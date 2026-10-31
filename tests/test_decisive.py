"""Tests for the decisive oversight study's scripts (experiments/decisive_*.py)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import so_arena as soa  # noqa: E402
from so_arena.core.ground_truth import JudgeCorrectness, StanceValue  # noqa: E402
from so_arena.core.items import AnswerOption, GroundTruth, TaskItem  # noqa: E402
from so_arena.core.runner import Profile, PlayerSpec, build_players, run_sync  # noqa: E402
from so_arena.mechanisms import Consultancy, Debate, DirectJudge  # noqa: E402
from so_arena.models.base import FunctionModel  # noqa: E402
from so_arena.samplers.arms import ASDExperiment  # noqa: E402
from so_arena.samplers.pools import expand_tree  # noqa: E402

import decisive_common as dc  # noqa: E402
import decisive_optimize as dopt  # noqa: E402
import decisive_protocols as dp  # noqa: E402
import decisive_report as dr  # noqa: E402
import decisive_select as ds  # noqa: E402


def _items(n=3):
    return [TaskItem(id=f"q{i}", domain="t", question=f"Is {i} even?",
                     answers=[AnswerOption(label="A", text="yes", value=1.0 if i % 2 == 0 else -1.0),
                              AnswerOption(label="B", text="no", value=-1.0 if i % 2 == 0 else 1.0)],
                     ground_truth=GroundTruth(correct="A" if i % 2 == 0 else "B")) for i in range(n)]


def _judge(p_a: float, name: str):
    return FunctionModel(lambda m, o, i: f'{{"A": {p_a}, "B": {1 - p_a}}}', name=name)


# --------------------------------------------------------------------------------------------- selection

def test_select_requires_every_debater_two_of_two_and_low_pooled_judges():
    t = pd.DataFrame({"j1_acc": [0, 1, 1 / 3, 0], "j1_n": [3, 3, 3, 3], "j2_acc": [2 / 3, 1, 1 / 3, 0],
                      "j2_n": [3, 3, 3, 3], "d1_acc": [1, 1, 1, 0.5], "d1_n": [2, 2, 2, 2],
                      "d2_acc": [1, 1, 1.0, 1], "d2_n": [2, 2, 1, 2]}, index=list("wxyz"))
    t["judges_acc"] = ds.pooled_judges(t, ["j1", "j2"])
    assert t.judges_acc.tolist() == pytest.approx([1 / 3, 1, 1 / 3, 0])
    mask = ds.select(t, ["j1", "j2"], ["d1", "d2"], 0.6)
    # x: judges right; y: d2 answered once only; z: d1 wrong once
    assert mask.tolist() == [True, False, False, False]


def test_pooled_judges_weights_by_samples_and_skips_missing():
    t = pd.DataFrame({"a_acc": [1.0, np.nan], "a_n": [1, np.nan], "b_acc": [0.0, np.nan], "b_n": [3, np.nan]})
    assert ds.pooled_judges(t, ["a", "b"]).iloc[0] == pytest.approx(0.25)
    assert np.isnan(ds.pooled_judges(t, ["a", "b"]).iloc[1])


def test_ledger_sets_stage_totals(tmp_path):
    led = dc.Ledger(tmp_path, budget=10)
    led.set("a", 3)
    led.set("a", 4)  # a stage's total is replaced, not added
    led.set("b", 5)
    again = dc.Ledger(tmp_path, budget=10)
    assert again.total == pytest.approx(9) and again.check(1) and not again.check(1.5)


# --------------------------------------------------------------------------------------------- protocols

def _record_debater_calls(mech, judge_p):
    calls = []

    def speak(messages, options, sample_index):
        calls.append(json.dumps([[m.model_dump() for m in messages], options.model_dump(exclude_none=True),
                                 sample_index], sort_keys=True))
        return "An argument."

    agent = soa.LLMPolicy(FunctionModel(speak, name="debater"))
    judge = soa.LLMPolicy(_judge(judge_p, f"judge{judge_p}"), cot=True)
    ASDExperiment([mech], _items(), agent=agent, fixtures={"judge": judge}, ground_truth=[StanceValue(),
                  JudgeCorrectness()], repeats=2).run()
    return sorted(calls)


@pytest.mark.parametrize("mech", [Debate(rounds=2, simultaneous=False), Debate(rounds=2, simultaneous=True)])
def test_judge_independent_protocols_make_identical_agent_calls_for_any_judge(mech):
    # the response cache keys on exactly these, so a second judge replays the first judge's transcripts
    a, b = _record_debater_calls(mech, 0.8), _record_debater_calls(mech, 0.3)
    assert a == b and len(set(a)) == len(a)  # and repeats are distinct samples


def test_consultancy_agent_calls_depend_on_the_judge():
    def ask(p):
        calls = []

        def speak(messages, options, sample_index):
            calls.append(json.dumps([m.model_dump() for m in messages], sort_keys=True))
            return "Consultant message."
        agent = soa.LLMPolicy(FunctionModel(speak, name="c"))
        judge = soa.LLMPolicy(FunctionModel(lambda m, o, i: f"Question {p}? " + f'{{"A": {p}, "B": {1 - p}}}',
                                            name=f"j{p}"))
        ASDExperiment([Consultancy(rounds=2)], _items(1), agent=agent, fixtures={"judge": judge}).run()
        return calls

    a, b = ask(0.8), ask(0.3)
    assert a[0] == b[0] and a[1] != b[1]  # the first message is shared; later ones see the judge's question


def test_episode_rows_and_item_table():
    agent = soa.LLMPolicy(FunctionModel(lambda m, o, i: "An argument.", name="d"))
    judge = soa.LLMPolicy(_judge(0.9, "j"))  # always says A: right on even items, wrong on odd ones
    eps = ASDExperiment([DirectJudge(), Debate(rounds=1, name="debate_seq")], _items(4), agent=agent,
                        fixtures={"judge": judge}, ground_truth=[StanceValue(), JudgeCorrectness()]).run()
    rows = pd.DataFrame(dp.episode_rows(eps, judge="j", debater="d"))
    assert set(rows.mechanism) == {"direct", "debate_seq"}
    deb = rows[rows.mechanism == "debate_seq"]
    assert set(deb.role) == {"debater_a", "debater_b"} and (deb.groupby("episode").size() == 2).all()
    assert rows.drop_duplicates("episode").cost.sum() == pytest.approx(rows.cost.sum())  # cost counted once
    it = dr.item_table(rows)
    assert it[it.mechanism == "direct"].set_index("item").accuracy.to_dict() == {"q0": 1, "q1": 0, "q2": 1, "q3": 0}
    # judge-alone verdict ASD on the Brier scale: 2 (2p - 1) on the true side's probability
    d = it[it.mechanism == "direct"].set_index("item")
    assert d.asd_brier["q0"] == pytest.approx(2 * (2 * 0.9 - 1))
    assert d.asd_brier["q1"] == pytest.approx(2 * (2 * 0.1 - 1))


def test_brier_reward():
    r = dr.brier_reward(pd.Series([1.0, 0.25, 0.25]), pd.Series([1, 1, -1]))
    assert list(r) == pytest.approx([0.0, -2 * 0.75 ** 2, -2 * 0.25 ** 2])


def _synthetic_items_table():
    rows = []
    for k in range(20):
        item = f"i{k}"
        rows += [{"judge": "j", "debater": "-", "mechanism": "direct", "item": item, "accuracy": 0.5,
                  "asd_brier": 0.0, "asd": 0.0, "concordance": 0.5},
                 {"judge": "-", "debater": "d", "mechanism": "direct_strong", "item": item, "accuracy": 1.0,
                  "asd_brier": 2.0, "asd": 1.0, "concordance": 1.0},
                 {"judge": "j", "debater": "d", "mechanism": "debate_seq", "item": item,
                  "accuracy": 1.0 if k % 2 else 0.5, "asd_brier": 1.0, "asd": 0.5, "concordance": 1.0}]
    return pd.DataFrame(rows)


def test_summarize_gap_recovered_and_igr():
    it = dr.with_references(_synthetic_items_table())
    s = dr.summarize(it, ["judge", "debater", "mechanism"], n_boot=200).iloc[0]
    assert s.accuracy == pytest.approx(0.75) and s.judge_alone == 0.5 and s.debater_alone == 1.0
    assert s.delta == pytest.approx(0.25) and s.gap == pytest.approx(0.5)
    assert s.gap_recovered == pytest.approx(0.5) and s.igr == pytest.approx(0.5)
    assert s.delta_lo < 0.25 < s.delta_hi and s["items"] == 20


def test_committed_filter_drops_both_arms_of_a_sample_with_a_conceding_liar():
    rows = pd.DataFrame([{"judge": "j", "debater": "d", "mechanism": "consultancy", "item": "q", "repeat": r,
                          "arm": arm, "episode": f"{r}{arm}"} for r in range(2) for arm in ("true", "false")])
    verdicts = pd.DataFrame([{"kind": "prompted", "judge": "j", "debater": "d", "mechanism": "consultancy",
                              "item": "q", "repeat": 0, "verdict": "conceded"},
                             {"kind": "prompted", "judge": "j", "debater": "d", "mechanism": "consultancy",
                              "item": "q", "repeat": 1, "verdict": "committed"}])
    kept = dr.committed_filter(rows, verdicts)
    assert sorted(kept.episode) == ["1false", "1true"]


# --------------------------------------------------------------------------------------------- optimization

def test_liar_roles_and_policies():
    assert dopt.liar_role("debate_1r", "true") == "debater_b" and dopt.liar_role("debate_1r", "false") == "debater_a"
    assert dopt.liar_role("consultancy_1r", "false") == "consultant"
    assert dopt.policies("consultancy_1r", "true", "liar", 8) == {}  # no liar on the consultant's true arm
    assert set(dopt.policies("debate_1r", "true", "liar", 8)) == {"debater_b"}
    assert set(dopt.policies("debate_1r", "false", "both", 8)) == {"debater_a", "debater_b"}
    assert dopt.policies("debate_1r", "true", "both", 4)["debater_a"].n == 4


def test_candidate_texts_recover_every_pool_candidate():
    seen = {"n": 0}

    def speak(messages, options, sample_index):
        seen["n"] += 1
        return f"speech number {sample_index}"

    agent = soa.LLMPolicy(FunctionModel(speak, name="d"))
    judge = soa.LLMPolicy(_judge(0.6, "j"))
    item = _items(1)[0]
    prof = Profile(name="p", players={"debater_a": PlayerSpec(policy=agent, stance="true"),
                                      "debater_b": PlayerSpec(policy=agent, stance="false"),
                                      "judge": PlayerSpec(policy=judge)})
    tree, eps = run_sync(expand_tree(Debate(rounds=1), item, build_players(prof, item),
                                     pool_sizes={"debater_a": 3, "debater_b": 3}, keep_episodes=True))
    cands = dopt.candidate_texts(tree, eps)
    assert len(cands) == 6 and {c["role"] for c in cands} == {"debater_a", "debater_b"}
    by_role = {r: sorted(c["text"] for c in cands if c["role"] == r) for r in ("debater_a", "debater_b")}
    assert all(len(set(v)) == 3 and all(t.startswith("speech number") for t in v) for v in by_role.values())


def test_selected_audit_weights_verdicts_by_selection():
    sel = pd.DataFrame([{"debater": "d", "judge": "j", "mechanism": "debate_1r", "item": "q", "arm": "true", "n": n,
                         "key": "k", "index": i, "weight": w}
                        for n, ws in ((1, [0.5, 0.5]), (2, [1.0, 0.0])) for i, w in enumerate(ws)])
    ver = pd.DataFrame([{"kind": "optimized", "debater": "d", "mechanism": "debate_1r", "item": "q", "arm": "true",
                         "pool_key": "k", "index": i, "verdict": v} for i, v in enumerate(["committed", "conceded"])])
    out = dr.selected_audit(sel, ver).set_index("n")
    assert out.loc[1, "conceded"] == pytest.approx(0.5) and out.loc[2, "conceded"] == pytest.approx(0.0)
