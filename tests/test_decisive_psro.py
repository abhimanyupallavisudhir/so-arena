"""Tests for the instant-judge replay (decisive_instant.py) and the PSRO self-play stage (decisive_psro*.py)."""

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import so_arena as soa  # noqa: E402
from so_arena.core.actions import ActionRequest  # noqa: E402
from so_arena.core.items import AnswerOption, GroundTruth, TaskItem  # noqa: E402
from so_arena.core.policy import format_instructions  # noqa: E402
from so_arena.core.runner import PlayerSpec, Profile, build_players, run_sync  # noqa: E402
from so_arena.mechanisms import Consultancy, Debate, RecursiveDebate  # noqa: E402
from so_arena.models.base import FunctionModel  # noqa: E402
from so_arena.samplers.arms import ASDExperiment, arm_profiles  # noqa: E402
from so_arena.samplers.pools import expand_tree  # noqa: E402

import decisive_instant as di  # noqa: E402
import decisive_optimize as dopt  # noqa: E402
import decisive_protocols as dp  # noqa: E402
import decisive_psro as ps  # noqa: E402
import decisive_psro_report as pr  # noqa: E402


def _items(n=4):
    return [TaskItem(id=f"q{i}", domain="t", question=f"Is {i} even?",
                     answers=[AnswerOption(label="A", text="yes", value=1.0 if i % 2 == 0 else -1.0),
                              AnswerOption(label="B", text="no", value=-1.0 if i % 2 == 0 else 1.0)],
                     ground_truth=GroundTruth(correct="A" if i % 2 == 0 else "B")) for i in range(n)]


def _judge(p_a: float, name: str):
    return FunctionModel(lambda m, o, i: f'Question? {{"A": {p_a}, "B": {1 - p_a}}}', name=name)


def _speaker(name="deb"):
    return soa.LLMPolicy(FunctionModel(lambda m, o, i: f"speech {i} " + m[-1].content[-30:].replace("\n", " "),
                                       name=name))


# ------------------------------------------------------------------------------------------ instant judges

def test_ask_now_asks_for_the_json_at_once():
    req = ActionRequest(kind="probabilities", options=["A", "B"])
    now, ask = format_instructions(req, "ask_now"), format_instructions(req, "ask")
    assert "Think" in ask and "Think" not in now and "nothing else" in now and '"A"' in now


def test_instant_judge_parses_a_bare_json_answer():
    pol = di.instant_judge(FunctionModel(lambda m, o, i: '{"A": 0.3, "B": 0.7}', name="inst"))
    assert pol.cot is False and pol.temperature == 0.0 and pol.elicitation == "ask_now"
    eps = ASDExperiment([Debate(rounds=1)], _items(2), agent=_speaker(), fixtures={"judge": pol},
                        ground_truth=dp.GT).run()
    verdicts = [t for e in eps for t in e.turns if t.probs]
    assert verdicts and all(t.parse_ok and abs(t.probs["B"] - 0.7) < 1e-9 for t in verdicts)


def test_transcript_replay_keeps_consultations_and_only_reasks_the_verdict():
    """The consultant's speeches and the reasoning judge's questions replay exactly; only the verdict is new."""
    mech = Consultancy(rounds=2)
    first = ASDExperiment([mech], _items(), agent=_speaker(), fixtures={"judge": soa.LLMPolicy(_judge(0.8, "j1"))},
                          ground_truth=dp.GT, repeats=2).run()
    replay = di.TranscriptReplay(first)
    assert replay.items == {it.id for it in _items()} and replay.repeats == 2
    seen = []

    def new_judge(m, o, i):
        seen.append(m[-1].content)
        return '{"A": 0.1, "B": 0.9}'

    judge = di.TranscriptReplay(verdict=soa.LLMPolicy(FunctionModel(new_judge, name="j2")))
    judge.text = replay.text
    again = ASDExperiment([mech], _items(), agent=replay, fixtures={"judge": judge}, ground_truth=dp.GT,
                          repeats=2).run()
    assert all(e.error is None for e in again) and len(again) == len(first)
    key = lambda e: (e.item_id, e.repeat, e.tags.get("arm"))  # noqa: E731
    texts = lambda e: [(t.role, t.phase, t.text) for t in e.turns if t.kind == "text"]  # noqa: E731
    old = {key(e): e for e in first}
    for e in again:
        assert texts(e) == texts(old[key(e)])
        assert next(t for t in e.turns if t.probs).probs["B"] == pytest.approx(0.9)
    assert len(seen) == len(again) and all("Question?" in s for s in seen)  # the replayed questions are shown


def test_pool_replay_regrows_the_same_tree_under_a_new_judge(tmp_path):
    mech = Debate(rounds=1, simultaneous=True)
    item = _items(1)[0]
    agent = soa.LLMPolicy(FunctionModel(lambda m, o, i: f"candidate {i % 10_000}", name="d"))
    prof = arm_profiles(mech, agent=agent, fixtures={"judge": soa.LLMPolicy(_judge(0.6, "j"))})[0]
    sizes = {"debater_a": 3, "debater_b": 3}
    tree, eps = run_sync(expand_tree(mech, item, build_players(prof, item), pool_sizes=sizes, keep_episodes=True,
                                     ground_truth=dp.GT))
    path = tmp_path / "c.jsonl"
    with open(path, "w") as f:
        for c in dopt.candidate_texts(tree, eps):
            f.write(json.dumps({"item": item.id, **c}) + "\n")
    replay = di.PoolReplay(path)
    prof2 = arm_profiles(mech, agent=replay, fixtures={"judge": soa.LLMPolicy(_judge(0.2, "j2"))})[0]
    tree2, eps2 = run_sync(expand_tree(mech, item, build_players(prof2, item), pool_sizes=sizes, keep_episodes=True,
                                       ground_truth=dp.GT))
    assert len(tree2.leaves) == len(tree.leaves) == 9 and all(e.error is None for e in eps2)
    assert sorted(t.text for e in eps2 for t in e.turns if t.kind == "text") == \
        sorted(t.text for e in eps for t in e.turns if t.kind == "text")
    assert {round(lf.values["judge_p_true"], 6) for lf in tree2.leaves.values()} == {0.2}


# ------------------------------------------------------------------------------------------------- PSRO

def test_mean_judge_averages_independent_samples_and_keeps_them():
    calls = []

    def fn(m, o, i):
        calls.append(i)
        p = [0.2, 0.5, 0.8][len(calls) - 1]
        return f'{{"A": {p}, "B": {1 - p}}}'

    judge = ps.MeanJudge(soa.LLMPolicy(FunctionModel(fn, name="j")), k=3)
    eps = ASDExperiment([Debate(rounds=1)], _items(1), agent=_speaker(), fixtures={"judge": judge},
                        ground_truth=dp.GT, arms=["true"]).run()
    v = next(t for t in eps[0].turns if t.probs)
    assert v.probs["A"] == pytest.approx(0.5) and len(set(calls)) == 3  # three distinct sample indices
    assert [s["A"] for s in v.metadata["samples"]] == pytest.approx([0.2, 0.5, 0.8])
    rec = ps.episode_record(_tag(eps[0]), {"q0": "A"})
    assert rec["p_true_samples"] == pytest.approx([0.2, 0.5, 0.8])


def _tag(ep):
    ep.tags["seat_h"] = "debater_a"
    return ep


def test_seats_are_exactly_balanced_and_fixed():
    items = _items(10)
    a, b = ps.seat_map(items), ps.seat_map(list(reversed(items)))
    assert a == b and sorted(a.values()).count("debater_a") == 5


def test_play_puts_the_honest_strategy_on_the_true_side_in_both_seats():
    honest = soa.LLMPolicy(FunctionModel(lambda m, o, i: "I am honest.", name="h"), label="H")
    liar = soa.LLMPolicy(FunctionModel(lambda m, o, i: "I am lying.", name="l"), label="L")
    items = _items(4)
    recs, eps = run_sync(ps.play(Debate(rounds=1), items, honest, liar, soa.LLMPolicy(_judge(0.7, "j")),
                                 concurrency=4, truth=ps.truth_letters(items), keep_text=True))
    assert len(recs) == 4 and {r["seat_h"] for r in recs} == {"debater_a", "debater_b"}
    for r, e in zip(recs, eps):
        honest_turns = [t for t in e.turns if t.role == r["seat_h"]]
        assert all(t.text == "I am honest." for t in honest_turns)
        assert e.positions[r["seat_h"]] == next(it.true_label for it in items if it.id == e.item_id)
        # the judge always says A with 0.7: the honest side's reward is log p(true answer)
        p_true = 0.7 if e.positions[r["seat_h"]] == "A" else 0.3
        assert r["r_h"] == pytest.approx(math.log(p_true)) and r["r_l"] == pytest.approx(math.log(1 - p_true))
        assert all(s["honest"] == (s["role"] == r["seat_h"]) for s in r["speeches"])


def test_recursive_debate_self_play_alternates_prover_and_opponent():
    honest = soa.LLMPolicy(FunctionModel(lambda m, o, i: "I am honest.", name="h"), label="H")
    liar = soa.LLMPolicy(FunctionModel(lambda m, o, i: "I am lying.", name="l"), label="L")
    judge = soa.LLMPolicy(FunctionModel(lambda m, o, i: 'Hmm. {"true": 0.8, "false": 0.2}', name="j"))
    items = _items(4)
    mech = RecursiveDebate(depth=1, width=2)
    assert ps.seats_of(mech) == ("prover", "opponent") and ps.other("opponent") == "prover"
    recs, eps = run_sync(ps.play(mech, items, honest, liar, judge, concurrency=4, truth=ps.truth_letters(items)))
    assert {r["seat_h"] for r in recs} == {"prover", "opponent"}
    truth = ps.truth_letters(items)
    for r, e in zip(recs, eps):
        assert not r["error"] and "p_true_samples" not in r
        # the prover always defends the honest side's answer when honest, a false one when lying
        assert (e.positions["prover"] == truth[e.item_id]) == (r["seat_h"] == "prover")
        assert all(t.text == "I am honest." for t in e.turns if t.role == r["seat_h"] and t.kind == "text")
        # the judge gives every final claim 0.8: the prover earns 0.8, the opponent 0.2
        assert r["r_h"] == pytest.approx(0.8 if r["seat_h"] == "prover" else 0.2)
        assert r["r_h"] + r["r_l"] == pytest.approx(1.0)


def _write_pair(d: Path, split, h, l, rows):
    p = d / "pairs" / split / f"{h}__{l}.jsonl.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_report_evaluates_each_equilibrium_on_held_out_items(tmp_path):
    d = tmp_path / "psro" / "debate_sim" / "j"
    d.mkdir(parents=True)
    pops = {"honest": ["base"], "liar": ["base", "br1"]}
    state = {"protocol": "debate_sim", "judge": "j", "pop": {"honest": {"base": ""}, "liar": {"base": "", "br1": "x"}},
             "iterations": [
                 {"iteration": 0, "populations": {"honest": ["base"], "liar": ["base"]},
                  "sigma": {"honest": [1.0], "liar": [1.0], "value": {}, "nash_conv": 0.0}},
                 {"iteration": 1, "populations": pops,
                  "sigma": {"honest": [1.0], "liar": [0.25, 0.75], "value": {}, "nash_conv": 0.0}}]}
    (d / "state.json").write_text(json.dumps(state))
    for l, (rh, rl) in {"base": (-0.2, -2.0), "br1": (-1.0, -0.6)}.items():
        _write_pair(d, "test", "base", l, [{"item": f"q{i}", "seat_h": "debater_a", "error": False, "r_h": rh, "r_l": rl,
                                            "correct": 1.0, "p_true_samples": [0.6, 0.8]} for i in range(5)])
    st, pairs = pr.load_cell(d)
    rows, _, per = pr.cell_rows(st, pairs)
    assert [r["iteration"] for r in rows] == [0, 1]
    assert rows[0]["asd"] == pytest.approx(1.8) and rows[0]["conc"] == 1.0
    assert rows[1]["asd"] == pytest.approx(0.25 * 1.8 + 0.75 * -0.4) and rows[1]["support_l"] == 2
    single = np.mean([math.log(0.6), math.log(0.8)]) - np.mean([math.log(0.4), math.log(0.2)])
    assert rows[0]["asd_1"] == pytest.approx(single)


def test_seat_rows_split_by_the_honest_seat_and_difference(tmp_path):
    per = pd.DataFrame({"asd": [1.0, 1.2, -0.2, 0.0], "acc": [1, 1, 0, 1],
                        "seat_h": ["debater_b", "debater_b", "debater_a", "debater_a"]},
                       index=["q1", "q2", "q3", "q4"])
    rows = {r["seat_h"]: r for r in pr.seat_rows(per, protocol="debate_seq")}
    assert rows["debater_b"]["asd"] == pytest.approx(1.1) and rows["debater_a"]["asd"] == pytest.approx(-0.1)
    assert rows["last - first"]["asd"] == pytest.approx(1.2) and rows["last - first"]["acc"] == pytest.approx(0.5)
    assert rows["last - first"]["asd_lo"] <= 1.2 <= rows["last - first"]["asd_hi"]


def test_cache_can_drop_native_reasoning(tmp_path):
    from so_arena.config import settings
    from so_arena.core.types import Completion, GenerateOptions, Message
    from so_arena.models.base import Model
    from so_arena.models.cache import CachedModel, ResponseCache

    class Reasoner(Model):
        name = "reasoner"
        supports_logprobs = False
        calls = 0

        async def generate(self, messages, options=None, *, sample_index=0):
            Reasoner.calls += 1
            return Completion(text="answer", reasoning="long hidden reasoning", model=self.name)

    old = settings.cache_reasoning
    try:
        settings.cache_reasoning = False
        m = CachedModel(Reasoner(), ResponseCache(tmp_path / "c.sqlite"))
        msgs = [Message(role="user", content="q")]
        first = run_sync(m.generate(msgs, GenerateOptions()))
        again = run_sync(m.generate(msgs, GenerateOptions()))
        assert first.reasoning == "long hidden reasoning"  # the live call keeps it
        assert again.cached and again.text == "answer" and again.reasoning is None and Reasoner.calls == 1
    finally:
        settings.cache_reasoning = old
