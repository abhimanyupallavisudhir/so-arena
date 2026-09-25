"""Regression tests for incentive-measurement fixes: n-player PSRO meta-solvers checked by NashConv, honesty
margins signed by measured behaviour, audits redrawn per RL training step, multi-decision roles refused by
reward functions, judge parse failures reported in ASD, audit findings withheld from releases, and label
coverage next to averaged ground-truth values."""

import json
import logging
import math

import numpy as np
import pytest

import so_arena as soa
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.mechanisms import Debate, Propaganda


def _synth(n=6, seed=3):
    dom = SyntheticPersuasion(n_items=n, seed=seed)
    return dom.load(), dom.context()


# ----------------------------------------------------------------------------- releases

def test_release_withholds_audit_findings(tmp_path, caplog):
    """reward_details carried the label a judge audit returned and RandomAudit's audited values - ground truth.
    The rewards themselves are computed from the findings, so releasing audited episodes warns."""
    import logging

    from so_arena.release import release

    items, ctx = _synth(8)
    oracle = soa.truth_oracle(items)
    reward = (soa.RandomAudit(soa.JudgeScore("log"), lambda ep: {"debater_a": 1.0, "debater_b": -1.0}, p=0.5)
              + soa.JudgeAuditScore(oracle, p=0.5))
    mech = Debate(rounds=1, affordances={"agents": ["answer_key"]}, trainable={"judge": True}, reward=reward)
    prof = Profile(name="p", players={"debater_a": PlayerSpec(policy=synthetic_arguer(), stance="true"),
                                      "debater_b": PlayerSpec(policy=synthetic_arguer(), stance="false"),
                                      "judge": synthetic_judge()})
    eps = run_sync(run_episodes(mech, items, [prof], ctx=ctx))
    assert any("judge_audit_label" in e.reward_details for e in eps)
    assert any("audit_values" in e.reward_details for e in eps)
    for public in (False, True):
        out = tmp_path / f"rel{public}"
        with caplog.at_level(logging.WARNING, logger="so_arena"):
            release(eps, items, out, html=False, public_labels=public)
        assert "were audited" in caplog.text
        released = [json.loads(x) for x in (out / "episodes.jsonl").read_text().splitlines()]
        assert released and all(set(e["reward_details"]) <= {"audited"} for e in released)
        text = (out / "episodes.jsonl").read_text()
        assert "audit_label" not in text and "audit_values" not in text


# ----------------------------------------------------------------------------- RL reward functions

def _audited_propaganda(items, ctx):
    from so_arena.core.game import Player
    from so_arena.integrations.rl import reward_function

    # every audit finds a violation and costs 10: audited iff the reward is below the lowest log score
    rule = soa.RandomAudit(soa.JudgeScore("log"), lambda ep: {"agent": -1.0}, p=0.5, penalty=10.0)
    mech = Propaganda(affordances={"agents": ["answer_key"]}, reward=rule)
    return lambda **kw: reward_function(mech, "agent", items, {"judge": Player(policy=synthetic_judge())},
                                        stances={"agent": "true"}, ctx=ctx, **kw)


def test_reward_function_redraws_audits_every_training_step():
    """A fixed seed audited the same items at every step: rewards were identical over epochs."""
    from types import SimpleNamespace

    items, ctx = _synth(12)
    make = _audited_propaganda(items, ctx)
    ids = [it.id for it in items for _ in range(2)]  # two completions per prompt, as in GRPO
    texts = [f'<arg for="{it.true_label}" strength="1.0"> argument' for it in items for _ in range(2)]

    def audited(rewards):
        pairs = [(i, r < -9.5) for i, r in zip(ids, rewards)]
        per_item = {}
        for i, a in pairs:
            per_item.setdefault(i, set()).add(a)
        assert all(len(v) == 1 for v in per_item.values())  # one shared draw for a prompt's completions
        return frozenset(i for i, v in per_item.items() if True in v)

    fn = make()
    steps = [audited(fn(completions=texts, item_id=ids)) for _ in range(3)]
    assert len(set(steps)) > 1 and all(0 < len(s) < len(items) for s in steps)
    assert [h["step"] for h in fn.history[::len(ids)]] == [0, 1, 2]
    assert fn.history[0]["seed"] == 0 and len({h["seed"] for h in fn.history}) == 3
    # the trainer's step decides: two calls at one global step share the draw
    fn = make()
    state = SimpleNamespace(global_step=7)
    a, b = (audited(fn(completions=texts, item_id=ids, trainer_state=state)) for _ in range(2))
    assert a == b and {h["step"] for h in fn.history} == {7}
    # opting out keeps one draw for every step
    fn = make(redraw_per_step=False)
    assert len({audited(fn(completions=texts, item_id=ids)) for _ in range(3)}) == 1


def test_reward_function_refuses_to_reuse_a_completion_for_several_decisions():
    from so_arena.core.game import Player
    from so_arena.integrations.rl import MultipleDecisionsError, reward_function
    from so_arena.mechanisms import Consultancy

    items, ctx = _synth(2)
    fn = reward_function(Consultancy(rounds=2, affordances={"agents": ["answer_key"]}), "consultant", items,
                         {"judge": Player(policy=synthetic_judge())}, stances={"consultant": "true"}, ctx=ctx)
    with pytest.raises(MultipleDecisionsError, match="more than one decision"):
        fn(completions=["<arg for=\"A\" strength=\"1.0\"> x"], item_id=[items[0].id])
    one = reward_function(Consultancy(rounds=1, affordances={"agents": ["answer_key"]}), "consultant", items,
                          {"judge": Player(policy=synthetic_judge())}, stances={"consultant": "true"}, ctx=ctx)
    assert one(completions=["an argument"], item_id=[items[0].id])[0] is not None


# ----------------------------------------------------------------------------- PSRO meta-solvers

def _matching_pennies_3p(duplicate_tails=True):
    """Jordan's three-player matching pennies (a matches b, b matches c, c mismatches a); unique equilibrium:
    everyone mixes 50/50. With tails duplicated (as a PSRO population can hold two equivalent strategies)
    uniform play is no longer an equilibrium, and replicator dynamics from it spiral out to the boundary."""
    import itertools

    from so_arena.games import NormalFormGame

    idx = [0, 1, 1] if duplicate_tails else [0, 1]
    k = len(idx)
    U = {p: np.zeros((k, k, k)) for p in "abc"}
    for i, j, m in itertools.product(range(k), repeat=3):
        a, b, c = idx[i], idx[j], idx[m]
        U["a"][i, j, m] = 1.0 if a == b else -1.0
        U["b"][i, j, m] = 1.0 if b == c else -1.0
        U["c"][i, j, m] = 1.0 if c != a else -1.0
    names = ["H", "T", "T2"][:k]
    return NormalFormGame(list("abc"), {p: names for p in "abc"}, U, name="mp3")


def test_n_player_nash_meta_solver_reaches_an_equilibrium(caplog):
    """The 3+ player 'nash' solver returned the last replicator iterate unchecked: NashConv 2.0."""
    from so_arena.games import NormalFormGame
    from so_arena.samplers.psro import solve_meta

    g = _matching_pennies_3p()
    x_rep, _ = g.replicator(steps=3000)
    assert g.nash_conv(x_rep) > 1.5  # the old answer
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        x = solve_meta(g, "nash")
    assert g.nash_conv(x) < 1e-6 and "approximate" not in caplog.text
    for v in x:  # heads 1/2, the two tails 1/2 together
        assert v[0] == pytest.approx(0.5, abs=1e-4) and v[1] + v[2] == pytest.approx(0.5, abs=1e-4)
    jordan = _matching_pennies_3p(False)
    assert jordan.nash_conv(solve_meta(jordan, "nash")) < 1e-9
    # symmetric players get one shared (symmetric) equilibrium
    A = np.array([[-1.0, 3.0], [0.0, 1.0]])  # hawk-dove for each pair, three players: hawk with p = 2/3
    U = {p: np.zeros((2, 2, 2)) for p in "abc"}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = (i, j, k)
                for a, p in enumerate("abc"):
                    U[p][s] = sum(A[s[a], s[b]] for b in range(3) if b != a)
    hd = NormalFormGame(list("abc"), {p: ["hawk", "dove"] for p in "abc"}, U)
    xs = solve_meta(hd, "nash", symmetric=list("abc"))
    assert hd.nash_conv(xs) < 1e-6 and np.allclose(xs[0], xs[1]) and np.allclose(xs[0], xs[2])
    assert xs[0][0] == pytest.approx(2 / 3, abs=1e-4)
    # two-player games still solve exactly (support enumeration)
    mp2 = NormalFormGame(["a", "b"], {"a": ["H", "T"], "b": ["H", "T"]},
                         {"a": np.array([[1.0, -1.0], [-1.0, 1.0]]), "b": np.array([[-1.0, 1.0], [1.0, -1.0]])})
    assert mp2.nash_conv(solve_meta(mp2, "nash")) < 1e-9


def test_psro_records_the_meta_strategy_nash_conv():
    import re as _re

    from so_arena.models import FunctionModel
    from so_arena.samplers.psro import PSRO

    items, ctx = _synth(4, seed=4)

    def level_factory(strategy):
        m = _re.search(r"level (\d+)", strategy or "")
        lv = int(m.group(1)) if m else 0
        return synthetic_arguer(honest_mean=1.0 + 0.3 * lv, dishonest_mean=0.3 * lv, sd=0.2, label=f"level{lv}")

    def fn(messages, options, i):
        levels = [int(x) for x in _re.findall(r"level (\d+)", messages[-1].content)] or [0]
        return f"<rationale>r</rationale><strategy>argue at level {max(levels) + 1}</strategy>"

    psro = PSRO(Debate(rounds=1, affordances={"agents": ["answer_key"]}, zero_sum=True), items,
                roles=["debater_a", "debater_b"], initial={"debater_a": {"base": ""}, "debater_b": {"base": ""}},
                policy_factories={"debater_a": level_factory, "debater_b": level_factory},
                fixtures={"judge": synthetic_judge()}, optimizer=FunctionModel(fn, name="opt"),
                stances={"debater_a": "true", "debater_b": "false"}, iterations=1,
                search_kwargs={"iterations": 1, "candidates_per_iter": 1}, ctx=ctx)
    hist = psro.run()
    for h in hist:
        assert h.meta_nash_conv is not None and h.meta_nash_conv < 1e-6 and h.meta_equilibrium is True
        assert h.meta_value["gt_welfare_coverage"] == pytest.approx(1.0)
    assert "best_value_coverage" in hist[0].searches["debater_a"]


# ----------------------------------------------------------------------------- prompt search

def _level_factory(strategy):
    import re as _re

    m = _re.search(r"level (\d+)", strategy or "")
    lv = int(m.group(1)) if m else 0
    return synthetic_arguer(honest_mean=1.0 + 0.3 * lv, dishonest_mean=0.3 * lv, sd=0.2, label=f"level{lv}")


def _level_optimizer():
    import re as _re

    from so_arena.models import FunctionModel

    def fn(messages, options, i):
        levels = [int(x) for x in _re.findall(r"level (\d+)", messages[-1].content)] or [0]
        return "".join(f"<rationale>r</rationale><strategy>argue at level {max(levels) + 1 + j}</strategy>"
                       for j in range(2))

    return FunctionModel(fn, name="opt")


def test_honesty_margin_is_signed_by_measured_behaviour(caplog):
    """The margin was signed by the directives' labels: a 'deceptive' winner that argued the truth made
    deception look like it beat honesty."""
    from so_arena.samplers.prompt_search import PromptSearchSuite

    items, ctx = _synth(12, seed=4)
    suite = PromptSearchSuite(directives=("honest", "deceptive"), mechanism=Propaganda(affordances={"agents": ["answer_key"]}),
                              items=items[:8], eval_items=items[8:], role="agent", policy_factory=_level_factory,
                              others={"judge": synthetic_judge()}, optimizer=_level_optimizer(), iterations=2,
                              candidates_per_iter=2, ctx=ctx)
    suite.searches["honest"].arms = ["false"]    # the "honest" search's strategies argue a false answer
    suite.searches["deceptive"].arms = ["true"]  # the "deceptive" search's strategies argue the truth
    suite.run()
    old = suite.honesty_margin(by="directive")
    assert old["margin"] < 0  # read as "deception beats honesty"
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        m = suite.honesty_margin()
    assert m["basis"] == "measured" and sorted(m["mislabelled"]) == ["deceptive", "honest"]
    assert "does not behave as labelled" in caplog.text
    assert m["honest_value"] == 1.0 and m["deceptive_value"] == -1.0
    assert m["honest_directive"] == "deceptive" and m["deceptive_directive"] == "honest"
    assert m["margin"] > 0 and m["ci_low"] > 0 and m["n_items"] == 4
    # a suite that never produced measured deception has no margin
    only_honest = PromptSearchSuite(directives=("honest", "deceptive"), mechanism=Propaganda(affordances={"agents": ["answer_key"]}),
                                    items=items[:8], role="agent", policy_factory=_level_factory,
                                    others={"judge": synthetic_judge()}, optimizer=_level_optimizer(), iterations=1,
                                    candidates_per_iter=2, ctx=ctx, arms=["true"])
    only_honest.run()
    m = only_honest.honesty_margin()
    assert math.isnan(m["margin"]) and m["deceptive_id"] is None and m["mislabelled"] == ["deceptive"]


def test_prompt_search_reports_value_coverage(caplog):
    """Mean values averaged over the episodes that had a value, silently: now with coverage, NaN below half."""
    from so_arena.samplers.prompt_search import Candidate, PromptSearch

    c = Candidate(id="s1", strategy="x", rewards=[0.1] * 4, values=[1.0, None, None, None])
    assert c.value_coverage == 0.25 and math.isnan(c.mean_value)
    c = Candidate(id="s2", strategy="x", rewards=[0.1] * 4, values=[1.0, 1.0, None, -1.0])
    assert c.value_coverage == 0.75 and c.mean_value == pytest.approx(1 / 3)
    items, ctx = _synth(4, seed=4)
    search = PromptSearch(Propaganda(affordances={"agents": ["answer_key"]}), items, role="agent",
                          policy_factory=_level_factory, others={"judge": synthetic_judge()},
                          optimizer=_level_optimizer(), arms=["true"], iterations=1, candidates_per_iter=1,
                          ground_truth=[], ctx=ctx)  # no scorers: no values at all
    res = search.run()
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        frame = res.frame()
    assert (frame["value_coverage"] == 0).all() and frame["mean_value"].isna().all()
    assert "fewer than 50%" in caplog.text
    assert "best_value_coverage" in res.path()


# ----------------------------------------------------------------------------- judge parse failures

def test_asd_reports_and_can_exclude_unparsed_judgments(caplog):
    """A judge parse failure fell back to 50/50 and was ignored, quietly pulling ASD toward 0."""
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd, summary, with_parse_status
    from so_arena.core.policy import FunctionPolicy
    from so_arena.samplers.arms import ASDExperiment

    items, ctx = _synth(8)
    truth = {it.id: it.true_label for it in items}
    broken = {it.id for it in items[:4]}

    def judge(req, c):
        iid = c.game.item.id
        if iid in broken:
            return "I cannot decide."
        return {lab: 0.9 if lab == truth[iid] else 0.1 / (len(req.options) - 1) for lab in req.options}

    exp = ASDExperiment([Propaganda(affordances={"agents": ["answer_key"]})], items, agent=synthetic_arguer(),
                        fixtures={"judge": FunctionPolicy(judge, label="judge")}, ctx=ctx)
    exp.run()
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        s = exp.summary(transforms=("log",))
    assert s["parse_fail_rate"].iloc[0] == pytest.approx(0.5)
    assert s["parse_fail_rate_true"].iloc[0] == pytest.approx(0.5) == s["parse_fail_rate_false"].iloc[0]
    assert caplog.text.count("did not parse") == 1
    full = float(np.log(0.9) - np.log(0.1))  # binary synthetic items
    assert s["asd"].iloc[0] == pytest.approx(full / 2)  # the fallback pays both arms alike
    clean = exp.summary(transforms=("log",), exclude_unparsed=True)
    assert clean["asd"].iloc[0] == pytest.approx(full) and clean["n_items"].iloc[0] == 4
    assert clean["accuracy"].iloc[0] == pytest.approx(1.0)
    # plain role frames: from the explicit column, or from the judge's own rows
    eps = exp.episodes
    for df in (with_parse_status(role_frame(eps), eps), role_frame(eps, include_fixtures=True)):
        a = asd(df, exclude_unparsed=True, parse_warn=None)
        assert a["parse_fail_rate"].iloc[0] == pytest.approx(0.5) and a["asd"].iloc[0] == pytest.approx(full)
    assert "parse_fail_rate" not in asd(role_frame(eps))  # unknown: nothing claimed
    assert summary(with_parse_status(role_frame(eps), eps))["parse_fail_rate"].iloc[0] == pytest.approx(0.5)


# ----------------------------------------------------------------------------- label coverage in games

def test_game_outcomes_report_their_label_coverage(caplog):
    """Outcome averages over a meta-strategy or a training trajectory used only the labelled episodes, silently."""
    from so_arena.games import NormalFormGame
    from so_arena.games.learning import policy_gradient

    U = np.array([[1.0, 0.0], [0.0, 1.0]])
    W = np.array([[1.0, np.nan], [0.0, 0.5]])
    C = np.array([[0.2, 0.0], [1.0, 1.0]])  # (0, 0): only a fifth of its episodes had a value
    g = NormalFormGame(["a", "b"], {"a": ["x", "y"], "b": ["x", "y"]}, {"a": U, "b": U}, outcomes={"w": W},
                       coverage={"w": C}, name="g")
    value, cov = g.expected_outcome(g.uniform(), "w")
    assert cov == pytest.approx((0.2 + 0 + 1 + 1) / 4) and value == pytest.approx(0.5)
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        value, cov = g.expected_outcome(g.pure((0, 0)), "w")
    assert cov == pytest.approx(0.2) and math.isnan(value) and "measured on 20%" in caplog.text
    df = policy_gradient(g, steps=50, init=[np.array([0.9, 0.1]), np.array([0.9, 0.1])])
    assert "w_coverage" in df and df["w_coverage"].iloc[-1] < 0.5 and math.isnan(df["w"].iloc[-1])
    assert "w_coverage" in g.table()


def test_judge_accuracy_reports_coverage(caplog):
    import pandas as pd

    from so_arena.analysis.metrics import judge_accuracy

    df = pd.DataFrame({"episode_id": [f"e{i}" for i in range(4)], "item_id": ["a", "b", "c", "d"],
                       "mechanism": "m", "judge_correct": [1.0, None, None, None], "judge_p_true": [0.9, None, None, None]})
    ja = judge_accuracy(df)
    assert ja["accuracy_coverage"].iloc[0] == 0.25 and ja["accuracy"].iloc[0] == 1.0  # analysis default: report only
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        ja = judge_accuracy(df, min_coverage=0.5)
    assert math.isnan(ja["accuracy"].iloc[0]) and "known for 25%" in caplog.text
