import math

import numpy as np
import pytest

import so_arena as soa
from so_arena.analysis.frames import role_frame
from so_arena.analysis.metrics import asd, graded_asd, incentive_alignment, open_probs, pairwise_concordance
from so_arena.core.parsing import (
    parse_choice,
    parse_probabilities,
    parse_score,
    probs_from_logprobs,
    split_reasoning,
)
from so_arena.core.types import TokenLogprob, TopLogprob
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.mechanisms import Consultancy, Debate, DirectJudge, Propaganda
from so_arena.samplers.arms import ASDExperiment, arm_profiles


# ----------------------------------------------------------------------------- parsing

def test_parse_probabilities_variants():
    assert parse_probabilities('blah {"A": 0.7, "B": 0.3}', ["A", "B"]) == pytest.approx({"A": 0.7, "B": 0.3})
    assert parse_probabilities("A: 70%\nB: 30%", ["A", "B"]) == pytest.approx({"A": 0.7, "B": 0.3})
    assert parse_probabilities('{"accept": 2, "reject": 6}', ["accept", "reject"]) == pytest.approx({"accept": 0.25, "reject": 0.75})
    assert parse_probabilities("Probability: 0.8", ["yes", "no"]) == pytest.approx({"yes": 0.8, "no": 0.2})
    assert parse_probabilities("no numbers here", ["A", "B"]) is None


def test_parse_choice_and_score():
    assert parse_choice("I think... Answer: B", ["A", "B"]) == "B"
    assert parse_choice("<answer>(A)</answer>", ["A", "B"]) == "A"
    assert parse_choice("report", ["report", "silent"]) == "report"
    assert parse_score("reasoning... Score: 7", 0, 10) == 7
    assert parse_score("<score>3.5</score>", 0, 10) == 3.5
    assert parse_score("Score: 42", 0, 10) is None


def test_logprob_parsing_pools_variants():
    lp = [TokenLogprob(token="A", logprob=-0.2, top=[
        TopLogprob(token="A", logprob=-0.2), TopLogprob(token=" B", logprob=-2.0), TopLogprob(token="(A", logprob=-3.0)])]
    p = probs_from_logprobs(lp, ["A", "B"])
    ea, eb = math.exp(-0.2) + math.exp(-3.0), math.exp(-2.0)
    assert p["A"] == pytest.approx(ea / (ea + eb))


def test_split_reasoning():
    pub, r = split_reasoning("<thinking>secret plan</thinking>Public text")
    assert pub == "Public text" and r == "secret plan"


# ----------------------------------------------------------------------------- rewards

def test_score_transforms():
    p = {"A": 0.8, "B": 0.2}
    assert soa.score_probability(p, "A", "log") == pytest.approx(math.log(0.8))
    assert soa.score_probability(p, "A", "brier") == pytest.approx(-(0.2 ** 2 + 0.2 ** 2))
    assert soa.score_probability(p, "B", "accuracy") == 0.0
    assert soa.score_probability({"A": 0.5, "B": 0.5}, "B", "accuracy") == 0.5


def _synthetic_setup(n=30, **arg_kw):
    dom = SyntheticPersuasion(n_items=n, hint_strength=0.5, seed=1)
    items = dom.load()
    ctx = dom.context()
    arguer = synthetic_arguer(**arg_kw)
    judge = synthetic_judge(skill=1.0)
    return dom, items, ctx, arguer, judge


def _aff(mech_cls, **kw):
    return mech_cls(affordances={"agents": ["answer_key"]}, **kw)


def test_asd_positive_when_honesty_is_easier():
    dom, items, ctx, arguer, judge = _synthetic_setup(honest_mean=1.5, dishonest_mean=0.0, sd=0.5)
    exp = ASDExperiment([_aff(Propaganda), _aff(Consultancy, rounds=1), _aff(Debate, rounds=1)], items,
                        agent=arguer, fixtures={"judge": judge}, ctx=ctx)
    eps = exp.run()
    assert all(e.error is None for e in eps), [e.error for e in eps if e.error][:1]
    df = role_frame(eps)
    table = asd(df)
    assert set(table["mechanism"]) == {"propaganda", "consultancy", "debate"}
    assert (table["asd"] > 0).all()
    # debate: both debaters appear on both arms for every item
    deb = df[df.mechanism == "debate"]
    assert set(deb["value"]) == {1.0, -1.0}
    assert deb.groupby("item_id")["value"].nunique().min() == 2


def test_asd_negative_when_lying_is_easier():
    dom, items, ctx, arguer, judge = _synthetic_setup(honest_mean=0.0, dishonest_mean=2.0, sd=0.3)
    exp = ASDExperiment([_aff(Propaganda)], items, agent=arguer, fixtures={"judge": judge}, ctx=ctx)
    exp.run()
    assert (asd(exp.frame())["asd"] < 0).all()


def test_direct_judge_phantom_agent_asd_matches_hint():
    dom, items, ctx, arguer, judge = _synthetic_setup()
    exp = ASDExperiment([DirectJudge()], items, agent=arguer, fixtures={"judge": judge}, ctx=ctx)
    eps = exp.run()
    assert all(e.error is None for e in eps)
    t = asd(exp.frame())
    # direct judge ASD = mean over items of log p(true) - log p(false) from the hint
    expected = []
    for it in items:
        z = it.context["judge_hint"]
        pa = 1 / (1 + math.exp(-z))
        pt = pa if it.true_label == "A" else 1 - pa
        expected.append(math.log(pt) - math.log(1 - pt))
    assert t["asd"].iloc[0] == pytest.approx(np.mean(expected), rel=1e-6)


def test_graded_asd_reduces_to_asd_for_binary():
    dom, items, ctx, arguer, judge = _synthetic_setup(honest_mean=1.0, dishonest_mean=0.0)
    exp = ASDExperiment([_aff(Propaganda)], items, agent=arguer, fixtures={"judge": judge}, ctx=ctx)
    exp.run()
    df = exp.frame()
    a = asd(df)["asd"].iloc[0]
    g = graded_asd(df)["graded_asd"].iloc[0]
    # sum_c (v_c - vbar) u_c with v in {+1,-1}: = u_true - u_false
    assert g == pytest.approx(a)


def test_concordance_is_auc():
    import pandas as pd

    df = pd.DataFrame({"item_id": ["x"] * 4, "reward": [3, 1, 2, 0], "value": [1, 1, 0, 0]})
    c, n = pairwise_concordance(df)
    assert n == 4 and c == pytest.approx(0.75)


def test_open_probs():
    p = open_probs({"A": 0.0, "B": math.log(3)}, 1.0)
    assert p["B"] == pytest.approx(0.75)
    assert open_probs({"A": 1.0, "B": 0.0}, 0.0) == {"A": 1.0, "B": 0.0}


def test_rescore_counterfactual_rule():
    dom, items, ctx, arguer, judge = _synthetic_setup(n=10)
    exp = ASDExperiment([_aff(Propaganda)], items, agent=arguer, fixtures={"judge": judge}, ctx=ctx)
    eps = exp.run()
    brier = soa.rescore(eps, soa.JudgeScore("brier"))
    for e, b in zip(eps, brier):
        p = e.outcome.probs
        assert b.rewards["agent"] == pytest.approx(soa.score_probability(p, e.positions["agent"], "brier"))


def test_verification_changes_judge():
    dom, items, ctx, _, judge = _synthetic_setup(n=20)
    liar_claims = synthetic_arguer(honest_mean=0.5, dishonest_mean=0.5, sd=0.1, claim_rate=1.0, lie_claim_rate=1.0)
    no_ver = _aff(Propaganda)
    with_ver = _aff(Propaganda, verification=soa.VerificationPolicy(verifiers=["fact"]), name="propaganda_verified")
    exp = ASDExperiment([no_ver, with_ver], items, agent=liar_claims, fixtures={"judge": judge}, ctx=ctx)
    eps = exp.run()
    t = asd(exp.frame()).set_index("mechanism")["asd"]
    assert t["propaganda_verified"] > t["propaganda"] + 1.0
    verified = [e for e in eps if e.mechanism == "propaganda_verified"]
    statuses = {v.status for e in verified for v in e.verifications()}
    assert statuses == {"verified", "refuted"}
    assert any("<failed" in t.shown for e in verified for t in e.turns)


def test_mock_llm_policy_end_to_end():
    judge = soa.LLMPolicy(soa.models.MockModel(['I think A. {"A": 0.9, "B": 0.1}']), label="mockjudge")
    agent = soa.LLMPolicy(soa.models.MockModel(["<thinking>hidden</thinking>My argument for my side."]), cot=True)
    item = soa.binary_item("q", "2+2?", correct="4", incorrect="5", shuffle_seed=0)
    exp = ASDExperiment([Consultancy(rounds=1)], [item], agent=agent, fixtures={"judge": judge})
    eps = exp.run()
    assert all(e.error is None for e in eps)
    t = eps[0].turns[0]
    assert t.reasoning == "hidden" and "hidden" not in t.shown


def test_arm_profiles_debate_distinct_assignments():
    profs = arm_profiles(Debate(), agent="mock", fixtures={"judge": "mock"})
    assert len(profs) == 2
    item = soa.binary_item("q", "?", correct="x", incorrect="y")
    from so_arena.core.runner import build_players

    assigns = {tuple(sorted((r, p.stance) for r, p in build_players(pr, item).items() if r != "judge")) for pr in profs}
    assert len(assigns) == 2


def test_role_overrides_affordances():
    mech = Debate(affordances={"agents": ["passage"]}, sees_reasoning={"judge": ["debater_a"]})
    specs = mech.role_specs()
    assert specs["debater_a"].affordances == ["passage"] and specs["judge"].affordances == []
    assert specs["judge"].sees_reasoning_of == ["debater_a"]


def test_repeats_draw_independent_samples_under_cache(tmp_path):
    from so_arena.models.cache import CachedModel, ResponseCache

    calls = []

    def fn(messages, options, i):
        calls.append(i)
        return f"sample {i}"

    model = CachedModel(soa.models.FunctionModel(fn, name="counting"), ResponseCache.at(tmp_path))
    agent = soa.LLMPolicy(model)
    judge = soa.ScriptedPolicy('{"A": 0.5, "B": 0.5}')
    item = soa.binary_item("q", "?", correct="x", incorrect="y", shuffle_seed=0)
    prof = soa.Profile(name="p", players={"agent": soa.PlayerSpec(policy=agent, stance="true"), "judge": judge})
    eps = soa.run_sync(soa.run_episodes(Propaganda(), [item], [prof], repeats=3))
    texts = {e.turns[0].text for e in eps}
    assert len(texts) == 3  # three distinct samples, not one cached completion replayed
    eps2 = soa.run_sync(soa.run_episodes(Propaganda(), [item], [prof], repeats=3))
    assert {e.turns[0].text for e in eps2} == texts and len(calls) == 3  # re-running hits the cache
