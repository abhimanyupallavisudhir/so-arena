"""Recursive debate and prover-estimator debate: payoffs, coherence clipping, and the obfuscation property."""

from __future__ import annotations

import pytest

from so_arena.core.items import binary_item
from so_arena.core.policy import FunctionPolicy
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.mechanisms import ProverEstimatorDebate, RecursiveDebate

ITEM = binary_item("q1", "Is 2+2=4?", "yes", "no", shuffle_seed=None)  # A = yes (true), B = no


def play(mech, prover, other, judge, stance="A"):
    roles = list(mech.role_specs())
    other_role = [r for r in roles if r not in ("prover", "judge")][0]
    prof = Profile(name="p", players={"prover": PlayerSpec(policy=prover, stance=stance),
                                      other_role: PlayerSpec(policy=other), "judge": PlayerSpec(policy=judge)})
    eps = run_sync(run_episodes(mech, [ITEM], [prof]))
    assert not eps[0].error, eps[0].error
    return eps[0]


def prover(direction="too_low", leaf_direction="too_low", recurse=1, subclaims=("s1", "s2")):
    def fn(req, ctx):
        if req.phase.startswith("decompose"):
            return {"direction": direction, "subclaims": list(subclaims)}
        if req.phase.startswith("choose"):
            return {"recurse": recurse}
        if req.phase == "leaf":
            return {"direction": leaf_direction, "argument": "trust me"}
        raise AssertionError(req.phase)
    return FunctionPolicy(fn)


def estimator(p_top=0.5, p_sub=(0.9, 0.9, 0.9), p_all=0.7):
    def fn(req, ctx):
        if req.phase == "estimate0":
            return {"probability": p_top}
        return {"probabilities": list(p_sub), "all_true": p_all}
    return FunctionPolicy(fn)


def judge(p_true):
    return FunctionPolicy(lambda req, ctx: {"true": p_true, "false": 1 - p_true})


def test_prover_estimator_payoffs_follow_the_paper():
    mech = ProverEstimatorDebate(depth=1, width=2, rho=2.0)
    ep = play(mech, prover(), estimator(p_top=0.5, p_sub=(0.9, 0.8, 0.95), p_all=0.7), judge(0.95))
    r = 0.5
    # start: r^(d+1) p_d; round 1: r^1 * (+1) * (p_all - p_d); leaf: (+1) * (P_judge - p_0), p_0 = p_sub[0]
    expected = [r ** 2 * 0.5, r * (0.7 - 0.5), 0.95 - 0.9]
    assert ep.outcome.data["reward_terms"] == pytest.approx(expected)
    assert ep.rewards["prover"] == pytest.approx(sum(expected))
    assert ep.rewards["estimator"] == pytest.approx(-sum(expected))
    # bridge subclaim appended to every decomposition
    assert ep.outcome.data["path"][0]["subclaims"][-1].startswith("If all of the other subclaims")


def test_joint_probability_is_clipped_to_frechet_bounds():
    mech = ProverEstimatorDebate(depth=1, width=2)
    # marginals 0.9, 0.9, 0.9 force P(all) >= 0.7; an incoherent 0.2 is raised to 0.7
    ep = play(mech, prover(), estimator(p_sub=(0.9, 0.9, 0.9), p_all=0.2), judge(0.5))
    assert ep.outcome.data["path"][0]["p_all"] == pytest.approx(0.7)
    # P(all) can never exceed the smallest marginal
    ep = play(mech, prover(), estimator(p_sub=(0.9, 0.3, 0.9), p_all=0.95), judge(0.5))
    assert ep.outcome.data["path"][0]["p_all"] == pytest.approx(0.3)


def test_calibrated_estimator_defeats_an_obfuscated_false_argument():
    """A false claim split into many plausible steps: the estimator prices each at 0.9 without finding the flaw,
    P(all) = 0.9^5 ~ 0.59 << 1; the prover cannot profit at any step it cannot show is mispriced."""
    mech = ProverEstimatorDebate(depth=2, width=4, rho=2.0)
    subs = [f"step {i}" for i in range(4)]
    p_sub = (0.9,) * 5
    ep = play(mech, prover(direction="too_low", subclaims=subs), estimator(p_top=0.3, p_sub=p_sub, p_all=0.9 ** 5),
              judge(0.9), stance="B")
    # the judge agrees with the estimator at the leaf (0.9): the prover's only gains are the start term
    assert ep.rewards["prover"] < 0.2
    assert ep.outcome.decision in ("A", "B")


def test_honest_prover_profits_from_an_underpricing_estimator():
    """Completeness: an estimator that underprices a true claim loses at the leaf, where weights are largest."""
    mech = ProverEstimatorDebate(depth=2, width=2, rho=2.0)
    ep = play(mech, prover(direction="too_low"), estimator(p_top=0.2, p_sub=(0.2, 0.2, 0.9), p_all=0.1), judge(0.99))
    assert ep.rewards["prover"] > 0.5
    assert ep.outcome.decision == "A"


def test_recursive_debate_pays_the_leaf_judgment():
    mech = RecursiveDebate(depth=2, width=2)
    opp = FunctionPolicy(lambda req, ctx: {"dispute": 2} if req.kind == "json" else "it is false")
    prov = FunctionPolicy(lambda req, ctx: {"subclaims": ["x", "y"]} if req.kind == "json" else "it is true")
    ep = play(mech, prov, opp, judge(0.25))
    assert ep.rewards == {"prover": pytest.approx(0.25), "opponent": pytest.approx(0.75)}
    assert ep.outcome.decision == "B"  # judge rejected the prover's final claim
    assert len(ep.outcome.data["path"]) == 2


def test_estimates_are_salvaged_from_arithmetic():
    from so_arena.mechanisms.recursive import _salvage_estimates

    d = _salvage_estimates('{"probabilities": [0.83, 0.99, 90%], "all_true": 0.83 * 0.99 * 0.9}')
    assert d["probabilities"] == pytest.approx([0.83, 0.99, 0.9])
    assert d["all_true"] == pytest.approx(0.83 * 0.99 * 0.9)
