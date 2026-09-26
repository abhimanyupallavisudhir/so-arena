import itertools
import math

import numpy as np
import pandas as pd
import pytest

import so_arena as soa
from so_arena.analysis.optimization import (
    BestOfN,
    Tilted,
    Uniform,
    evaluate_tree,
    first_order_gain,
    optimization_grid,
    pool_curve,
)
from so_arena.core.runner import Profile, PlayerSpec, build_players, run_sync
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.mechanisms import Debate, Propaganda
from so_arena.samplers.pools import OptimizationExperiment, expand_tree


def brute_bon_unbiased(w, n):
    k = len(w)
    probs = np.zeros(k)
    subsets = list(itertools.combinations(range(k), n))
    for s in subsets:
        m = max(w[i] for i in s)
        winners = [i for i in s if w[i] == m]
        for i in winners:
            probs[i] += 1 / len(winners)
    return probs / len(subsets)


def brute_bon_plugin(w, n):
    k = len(w)
    probs = np.zeros(k)
    for s in itertools.product(range(k), repeat=n):
        m = max(w[i] for i in s)
        winners = sorted({i for i in s if w[i] == m})
        # ties among *distinct candidates* with equal payoff: split uniformly over tied candidates
        tied = [i for i in range(k) if w[i] == m]
        for i in tied:
            probs[i] += 1 / len(tied)
    return probs / k ** n


@pytest.mark.parametrize("w", [[0.1, 0.5, 0.3, 0.9, 0.2], [1, 1, 0, 2, 2, 0], [3, 3, 3]])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_bon_exact(w, n):
    w = np.array(w, dtype=float)
    if n <= len(w):
        assert BestOfN(n, "unbiased").probs(w) == pytest.approx(brute_bon_unbiased(list(w), n))
    assert BestOfN(n, "plugin").probs(w) == pytest.approx(brute_bon_plugin(list(w), n))


def test_tilted_limits_and_kl():
    w = np.array([0.0, 1.0, 2.0])
    assert Tilted(0.0).probs(w) == pytest.approx(np.ones(3) / 3)
    assert Tilted(50.0).probs(w)[2] == pytest.approx(1.0, abs=1e-6)
    assert Uniform().kl(w) == pytest.approx(0.0)
    assert BestOfN(2).kl(w) > 0


def test_first_order_gain_is_derivative():
    rng = np.random.default_rng(0)
    rows = []
    for item in range(20):
        u = rng.normal(size=10)
        v = 0.5 * u + rng.normal(size=10)
        rows += [{"item_id": item, "reward": a, "value": b} for a, b in zip(u, v)]
    pool = pd.DataFrame(rows)
    eps = 1e-4
    c = pool_curve(pool, selections=[Tilted(eps), Tilted(-eps)])
    numeric = (c["value"].iloc[0] - c["value"].iloc[1]) / (2 * eps)
    assert numeric == pytest.approx(first_order_gain(pool), rel=1e-3)


def _setup(n=6, **kw):
    dom = SyntheticPersuasion(n_items=n, seed=3)
    items = dom.load()
    ctx = dom.context()
    arguer = synthetic_arguer(**kw)
    judge = synthetic_judge()
    return items, ctx, arguer, judge


def test_single_level_tree_matches_pool():
    items, ctx, arguer, judge = _setup(honest_mean=0.0, dishonest_mean=0.0, sd=1.0)
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    prof = Profile(name="p", players={"agent": PlayerSpec(policy=arguer, stance="false"), "judge": judge})
    item = items[0]
    tree, eps = run_sync(expand_tree(mech, item, build_players(prof, item), pool_sizes={"agent": 5}, ctx=ctx,
                                     keep_episodes=True))
    assert tree.n_leaves == 5 and len(tree.nodes) == 1
    root = tree.nodes[tree.root]
    rewards = np.array([tree.leaves[c].rewards["agent"] for c in root.children])
    assert len(set(np.round(rewards, 8))) == 5  # distinct samples
    best = evaluate_tree(tree, {"agent": BestOfN(5)})
    assert best.rewards["agent"] == pytest.approx(rewards.max())
    base = evaluate_tree(tree)
    assert base.rewards["agent"] == pytest.approx(rewards.mean())


def test_tree_replay_is_deterministic():
    items, ctx, arguer, judge = _setup()
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    prof = Profile(name="p", players={"agent": PlayerSpec(policy=arguer, stance="true"), "judge": judge})
    item = items[1]
    t1, _ = run_sync(expand_tree(mech, item, build_players(prof, item), pool_sizes={"agent": 4}, ctx=ctx))
    t2, _ = run_sync(expand_tree(mech, item, build_players(prof, item), pool_sizes={"agent": 4}, ctx=ctx))
    r1 = sorted(l.rewards["agent"] for l in t1.leaves.values())
    r2 = sorted(l.rewards["agent"] for l in t2.leaves.values())
    assert r1 == pytest.approx(r2)


def test_simultaneous_debate_tree_uniform_equals_leaf_mean():
    items, ctx, arguer, judge = _setup()
    mech = Debate(rounds=1, affordances={"agents": ["answer_key"]})
    prof = Profile(name="p", players={"debater_a": PlayerSpec(policy=arguer, stance="true"),
                                      "debater_b": PlayerSpec(policy=arguer, stance="false"), "judge": judge})
    item = items[0]
    tree, _ = run_sync(expand_tree(mech, item, build_players(prof, item), pool_sizes={"debater_a": 3, "debater_b": 3}, ctx=ctx))
    assert tree.n_leaves == 9
    # B's pool is shared across A's candidates (same information set)
    root = tree.nodes[tree.root]
    b_keys = {tree.nodes[c].key for c in root.children}
    assert len(b_keys) == 1
    uni = evaluate_tree(tree)
    mean_a = np.mean([l.rewards["debater_a"] for l in tree.leaves.values()])
    assert uni.rewards["debater_a"] == pytest.approx(mean_a)
    opt = evaluate_tree(tree, {"debater_a": BestOfN(3), "debater_b": BestOfN(3)})
    assert np.isfinite(opt.rewards["debater_a"])


def test_goodhart_sophistry_found_by_bon():
    # honest arguments better on average, but dishonest ones have a heavy tail (sophistry):
    # optimizing the liar finds it, so the liar's reward rises steeply with n.
    items, ctx, arguer, judge = _setup(n=8, honest_mean=1.0, dishonest_mean=0.0, sd=0.3,
                                       sophistry_rate=0.2, sophistry_boost=4.0)
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    prof = Profile(name="liar", players={"agent": PlayerSpec(policy=arguer, stance="false"), "judge": judge})
    exp = OptimizationExperiment(mech, items, prof, pool_sizes={"agent": 8}, ctx=ctx)
    exp.run()
    grid = exp.grid({"agent": [1, 8]})
    r1, r8 = grid["reward_agent"].tolist()
    assert r8 > r1 + 0.5
    # and the judge is fooled more often
    assert grid["judge_correct"].iloc[1] < grid["judge_correct"].iloc[0]
