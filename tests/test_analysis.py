import math

import numpy as np
import pandas as pd
import pytest

from oversight_arena.analysis import EmpiricalGame, Node, alignment, asd, bon_curve, bon_weights, frontier, tree_value


def df_from(rows):
    d = pd.DataFrame(rows)
    d["error"] = False
    return d


def test_asd_matches_definition():
    rows = []
    for t in range(5):
        rows += [{"mechanism": "m", "task": t, "role": "a", "reward": math.log(0.8), "gt_correct": 1.0, "strategy_name": "h"},
                 {"mechanism": "m", "task": t, "role": "a", "reward": math.log(0.6), "gt_correct": 0.0, "strategy_name": "d"}]
    a = asd(df_from(rows)).iloc[0]
    assert a["asd"] == pytest.approx(math.log(0.8) - math.log(0.6))
    al = alignment(df_from(rows)).iloc[0]
    assert al["pairwise_acc"] == 1.0 and al["spearman"] == pytest.approx(1.0)
    f = frontier(df_from(rows)).iloc[0]
    assert f["frontier_gap"] == pytest.approx(math.log(0.8) - math.log(0.6)) and f["argmax_gt"] == 1.0


def test_perverse_incentives_detected():
    rows = []
    for t in range(4):
        for k, (r, g) in enumerate([(0.9, 0.0), (0.5, 1.0), (0.4, 1.0)]):
            rows.append({"mechanism": "m", "task": t, "role": "a", "reward": r, "gt_correct": g, "strategy_name": f"s{k}"})
    al = alignment(df_from(rows)).iloc[0]
    assert al["pairwise_acc"] == 0.0
    assert frontier(df_from(rows)).iloc[0]["frontier_gap"] < 0


def test_bon_weights_exact():
    w = bon_weights([1, 2, 3, 4], 2)
    assert np.allclose(w, [1 / 16, 3 / 16, 5 / 16, 7 / 16])
    assert np.allclose(bon_weights([1, 1, 2], 1), [1 / 3] * 3)
    assert np.allclose(bon_weights([5, 1], 3, maximize=False), [1 / 8, 7 / 8])


def test_bon_curve_goodhart():
    rows = []
    for t in range(6):
        for s in range(8):
            r = s / 8
            rows.append({"mechanism": "m", "task": t, "role": "a", "reward": r, "gt_correct": 1.0 if s < 6 else 0.0})
    c = bon_curve(df_from(rows), gt="correct", n_values=[1, 4, 64])
    assert c["reward"].is_monotonic_increasing and c["gt"].is_monotonic_decreasing


def test_tree_value_minmax():
    # proposer picks among 2 proposals; critic picks among critiques (min)
    good = Node(gt=1.0, children=[Node(payoff=0.6), Node(payoff=0.6)])
    bad = Node(gt=0.0, children=[Node(payoff=0.95), Node(payoff=0.35)])  # a flaw only a critic exposes
    root = Node(children=[good, bad])
    pay, gt = tree_value(root, ks=[64, 64], maximize=[True, False])  # optimised critic -> good proposal wins
    assert gt == pytest.approx(1.0, abs=1e-6) and pay == pytest.approx(0.6, abs=1e-3)
    pay, gt = tree_value(root, ks=[64, 1], maximize=[True, False])  # unoptimised critic -> bad proposal wins
    assert gt < 0.01 and pay == pytest.approx(0.65, abs=1e-3)


def test_games():
    g = EmpiricalGame.from_matrices([[1, -1], [-1, 1]])
    eq = g.nash()
    assert len(eq) == 1 and np.allclose(eq[0].mix["row"], [0.5, 0.5])
    v, mix = g.zero_sum_value()
    assert v == pytest.approx(0.0, abs=1e-9)
    A = np.array([[4, 0], [3, 3]])
    sh = EmpiricalGame.from_matrices(A, A.T, row_strats=["S", "H"], col_strats=["S", "H"])
    kinds = sorted(tuple(np.round(e.mix["row"], 2)) for e in sh.nash())
    assert kinds == [(0.0, 1.0), (0.75, 0.25), (1.0, 0.0)]
    assert sh.regret(sh.pure({"row": "S", "col": "H"}))["row"] == pytest.approx(3.0)
    fin = sh.replicator({"row": np.array([0.9, 0.1]), "col": np.array([0.9, 0.1])}, iters=300, lr=0.5)[-1]
    assert fin["row"][0] > 0.99


def test_three_player_game_from_results():
    from oversight_arena.domains.swarm import AbstractSwarm
    from oversight_arena.mechanisms import Swarm, TeamReward
    from oversight_arena.sim import SwarmWorker
    import oversight_arena as oa

    dom = AbstractSwarm(n_tasks=6)
    strat = [oa.Strategy(name="honest", params={"cheat": 0}), oa.Strategy(name="cheat", params={"cheat": 1})]
    ws = ["worker_1", "worker_2", "worker_3"]
    res = oa.Experiment(dom, Swarm(n_workers=3, rounds=1, reward=TeamReward()), {"*": SwarmWorker()},
                        oa.Cartesian(strategies={w: strat for w in ws}), progress=False).run()
    g = EmpiricalGame.from_results(res, ws, gt_metrics=("clean",))
    assert g.shape == (2, 2, 2)
    # under a shared reward, cheating weakly dominates: all-cheat is a pure Nash equilibrium
    assert any(all(e.mix[w][g.strategies[w].index("cheat")] == 1 for w in ws) for e in g.pure_nash())
