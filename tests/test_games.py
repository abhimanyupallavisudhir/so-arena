import numpy as np
import pytest

from so_arena.games import NormalFormGame, game_from_function, zero_sum_value
from so_arena.theory import whistleblower as wb
from so_arena.theory.audits import audit_corrected, label_efficiency


def two_player(A, B, s1=("a", "b"), s2=("a", "b")):
    return NormalFormGame(["row", "col"], {"row": list(s1), "col": list(s2)}, {"row": np.array(A), "col": np.array(B)})


def test_prisoners_dilemma():
    # strategies: cooperate, defect
    g = two_player([[3, 0], [5, 1]], [[3, 5], [0, 1]], ("C", "D"), ("C", "D"))
    assert g.pure_nash() == [(1, 1)]
    assert g.dominant_strategies(strict=True) == {"row": ["D"], "col": ["D"]}


def test_matching_pennies_support_enumeration():
    g = two_player([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    eqs = g.support_enumeration()
    assert len(eqs) == 1
    assert eqs[0][0] == pytest.approx([0.5, 0.5]) and eqs[0][1] == pytest.approx([0.5, 0.5])
    v, x, y = zero_sum_value(np.array([[1, -1], [-1, 1]]))
    assert v == pytest.approx(0.0) and x == pytest.approx([0.5, 0.5])


def test_rps_value_and_fictitious_play():
    A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    v, x, _ = zero_sum_value(A)
    assert v == pytest.approx(0.0, abs=1e-9) and x == pytest.approx(np.ones(3) / 3, abs=1e-6)
    g = two_player(A, -A, ("R", "P", "S"), ("R", "P", "S"))
    fp = g.fictitious_play(3000)
    assert g.nash_conv(fp) < 0.05


def test_coordination_game_equilibria_and_ce_welfare():
    # stag hunt
    A = np.array([[4, 0], [3, 3]])
    g = two_player(A, A.T, ("stag", "hare"), ("stag", "hare"))
    assert set(g.pure_nash()) == {(0, 0), (1, 1)}
    assert len(g.support_enumeration()) == 3  # two pure + one mixed
    g.outcomes["welfare"] = g.payoffs["row"] + g.payoffs["col"]
    lo, hi = g.outcome_range("welfare")
    assert hi == pytest.approx(8.0) and lo <= 6.0 + 1e-9
    joint = g.regret_matching(4000)
    assert joint.sum() == pytest.approx(1.0)


def test_replicator_basins_stag_hunt():
    A = np.array([[4, 0], [3, 3]])
    g = two_player(A, A.T)
    xf, _ = g.replicator([np.array([0.9, 0.1]), np.array([0.9, 0.1])])
    assert xf[0][0] > 0.99
    xf, _ = g.replicator([np.array([0.2, 0.8]), np.array([0.2, 0.8])])
    assert xf[0][1] > 0.99


def test_whistleblower_common_reward_silence_weakly_dominant():
    g = wb.whistleblower_game(3, R=1.0, delta=0.5, s=0.0)
    dom = g.dominant_strategies()
    assert all("silent" in v for v in dom.values())
    eqs = wb.symmetric_equilibria(3, R=1.0, delta=0.5, s=0.0)
    assert any(e["p_report"] == 0.0 and e["stable"] for e in eqs)


def test_whistleblower_thresholds_match_closed_form():
    for n in (2, 3, 6):
        s, d = 0.2, 0.5
        eqs = wb.symmetric_equilibria(n, R=1.0, delta=d, s=s, split="each")
        mixed = [e["p_report"] for e in eqs if e["kind"] == "mixed"]
        assert len(mixed) == 1
        assert mixed[0] == pytest.approx(wb.mixed_threshold_each(n, s, d), abs=1e-6)
        assert not [e for e in eqs if e["kind"] == "mixed"][0]["stable"]
        assert {e["p_report"] for e in eqs if e["stable"]} == {0.0, 1.0}
    # large bounty: reporting dominant, unique equilibrium
    g = wb.whistleblower_game(3, R=1.0, delta=0.5, s=0.6)
    assert g.pure_nash() == [(1, 1, 1)]


def test_whistleblower_game_matches_symmetric_analysis():
    g = wb.whistleblower_game(3, R=1.0, delta=0.5, s=0.2)
    assert g.is_symmetric()
    pure = g.pure_nash()
    assert (0, 0, 0) in pure and (1, 1, 1) in pure
    p = wb.mixed_threshold_each(3, 0.2, 0.5)
    f = g.symmetric_payoff(np.array([1 - p, p]))
    assert f[0] == pytest.approx(f[1], abs=1e-9)
    sym = g.symmetric_equilibria()
    assert len(sym) == 3
    # silent basin shrinks with team size
    sw = wb.sweep([0.2], ns=(2, 5, 20), delta=1.0)
    thr = sw["p_threshold"].tolist()
    assert thr[0] == pytest.approx(0.8) and thr == sorted(thr, reverse=True)


def test_collective_liability_can_make_reporting_dominant():
    # bounty alone too small, but independent audits + penalties for silence tip it
    g = wb.whistleblower_game(3, R=1.0, delta=0.5, s=0.1, q=0.5, c=1.0)
    assert g.pure_nash() == [(1, 1, 1)]


def test_audit_control_variate_unbiased_and_efficient():
    rng = np.random.default_rng(0)
    n = 20000
    v = rng.normal(size=n)
    u = v + 0.5 * rng.normal(size=n)  # correlated mechanism reward
    p = 0.1
    est = []
    for s in range(200):
        a = rng.random(n) < p
        est.append(audit_corrected(u, v, a, p).mean())
    assert np.mean(est) == pytest.approx(v.mean(), abs=0.01)
    assert label_efficiency(u, v) == pytest.approx(1 / (1 - np.corrcoef(u, v)[0, 1] ** 2))


def test_game_from_function_three_players():
    g = game_from_function(["a", "b", "c"], {p: ["x", "y"] for p in "abc"},
                           lambda names: {p: float(sum(v == names[p] for v in names.values())) for p in names})
    assert set(g.pure_nash()) == {(0, 0, 0), (1, 1, 1)}
