"""Swarms with partial observation: the analytic model (Propositions 5 and 6), the Team mechanism's
observation, self-reports, leniency and misprision, and learning dynamics (natural vs. vanilla gradient)."""

import math

import numpy as np
import pytest
from scipy.stats import binom

import so_arena as soa
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.domains.synthetic import SHORTCUT_BRIEF, SyntheticTeam, team_worker
from so_arena.games import EmpiricalGameExperiment, NormalFormGame
from so_arena.games.learning import StrategyGradient, natural_estimate, policy_gradient, vanilla_estimate
from so_arena.mechanisms import Team
from so_arena.theory import whistleblower as wb

# the synthetic team's numbers: hacked score 0.9, honest 0.6 (every member's stake: 0.3)
TEAM = dict(R=0.9, delta=0.3, split="shared")


def run(mech, items, profiles, ctx=None, gt=None, seed=0):
    eps = run_sync(run_episodes(mech, items, profiles, ctx=ctx, ground_truth=gt, seed=seed))
    errs = [e.error for e in eps if e.error]
    assert not errs, errs[0]
    return eps


def profile(dom, workers):
    players = {"grader": dom.grader()}
    for i, (work, report) in enumerate(workers):
        players[f"worker_{i + 1}"] = PlayerSpec(policy=team_worker(work=work, report=report), label=f"{work}·{report}")
    return Profile(name="|".join(f"{w}·{r}" for w, r in workers), players=players)


# ----------------------------------------------------------------------------- Proposition 5


def test_report_equilibrium_bounty_is_a_third_of_the_stake_for_two_witnesses_at_o_08():
    # a team of 3 (two potential witnesses of a violation), o = 0.8, no audits: b_R = g/3
    for g in (0.3, 1.0):
        assert wb.report_equilibrium_bounty(2, delta=g, o=0.8, split="shared") == pytest.approx(g / 3)
        assert wb.report_equilibrium_bounty(2, delta=g, o=0.8, split="first") == pytest.approx(g / 3)
        # paid to each verified reporter: P(K = 0) g
        assert wb.report_equilibrium_bounty(2, delta=g, o=0.8, split="each") == pytest.approx(0.2 * g)
        assert wb.dominance_bounty(2, delta=g, o=0.8, split="shared") == pytest.approx(g)


def test_b_r_matches_the_closed_form_with_misprision_and_audits():
    n, o, d, q, c, m = 3, 0.5, 1.0, 0.2, 0.3, 0.02
    P = binom.pmf(np.arange(n), n - 1, o)
    T = (1 - q) * d - (c + m) * q
    closed = (P[0] * T - (1 - P[0]) * m) / (P[0] + sum(P[k] / (1 + k) for k in range(1, n)))
    kw = dict(delta=d, o=o, q=q, c=c, m=m, split="shared")
    assert closed > 0 and wb.report_equilibrium_bounty(n, **kw) == pytest.approx(closed)
    assert wb.dominance_bounty(n, **kw) == pytest.approx(T)
    # the advantage of reporting, K = 0 vs K >= 1 (Proposition 5)
    s = 0.25
    assert wb.advantage(0, s=s, **kw) == pytest.approx(s - T)
    for k in range(1, n):
        assert wb.advantage(k, s=s, **kw) == pytest.approx(m + s / (k + 1))


def test_full_observation_recovers_the_closed_forms():
    for n in (2, 3, 6):
        s, d = 0.2, 0.5
        eqs = wb.symmetric_equilibria(n, R=1.0, delta=d, s=s, o=1.0)
        mixed = [e["p_report"] for e in eqs if e["kind"] == "mixed"]
        assert mixed == [pytest.approx(1 - (s / d) ** (1 / (n - 1)), abs=1e-6)]
        assert wb.report_equilibrium_bounty(n, delta=d) == 0.0  # everyone sees it: never alone
        for p in (0.0, 0.3, 1.0):  # the model without observation noise: E_{K~Bin(n-1,p)}[s - delta 1{K=0}]
            assert wb.report_gain(p, n, R=1.0, delta=d, s=s) == pytest.approx(s - d * (1 - p) ** (n - 1))
    # observation probability o only thins the other reporters: Bin(n-1, o p)
    assert wb.report_gain(0.5, 3, delta=0.5, s=0.2, o=0.6) == pytest.approx(wb.report_gain(0.3, 3, delta=0.5, s=0.2))


def test_regimes_under_partial_observation():
    kw = dict(TEAM, o=0.8)
    # below b_R = 0.1 silence is the only equilibrium although the bounty is positive
    assert wb.regime(2, s=0.05, **kw) == "silence"
    g = wb.whistleblower_game(2, s=0.05, **kw)
    assert g.pure_nash() == [(0, 0)]
    # between b_R and the stake: coordination with a unique mixed equilibrium between the two
    assert wb.regime(2, s=0.2, **kw) == "coordination"
    eqs = wb.symmetric_equilibria(2, s=0.2, **kw)
    assert [e["kind"] for e in eqs].count("mixed") == 1
    assert set(wb.whistleblower_game(2, s=0.2, **kw).strict_nash()) == {(0, 0), (1, 1)}
    # above the stake reporting is dominant
    assert wb.regime(2, s=0.45, **kw) == "reporting"
    assert wb.whistleblower_game(2, s=0.45, **kw).dominant_strategies(strict=True) == {"agent_1": ["report"],
                                                                                       "agent_2": ["report"]}
    # with full observation the same small bounty makes a coordination game
    assert wb.regime(2, s=0.05, **TEAM) == "coordination"


@pytest.mark.parametrize("n,o,q,c,m", [(3, 0.5, 0.0, 0.0, 0.0), (5, 0.9, 0.2, 0.1, 0.05), (4, 0.3, 0.1, 0.0, 0.2)])
def test_one_mixed_equilibrium_between_b_r_and_the_stake(n, o, q, c, m):
    kw = dict(delta=1.0, o=o, q=q, c=c, m=m, split="shared")
    lo, hi = wb.report_equilibrium_bounty(n, **kw), wb.dominance_bounty(n, **kw)
    for s in np.linspace(lo, hi, 7)[1:-1]:
        eqs = wb.symmetric_equilibria(n, s=float(s), **kw)
        assert [e["kind"] for e in eqs].count("mixed") == 1  # Descartes' rule of signs
        assert wb.regime(n, s=float(s), **kw) == "coordination"


def test_risk_dominance_for_two_witnesses():
    # shared bounty, o = 1: reporting is risk dominant iff 3/2 s + m + (c + m) q > (1 - q) delta
    for s, m, q, c in [(0.25, 0, 0, 0), (0.15, 0, 0, 0), (0.1, 0.1, 0.2, 0.1), (0.1, 0.0, 0.2, 0.1)]:
        want = 1.5 * s + m + (c + m) * q > (1 - q) * 0.3
        got = wb.risk_dominant(delta=0.3, s=s, m=m, q=q, c=c, split="shared")
        assert got == ("all report" if want else "all silent")


def test_unknown_parameters_are_rejected():
    with pytest.raises(TypeError):
        wb.report_gain(0.5, 3, delta=0.5, bounty=0.2)


def test_ex_ante_game_and_reverted_outcome():
    kw = dict(TEAM, s=0.2, o=0.7, q=0.1)
    g = wb.whistleblower_game(3, **kw)
    assert g.is_symmetric()
    # the ex-ante gain of "report if you witness it" is o x the gain of a witness
    for p in (0.2, 0.6):
        f = g.symmetric_payoff(np.array([1 - p, p]))
        assert f[1] - f[0] == pytest.approx(0.7 * wb.report_gain(p, 3, **kw))
    assert g.outcomes["reverted"][(1, 1, 1)] == pytest.approx(1 - 0.9 * 0.3 ** 3)


# ----------------------------------------------------------------------------- the whole game


def test_nobody_violating_is_nash_but_not_subgame_perfect():
    kw = dict(TEAM, s=0.05, P=0.3, o=0.8)  # s < b_R = 0.1
    whole = wb.team_game(2, **kw)
    honest_report = whole.index({p: "honest·report" for p in whole.players})
    assert whole.is_nash(honest_report)  # the threat of reports deters every violation ...
    # ... but it is never carried out: after a violation, a witness prefers silence
    assert not wb.whistleblower_game(2, **kw).is_nash((1, 1))
    assert wb.offender_gain(1.0, 2, **kw) < 0 < wb.offender_gain(0.0, 2, **kw)


def test_natural_gradient_on_the_whole_game_is_the_mean_field():
    """Proposition 6: multiplicative weights keep the two decisions independent; the marginals follow
    the two-dimensional mean field exactly."""
    kw = dict(TEAM, s=0.2, P=0.3, o=0.8)
    g = wb.team_game(2, **kw)
    df = policy_gradient(g, init=[wb.team_mix(0.5, 0.3)] * 3, shared=[g.players], lr=1.0, steps=60,
                         natural=True, record_every=20)
    mf = wb.mean_field(0.5, 0.3, 2, steps=60, lr=1.0, **kw).set_index("step")
    for _, row in df.iterrows():
        mix = [row[f"p_worker_1_{s}"] for s in wb.TEAM_STRATEGIES]
        x, p = wb.team_marginals(mix)
        assert x == pytest.approx(mf.loc[row["step"], "x"], abs=1e-9)
        assert p == pytest.approx(mf.loc[row["step"], "p"], abs=1e-9)
        assert np.outer([1 - x, x], [1 - p, p]).ravel() == pytest.approx(mix, abs=1e-9)  # still a product


def test_training_algorithm_selects_the_equilibrium():
    kw = dict(TEAM, s=0.15, P=0.3, o=0.8)  # coordination: both "violate, silent" and "deterred" are stable
    g = wb.team_game(2, **kw)
    final = {}
    for natural in (True, False):
        df = policy_gradient(g, init=[wb.team_mix(0.5, 0.6)] * 3, shared=[g.players], lr=2.0, steps=2000,
                             natural=natural, record_every=2000)
        final[natural] = wb.team_marginals([df.iloc[-1][f"p_worker_1_{s}"] for s in wb.TEAM_STRATEGIES])[0]
    assert final[True] < 0.05 and final[False] > 0.95  # same start: NPG deters, REINFORCE entrenches
    # the mean field's basins differ too
    kw = dict(TEAM, s=0.3, P=0.3, o=0.8)
    assert wb.basin_of_deterrence(2, natural=True, **kw) > wb.basin_of_deterrence(2, natural=False, **kw)


def test_natural_and_vanilla_gradient_share_rest_points():
    # KL-regularized ascent for one player: both reach the tilted policy softmax(u / tau)
    u = np.array([0.0, 1.0, 0.5])
    g = NormalFormGame(["p"], {"p": ["a", "b", "c"]}, {"p": u})
    target = np.exp(u / 0.5) / np.exp(u / 0.5).sum()
    for natural in (True, False):
        last = policy_gradient(g, lr=0.5, steps=4000, kl=0.5, natural=natural).iloc[-1]
        assert np.array([last["p_p_a"], last["p_p_b"], last["p_p_c"]]) == pytest.approx(target, abs=1e-3)


# ----------------------------------------------------------------------------- mechanism and domain


def test_observation_is_a_chance_move_recorded_as_witnesses():
    dom = SyntheticTeam(n_items=60)
    items, ctx = dom.load(), dom.context()
    mech = Team(n_workers=3, observe_prob=0.5, regrade_on_report=True, reward=soa.Whistleblower(bounty=0.5))
    a = run(mech, items, [profile(dom, [("hack", "silent"), ("honest", "report"), ("honest", "report")])], ctx=ctx,
            gt=dom.ground_truth_scorers())
    b = run(mech, items, [profile(dom, [("hack", "report"), ("honest", "silent"), ("honest", "report")])], ctx=ctx)
    seen = []
    for ea, eb in zip(a, b):
        w = ea.outcome.data["witnesses"]
        assert w == eb.outcome.data["witnesses"]  # common random numbers across profiles
        seen += [v in w["worker_1"] for v in ("worker_2", "worker_3")]
        # only a witness can report; the grader sees every contribution
        assert {r for r, x in ea.outcome.data["reports"].items() if x["reported"]} == set(w["worker_1"])
        assert ea.outcome.data["team_score"] == pytest.approx(0.9)
        # ground truth: silence is fine for a worker who did not see the hack
        assert ea.ground_truth["role_values"]["worker_2"] == 1.0
    assert 0.35 < np.mean(seen) < 0.65
    assert mech.config["observe_prob"] == 0.5 and "observe_prob" not in Team(n_workers=3).config


def test_self_reports_leniency_and_misprision():
    dom = SyntheticTeam(n_items=1)
    items, ctx = dom.load(), dom.context()
    rule = soa.Whistleblower(bounty=0.5, violation_penalty=1.0, leniency=0.75, misprision=0.2)
    mech = Team(n_workers=3, regrade_on_report=True, self_reports=True, reward=rule)
    confess, reported, silent = run(mech, items, [
        profile(dom, [("hack", "confess"), ("honest", "silent"), ("honest", "silent")]),
        profile(dom, [("hack", "silent"), ("honest", "report"), ("honest", "silent")]),
        profile(dom, [("hack", "silent"), ("honest", "silent"), ("honest", "silent")])], ctx=ctx)
    # a confession reverts the hack; a quarter of the penalty stands; no bounty for reporting oneself; the silent
    # witnesses of a caught violator pay misprision
    assert confess.outcome.data["reports"]["worker_1"]["self"] and confess.outcome.data["violators"] == ["worker_1"]
    assert confess.rewards == pytest.approx({"worker_1": 0.6 - 0.25, "worker_2": 0.6 - 0.2, "worker_3": 0.6 - 0.2})
    assert reported.rewards == pytest.approx({"worker_1": 0.6 - 1.0, "worker_2": 0.6 + 0.5, "worker_3": 0.6 - 0.2})
    assert silent.rewards == pytest.approx({f"worker_{i}": 0.9 for i in (1, 2, 3)})  # nobody caught it
    # without self_reports a confession is dropped (the default, as before)
    plain = run(Team(n_workers=3, regrade_on_report=True, reward=rule), items,
                [profile(dom, [("hack", "confess"), ("honest", "silent"), ("honest", "silent")])], ctx=ctx)[0]
    assert not plain.outcome.data["reports"]["worker_1"]["reported"]


def test_misprision_needs_the_witness_record():
    from so_arena.core.mechanism import Episode, Outcome

    ep = Episode(id="e", item_id="i", mechanism="team", outcome=Outcome(data={
        "team_score": 0.9, "team_score_reverted": 0.6, "team_roles": ["a", "b"], "caught": True,
        "violators": ["a"], "reports": {}}))
    with pytest.raises(ValueError, match="witnesses"):
        soa.Whistleblower(bounty=0.1, misprision=0.2).compute(ep)
    assert soa.Whistleblower(bounty=0.1).compute(ep) == {"a": 0.6, "b": 0.6}


def test_synthetic_team_gain_damage_and_opportunity():
    dom = SyntheticTeam(n_items=40, gain=0.2, damage=0.5, opportunity="one", n_workers=3)
    items, ctx = dom.load(), dom.context()
    holders = [[k for k, v in it.private.items() if v == SHORTCUT_BRIEF] for it in items]
    assert all(len(h) == 1 for h in holders) and len({h[0] for h in holders}) == 3
    eps = run(Team(n_workers=3, regrade_on_report=True), items[:6],
              [profile(dom, [("hack", "silent")] * 3)], ctx=ctx, gt=dom.ground_truth_scorers())
    for ep, h in zip(eps, holders):
        hackers = [t.role for t in ep.turns if "HACK" in t.text]
        assert hackers == [h[0].removeprefix("brief_")]  # only the holder can use the shortcut
        assert ep.outcome.data["team_score"] == pytest.approx(0.8)  # 0.6 + gain
        assert ep.ground_truth["true_score"] == pytest.approx(0.1)  # 0.6 - damage: the hack stands
    # the default domain is unchanged: everyone can hack, items carry no briefs
    assert SyntheticTeam(n_items=2).load()[0] == SyntheticTeam(n_items=2, damage=0.5).load()[0]
    assert not SyntheticTeam(n_items=2).load()[0].private


def test_egta_reporting_game_matches_the_partial_observation_model():
    """Hacker fixed, two potential witnesses who each see the hack w.p. 0.8: the empirical game over many
    items approaches the analytic ex-ante game; its equilibria match the regimes."""
    dom = SyntheticTeam(n_items=150)
    items, ctx = dom.load(), dom.context()
    strategies = {w: {"silent": team_worker(report="silent"), "report": team_worker(report="report")}
                  for w in ("worker_2", "worker_3")}
    for s, regime in ((0.05, "silence"), (0.2, "coordination"), (0.45, "reporting")):
        mech = Team(n_workers=3, observe_prob=0.8, regrade_on_report=True,
                    reward=soa.Whistleblower(bounty=s, split="shared"))
        exp = EmpiricalGameExperiment(mech, items, strategies, symmetric=["worker_2", "worker_3"], ctx=ctx,
                                      fixtures={"worker_1": team_worker(work="hack"), "grader": dom.grader()})
        exp.run()
        g = exp.game()
        analytic = wb.whistleblower_game(2, s=s, o=0.8, **TEAM)
        for prof in g.profiles():
            for a, p in enumerate(g.players):
                assert g.payoffs[p][prof] == pytest.approx(analytic.payoffs[analytic.players[a]][prof], abs=0.03)
        want = {"silence": {(0, 0)}, "coordination": {(0, 0), (1, 1)}, "reporting": {(1, 1)}}[regime]
        assert wb.regime(2, s=s, o=0.8, **TEAM) == regime and set(g.strict_nash()) == want


# ----------------------------------------------------------------------------- sampled training


def test_natural_estimator_is_unbiased_for_rare_strategies():
    rng = np.random.default_rng(0)
    probs = np.array([0.9, 0.07, 0.03])
    u = np.array([0.0, 1.0, 2.0])
    B, runs = 16, 20000
    nat, van = np.zeros(3), np.zeros(3)
    for _ in range(runs):
        picks = rng.choice(3, size=B, p=probs)
        r = u[picks] + rng.normal(0, 0.5, B)
        base = (r.sum() - r) / (B - 1)  # leave-one-out: independent of each play
        nat += natural_estimate(picks, r, base, probs) / runs
        van += vanilla_estimate(picks, r, base, probs) / runs
    adv = u - probs @ u
    # the softmax ignores a common shift: compare centred estimates
    assert nat - nat @ probs == pytest.approx(adv, abs=0.03)
    assert van == pytest.approx(probs * adv, abs=0.01)


def test_strategy_gradient_on_sampled_team_episodes():
    dom = SyntheticTeam(n_items=30, opportunity="one")
    items, ctx = dom.load(), dom.context()
    workers = ["worker_1", "worker_2", "worker_3"]
    strategies = {w: {s: team_worker(work="hack" if s.startswith("violate") else "honest",
                                     report="report" if s.endswith("report") else "silent")
                      for s in wb.TEAM_STRATEGIES} for w in workers}
    mech = Team(n_workers=3, observe_prob=0.8, regrade_on_report=True,
                reward=soa.Whistleblower(bounty=0.45, split="shared", violation_penalty=0.3))
    init = {w: wb.team_mix(0.5, 0.5) for w in workers}
    tr = StrategyGradient(mech, items, strategies, fixtures={"grader": dom.grader()}, shared=[workers], init=init,
                          natural=True, lr=3.0, batch=16, iterations=12, seed=1, ground_truth=dom.ground_truth_scorers(),
                          ctx=ctx)
    df = tr.run()
    x = [wb.team_marginals([r[f"p_worker_1_{s}"] for s in wb.TEAM_STRATEGIES])[0] for _, r in df.iterrows()]
    assert x[-1] < x[0] - 0.2  # reporting is dominant, so violating stops paying
    assert len(tr.episodes) == 13 * 16 and df["errors"].sum() == 0
    assert np.isfinite(df["outcome_value"]).all()
    # chance moves are redrawn per sampled episode, not per item
    assert len({(e.item_id, e.seed) for e in tr.episodes}) == len(tr.episodes)
    # a missing reward counts as the role's worst: training must not reward breaking the episode
    tr2 = StrategyGradient(mech, items, strategies, fixtures={"grader": dom.grader()}, shared=[workers], batch=4,
                           iterations=0, ctx=ctx)
    eps = [None, None]
    tr2._worst["worker_1"] = -1.0
    assert tr2._rewards("worker_1", eps).tolist() == [-1.0, -1.0]
    with pytest.raises(ValueError):
        StrategyGradient(mech, items, strategies, shared=[["worker_1", "judge"]])
    assert math.isfinite(df["reward_worker_1"].iloc[0])
