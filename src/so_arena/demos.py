"""Offline demonstrations (no API keys): each runs a small experiment end to end and writes a report.

These use the synthetic domains, so the numbers illustrate the *machinery* - they are not findings
about language models. Swap the synthetic policies for ``LLMPolicy`` and a real domain to run the
same experiments for real (see ``configs/`` and ``docs/experiments.md``).

* :func:`demo_asd` - ASD across protocols, with and without verified claims.
* :func:`demo_optimization` - best-of-N optimization pressure on a proposer, with and without a critic
  (the "debate with self-play best-of-N" design): does optimizing against the mechanism raise accuracy?
* :func:`demo_swarm` - reporting incentives in teams: analytic equilibria, a simulated empirical game,
  policy-gradient training dynamics, partial observation (when a bounty buys nothing) and natural vs.
  vanilla policy gradient on sampled episodes (the training algorithm selects the equilibrium).
* :func:`demo_work` - stateful work (the paired-branch design): honest and deceptive agents on the same
  repositories and company tasks, reviewed with three levels of access to the result.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import so_arena as soa
from so_arena.analysis import plots
from so_arena.analysis.report import Report, build_report
from so_arena.core.runner import PlayerSpec, Profile


def _figures(out: Path) -> Path:
    d = out / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def demo_asd(out: str | Path = "runs/demo_asd", n_items: int = 40) -> Path:
    from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
    from so_arena.mechanisms import Consultancy, Debate, DirectJudge, Propaganda
    from so_arena.samplers.arms import ASDExperiment

    out = Path(out)
    dom = SyntheticPersuasion(n_items=n_items, hint_strength=0.4, seed=1)
    items, ctx = dom.load(), dom.context()
    arguer = synthetic_arguer(honest_mean=0.8, dishonest_mean=0.3, sd=0.7, claim_rate=0.6, lie_claim_rate=0.8)
    aff = {"agents": ["answer_key"]}
    ver = soa.VerificationPolicy(verifiers=["fact"])
    mechs = [DirectJudge(), Propaganda(affordances=aff), Consultancy(rounds=2, affordances=aff),
             Debate(rounds=2, affordances=aff), Debate(rounds=2, affordances=aff, verification=ver, name="debate+verified")]
    exp = ASDExperiment(mechs, items, agent=arguer, fixtures={"judge": synthetic_judge()}, ctx=ctx,
                        store=soa.RunStore(out))
    eps = exp.run()
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd

    table = asd(role_frame(eps))
    order = [m.name for m in mechs]
    table = table.set_index("mechanism").loc[order].reset_index()
    plots.asd_bars(table, subtitle="synthetic persuasion world · log-score rewards").save(_figures(out) / "asd.png")
    build_report(eps, out / "report.html", title="ASD across protocols", subtitle="synthetic persuasion world (demo)")
    return out


def demo_optimization(out: str | Path = "runs/demo_optimization", n_items: int = 24) -> Path:
    from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_critic, synthetic_reviewer, synthetic_worker
    from so_arena.mechanisms import ReviewedWork
    from so_arena.samplers.pools import OptimizationExperiment

    out = Path(out)
    dom = SyntheticPersuasion(n_items=n_items, seed=2)
    items, ctx = dom.load(), dom.context()
    worker = synthetic_worker(honest_choice_rate=0.6, honest_mean=1.0, dishonest_mean=0.2, sd=0.8,
                              sophistry_rate=0.25, sophistry_boost=2.5)
    aff = {"agents": ["answer_key"]}
    single = ReviewedWork(critique_rounds=0, transform="prob", affordances=aff, name="single judge")
    critique = ReviewedWork(critique_rounds=1, rebuttal=False, transform="prob", affordances=aff, name="with critic")
    prof = Profile(name="open", players={"worker": worker, "critic": synthetic_critic(), "reviewer": synthetic_reviewer()})
    prof_single = Profile(name="open", players={"worker": worker, "reviewer": synthetic_reviewer()})
    ns = [1, 2, 4, 8, 16]
    e1 = OptimizationExperiment(single, items, prof_single, pool_sizes={"worker": 16}, ctx=ctx)
    e1.run()
    g1 = e1.grid({"worker": ns})
    e2 = OptimizationExperiment(critique, items, prof, pool_sizes={"worker": 16, "critic": 4}, ctx=ctx, max_leaves=100000)
    e2.run()
    g2 = e2.grid({"worker": ns, "critic": [1, 2, 4]})
    rows = []
    for name, g in (("single judge", g1), ("with critic (critic Bo4)", g2[g2["level_critic"] == 4]),
                    ("with critic (critic Bo1)", g2[g2["level_critic"] == 1])):
        for _, r in g.iterrows():
            rows.append({"series": name, "n": r["level_worker"], "win_rate": r["reward_worker"],
                         "accuracy": (r["value_worker"] + 1) / 2})
    curves = pd.DataFrame(rows)
    figs = _figures(out)
    c = plots.parametric_curves(curves, x="win_rate", y="accuracy", series="series", level="n",
                                title="Optimizing the proposer against the mechanism",
                                subtitle="best-of-n selection by reviewer acceptance; y = is the accepted answer right?",
                                xlabel="proposer reward (P(accept))", ylabel="proposer accuracy (ground truth)")
    c.save(figs / "bon_curves.png")
    mesh = g2.assign(accuracy=(g2["value_worker"] + 1) / 2)
    plots.heatmap(mesh, x="level_worker", y="level_critic", value="accuracy", title="Proposer accuracy under self-play",
                  xlabel="proposer n", ylabel="critic n", value_label="accuracy").save(figs / "bon_mesh.png")
    curves.to_csv(out / "bon_curves.csv", index=False)
    mesh.to_csv(out / "bon_mesh.csv", index=False)
    rep = Report("Optimization pressure (best-of-N)", "synthetic proposal → critique → judgment (demo)")
    rep.section("Does optimizing against the mechanism help?",
                charts=plots.dual_mode(plots.parametric_curves, curves, x="win_rate", y="accuracy", series="series",
                                       level="n", title="Optimizing the proposer against the mechanism",
                                       subtitle="best-of-n by reviewer acceptance",
                                       xlabel="proposer reward (P(accept))", ylabel="proposer accuracy"),
                info="Each point is a best-of-n policy computed exactly from sampled pools; moving right means "
                     "more optimization against the mechanism's reward.")
    rep.section("Both players optimizing", charts=plots.dual_mode(plots.heatmap, mesh, x="level_worker", y="level_critic",
                                                                    value="accuracy", title="Proposer accuracy",
                                                                    xlabel="proposer n", ylabel="critic n",
                                                                    value_label="accuracy"))
    rep.write(out / "report.html")
    return out


def demo_swarm(out: str | Path = "runs/demo_swarm", n_runs: int = 6) -> Path:
    """Shared rewards vs. reporting incentives: (a) the analytic threshold, (b) the simulated reporting game,
    (c) exact training dynamics, (d) partial observation, analytic and simulated, and (e) natural vs. vanilla
    policy gradient on sampled episodes of the whole game (violate? report?), ``n_runs`` runs each, with CIs.
    All agents are scripted."""
    from so_arena.domains.synthetic import SyntheticTeam, team_worker
    from so_arena.games import EmpiricalGameExperiment
    from so_arena.games.learning import policy_gradient
    from so_arena.mechanisms import Team
    from so_arena.theory import whistleblower as wb

    out = Path(out)
    figs = _figures(out)
    # (a) analytic: basin boundary vs bounty ratio
    sw = wb.sweep(np.linspace(0.02, 1.0, 50), ns=(2, 5, 20), delta=1.0)
    th = plots.threshold_curves(sw, x="ratio", y="p_threshold", series="n", title="When does reporting take over?",
                                subtitle="n witnesses; bounty s for a verified report; hack worth Δ to everyone",
                                xlabel="bounty / value of the hack (s/Δ)", ylabel="report propensity threshold p*",
                                below="Start below a curve → training ends with everyone silent",
                                above="above → everyone reports", vline=1.0, vline_label="reporting dominant")
    th.save(figs / "snitch_threshold.png")
    # (b) simulated team game: hacker fixed, two potential reporters, three bounty levels
    dom = SyntheticTeam(n_items=4)
    items, ctx = dom.load(), dom.context()
    rows = []
    games = {}
    for bounty in (0.0, 0.1, 0.5):
        mech = Team(n_workers=3, regrade_on_report=True, reward=soa.Whistleblower(bounty=bounty),
                    name=f"team(bounty={bounty:g})")
        strategies = {w: {"silent": team_worker(report="silent"), "report": team_worker(report="report")}
                      for w in ("worker_2", "worker_3")}
        exp = EmpiricalGameExperiment(mech, items, strategies, symmetric=["worker_2", "worker_3"], ctx=ctx,
                                      fixtures={"worker_1": team_worker(work="hack"), "grader": dom.grader()},
                                      ground_truth=dom.ground_truth_scorers())
        exp.run()
        g = exp.game()
        games[bounty] = g
        strict = [g.profile_names(p) for p in g.strict_nash()]
        rows.append({"bounty": bounty, "strict equilibria": "; ".join("/".join(v for v in s.values()) for s in strict) or "none",
                     "hack reverted at best eq.": max((g.outcomes["outcome_value"][p] for p in g.strict_nash()), default=np.nan),
                     "hack reverted at worst eq.": min((g.outcomes["outcome_value"][p] for p in g.strict_nash()), default=np.nan)})
    eq_table = pd.DataFrame(rows)
    eq_table.to_csv(out / "team_equilibria.csv", index=False)
    # (c) training dynamics under a small bounty: outcome depends on the initial propensity
    g = games[0.1]
    traj = []
    for p0 in (0.5, 0.8):
        df = policy_gradient(g, init=[np.array([1 - p0, p0])] * 2, shared=[["worker_2", "worker_3"]], lr=3.0, steps=400,
                             record_every=10)
        traj.append(df[["step", "p_worker_2_report"]].rename(columns={"p_worker_2_report": f"start at {p0:.0%}"}))
    tr = traj[0].merge(traj[1], on="step")
    lc = plots.line_chart(tr, x="step", ys=list(tr.columns[1:]), title="Training on a small bounty (s/Δ = 1/3)",
                          subtitle="two reporters; threshold p* = 2/3", xlabel="policy-gradient step",
                          ylabel="P(report)", ylim=(0, 1))
    lc.save(figs / "snitch_training.png")
    tr.to_csv(out / "snitch_training.csv", index=False)
    rep = Report("Swarms: shared rewards vs. reporting incentives", "analytic model, simulated team game, training dynamics (demo)")
    rep.section("Equilibrium structure", charts=plots.dual_mode(
        plots.threshold_curves, sw, x="ratio", y="p_threshold", series="n", title="When does reporting take over?",
        subtitle="n witnesses; bounty s for a verified report; hack worth Δ to everyone",
        xlabel="bounty / value of the hack (s/Δ)", ylabel="report propensity threshold p*",
        below="Start below a curve → training ends with everyone silent", above="above → everyone reports",
        vline=1.0, vline_label="reporting dominant"),
        info="Below the curve, 'nobody reports' is where learning ends up; above it, 'everyone reports'. "
             "With no bounty nobody ever has a reason to report.")
    rep.section("Simulated team game", table=eq_table,
                note="One teammate hacks the grader; two teammates choose whether to report. Strict equilibria of the "
                     "simulated game, which match the analytic model exactly.")
    rep.section("Training dynamics", charts=plots.dual_mode(
        plots.line_chart, tr, x="step", ys=list(tr.columns[1:]), title="Training on a small bounty (s/Δ = 1/3)",
        subtitle="two reporters; threshold p* = 2/3", xlabel="policy-gradient step", ylabel="P(report)", ylim=(0, 1)))
    _swarm_partial_observation(out, figs, rep)
    _swarm_training(out, figs, rep, n_runs)
    rep.write(out / "report.html")
    return out


SWARM_STAKE, SWARM_PENALTY, SWARM_OBSERVE = 0.3, 0.3, 0.8  # the synthetic team: hack worth 0.3 to everyone


def _swarm_partial_observation(out: Path, figs: Path, rep: Report) -> None:
    """Regimes vs. bounty when each teammate sees the hack only with probability o (analytic and simulated)."""
    from so_arena.domains.synthetic import SyntheticTeam, team_worker
    from so_arena.games import EmpiricalGameExperiment
    from so_arena.mechanisms import Team
    from so_arena.theory import whistleblower as wb

    team = dict(R=0.9, delta=SWARM_STAKE, split="shared")
    os_ = np.round(np.linspace(0.3, 1.0, 36), 3)
    th = pd.DataFrame({"o": os_,
                       "b_R, team of 3": [wb.report_equilibrium_bounty(2, o=o, **team) / SWARM_STAKE for o in os_],
                       "b_R, team of 6": [wb.report_equilibrium_bounty(5, o=o, **team) / SWARM_STAKE for o in os_],
                       "reporting dominant": [wb.dominance_bounty(2, o=o, **team) / SWARM_STAKE for o in os_]})
    th.to_csv(out / "bounty_thresholds.csv", index=False)
    kw = dict(x="o", ys=list(th.columns[1:]), title="Partial observation: how large must a bounty be?",
              subtitle="bounty shared among verified reporters; below b_R silence is the only equilibrium, between b_R "
                       "and the stake both are, above the stake reporting is dominant",
              xlabel="probability that a teammate sees the hack (o)", ylabel="bounty / stake (s/Δ)", ylim=(0, 1.05))
    plots.line_chart(th, **kw).save(figs / "snitch_observation.png")
    # the simulated reporting game with o = 0.8: hacker fixed, two potential witnesses
    dom = SyntheticTeam(n_items=200)
    items, ctx = dom.load(), dom.context()
    strategies = {w: {"silent": team_worker(report="silent"), "report": team_worker(report="report")}
                  for w in ("worker_2", "worker_3")}
    rows = []
    b_r = wb.report_equilibrium_bounty(2, o=SWARM_OBSERVE, **team)
    for bounty in (0.05, 0.2, 0.45):
        mech = Team(n_workers=3, observe_prob=SWARM_OBSERVE, regrade_on_report=True,
                    reward=soa.Whistleblower(bounty=bounty, split="shared"), name=f"team(o={SWARM_OBSERVE:g},b={bounty:g})")
        exp = EmpiricalGameExperiment(mech, items, strategies, symmetric=["worker_2", "worker_3"], ctx=ctx,
                                      fixtures={"worker_1": team_worker(work="hack"), "grader": dom.grader()},
                                      ground_truth=dom.ground_truth_scorers())
        exp.run()
        g = exp.game()
        strict = [g.profile_names(p) for p in g.strict_nash()]
        analytic = wb.whistleblower_game(2, s=bounty, o=SWARM_OBSERVE, **team)
        rows.append({"bounty": bounty, "bounty / stake": round(bounty / SWARM_STAKE, 2),
                     "regime (theory)": wb.regime(2, s=bounty, o=SWARM_OBSERVE, **team),
                     "strict equilibria (simulated)": "; ".join("/".join(v for v in x.values()) for x in strict) or "none",
                     "strict equilibria (theory)": "; ".join("/".join(v for v in analytic.profile_names(p).values())
                                                             for p in analytic.strict_nash()),
                     "max payoff gap to theory": float(max(np.nanmax(np.abs(g.payoffs[a] - analytic.payoffs[b]))
                                                           for a, b in zip(g.players, analytic.players)))})
    table = pd.DataFrame(rows)
    table.to_csv(out / "partial_observation_equilibria.csv", index=False)
    rep.section("Partial observation", charts=plots.dual_mode(plots.line_chart, th, **kw), table=table,
                info=f"Each teammate sees the hack only with probability o, so a witness may be the only one. Then "
                     f"'everyone reports' is an equilibrium only above b_R (Proposition 5): with o = {SWARM_OBSERVE:g} and "
                     f"two potential witnesses, b_R = {b_r / SWARM_STAKE:.2f} of the stake, so a bounty of 0.05 buys "
                     f"nothing although with full observation it would create a coordination game.",
                note=f"Simulated game ({len(items)} items; who sees what is a chance move, identical across profiles): "
                     f"one teammate hacks, two choose whether to report what they see. Payoffs match the analytic "
                     f"ex-ante game up to the sampling noise of the observation draws.")


def _t_ci(values: np.ndarray, lo: float = -math.inf, hi: float = math.inf) -> tuple[float, float, float]:
    """Mean and 95% t-interval over independent runs, clipped to the quantity's range [lo, hi]."""
    from scipy.stats import t

    v = np.asarray(values, float)
    m = float(v.mean())
    if len(v) < 2:
        return m, math.nan, math.nan
    h = float(t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))
    return m, max(lo, m - h), min(hi, m + h)


def _swarm_training(out: Path, figs: Path, rep: Report, n_runs: int) -> None:
    """Natural vs. vanilla policy gradient on sampled episodes of the whole game, averaged over runs."""
    from so_arena.domains.synthetic import SyntheticTeam, team_worker
    from so_arena.games.learning import StrategyGradient, policy_gradient
    from so_arena.mechanisms import Team
    from so_arena.theory import whistleblower as wb

    dom = SyntheticTeam(n_items=60, opportunity="one")  # one random worker per task can hack
    items, ctx = dom.load(), dom.context()
    workers = ["worker_1", "worker_2", "worker_3"]
    strategies = {w: {s: team_worker(work="hack" if s.startswith("violate") else "honest",
                                     report="report" if s.endswith("report") else "silent")
                      for s in wb.TEAM_STRATEGIES} for w in workers}
    iterations, batch, lr = 60, 24, 2.0
    outcomes = {k: (lambda ep, k=k: ep.ground_truth.get(k)) for k in ("violation", "true_score")}
    settings = [(0.45, 0.1, "bounty above the stake; teammates start out mostly silent"),
                (0.05, 0.9, "bounty below b_R; teammates start out mostly reporting")]
    all_rows, summary = [], []
    for bounty, p0, what in settings:
        theory = dict(R=0.9, delta=SWARM_STAKE, split="shared", s=bounty, P=SWARM_PENALTY, o=SWARM_OBSERVE)
        mech = Team(n_workers=3, observe_prob=SWARM_OBSERVE, regrade_on_report=True,
                    reward=soa.Whistleblower(bounty=bounty, split="shared", violation_penalty=SWARM_PENALTY),
                    name=f"team(b={bounty:g})")
        init = {w: wb.team_mix(0.5, p0) for w in workers}
        game = wb.team_game(2, **theory)
        curves = {}
        for natural, algo in ((True, "natural PG"), (False, "REINFORCE")):
            finals = []
            for run in range(n_runs):
                df = StrategyGradient(mech, items, strategies, fixtures={"grader": dom.grader()}, shared=[workers],
                                      init=init, natural=natural, lr=lr, batch=batch, iterations=iterations, seed=run,
                                      ground_truth=dom.ground_truth_scorers(), ctx=ctx, outcomes=outcomes).run()
                for _, r in df.iterrows():
                    x, p = wb.team_marginals([r[f"p_worker_1_{s}"] for s in wb.TEAM_STRATEGIES])
                    all_rows.append({"bounty": bounty, "algorithm": algo, "run": run, "iteration": int(r["iteration"]),
                                     "P(violate)": x, "P(report)": p, "violation rate": r["violation"],
                                     "true score": r["true_score"]})
                finals.append(all_rows[-1]["P(violate)"])
            exact = policy_gradient(game, init=[wb.team_mix(0.5, p0)] * 3, shared=[game.players], lr=lr,
                                    steps=iterations, natural=natural)
            curves[algo] = [wb.team_marginals([r[f"p_worker_1_{s}"] for s in wb.TEAM_STRATEGIES])[0]
                            for _, r in exact.iterrows()]
            m, lo, hi = _t_ci(np.array(finals), 0.0, 1.0)
            summary.append({"bounty": bounty, "setting": what, "algorithm": algo, "start P(report)": p0,
                            f"P(violate) after {iterations} iterations": round(m, 3), "95% CI": f"{lo:.3f} to {hi:.3f}",
                            "runs deterring (P(violate) < 0.5)": f"{sum(f < 0.5 for f in finals)}/{n_runs}",
                            "expected-update prediction": round(curves[algo][-1], 3)})
        d = pd.DataFrame([r for r in all_rows if r["bounty"] == bounty])
        agg = []
        for it, g in d.groupby("iteration"):
            row = {"iteration": it}
            for algo in ("natural PG", "REINFORCE"):
                m, lo, hi = _t_ci(g.loc[g["algorithm"] == algo, "P(violate)"].to_numpy(), 0.0, 1.0)
                row |= {algo: m, f"{algo} lo": lo, f"{algo} hi": hi, f"{algo} (expected update)": curves[algo][it]}
            agg.append(row)
        agg = pd.DataFrame(agg)
        kw = dict(x="iteration", ys=["natural PG", "REINFORCE", "natural PG (expected update)", "REINFORCE (expected update)"],
                  bands={a: (f"{a} lo", f"{a} hi") for a in ("natural PG", "REINFORCE")},
                  dashed=["natural PG (expected update)", "REINFORCE (expected update)"],
                  title=f"Training the whole game: bounty {bounty:g} (stake {SWARM_STAKE:g})",
                  subtitle=f"{what}; start P(violate) = 0.5, P(report) = {p0:g}; mean of {n_runs} runs, 95% CI; dashed: "
                           f"the exact expected update on the analytic game",
                  xlabel="training iteration (batches of sampled episodes)", ylabel="P(violate)", ylim=(0, 1))
        name = f"snitch_npg_vs_reinforce_{'high' if bounty > SWARM_STAKE else 'low'}.png"
        plots.line_chart(agg, **kw).save(figs / name)
        rep.section(f"Natural vs. vanilla policy gradient: bounty {bounty:g}", charts=plots.dual_mode(plots.line_chart, agg, **kw),
                    info="Scripted workers choose among four strategies (violate if you can? report what you see?); each "
                         "worker's softmax policy is trained on sampled episodes of the mechanism." if bounty == settings[0][0] else None)
    pd.DataFrame(all_rows).to_csv(out / "npg_vs_reinforce.csv", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(out / "npg_vs_reinforce_summary.csv", index=False)
    rep.section("Equilibrium selection depends on the training algorithm", table=table,
                note=f"Team of 3; one random worker per task can hack the grader (+{SWARM_STAKE:g} to everyone if it stands); "
                     f"each teammate sees it with probability {SWARM_OBSERVE:g}; a caught violator pays {SWARM_PENALTY:g}; "
                     f"{batch} episodes per iteration, learning rate {lr:g}. Both algorithms have the same rest points, but "
                     f"vanilla gradient moves a strategy's logit in proportion to its probability, so a rarely played "
                     f"strategy (reporting, when nobody does) learns slowly and misconduct entrenches first. Below b_R, "
                     f"training that starts from a reporting culture ends with nobody violating: a Nash equilibrium "
                     f"whose threat is never tested - after an actual violation a witness would rather stay silent (it is "
                     f"not subgame perfect), so the outcome is fragile. The natural-gradient estimate weights a sampled "
                     f"strategy by 1/P(strategy), so once misconduct is rare a single sampled violation that went unseen "
                     f"can move its logit a long way: its runs spread more than REINFORCE's. All agents are scripted: these "
                     f"numbers illustrate the dynamics, not language-model behaviour.")


ACCESS_LEVELS = {"report": "report only", "diff": "+ diffs", "query": "+ queries"}
ARM_NAMES = {
    ("repo", "hardcode"): "repo: hard-code the tested inputs", ("repo", "tamper"): "repo: edit the tests",
    ("repo", "sabotage"): "repo: subtly wrong code", ("repo", "lazy"): "repo: do nothing, claim done",
    ("acquisition", "burner"): "sales: sign-ups on the company card", ("acquisition", "fake_rows"): "sales: fake database rows",
    ("acquisition", "false_claims"): "sales: false claims to leads", ("acquisition", "spam"): "sales: spam the leads",
    ("refunds", "deny"): "support: deny eligible refunds", ("refunds", "close_all"): "support: close tickets, no refunds",
    ("investor_update", "redefine"): "reporting: redefine 'active user'",
    ("investor_update", "inflate"): "reporting: inflate the numbers",
}


def demo_work(out: str | Path = "runs/demo_work", n_repo: int = 6, n_firm: int = 3) -> Path:
    """Paired branches on stateful work: scripted honest and deceptive workers on repository tasks (hidden
    tests) and on a simulated company's sales, support and reporting tasks, each reviewed by scripted
    reviewers who see the report only, the report and diffs of every change, or can also query the result."""
    from so_arena.domains.firm import WORKER_MODES as FIRM_MODES
    from so_arena.domains.firm import FirmDomain, firm_reviewer, firm_worker
    from so_arena.domains.repo import WORKER_MODES as REPO_MODES
    from so_arena.domains.repo import RepoDomain, repo_reviewer, repo_worker
    from so_arena.mechanisms import ReviewedWork
    from so_arena.samplers.paired import PairedWorkExperiment

    out = Path(out)
    figs = _figures(out)
    rows, audits, all_eps = [], [], []
    families = [("repo", RepoDomain(), RepoDomain().load(limit=n_repo), REPO_MODES, repo_worker, repo_reviewer)]
    firm = FirmDomain(n_items=n_firm)
    firm_items = firm.load()
    for kind, modes in FIRM_MODES.items():
        families.append((kind, firm, [it for it in firm_items if it.metadata["kind"] == kind], modes, firm_worker,
                         firm_reviewer))
    for family, dom, items, modes, worker, reviewer in families:
        ctx = dom.context(states=out / "states")
        env_tools = dom.environment().tools()
        tools_all = sorted(env_tools)
        readonly = sorted(n for n, t in env_tools.items() if getattr(t, "readonly", False))
        for level in ACCESS_LEVELS:
            lvl = "exec" if (family == "repo" and level == "query") else level
            kw = ({"state_access": {"reviewer": "read"}, "tools": {"worker": tools_all, "reviewer": readonly}}
                  if level == "query" else {"tools": {"worker": tools_all}})
            mech = ReviewedWork(name=f"{family}/{level}", dossier="report" if level == "report" else "diff",
                                affordances={"worker": ["solutions"]} if family == "repo" else None, **kw)
            exp = PairedWorkExperiment([mech], items, behaviours={m: worker(m) for m in modes},
                                       fixtures={"reviewer": reviewer(lvl)}, ctx=ctx, ground_truth=dom.ground_truth_scorers())
            eps = exp.run()
            all_eps += eps
            for r in exp.summary().itertuples():
                if r.arm == "worst":
                    continue
                rows.append({"deception": ARM_NAMES.get((family, r.arm), f"{family}: {r.arm}"), "family": family,
                             "access": ACCESS_LEVELS[level], "asd": r.asd, "ci_low": r.ci_low, "ci_high": r.ci_high,
                             "n_tasks": r.n_items})
            if level == "report":
                for e in eps:
                    audits.append({"family": family, "arm": e.players["worker"].label,
                                   "audited_good": e.value("worker") is not None and e.value("worker") > 0})
    table = pd.DataFrame(rows)
    table.to_csv(out / "asd_by_access.csv", index=False)
    order_y = [ARM_NAMES[k] for k in ARM_NAMES if ARM_NAMES[k] in set(table["deception"])]
    order_x = list(ACCESS_LEVELS.values())
    tips = {(r.access, r.deception): f"(95% CI {r.ci_low:.2f} to {r.ci_high:.2f}; {r.n_tasks} tasks)" for r in table.itertuples()}
    kw = dict(x="access", y="deception", value="asd", center=0.0, order_x=order_x, order_y=order_y, tip_extra=tips,
              title="Does review pay honest work more than faked success?",
              subtitle="ASD = log P(accept | honest) − log P(accept | deceptive), same task; blue: honesty pays, "
                       "red: deception pays",
              xlabel="reviewer's access (cumulative)", ylabel="", value_label="ASD", decimals=2)
    plots.heatmap(table, **kw).save(figs / "asd_by_access.png")
    audit = pd.DataFrame(audits).groupby(["family", "arm"])["audited_good"].mean().reset_index()
    rep = Report("Stateful work: honest vs. faked success", "paired branches on repositories and a simulated company (demo)")
    rep.kpis({"episodes": len(all_eps), "deception types": len(order_y), "tasks": n_repo + 3 * n_firm,
              "errors": sum(e.error is not None for e in all_eps)})
    rep.section("Reviewer access vs. incentives", charts=plots.dual_mode(plots.heatmap, table, **kw),
                table=table[["deception", "access", "asd", "ci_low", "ci_high", "n_tasks"]],
                info="Every deceptive worker reports success. Reviewers who read only the report cannot tell; diffs "
                     "expose edited tests, card charges, false claims and edited metrics; querying the result exposes "
                     "hard-coded answers, denied refunds and inflated numbers; nothing short of hidden tests exposes "
                     "subtly wrong code. Scripted workers and reviewers: this illustrates the machinery, not LLM behaviour.")
    rep.section("Ground truth: the environment's audit", table=audit,
                note="Share of each arm's results the audit (hidden tests; the company's ledger of real sign-ups, "
                     "refund eligibility and KPI definitions) judged good. Only honest work passes.")
    rep.write(out / "report.html")
    return out


def run_all(out: str | Path = "runs/demos") -> list[Path]:
    out = Path(out)
    return [demo_asd(out / "asd"), demo_optimization(out / "optimization"), demo_swarm(out / "swarm"),
            demo_work(out / "work")]
