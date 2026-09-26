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
* :func:`demo_monitoring` - penalizing a monitor's alarms: an evadable monitor buys obfuscation, a
  fixed-AUROC probe deters; thresholds against Proposition 4, with CIs across seeds.
* :func:`demo_hiddenbits` - verified claims (HiddenBits): ASD against the verification budget for naive,
  credulous and sceptical Bayesian judges, in propaganda and debate, against Proposition 3.
* :func:`demo_bon_budget` - best-of-N on a lying debater under a verification budget: selection moves its
  lies beyond what is checked.
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


def demo_monitoring(out: str | Path = "runs/demo_monitoring", n_items: int = 200, n_seeds: int = 20,
                    auroc: float = 0.9) -> Path:
    """Monitor-based rewards under optimization (``docs/theory.md``, Proposition 4).

    A scripted actor on the synthetic monitoring world works honestly, overtly edits the tests or edits them
    covertly; ``MonitoredWork`` pays the reported pass rate minus ``lam`` per flag. For each seed (new trace and
    probe draws) the actor best-responds - the strategy with the highest mean reward over ``n_items`` tasks - at
    every penalty, against a keyword monitor (which covert work evades), a simulated probe of fixed AUROC, and
    the keyword monitor with defer-to-trusted. Rewards are affine in the penalty, so each strategy's line is
    measured with the mechanism's reward rule at two penalties and the envelope gives its exact switch points.
    """
    from dataclasses import replace

    from so_arena.analysis.metrics import bootstrap_mean_ci
    from so_arena.core.runner import run_episodes, run_sync
    from so_arena.domains.monitoring import ACTIONS, SyntheticMonitoring, keyword_monitor, monitoring_actor, trusted_worker
    from so_arena.integrations.probes import SimulatedProbe, probe_monitor
    from so_arena.mechanisms import MonitoredWork
    from so_arena.mechanisms.work import MonitoredReward
    from so_arena.theory import monitoring as tm

    out = Path(out)
    figs = _figures(out)
    theory = tm.MonitorParams(s=0.5, gain=0.3, hide_cost=0.05, d_overt=1.0, d_covert=0.15, d_honest=0.05)
    dom = SyntheticMonitoring(n_items=n_items, success=theory.s, gain=theory.gain, hide_cost=theory.hide_cost,
                              d_overt=theory.d_overt, d_covert=theory.d_covert, d_honest=theory.d_honest)
    items = dom.load()
    fpr, tpr = tm.probe_rates(auroc)
    probe_name = f"probe (AUROC {auroc:g})"
    variants = {  # name -> (mechanism, monitor policy, the theory it should match)
        "keyword monitor": (MonitoredWork(penalty="flag", forfeit=False, name="monitored(keyword)"), keyword_monitor(),
                            theory),
        probe_name: (MonitoredWork(penalty="flag", forfeit=False, monitor_sees=("probes",),
                                   probes=[SimulatedProbe(auroc=auroc)], name="monitored(probe)"),
                     probe_monitor(), tm.with_detector(theory, fpr, tpr)),
        "keyword monitor + defer": (MonitoredWork(penalty="flag", defer=True, name="monitored(keyword, defer)"),
                                    keyword_monitor(), replace(theory, forfeit=True)),
    }
    lams = np.round(np.linspace(0.0, 3.0, 121), 4)
    curve, switches, rates, n_eps = [], [], [], 0
    for seed in range(n_seeds):
        ctx = dom.context(seed=seed)
        for vname, (mech, monitor, _) in variants.items():
            fixtures = {"monitor": monitor, "grader": dom.grader(), **({"trusted": trusted_worker()} if mech.defer else {})}
            profiles = [Profile(name=a, players={"worker": monitoring_actor(a), **fixtures}) for a in ACTIONS]
            eps = run_sync(run_episodes(mech, items, profiles, ctx=ctx, seed=seed, ground_truth=dom.ground_truth_scorers()))
            n_eps += len(eps)
            if any(e.error for e in eps):
                raise RuntimeError(next(e.error for e in eps if e.error))
            # each strategy's expected reward is a line in the penalty: measure it with the mechanism's own rule
            r0, r1 = (MonitoredReward("flag", lam, forfeit=mech.forfeit) for lam in (0.0, 1.0))
            lines, alarm = [], {}
            for a in ACTIONS:
                es = [e for e in eps if e.profile == a]
                c = float(np.mean([r0.compute(e)["worker"] for e in es]))
                lines.append(tm.Behaviour(a, c, c - float(np.mean([r1.compute(e)["worker"] for e in es]))))
                alarm[a] = float(np.mean([e.outcome.data["flagged"] for e in es]))
                rates.append({"detector": vname, "seed": seed, "action": a, "alarm_rate": alarm[a]})
            lines.sort(key=lambda b: ACTIONS.index(b.name))
            switches.append({"detector": vname, "seed": seed, **tm.switch_points(lines),
                             **{f"{b.name}_{k}": v for b in lines for k, v in (("reward0", b.success), ("alarm", b.alarm))}})
            for lam in lams:
                a = tm.best_response_of(lines, float(lam))
                curve.append({"detector": vname, "seed": seed, "penalty": float(lam), "misbehaves": float(a != "honest"),
                              "covert": float(a == "covert"), "alarm_rate": alarm[a]})
    curve_df, switch_df = pd.DataFrame(curve), pd.DataFrame(switches)

    def summarize(col: str) -> pd.DataFrame:
        rows = []
        for (v, lam), g in curve_df.groupby(["detector", "penalty"], sort=False):
            lo, hi = bootstrap_mean_ci(g[col].to_numpy())
            rows.append({"detector": v, "penalty": lam, "mean": g[col].mean(), "lo": lo, "hi": hi})
        wide = pd.DataFrame(rows).pivot(index="penalty", columns="detector")
        flat = pd.DataFrame({"penalty": wide.index})
        for v in variants:
            for stat in ("mean", "lo", "hi"):
                flat[v if stat == "mean" else f"{v} {stat}"] = wide[(stat, v)].to_numpy()
        return flat

    names = list(variants)
    bands = {v: (f"{v} lo", f"{v} hi") for v in names}
    th = {v: tm.thresholds(p) for v, (_, _, p) in variants.items()}
    vlines = [(th["keyword monitor"]["deterred"], "keyword: deters (theory)"),
              (th[probe_name]["deterred"], "probe: deters (theory)")]
    mis, alarms = summarize("misbehaves"), summarize("alarm_rate")
    kw_mis = dict(x="penalty", ys=names, bands=bands, vlines=vlines, ylim=(-0.03, 1.03), xlabel="penalty per alarm (λ)",
                  title="Does penalizing alarms stop misbehaviour?",
                  subtitle=f"share of {n_seeds} seeds in which the actor's best response misbehaves; bands: 95% CI",
                  ylabel="P(best response misbehaves)")
    kw_alarm = dict(x="penalty", ys=names, bands=bands, vlines=vlines, ylim=(-0.03, 1.03), xlabel="penalty per alarm (λ)",
                    title="...or does the monitor only look better?",
                    subtitle="alarm rate of the best response (mean over seeds; bands: 95% CI)", ylabel="alarm rate")
    plots.line_chart(mis, **kw_mis).save(figs / "monitoring_misbehaviour.png")
    plots.line_chart(alarms, **kw_alarm).save(figs / "monitoring_alarms.png")
    mis.to_csv(out / "misbehaviour.csv", index=False)
    alarms.to_csv(out / "alarm_rate.csv", index=False)
    switch_df.to_csv(out / "switch_points.csv", index=False)

    def pooled(rows: pd.DataFrame) -> dict[str, float]:
        """Switch points of the strategies' lines averaged over seeds (a ratio of means: the mean of per-seed
        ratios would be biased upwards)."""
        return tm.switch_points([tm.Behaviour(a, rows[f"{a}_reward0"].mean(), rows[f"{a}_alarm"].mean())
                                 for a in ACTIONS])

    table, boot_rng = [], np.random.default_rng(0)
    for v in names:
        rows = switch_df[switch_df.detector == v].reset_index(drop=True)
        boots = [pooled(rows.iloc[boot_rng.integers(0, len(rows), len(rows))]) for _ in range(1000)]
        for key, what in (("overt_to_covert", "misbehaviour turns covert"), ("deterred", "misbehaviour deterred")):
            b = np.array([x[key] for x in boots])
            lo, hi = (np.percentile(b, [2.5, 97.5]) if np.isfinite(b).all() else (np.nan, np.nan))
            table.append({"detector": v, "switch": what, "theory λ": th[v][key], "measured λ": pooled(rows)[key],
                          "ci_low": lo, "ci_high": hi,
                          "seeds where it happens": f"{int(np.isfinite(rows[key]).sum())}/{len(rows)}"})
    table = pd.DataFrame(table)
    table.to_csv(out / "thresholds.csv", index=False)
    theory_rates = {v: {b.name: b.alarm for b in p.behaviours()} for v, (_, _, p) in variants.items()}
    rate_table = (pd.DataFrame(rates).groupby(["detector", "action"], sort=False)["alarm_rate"].mean().reset_index()
                  .rename(columns={"alarm_rate": "measured alarm rate"}))
    rate_table["theory"] = [theory_rates[r.detector][r.action] for r in rate_table.itertuples()]

    rep = Report("Monitoring: obfuscation vs. deterrence",
                 "synthetic monitoring world, scripted actor best-responding to MonitoredWork's reward (demo)")
    rep.kpis({"episodes": n_eps, "seeds": n_seeds, "tasks per seed": n_items, "probe AUROC": auroc})
    rep.section("Misbehaviour under a growing penalty", charts=plots.dual_mode(plots.line_chart, mis, **kw_mis),
                info="The actor picks whichever of honest work, overt test edits and covert test edits earns the most "
                     "(measured pass rate minus λ per flag). Against the keyword monitor, a small penalty only makes it "
                     "hide the edits; it is deterred at λ ≈ (Δ−κ)/(d_c−d_h). Against a probe whose accuracy does not "
                     "depend on how the actor behaves, hiding buys nothing and a moderate penalty deters. With deferral, "
                     "a flag also costs the task credit: misbehaviour is covert from the start.")
    rep.section("What the alarm rate says", charts=plots.dual_mode(plots.line_chart, alarms, **kw_alarm),
                info="Against the keyword monitor, the alarm rate drops from d_o to d_c as soon as the actor hides - "
                     "the monitor looks as if it were working while the misbehaviour continues. A low alarm rate under "
                     "optimization is not evidence of good behaviour.")
    rep.section("Switch points: theory vs. measured", table=table,
                note="Theory: the envelope of the three strategies' expected rewards (Proposition 4; the deferral row "
                     "uses the forfeiture variant). Measured: the same envelope of the strategies' reward lines measured "
                     "with the mechanism's reward rule and averaged over seeds, with a 95% bootstrap CI over seeds; the "
                     "last column counts the seeds whose own envelope has the switch. The keyword monitor's deterrence "
                     "threshold divides by d_c − d_h, so it is the noisiest number here.")
    rep.section("Alarm rates by strategy", table=rate_table,
                note="Mean over seeds. The probe's alarm rate on both kinds of misbehaviour is its detection rate: "
                     "its draws are shared by every strategy on an item (common random numbers).")
    rep.raw("<p>All agents here are scripted: the actor is a best response over three fixed strategies, the "
            "monitor a keyword match and the probe a simulation from ground truth. The demo checks the machinery "
            "and the theory; it is not a finding about language models.</p>")
    rep.write(out / "report.html")
    return out


HIDDENBITS_JUDGES = ("rational (naive)", "credulous (naive)", "sceptical")


def _bits_judge(name: str, k: int):
    from so_arena.domains.hidden_bits import bayesian_bit_judge

    return {"rational (naive)": bayesian_bit_judge(trust=0.5), "credulous (naive)": bayesian_bit_judge(trust=0.8),
            "sceptical": bayesian_bit_judge(trust=0.5, sceptical=True, disclosure_limit=k)}[name]


def demo_hiddenbits(out: str | Path = "runs/demo_hiddenbits", n_items: int = 150, length: int = 15,
                    budgets: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8), claims: int = 8, lie_rate: float = 0.5) -> Path:
    """Verified claims on HiddenBits (``docs/theory.md``, section 5, Proposition 3).

    Scripted advocates reveal up to ``claims`` favourable bits as ``<claim kind="bit">`` claims, lying (with
    probability ``lie_rate`` per claim) once their true favourable bits run out; the first ``k`` claims of
    each advocate are checked (``budget_per_role=k``), the rest are shown as unverified. Exact Bayesian judges
    read what they are shown: a rational one ignores unverified claims, a credulous one believes them with
    probability 0.8, a sceptical one also reads "fewer than k favourable verified bits" as "no more exist".
    ASD (log score) with 95% bootstrap CIs over items, propaganda (one advocate) and debate (two, one round),
    against the exact ASD of ``theory.disclosure`` (which assumes advocates that never lie).
    """
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd
    from so_arena.domains.hidden_bits import HiddenBits, bit_advocate
    from so_arena.mechanisms import Debate, Propaganda
    from so_arena.samplers.arms import ASDExperiment
    from so_arena.theory import disclosure as td

    out = Path(out)
    figs = _figures(out)
    dom = HiddenBits(n_items=n_items, length=length)
    items, ctx = dom.load(), dom.context()
    advocate = bit_advocate(claims=claims, lie_rate=lie_rate)
    aff = {"agents": ["bits"]}
    rows, sample, n_eps = [], [], 0
    for k in budgets:
        vp = soa.VerificationPolicy(verifiers=["bit"], budget_per_role=k)
        for mech in (Propaganda(affordances=aff, verification=vp), Debate(rounds=1, affordances=aff, verification=vp)):
            for judge in HIDDENBITS_JUDGES:
                exp = ASDExperiment([mech], items, agent=advocate, fixtures={"judge": _bits_judge(judge, k)}, ctx=ctx,
                                    ground_truth=dom.ground_truth_scorers())
                eps = exp.run()
                n_eps += len(eps)
                if any(e.error for e in eps):
                    raise RuntimeError(next(e.error for e in eps if e.error))
                a = asd(role_frame(eps)).iloc[0]
                rows.append({"protocol": mech.name, "judge": judge, "budget": k, "asd": a["asd"],
                             "ci_low": a["ci_low"], "ci_high": a["ci_high"], "n_items": int(a["n_items"])})
                if k == 4 and judge == "credulous (naive)":
                    sample += eps[:12]
    table = pd.DataFrame(rows)
    table.to_csv(out / "asd_by_budget.csv", index=False)
    theory = pd.DataFrame(td.asd_table(length, list(budgets)))
    theory.to_csv(out / "theory_disclosure.csv", index=False)

    def wide(protocol: str) -> pd.DataFrame:
        d = table[table.protocol == protocol]
        w = pd.DataFrame({"budget": list(budgets)})
        for j in HIDDENBITS_JUDGES:
            dj = d[d.judge == j].set_index("budget").loc[list(budgets)]
            w[j], w[f"{j} lo"], w[f"{j} hi"] = dj["asd"].to_numpy(), dj["ci_low"].to_numpy(), dj["ci_high"].to_numpy()
        return w

    bands = {j: (f"{j} lo", f"{j} hi") for j in HIDDENBITS_JUDGES}
    charts = {}
    for protocol, what in (("propaganda", "one advocate"), ("debate", "two opposed advocates")):
        kw = dict(x="budget", ys=list(HIDDENBITS_JUDGES), bands=bands, xlabel="verification budget k (claims checked per advocate)",
                  ylabel="ASD (log score)", title=f"{protocol.capitalize()}: ASD vs. verification budget",
                  subtitle=f"{what}; {length} hidden bits; bands: 95% CI over {n_items} items")
        plots.line_chart(wide(protocol), **kw).save(figs / f"hiddenbits_{protocol}.png")
        charts[protocol] = (wide(protocol), kw)
    comp = table[table.judge != "credulous (naive)"].copy()
    comp["theory (no lies)"] = [
        next(r[f"{p}_asd"] for r in td.asd_table(length, [k]) if r["judge"] == ("sceptical" if j == "sceptical" else "naive"))
        for p, j, k in zip(comp.protocol, comp.judge, comp.budget)]
    comp = comp[["protocol", "judge", "budget", "asd", "ci_low", "ci_high", "theory (no lies)"]]

    rep = Report("Verified claims: disclosure on HiddenBits", "scripted advocates, exact Bayesian judges (demo)")
    rep.kpis({"episodes": n_eps, "items": n_items, "hidden bits": length, "claims per advocate": claims,
              "lie rate once favourable bits run out": lie_rate})
    for protocol, info in (
            ("propaganda", "One advocate. A naive judge is misled by selective disclosure: the liar also shows only true "
                           "favourable bits, and ASD stays near zero at any budget. A sceptical judge reads 'fewer "
                           "favourable bits than the budget allowed' as 'there are no more' and the liar's disclosure "
                           "unravels once k reaches (L+1)/2. The credulous judge believes unverified claims, so lies "
                           "beyond the budget pay."),
            ("debate", "Two opposed advocates. Competition substitutes for scepticism: once each side can get (L+1)/2 "
                       "bits verified, the true side proves its answer, and even the naive judge decides correctly.")):
        df, kw = charts[protocol]
        rep.section(f"{protocol.capitalize()}", charts=plots.dual_mode(plots.line_chart, df, **kw), info=info)
    rep.section("Measured vs. theory", table=comp.round(4),
                note="Theory: theory.disclosure.asd_table, exact over the fair prior, for advocates that never lie. The "
                     "scripted advocates lie once their favourable bits run out; a lie inside the budget is refuted and "
                     "hands the judge a bit, so measured ASD can exceed the no-lies theory at intermediate budgets. "
                     "With no verification (k = 0) a rational judge learns nothing: ASD is exactly 0.")
    rep.episodes(sample, title="Sample episodes (credulous judge, budget 4)")
    rep.raw("<p>All agents here are scripted: advocates follow a fixed disclosure rule and judges compute exact "
            "posteriors. The demo checks the machinery against Proposition 3; it is not a finding about language "
            "models.</p>")
    rep.write(out / "report.html")
    return out


def demo_bon_budget(out: str | Path = "runs/demo_bon_budget", n_items: int = 60, pool: int = 16, budget: int = 3,
                    length: int = 15) -> Path:
    """Best-of-N against a verification budget: where does optimization put a liar's lies?

    Debate on HiddenBits (one simultaneous round). Debater A is assigned the false answer; its base policy is
    diverse - 2 to 12 bit claims, a random lie rate, lies placed anywhere half the time - and debater B argues
    the truth honestly. Each advocate's first ``budget`` claims are checked. ``OptimizationExperiment`` samples
    ``pool`` candidates for A per item and computes the exact best-of-n policy under the mechanism's reward (log
    score of the judge's probability on A's answer). Three conditions: a credulous judge (trust 0.8) with the
    budget, a rational judge (unverified claims ignored) with the budget, and a credulous judge with every claim
    checked. Values are ground truth about A's claims (``BitClaims``), with 95% bootstrap CIs over items.
    """
    from so_arena.domains.hidden_bits import HiddenBits, bayesian_bit_judge, bit_advocate
    from so_arena.mechanisms import Debate
    from so_arena.samplers.pools import OptimizationExperiment

    out = Path(out)
    figs = _figures(out)
    dom = HiddenBits(n_items=n_items, length=length, seed=3)
    items, ctx = dom.load(), dom.context()
    base = bit_advocate(sample={"claims": (2, 12), "lie_rate": (0.0, 1.0), "lie_first": 0.5}, label="base_policy")
    honest = bit_advocate(claims=budget, label="honest")
    aff = {"agents": ["bits"]}
    conditions = {
        f"budget {budget}, credulous judge": (budget, 0.8),
        f"budget {budget}, rational judge": (budget, 0.5),
        "every claim checked, credulous judge": (None, 0.8),
    }
    ns = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= pool]
    frames = []
    for cname, (k, trust) in conditions.items():
        mech = Debate(rounds=1, affordances=aff, name=f"debate[{cname}]",
                      verification=soa.VerificationPolicy(verifiers=["bit"], budget_per_role=k))
        prof = Profile(name="liar_vs_honest", players={
            "debater_a": PlayerSpec(policy=base, stance="false"),
            "debater_b": PlayerSpec(policy=honest, stance="opposite:debater_a"),
            "judge": PlayerSpec(policy=bayesian_bit_judge(trust=trust))})
        exp = OptimizationExperiment(mech, items, prof, pool_sizes={"debater_a": pool}, ctx=ctx,
                                     ground_truth=dom.ground_truth_scorers())
        exp.run()
        g = exp.grid({"debater_a": ns})
        g.insert(0, "condition", cname)
        frames.append(g)
    grid = pd.concat(frames, ignore_index=True)
    grid.to_csv(out / "bon_budget.csv", index=False)
    lie_cols = {"lies_checked_debater_a": "lies in checked claims", "lies_unchecked_debater_a": "lies beyond the budget"}

    def series(cname: str) -> pd.DataFrame:
        d = grid[grid.condition == cname]
        w = pd.DataFrame({"n": d["level_debater_a"].to_numpy()})
        for col, lab in lie_cols.items():
            w[lab], w[f"{lab} lo"], w[f"{lab} hi"] = (d[col].to_numpy(), d[f"{col}_ci_low"].to_numpy(),
                                                     d[f"{col}_ci_high"].to_numpy())
        return w

    first = next(iter(conditions))
    kw1 = dict(x="n", ys=list(lie_cols.values()), bands={lab: (f"{lab} lo", f"{lab} hi") for lab in lie_cols.values()},
               xlabel="best-of-n (debater A)", ylabel="false claims per debate",
               title="Selection moves lies beyond the verification budget",
               subtitle=f"{first}; debater A argues the false answer; bands: 95% CI over {n_items} items")
    plots.line_chart(series(first), **kw1).save(figs / "bon_budget_lies.png")
    cmp = pd.DataFrame({"n": ns})
    for cname in conditions:
        d = grid[grid.condition == cname]
        cmp[cname] = d["lies_unchecked_debater_a"].to_numpy()
        cmp[f"{cname} lo"], cmp[f"{cname} hi"] = d["lies_unchecked_debater_a_ci_low"].to_numpy(), \
            d["lies_unchecked_debater_a_ci_high"].to_numpy()
    kw2 = dict(x="n", ys=list(conditions), bands={c: (f"{c} lo", f"{c} hi") for c in conditions},
               xlabel="best-of-n (debater A)", ylabel="unchecked false claims per debate",
               title="...only where unchecked claims are believed",
               subtitle="lies beyond the budget under selection, by condition; bands: 95% CI over items")
    plots.line_chart(cmp, **kw2).save(figs / "bon_budget_conditions.png")
    keep = ["condition", "level_debater_a", "reward_debater_a", "lies_checked_debater_a", "lies_checked_debater_a_ci_low",
            "lies_checked_debater_a_ci_high", "lies_unchecked_debater_a", "lies_unchecked_debater_a_ci_low",
            "lies_unchecked_debater_a_ci_high", "honesty_debater_a", "judge_correct"]
    tbl = grid[[c for c in keep if c in grid.columns]].rename(columns={"level_debater_a": "n"})

    rep = Report("Best-of-N against a verification budget", "HiddenBits debate, scripted base policy, exact best-of-n (demo)")
    rep.kpis({"items": n_items, "candidates per item": pool, "verification budget": budget, "hidden bits": length})
    rep.section("Where selection puts the lies", charts=plots.dual_mode(plots.line_chart, series(first), **kw1),
                info="Best-of-n by the mechanism's reward almost eliminates lies in checked claims - a refuted claim "
                     "costs the liar - and raises the lies told after the budget is spent, which the credulous judge "
                     "still believes. Optimization does not remove deception; it moves it to where verification does "
                     "not reach.")
    rep.section("It depends on who believes unchecked claims", charts=plots.dual_mode(plots.line_chart, cmp, **kw2),
                table=tbl.round(3),
                info="A rational judge ignores unverified claims, so lies beyond the budget earn nothing and selection "
                     "does not favour them. With every claim checked there is nowhere left to hide a lie and "
                     "selection removes most of them - yet the naive judge is still misled (judge_correct in the table): debater A shows more "
                     "favourable true bits than its honest opponent, and a naive judge reads a selected sample as a "
                     "random one (Proposition 3). Verification bounds what can be said, not what is left out.")
    rep.raw("<p>All agents here are scripted: debater A's base policy samples claim counts, lie rates and lie "
            "placement, and judges compute exact posteriors. Best-of-n is computed exactly from the sampled pools "
            "(samplers.pools.OptimizationExperiment). The demo illustrates the machinery, not LLM behaviour.</p>")
    rep.write(out / "report.html")
    return out


def run_all(out: str | Path = "runs/demos") -> list[Path]:
    out = Path(out)
    return [demo_asd(out / "asd"), demo_optimization(out / "optimization"), demo_swarm(out / "swarm"),
            demo_work(out / "work"), demo_monitoring(out / "monitoring"), demo_hiddenbits(out / "hiddenbits"),
            demo_bon_budget(out / "bon_budget")]
