"""Offline demonstrations (no API keys): each runs a small experiment end to end and writes a report.

These use the synthetic domains, so the numbers illustrate the *machinery* - they are not findings
about language models. Swap the synthetic policies for ``LLMPolicy`` and a real domain to run the
same experiments for real (see ``configs/`` and ``docs/experiments.md``).

* :func:`demo_asd` - ASD across protocols, with and without verified claims.
* :func:`demo_optimization` - best-of-N optimization pressure on a proposer, with and without a critic
  (the "debate with self-play best-of-N" design): does optimizing against the mechanism raise accuracy?
* :func:`demo_swarm` - reporting incentives in teams: analytic equilibria, a simulated empirical game,
  and policy-gradient training dynamics.
"""

from __future__ import annotations

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


def demo_swarm(out: str | Path = "runs/demo_swarm") -> Path:
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
                                subtitle="Team of n; bounty s for a verified report; hack worth Δ to everyone",
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
        subtitle="Team of n; bounty s for a verified report; hack worth Δ to everyone",
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
    rep.write(out / "report.html")
    return out


def run_all(out: str | Path = "runs/demos") -> list[Path]:
    out = Path(out)
    return [demo_asd(out / "asd"), demo_optimization(out / "optimization"), demo_swarm(out / "swarm")]
