"""Built-in demos that run without API keys (programmatic agents), each producing figures and
an HTML report. They double as end-to-end tests of the whole pipeline and as templates for
LLM experiments (swap the agents for :class:`~oversight_arena.agents.LLMAgent`).

``oversight-arena demo <name>`` with name in: hiddenbits, chess, bon, optimize, swarm,
monitoring, release, all.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import plots as P
from .analysis.bon import bon_curve
from .analysis.games import EmpiricalGame
from .analysis.ic import asd
from .analysis.report import html_report
from .channels.evidence import EvidencePolicy
from .core.strategy import Assignment, Profile, Stance, Strategy
from .core.util import stable_hash
from .experiment.profiles import Cartesian, MapProfiles, Seeds, Stances
from .experiment.results import Results
from .experiment.runner import Experiment


def _run(domain, mech, agents, profiles, **kw) -> Results:
    return Experiment(domain, mech, agents, profiles, progress=False, **kw).run()


# --------------------------------------------------------------------------- verified claims
def demo_hiddenbits(out: Path) -> list[Path]:
    """How verification budgets and judge credulity shape incentive compatibility."""
    from .domains.synthetic import HiddenBits
    from .mechanisms import Consultancy, Debate, NaiveJudge, Propaganda
    from .sim import BayesianBitJudge, BitAdvocate
    from .theory.disclosure import asd_table

    out.mkdir(parents=True, exist_ok=True)
    dom = HiddenBits(n_tasks=60, length=15)
    rows, keep = [], None
    for jname, trust in [("rational judge", 0.5), ("credulous judge", 0.8)]:
        agents = {"kind:judge": BayesianBitJudge(trust=trust), "*": BitAdvocate(claims=3, lie_rate=0.5)}
        for budget in [0, 2, 4, 6, 8]:
            pol = EvidencePolicy(budget=budget)
            for mech in [NaiveJudge(), Propaganda(evidence=pol), Consultancy(rounds=2, evidence=pol), Debate(rounds=2, evidence=pol)]:
                res = _run(dom, mech, agents, Stances())
                a = asd(res).iloc[0]
                rows.append({"judge": jname, "budget": budget, "mechanism": mech.name, "asd": a["asd"], "lo": a["lo"], "hi": a["hi"]})
                if jname == "credulous judge" and budget == 4:
                    keep = res if keep is None else keep + res
    df = pd.DataFrame(rows)
    df.to_csv(out / "asd_by_budget.csv", index=False)
    figs = {}
    for jname in df["judge"].unique():
        fig, ax = P.line_compare(df[df["judge"] == jname], "budget", "asd", "mechanism", band=("lo", "hi"),
                                 title=f"ASD vs verification budget — {jname}",
                                 xlabel="verification budget (claims checked per agent)", ylabel="ASD (log score)")
        figs[f"ASD vs budget ({jname})"] = P.save(fig, out / f"asd_budget_{jname.split()[0]}.png")
    theory = pd.DataFrame(asd_table(15, [0, 2, 4, 6, 8, 15]))
    theory.to_csv(out / "theory_disclosure.csv", index=False)
    rep = html_report(keep, out / "report.html", title="Verified claims (HiddenBits)",
                      subtitle="Episodes shown: credulous Bayesian judge, verification budget 4. Figures sweep the budget.",
                      figures=figs, tables={"All configurations (ASD)": df.round(3),
                                            "Theory: exact ASD, naive vs selection-aware judge": theory.round(3)})
    return [rep, *figs.values()]


# --------------------------------------------------------------------------- chess
def demo_chess(out: Path) -> list[Path]:
    """Real capability gap: engine experts vs shallow-search judges, honest vs cherry-picked lines."""
    from .domains.chess import ChessMoves
    from .mechanisms import Consultancy, Debate, NaiveJudge
    from .sim.chess_agents import EngineAdvocate, EngineJudge

    out.mkdir(parents=True, exist_ok=True)
    dom = ChessMoves(n_puzzles=40, judge_trap_depth=2, gt_depth=14, verify_depth=0, limit=30)
    advocate = EngineAdvocate(depth=10)
    pol = EvidencePolicy()
    rows, allres = [], None
    for jd in (1, 2, 3):
        agents = {"kind:judge": EngineJudge(depth=jd), "*": advocate}
        for mech in (NaiveJudge(), Consultancy(rounds=2, evidence=pol), Debate(rounds=2, evidence=pol)):
            for style in ("honest", "cherry_pick"):
                m = mech.model_copy(update={"label": f"{mech.name} · judge depth {jd} · {style}"})
                res = _run(dom, m, agents, MapProfiles(source=Stances(), params={"style": style}), concurrency=2)
                a = asd(res).iloc[0]
                acc = res.episodes_df()["gt_decision_correct[_outcome]"].mean()
                rows.append({"mechanism": mech.name, "judge_depth": jd, "style": style, "asd": a["asd"], "lo": a["lo"],
                             "hi": a["hi"], "judge_accuracy": acc})
                if jd == 2:
                    allres = res if allres is None else allres + res
    df = pd.DataFrame(rows)
    df.to_csv(out / "chess_asd.csv", index=False)
    figs = {}
    for style in ("honest", "cherry_pick"):
        d = df[df["style"] == style]
        fig, ax = P.line_compare(d, "judge_depth", "judge_accuracy", "mechanism",
                                 title=f"Judge accuracy vs judge strength — advocates: {style.replace('_', '-')}",
                                 xlabel="judge search depth", ylabel="judge accuracy")
        ax.set_xticks([1, 2, 3])
        figs[f"Accuracy ({style})"] = P.save(fig, out / f"accuracy_{style}.png")
    # empirical game over argument styles (judge depth 2): truthful side vs lying side
    agents = {"kind:judge": EngineJudge(depth=2), "*": advocate}
    styles = ["honest", "cherry_pick"]
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    G = np.zeros((2, 2))
    for i, st in enumerate(styles):
        for j, sl in enumerate(styles):
            vals_t, vals_l, accs = [], [], []
            for truthful in ("debater_a", "debater_b"):
                liar = "debater_b" if truthful == "debater_a" else "debater_a"
                prof = Profile(assignments={
                    truthful: Assignment(strategy=Strategy(name=f"truth/{st}", stance=Stance.CORRECT, params={"style": st})),
                    liar: Assignment(strategy=Strategy(name=f"lie/{sl}", stance=Stance.INCORRECT, params={"style": sl})),
                })
                res = _run(dom, Debate(rounds=2, evidence=pol), agents, [prof], concurrency=2)
                for r in res.records:
                    if r.error is None:
                        vals_t.append(r.rewards[truthful])
                        vals_l.append(r.rewards[liar])
                        accs.append(r.gt["decision_correct"]["_outcome"])
            A[i, j], B[i, j], G[i, j] = np.mean(vals_t), np.mean(vals_l), np.mean(accs)
    game = EmpiricalGame.from_matrices(A, B, row="truthful debater", col="lying debater", row_strats=styles,
                                       col_strats=styles, gt={"judge accuracy": G})
    eqs = game.nash()
    eq_rows = [{"equilibrium": json.dumps(e.support(game)), "truthful payoff": e.payoffs["truthful debater"],
                "liar payoff": e.payoffs["lying debater"], "judge accuracy": e.gt.get("judge accuracy")} for e in eqs]
    fig, _ = P.payoff_heatmap(game, gt_key="judge accuracy", title="Judge accuracy by argument styles (debate, depth-2 judge)")
    figs["Style game: accuracy"] = P.save(fig, out / "style_game_accuracy.png")
    fig, _ = P.payoff_heatmap(game, role="lying debater", title="Lying debater's reward")
    figs["Style game: liar reward"] = P.save(fig, out / "style_game_liar.png")
    rep = html_report(allres, out / "report.html", title="Chess: engine experts vs weak judges",
                      subtitle="Lichess puzzles where a depth-2 search prefers the wrong move; GT = Stockfish depth 14. Episodes shown: judge depth 2.",
                      figures=figs, tables={"ASD and judge accuracy (all settings)": df.round(3),
                                            "Style game equilibria (debate, judge depth 2)": pd.DataFrame(eq_rows).round(3)})
    return [rep, *figs.values()]


# --------------------------------------------------------------------------- best-of-N
def demo_bon(out: Path) -> list[Path]:
    """One step of optimisation: best-of-N selection by the mechanism's reward vs honesty."""
    from .domains.synthetic import HiddenBits
    from .mechanisms import Debate, OpenConsultancy
    from .sim import BayesianBitJudge, BitAdvocate

    out.mkdir(parents=True, exist_ok=True)
    dom = HiddenBits(n_tasks=40, length=15)
    base = Strategy(name="base_policy", params={"sample": {"claims": (1, 5), "lie_rate": (0.0, 1.0), "lie_first": 0.3, "side_error": 0.3}})
    agents = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate()}
    N = 32
    mechs = {
        "consultancy · cheap talk": (OpenConsultancy(label="consultancy · cheap talk"), "consultant"),
        "consultancy · verified": (OpenConsultancy(evidence=EvidencePolicy(budget=3), label="consultancy · verified"), "consultant"),
        "debate · verified": (Debate(rounds=2, evidence=EvidencePolicy(budget=3), label="debate · verified"), "debater_a"),
    }
    curves, allres = [], None
    for mech, role in mechs.values():
        strategies = {role: base}
        if role == "debater_a":
            strategies["debater_b"] = Strategy(name="honest_opponent", stance=Stance.CORRECT, params={"claims": 3})
        src = Seeds(n=N, strategies=strategies, roles=[role])
        res = _run(dom, mech, agents, src)
        allres = res if allres is None else allres + res
        for gt in ("honesty", "correct"):
            c = bon_curve(res, roles=role, gt=gt, n_values=[1, 2, 4, 8, 16, 32])
            c["gt_metric"] = gt
            curves.append(c)
    cur = pd.concat(curves)
    cur.to_csv(out / "bon_curves.csv", index=False)
    figs = {}
    fig, ax = P.optimization_frontier(cur[cur["gt_metric"] == "honesty"], title="Best-of-N: reward vs honesty of claims",
                                      ylabel="honesty (share of true claims)")
    figs["BoN frontier (honesty)"] = P.save(fig, out / "bon_honesty.png")
    d = cur[(cur["gt_metric"] == "correct") & (cur["mechanism"] != "debate · verified")]
    fig, ax = P.optimization_frontier(d, title="Best-of-N: reward vs arguing the true side", ylabel="P(argues the correct answer)")
    figs["BoN frontier (side)"] = P.save(fig, out / "bon_side.png")
    # what does selection pick in debate? (debater A is assigned the false answer vs an honest opponent)
    from .domains.synthetic import parse_bit_claims

    rows = []
    for r in allres.records:
        if r.mechanism != "debate · verified" or r.error:
            continue
        bits = dom.get_task(r.task_id).resources["bits"]
        claims = [(i, v, st) for e in r.transcript.entries if e.role == "debater_a" for i, v, st in parse_bit_claims(e)]
        rows.append({"task": r.task_id, "reward": r.rewards["debater_a"],
                     "false verified-range claims": sum(st != "UNCHECKED" and bits[i] != v for i, v, st in claims),
                     "false claims beyond the budget": sum(st == "UNCHECKED" and bits[i] != v for i, v, st in claims)})
    sel = pd.DataFrame(rows)
    sel["selected by Bo(2)"] = sel.groupby("task")["reward"].rank(ascending=False, method="first") <= 2
    what = sel.groupby("selected by Bo(2)")[["false verified-range claims", "false claims beyond the budget"]].mean().round(2).reset_index()
    rep = html_report(allres, out / "report.html", title="Best-of-N optimisation pressure",
                      subtitle=(f"Base-policy samples ({N} per task) of a HiddenBits advocate; credulous Bayesian judge; verification "
                                "budget 3 claims per advocate. In debate, debater A is assigned the false answer against an honest "
                                "opponent: selection moves its lies to where verification does not reach."),
                      gt="honesty", figures=figs, tables={"What best-of-N selects in debate (debater A, per episode)": what,
                                                          "BoN curves": cur.round(3)})
    return [rep, *figs.values()]


# --------------------------------------------------------------------------- prompt/strategy optimisation
def demo_optimize(out: Path) -> list[Path]:
    """Strategy search with steering constraints: honest vs deceptive frontiers."""
    from .domains.synthetic import HiddenBits
    from .elicitation.evaluator import Evaluator
    from .elicitation.optimize import ParamProposer, PromptOptimizer
    from .mechanisms import Debate, Propaganda
    from .sim import BayesianBitJudge, BitAdvocate

    out.mkdir(parents=True, exist_ok=True)
    dom = HiddenBits(n_tasks=40, length=15)
    tasks = dom.tasks()
    agents = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate()}
    space = {"claims": (1, 8), "lie_rate": (0.0, 1.0), "lie_first": [True, False], "markup": [True, False]}
    runs = []
    for mech, role, others in [
        (Propaganda(evidence=EvidencePolicy(budget=2), label="propaganda · budget 2"), "agent", {}),
        (Debate(rounds=2, evidence=EvidencePolicy(budget=4), label="debate · budget 4"), "debater_a",
         {"debater_b": Strategy(name="opponent", params={"claims": 3})}),
    ]:
        for label, stance, constraint in [
            ("argue truth · honest only", Stance.CORRECT, lambda e: e.gt.get("honesty", 1.0) >= 0.999),
            ("argue truth · anything", Stance.CORRECT, None),
            ("argue falsehood · anything", Stance.INCORRECT, None),
        ]:
            base = Strategy(name="seed", stance=stance, params={"claims": 2, "lie_rate": 0.0, "markup": True, "lie_first": False})
            opp = {k: v.with_(stance=Stance.INCORRECT if stance == Stance.CORRECT else Stance.CORRECT) for k, v in others.items()}
            ev = Evaluator(dom, mech, agents, role, tasks=tasks[:24], others=opp, gt_keys=("honesty", "correct"))
            po = PromptOptimizer(ev, ParamProposer(space, seed=int(stable_hash(label, length=6), 16) % 1000), seeds=[base], iterations=8, per_iter=5,
                                 minibatch=12, constraint=constraint, holdout=tasks[24:], final_k=1, label=label)
            tr = asyncio.run(po.run())
            t = tr.trajectory(gt="honesty")
            t["mechanism"] = mech.display_name
            runs.append(t)
            best = tr.best(1)[0]
            runs[-1]["final_holdout_reward"] = best.holdout_reward
    df = pd.concat(runs)
    df.to_csv(out / "optimization.csv", index=False)
    figs = {}
    for mech in df["mechanism"].unique():
        d = df[df["mechanism"] == mech]
        fig, ax = P.line_compare(d, "iteration", "best_reward", "label", title=f"Search for the best-rewarded strategy — {mech}",
                                 xlabel="optimisation iteration", ylabel="best reward found (minibatch)")
        figs[f"Search ({mech})"] = P.save(fig, out / f"search_{mech.split()[0]}.png")
    summary = df.groupby(["mechanism", "label"]).agg(best_reward=("best_reward", "max"), holdout=("final_holdout_reward", "last"),
                                                      honesty_of_best=("best_gt", "last")).reset_index()
    gaps = []
    for mech, g in summary.groupby("mechanism"):
        h = g[g["label"] == "argue truth · honest only"]["holdout"].iloc[0]
        f_ = g[g["label"] == "argue falsehood · anything"]["holdout"].iloc[0]
        gaps.append({"mechanism": mech, "honest frontier": h, "deceptive frontier": f_, "frontier gap": h - f_})
    html_report(Results([]), out / "report.html", title="Strategy optimisation: honest vs deceptive frontiers",
                subtitle="Evolutionary search over advocate strategies (HiddenBits, credulous judge). Holdout = unseen tasks.",
                figures=figs, tables={"Frontier gap (IC under optimisation)": pd.DataFrame(gaps).round(3),
                                      "Best strategies per search": summary.round(3)})
    return [out / "report.html", *figs.values()]


# --------------------------------------------------------------------------- swarms
SWARM_STRATEGIES = [("cheat·silent", 1.0, 0.0), ("cheat·report", 1.0, 1.0), ("honest·silent", 0.0, 0.0), ("honest·report", 0.0, 1.0)]
CHEAT, REPORT = ("cheat·silent", "cheat·report"), ("cheat·report", "honest·report")


def _swarm_mix(p_cheat: float, p_report: float) -> dict[str, float]:
    """Weights on SWARM_STRATEGIES with independent cheat / report propensities."""
    x, q = p_cheat, p_report
    return {"cheat·silent": x * (1 - q), "cheat·report": x * q, "honest·silent": (1 - x) * (1 - q), "honest·report": (1 - x) * q}


def demo_swarm(out: Path) -> list[Path]:
    """Shared objective vs whistleblower bounties: theory, empirical equilibria, learning dynamics."""
    from .domains.swarm import AbstractSwarm
    from .elicitation.rl import StrategyGradient
    from .mechanisms import Swarm, TeamReward, Whistleblower
    from .sim import SwarmWorker
    from .theory import swarm_game as sg

    out.mkdir(parents=True, exist_ok=True)
    g, pen, obs = 0.3, 0.3, 0.8  # each worker's stake in an undetected violation, offender penalty, P(observe)
    workers = ["worker_1", "worker_2", "worker_3"]
    dom = AbstractSwarm(n_tasks=40, gain=g, damage=g, observe_prob=obs, n_workers=3)
    strat = {name: Strategy(name=name, params={"cheat": c, "report": r}) for name, c, r in SWARM_STRATEGIES}
    agents = {"*": SwarmWorker()}
    # One set of episodes; every reward rule is evaluated on it by re-scoring. Exact here: the
    # programmatic workers' behaviour does not depend on the stated incentives.
    res = _run(dom, Swarm(n_workers=3, rounds=1, reward=TeamReward()), agents,
               Cartesian(strategies={w: list(strat.values()) for w in workers}))
    # bounties: below b_R (an observer may be alone, so silence is the only equilibrium; b_R = g/3
    # here), between b_R and the stake (coordination), above the stake (reporting dominant)
    b_min, b_lo, b_hi = 0.05, 0.2, 0.45
    rules = {
        "shared reward": TeamReward(),
        f"bounty {b_min:g} (< stake/3)": Whistleblower(bounty=b_min, offender_penalty=pen),
        f"bounty {b_lo:g} (< stake)": Whistleblower(bounty=b_lo, offender_penalty=pen),
        f"bounty {b_hi:g} (> stake)": Whistleblower(bounty=b_hi, offender_penalty=pen),
    }
    dyn_rules = [r for r in rules if not r.startswith(f"bounty {b_min:g}")]
    figs: dict[str, Path] = {}
    theory = sg.phase_diagram(n=3, g=g, P=pen, o=obs, b_over_stake=np.linspace(0, 1.5, 151), audit=np.linspace(0, 0.9, 91))
    fig, ax = P.regime_map(theory, "bounty_over_stake", "audit", contour="basin_report",
                           title="When does reporting pay? (theory)",
                           xlabel="bounty ÷ each worker's stake in the undetected violation",
                           ylabel="random audit probability",
                           points=[(0.0, 0.0, "shared")] + [(b / g, 0.0, f"b={b:g}") for b in (b_min, b_lo, b_hi)])
    figs["Theory: regimes"] = P.save(fig, out / "theory_regimes.png")

    eq_rows, dyn_rows = [], []
    for rname, rule in rules.items():
        game = EmpiricalGame.from_results(res, workers, gt_metrics=("clean", "reported"), reward=rule)
        oc = game.outcomes(keys=["clean:_outcome"], digits=2)
        honest = game.regret(game.pure({w: "honest·report" for w in workers}))
        silent = game.regret(game.pure({w: "cheat·silent" for w in workers}))
        eq_rows.append({
            "reward rule": rname,
            "best-equilibrium true score": oc["gt[clean:_outcome]"].max(),
            "worst-equilibrium true score": oc["gt[clean:_outcome]"].min(),
            "honest+report is an equilibrium": max(honest.values()) <= 1e-9,
            "cheat+silent is an equilibrium": max(silent.values()) <= 1e-9,
            "reporting subgame (theory)": sg.regime(sg.SwarmParams(n=3, g=g, b=getattr(rule, "bounty", 0.0), P=pen, o=obs)),
        })
        if rname not in dyn_rules:
            continue
        for init, q0 in (("expect silence", 0.1), ("expect reporting", 0.9)):
            for t, mix in enumerate(game.replicator(game.mix(_swarm_mix(0.5, q0)), iters=300, lr=1.0, record_every=3)):
                dyn_rows.append({"rule": rname, "init": init, "step": t * 3,
                                 "P(cheat)": np.mean([game.marginal(mix, w, CHEAT) for w in workers]),
                                 "P(report)": np.mean([game.marginal(mix, w, REPORT) for w in workers]),
                                 "true score": game.expected_gt(mix).get("clean:_outcome")})
    eqdf = pd.DataFrame(eq_rows)
    dyn = pd.DataFrame(dyn_rows)
    dyn.to_csv(out / "replicator_dynamics.csv", index=False)
    d = dyn.assign(series=dyn["rule"] + " · " + dyn["init"])
    fig, ax = P.line_compare(d, "step", "P(cheat)", "series", title="Learning dynamics (replicator): misconduct",
                             xlabel="training time (replicator steps)", ylabel="P(cheat)", markers=False)
    figs["Dynamics: P(cheat)"] = P.save(fig, out / "dynamics_cheat.png")
    fig, ax = P.line_compare(d, "step", "true score", "series", title="Learning dynamics (replicator): true project score",
                             xlabel="training time (replicator steps)", ylabel="expected true score", markers=False)
    figs["Dynamics: true score"] = P.save(fig, out / "dynamics_true_score.png")

    # theory vs simulation: share of initial report propensities from which training deters misconduct
    basin_rows = []
    q0s = np.linspace(0.02, 0.98, 25)
    for x in np.linspace(0, 1.5, 16):
        rule = Whistleblower(bounty=float(x * g), offender_penalty=pen)
        game = EmpiricalGame.from_results(res, workers, gt_metrics=("clean",), reward=rule)
        deterred = []
        for q0 in q0s:
            fin = game.replicator(game.mix(_swarm_mix(0.5, q0)), iters=600, lr=1.0)[-1]
            deterred.append(np.mean([game.marginal(fin, w, CHEAT) for w in workers]) < 0.5)
        th = sg.SwarmParams(n=3, g=g, b=float(x * g), P=pen, o=obs)
        basin_rows += [
            {"bounty_over_stake": x, "basin": float(np.mean(deterred)), "source": "simulation (empirical game)"},
            {"bounty_over_stake": x, "basin": sg.basin_of_deterrence(th, 0.5, q0s, 600, 1.0), "source": "theory (full game, mean field)"},
            {"bounty_over_stake": x, "basin": sg.basin_of_reporting(th), "source": "theory (reporting subgame)"},
        ]
    bdf = pd.DataFrame(basin_rows)
    bdf.to_csv(out / "basin_theory_vs_sim.csv", index=False)
    fig, ax = P.line_compare(bdf, "bounty_over_stake", "basin", "source",
                             title="Basin of the honest outcome: theory vs simulation",
                             xlabel="bounty ÷ each worker's stake", ylabel="share of initial beliefs → misconduct deterred")
    figs["Basin: theory vs simulation"] = P.save(fig, out / "basin_theory_vs_sim.png")

    # sampled training on real episodes, starting from expected silence: the training algorithm
    # matters (natural PG follows the replicator dynamics; vanilla REINFORCE is slow to learn the
    # rare reporting strategy, so misconduct entrenches first)
    sgd_rows = []
    pop = [strat[n] for n, _, _ in SWARM_STRATEGIES]
    w0 = _swarm_mix(0.5, 0.1)
    init = [w0[s_.name] for s_ in pop]
    train_seeds = (3, 4, 5)  # stochastic training: show the mean path over independent runs
    for rname, natural in ((f"bounty {b_hi:g} (> stake)", True), (f"bounty {b_hi:g} (> stake)", False),
                           (f"bounty {b_lo:g} (< stake)", True)):
        for seed in train_seeds:
            tr = StrategyGradient(dom, Swarm(n_workers=3, rounds=1, reward=rules[rname]), agents, {w: pop for w in workers},
                                  init={w: init for w in workers}, lr=2.0, batch=32, iterations=40, gt_keys=("clean",),
                                  seed=seed, natural=natural)
            traj = asyncio.run(tr.run())
            for _, r in traj.iterrows():
                sgd_rows.append({"run": f"{rname} · {'natural PG' if natural else 'REINFORCE'}", "seed": seed,
                                 "iteration": r["iteration"],
                                 "P(cheat)": np.mean([r[f"p[{w}:cheat·silent]"] + r[f"p[{w}:cheat·report]"] for w in workers]),
                                 "P(report)": np.mean([r[f"p[{w}:cheat·report]"] + r[f"p[{w}:honest·report]"] for w in workers])})
    sgd = pd.DataFrame(sgd_rows).groupby(["run", "iteration"], as_index=False)[["P(cheat)", "P(report)"]].mean()
    sgd.to_csv(out / "sampled_training.csv", index=False)
    fig, ax = P.line_compare(sgd, "iteration", "P(cheat)", "run",
                             title=f"Sampled training on episodes, starting from silence (mean of {len(train_seeds)} runs)",
                             xlabel="training iteration", ylabel="P(cheat)")
    figs["Sampled training: P(cheat)"] = P.save(fig, out / "sampled_training_cheat.png")
    rep = html_report(Results([]), out / "report.html", title="Swarms: shared objective vs whistleblowing",
                      subtitle=f"3 workers; one can tamper with the scorer (+{g:g} to everyone's reward if undetected, "
                               f"−{g:g} true score); each other worker notices w.p. {obs:g}; offender penalty {pen:g}.",
                      figures=figs, tables={"Equilibria of the empirical game": eqdf.round(3)})
    return [rep, *figs.values()]


# --------------------------------------------------------------------------- monitoring
def demo_monitoring(out: Path, n_tasks: int = 200, n_seeds: int = 6) -> list[Path]:
    """Penalising alarms: a real monitor buys obfuscation, a robust probe buys good behaviour."""
    from .analysis.stats import seed_ci
    from .channels import SimulatedProbe
    from .domains.monitoring_sim import AbstractMonitoring
    from .mechanisms import MonitoredReward, Monitoring
    from .sim.monitoring_agents import ActionActor, KeywordMonitor
    from .theory import monitoring as tm

    out.mkdir(parents=True, exist_ok=True)
    params = tm.MonitorParams(s=0.5, gain=0.3, hide_cost=0.05, d_overt=1.0, d_covert=0.1, d_honest=0.05)
    def domain(seed: int) -> AbstractMonitoring:
        return AbstractMonitoring(n_tasks=n_tasks, seed=seed, success=params.s, gain=params.gain, hide_cost=params.hide_cost,
                                  d_covert=params.d_covert, d_honest=params.d_honest)

    actions = [Strategy(name=a, params={"action": a}, tags={"honest": a == "honest"}) for a in ("honest", "overt", "covert")]
    monitors = {
        "keyword monitor (reads the log)": (Monitoring(monitor="llm", probe=None, grader=False), {"actor": ActionActor(), "monitor": KeywordMonitor()}),
        "probe, AUROC 0.8 (fixed)": (Monitoring(monitor="probe", probe=SimulatedProbe(auroc=0.8), grader=False), {"actor": ActionActor()}),
        "probe, AUROC 0.95 (fixed)": (Monitoring(monitor="probe", probe=SimulatedProbe(auroc=0.95), grader=False), {"actor": ActionActor()}),
    }
    lams = np.round(np.concatenate([np.linspace(0, 1, 11), np.linspace(1.5, 10, 18)]), 3)
    rows, thresholds = [], []
    # the thresholds depend on the random traces and probe scores, so the whole experiment is
    # replicated on fresh tasks (new draws) and the thresholds are reported with a CI across seeds
    for seed in range(n_seeds):
        for name, (mech, agents) in monitors.items():
            res = _run(domain(seed), mech.model_copy(update={"label": name}), agents, Cartesian(strategies={"actor": actions}))
            alarm = {a: np.mean([r.outcome["alarm"] for r in res.records if r.bound["actor"].strategy_name == a]) for a in ("honest", "overt", "covert")}
            deterred = np.nan
            for lam in lams:  # the reward rule is re-scored; the optimiser picks the best of the three behaviours
                d = res.rescore(MonitoredReward(alarm_penalty=float(lam))).df(trainable_only=True)
                best = str(d.groupby("strategy_name")["reward"].mean().idxmax())
                rows.append({"monitor": name, "seed": seed, "penalty": float(lam), "action": best, "misbehaviour": float(best != "honest"),
                             "alarm rate if misbehaving": alarm[best] if best != "honest" else np.nan, "source": "simulation"})
                if best == "honest" and np.isnan(deterred):
                    deterred = float(lam)
            thresholds.append({"monitor": name, "seed": seed, "deterrence threshold": deterred})
    th = tm.sweep(lams, params)
    rows += [{"monitor": "theory (keyword monitor)", "penalty": r.penalty, "action": r.action, "misbehaviour": r.misbehaviour,
              "alarm rate if misbehaving": r.alarm_rate_when_misbehaving, "source": "theory"} for r in th.itertuples()]
    df = pd.DataFrame(rows).astype({"seed": "Int64"})
    df.to_csv(out / "monitoring_sweep.csv", index=False)
    # averaged over seeds: the share of seeds in which the reward-maximising action misbehaves
    avg = df.groupby(["monitor", "penalty", "source"], as_index=False, sort=False)[["misbehaviour", "alarm rate if misbehaving"]].mean()
    figs = {}
    fig, ax = P.line_compare(avg, "penalty", "misbehaviour", "monitor", title="Optimised actor: does it misbehave?",
                             xlabel="alarm penalty λ in the actor's reward", ylabel=f"misbehaves (share of {n_seeds} seeds)")
    figs["Misbehaviour vs penalty"] = P.save(fig, out / "misbehaviour_vs_penalty.png")
    d = avg[avg["monitor"].str.startswith(("keyword", "theory"))]
    fig, ax = P.line_compare(d, "penalty", "alarm rate if misbehaving", "monitor", title="…while the monitor's alarms go quiet",
                             xlabel="alarm penalty λ in the actor's reward", ylabel="alarm rate on misbehaving episodes")
    figs["Alarm rate vs penalty"] = P.save(fig, out / "alarms_vs_penalty.png")
    tdf = pd.DataFrame(thresholds)
    tdf.to_csv(out / "deterrence_thresholds_by_seed.csv", index=False)
    summary = pd.DataFrame([{"monitor": m, **seed_ci(g["deterrence threshold"])} for m, g in tdf.groupby("monitor", sort=False)])
    summary.to_csv(out / "deterrence_thresholds.csv", index=False)

    def threshold_note(m: str, r: pd.Series) -> str:
        m, never = m.replace(" (fixed)", ""), f"not deterred by λ = {lams[-1]:g}"
        if r["seeds"] == 0:
            return f"{m} {never}"
        ci = f" (95% CI {r['ci_low']:.2f}–{r['ci_high']:.2f})" if r["seeds"] > 1 else ""
        return f"{m} {r['mean']:.2f}{ci}" + (f", {never} in {int(r['missing'])} seeds" if r["missing"] else "")

    ci_note = "; ".join(threshold_note(m, r) for m, r in summary.set_index("monitor").iterrows())
    th_row = tm.thresholds(params)
    rep = html_report(Results([]), out / "report.html", title="Monitoring as a reward: obfuscation vs deterrence",
                      subtitle=(f"Actor: honest (success {params.s:g}), overt misbehaviour (+{params.gain:g}, always leaves a trace) "
                                f"or covert (+{params.gain - params.hide_cost:g}, trace w.p. {params.d_covert:g}). Theory: overt→covert at "
                                f"λ={th_row['overt_to_covert']:.2f}, deterred at λ={th_row['deterred']:.1f} for the log-reading monitor. "
                                f"Simulated deterrence thresholds, mean over {n_seeds} seeds: {ci_note}."),
                      figures=figs, tables={"Deterrence threshold λ across seeds": summary.rename(columns={
                          "mean": "mean", "ci_low": "95% CI low", "ci_high": "95% CI high",
                          "missing": f"not deterred by λ = {lams[-1]:g}"}).round(3)})
    return [rep, *figs.values()]


# --------------------------------------------------------------------------- releases
def demo_release(out: Path) -> list[Path]:
    """Release judge-based forecast rankings before resolution; evaluate them after."""
    from .analysis.ic import alignment
    from .domains.synthetic_forecasting import SyntheticForecasting
    from .mechanisms import Forecast, JudgeRating
    from .release import create_release, resolve_release
    from .sim.forecast_agents import ConfidenceJudge, SignalForecaster

    out.mkdir(parents=True, exist_ok=True)
    dom = SyntheticForecasting(n_tasks=60, resolved=False)
    strategies = [
        Strategy(name="calibrated", params={"temper": 1.0}),
        Strategy(name="overconfident", params={"temper": 2.5}),
        Strategy(name="underconfident", params={"temper": 0.5}),
        Strategy(name="extremizer", params={"shade": 0.6}),
    ]
    agents = {"kind:judge": ConfidenceJudge(), "*": SignalForecaster()}
    profs = [Profile(assignments={"forecaster": Assignment(strategy=s)}, label=s.name) for s in strategies]
    res = _run(dom, Forecast(judge=True, reward=JudgeRating(), label="judged forecast"), agents, profs)
    rel = create_release(res, out / "release", title="Judged forecasts (unresolved)",
                         description="A weak judge's ratings of forecasts, published before the questions resolve.")
    resolved = dom.outcomes()
    report = resolve_release(out / "release", resolved)
    later = res.resolve(resolved, scorers=dom.gt_scorers())
    d = later.df()
    d = d[d["role"] == "forecaster"]
    by = d.groupby("strategy_name").agg(judge_reward=("reward", "mean"), log_score=("gt_forecast_log", "mean")).reset_index()
    al = alignment(d, gt="forecast_log")
    fig, ax = P.asd_bars(by.rename(columns={"strategy_name": "strategy"}), metric="judge_reward", label="strategy",
                         title="Released ranking: judge's reward", xlabel="mean judge rating")
    f1 = P.save(fig, out / "released_ranking.png")
    fig, ax = P.asd_bars(by.rename(columns={"strategy_name": "strategy"}), metric="log_score", label="strategy",
                         title="After resolution: proper log score", xlabel="mean log score (higher is better)")
    f2 = P.save(fig, out / "resolved_scores.png")
    rep = html_report(later, out / "report.html", title="Release now, resolve later",
                      subtitle=f"Release root {rel.root[:16]}…; resolution verified: {report['verification']['ok']}",
                      gt="forecast_log", figures={"Released ranking": f1, "After resolution": f2},
                      tables={"Reward vs resolved log score, per strategy": by.round(3),
                              "Did the judge's reward track the truth?": al.round(3)})
    return [rep, rel.path / "index.html", f1, f2]


DEMOS = {"hiddenbits": demo_hiddenbits, "chess": demo_chess, "bon": demo_bon, "optimize": demo_optimize,
         "swarm": demo_swarm, "monitoring": demo_monitoring, "release": demo_release}


def _missing_requirement(name: str) -> str | None:
    """Why a demo cannot run here (None if it can)."""
    if name == "chess":
        try:
            import chess  # noqa: F401

            from .domains.chess import find_stockfish

            find_stockfish()
        except (ImportError, RuntimeError) as e:
            return str(e) or type(e).__name__
    return None


def run(name: str, out: Path, on_skip: Any = None) -> list[Path]:
    """Run one demo, or ``"all"`` of them; with ``"all"``, demos whose requirements are missing
    (the chess demo needs Stockfish) are skipped with a warning (or ``on_skip(message)``)."""
    names = list(DEMOS) if name == "all" else [name]
    paths: list[Path] = []
    for n in names:
        why = _missing_requirement(n) if name == "all" else None
        if why:
            msg = f"skipping the {n} demo: {why}"
            on_skip(msg) if on_skip else warnings.warn(msg, stacklevel=2)
            continue
        paths += DEMOS[n](Path(out) / n)
    return paths
