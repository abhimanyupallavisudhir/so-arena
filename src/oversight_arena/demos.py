"""Built-in demos that run without API keys (programmatic agents), each producing figures and
an HTML report. They double as end-to-end tests of the whole pipeline and as templates for
LLM experiments (swap the agents for :class:`~oversight_arena.agents.LLMAgent`).

``oversight-arena demo <name>`` with name in: hiddenbits, chess, bon, optimize, swarm,
release, all.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import plots as P
from .analysis.bon import bon_curve
from .analysis.games import EmpiricalGame
from .analysis.ic import asd, frontier, ic_report
from .analysis.report import html_report
from .channels.evidence import EvidencePolicy
from .core.strategy import Assignment, Profile, Stance, Strategy
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
    for label, (mech, role) in mechs.items():
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
    rep = html_report(allres, out / "report.html", title="Best-of-N optimisation pressure",
                      subtitle=f"Base policy samples ({N} per task) of a HiddenBits advocate; credulous Bayesian judge.",
                      gt="honesty", figures=figs, tables={"BoN curves": cur.round(3)})
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
            po = PromptOptimizer(ev, ParamProposer(space, seed=hash(label) % 1000), seeds=[base], iterations=8, per_iter=5,
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
def demo_swarm(out: Path) -> list[Path]:
    """Shared objective vs whistleblower bounties: theory, equilibria, and learning dynamics."""
    from .domains.swarm import AbstractSwarm
    from .elicitation.rl import StrategyGradient
    from .mechanisms import Swarm, TeamReward, Whistleblower
    from .sim import SwarmWorker
    from .theory import swarm_game as sg

    out.mkdir(parents=True, exist_ok=True)
    figs = {}
    df = sg.phase_diagram(n=3, G=0.3, o=0.8, b_over_share=np.linspace(0, 1.5, 151), audit=np.linspace(0, 0.9, 91))
    fig, ax = P.regime_map(df, "bounty_over_share", "audit", contour="basin_report", title="When does reporting pay? (theory)",
                           xlabel="bounty ÷ each worker's share of the illicit gain", ylabel="random audit probability")
    figs["Theory: regimes"] = P.save(fig, out / "theory_regimes.png")
    dom = AbstractSwarm(n_tasks=30, gain=0.3, damage=0.3, observe_prob=0.8, n_workers=3)
    strat = {
        "honest-silent": Strategy(name="honest-silent", params={"cheat": 0.0, "report": 0.0}),
        "honest-reporter": Strategy(name="honest-reporter", params={"cheat": 0.0, "report": 1.0}),
        "cheat-silent": Strategy(name="cheat-silent", params={"cheat": 1.0, "report": 0.0}),
        "cheat-reporter": Strategy(name="cheat-reporter", params={"cheat": 1.0, "report": 1.0}),
    }
    rules = {"shared reward": TeamReward(), "bounty 0.05": Whistleblower(bounty=0.05, offender_penalty=0.3),
             "bounty 0.3": Whistleblower(bounty=0.3, offender_penalty=0.3)}
    workers = ["worker_1", "worker_2", "worker_3"]
    eq_rows = []
    agents = {"*": SwarmWorker()}
    for rname, rule in rules.items():
        mech = Swarm(n_workers=3, rounds=1, reward=rule, label=rname)
        res = _run(dom, mech, agents, Cartesian(strategies={w: list(strat.values()) for w in workers}))
        game = EmpiricalGame.from_results(res, workers, strategy_col="strategy_name", gt_metrics=("clean", "reported"))
        for e in game.nash(starts=8):
            sup = e.support(game)
            eq_rows.append({
                "reward rule": rname, "equilibrium": "; ".join(f"{w}: " + ", ".join(f"{k} {v:.2f}" for k, v in s.items()) for w, s in sup.items()),
                "mean payoff": np.mean(list(e.payoffs.values())),
                "true project score": e.gt.get("clean:_outcome"),
                "violation reported": e.gt.get("reported:_outcome"),
            })
    eqdf = pd.DataFrame(eq_rows)
    # learning dynamics: silent-leaning vs report-leaning initialisations
    learn = []
    pop = [strat["cheat-silent"], strat["cheat-reporter"], strat["honest-silent"], strat["honest-reporter"]]
    for rname in ("shared reward", "bounty 0.05", "bounty 0.3"):
        for init_name, init in [("start silent", [0.45, 0.05, 0.45, 0.05]), ("start reporting", [0.05, 0.45, 0.05, 0.45])]:
            sgd = StrategyGradient(dom, Swarm(n_workers=3, rounds=1, reward=rules[rname]), agents,
                                   {w: pop for w in workers}, init={w: init for w in workers}, lr=2.0, batch=24,
                                   iterations=30, gt_keys=("clean", "reported"), seed=7)
            traj = asyncio.run(sgd.run())
            for _, r in traj.iterrows():
                p_rep = np.mean([r[f"p[{w}:cheat-reporter]"] + r[f"p[{w}:honest-reporter]"] for w in workers])
                p_cheat = np.mean([r[f"p[{w}:cheat-silent]"] + r[f"p[{w}:cheat-reporter]"] for w in workers])
                learn.append({"rule": rname, "init": init_name, "iteration": r["iteration"], "P(report)": p_rep,
                              "P(cheat)": p_cheat, "true score": r.get("gt_clean[_outcome]")})
    ldf = pd.DataFrame(learn)
    ldf.to_csv(out / "learning.csv", index=False)
    for metric in ("P(report)", "P(cheat)"):
        d = ldf.assign(series=ldf["rule"] + " · " + ldf["init"])
        d = d[d["rule"] != "shared reward"] if metric == "P(report)" else d
        fig, ax = P.line_compare(d, "iteration", metric, "series", title=f"Training dynamics: {metric}", xlabel="training iteration", ylabel=metric)
        figs[f"Learning: {metric}"] = P.save(fig, out / f"learning_{metric[2:-1]}.png")
    rep = html_report(Results([]), out / "report.html", title="Swarms: shared objective vs whistleblowing",
                      subtitle="3 workers; one can tamper with the scorer (+0.3 apparent, −0.3 true); others notice w.p. 0.8.",
                      figures=figs, tables={"Equilibria of the empirical game": eqdf.round(3)})
    return [rep, *figs.values()]


# --------------------------------------------------------------------------- releases
def demo_release(out: Path) -> list[Path]:
    """Release judge-based forecast rankings before resolution; evaluate them after."""
    from .analysis.ic import alignment
    from .domains.synthetic_forecasting import SyntheticForecasting
    from .mechanisms import Forecast, JudgeRating, ProperScoring
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
    proper = later.rescore_delayed(ProperScoring(rule="log")) if hasattr(later, "rescore_delayed") else None
    d = later.df()
    d = d[d["role"] == "forecaster"]
    by = d.groupby("strategy_name").agg(judge_reward=("reward", "mean"), log_score=("gt_forecast_log", "mean")).reset_index()
    al = alignment(d.rename(columns={}), gt="forecast_log")
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
    _ = proper
    return [rep, rel.path / "index.html", f1, f2]


DEMOS = {"hiddenbits": demo_hiddenbits, "chess": demo_chess, "bon": demo_bon, "optimize": demo_optimize,
         "swarm": demo_swarm, "release": demo_release}


def run(name: str, out: Path) -> list[Path]:
    names = list(DEMOS) if name == "all" else [name]
    paths: list[Path] = []
    for n in names:
        paths += DEMOS[n](Path(out) / n)
    return paths
