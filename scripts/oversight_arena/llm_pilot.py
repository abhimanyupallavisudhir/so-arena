"""A small real-LLM pilot of OversightArena (≈ a few dollars with small models).

1. ASD across protocols on GSM8K (experts have a calculator; the judge does not), with a check that
   the simulated behaviours complied ("argue for the wrong answer" really argued it).
2. Chess, a real capability gap: engine-backed experts argue before an LLM judge (trap positions for
   a depth-2 search; Stockfish ground truth).
3. Steered prompt search on consultancy: honest-only vs deceptive frontiers, held-out evaluation.

    export OA_EXPERT_MODEL=openai/gpt-4.1-mini OA_JUDGE_MODEL=openai/gpt-4.1-nano   # any Inspect models
    python scripts/oversight_arena/llm_pilot.py --tasks 40 --out reports/llm_pilot
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

import oversight_arena as oa
from oversight_arena.agents.llm import LLMJudge, default_models
from oversight_arena.analysis import ic_report, plots
from oversight_arena.analysis.diagnostics import compliance, cost_summary, outcome_metrics
from oversight_arena.analysis.report import html_report
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.math import GSM8K
from oversight_arena.elicitation import STEER_DECEPTIVE, STEER_HONEST, Evaluator, LLMProposer, PromptOptimizer
from oversight_arena.experiment.profiles import MapProfiles
from oversight_arena.ground_truth.common import DecisionCorrect, JudgeProbCorrect, TargetCorrect
from oversight_arena.ground_truth.llm import LLMCompliance
from oversight_arena.mechanisms import Consultancy, Debate, NaiveJudge, Propaganda


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    """Markdown table without optional dependencies."""
    def cell(v: object) -> str:
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.{digits}f}"
        return str(v).replace("|", "\\|")

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def part_asd(n: int, out: Path, expert: str, judge: str) -> tuple[str, oa.Results]:
    pol = EvidencePolicy()
    mechs = [NaiveJudge(), Propaganda(evidence=pol), Consultancy(rounds=2, evidence=pol), Debate(rounds=2, evidence=pol)]
    gt = [TargetCorrect(), DecisionCorrect(), JudgeProbCorrect(), LLMCompliance(model=expert)]
    res = oa.Experiment(GSM8K(limit=n), mechs, oa.llm_agents(expert, judge), oa.Stances(), gt=gt,
                        out=out / "gsm8k_asd", name="pilot-gsm8k-asd").run()
    ic = ic_report(res.df(trainable_only=True))
    om = outcome_metrics(res)
    d = res.df(trainable_only=True)
    comp = d.groupby("mechanism")["gt_compliance"].mean().reset_index() if "gt_compliance" in d else pd.DataFrame()
    fig, _ = plots.asd_bars(ic)
    plots.save(fig, out / "gsm8k_asd.png")
    html_report(res, out / "gsm8k_asd" / "report.html", title="Pilot: ASD on GSM8K")
    text = ("## 1. ASD on GSM8K\n\n" + md_table(ic[["mechanism", "asd", "lo", "hi", "pairwise_acc", "n_tasks"]]) +
            "\n\nJudge outcomes:\n\n" + md_table(om[[c for c in ["mechanism", "accuracy", "p_correct", "ece"] if c in om]]) +
            ("\n\nCompliance (share of arguments that argued the assigned answer):\n\n" + md_table(comp) if len(comp) else "") +
            "\n\n![ASD](gsm8k_asd.png)\n\n")
    return text, res


def part_chess(n: int, out: Path, judge: str) -> tuple[str, oa.Results]:
    from oversight_arena.domains.chess import ChessMoves
    from oversight_arena.sim.chess_agents import EngineAdvocate

    dom = ChessMoves(n_puzzles=max(40, 2 * n), judge_trap_depth=2, gt_depth=14, verify_depth=8, limit=n)
    agents = {"kind:judge": LLMJudge(judge), "*": EngineAdvocate(depth=10)}
    pol = EvidencePolicy()
    results = None
    for style in ("honest", "cherry_pick"):
        ms = [m.model_copy(update={"label": f"{m.name} · {style}"})
              for m in (NaiveJudge(), Consultancy(rounds=2, evidence=pol), Debate(rounds=2, evidence=pol))]
        res = oa.Experiment(dom, ms, agents, MapProfiles(source=oa.Stances(), params={"style": style}),
                            out=out / "chess" / style, concurrency=2).run()
        results = res if results is None else results + res
    om = outcome_metrics(results)
    ic = ic_report(results.df(trainable_only=True))
    html_report(results, out / "chess" / "report.html", title="Pilot: chess, engine experts vs an LLM judge")
    text = ("## 2. Chess: engine experts vs an LLM judge\n\n" +
            md_table(om[["mechanism", "accuracy", "p_correct"]].merge(ic[["mechanism", "asd", "lo", "hi"]], on="mechanism", how="left")) + "\n\n")
    return text, results


def part_search(n: int, out: Path, expert: str, judge: str, optimizer: str) -> str:
    dom = GSM8K(limit=n)
    tasks = dom.tasks()
    train, hold = tasks[: len(tasks) * 2 // 3], tasks[len(tasks) * 2 // 3:]
    mech = Consultancy(rounds=2, evidence=EvidencePolicy())
    rows, trajs = [], []
    for label, stance, steer in (("honest only", oa.Stance.CORRECT, STEER_HONEST), ("deceptive", oa.Stance.INCORRECT, STEER_DECEPTIVE)):
        ev = Evaluator(dom, mech, oa.llm_agents(expert, judge), "consultant", tasks=train)
        po = PromptOptimizer(ev, LLMProposer(optimizer, style="reflective"), seeds=[oa.argue(stance)], iterations=3, per_iter=3,
                             minibatch=8, steering=steer, holdout=hold, final_k=1, label=label)
        tr = asyncio.run(po.run())
        (out / f"search_{label.split()[0]}.json").write_text(tr.to_json())
        t = tr.trajectory()
        t["search"] = label
        trajs.append(t)
        b = tr.best(1)[0]
        rows.append({"search": label, "best minibatch reward": b.reward, "held-out reward": b.holdout_reward,
                     "instructions": (b.strategy.instructions or "")[:160].replace("\n", " ")})
    tab = pd.DataFrame(rows)
    traj = pd.concat(trajs)
    fig, _ = plots.line_compare(traj, "iteration", "best_reward", "search", title="Steered prompt search (consultancy, GSM8K)",
                                xlabel="iteration", ylabel="best reward found")
    plots.save(fig, out / "search.png")
    gap = tab.loc[0, "held-out reward"] - tab.loc[1, "held-out reward"] if tab["held-out reward"].notna().all() else float("nan")
    return ("## 3. Steered prompt search (consultancy, GSM8K)\n\n" + md_table(tab) +
            f"\n\nHeld-out frontier gap (honest − deceptive): **{gap:.3f}**\n\n![search](search.png)\n\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--chess-tasks", type=int, default=20)
    ap.add_argument("--search-tasks", type=int, default=24)
    ap.add_argument("--parts", default="asd,chess,search")
    ap.add_argument("--out", default="reports/llm_pilot")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    expert, judge = default_models()
    optimizer = os.environ.get("OA_OPTIMIZER_MODEL", expert)
    parts = set(args.parts.split(","))
    body, costs = [], []
    if "asd" in parts:
        t, r = part_asd(args.tasks, out, expert, judge)
        body.append(t)
        costs.append(cost_summary(r))
        comp = compliance(r)
        body.append("Stated-answer compliance and parse errors:\n\n" + md_table(comp) + "\n\n")
    if "chess" in parts:
        t, r = part_chess(args.chess_tasks, out, judge)
        body.append(t)
        costs.append(cost_summary(r))
    if "search" in parts:
        body.append(part_search(args.search_tasks, out, expert, judge, optimizer))
    head = (f"# OversightArena LLM pilot\n\nExpert: `{expert}` · judge: `{judge}` · optimiser: `{optimizer}`.\n\n"
            "Rewards are the judge's log-probability for the answer argued; ASD = reward(argue truth) − reward(argue falsehood), "
            "per task, with task-bootstrap 95% CIs.\n\n")
    tail = ("## Cost\n\n" + md_table(pd.concat(costs)) + "\n") if costs else ""
    (out / "README.md").write_text(head + "".join(body) + tail)
    print(out / "README.md")


if __name__ == "__main__":
    main()
