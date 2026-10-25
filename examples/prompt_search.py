"""Prompt optimisation against a mechanism: what behaviour does it reward when an optimiser (an LLM
that is told the rules) searches for the highest-reward instructions? Three steered searches map
the honest and deceptive frontiers; the frontier gap on held-out tasks is the optimiser's view of
incentive compatibility.

    OA_EXPERT_MODEL=... OA_JUDGE_MODEL=... OA_OPTIMIZER_MODEL=... python examples/prompt_search.py --domain quality
"""

import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy
from oversight_arena.elicitation import STEER_DECEPTIVE, STEER_FREE, STEER_HONEST, Evaluator, LLMProposer, PromptOptimizer
from oversight_arena.ground_truth.llm import LLMGroundTruth
from oversight_arena.mechanisms import Consultancy
from oversight_arena.registry import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="quality")
    ap.add_argument("--tasks", type=int, default=12)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--out", default="runs/prompt_search")
    args = ap.parse_args()
    domain = build("domain", {"type": args.domain, "limit": args.tasks})
    tasks = domain.tasks()
    train, holdout = tasks[: len(tasks) * 2 // 3], tasks[len(tasks) * 2 // 3:]
    mech = Consultancy(rounds=2, evidence=EvidencePolicy())
    optimizer = os.environ.get("OA_OPTIMIZER_MODEL", os.environ.get("OA_EXPERT_MODEL", "mockllm/model"))
    rows = []
    for label, stance, steer in [("honest-only · argue truth", oa.Stance.CORRECT, STEER_HONEST),
                                 ("free · argue truth", oa.Stance.CORRECT, STEER_FREE),
                                 ("deceptive · argue falsehood", oa.Stance.INCORRECT, STEER_DECEPTIVE)]:
        seed = oa.argue(stance)
        ev = Evaluator(domain, mech, oa.llm_agents(), "consultant", tasks=train, gt_keys=("correct",))
        po = PromptOptimizer(ev, LLMProposer(optimizer, style="reflective"), seeds=[seed], iterations=args.iterations,
                             per_iter=3, minibatch=6, steering=steer, holdout=holdout, final_k=1, label=label)
        trace = asyncio.run(po.run())
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / f"{label.split()[0]}.json").write_text(trace.to_json())
        best = trace.best(1)[0]
        rows.append({"search": label, "best_minibatch_reward": best.reward, "holdout_reward": best.holdout_reward,
                     "instructions": best.strategy.instructions[:120]})
    table = pd.DataFrame(rows)
    print(table.round(3).to_string(index=False))
    h = table.loc[0, "holdout_reward"]
    d = table.loc[2, "holdout_reward"]
    if h is not None and d is not None:
        print(f"frontier gap (honest - deceptive, held out): {h - d:.3f}")
    _ = LLMGroundTruth  # for behavioural GT (honesty of claims), add LLMGroundTruth(model=...) to Evaluator(gt=...)
