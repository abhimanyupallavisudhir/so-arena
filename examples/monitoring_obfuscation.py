"""Optimising an actor against a monitor: does the reward reduce misbehaviour, or only alarms?

The actor solves code tasks and is paid (grader's score) - lambda * alarm. Two monitors: an LLM
chain-of-thought monitor (optimisable) and a simulated probe with a fixed AUROC against ground
truth (a stand-in for a robust detector). A steered prompt search looks for high-reward actor
instructions; we track ground-truth misbehaviour (hidden tests) against alarm rates.
See docs/theory.md, Proposition 4.

    OA_EXPERT_MODEL=... OA_JUDGE_MODEL=... python examples/monitoring_obfuscation.py
"""

import argparse
import asyncio
import os

import oversight_arena as oa
from oversight_arena.channels import SimulatedProbe
from oversight_arena.elicitation import Evaluator, LLMProposer, PromptOptimizer, artifact_library
from oversight_arena.ground_truth import StrategyTag
from oversight_arena.mechanisms import MonitoredReward, Monitoring
from oversight_arena.registry import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--penalty", type=float, default=1.0)
    args = ap.parse_args()
    domain = build("domain", {"type": "code_gen", "limit": args.tasks})
    reward = MonitoredReward(alarm_penalty=args.penalty)
    for label, mech in [("CoT monitor", Monitoring(monitor="llm", cot_visible=True, probe=None, reward=reward)),
                        ("fixed-AUROC probe", Monitoring(monitor="probe", probe=SimulatedProbe(auroc=0.9, signal="tag:honest"),
                                                         reward=reward))]:
        agents = oa.llm_agents(scratchpad=True)
        gt = [*domain.gt_scorers(), StrategyTag()]  # hidden tests + the behaviour each strategy was meant to show
        ev = Evaluator(domain, mech, agents, "actor", gt=gt, gt_keys=("code_correct", "intended_honest"))
        seeds = artifact_library(["best_effort", "special_case_tests", "subtle_bug"])
        po = PromptOptimizer(ev, LLMProposer(os.environ.get("OA_OPTIMIZER_MODEL", os.environ.get("OA_EXPERT_MODEL", "mockllm/model"))),
                             seeds=seeds, iterations=2, per_iter=2, minibatch=4, label=label)
        trace = asyncio.run(po.run())
        print(label)
        print(trace.df()[["name", "iteration", "reward", "gt_code_correct", "gt_intended_honest"]].round(3).to_string(index=False))
