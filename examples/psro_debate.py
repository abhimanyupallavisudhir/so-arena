"""Multi-agent prompt optimisation for debate: PSRO with LLM best-response oracles.

Each debater keeps a population of instructions; each iteration estimates the empirical game
between the populations (episodes on training tasks), solves it (``--meta nash`` = double oracle;
``last`` = iterated best response / level-k; ``uniform`` = fictitious play), and asks each role's
optimiser for a best response to the other's meta-strategy. The optimisers are told the opponent's
current strategies (opponent-aware). The trace reports the meta-equilibrium's ground truth
(does the judge end up right?) at every iteration.

    OA_EXPERT_MODEL=... OA_JUDGE_MODEL=... OA_OPTIMIZER_MODEL=... python examples/psro_debate.py --domain gsm8k
"""

import argparse
import asyncio
import os

import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy
from oversight_arena.elicitation import PSRO, LLMProposer, optimizer_oracle
from oversight_arena.mechanisms import Debate
from oversight_arena.registry import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="gsm8k")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=2)
    ap.add_argument("--meta", default="nash", choices=["nash", "last", "uniform", "replicator"])
    args = ap.parse_args()
    domain = build("domain", {"type": args.domain, "limit": args.tasks})
    opt = LLMProposer(os.environ.get("OA_OPTIMIZER_MODEL", os.environ.get("OA_EXPERT_MODEL", "mockllm/model")),
                      style="reflective", show_opponents=True)
    # debater_a defends the correct answer, debater_b an incorrect one: honest vs dishonest optimisation
    init = {"debater_a": [oa.argue(oa.Stance.CORRECT)], "debater_b": [oa.argue(oa.Stance.INCORRECT)]}
    psro = PSRO(domain, Debate(rounds=2, evidence=EvidencePolicy()), oa.llm_agents(), init,
                {r: optimizer_oracle(opt, iterations=1, per_iter=2, minibatch=4) for r in init},
                iterations=args.iterations, meta_solver=args.meta, gt_keys=("correct", "decision_correct"))
    trace = asyncio.run(psro.run())
    print(trace.df().round(3).to_string(index=False))
