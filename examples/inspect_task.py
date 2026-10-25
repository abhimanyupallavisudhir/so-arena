"""OversightArena experiments as Inspect tasks.

    inspect eval examples/inspect_task.py@gsm8k_debate --model openai/gpt-4.1 --model-role trusted=openai/gpt-4.1-nano
    inspect view                                    # browse debates; rewards and GT are scores
    python -c "from oversight_arena.integrations.inspect import results_from_logs; print(len(results_from_logs('logs')))"
"""

from inspect_ai import task

import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.math import GSM8K
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.integrations.inspect import oversight_task
from oversight_arena.mechanisms import Debate
from oversight_arena.sim import BayesianBitJudge, BitAdvocate


@task
def gsm8k_debate(limit: int = 20, rounds: int = 2):
    return oversight_task(GSM8K(limit=limit), Debate(rounds=rounds, evidence=EvidencePolicy()), profiles=oa.Stances())


@task
def hiddenbits_programmatic(n_tasks: int = 20, budget: int = 4):
    """No model calls: exact Bayesian judge and programmatic advocates (any --model works)."""
    return oversight_task(HiddenBits(n_tasks=n_tasks), Debate(rounds=2, evidence=EvidencePolicy(budget=budget)),
                          {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate()}, oa.Stances())
