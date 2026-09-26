"""Behaviour elicitation: strategy libraries, evaluators, prompt optimisation, multi-agent
optimisation (PSRO / level-k) and RL."""

from .evaluator import Evaluation, Evaluator
from .multiagent import PSRO, PSROTrace, optimizer_oracle, pool_oracle
from .optimize import (
    AgenticOptimizer, Candidate, LLMProposer, OptimizationTrace, ParamProposer, PromptOptimizer, Proposer,
)
from .rl import MechanismEnv, StrategyGradient, preference_pairs, reward_function
from .strategies import (
    ARCHETYPES, ARTIFACT_ARCHETYPES, STEER_DECEPTIVE, STEER_FREE, STEER_HONEST, archetype, artifact_archetype,
    artifact_library, library,
)

__all__ = [
    "Evaluation", "Evaluator", "PSRO", "PSROTrace", "optimizer_oracle", "pool_oracle", "AgenticOptimizer",
    "Candidate", "LLMProposer", "OptimizationTrace", "ParamProposer", "PromptOptimizer", "Proposer",
    "MechanismEnv", "StrategyGradient", "preference_pairs", "reward_function", "ARCHETYPES", "ARTIFACT_ARCHETYPES",
    "STEER_DECEPTIVE", "STEER_FREE", "STEER_HONEST", "archetype", "artifact_archetype", "artifact_library", "library",
]
