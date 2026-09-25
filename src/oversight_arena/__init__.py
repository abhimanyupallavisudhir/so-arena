"""OversightArena — principled experiments on scalable-oversight mechanisms.

Core idea: a *mechanism* is a protocol that also assigns rewards to (trainable) agents. We
evaluate mechanisms by how well those rewards track *ground truth* about the agents'
behaviour — incentive compatibility — under increasing optimisation pressure, and in
multi-agent equilibrium.
"""

from .agents import Action, Agent, ConstantAgent, HumanAgent, LLMAgent, LLMJudge, Observation, ResponseSpec, ScriptedAgent
from .core.episode import EpisodeRecord
from .core.rewards import JudgeProbability, RewardRule, score_prob
from .core.roles import RoleSpec
from .core.strategy import DISHONEST, HONEST, Assignment, Profile, Stance, Strategy, argue
from .core.task import Answer, InfoBlock, Task, TaskView
from .core.tools import Tool, tool
from .experiment import (
    AgentTable, Cartesian, Experiment, Fixed, Results, Seeds, Stances, run_episode, sweep,
)
from .mechanisms import EpisodeContext, Mechanism, mechanism
from .models import get_model

__version__ = "0.1.0"

__all__ = [
    "Action", "Agent", "ConstantAgent", "HumanAgent", "LLMAgent", "LLMJudge", "Observation", "ResponseSpec",
    "ScriptedAgent", "EpisodeRecord", "JudgeProbability", "RewardRule", "score_prob", "RoleSpec",
    "DISHONEST", "HONEST", "Assignment", "Profile", "Stance", "Strategy", "argue", "Answer", "InfoBlock",
    "Task", "TaskView", "Tool", "tool", "AgentTable", "Cartesian", "Experiment", "Fixed", "Results", "Seeds",
    "Stances", "run_episode", "sweep", "EpisodeContext", "Mechanism", "mechanism", "get_model",
]
