"""Simulated (programmatic) agents for cheap, exact, reproducible experiments."""

from .bits_agents import BayesianBitJudge, BitAdvocate
from .swarm_agents import SwarmWorker


def __getattr__(name):  # chess agents need python-chess: import lazily
    if name in ("EngineAdvocate", "EngineJudge"):
        from . import chess_agents

        return getattr(chess_agents, name)
    raise AttributeError(name)

__all__ = ["BayesianBitJudge", "BitAdvocate", "SwarmWorker", "EngineAdvocate", "EngineJudge"]
