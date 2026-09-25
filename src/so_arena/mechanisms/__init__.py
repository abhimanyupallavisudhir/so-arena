"""Built-in oversight mechanisms (game form + reward rule)."""

from so_arena.core.mechanism import Mechanism
from so_arena.mechanisms._common import NullPolicy
from so_arena.mechanisms.elicitation import Confession, PeerPrediction
from so_arena.mechanisms.markets import MarketMaking, MarketScoringReward, PredictionMarket
from so_arena.mechanisms.qa import Consultancy, Debate, DirectJudge, Propaganda
from so_arena.mechanisms.swarm import Team
from so_arena.mechanisms.work import MonitoredWork, ReviewedWork, Work

MECHANISMS: dict[str, type[Mechanism]] = {
    cls.name: cls
    for cls in (DirectJudge, Propaganda, Consultancy, Debate, ReviewedWork, MonitoredWork, Work, Team,
                MarketMaking, PredictionMarket, PeerPrediction, Confession)
}
MECHANISMS["naive_judge"] = DirectJudge
MECHANISMS["critique"] = ReviewedWork


def register_mechanism(cls: type[Mechanism], name: str | None = None) -> type[Mechanism]:
    MECHANISMS[name or cls.name] = cls
    return cls


def get_mechanism(mechanism: str, /, **config) -> Mechanism:
    """Instantiate a registered mechanism; ``config`` may include ``name=`` to label this variant."""
    if mechanism not in MECHANISMS:
        raise KeyError(f"unknown mechanism {mechanism!r}; available: {sorted(MECHANISMS)}")
    return MECHANISMS[mechanism](**config)


__all__ = [
    "DirectJudge", "Propaganda", "Consultancy", "Debate", "ReviewedWork", "MonitoredWork", "Work", "Team",
    "MarketMaking", "PredictionMarket", "MarketScoringReward", "PeerPrediction", "Confession", "NullPolicy",
    "MECHANISMS", "get_mechanism", "register_mechanism",
]
