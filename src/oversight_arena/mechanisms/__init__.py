"""Scalable-oversight mechanisms (protocol + reward rule), all domain-general."""

from .base import EpisodeContext, Mechanism, mechanism
from .critique import AcceptReward, ProposerCritic
from .debate import CrossExamination, Debate, OpenDebate
from .forecasting import Forecast, JudgeAgreement, JudgeRating, MarketScoring, PredictionMarket, ProperScoring
from .judging import Consultancy, NaiveJudge, OpenConsultancy, Propaganda
from .monitoring import MonitoredReward, Monitoring
from .peer_prediction import BTS, DMI, CorrelatedAgreement, OutputAgreement, Reporters
from .preference import Comparison, PreferenceReward, ProverReward, ProverVerifier
from .swarm import Swarm, SwarmAudit, SwarmEnvironment, TeamReward, Whistleblower

__all__ = [
    "EpisodeContext", "Mechanism", "mechanism", "AcceptReward", "ProposerCritic", "CrossExamination", "Debate", "OpenDebate",
    "Comparison", "PreferenceReward", "ProverReward", "ProverVerifier",
    "Forecast", "JudgeAgreement", "JudgeRating", "MarketScoring", "PredictionMarket", "ProperScoring",
    "Consultancy", "NaiveJudge", "OpenConsultancy", "Propaganda", "MonitoredReward", "Monitoring",
    "BTS", "DMI", "CorrelatedAgreement", "OutputAgreement", "Reporters", "Swarm", "SwarmAudit",
    "SwarmEnvironment", "TeamReward", "Whistleblower",
]
