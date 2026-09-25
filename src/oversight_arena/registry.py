"""Component registry: build domains, mechanisms, reward rules, agents, GT scorers, profile
sources and channels from names + kwargs (used by YAML configs and the CLI).

Specs are a name (``"debate"``) or a dict with ``type`` (``{"type": "debate", "rounds": 3}``);
nested dicts with a ``type`` key (e.g. a mechanism's ``reward``) are built recursively.
Register your own components with :func:`register`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

_P = "oversight_arena"
REGISTRY: dict[str, dict[str, str | Callable[..., Any]]] = {
    "domain": {
        "hidden_bits": f"{_P}.domains.synthetic:HiddenBits",
        "gsm8k": f"{_P}.domains.math:GSM8K",
        "quality": f"{_P}.domains.quality:QuALITY",
        "mmlu_pro": f"{_P}.domains.mcq:MMLUPro",
        "gpqa": f"{_P}.domains.mcq:GPQA",
        "hf_mcq": f"{_P}.domains.mcq:HFMultipleChoice",
        "code": f"{_P}.domains.code:HiddenTestsCode",
        "code_gen": f"{_P}.domains.code:CodeProposal",
        "sql": f"{_P}.domains.sql:PrivateSQL",
        "chess": f"{_P}.domains.chess:ChessMoves",
        "forecasting": f"{_P}.domains.forecasting:ManifoldForecasting",
        "forecasting_file": f"{_P}.domains.forecasting:FileForecasting",
        "minif2f": f"{_P}.domains.lean:MiniF2F",
        "abstract_swarm": f"{_P}.domains.swarm:AbstractSwarm",
        "simops": f"{_P}.domains.swarm:SimOps",
        "tasklist": f"{_P}.domains.base:TaskListDomain",
    },
    "mechanism": {
        "naive_judge": f"{_P}.mechanisms.judging:NaiveJudge",
        "propaganda": f"{_P}.mechanisms.judging:Propaganda",
        "consultancy": f"{_P}.mechanisms.judging:Consultancy",
        "open_consultancy": f"{_P}.mechanisms.judging:OpenConsultancy",
        "debate": f"{_P}.mechanisms.debate:Debate",
        "cross_examination": f"{_P}.mechanisms.debate:CrossExamination",
        "proposer_critic": f"{_P}.mechanisms.critique:ProposerCritic",
        "monitoring": f"{_P}.mechanisms.monitoring:Monitoring",
        "reporters": f"{_P}.mechanisms.peer_prediction:Reporters",
        "forecast": f"{_P}.mechanisms.forecasting:Forecast",
        "prediction_market": f"{_P}.mechanisms.forecasting:PredictionMarket",
        "swarm": f"{_P}.mechanisms.swarm:Swarm",
    },
    "reward": {
        "judge_probability": f"{_P}.core.rewards:JudgeProbability",
        "judge_label_score": f"{_P}.core.rewards:JudgeLabelScore",
        "combined": f"{_P}.core.rewards:Combined",
        "outcome_field": f"{_P}.core.rewards:OutcomeField",
        "none": f"{_P}.core.rewards:NoReward",
        "accept": f"{_P}.mechanisms.critique:AcceptReward",
        "monitored": f"{_P}.mechanisms.monitoring:MonitoredReward",
        "output_agreement": f"{_P}.mechanisms.peer_prediction:OutputAgreement",
        "bts": f"{_P}.mechanisms.peer_prediction:BTS",
        "correlated_agreement": f"{_P}.mechanisms.peer_prediction:CorrelatedAgreement",
        "dmi": f"{_P}.mechanisms.peer_prediction:DMI",
        "proper_scoring": f"{_P}.mechanisms.forecasting:ProperScoring",
        "market_scoring": f"{_P}.mechanisms.forecasting:MarketScoring",
        "judge_agreement": f"{_P}.mechanisms.forecasting:JudgeAgreement",
        "judge_rating": f"{_P}.mechanisms.forecasting:JudgeRating",
        "team": f"{_P}.mechanisms.swarm:TeamReward",
        "whistleblower": f"{_P}.mechanisms.swarm:Whistleblower",
    },
    "channel": {
        "auditor": f"{_P}.channels.gt_channels:Auditor",
        "probe": f"{_P}.channels.gt_channels:SimulatedProbe",
        "label": f"{_P}.channels.gt_channels:Label",
        "resolution": f"{_P}.channels.gt_channels:Resolution",
        "swarm_audit": f"{_P}.mechanisms.swarm:SwarmAudit",
        "evidence": f"{_P}.channels.evidence:EvidencePolicy",
    },
    "gt": {
        "correct": f"{_P}.ground_truth.common:TargetCorrect",
        "value": f"{_P}.ground_truth.common:TargetValue",
        "decision_correct": f"{_P}.ground_truth.common:DecisionCorrect",
        "judge_p_correct": f"{_P}.ground_truth.common:JudgeProbCorrect",
        "intended_honest": f"{_P}.ground_truth.common:StrategyTag",
        "claim_accuracy": f"{_P}.ground_truth.common:ClaimAccuracy",
        "accept_correct": f"{_P}.ground_truth.common:AcceptCorrect",
        "env": f"{_P}.ground_truth.common:EnvGT",
        "llm_judge_gt": f"{_P}.ground_truth.llm:LLMGroundTruth",
        "llm_compliance": f"{_P}.ground_truth.llm:LLMCompliance",
    },
    "profiles": {
        "stances": f"{_P}.experiment.profiles:Stances",
        "cartesian": f"{_P}.experiment.profiles:Cartesian",
        "seeds": f"{_P}.experiment.profiles:Seeds",
        "fixed": f"{_P}.experiment.profiles:Fixed",
        "map": f"{_P}.experiment.profiles:MapProfiles",
    },
    "agent": {
        "llm": f"{_P}.agents.llm:LLMAgent",
        "llm_judge": f"{_P}.agents.llm:LLMJudge",
        "human": f"{_P}.agents.scripted:HumanAgent",
        "constant": f"{_P}.agents.scripted:ConstantAgent",
        "bit_advocate": f"{_P}.sim.bits_agents:BitAdvocate",
        "bayes_bit_judge": f"{_P}.sim.bits_agents:BayesianBitJudge",
        "engine_advocate": f"{_P}.sim.chess_agents:EngineAdvocate",
        "engine_judge": f"{_P}.sim.chess_agents:EngineJudge",
        "swarm_worker": f"{_P}.sim.swarm_agents:SwarmWorker",
    },
}


def register(kind: str, name: str | None = None) -> Callable[[Any], Any]:
    """Decorator: ``@register("mechanism", "my_protocol") class MyProtocol(Mechanism): ...``"""

    def deco(obj: Any) -> Any:
        REGISTRY.setdefault(kind, {})[name or getattr(obj, "name", None) or obj.__name__.lower()] = obj
        return obj

    return deco


def resolve(kind: str, name: str) -> Any:
    table = REGISTRY.get(kind, {})
    if name not in table:
        if ":" in name:  # "package.module:Class"
            mod, attr = name.split(":", 1)
            return getattr(importlib.import_module(mod), attr)
        raise KeyError(f"unknown {kind} {name!r}; known: {sorted(table)}")
    ref = table[name]
    if isinstance(ref, str):
        mod, attr = ref.split(":")
        ref = getattr(importlib.import_module(mod), attr)
        table[name] = ref
    return ref


FIELD_KINDS = {"reward": "reward", "rules": "reward", "audit": "channel", "probe": "channel",
               "judge_labels": "channel", "evidence": "channel", "source": "profiles", "sources": "profiles"}


def _build_nested(key: str, v: Any) -> Any:
    kind = FIELD_KINDS.get(key)
    if isinstance(v, dict) and "type" in v and kind:
        return build(kind, v)
    if isinstance(v, dict) and kind == "channel" and key == "evidence":
        return build("channel", {"type": "evidence", **v})
    if isinstance(v, list) and kind:
        return [build(kind, x) if isinstance(x, (dict, str)) else x for x in v]
    return v


def build(kind: str, spec: Any) -> Any:
    if not isinstance(spec, (str, dict)):
        return spec
    if isinstance(spec, str):
        spec = {"type": spec}
    spec = dict(spec)
    cls = resolve(kind, spec.pop("type"))
    kwargs = {k: _build_nested(k, v) for k, v in spec.items()}
    return cls(**kwargs)


def list_components(kind: str | None = None) -> dict[str, list[str]]:
    kinds = [kind] if kind else sorted(REGISTRY)
    return {k: sorted(REGISTRY.get(k, {})) for k in kinds}
