"""Serializable contracts. Evaluation labels never live in a mechanism's Task."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

JSON = Any


def canonical(value: JSON) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: JSON) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Scores must be finite numbers, not booleans")
    return float(value)


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    data: dict[str, JSON] = field(default_factory=dict)
    split: Literal["train", "validation", "test", "unresolved"] = "test"
    snapshot: str = "stateless"

    def __post_init__(self):
        if not self.id or self.split not in {"train", "validation", "test", "unresolved"}:
            raise ValueError("Task needs an id and a valid split")
        canonical(asdict(self))


@dataclass(frozen=True)
class Role:
    id: str
    trainable: bool = True
    description: str = ""
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class Claim:
    statement: str
    artifact: JSON
    verifier: str
    scope: str = "artifact validity"

    @property
    def id(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class Action:
    text: str = ""
    data: dict[str, JSON] = field(default_factory=dict)
    claims: tuple[Claim, ...] = ()

    def __post_init__(self):
        canonical(asdict(self))


@dataclass(frozen=True)
class Event:
    index: int
    actor: str
    kind: str
    data: dict[str, JSON]
    # None = public; () = experimenter-only; otherwise explicit role IDs.
    audience: tuple[str, ...] | None = None

    def visible_to(self, role: str) -> bool:
        return self.audience is None or role in self.audience


@dataclass(frozen=True)
class Observation:
    task: Task
    role: str
    instruction: str
    events: tuple[Event, ...]
    seed: int
    tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class Reward:
    value: float
    components: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        finite(self.value)
        for value in self.components.values():
            finite(value)


@dataclass(frozen=True)
class Outcome:
    rewards: dict[str, Reward]
    output: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class Measurement:
    value: float | None
    status: Literal["observed", "pending", "unavailable", "error"]
    source: str
    kind: Literal["exact", "proxy", "human", "outcome"] = "exact"
    detail: str = ""

    def __post_init__(self):
        if self.status not in {"observed", "pending", "unavailable", "error"}:
            raise ValueError("Unknown measurement status")
        if self.kind not in {"exact", "proxy", "human", "outcome"} or not self.source:
            raise ValueError("Measurements require an explicit source and kind")
        if self.status == "observed":
            finite(self.value)
        elif self.value is not None:
            raise ValueError("Unobserved measurements must have value=None")


@dataclass(frozen=True)
class Evaluation:
    run_id: str
    scorer: str
    version: str
    # role -> dimension -> measurement. Includes behavior and task quality separately.
    scores: dict[str, dict[str, Measurement]]


@dataclass(frozen=True)
class Run:
    id: str
    task: Task
    mechanism: str
    roles: tuple[Role, ...]
    seed: int
    events: tuple[Event, ...]
    outcome: Outcome | None
    status: Literal["complete", "failed"]
    usage: dict[str, float]
    manifest: dict[str, JSON]
    error: str | None = None
    schema_version: int = 1

    def reward(self, role: str) -> float:
        if self.status != "complete" or self.outcome is None:
            raise ValueError("Failed runs have no training reward")
        return self.outcome.rewards[role].value


def action_from_dict(data: dict) -> Action:
    return Action(
        data.get("text", ""),
        data.get("data", {}),
        tuple(Claim(**c) for c in data.get("claims", [])),
    )


def run_from_dict(data: dict) -> Run:
    if data.get("schema_version") != 1:
        raise ValueError("Unsupported run schema")
    data = dict(data)
    data["task"] = Task(**data["task"])
    data["roles"] = tuple(Role(**{**r, "tools": tuple(r["tools"])}) for r in data["roles"])
    data["events"] = tuple(
        Event(**{**e, "audience": None if e["audience"] is None else tuple(e["audience"])})
        for e in data["events"]
    )
    if data["outcome"] is not None:
        data["outcome"] = Outcome(
            {k: Reward(**v) for k, v in data["outcome"]["rewards"].items()},
            data["outcome"]["output"],
        )
    return Run(**data)


def evaluation_from_dict(data: dict) -> Evaluation:
    return Evaluation(
        data["run_id"],
        data["scorer"],
        data["version"],
        {
            role: {name: Measurement(**m) for name, m in scores.items()}
            for role, scores in data["scores"].items()
        },
    )
