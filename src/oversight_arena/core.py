"""Public protocol data and private evaluation data have separate types and paths."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

JSON = Any


def canonical(value: JSON) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: JSON) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Role:
    id: str
    trainable: bool = True
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or type(self.trainable) is not bool:
            raise ValueError("Role needs a nonempty string id and boolean trainability")


@dataclass(frozen=True)
class PublicTask:
    id: str
    prompt: str
    data: dict[str, JSON] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or not isinstance(self.prompt, str):
            raise ValueError("Public task needs a nonempty string id and string prompt")
        canonical(self.data)


@dataclass(frozen=True)
class Task:
    public: PublicTask
    # These are deliberately omitted from Episode and from mechanism input.
    evaluator_data: dict[str, JSON] = field(default_factory=dict)
    role_data: dict[str, dict[str, JSON]] = field(default_factory=dict)
    split: Literal["train", "validation", "test", "unspecified"] = "unspecified"
    provenance: dict[str, JSON] = field(default_factory=dict)

    def __post_init__(self):
        if self.split not in {"train", "validation", "test", "unspecified"}:
            raise ValueError("Invalid dataset split")
        canonical(self.role_data)
        canonical(self.provenance)


@dataclass(frozen=True)
class Response:
    content: JSON
    channels: dict[str, JSON] = field(default_factory=dict)
    # Provider accounting must be supplied by adapters. Unknown cost is not zero.
    cost: float | None = None
    tokens: int | None = None

    def __post_init__(self):
        canonical(asdict(self))
        if self.cost is not None and finite(self.cost) < 0:
            raise ValueError("Negative cost")
        if self.tokens is not None and (type(self.tokens) is not int or self.tokens < 0):
            raise ValueError("Tokens must be a nonnegative integer")


@dataclass(frozen=True)
class Event:
    id: str
    actor: str
    kind: str
    content: JSON
    recipients: tuple[str, ...] | None = None  # None = protocol-public
    channels: dict[str, JSON] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    task: PublicTask
    role: str
    instruction: str
    history: tuple[Event, ...]
    private: dict[str, JSON]
    seed: int


class Policy(Protocol):
    async def __call__(self, observation: Observation) -> Response: ...


@dataclass(frozen=True)
class Reward:
    """Scalar utility consumed by optimizers; components are explanatory, not summed implicitly."""

    value: float
    components: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self):
        finite(self.value)
        for value in self.components.values():
            finite(value)


@dataclass(frozen=True)
class Evaluation:
    scorer: str
    version: str
    per_agent: dict[str, dict[str, float]] = field(default_factory=dict)
    outcomes: dict[str, float] = field(default_factory=dict)
    status: Literal["resolved", "pending", "unavailable"] = "resolved"
    evidence: dict[str, JSON] = field(default_factory=dict)

    def __post_init__(self):
        if (
            not self.scorer
            or not self.version
            or self.status not in {"resolved", "pending", "unavailable"}
        ):
            raise ValueError("Evaluation needs scorer, version, and a valid status")
        if self.status != "resolved" and (self.per_agent or self.outcomes):
            raise ValueError("Unresolved evaluation cannot contain truth scores")
        for values in [self.outcomes, *self.per_agent.values()]:
            for value in values.values():
                finite(value)
        canonical(self.evidence)


@dataclass(frozen=True)
class Episode:
    id: str
    task: PublicTask
    mechanism: str
    config: dict[str, JSON]
    roles: tuple[Role, ...]
    seed: int
    events: tuple[Event, ...]
    rewards: dict[str, Reward]
    output: JSON
    usage: dict[str, JSON]
    provenance: dict[str, JSON] = field(default_factory=dict)
    evaluations: tuple[Evaluation, ...] = ()
    schema_version: int = 1

    def mechanism_payload(self) -> dict[str, JSON]:
        value = asdict(self)
        value.pop("id")
        value.pop("evaluations")
        return value

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Unsupported episode schema")
        role_ids = {role.id for role in self.roles}
        if len(role_ids) != len(self.roles):
            raise ValueError("Duplicate roles")
        if set(self.rewards) - role_ids:
            raise ValueError("Rewards for unknown roles")
        missing = {role.id for role in self.roles if role.trainable} - self.rewards.keys()
        if missing:
            raise ValueError(f"Missing rewards for trainable roles: {sorted(missing)}")
        for reward in self.rewards.values():
            reward.__post_init__()
        for evaluation in self.evaluations:
            evaluation.__post_init__()
            if set(evaluation.per_agent) - role_ids:
                raise ValueError("Evaluation of unknown role")
        if len({e.id for e in self.events}) != len(self.events):
            raise ValueError("Duplicate event ids")
        for event in self.events:
            if event.actor not in role_ids | {"mechanism"}:
                raise ValueError("Unknown event actor")
            if event.recipients is not None and set(event.recipients) - role_ids:
                raise ValueError("Unknown recipient")
        if digest(self.mechanism_payload()) != self.id:
            raise ValueError("Episode content does not match its id")


class Scorer(Protocol):
    async def __call__(self, task: Task, episode: Episode) -> Evaluation: ...


@dataclass(frozen=True)
class Outcome:
    rewards: dict[str, Reward]
    output: JSON = None


def episode_from_dict(data: dict[str, JSON]) -> Episode:
    data = dict(data)
    data["task"] = PublicTask(**data["task"])
    data["roles"] = tuple(Role(**r) for r in data["roles"])
    data["events"] = tuple(
        Event(
            **{**e, "recipients": tuple(e["recipients"]) if e["recipients"] is not None else None}
        )
        for e in data["events"]
    )
    data["rewards"] = {k: Reward(**v) for k, v in data["rewards"].items()}
    data["evaluations"] = tuple(Evaluation(**e) for e in data["evaluations"])
    episode = Episode(**data)
    episode.validate()
    return episode
