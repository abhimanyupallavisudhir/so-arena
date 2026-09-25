"""Async workflow execution, visibility, budgets, and oracle isolation."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Protocol

from .core import (
    JSON,
    Episode,
    Event,
    Observation,
    Outcome,
    Policy,
    PublicTask,
    Role,
    Scorer,
    Task,
    canonical,
    digest,
    finite,
)


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class Budget:
    calls: int = 100
    tool_calls: int = 100
    cost: float | None = None
    timeout: float = 300

    def __post_init__(self):
        if type(self.calls) is not int or type(self.tool_calls) is not int:
            raise ValueError("Call budgets must be integers")
        if self.calls < 1 or self.tool_calls < 0 or finite(self.timeout) <= 0:
            raise ValueError("Invalid budget")
        if self.cost is not None and finite(self.cost) < 0:
            raise ValueError("Invalid cost budget")


@dataclass(frozen=True)
class Tool:
    name: str
    call: Callable[[JSON], Awaitable[JSON]]
    allowed_roles: frozenset[str]
    version: str = "1"
    cost: float = 0.0
    description: str = ""
    parameters: dict[str, JSON] = field(default_factory=dict)

    def __post_init__(self):
        if finite(self.cost) < 0:
            raise ValueError("Negative tool cost")


@dataclass
class Context:
    task: PublicTask
    roles: tuple[Role, ...]
    policies: Mapping[str, Policy]
    seed: int
    private: dict[str, dict[str, JSON]] = field(default_factory=dict)
    tools: Mapping[str, Tool] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    # Per-recipient allowlist of optional channels, e.g. reasoning or activations.
    channel_access: Mapping[str, frozenset[str]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    calls: int = 0
    tool_calls: int = 0
    known_cost: float = 0.0
    unknown_cost_calls: int = 0
    tokens: int = 0
    unknown_token_calls: int = 0
    _role_calls: dict[str, int] = field(default_factory=dict)

    def emit(
        self,
        actor: str,
        kind: str,
        content: JSON,
        *,
        recipients: Sequence[str] | None = None,
        channels: dict[str, JSON] | None = None,
    ) -> Event:
        ids = {r.id for r in self.roles}
        if actor not in ids | {"mechanism"} or (recipients is not None and set(recipients) - ids):
            raise ValueError("Unknown actor or recipient")
        canonical(content)
        event = Event(
            f"e{len(self.events)}",
            actor,
            kind,
            copy.deepcopy(content),
            tuple(recipients) if recipients is not None else None,
            copy.deepcopy(channels or {}),
        )
        self.events.append(event)
        return copy.deepcopy(event)

    def visible_history(
        self, role: str, history: Sequence[Event] | None = None
    ) -> tuple[Event, ...]:
        allowed = self.channel_access.get(role, frozenset())
        return tuple(
            copy.deepcopy(
                replace(e, channels={k: v for k, v in e.channels.items() if k in allowed})
            )
            for e in (self.events if history is None else history)
            if e.recipients is None or role in e.recipients
        )

    async def ask(
        self,
        role: str,
        instruction: str = "",
        *,
        recipients: Sequence[str] | None = None,
        history: Sequence[Event] | None = None,
    ) -> Event:
        if role not in self.policies:
            raise ValueError(f"No policy for {role}")
        if recipients is not None and set(recipients) - set(self.policies):
            raise ValueError("Unknown recipient")
        if self.calls >= self.budget.calls:
            raise BudgetExceeded("Model call budget exhausted")
        self.calls += 1  # Reservation occurs before await, including concurrent calls.
        index = self._role_calls.get(role, 0)
        self._role_calls[role] = index + 1
        call_seed = int(digest([self.seed, role, index])[:8], 16)
        observation = Observation(
            copy.deepcopy(self.task),
            role,
            instruction,
            self.visible_history(role, history),
            copy.deepcopy(self.private.get(role, {})),
            call_seed,
        )
        response = await self.policies[role](observation)
        response.__post_init__()
        if response.cost is None:
            self.unknown_cost_calls += 1
            if self.budget.cost is not None:
                raise BudgetExceeded("Cannot enforce cost budget with unpriced responses")
        else:
            self.known_cost += response.cost
        if response.tokens is None:
            self.unknown_token_calls += 1
        else:
            self.tokens += response.tokens
        self._check_cost()
        return self.emit(
            role, "message", response.content, recipients=recipients, channels=response.channels
        )

    async def interact(
        self,
        role: str,
        instruction: str,
        *,
        max_steps: int = 10,
        recipients: Sequence[str] | None = None,
    ) -> Event:
        """Bounded JSON tool loop for arbitrary model policies and domain tools.

        Request {"tool": name, "arguments": ...}, or finish with {"final": ...}.
        The mechanism controls recipients; policies cannot widen visibility.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        available = {
            name: {"description": tool.description, "parameters": tool.parameters}
            for name, tool in self.tools.items()
            if role in tool.allowed_roles
        }
        contract = (
            '\nReturn {"tool": name, "arguments": ...} or {"final": ...}. '
            "Available tools: " + canonical(available)
        )
        for _ in range(max_steps):
            message = await self.ask(role, instruction + contract, recipients=recipients)
            content = message.content
            if not isinstance(content, dict):
                raise ValueError("Tool policy must return a JSON object")
            if "final" in content and "tool" not in content:
                return self.emit(role, "message", content["final"], recipients=recipients)
            if "tool" not in content or "final" in content:
                raise ValueError("Expected exactly one of tool/final")
            await self.use_tool(
                role, content["tool"], content.get("arguments", {}), recipients=recipients
            )
        raise BudgetExceeded("Agent tool loop exhausted before final answer")

    def _check_cost(self) -> None:
        if self.budget.cost is not None and self.known_cost > self.budget.cost:
            raise BudgetExceeded("Cost budget exceeded after returned call")

    async def use_tool(
        self, actor: str, name: str, arguments: JSON, *, recipients: Sequence[str] | None = None
    ) -> Event:
        if actor not in self.policies:
            raise ValueError("Unknown tool actor")
        if recipients is not None and set(recipients) - set(self.policies):
            raise ValueError("Unknown recipient")
        tool = self.tools[name]
        if actor not in tool.allowed_roles:
            raise PermissionError(f"{actor} cannot use {name}")
        if self.tool_calls >= self.budget.tool_calls:
            raise BudgetExceeded("Tool budget exhausted")
        self.tool_calls += 1
        self.known_cost += tool.cost
        self._check_cost()
        result = await tool.call(copy.deepcopy(arguments))
        return self.emit(
            actor,
            "tool",
            {"name": name, "version": tool.version, "arguments": arguments, "result": result},
            recipients=recipients,
        )


class Mechanism(Protocol):
    name: str
    config: dict[str, JSON]

    async def run(self, context: Context) -> Outcome: ...


@dataclass(frozen=True)
class Workflow:
    """Any async program can be a mechanism; no prescribed judge or turn structure."""

    name: str
    execute: Callable[[Context], Awaitable[Outcome]]
    config: dict[str, JSON] = field(default_factory=dict)

    async def run(self, context: Context) -> Outcome:
        return await self.execute(context)


async def evaluate(task: Task, episode: Episode, scorer: Scorer) -> Episode:
    episode.validate()
    if task.public != episode.task:
        raise ValueError("Evaluation task does not match episode")
    result = await scorer(copy.deepcopy(task), copy.deepcopy(episode))
    updated = replace(episode, evaluations=(*episode.evaluations, result))
    updated.validate()
    return updated


async def run_episode(
    task: Task,
    mechanism: Mechanism,
    roles: Sequence[Role],
    policies: Mapping[str, Policy],
    *,
    seed: int = 0,
    scorer: Scorer | None = None,
    budget: Budget | None = None,
    tools: Mapping[str, Tool] | None = None,
    channel_access: Mapping[str, frozenset[str]] | None = None,
    provenance: dict[str, JSON] | None = None,
) -> Episode:
    ids = {r.id for r in roles}
    if len(ids) != len(roles) or ids != set(policies) or not ids:
        raise ValueError("Roles must be unique and have exactly one policy each")
    if set(task.role_data) - ids or set(channel_access or {}) - ids:
        raise ValueError("Private data or channel access for unknown role")
    for name, tool in (tools or {}).items():
        if tool.name != name or set(tool.allowed_roles) - ids:
            raise ValueError("Tool names and permitted roles must match the run")
    budget = budget or Budget()
    context = Context(
        copy.deepcopy(task.public),
        tuple(roles),
        dict(policies),
        seed,
        copy.deepcopy(task.role_data),
        tools or {},
        budget,
        channel_access or {},
    )
    async with asyncio.timeout(budget.timeout):
        outcome = await mechanism.run(context)
    usage = {
        "calls": context.calls,
        "tool_calls": context.tool_calls,
        "known_cost": context.known_cost,
        "unknown_cost_calls": context.unknown_cost_calls,
        "tokens": context.tokens,
        "unknown_token_calls": context.unknown_token_calls,
    }
    manifest = {
        **(provenance or {}),
        "task": task.provenance,
        "split": task.split,
        "budget": asdict(budget),
        "channel_access": {k: sorted(v) for k, v in (channel_access or {}).items()},
        "tools": {
            k: {"version": v.version, "allowed_roles": sorted(v.allowed_roles)}
            for k, v in (tools or {}).items()
        },
    }
    episode = Episode(
        "",
        copy.deepcopy(task.public),
        mechanism.name,
        copy.deepcopy(mechanism.config),
        tuple(roles),
        seed,
        tuple(context.events),
        copy.deepcopy(outcome.rewards),
        copy.deepcopy(outcome.output),
        usage,
        copy.deepcopy(manifest),
    )
    episode = replace(episode, id=digest(episode.mechanism_payload()))
    episode.validate()
    return await evaluate(task, episode, scorer) if scorer else episode
