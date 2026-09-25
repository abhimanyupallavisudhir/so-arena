"""Async workflows, explicit information boundaries, resource limits and replayable logs."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

from .types import (
    JSON,
    Action,
    Event,
    Observation,
    Outcome,
    Reward,
    Role,
    Run,
    Task,
    canonical,
    digest,
)


class Policy(Protocol):
    # Optional prepare(Observation) hook adds private policy instructions before logging.
    async def __call__(self, observation: Observation) -> Action: ...


class Mechanism(Protocol):
    async def __call__(self, context: Context) -> Outcome: ...


Tool = Callable[[JSON], Awaitable[JSON]]


@dataclass(frozen=True)
class Budget:
    calls: int = 100
    tool_calls: int = 100
    seconds: float = 300

    def __post_init__(self):
        if self.calls < 0 or self.tool_calls < 0 or not 0 < self.seconds < float("inf"):
            raise ValueError("Invalid budget")


class Context:
    """Only public task data enters a mechanism. Scorers run after this context is closed.

    This is an API boundary, not a sandbox against malicious Python plugins.
    Tool providers must isolate any external state themselves.
    """

    def __init__(
        self,
        task: Task,
        roles: Sequence[Role],
        policies: Mapping[str, Policy],
        seed: int,
        budget: Budget,
        tools: Mapping[str, Tool] | None = None,
    ):
        self.task = copy.deepcopy(task)
        self.roles = {r.id: r for r in roles}
        if not roles or len(self.roles) != len(roles) or any(not r.id for r in roles):
            raise ValueError("Roles must have unique, nonempty IDs")
        if set(policies) != set(self.roles):
            raise ValueError("Exactly one policy is required for every role, including fixtures")
        self._policies = policies
        self.seed, self.budget = seed, budget
        self._tools = tools or {}
        self.events: list[Event] = []
        self.usage = {"calls": 0, "tool_calls": 0}
        self._role_calls = dict.fromkeys(self.roles, 0)

    def emit(
        self, actor: str, kind: str, data: dict, audience: tuple[str, ...] | None = None
    ) -> Event:
        if audience is not None and not set(audience) <= self.roles.keys():
            raise ValueError("Unknown event audience")
        canonical(data)
        event = Event(len(self.events), actor, kind, copy.deepcopy(data), audience)
        self.events.append(event)
        return copy.deepcopy(event)

    def _observe(self, role: str, instruction: str) -> Observation:
        if role not in self.roles:
            raise ValueError(f"Unknown role: {role}")
        if self.usage["calls"] >= self.budget.calls:
            raise RuntimeError("Policy call budget exceeded")
        self.usage["calls"] += 1
        turn = self._role_calls[role]
        self._role_calls[role] += 1
        seed = int(digest([self.seed, self.task.id, self.task.snapshot, role, turn])[:16], 16)
        observation = Observation(
            copy.deepcopy(self.task),
            role,
            instruction,
            tuple(copy.deepcopy(e) for e in self.events if e.visible_to(role)),
            seed,
            self.roles[role].tools,
        )
        prepare = getattr(self._policies[role], "prepare", None)
        return prepare(observation) if prepare else observation

    async def act(
        self, role: str, instruction: str = "", audience: tuple[str, ...] | None = None
    ) -> Action:
        obs = self._observe(role, instruction)
        self.emit(obs.role, "policy_input", asdict(obs), ())
        action = await self._policies[role](copy.deepcopy(obs))
        self._record(obs, action, audience)
        return copy.deepcopy(action)

    def _record(self, obs: Observation, action: Action, audience: tuple[str, ...] | None):
        if not isinstance(action, Action):
            raise TypeError("Policies must return Action")
        # Inputs are experimenter-only; outputs follow mechanism visibility.
        self.emit(obs.role, "action", asdict(action), audience)

    async def simultaneous(
        self, requests: Mapping[str, str], audience: tuple[str, ...] | None = None
    ) -> dict[str, Action]:
        # Every player observes the same pre-round history. Logging order is stable.
        observations = [self._observe(role, text) for role, text in requests.items()]
        for obs in observations:
            self.emit(obs.role, "policy_input", asdict(obs), ())
        # A failed peer cancels the other in-flight calls before the run returns.
        async with asyncio.TaskGroup() as group:
            pending = [
                group.create_task(self._policies[o.role](copy.deepcopy(o))) for o in observations
            ]
        actions = [task.result() for task in pending]
        for obs, action in zip(observations, actions, strict=True):
            self._record(obs, action, audience)
        return {o.role: copy.deepcopy(a) for o, a in zip(observations, actions, strict=True)}

    async def tool(
        self, role: str, name: str, arguments: JSON, audience: tuple[str, ...] | None = None
    ) -> JSON:
        if name not in self.roles[role].tools or name not in self._tools:
            raise PermissionError(f"{role} cannot use {name}")
        if self.usage["tool_calls"] >= self.budget.tool_calls:
            raise RuntimeError("Tool call budget exceeded")
        self.usage["tool_calls"] += 1
        result = await self._tools[name](copy.deepcopy(arguments))
        self.emit(role, "tool", {"name": name, "arguments": arguments, "result": result}, audience)
        return copy.deepcopy(result)


async def run(
    task: Task,
    roles: Sequence[Role],
    policies: Mapping[str, Policy],
    mechanism: Mechanism,
    *,
    name: str,
    seed: int = 0,
    manifest: dict | None = None,
    budget: Budget | None = None,
    tools: Mapping[str, Tool] | None = None,
) -> Run:
    """Execute without an oracle. Failures are recorded and never assigned zero reward.

    Include policy/checkpoint IDs, mechanism config and an experiment ID in manifest;
    these form run identity. Seeds describe requested randomness, not provider determinism.
    """
    budget = budget or Budget()
    manifest = copy.deepcopy(manifest or {})
    manifest["budget"] = asdict(budget)
    identity = {
        "task": asdict(task),
        "roles": [asdict(r) for r in roles],
        "mechanism": name,
        "seed": seed,
        "manifest": manifest,
    }
    run_id = digest(identity)
    ctx = Context(task, roles, policies, seed, budget, tools)
    outcome, error = None, None
    try:
        async with asyncio.timeout(budget.seconds):
            candidate = await mechanism(ctx)
            if not isinstance(candidate, Outcome):
                raise TypeError("Mechanisms must return Outcome")
            if any(not isinstance(reward, Reward) for reward in candidate.rewards.values()):
                raise TypeError("Outcome rewards must be Reward instances")
            unknown = candidate.rewards.keys() - ctx.roles.keys()
            missing = {r.id for r in roles if r.trainable} - candidate.rewards.keys()
            if unknown or missing:
                raise ValueError(
                    f"Reward contract: unknown={sorted(unknown)}, missing={sorted(missing)}"
                )
            canonical(asdict(candidate))
            outcome = copy.deepcopy(candidate)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return Run(
        run_id,
        copy.deepcopy(task),
        name,
        tuple(roles),
        seed,
        tuple(ctx.events),
        outcome,
        "failed" if error else "complete",
        ctx.usage,
        manifest,
        error,
    )
