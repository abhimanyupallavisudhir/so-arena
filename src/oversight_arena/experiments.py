"""Paired interventions, fresh policy instances and explicit environment factories."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict, dataclass, field

from .runtime import Budget, Mechanism, Policy, Tool, run
from .types import Action, Observation, Role, Run, Task


@dataclass(frozen=True)
class Condition:
    id: str
    prompts: dict[str, str] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class PromptPolicy:
    """A runtime policy whose private prompt is included in recorded model inputs."""

    base: Policy
    prompt: str

    def prepare(self, observation: Observation) -> Observation:
        from dataclasses import replace

        observation = replace(
            observation, instruction=self.prompt + "\n\n" + observation.instruction
        )
        prepare = getattr(self.base, "prepare", None)
        return prepare(observation) if prepare else observation

    async def __call__(self, observation: Observation) -> Action:
        return await self.base(observation)


@asynccontextmanager
async def stateless_environment(task: Task, seed: int):
    if task.snapshot != "stateless":
        raise ValueError("Stateful tasks require an environment factory that forks the snapshot")
    yield {}


EnvironmentFactory = Callable[[Task, int], AbstractAsyncContextManager[Mapping[str, Tool]]]


@dataclass
class Experiment:
    name: str
    roles: tuple[Role, ...]
    policies: Mapping[str, Callable[[], Policy]]
    mechanism: Callable[[], Mechanism]
    manifest: dict
    budget: Budget = field(default_factory=Budget)
    environment: EnvironmentFactory = stateless_environment

    async def trial(self, task: Task, condition: Condition, seed: int) -> Run:
        if not condition.prompts.keys() <= self.policies.keys():
            raise ValueError("Intervention names an unknown role")
        policies = {
            r: PromptPolicy(factory(), condition.prompts.get(r, ""))
            for r, factory in self.policies.items()
        }
        manifest = {**self.manifest, "condition": asdict(condition)}
        # Factory is entered anew for every arm, including repeats and failed runs.
        async with self.environment(task, seed) as tools:
            return await run(
                task,
                self.roles,
                policies,
                self.mechanism(),
                name=self.name,
                seed=seed,
                manifest=manifest,
                budget=self.budget,
                tools=tools,
            )

    async def sweep(
        self, tasks: Sequence[Task], conditions: Sequence[Condition], seeds: Sequence[int] = (0,)
    ) -> list[Run]:
        if len({t.id for t in tasks}) != len(tasks):
            raise ValueError("Task IDs must be unique")
        if len({c.id for c in conditions}) != len(conditions) or len(set(seeds)) != len(seeds):
            raise ValueError("Condition IDs and seeds must be unique")
        return [await self.trial(t, c, seed) for t in tasks for seed in seeds for c in conditions]


def parameter_grid(axes: Mapping[str, Sequence]) -> list[dict]:
    """Cross model families, capabilities, tool access, budgets and protocol settings."""
    from itertools import product

    if any(not values for values in axes.values()):
        raise ValueError("Sweep axes cannot be empty")
    return [dict(zip(axes, values, strict=True)) for values in product(*axes.values())]
