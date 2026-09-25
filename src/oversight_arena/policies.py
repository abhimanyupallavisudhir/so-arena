"""Composable policy interventions; prompt labels never imply ground-truth labels."""

from __future__ import annotations

import copy
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace

from .core import JSON, Observation, Policy, Response


@dataclass(frozen=True)
class FunctionPolicy:
    function: Callable[[Observation], Awaitable[Response]]

    async def __call__(self, observation: Observation) -> Response:
        return await self.function(observation)


@dataclass(frozen=True)
class ConstantPolicy:
    content: JSON

    async def __call__(self, observation: Observation) -> Response:
        return Response(copy.deepcopy(self.content), cost=0, tokens=0)


@dataclass(frozen=True)
class PromptedPolicy:
    base: Policy
    prompt: str

    async def __call__(self, observation: Observation) -> Response:
        return await self.base(
            replace(observation, instruction=f"{self.prompt}\n\n{observation.instruction}")
        )


@dataclass(frozen=True)
class MixturePolicy:
    """Seeded mixture over arbitrary policies; one draw per observation."""

    policies: Sequence[Policy]
    weights: Sequence[float] | None = None

    def __post_init__(self):
        from .optimization import probabilities

        probabilities(self.weights, len(self.policies))

    async def __call__(self, observation: Observation) -> Response:
        from .optimization import probabilities

        weights = probabilities(self.weights, len(self.policies))
        index = random.Random(observation.seed).choices(range(len(weights)), weights)[0]
        return await self.policies[index](observation)
