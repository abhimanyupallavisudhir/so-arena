"""Natural, replay and stratified policy sampling without assigning truth from prompts."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .experiments import PromptPolicy
from .optimization import distribution
from .runtime import Policy
from .types import Action, Observation


@dataclass
class ReplayPolicy:
    actions: Sequence[Action]
    position: int = 0

    async def __call__(self, observation: Observation) -> Action:
        if self.position >= len(self.actions):
            raise ValueError("Replay exhausted; the protocol requested an unrecorded continuation")
        action = self.actions[self.position]
        self.position += 1
        return action


@dataclass(frozen=True)
class Strategy:
    id: str
    prompt: str
    stratum: str = "natural"


@dataclass(frozen=True)
class SampledPolicy:
    strategy: Strategy
    policy: Policy


def sample_strategies(
    factory: Callable[[], Policy],
    strategies: Sequence[Strategy],
    *,
    count: int,
    seed: int = 0,
    weights: Sequence[float] | None = None,
    balanced: bool = False,
) -> list[SampledPolicy]:
    """Sample complete policy interventions, not candidate answers from unrelated tasks.

    Balanced sampling cycles randomized strata, then samples uniformly within each stratum.
    Stratum names describe elicitation intent only. Audit the resulting trajectories separately.
    """
    if not strategies or count < 1 or len({s.id for s in strategies}) != len(strategies):
        raise ValueError("Need distinct strategies and a positive count")
    if balanced and weights is not None:
        raise ValueError("Choose balanced strata or weighted strategy sampling")
    rng = random.Random(seed)
    if balanced:
        strata = sorted({s.stratum for s in strategies})
        order = []
        while len(order) < count:
            rng.shuffle(strata)
            order.extend(strata)
        choices = [
            rng.choice([s for s in strategies if s.stratum == group]) for group in order[:count]
        ]
    else:
        probs = distribution(
            weights if weights is not None else [1 / len(strategies)] * len(strategies)
        )
        if len(probs) != len(strategies):
            raise ValueError("Weights do not match strategies")
        choices = rng.choices(strategies, weights=probs, k=count)
    return [SampledPolicy(s, PromptPolicy(factory(), s.prompt)) for s in choices]
