"""Actual REINFORCE on categorical policy populations plus backend-neutral experience export."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .core import Episode, Policy, Role, Scorer, Task, finite
from .runtime import Mechanism, run_episode


@dataclass(frozen=True)
class Experience:
    episode_id: str
    role: str
    observations_and_actions: tuple[dict, ...]
    reward: float


def training_batch(records: Sequence[Episode]) -> list[Experience]:
    """Only mechanism utility and role-visible transcript; other private messages/channels excluded.

    Backends needing exact model prompts/logprobs should log those separately via
    their policy adapter. This export is not sufficient for token-level PPO.
    """
    return [
        Experience(
            r.id,
            role.id,
            tuple(
                {"actor": e.actor, "kind": e.kind, "content": copy.deepcopy(e.content)}
                for e in r.events
                if e.recipients is None or role.id in e.recipients
            ),
            r.rewards[role.id].value,
        )
        for r in records
        for role in r.roles
        if role.trainable
    ]


class Trainer(Protocol):
    async def update(self, experiences: Sequence[Experience]) -> Mapping[str, Policy]: ...


@dataclass
class CategoricalREINFORCE:
    """One categorical action selects a complete policy per role per episode.

    Updates are simultaneous; each role uses its own mechanism reward and a
    previous-batches running baseline. This trains mixture logits, not LLM weights.
    """

    populations: Mapping[str, Sequence[Policy]]
    learning_rate: float = 0.1
    seed: int = 0
    logits: dict[str, list[float]] = field(init=False)
    baselines: dict[str, float] = field(init=False)
    updates: int = field(default=0, init=False)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self):
        if finite(self.learning_rate) <= 0 or not self.populations:
            raise ValueError("Need positive learning rate and populations")
        if any(not p for p in self.populations.values()):
            raise ValueError("Empty population")
        self.logits = {r: [0.0] * len(p) for r, p in self.populations.items()}
        self.baselines = dict.fromkeys(self.populations, 0.0)
        self._rng = random.Random(self.seed)

    def distributions(self) -> dict[str, list[float]]:
        result = {}
        for role, logits in self.logits.items():
            exps = [math.exp(x - max(logits)) for x in logits]
            result[role] = [x / sum(exps) for x in exps]
        return result

    def checkpoint(self) -> dict:
        return {
            "logits": {r: list(v) for r, v in self.logits.items()},
            "baselines": dict(self.baselines),
            "updates": self.updates,
            "seed": self.seed,
            "learning_rate": self.learning_rate,
            "rng_state": self._rng.getstate(),
        }

    def restore(self, checkpoint: dict) -> None:
        if set(checkpoint["logits"]) != set(self.logits):
            raise ValueError("Checkpoint roles do not match")
        for role, values in checkpoint["logits"].items():
            if len(values) != len(self.logits[role]):
                raise ValueError("Checkpoint population size does not match")
            for value in values:
                finite(value)
        if set(checkpoint["baselines"]) != set(self.logits):
            raise ValueError("Checkpoint baseline roles do not match")
        baselines = {r: finite(v) for r, v in checkpoint["baselines"].items()}
        updates = checkpoint["updates"]
        learning_rate = finite(checkpoint["learning_rate"])
        if type(updates) is not int or updates < 0 or learning_rate <= 0:
            raise ValueError("Invalid checkpoint update count or learning rate")
        state = checkpoint["rng_state"]
        rng = random.Random()
        rng.setstate((state[0], tuple(state[1]), state[2]))
        self.logits = {r: list(v) for r, v in checkpoint["logits"].items()}
        self.baselines = baselines
        self.updates = updates
        self.learning_rate = learning_rate
        self.seed = checkpoint["seed"]
        self._rng = rng

    async def step(
        self,
        tasks: Sequence[Task],
        mechanism: Mechanism,
        roles: Sequence[Role],
        fixtures: Mapping[str, Policy],
        *,
        scorer: Scorer | None = None,
        **runtime_options,
    ) -> tuple[Episode, ...]:
        if not tasks or any(t.split != "train" for t in tasks):
            raise ValueError("REINFORCE updates require tasks explicitly marked train")
        trainable = {r.id for r in roles if r.trainable}
        if set(self.populations) - trainable or set(fixtures) & set(self.populations):
            raise ValueError("Populations must be trainable and disjoint from fixed policies")
        distributions = self.distributions()
        gradients = {r: [0.0] * len(v) for r, v in distributions.items()}
        returns = {r: [] for r in self.populations}
        records = []
        for task in tasks:
            actions = {r: self._rng.choices(range(len(p)), p)[0] for r, p in distributions.items()}
            profile = {**fixtures, **{r: self.populations[r][a] for r, a in actions.items()}}
            record = await run_episode(
                task,
                mechanism,
                roles,
                profile,
                seed=self._rng.randrange(2**32),
                scorer=scorer,
                provenance={
                    "trainer": "categorical_reinforce",
                    "update": self.updates,
                    "actions": actions,
                    "distributions": distributions,
                },
                **runtime_options,
            )
            records.append(record)
            for role, action in actions.items():
                reward = record.rewards[role].value
                returns[role].append(reward)
                advantage = reward - self.baselines[role]
                for index, p in enumerate(distributions[role]):
                    gradients[role][index] += advantage * (float(index == action) - p)
        for role in self.populations:
            for i, gradient in enumerate(gradients[role]):
                self.logits[role][i] += self.learning_rate * gradient / len(tasks)
            observed = sum(returns[role]) / len(tasks)
            self.baselines[role] += (observed - self.baselines[role]) / (self.updates + 1)
        self.updates += 1
        return tuple(records)
