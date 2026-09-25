"""Finite-support optimization. Utilities, never ground truth, determine selection."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .analysis import truth
from .core import Episode, finite


def probabilities(weights: Sequence[float] | None, size: int) -> list[float]:
    if size < 1:
        raise ValueError("Empty support")
    result = list(weights) if weights is not None else [1 / size] * size
    if len(result) != size or any(finite(x) < 0 for x in result):
        raise ValueError("Invalid probability weights")
    total = sum(result)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Probability weights need positive finite mass")
    return [x / total for x in result]


def best_of_n_weights(
    values: Sequence[float], n: int, *, base: Sequence[float] | None = None
) -> list[float]:
    """Exact with-replacement BoN on empirical support, uniform among tied draws.

    A tied reward group's probability is F(r)^n - F(r-)^n; allocate within
    that group in proportion to its base probability. n=1 recovers the base.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    for value in values:
        finite(value)
    weights = probabilities(base, len(values))
    result = [0.0] * len(values)
    cumulative = 0.0
    levels = sorted({v for v, w in zip(values, weights, strict=True) if w > 0})
    for level, value in enumerate(levels):
        group = [i for i, v in enumerate(values) if v == value]
        mass = sum(weights[i] for i in group)
        upper = 1.0 if level == len(levels) - 1 else min(1.0, cumulative + mass)
        selected = upper**n - cumulative**n
        if mass:
            for i in group:
                result[i] = selected * weights[i] / mass
        cumulative = upper
    return probabilities(result, len(result))


def _validate_pool(records: Sequence[Episode]) -> None:
    if not records:
        raise ValueError("Need a nonempty candidate pool")
    first = records[0]
    if any(
        (r.task, r.mechanism, r.config, r.roles)
        != (first.task, first.mechanism, first.config, first.roles)
        for r in records
    ):
        raise ValueError("Candidates must share one task, mechanism configuration and role set")


def select_best(records: Sequence[Episode], role: str, *, seed: int = 0) -> Episode:
    _validate_pool(records)
    best = max(record.rewards[role].value for record in records)
    return random.Random(seed).choice([r for r in records if r.rewards[role].value == best])


@dataclass(frozen=True)
class OptimizationPoint:
    pressure: dict[str, int]
    rewards: dict[str, float]
    truth: dict[str, float | None]
    support: int
    effective_support: float


def best_of_n(
    records: Sequence[Episode],
    role: str,
    n: int,
    *,
    metric: str = "quality",
    scorer: str | None = None,
) -> OptimizationPoint:
    _validate_pool(records)
    keys = set(records[0].rewards)
    if any(set(record.rewards) != keys for record in records):
        raise ValueError("Inconsistent reward roles")
    weights = best_of_n_weights([r.rewards[role].value for r in records], n)
    rewards = {
        key: sum(w * r.rewards[key].value for w, r in zip(weights, records, strict=True))
        for key in keys
    }
    values = {}
    for key in keys:
        qualities = [truth(r, key, metric, scorer=scorer) for r in records]
        values[key] = (
            None
            if any(q is None and w > 0 for q, w in zip(qualities, weights, strict=False))
            else sum(w * q for w, q in zip(weights, qualities, strict=False) if q is not None)
        )
    return OptimizationPoint(
        {role: n}, rewards, values, len(records), 1 / sum(w * w for w in weights)
    )


@dataclass(frozen=True)
class Leaf:
    rewards: dict[str, float]
    truth: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    player: str
    children: tuple[Decision | Leaf, ...]
    base: tuple[float, ...] | None = None


@dataclass(frozen=True)
class TreeResult:
    rewards: dict[str, float]
    truth: dict[str, float | None]
    leaf_probabilities: dict[tuple[int, ...], float]


def nested_best_of_n(tree: Decision | Leaf, pressure: Mapping[str, int]) -> TreeResult:
    """Backward induction for an observed, perfect-information, general-sum game tree.

    Each node optimizes its player's own continuation utility. Each child must
    be a continuation sampled conditional on that branch, not an unrelated trace.
    """
    if any(type(n) is not int or n < 1 for n in pressure.values()):
        raise ValueError("Invalid pressure")
    if isinstance(tree, Leaf):
        if not tree.rewards:
            raise ValueError("Leaf needs rewards")
        for value in tree.rewards.values():
            finite(value)
        for value in tree.truth.values():
            if value is not None:
                finite(value)
        return TreeResult(dict(tree.rewards), dict(tree.truth), {(): 1.0})
    children = [nested_best_of_n(child, pressure) for child in tree.children]
    if not children or any(set(c.rewards) != set(children[0].rewards) for c in children):
        raise ValueError("Nonempty, consistently rewarded branches required")
    weights = best_of_n_weights(
        [c.rewards[tree.player] for c in children], pressure.get(tree.player, 1), base=tree.base
    )
    rewards = {
        key: sum(w * c.rewards[key] for w, c in zip(weights, children, strict=False))
        for key in children[0].rewards
    }
    metrics = set().union(*(c.truth.keys() for c in children))
    qualities = {}
    for key in metrics:
        qualities[key] = (
            None
            if any(
                w > 0 and c.truth.get(key) is None for w, c in zip(weights, children, strict=False)
            )
            else sum(
                w * c.truth[key]
                for w, c in zip(weights, children, strict=False)
                if c.truth.get(key) is not None
            )
        )
    leaves = {
        (index, *path): weight * probability
        for index, (weight, child) in enumerate(zip(weights, children, strict=False))
        for path, probability in child.leaf_probabilities.items()
    }
    return TreeResult(rewards, qualities, leaves)


@dataclass(frozen=True)
class SamplingTurn:
    player: str
    samples: int


async def sample_game_tree(initial, turns: Sequence[SamplingTurn], expand, score, *, seed: int = 0):
    """Build a conditional game tree using async domain-independent callbacks.

    expand(state, player, candidate_index, seed) returns a new continuation state.
    score(state) returns a Leaf. Each sibling receives a deep copy, so all draws
    condition on the same prefix. This builds the full product tree: set small
    sample budgets first. Model callbacks must honor seeds where supported.
    """
    import copy

    from .core import digest

    if any(type(t.samples) is not int or t.samples < 1 for t in turns):
        raise ValueError("Each turn needs a positive sample budget")

    async def visit(state, depth, path):
        if depth == len(turns):
            leaf = await score(copy.deepcopy(state))
            if not isinstance(leaf, Leaf):
                raise TypeError("score must return a Leaf")
            return leaf
        turn = turns[depth]
        children = []
        for index in range(turn.samples):
            branch_seed = int(digest([seed, depth, list(path), index])[:8], 16)
            child = await expand(copy.deepcopy(state), turn.player, index, branch_seed)
            children.append(await visit(child, depth + 1, (*path, index)))
        return Decision(turn.player, tuple(children))

    return await visit(initial, 0, ())
