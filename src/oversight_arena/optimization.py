"""Finite-sample optimization and reward-only prompt search, with explicit scope."""

from __future__ import annotations

import math
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from .types import Run, Task, finite


def distribution(weights: Sequence[float]) -> list[float]:
    weights = list(map(finite, weights))
    if (
        not weights
        or any(w < 0 for w in weights)
        or not math.isclose(sum(weights), 1, abs_tol=1e-9)
    ):
        raise ValueError("Expected a probability distribution summing to 1")
    total = sum(weights)
    return [w / total for w in weights]


def best_of_n(scores: Sequence[float], n: int, base: Sequence[float] | None = None) -> list[float]:
    """Exact with-replacement BoN distribution on an empirical pool.

    Ties split mass proportional to base probability. No fabricated ordering of ties.
    n=1 recovers the base distribution. Finite support cannot discover new behavior.
    """
    if not scores or isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Need nonempty scores and positive integer n")
    scores = list(map(finite, scores))
    probs = distribution(base if base is not None else [1 / len(scores)] * len(scores))
    if len(probs) != len(scores):
        raise ValueError("Base distribution size mismatch")
    result = [0.0] * len(scores)
    lower = 0.0
    highest_supported = max(s for s, p in zip(scores, probs, strict=True) if p > 0)
    for score in sorted(set(scores)):
        indices = [i for i, s in enumerate(scores) if s == score]
        mass = sum(probs[i] for i in indices)
        upper = 1.0 if score >= highest_supported else min(1.0, lower + mass)
        selected = upper**n - lower**n
        for i in indices:
            result[i] = selected * probs[i] / mass if mass else 0.0
        lower = upper
    return distribution(result)


def optimization_curve(
    rewards: Sequence[float],
    qualities: Sequence[float] | None,
    budgets: Sequence[int],
    base: Sequence[float] | None = None,
) -> list[dict]:
    if qualities is not None and len(qualities) != len(rewards):
        raise ValueError("Quality size mismatch")
    if qualities is not None:
        qualities = list(map(finite, qualities))
    rows = []
    for n in budgets:
        weights = best_of_n(rewards, n, base)
        rows.append(
            {
                "budget": n,
                "reward": sum(p * r for p, r in zip(weights, rewards, strict=True)),
                "quality": None
                if qualities is None
                else sum(p * q for p, q in zip(weights, qualities, strict=True)),
                "effective_support": 1 / sum(p * p for p in weights),
            }
        )
    return rows


@dataclass(frozen=True)
class Leaf:
    rewards: dict[str, float]
    qualities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    # None = chance, fixed base probabilities; otherwise the optimizing player ID.
    player: str | None
    children: tuple[Leaf | Decision, ...]
    base: tuple[float, ...] | None = None


def nested_best_of_n(tree: Leaf | Decision, budgets: dict[str, int]) -> Leaf:
    """Backward induction over a conditional, perfect-information game tree.

    General-sum: each node selects using its OWN player's reward. This is not a solver
    for hidden-information games, where tied information sets constrain joint policies.
    """
    if isinstance(tree, Leaf):
        for value in (*tree.rewards.values(), *tree.qualities.values()):
            finite(value)
        return tree
    values = [nested_best_of_n(c, budgets) for c in tree.children]
    if not values:
        raise ValueError("Decision has no children")
    for value in values:
        if (
            value.rewards.keys() != values[0].rewards.keys()
            or value.qualities.keys() != values[0].qualities.keys()
        ):
            raise ValueError("All leaves must have matching reward and quality dimensions")
    base = tree.base if tree.base is not None else (1 / len(values),) * len(values)
    if len(base) != len(values):
        raise ValueError("Chance/base distribution size mismatch")
    weights = (
        distribution(base)
        if tree.player is None
        else best_of_n([v.rewards[tree.player] for v in values], budgets.get(tree.player, 1), base)
    )
    return Leaf(
        {
            r: sum(p * v.rewards[r] for p, v in zip(weights, values, strict=True))
            for r in values[0].rewards
        },
        {
            r: sum(p * v.qualities[r] for p, v in zip(weights, values, strict=True))
            for r in values[0].qualities
        },
    )


@dataclass(frozen=True)
class SearchFeedback:
    prompt: str
    reward: float | None
    feasible: bool
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class SearchResult:
    stratum: str
    selected: str | None
    history: tuple[SearchFeedback, ...]
    holdout_run_ids: tuple[str, ...]
    opponent_snapshot: str


@dataclass(frozen=True)
class PromptUpdate:
    step: int
    role: str
    opponents: dict[str, str]
    prompt: str
    reward: float
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class JointSearchResult:
    prompts: dict[str, str]
    updates: tuple[PromptUpdate, ...]
    holdout_run_ids: tuple[str, ...]
    schedule: str


PromptProposer = Callable[[tuple[SearchFeedback, ...], str, random.Random], Awaitable[str]]
PromptEvaluator = Callable[[str, Sequence[Task], int], Awaitable[list[Run]]]


async def prompt_search(
    *,
    propose: PromptProposer,
    evaluate: PromptEvaluator,
    train: Sequence[Task],
    holdout: Sequence[Task],
    role: str,
    iterations: int,
    stratum: str = "unrestricted",
    seed: int = 0,
    opponent_snapshot: str,
    constraint: Callable[[Run], bool] | None = None,
) -> SearchResult:
    """Optimize mechanism reward on train only; evaluate the selected prompt on holdout.

    A stratum is an instruction to the searcher, not a truth label. An optional independent
    constraint audits actual behavior. Feedback contains reward/feasibility, never quality.
    GEPA/DSPy/custom LLMs can implement propose; evaluate runs the real mechanism.
    """
    if (
        not train
        or not holdout
        or iterations < 1
        or not opponent_snapshot
        or any(t.split != "train" for t in train)
        or any(t.split not in {"validation", "test"} for t in holdout)
        or {t.id for t in train} & {t.id for t in holdout}
    ):
        raise ValueError("Search needs disjoint train/holdout tasks and an opponent snapshot")
    rng = random.Random(seed)
    history = []
    for _ in range(iterations):
        prompt = await propose(tuple(history), stratum, rng)
        results = await evaluate(prompt, train, seed)
        if {r.task.id for r in results} != {t.id for t in train}:
            raise ValueError("Evaluator returned incomplete or unexpected tasks")
        if any(not any(s.id == role and s.trainable for s in r.roles) for r in results):
            raise ValueError("Cannot optimize a fixture or unknown role")
        feasible = all(
            r.status == "complete" and (constraint is None or constraint(r)) for r in results
        )
        reward = sum(r.reward(role) for r in results) / len(results) if feasible else None
        history.append(SearchFeedback(prompt, reward, feasible, tuple(r.id for r in results)))
    feasible = [h for h in history if h.feasible]
    selected = max(feasible, key=lambda h: h.reward).prompt if feasible else None
    held = await evaluate(selected, holdout, seed) if selected is not None else []
    if selected is not None and {r.task.id for r in held} != {t.id for t in holdout}:
        raise ValueError("Holdout evaluator returned incomplete or unexpected tasks")
    return SearchResult(
        stratum, selected, tuple(history), tuple(r.id for r in held), opponent_snapshot
    )


async def joint_prompt_search(
    *,
    initial: dict[str, str],
    proposers: dict[str, PromptProposer],
    evaluate: Callable[[dict[str, str], Sequence[Task], int], Awaitable[list[Run]]],
    train: Sequence[Task],
    holdout: Sequence[Task],
    steps: int,
    schedule: str = "simultaneous",
    seed: int = 0,
    strata: dict[str, str] | None = None,
) -> JointSearchResult:
    """Coordinate prompt improvement with recorded opponent assumptions.

    Each role compares its incumbent with one proposal against a frozen opponent profile.
    Simultaneous updates share the pre-step profile; alternating updates see earlier updates.
    Peers' prompt contents are experimenter state, not injected into participant observations.
    The last joint profile is evaluated on holdout once, after all selection has finished.
    """
    if (
        not initial
        or not proposers
        or not proposers.keys() <= initial.keys()
        or steps < 1
        or schedule not in {"simultaneous", "alternating"}
        or not train
        or not holdout
        or any(t.split != "train" for t in train)
        or any(t.split not in {"validation", "test"} for t in holdout)
        or {t.id for t in train} & {t.id for t in holdout}
    ):
        raise ValueError("Invalid joint search configuration or train/holdout split")
    rng = random.Random(seed)
    profile, updates = dict(initial), []
    for step in range(steps):
        before, pending = dict(profile), dict(profile)
        for role, propose in proposers.items():
            reference = before if schedule == "simultaneous" else dict(pending)
            history = tuple(
                SearchFeedback(u.prompt, u.reward, True, u.run_ids)
                for u in updates
                if u.role == role
            )
            proposal = await propose(history, (strata or {}).get(role, "unrestricted"), rng)
            options = []
            for prompt in (reference[role], proposal):
                results = await evaluate({**reference, role: prompt}, train, seed)
                if len(results) != len(train) or {r.task.id for r in results} != {
                    t.id for t in train
                }:
                    raise ValueError("Joint search requires one result per task")
                if any(not any(s.id == role and s.trainable for s in r.roles) for r in results):
                    raise ValueError("Cannot optimize a fixture or unknown role")
                if all(r.status == "complete" for r in results):
                    options.append(
                        (
                            sum(r.reward(role) for r in results) / len(results),
                            prompt,
                            tuple(r.id for r in results),
                        )
                    )
            if not options:
                raise ValueError("Both incumbent and proposal failed; no policy update is defined")
            reward, prompt, ids = max(options, key=lambda option: option[0])
            pending[role] = prompt
            updates.append(
                PromptUpdate(
                    step,
                    role,
                    {r: p for r, p in reference.items() if r != role},
                    prompt,
                    reward,
                    ids,
                )
            )
        profile = pending
    held = await evaluate(profile, holdout, seed)
    if len(held) != len(holdout) or {r.task.id for r in held} != {t.id for t in holdout}:
        raise ValueError("Joint search requires one holdout result per task")
    return JointSearchResult(profile, tuple(updates), tuple(r.id for r in held), schedule)
