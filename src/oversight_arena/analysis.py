"""Incentive diagnostics on matched experiments, with task-clustered uncertainty."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from .core import Episode, Evaluation, canonical, finite


def truth(
    episode: Episode, role: str, metric: str = "quality", *, scorer: str | None = None
) -> float | None:
    evaluations = [e for e in episode.evaluations if scorer is None or e.scorer == scorer]
    if scorer is None and len({e.scorer for e in evaluations}) > 1:
        raise ValueError("Multiple scorers: select one explicitly")
    if not evaluations or evaluations[-1].status != "resolved":
        return None
    return evaluations[-1].per_agent.get(role, {}).get(metric)


def latest_evaluation(episode: Episode, scorer: str) -> Evaluation | None:
    return next((e for e in reversed(episode.evaluations) if e.scorer == scorer), None)


@dataclass(frozen=True)
class Estimate:
    mean: float
    low: float
    high: float
    tasks: int
    observations: int
    minimum_task_mean: float


def clustered_mean(
    values: Sequence[tuple[str, float]],
    *,
    seed: int = 0,
    bootstrap: int = 2000,
    confidence: float = 0.95,
) -> Estimate:
    """Equal weight per task; repetitions stay inside their task cluster."""
    if not values or bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("Need observations, positive bootstrap count and valid confidence")
    clusters = defaultdict(list)
    for task_id, value in values:
        clusters[task_id].append(finite(value))
    means = [mean(group) for group in clusters.values()]
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(means, k=len(means))) for _ in range(bootstrap))
    tail = (1 - confidence) / 2
    return Estimate(
        mean(means),
        samples[int(tail * (bootstrap - 1))],
        samples[int((1 - tail) * (bootstrap - 1))],
        len(means),
        len(values),
        min(means),
    )


def paired_asd(good: Sequence[Episode], bad: Sequence[Episode], role: str, **kwargs) -> Estimate:
    """Generalized ASD: E[R_i(good intervention) - R_i(bad intervention)].

    Labels describe interventions; this function does not assert realized honesty.
    Configure log/Brier utilities in the mechanism when reproducing the ASD paper.
    """

    def indexed(records):
        result = {}
        for record in records:
            record.validate()
            key = (record.task.id, record.seed)
            if key in result:
                raise ValueError(f"Duplicate task/seed pair: {key}")
            result[key] = record
        return result

    left, right = indexed(good), indexed(bad)
    if not left or left.keys() != right.keys():
        raise ValueError("ASD requires identical nonempty task/seed pairs")
    differences = []
    for key, a in left.items():
        b = right[key]
        if (a.task, a.mechanism, a.config, a.roles) != (b.task, b.mechanism, b.config, b.roles):
            raise ValueError("ASD pairs must use the same task and mechanism configuration")
        differences.append((a.task.id, a.rewards[role].value - b.rewards[role].value))
    return clustered_mean(differences, **kwargs)


def ranking_diagnostics(
    records: Sequence[Episode], role: str, metric: str = "quality", *, scorer: str | None = None
) -> dict[str, float | int | None]:
    """Within-task reward/quality ordering, including coverage and ties."""
    groups = defaultdict(list)
    known = 0
    for record in records:
        quality = truth(record, role, metric, scorer=scorer)
        if quality is not None:
            known += 1
            groups[
                (canonical(record.task.__dict__), record.mechanism, canonical(record.config))
            ].append((record.rewards[role].value, quality))
    concordant = discordant = reward_ties = 0
    regrets = []
    for candidates in groups.values():
        if len(candidates) < 2:
            continue
        best_reward = max(r for r, _ in candidates)
        chosen = [q for r, q in candidates if r == best_reward]
        regrets.append(max(q for _, q in candidates) - mean(chosen))
        for i, (ra, qa) in enumerate(candidates):
            for rb, qb in candidates[i + 1 :]:
                if qa == qb:
                    continue
                product = (ra - rb) * (qa - qb)
                concordant += product > 0
                discordant += product < 0
                reward_ties += product == 0
    comparisons = concordant + discordant + reward_ties
    return {
        "records": len(records),
        "truth_coverage": known / len(records) if records else 0,
        "comparable_pairs": comparisons,
        "reward_ties": reward_ties,
        "ordering_agreement": (
            (concordant + 0.5 * reward_ties) / comparisons if comparisons else None
        ),
        "mean_selection_regret_observed_support": mean(regrets) if regrets else None,
    }


def calibration(probabilities: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> dict:
    if len(probabilities) != len(outcomes) or not outcomes or bins < 1:
        raise ValueError("Need matched probabilities and binary outcomes")
    buckets = defaultdict(list)
    errors = []
    for p, y in zip(probabilities, outcomes, strict=True):
        if not 0 <= finite(p) <= 1 or y not in (0, 1):
            raise ValueError("Invalid probability or outcome")
        buckets[min(int(p * bins), bins - 1)].append((p, y))
        errors.append((p - y) ** 2)
    ece = sum(
        len(b) * abs(mean(p for p, _ in b) - mean(y for _, y in b)) for b in buckets.values()
    ) / len(outcomes)
    return {"brier": mean(errors), "ece": ece, "n": len(outcomes)}


def binary_score(probability: float, outcome: int = 1, rule: str = "brier") -> float:
    """Negative two-class Brier (range [-2,0]) or log probability. No silent clipping."""
    p = finite(probability)
    if not 0 <= p <= 1 or outcome not in (0, 1):
        raise ValueError("Invalid probability or outcome")
    if rule == "brier":
        return -2 * (p - outcome) ** 2
    if rule == "log":
        assigned = p if outcome else 1 - p
        if assigned == 0:
            raise ValueError("Log score is -infinity; choose and document explicit smoothing")
        return math.log(assigned)
    raise ValueError(f"Unknown scoring rule: {rule}")


def optimization_response(
    records: Sequence[Episode], role: str, *, metric: str = "quality", scorer: str | None = None
) -> dict[str, float | int | None]:
    """Within-task covariance: local quality response to exponential reward tilting.

    Missing truth is omitted with explicit coverage. This is a sampled-support
    derivative at zero pressure, not a general RL improvement guarantee.
    """
    groups = defaultdict(list)
    covered = 0
    for record in records:
        q = truth(record, role, metric, scorer=scorer)
        if q is not None:
            groups[
                (canonical(record.task.__dict__), record.mechanism, canonical(record.config))
            ].append((record.rewards[role].value, q))
            covered += 1
    covariances = []
    for group in groups.values():
        if len(group) < 2:
            continue
        rbar, qbar = mean(r for r, _ in group), mean(q for _, q in group)
        covariances.append(mean((r - rbar) * (q - qbar) for r, q in group))
    return {
        "covariance": mean(covariances) if covariances else None,
        "tasks": len(covariances),
        "truth_coverage": covered / len(records) if records else 0,
    }
