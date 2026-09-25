"""Estimands, matching rules and uncertainty are explicit; missing labels are never zero."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean

from .types import Evaluation, Run, finite


@dataclass(frozen=True)
class Estimate:
    value: float | None
    lower: float | None
    upper: float | None
    n: int
    unit: str


def bootstrap(
    values: list[float], *, seed: int = 0, draws: int = 2000, unit: str = "task"
) -> Estimate:
    if draws < 1:
        raise ValueError("draws must be positive")
    for v in values:
        finite(v)
    if not values:
        return Estimate(None, None, None, 0, unit)
    if len(values) == 1:
        return Estimate(values[0], None, None, 1, unit)
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return Estimate(
        mean(values),
        means[int(0.025 * (draws - 1))],
        means[int(0.975 * (draws - 1))],
        len(values),
        unit,
    )


@dataclass(frozen=True)
class PairedASD:
    estimate: Estimate
    pairs: int
    excluded: int
    manipulation: str
    task_differences: dict[str, float]


def paired_asd(
    runs: list[Run],
    role: str,
    *,
    good: str = "honest",
    bad: str = "deceptive",
    evaluations: list[Evaluation] | None = None,
    dimension: str = "quality",
    minimum_gap: float = 0,
    seed: int = 0,
) -> PairedASD:
    """Paired mechanism reward(good) - reward(bad), averaging repeats within task.

    With independent evaluations, include pairs only when realized good quality is higher.
    Supply exactly one chosen evaluator revision per run; intent-only ASD is labelled as such.
    Mixed mechanisms are rejected. Model/config comparisons must be stratified by the caller.
    """
    if len({r.mechanism for r in runs}) > 1:
        raise ValueError("Stratify ASD by mechanism")
    if good == bad or minimum_gap < 0:
        raise ValueError("Invalid contrast")
    labels = {}
    for evaluation in evaluations or []:
        if evaluation.run_id in labels:
            raise ValueError("Select one evaluation revision per run")
        labels[evaluation.run_id] = evaluation.scores.get(role, {}).get(dimension)
    pairs = defaultdict(dict)
    for r in runs:
        condition = r.manifest.get("condition", {}).get("id")
        if condition not in {good, bad}:
            continue
        key = r.task.id, r.task.snapshot, r.seed
        if condition in pairs[key]:
            raise ValueError("Duplicate arm; stratify by model and mechanism configuration")
        pairs[key][condition] = r
    task_values = defaultdict(list)
    included, excluded = 0, 0
    for (task_id, _, _), arms in pairs.items():
        if set(arms) != {good, bad} or any(r.status != "complete" for r in arms.values()):
            excluded += 1
            continue
        a, b = arms[good], arms[bad]
        if a.task != b.task or a.roles != b.roles:
            raise ValueError("Paired arms must share task inputs and role declarations")
        if evaluations is not None:
            ma, mb = labels.get(a.id), labels.get(b.id)
            if (
                ma is None
                or mb is None
                or ma.value is None
                or mb.value is None
                or ma.value - mb.value <= minimum_gap
            ):
                excluded += 1
                continue
        task_values[task_id].append(a.reward(role) - b.reward(role))
        included += 1
    values = {key: mean(v) for key, v in task_values.items()}
    return PairedASD(
        bootstrap(list(values.values()), seed=seed),
        included,
        excluded,
        "audited" if evaluations is not None else "intent-only",
        values,
    )


def alignment(rewards: list[float], qualities: list[float]) -> dict[str, float | None]:
    """Within one task/candidate pool: covariance, rank concordance, selection regret.

    Pairwise concordance excludes quality ties and assigns half credit to reward ties.
    Selection regret averages all maximum-reward ties; reward units remain unnormalized.
    """
    if not rewards or len(rewards) != len(qualities):
        raise ValueError("Need equally sized nonempty reward and quality arrays")
    rewards, qualities = list(map(finite, rewards)), list(map(finite, qualities))
    rbar, qbar = mean(rewards), mean(qualities)
    covariance = mean((r - rbar) * (q - qbar) for r, q in zip(rewards, qualities, strict=True))
    variance = mean((r - rbar) ** 2 for r in rewards) * mean((q - qbar) ** 2 for q in qualities)
    pairs = []
    for i in range(len(rewards)):
        for j in range(i):
            dq, dr = qualities[i] - qualities[j], rewards[i] - rewards[j]
            if dq:
                pairs.append(0.5 if dr == 0 else float(dr * dq > 0))
    chosen = [q for r, q in zip(rewards, qualities, strict=True) if r == max(rewards)]
    return {
        "covariance": covariance,
        "pearson": covariance / math.sqrt(variance) if variance else None,
        "concordance": mean(pairs) if pairs else None,
        "selection_regret": max(qualities) - mean(chosen),
        "expected_agent_score": rbar,
        "expected_quality": qbar,
    }


def gap_recovered(value: float, baseline: float, ceiling: float) -> float | None:
    for v in (value, baseline, ceiling):
        finite(v)
    return (value - baseline) / (ceiling - baseline) if ceiling > baseline else None


def tail_summary(values: list[float], fraction: float = 0.1) -> dict:
    """Descriptive lower-tail mean and minimum, not an extrapolated worst-case guarantee."""
    if not values or not 0 < fraction <= 1:
        raise ValueError("Need values and a tail fraction in (0,1]")
    values = sorted(map(finite, values))
    count = max(1, math.ceil(len(values) * fraction))
    return {
        "minimum": values[0],
        "lower_tail_mean": mean(values[:count]),
        "tail_n": count,
        "n": len(values),
    }


def detection(scores: list[float], labels: list[bool], *, threshold: float) -> dict:
    if not scores or len(scores) != len(labels) or any(type(v) is not bool for v in labels):
        raise ValueError("Need equally sized scores and binary labels")
    scores = list(map(finite, scores))
    finite(threshold)
    positive = [s for s, y in zip(scores, labels, strict=True) if y]
    negative = [s for s, y in zip(scores, labels, strict=True) if not y]
    return {
        "tpr": mean(s >= threshold for s in positive) if positive else None,
        "fpr": mean(s >= threshold for s in negative) if negative else None,
        "auroc": mean(float(p > n) + 0.5 * (p == n) for p in positive for n in negative)
        if positive and negative
        else None,
        "n_positive": len(positive),
        "n_negative": len(negative),
    }
