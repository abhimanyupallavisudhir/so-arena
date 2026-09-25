"""Evaluating candidate strategies for one role by running episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.episode import EpisodeRecord
from ..core.strategy import Assignment, Profile, Strategy
from ..core.task import Task
from ..core.util import gather_limited, rng_for
from ..domains.base import Domain
from ..experiment.runner import AgentTable, run_episode
from ..mechanisms.base import Mechanism

Mixture = list[tuple[Strategy, float]]


@dataclass
class Evaluation:
    strategy: Strategy
    reward: float
    reward_se: float
    gt: dict[str, float]
    per_task: dict[str, float]
    records: list[EpisodeRecord] = field(default_factory=list, repr=False)

    @property
    def n(self) -> int:
        return len(self.records)


class Evaluator:
    """Runs episodes with ``role`` playing a candidate strategy against fixed/mixed others.

    Args:
        others: per other role, a fixed Strategy or a mixture ``[(strategy, weight), ...]``
            (sampled deterministically per task/seed) — e.g. an opponent meta-strategy in PSRO.
        seeds: independent samples per task.
        gt_keys: ground-truth scorer names to average (reported, never shown to optimisers).
    """

    def __init__(
        self,
        domain: Domain,
        mechanism: Mechanism,
        agents: "AgentTable | dict[str, Any] | Any",
        role: str,
        tasks: Sequence[Task] | None = None,
        others: dict[str, "Strategy | Mixture"] | None = None,
        seeds: int = 1,
        gt_keys: Sequence[str] = ("correct",),
        clearances: dict[str, Sequence[str]] | None = None,
        concurrency: int = 8,
        position: str | None = None,
    ):
        self.domain = domain
        self.mechanism = mechanism
        self.agents = agents if isinstance(agents, AgentTable) else AgentTable(agents)
        self.role = role
        self.tasks = list(tasks) if tasks is not None else domain.tasks()
        self.others = others or {}
        self.seeds = seeds
        self.gt_keys = list(gt_keys)
        self.clearances = clearances
        self.concurrency = concurrency
        self.position = position
        self.n_episodes = 0

    def _other(self, role: str, spec: "Strategy | Mixture", task: Task, seed: int) -> Strategy:
        if isinstance(spec, Strategy):
            return spec
        rng = rng_for("mixture", role, task.id, seed)
        ws = np.array([w for _, w in spec], float)
        ws = ws / ws.sum()
        return spec[int(rng.choices(range(len(spec)), weights=list(ws))[0])][0]

    def profile(self, strategy: Strategy, task: Task, seed: int) -> Profile:
        asg = {self.role: Assignment(strategy=strategy, seed=seed, position=self.position)}
        for r, spec in self.others.items():
            asg[r] = Assignment(strategy=self._other(r, spec, task, seed), seed=seed)
        return Profile(assignments=asg)

    async def evaluate(self, strategy: Strategy, tasks: Sequence[Task] | None = None) -> Evaluation:
        tasks = list(tasks) if tasks is not None else self.tasks
        jobs = [(t, s) for t in tasks for s in range(self.seeds)]

        def make(t: Task, s: int):
            return lambda: run_episode(
                self.mechanism, t, self.profile(strategy, t, s), self.agents, self.domain,
                clearances=self.clearances, seed=s,
            )

        recs = await gather_limited([make(t, s) for t, s in jobs], self.concurrency)
        self.n_episodes += len(recs)
        return summarize(strategy, recs, self.role, self.gt_keys)


def summarize(strategy: Strategy, recs: list[EpisodeRecord], role: str, gt_keys: Sequence[str]) -> Evaluation:
    ok = [r for r in recs if r.error is None and role in r.rewards]
    rewards = np.array([r.rewards[role] for r in ok], float)
    per_task: dict[str, list[float]] = {}
    for r in ok:
        per_task.setdefault(r.task_id, []).append(r.rewards[role])
    gt: dict[str, float] = {}
    for k in gt_keys:
        vals = [r.gt.get(k, {}).get(role) for r in ok]
        vals = [v for v in vals if v is not None]
        if vals:
            gt[k] = float(np.mean(vals))
        outs = [r.gt.get(k, {}).get("_outcome") for r in ok]
        outs = [v for v in outs if v is not None]
        if outs:
            gt[f"{k}:_outcome"] = float(np.mean(outs))
    task_means = np.array([np.mean(v) for v in per_task.values()], float)
    return Evaluation(
        strategy=strategy,
        reward=float(rewards.mean()) if len(rewards) else float("nan"),
        # tasks are the independent units (seeds within a task are correlated)
        reward_se=float(task_means.std(ddof=1) / np.sqrt(len(task_means))) if len(task_means) > 1 else float("nan"),
        gt=gt,
        per_task={k: float(np.mean(v)) for k, v in per_task.items()},
        records=recs,
    )
