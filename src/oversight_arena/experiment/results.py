"""Results: a collection of episode records with tabular views, re-scoring and GT resolution."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.episode import EpisodeRecord
from ..core.rewards import RewardRule
from ..core.task import Task
from ..core.util import read_jsonl, run_sync, write_jsonl


class Results:
    def __init__(self, records: Iterable[EpisodeRecord], tasks: dict[str, Task] | None = None):
        self.records: list[EpisodeRecord] = list(records)
        self.tasks: dict[str, Task] = dict(tasks or {})

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __add__(self, other: "Results") -> "Results":
        return Results(self.records + other.records, {**self.tasks, **other.tasks})

    def filter(self, fn: Callable[[EpisodeRecord], bool] | None = None, **eq: Any) -> "Results":
        out = []
        for r in self.records:
            if fn is not None and not fn(r):
                continue
            if any(getattr(r, k) != v for k, v in eq.items()):
                continue
            out.append(r)
        return Results(out, self.tasks)

    @property
    def errors(self) -> list[EpisodeRecord]:
        return [r for r in self.records if r.error]

    def ok(self) -> "Results":
        return self.filter(lambda r: r.error is None)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        write_jsonl(path, self.records)
        return path

    @classmethod
    def load(cls, path: str | Path, tasks: dict[str, Task] | None = None) -> "Results":
        path = Path(path)
        if path.is_dir():
            path = path / "episodes.jsonl"
        return cls([EpisodeRecord.model_validate(r) for r in read_jsonl(path)], tasks)

    # ------------------------------------------------------------------ tables
    def df(self, trainable_only: bool = False, include_fixtures: bool = True) -> pd.DataFrame:
        """Long format: one row per (episode, role).

        Ground truth: ``gt_<scorer>`` is the *role's* value; principal-level values a scorer
        reports under ``_outcome*`` keys are repeated on every row of the episode as
        ``gt_<scorer>[_outcome]`` (same names as in :meth:`episodes_df`).
        """
        rows = []
        for r in self.records:
            outcome_gt = {
                f"gt_{s}[{k}]": v
                for s, vals in r.gt.items()
                for k, v in vals.items()
                if k.startswith("_outcome")
            }
            for spec in r.roles:
                if trainable_only and not spec.trainable:
                    continue
                if not include_fixtures and not spec.trainable:
                    continue
                b = r.bound.get(spec.name)
                row: dict[str, Any] = {
                    "episode": r.id,
                    "key": r.key,
                    "experiment": r.experiment,
                    "mechanism": r.mechanism,
                    "mechanism_hash": r.mechanism_hash,
                    "reward_rule": r.reward_rule,
                    "task": r.task_id,
                    "domain": r.domain,
                    "profile": r.profile.id,
                    "profile_label": r.profile.label,
                    "seed": r.seed,
                    "role": spec.name,
                    "role_kind": spec.kind,
                    "trainable": spec.trainable,
                    "agent": b.agent if b else None,
                    "strategy": b.strategy_id if b else None,
                    "strategy_name": b.strategy_name if b else None,
                    "sample": b.seed if b else 0,
                    "target": b.target if b else None,
                    "reward": r.rewards.get(spec.name),
                    "decision": r.outcome.get("decision"),
                    "error": r.error is not None,
                    "gt_status": r.gt_status,
                    "tokens": r.usage.get(spec.name).total_tokens if r.usage.get(spec.name) else 0,
                    "cost_usd": r.usage.get(spec.name).cost_usd if r.usage.get(spec.name) else 0.0,
                    "n_words": sum(len(e.content.split()) for e in r.transcript.entries if e.role == spec.name),
                }
                if b:
                    for k, v in b.tags.items():
                        if isinstance(v, (str, int, float, bool)) or v is None:
                            row[f"tag_{k}"] = v
                for s, vals in r.gt.items():
                    row[f"gt_{s}"] = vals.get(spec.name)
                row.update(outcome_gt)
                rows.append(row)
        return pd.DataFrame(rows)

    def episodes_df(self) -> pd.DataFrame:
        """Wide format: one row per episode (rewards/GT per role as columns)."""
        rows = []
        for r in self.records:
            row: dict[str, Any] = {
                "episode": r.id, "mechanism": r.mechanism, "task": r.task_id, "domain": r.domain,
                "profile": r.profile.id, "profile_label": r.profile.label, "seed": r.seed,
                "decision": r.outcome.get("decision"), "error": r.error is not None,
                "gt_status": r.gt_status, "tokens": r.total_usage().total_tokens,
                "cost_usd": r.total_usage().cost_usd, "gt_channel_cost": sum(c.cost for c in r.channels),
            }
            for k, v in r.rewards.items():
                row[f"reward[{k}]"] = v
            for role, b in r.bound.items():
                row[f"strategy[{role}]"] = b.strategy_name
                row[f"target[{role}]"] = b.target
            for s, vals in r.gt.items():
                for k, v in vals.items():
                    row[f"gt_{s}[{k}]"] = v
            probs = r.outcome.get("probs")
            if isinstance(probs, dict):
                for k, v in probs.items():
                    row[f"p[{k}]"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ re-scoring
    def rescore(self, rule: RewardRule, mechanism: str | None = None, mechanism_hash: str | None = None) -> "Results":
        """Recompute rewards with a different reward rule (no model calls), for all records or
        those of one mechanism (by display name or, unambiguously, by config hash).

        Batch rules (multi-task peer prediction) are computed per *population*: episodes with
        the same mechanism config, profile and seed — the same agents answering many tasks.
        """

        def sel(r: EpisodeRecord) -> bool:
            return (mechanism is None or r.mechanism == mechanism) and (mechanism_hash is None or r.mechanism_hash == mechanism_hash)

        target = [r for r in self.records if sel(r)]
        others = [r for r in self.records if not sel(r)]
        new = [r.model_copy(deep=True) for r in target]
        if type(rule).batch:
            groups: dict[tuple, list[EpisodeRecord]] = {}
            for r in new:
                groups.setdefault((r.mechanism_hash, r.profile.id, r.seed), []).append(r)
            for grp in groups.values():
                ok = [r for r in grp if r.error is None]
                for r, rw in zip(ok, rule.compute_batch(ok)):  # type: ignore[attr-defined]
                    r.rewards = {k: float(v) for k, v in rw.items()}
                    r.reward_rule = rule.name
        else:
            for r in new:
                if r.error is None:
                    r.rewards = {k: float(v) for k, v in rule(r).items()}
                    r.reward_rule = rule.name
        return Results(others + new, self.tasks)

    async def aregt(self, scorers: Sequence[Any], tasks: dict[str, Task] | None = None) -> "Results":
        """Recompute ground truth (e.g. after questions resolve) — no model calls."""
        from ..ground_truth.base import compute_gt

        tasks = {**self.tasks, **(tasks or {})}
        new = []
        for r in self.records:
            r2 = r.model_copy(deep=True)
            t = tasks.get(r.task_id)
            if t is not None and r2.error is None:
                r2.gt = {}
                await compute_gt(t, r2, list(scorers))
            new.append(r2)
        return Results(new, tasks)

    def regt(self, scorers: Sequence[Any], tasks: dict[str, Task] | None = None) -> "Results":
        return run_sync(self.aregt(scorers, tasks))

    async def aresolve(
        self, tasks: dict[str, Task] | Sequence[Task], scorers: Sequence[Any] | None = None,
        rule: RewardRule | None = None,
    ) -> "Results":
        """Attach late-arriving ground truth: update tasks, recompute GT and delayed rewards."""
        from ..ground_truth.base import compute_gt

        if not isinstance(tasks, dict):
            tasks = {t.id: t for t in tasks}
        self.tasks.update(tasks)
        new = []
        for r in self.records:
            r2 = r.model_copy(deep=True)
            t = self.tasks.get(r.task_id)
            if t is not None and r2.error is None and t.resolved:
                if scorers is not None:
                    r2.gt = {}
                    await compute_gt(t, r2, list(scorers))
                if rule is not None and type(rule).delayed:
                    r2.rewards = {k: float(v) for k, v in rule.score(r2, t).items()}  # type: ignore[attr-defined]
                    r2.reward_rule = rule.name
                    r2.meta.pop("rewards_pending", None)
            new.append(r2)
        return Results(new, self.tasks)

    def resolve(self, tasks, scorers=None, rule=None) -> "Results":
        return run_sync(self.aresolve(tasks, scorers, rule))

    def summary(self) -> pd.DataFrame:
        d = self.df()
        if d.empty:
            return d
        gt_cols = [c for c in d.columns if c.startswith("gt_") and c != "gt_status"]
        agg = {"reward": "mean", **{c: "mean" for c in gt_cols}, "episode": "nunique"}
        return d.groupby(["mechanism", "role"]).agg(agg).rename(columns={"episode": "episodes"})
