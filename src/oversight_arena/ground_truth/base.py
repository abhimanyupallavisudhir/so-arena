"""Ground-truth scorers: the *evaluation* side, never visible to mechanisms.

A GT scorer maps (full task, episode record) to per-role scores ("how good was this role's
behaviour, really?") and/or principal-level scores under the key ``"_outcome"``. Incentive
compatibility = alignment between mechanism rewards and these scores.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..core.episode import EpisodeRecord
from ..core.task import Task

OUTCOME = "_outcome"


class GTScorer(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: str = "gt"
    expensive: bool = False  # e.g. human labels / strong-model labels (tracked in reports)

    @abstractmethod
    def score(self, task: Task, record: EpisodeRecord) -> dict[str, float | None]: ...

    async def ascore(self, task: Task, record: EpisodeRecord) -> dict[str, float | None]:
        return self.score(task, record)

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, **self.model_dump()}


async def compute_gt(task: Task, record: EpisodeRecord, scorers: list[GTScorer]) -> None:
    """Fill ``record.gt`` / ``record.gt_status`` in place."""
    if not task.resolved:
        record.gt_status = "pending"
        return
    any_val = False
    for s in scorers:
        try:
            vals = await s.ascore(task, record)
        except Exception as e:  # GT failures should not kill the run
            record.meta.setdefault("gt_errors", {})[s.name] = f"{type(e).__name__}: {e}"
            continue
        record.gt[s.name] = {k: (None if v is None else float(v)) for k, v in vals.items()}
        any_val = any_val or any(v is not None for v in vals.values())
    record.gt_status = "complete" if any_val else "none"
