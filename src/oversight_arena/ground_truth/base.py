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

    def fingerprint(self) -> str:
        """Identity of this scorer's configuration *and code*: stored with its values, so a
        resumed experiment recomputes ground truth whose scorer has changed."""
        from ..core.util import code_hash, stable_hash

        fns = {k: code_hash(v) for k, v in vars(self).items() if callable(v)}
        return stable_hash(self.describe(), fns, code_hash(type(self).score), length=12)


def stale_scorers(record: EpisodeRecord, scorers: list[GTScorer]) -> list[GTScorer]:
    """Scorers whose values are missing from ``record``, failed, or came from a different
    configuration or code (see :meth:`GTScorer.fingerprint`)."""
    done = record.meta.get("gt_fingerprints") or {}
    failed = record.meta.get("gt_errors") or {}
    return [s for s in scorers if s.name not in record.gt or s.name in failed or done.get(s.name) != s.fingerprint()]


async def compute_gt(task: Task, record: EpisodeRecord, scorers: list[GTScorer]) -> None:
    """Fill ``record.gt`` / ``record.gt_status`` in place (entries of other scorers are kept)."""
    if not task.resolved:
        record.gt_status = "pending"
        return
    fps = record.meta.setdefault("gt_fingerprints", {})
    for s in scorers:
        try:
            vals = await s.ascore(task, record)
        except Exception as e:  # GT failures should not kill the run; a stale value must not survive
            record.meta.setdefault("gt_errors", {})[s.name] = f"{type(e).__name__}: {e}"
            record.gt.pop(s.name, None)
            fps.pop(s.name, None)
            continue
        record.gt[s.name] = {k: (None if v is None else float(v)) for k, v in vals.items()}
        fps[s.name] = s.fingerprint()
        (record.meta.get("gt_errors") or {}).pop(s.name, None)
    any_val = any(v is not None for vals in record.gt.values() for v in vals.values())
    record.gt_status = "complete" if any_val else "none"
