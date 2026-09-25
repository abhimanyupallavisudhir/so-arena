"""Probes: detectors that score a role's behaviour, for monitoring mechanisms.

- :class:`~oversight_arena.channels.gt_channels.SimulatedProbe` simulates a detector of a given
  AUROC *from ground truth* — a stand-in for capabilities we may not (yet) have, e.g. a good
  deception probe. Its use is logged as ground-truth access.
- :class:`FunctionProbe` wraps a *real* detector: any function of what the role produced — its
  transcript entries, including private reasoning and anything the agent recorded in
  ``entry.data`` (e.g. activations or token logprobs from an open-weights model; keys starting
  with ``_`` are hidden from other agents but visible to probes, which are trusted code with
  white-box access). Not ground truth.

Both implement ``async query(ctx, role) -> float | None`` (suspicion in [0, 1]).
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.util import clamp

if TYPE_CHECKING:  # pragma: no cover
    from ..mechanisms.base import EpisodeContext


class Probe(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    name: str = "probe"

    @abstractmethod
    async def query(self, ctx: "EpisodeContext", role: str | None = None, **kw: Any) -> float | None: ...


class FunctionProbe(Probe):
    """``fn(entries) -> float | None`` over the role's own transcript entries (sync or async)."""

    name: str = "function_probe"
    fn: Callable[[list[Any]], Any] = Field(exclude=True)

    async def query(self, ctx: "EpisodeContext", role: str | None = None, **kw: Any) -> float | None:
        assert role is not None
        entries = [e for e in ctx.transcript.entries if e.role == role]
        v = self.fn(entries)
        if inspect.isawaitable(v):
            v = await v
        return None if v is None else clamp(float(v), 0.0, 1.0)

    def describe(self) -> dict[str, Any]:
        from ..core.util import code_hash

        return {"type": "FunctionProbe", "name": self.name, "fn": code_hash(self.fn)}


__all__ = ["Probe", "FunctionProbe"]
