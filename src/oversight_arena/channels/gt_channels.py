"""Declared, budgeted ground-truth channels available to mechanisms.

Scalable oversight mechanisms should not see ground truth — but realistic mechanisms may use
*some*: random audits (costly state verification), delayed resolution (forecasting),
or tools whose quality we want to *simulate* (e.g. an interpretability probe with a given
AUROC). Such access must go through a :class:`GTChannel`; every use is logged on the episode
(``record.channels``) so analyses can account for how much ground truth a mechanism consumed.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from statistics import NormalDist
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ..core.util import rng_for

if TYPE_CHECKING:  # pragma: no cover
    from ..mechanisms.base import EpisodeContext


class GTChannel(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    name: str = "gt_channel"
    cost: float = 1.0

    @abstractmethod
    async def query(self, ctx: "EpisodeContext", role: str | None = None, **kw: Any) -> Any: ...


def role_is_bad(ctx: "EpisodeContext", role: str, signal: str = "auto") -> bool | None:
    """Ground-truth 'badness' of a role's behaviour, for audits/probes.

    ``signal``: ``"stance"`` (argued an incorrect option), ``"tag:<name>"`` (strategy tag is
    falsy, e.g. ``tag:honest``), ``"env"`` (environment GT log flags the role), or ``"auto"``.
    """
    task = ctx._task_full
    b = ctx.bound.get(role)
    if signal.startswith("tag:"):
        tag = signal[4:]
        if b is None or tag not in b.tags:
            return None
        return not bool(b.tags[tag])
    if signal in ("env", "auto") and ctx.env is not None:
        flags = getattr(ctx.env, "bad_roles", None)
        if callable(flags):
            res = flags()
            if res is not None:
                return role in res
    if signal in ("stance", "auto"):
        target = None
        pos = ctx.outcome.get("positions", {}).get(role) if isinstance(ctx.outcome.get("positions"), dict) else None
        target = pos or (b.target if b else None)
        if target is not None and task.has_values():
            return target not in task.correct_ids()
    if signal == "auto" and b is not None and "honest" in b.tags:
        return not bool(b.tags["honest"])
    return None


class Auditor(GTChannel):
    """Random audit: with probability ``prob`` reveal whether ``role`` behaved badly."""

    name: str = "audit"
    prob: float = 0.1
    signal: str = "auto"
    cost: float = 10.0

    async def query(self, ctx: "EpisodeContext", role: str | None = None, **kw: Any) -> dict[str, Any]:
        assert role is not None
        rng = rng_for("audit", ctx.episode_key, role, ctx.transcript_len())
        if rng.random() >= self.prob:
            return {"audited": False, "bad": None}
        bad = role_is_bad(ctx, role, self.signal)
        ctx.log_channel(self.name, role, self.cost, {"bad": bad})
        return {"audited": True, "bad": bad}


class SimulatedProbe(GTChannel):
    """A simulated detector (e.g. an interpretability probe) with a target AUROC.

    Scores are Gaussian: honest $\\sim N(0,1)$, bad $\\sim N(d', 1)$ with
    $d' = \\sqrt{2}\\,\\Phi^{-1}(\\mathrm{AUROC})$; returned as $\\Phi(z - d'/2) \\in (0,1)$.
    It *simulates a capability we may not have*, so its use is logged as GT access.
    """

    name: str = "probe"
    auroc: float = 0.8
    signal: str = "auto"
    cost: float = 0.0

    async def query(self, ctx: "EpisodeContext", role: str | None = None, **kw: Any) -> float | None:
        assert role is not None
        bad = role_is_bad(ctx, role, self.signal)
        if bad is None:
            return None
        nd = NormalDist()
        dprime = math.sqrt(2) * nd.inv_cdf(min(max(self.auroc, 1e-6), 1 - 1e-6))
        rng = rng_for("probe", ctx.episode_key, role, kw.get("step"), ctx.transcript_len())
        z = rng.gauss(dprime if bad else 0.0, 1.0)
        score = nd.cdf(z - dprime / 2)
        ctx.log_channel(self.name, role, self.cost, {"score": score})
        return score


class Resolution(GTChannel):
    """Delayed ground truth (e.g. a forecasting question's resolution).

    Returns the resolved value if available, else ``None``. Rewards that depend on it are
    *delayed rewards*; see :class:`~oversight_arena.mechanisms.forecasting.ProperScoring`.
    """

    name: str = "resolution"
    cost: float = 0.0
    key: str = "outcome"

    async def query(self, ctx: "EpisodeContext", role: str | None = None, **kw: Any) -> Any:
        task = ctx._task_full
        if task.gt.get("pending"):
            return None
        val = task.gt.get(self.key)
        ctx.log_channel(self.name, role, self.cost, {"value": val})
        return val


class Label(GTChannel):
    """Ground-truth labels for a random fraction of tasks (e.g. easy/verified items used to train a
    weak judge — the weak-to-strong / sandwiching setup). Returns the correct option ids or None."""

    name: str = "label"
    prob: float = 1.0
    cost: float = 1.0

    async def query(self, ctx: "EpisodeContext", role: str | None = None, **kw: Any) -> list[str] | None:
        if rng_for("label", ctx.task.id).random() >= self.prob:
            return None
        ids = ctx._task_full.correct_ids()
        ctx.log_channel(self.name, role, self.cost, {"label": ids})
        return ids or None
