"""Synthetic forecasting questions with a known latent probability and a delayed outcome."""

from __future__ import annotations

from typing import ClassVar

from ..core.task import Task
from ..core.util import rng_for
from .base import Domain
from .forecasting import default_forecast_gt, forecast_task


class SyntheticForecasting(Domain):
    name: ClassVar[str] = "synthetic_forecasting"
    n_tasks: int = 60
    resolved: bool = False  # False: outcomes pending (release now, resolve later)
    signal_noise: float = 0.3
    expert_clearance: list[str] = ["signal"]
    judge_clearance: list[str] = []

    def load(self) -> list[Task]:
        import math

        out = []
        for k in range(self.n_tasks):
            rng = rng_for("synth-fc", self.seed, k)
            p = rng.betavariate(0.7, 0.7)
            y = float(rng.random() < p)
            z = math.log(max(p, 1e-4) / max(1 - p, 1e-4)) + rng.gauss(0, self.signal_noise)
            sig = 1 / (1 + math.exp(-z))
            t = forecast_task(f"q{k}", f"Synthetic event #{k}: will it happen?", y if self.resolved else None,
                              source=f"synth{self.seed}")
            t = t.model_copy(update={
                "resources": {"signal": round(sig, 3)}, "resource_access": {"signal": "signal"},
                "gt": {**t.gt, "_p": p} if self.resolved else t.gt,
            })
            out.append(t)
        return out

    def outcomes(self) -> list[Task]:
        """The same questions, resolved (what arrives 'later')."""
        return SyntheticForecasting(n_tasks=self.n_tasks, resolved=True, seed=self.seed,
                                    signal_noise=self.signal_noise).tasks()

    def gt_scorers(self):  # type: ignore[override]
        return default_forecast_gt()
