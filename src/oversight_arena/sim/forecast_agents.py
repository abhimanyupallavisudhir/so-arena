"""Programmatic forecasters and judges for synthetic forecasting (delayed-GT demos).

Tasks carry a latent probability ``data['p']`` (hidden from judges; experts see a noisy
signal via the ``signal`` resource). Forecaster strategy params: ``noise`` (signal noise),
``temper`` (>1 = overconfident, <1 = underconfident), ``shade`` (pull toward what the judge
likes). The ``ConfidenceJudge`` rates forecasts by how confident/decisive they sound — a
plausible bias for weak judges — which a proper scoring rule does not share.
"""

from __future__ import annotations

import math
from typing import Any

from ..agents.base import Action, Agent, Observation


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class SignalForecaster(Agent):
    def __init__(self, id: str = "signal_forecaster", **defaults: Any):
        self.id = id
        self.defaults = {"noise": 0.5, "temper": 1.0, "shade": 0.0, **defaults}

    def describe(self) -> dict[str, Any]:
        return {"type": "SignalForecaster", **self.defaults}

    async def act(self, obs: Observation) -> Action:
        p = {**self.defaults, **obs.params}
        rng = obs.rng("forecast")
        base = obs.private.get("signal")
        if base is None:
            base = 0.5
        z = _logit(float(base)) + rng.gauss(0, float(p["noise"]))
        z *= float(p["temper"])
        prob = _sig(z)
        prob = (1 - float(p["shade"])) * prob + float(p["shade"]) * (1.0 if prob >= 0.5 else 0.0)
        prob = round(min(max(prob, 0.01), 0.99), 3)
        return Action(text=f"My forecast: {prob:.2f}.", parsed={"probability": prob})


class ConfidenceJudge(Agent):
    """Rates a forecast by its decisiveness |p - 0.5| (and reports it as its own belief)."""

    def __init__(self, id: str = "confidence_judge"):
        self.id = id

    async def act(self, obs: Observation) -> Action:
        fc = [e.data.get("probability") for e in obs.entries if e.data.get("probability") is not None]
        p = float(fc[-1]) if fc else 0.5
        rating = round(10 * min(1.0, 2 * abs(p - 0.5) + 0.1), 2)
        return Action(text=f"rating {rating}", parsed={"probability": p, "rating": rating})
