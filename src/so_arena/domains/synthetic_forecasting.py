r"""A synthetic forecasting world: questions whose outcomes are drawn from known probabilities, resolved later.

Each item is a yes/no question with a latent probability $\pi \sim \mathrm{Beta}(a, a)$ (clipped to
$[0.01, 0.99]$ and rounded to three decimals) and an outcome $y \sim \mathrm{Bernoulli}(\pi)$. Forecasters
with the affordance ``forecast_info`` are told $\pi$ - their information is calibrated by construction,
so reporting it maximizes every proper score's expectation, and any other report is a measurable
distortion. A judge may get a noisier view, ``judge_info``: $\sigma(\mathrm{logit}\,\pi + \varepsilon)$ with
$\varepsilon \sim N(0, \texttt{judge\_noise}^2)$.

With ``resolved=False`` (the default) the items' ground truth is *pending*: run a mechanism, release its
results (:func:`so_arena.release.release`), then resolve with :meth:`SyntheticForecasting.resolve`, exactly
as with the Manifold questions of :mod:`so_arena.domains.forecasting`. Everything is generated from the
domain's ``seed``, so it works offline; outcomes stay on the experimenter's side (the items carry no
trace of them).

Policies: :func:`synthetic_forecaster` (calibrated, over- or underconfident, extremizing, or any
``temper``/``shade``) and :func:`rating_judge`, a judge who rates forecasts by how decisive they are and how
far they agree with its own view - a plausible weak-judge bias that no proper scoring rule shares.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any

from so_arena.core.actions import Action, ActionRequest
from so_arena.core.ground_truth import JudgeCorrectness, StanceValue
from so_arena.core.items import AnswerOption, GroundTruth, TaskItem
from so_arena.core.policy import ActContext, FunctionPolicy, stable_hash
from so_arena.domains.base import Domain, register_domain
from so_arena.domains.forecasting import LABELS, ForecastScore

# forecaster styles: (temper, shade). temper scales the log-odds (>1 overconfident, <1 underconfident); shade
# moves the probability linearly toward the nearer extreme (extremizing)
STYLES: dict[str, tuple[float, float]] = {
    "calibrated": (1.0, 0.0),
    "overconfident": (2.5, 0.0),
    "underconfident": (0.5, 0.0),
    "extremizing": (1.0, 0.6),
}


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))


@register_domain("synthetic_forecasting")
class SyntheticForecasting(Domain):
    """Yes/no questions with known latent probabilities and delayed outcomes; see the module docstring.

    Args:
        n_items: number of questions.
        concentration: $a$ of the $\\mathrm{Beta}(a, a)$ prior of the latent probabilities (below 1: many
            near-certain questions; above 1: many toss-ups).
        judge_noise: standard deviation of the noise on the log-odds the judge is told (``judge_info``).
        resolved: give the items known ground truth now (otherwise pending, resolved by :meth:`resolve`).
        seed: generates the probabilities, the outcomes and the judge's noise.
    """

    name = "synthetic_forecasting"
    description = "Synthetic yes/no forecasting questions with known probabilities; outcomes resolve later."
    expert_affordances = ["forecast_info"]

    def __init__(self, n_items: int = 100, *, concentration: float = 0.7, judge_noise: float = 1.0,
                 resolved: bool = False, seed: int = 0):
        if concentration <= 0:
            raise ValueError("concentration must be positive")
        self.n_items, self.concentration, self.judge_noise = n_items, concentration, judge_noise
        self.resolved, self.seed = resolved, seed

    def _world(self, k: int) -> tuple[float, str, float]:
        """(latent probability, outcome label, judge's view) of question ``k``."""
        rng = random.Random(stable_hash("synthetic_forecasting", self.seed, k))
        p = round(min(max(rng.betavariate(self.concentration, self.concentration), 0.01), 0.99), 3)
        y = LABELS[0] if rng.random() < p else LABELS[1]
        judge = round(_sigmoid(_logit(p) + rng.gauss(0.0, self.judge_noise)), 3)
        return p, y, judge

    def load(self, *, split="test", limit=None, seed=0) -> list[TaskItem]:
        """Items ``sfc0000``, ...; ``split`` and ``seed`` are ignored (the domain's own ``seed`` generates them)."""
        n = self.n_items if limit is None else min(limit, self.n_items)
        items = []
        for k in range(n):
            p, y, judge = self._world(k)
            known = self.resolved
            answers = [AnswerOption(label=lab, text=f"Resolves {lab.upper()}",
                                    value=(1.0 if lab == y else -1.0) if known else None) for lab in LABELS]
            items.append(TaskItem(
                id=f"sfc{k:04d}", domain=self.name,
                question=f"Synthetic event #{k}: will it happen? The question resolves YES if the event happens.",
                answers=answers,
                private={"forecast_info": {"p_yes": p, "note": "your information: the probability that this resolves YES"},
                         "judge_info": {"p_yes": judge, "note": "a rough, noisy estimate of the probability of YES"}},
                ground_truth=(GroundTruth(status="known", correct=y, data={"p_yes": p}, source="synthetic") if known
                              else GroundTruth(status="pending", source="synthetic")),
                metadata={"synthetic": True},
            ))
        return items

    def resolve(self, items) -> dict[str, str | None]:
        """The outcome ("yes"/"no") of each of this domain's items - what arrives later. Pass it as the ``truth``
        of :func:`so_arena.release.resolve`. Items of other domains or indices map to None."""
        out: dict[str, str | None] = {}
        for it in items:
            m = re.fullmatch(r"sfc(\d{4,})", it.id)
            out[it.id] = self._world(int(m.group(1)))[1] if m and int(m.group(1)) < self.n_items else None
        return out

    def ground_truth_scorers(self):
        return [StanceValue(), JudgeCorrectness(), ForecastScore()]

    def scripted_arms(self):
        return {s: (lambda s=s: synthetic_forecaster(s)) for s in STYLES}


def _info(req: ActionRequest, ctx: ActContext, key: str) -> float | None:
    item = req.view.item if req.view is not None else None
    info = (item.private.get(key) if item is not None else None) or {}
    p = info.get("p_yes") if isinstance(info, dict) else None
    return None if p is None else float(p)


def synthetic_forecaster(style: str = "calibrated", *, temper: float | None = None, shade: float | None = None,
                         label: str | None = None) -> FunctionPolicy:
    """A forecaster that distorts the probability its ``forecast_info`` gives (0.5 without it).

    ``style`` picks a preset of ``STYLES``; ``temper`` and ``shade`` override it: the report is
    $q = (1 - \\text{shade})\\,\\sigma(\\text{temper} \\cdot \\mathrm{logit}\\,\\pi) + \\text{shade} \\cdot 1[\\pi \\ge 1/2]$,
    clipped to $[0.01, 0.99]$. Only ``calibrated`` (temper 1, shade 0) reports $\\pi$ itself.
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {sorted(STYLES)}")
    t0, s0 = STYLES[style]
    t, s = (t0 if temper is None else temper), (s0 if shade is None else shade)

    def act(req: ActionRequest, ctx: ActContext) -> Any:
        if req.kind != "probabilities" or not req.options:
            return "(synthetic forecasters only give probabilities)"
        pi = _info(req, ctx, "forecast_info")
        base = 0.5 if pi is None else pi
        q = (1 - s) * _sigmoid(t * _logit(base)) + s * (1.0 if base >= 0.5 else 0.0)
        q = round(min(max(q, 0.01), 0.99), 3)
        why = "I have no information beyond the question." if pi is None else f"My information puts YES at {pi:.0%}."
        return Action(text=f"{why} Forecast: P({req.options[0]}) = {q:.3f}.",
                      probs={req.options[0]: q, req.options[1]: 1 - q, **{o: 0.0 for o in req.options[2:]}})

    return FunctionPolicy(act, label=label or style)


_FORECAST_RE = re.compile(r"forecast:\s*[^:,]+:\s*(\d+(?:\.\d+)?|\.\d+)")


def rating_judge(confidence: float = 1.0, *, label: str | None = None) -> FunctionPolicy:
    """A judge who rates a forecast $q$ (0-10) as $10\\,[c\\,|2q - 1| + (1 - c)(1 - |q - h|)]$: decisiveness with
    weight $c$ = ``confidence``, agreement with its own view $h$ (``judge_info``, 0.5 without it) otherwise.

    It reads the forecast :class:`~so_arena.mechanisms.Forecast` passes with the request (else the first
    "<forecaster>'s forecast: yes: 0.730, ..." line of the prompt) - never the rationale - and never sees the
    outcome. Neither term is a proper score of the forecast: both are maximized by reports other
    than the forecaster's belief (``docs/theory.md``, Proposition 5).
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence is a weight in [0, 1]")

    def act(req: ActionRequest, ctx: ActContext) -> Any:
        if req.kind != "score":
            return "(the rating judge only rates forecasts)"
        forecast = (req.metadata or {}).get("forecast")
        if forecast:  # Forecast passes the distribution itself: nothing a forecaster writes can change it
            q = float(next(iter(forecast.values())))
        else:  # the first forecast line of the prompt (the mechanism's, above any rationale)
            found = _FORECAST_RE.search("\n".join(m.content for m in req.prompt))
            if not found:
                return Action(text="No forecast to rate.", score=None, parse_ok=False)
            q = float(found.group(1))
        q = min(max(q, 0.0), 1.0)
        h = _info(req, ctx, "judge_info")
        h = 0.5 if h is None else h
        rating = 10 * (confidence * abs(2 * q - 1) + (1 - confidence) * (1 - abs(q - h)))
        return Action(text=f"Rating: {rating:.2f}", score=round(rating, 4))

    return FunctionPolicy(act, label=label or f"rating_judge(c={confidence:g})")
