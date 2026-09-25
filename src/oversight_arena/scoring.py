"""Independent post-run evaluation. Manipulation labels must be checked, not assumed."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .mechanisms import probability
from .types import Evaluation, Measurement, Run, action_from_dict


def last_action(run: Run, role: str):
    for event in reversed(run.events):
        if event.actor == role and event.kind == "action":
            return action_from_dict(event.data)
    raise ValueError(f"No action by {role}")


@dataclass
class AnswerScorer:
    answers: Mapping[str, object]
    roles: tuple[str, ...]
    field: str = "answer"
    version: str = "1"

    def __call__(self, run: Run) -> Evaluation:
        scores = {}
        for role in self.roles:
            if run.task.id not in self.answers:
                m = Measurement(None, "unavailable", "reference answer")
            elif run.status != "complete":
                m = Measurement(None, "error", "reference answer", detail="Run failed")
            else:
                try:
                    answer = last_action(run, role).data[self.field]
                    m = Measurement(
                        float(answer == self.answers[run.task.id]), "observed", "reference answer"
                    )
                except (KeyError, ValueError) as exc:
                    m = Measurement(None, "error", "reference answer", detail=str(exc))
            scores[role] = {"quality": m}
        return Evaluation(run.id, "answer", self.version, scores)


@dataclass
class ForecastScorer:
    resolutions: Mapping[str, bool]
    roles: tuple[str, ...]
    version: str = "1"

    def __call__(self, run: Run) -> Evaluation:
        scores = {}
        for role in self.roles:
            if run.status != "complete":
                m = Measurement(None, "error", "event resolution", "outcome", "Run failed")
            elif run.task.id not in self.resolutions:
                m = Measurement(None, "pending", "event resolution", "outcome")
            else:
                try:
                    outcome = self.resolutions[run.task.id]
                    if not isinstance(outcome, bool):
                        raise ValueError("Forecast resolutions must be boolean")
                    p = probability(last_action(run, role).data["probability"])
                    m = Measurement(
                        -2 * (p - outcome) ** 2, "observed", "event resolution", "outcome"
                    )
                except (KeyError, ValueError) as exc:
                    m = Measurement(None, "error", "event resolution", "outcome", str(exc))
            scores[role] = {"brier": m}
        return Evaluation(run.id, "forecast", self.version, scores)


@dataclass
class CallbackScorer:
    """Use hidden tests, kernel checks, human audits or expensive model assessments.

    Return per-role, per-dimension Measurements with exact/proxy/human/outcome provenance.
    Exceptions remain errors, not negative quality labels.
    """

    name: str
    version: str
    callback: Callable[[Run], dict[str, dict[str, Measurement]]]

    def __call__(self, run: Run) -> Evaluation:
        return Evaluation(run.id, self.name, self.version, self.callback(run))
