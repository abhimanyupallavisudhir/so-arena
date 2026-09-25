"""Task items: the problem instances a mechanism is run on.

Design rule: ground truth lives in ``TaskItem.ground_truth`` (and in ``AnswerOption.value``) and is
*never* shown to mechanisms or agents. The runner passes ``item.censored()`` to mechanisms; only
ground-truth scorers (the experimenter's measurement layer) see the uncensored item. Anything a
mechanism is legitimately allowed to use (a hidden passage, a private database, visible tests, a
budgeted engine) belongs in ``TaskItem.private`` and is exposed per-role via affordances, or via
verifiers/tools.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnswerOption(BaseModel):
    """A candidate answer ("answer case" in SOlib terms)."""

    label: str
    text: str
    # Ground-truth value of arguing for / selecting this answer. +1/-1 reproduces binary ASD;
    # graded values give graded ASD. Hidden from mechanisms.
    value: float | None = None

    def render(self) -> str:
        return f"({self.label}) {self.text}" if self.text != self.label else f"({self.label})"


GroundTruthStatus = Literal["known", "pending", "unknown"]


class GroundTruth(BaseModel):
    """Experimenter-side ground truth for an item.

    ``status="pending"`` marks ground truth that will arrive later (e.g. unresolved forecasting
    questions); ``"unknown"`` marks items where it may never arrive. Metrics that need ground truth
    skip such items, and results can be released and resolved later (see ``so_arena.release``).
    """

    status: GroundTruthStatus = "known"
    correct: str | None = None
    values: dict[str, float] | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    resolve_after: _dt.datetime | None = None
    source: str | None = None


class TaskItem(BaseModel):
    """One problem instance."""

    id: str
    domain: str = "generic"
    question: str
    answers: list[AnswerOption] | None = None
    # Public context visible to all roles (e.g. a FEN, a schema, background information).
    context: dict[str, Any] = Field(default_factory=dict)
    # Private information keyed by affordance name; a role sees ``private[k]`` iff ``k`` is in its
    # RoleSpec.affordances. Verifiers and tools may also read it.
    private: dict[str, Any] = Field(default_factory=dict)
    ground_truth: GroundTruth | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ------------------------------------------------------------------ answer helpers
    @property
    def labels(self) -> list[str]:
        return [a.label for a in self.answers or []]

    def answer(self, label: str) -> AnswerOption:
        for a in self.answers or []:
            if a.label == label:
                return a
        raise KeyError(f"item {self.id!r} has no answer {label!r}")

    def value_of(self, label: str | None) -> float | None:
        """Ground-truth value of an answer label (None if unknown)."""
        if label is None:
            return None
        gt = self.ground_truth
        if gt is not None and gt.values and label in gt.values:
            return gt.values[label]
        try:
            v = self.answer(label).value
        except KeyError:
            return None
        if v is not None:
            return v
        if gt is not None and gt.correct is not None and gt.status == "known":
            return 1.0 if label == gt.correct else -1.0
        return None

    @property
    def has_ground_truth(self) -> bool:
        gt = self.ground_truth
        if gt is None:
            return any(a.value is not None for a in self.answers or [])
        return gt.status == "known"

    @property
    def true_label(self) -> str | None:
        gt = self.ground_truth
        if gt is not None and gt.correct is not None:
            return gt.correct
        vals = [(a.value, a.label) for a in self.answers or [] if a.value is not None]
        if not vals:
            return None
        return max(vals)[1]

    @property
    def false_labels(self) -> list[str]:
        t = self.true_label
        return [lab for lab in self.labels if lab != t]

    def censored(self, *, keep_metadata: bool = False) -> "TaskItem":
        """Copy with ground truth (and experimenter-side metadata) removed: what mechanisms and agents receive.

        Public structured information belongs in ``context``; ``metadata`` is for the experimenter
        (e.g. which mutation produced a wrong answer, a market id used to fetch the resolution) and is
        not shown to mechanisms. Releases keep metadata (``keep_metadata=True``) so they can be resolved.
        """
        return self.model_copy(
            update={
                "ground_truth": None,
                "metadata": dict(self.metadata) if keep_metadata else {},
                "answers": (
                    [a.model_copy(update={"value": None}) for a in self.answers]
                    if self.answers is not None
                    else None
                ),
            },
            deep=True,
        )

    def fingerprint(self) -> str:
        """Hash of what mechanisms receive (the censored item): content changes change it, while
        resolving the item (ground truth, answer values) or editing its metadata does not."""
        payload = self.censored().model_dump(mode="json", exclude={"ground_truth", "metadata"})
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def render_question(self, *, with_options: bool = True) -> str:
        parts = [self.question.strip()]
        if with_options and self.answers:
            parts.append("Answer options:\n" + "\n".join(a.render() for a in self.answers))
        return "\n\n".join(parts)


def binary_item(
    id: str,
    question: str,
    correct: str,
    incorrect: str,
    *,
    domain: str = "generic",
    shuffle_seed: int | None = None,
    **kwargs: Any,
) -> TaskItem:
    """Convenience constructor for a two-answer item with +1/-1 values (the classic ASD setup)."""
    opts = [("correct", correct), ("incorrect", incorrect)]
    if shuffle_seed is not None:
        import random

        random.Random(shuffle_seed).shuffle(opts)
    answers = []
    correct_label = None
    for label, (kind, text) in zip("AB", opts):
        answers.append(AnswerOption(label=label, text=text, value=1.0 if kind == "correct" else -1.0))
        if kind == "correct":
            correct_label = label
    return TaskItem(
        id=id,
        domain=domain,
        question=question,
        answers=answers,
        ground_truth=GroundTruth(correct=correct_label),
        **kwargs,
    )
