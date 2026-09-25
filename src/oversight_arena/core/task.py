"""Tasks and the ground-truth firewall.

A :class:`Task` carries everything known about an instance, *including ground truth*
(`Answer.value`, `Task.gt`). Mechanisms and agents never receive a `Task`; they receive a
:class:`TaskView`, from which ground truth has been stripped. Ground truth reaches the
episode only through (a) the experiment harness binding *simulated behaviours* (e.g. "argue
for the correct answer"), and (b) explicitly declared, budgeted ground-truth channels
(audits, simulated probes, delayed resolution), whose use is recorded on the episode.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AnswerType = Literal["choice", "text", "number", "probability", "code", "sql", "proof", "artifact"]


class Answer(BaseModel):
    """A candidate answer / position. `value` is ground truth (hidden from mechanisms)."""

    id: str
    text: str
    value: float | None = None
    data: dict[str, Any] = Field(default_factory=dict)  # public structured payload (e.g. UCI move)

    def public(self) -> "AnswerView":
        return AnswerView(id=self.id, text=self.text, data=dict(self.data))


class AnswerView(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    text: str
    data: dict[str, Any] = Field(default_factory=dict)


class InfoBlock(BaseModel):
    """A named chunk of task information with an access level.

    `access="public"` blocks are shown to every role. Any other value names a *clearance*;
    only roles holding that clearance see the block (information asymmetry).
    """

    key: str
    content: str
    title: str | None = None
    access: str = "public"


class Task(BaseModel):
    """A task instance, including hidden ground truth."""

    id: str
    domain: str
    question: str
    options: list[Answer] = Field(default_factory=list)
    info: list[InfoBlock] = Field(default_factory=list)
    answer_type: AnswerType = "choice"
    data: dict[str, Any] = Field(default_factory=dict)  # public structured data
    resources: dict[str, Any] = Field(default_factory=dict)  # for trusted code (tools, verifiers)
    resource_access: dict[str, str] = Field(default_factory=dict)  # resource -> clearance (agents)
    gt: dict[str, Any] = Field(default_factory=dict)  # hidden ground-truth labels
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ---- ground-truth helpers (harness / GT scorers only) ----
    def option(self, oid: str) -> Answer:
        for o in self.options:
            if o.id == oid:
                return o
        raise KeyError(f"Task {self.id} has no option {oid!r}")

    def has_values(self) -> bool:
        return bool(self.options) and all(o.value is not None for o in self.options)

    def best_option(self) -> Answer | None:
        vals = [o for o in self.options if o.value is not None]
        return max(vals, key=lambda o: o.value) if vals else None  # type: ignore[arg-type]

    def worst_option(self) -> Answer | None:
        vals = [o for o in self.options if o.value is not None]
        return min(vals, key=lambda o: o.value) if vals else None  # type: ignore[arg-type]

    def correct_ids(self) -> list[str]:
        best = self.best_option()
        if best is None:
            return []
        return [o.id for o in self.options if o.value == best.value]

    def incorrect_ids(self) -> list[str]:
        good = set(self.correct_ids())
        return [o.id for o in self.options if o.value is not None and o.id not in good]

    def value_of(self, oid: str | None) -> float | None:
        if oid is None:
            return None
        for o in self.options:
            if o.id == oid:
                return o.value
        return None

    @property
    def resolved(self) -> bool:
        """False when ground truth is pending (e.g. unresolved forecasting question)."""
        return not self.gt.get("pending", False)

    # ---- firewall ----
    def view(self) -> "TaskView":
        return TaskView(
            id=self.id,
            domain=self.domain,
            question=self.question,
            options=[o.public() for o in self.options],
            info=[b.model_copy() for b in self.info],
            answer_type=self.answer_type,
            data=dict(self.data),
            metadata={k: v for k, v in self.metadata.items() if not str(k).startswith("_")},
        )


class TaskView(BaseModel):
    """What mechanisms (and, filtered by clearance, agents) may see. No ground truth."""

    model_config = ConfigDict(frozen=True)

    id: str
    domain: str
    question: str
    options: list[AnswerView] = Field(default_factory=list)
    info: list[InfoBlock] = Field(default_factory=list)
    answer_type: AnswerType = "choice"
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def option_ids(self) -> list[str]:
        return [o.id for o in self.options]

    def option(self, oid: str) -> AnswerView:
        for o in self.options:
            if o.id == oid:
                return o
        raise KeyError(oid)

    def visible_info(self, clearance: set[str] | frozenset[str]) -> list[InfoBlock]:
        return [b for b in self.info if b.access == "public" or b.access in clearance]
