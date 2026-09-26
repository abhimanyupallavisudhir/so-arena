"""Typed requests and actions exchanged between mechanisms and policies.

A mechanism never calls a model directly. At each decision point it sends the acting role an
:class:`ActionRequest` (what kind of action is wanted, a default prompt, and a structured
:class:`GameView` of what the role may see). The role's policy - an LLM, a script, a human, an
engine, a best-of-N wrapper, an RL trainee - returns an :class:`Action`. This keeps mechanisms
independent of how behaviour is produced, which is what lets the same mechanism be evaluated with
prompted, sampled, prompt-optimized or trained agents.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from so_arena.core.items import TaskItem
from so_arena.core.types import Message, Usage

ActionKind = Literal["text", "choice", "probabilities", "score", "json"]


class TurnView(BaseModel):
    """A transcript turn as seen by a particular role (after visibility filtering)."""

    index: int
    role: str
    phase: str = ""
    text: str
    reasoning: str | None = None  # only present if the viewer may see this role's reasoning


class GameView(BaseModel):
    """Everything a role is allowed to observe at a decision point."""

    role: str
    item: TaskItem  # censored; ``private`` filtered to the role's affordances
    transcript: list[TurnView] = Field(default_factory=list)
    stance: str | None = None  # the answer this role is assigned / has committed to argue for
    positions: dict[str, str | None] = Field(default_factory=dict)  # public positions of roles
    round: int = 0


class ActionRequest(BaseModel):
    kind: ActionKind = "text"
    # Default rendering of the decision point as chat messages (system + user). LLM policies use
    # it directly (adding their strategy and output-format instructions); other policies may build
    # their own view from ``view``.
    prompt: list[Message] = Field(default_factory=list)
    view: GameView | None = None
    phase: str = ""
    options: list[str] | None = None  # labels, for choice / probabilities
    option_texts: dict[str, str] | None = None
    score_range: tuple[float, float] | None = None
    score_meaning: str | None = None  # e.g. "suspicion that the agent is sabotaging"
    word_limit: int | None = None
    json_keys: list[str] | None = None  # for kind="json": keys expected in the returned object
    allow_tools: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    text: str = ""
    reasoning: str | None = None  # private chain of thought (visible only to allowed roles)
    choice: str | None = None
    probs: dict[str, float] | None = None
    score: float | None = None
    data: dict[str, Any] | None = None
    usage: Usage = Field(default_factory=Usage)
    parse_ok: bool = True
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def summary(self) -> str:
        if self.probs is not None:
            return ", ".join(f"{k}: {v:.3f}" for k, v in self.probs.items())
        if self.choice is not None:
            return self.choice
        if self.score is not None:
            return f"{self.score:g}"
        return self.text
