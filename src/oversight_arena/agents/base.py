"""Agent interface: observations in, actions out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..core.task import TaskView
from ..core.tools import Tool
from ..core.transcript import Entry
from ..core.types import ChatMessage, ToolTrace, Usage

ResponseKind = Literal["text", "choice", "distribution", "scalar", "json"]


class ResponseSpec(BaseModel):
    """What kind of response a mechanism expects from a role at a given step."""

    kind: ResponseKind = "text"
    options: list[str] | None = None  # option ids (choice/distribution)
    option_texts: dict[str, str] | None = None
    lo: float = 0.0  # scalar range
    hi: float = 1.0
    scalar_name: str = "value"
    fields: dict[str, str] | None = None  # json: field -> description
    max_words: int | None = None
    elicitation: Literal["verbal", "logprobs", "sample"] = "verbal"  # for distributions
    n_samples: int = 8
    reasoning: bool = True  # allow free-text reasoning before the final answer

    @classmethod
    def text(cls, max_words: int | None = None) -> "ResponseSpec":
        return cls(kind="text", max_words=max_words)

    @classmethod
    def choice(cls, options: list[str], texts: dict[str, str] | None = None) -> "ResponseSpec":
        return cls(kind="choice", options=options, option_texts=texts)

    @classmethod
    def distribution(
        cls, options: list[str], texts: dict[str, str] | None = None, elicitation: str = "verbal", **kw: Any
    ) -> "ResponseSpec":
        return cls(kind="distribution", options=options, option_texts=texts, elicitation=elicitation, **kw)  # type: ignore[arg-type]

    @classmethod
    def scalar(cls, lo: float = 0.0, hi: float = 1.0, name: str = "value") -> "ResponseSpec":
        return cls(kind="scalar", lo=lo, hi=hi, scalar_name=name)

    @classmethod
    def json(cls, fields: dict[str, str]) -> "ResponseSpec":
        return cls(kind="json", fields=fields)


class Observation(BaseModel):
    """Everything a role gets to see when asked to act."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    role: str
    role_title: str
    role_kind: str = "agent"
    mechanism: str = ""
    brief: str = ""  # rules of the game (incl. incentives) for this role
    strategy: str = ""  # private strategy instructions (bound)
    target: str | None = None  # assigned option id, if any
    target_text: str = ""
    task: TaskView
    task_text: str = ""  # rendered, clearance-filtered task
    private: dict[str, Any] = Field(default_factory=dict)  # clearance-filtered structured data
    entries: list[Entry] = Field(default_factory=list)  # visible transcript (structured)
    transcript_text: str = ""  # visible transcript (rendered)
    prompt: str = ""  # instruction for this turn
    response: ResponseSpec = Field(default_factory=ResponseSpec)
    tools: list[Tool] = Field(default_factory=list)
    claim_help: str = ""  # how to make verifiable claims (from the evidence policy)
    seed: int = 0
    turn: int | None = None
    step: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)  # strategy params (temperature, ...)
    tags: dict[str, Any] = Field(default_factory=dict)  # strategy tags (for programmatic agents)

    def rng(self, *extra: Any) -> Any:
        """Deterministic RNG for programmatic agents: varies with task, role, sample seed,
        turn and step (and ``extra``), so behaviour is reproducible yet independent across
        tasks and roles."""
        from ..core.util import rng_for

        return rng_for("agent-rng", self.task.id, self.role, self.seed, self.turn, self.step, *extra)


class Action(BaseModel):
    text: str = ""
    parsed: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    tool_trace: list[ToolTrace] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    messages: list[ChatMessage] | None = Field(default=None, exclude=True)
    error: str | None = None


class Agent(ABC):
    """A policy for a role. Subclasses implement :meth:`act`."""

    id: str = "agent"

    @abstractmethod
    async def act(self, obs: Observation) -> Action: ...

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "id": self.id}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.id!r})"
