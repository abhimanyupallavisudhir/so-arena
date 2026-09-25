"""Basic value types shared across the library: chat messages, completions, token usage."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """A chat message. Deliberately provider-agnostic (converted to Inspect/OpenAI formats in backends)."""

    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role="assistant", content=content)


class TopLogprob(BaseModel):
    token: str
    logprob: float


class TokenLogprob(BaseModel):
    token: str
    logprob: float
    top: list[TopLogprob] = Field(default_factory=list)


class Usage(BaseModel):
    """Token and cost accounting. Addable, so it can be accumulated per role / per run.

    Token convention (Inspect's): ``input_tokens`` are fresh (uncached) input tokens,
    ``cached_input_tokens`` those read from a provider's prompt cache, and ``output_tokens`` include
    any reasoning tokens (``reasoning_tokens`` breaks them out; it is not added again).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    cached_calls: int = 0
    cost_usd: float = 0.0
    # Human-judge time or other non-token oversight effort, in seconds.
    effort_seconds: float = 0.0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cached_input_tokens=self.cached_input_tokens + other.cached_input_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=self.calls + other.calls,
            cached_calls=self.cached_calls + other.cached_calls,
            cost_usd=self.cost_usd + other.cost_usd,
            effort_seconds=self.effort_seconds + other.effort_seconds,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class Completion(BaseModel):
    """Result of one model call."""

    text: str
    model: str
    reasoning: str | None = None
    logprobs: list[TokenLogprob] | None = None
    usage: Usage = Field(default_factory=Usage)
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateOptions(BaseModel):
    """Sampling options passed to a model backend. Hashable (used in cache keys)."""

    model_config = ConfigDict(frozen=True)

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    stop: tuple[str, ...] | None = None
    reasoning_effort: str | None = None

    def merged(self, **overrides: Any) -> "GenerateOptions":
        data = self.model_dump()
        data.update({k: v for k, v in overrides.items() if v is not None})
        return GenerateOptions(**data)
