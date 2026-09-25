"""Backend-agnostic message and usage types shared by models, agents and transcripts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token / call / cost accounting. Addable."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "Usage | None") -> "Usage":
        if other is None:
            return self.model_copy()
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            calls=self.calls + other.calls,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    """A chat message in provider-neutral form."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # tool name for tool results
    reasoning: str | None = None

    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> "ChatMessage":
        return cls(role="assistant", content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str | None = None) -> "ChatMessage":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


class TokenLogprob(BaseModel):
    token: str
    logprob: float
    top: list[tuple[str, float]] = Field(default_factory=list)


class ToolTrace(BaseModel):
    """Record of one tool invocation made by an agent during a turn."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    error: str | None = None
    trusted: bool = True  # executed by trusted code (so its output can serve as verified evidence)
