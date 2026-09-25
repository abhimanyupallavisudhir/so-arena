"""Provider-neutral model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ..core.types import ChatMessage, TokenLogprob, ToolCall, Usage


class GenConfig(BaseModel):
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    reasoning_effort: str | None = None
    reasoning_tokens: int | None = None
    seed: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def merged(self, **kw: Any) -> "GenConfig":
        data = self.model_dump()
        data.update({k: v for k, v in kw.items() if v is not None})
        return GenConfig(**data)


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )


class ModelOutput(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str | None = None
    logprobs: list[TokenLogprob] | None = None
    usage: Usage = Field(default_factory=Usage)
    model: str = ""
    stop_reason: str | None = None
    cached: bool = False


class Model(ABC):
    """An async text-generation backend.

    ``sample`` is the sample index: calls identical except for ``sample`` are distinct draws
    (and are cached separately), which is how best-of-N pools and game trees are built.
    """

    name: str = "model"

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        config: GenConfig | None = None,
        tools: list[ToolSpec] | None = None,
        sample: int = 0,
    ) -> ModelOutput: ...

    def describe(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"
