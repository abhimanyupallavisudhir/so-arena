"""Tools: private affordances a role can use while acting (an engine, a database, an interpreter).

Tools model the *capability gap* between roles ("smart generalist judge, narrow expert"): an expert
with an engine tool versus a judge without one. They differ from verifiers in that tool results
are private to the caller unless it chooses to share them (possibly as a verifiable claim).

LLM policies call tools with a model-agnostic text protocol: ``<tool name="NAME">ARGS</tool>``.
"""

from __future__ import annotations

import abc
import asyncio
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from so_arena.core.items import TaskItem
from so_arena.core.types import Usage

if TYPE_CHECKING:
    from so_arena.core.game import Game

TOOL_CALL_RE = re.compile(r"<tool\s+name\s*=\s*[\"'](?P<name>[\w\-\.]+)[\"']\s*>(?P<args>.*?)</tool>", re.S | re.I)


class ToolResult(BaseModel):
    output: str
    usage: Usage = Field(default_factory=Usage)
    error: bool = False


class Tool(abc.ABC):
    name: str = "tool"
    description: str = ""
    example: str = ""

    @abc.abstractmethod
    async def call(self, args: str, item: TaskItem, game: "Game | None" = None) -> ToolResult: ...

    def instructions(self) -> str:
        s = f'- <tool name="{self.name}">...</tool>: {self.description}'
        if self.example:
            s += f"\n  Example: {self.example}"
        return s


class FunctionTool(Tool):
    """Wraps ``fn(args: str, item: TaskItem) -> str`` (sync or async)."""

    def __init__(self, name: str, fn: Callable[..., Any], description: str = "", example: str = ""):
        self.name = name
        self.fn = fn
        self.description = description
        self.example = example

    async def call(self, args, item, game=None):
        try:
            res = self.fn(args, item)
            if asyncio.iscoroutine(res):
                res = await res
            return ToolResult(output=str(res))
        except Exception as e:  # tools report errors to the caller instead of crashing the episode
            return ToolResult(output=f"error: {e}", error=True)


def tool_instructions(tools: dict[str, Tool], max_calls: int) -> str:
    if not tools:
        return ""
    lines = [
        f"You have private tools (results are visible only to you). To call one, output exactly "
        f'<tool name="NAME">ARGUMENTS</tool> and stop; you will receive the result and can continue. '
        f"At most {max_calls} calls. When done, give your final response without any tool tags. Tools:"
    ]
    lines += [t.instructions() for t in tools.values()]
    return "\n".join(lines)
