"""Agent tools. A tool is trusted code an agent may invoke (engine, SQL, python, calculator...).

Tool outputs are produced by trusted code, so mechanisms may show them to other roles as
*verified evidence* (``share_tool_results``) — one of the main levers for "verified claims".
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..models.base import ToolSpec

_PY2JSON = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}


class Tool(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}, "required": []}
    )
    fn: Callable[..., Any] = Field(exclude=True)
    cost: float = 0.0
    group: str = "default"  # clearance/tool-group name used by domains

    def spec(self) -> ToolSpec:
        return ToolSpec(name=self.name, description=self.description, parameters=self.parameters)

    async def run(self, args: dict[str, Any]) -> str:
        out = self.fn(**args)
        if inspect.isawaitable(out):
            out = await out
        return out if isinstance(out, str) else str(out)

    def describe(self) -> str:
        return f"tool:{self.name}"


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    group: str = "default",
    cost: float = 0.0,
) -> Any:
    """Build a :class:`Tool` from a (sync or async) function using its signature/docstring.

    Parameter descriptions are taken from ``Args:`` lines in the docstring when present.
    """

    def build(f: Callable[..., Any]) -> Tool:
        sig = inspect.signature(f)
        hints = typing.get_type_hints(f)
        doc = inspect.getdoc(f) or ""
        arg_docs: dict[str, str] = {}
        if "Args:" in doc:
            for line in doc.split("Args:", 1)[1].splitlines():
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    arg_docs[k.strip().split(" ")[0]] = v.strip()
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in sig.parameters.values():
            t = hints.get(p.name, str)
            origin = typing.get_origin(t) or t
            props[p.name] = {"type": _PY2JSON.get(origin, "string")}
            if p.name in arg_docs:
                props[p.name]["description"] = arg_docs[p.name]
            if p.default is inspect.Parameter.empty:
                required.append(p.name)
        return Tool(
            name=name or f.__name__,
            description=description or doc.split("Args:")[0].strip() or f.__name__,
            parameters={"type": "object", "properties": props, "required": required},
            fn=f,
            group=group,
            cost=cost,
        )

    if fn is not None:
        return build(fn)
    return build
