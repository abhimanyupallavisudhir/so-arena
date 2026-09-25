"""Deterministic models for tests, simulations and fixtures."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from ..core.types import ChatMessage, ToolCall, Usage
from ..core.util import rng_for
from .base import GenConfig, Model, ModelOutput, ToolSpec

Responder = Callable[..., Any]


class FunctionModel(Model):
    """Wrap a Python function ``fn(messages, config, tools, sample) -> str | ModelOutput``.

    The function may be sync or async and may accept any prefix of those arguments.
    """

    def __init__(self, fn: Responder, name: str = "function"):
        self.fn = fn
        self.name = name
        self._nparams = len(inspect.signature(fn).parameters)

    async def generate(self, messages, config=None, tools=None, sample=0) -> ModelOutput:
        args = (messages, config or GenConfig(), tools or [], sample)[: self._nparams]
        out = self.fn(*args)
        if inspect.isawaitable(out):
            out = await out
        if isinstance(out, ModelOutput):
            return out
        if isinstance(out, dict) and "tool_calls" in out:
            return ModelOutput(
                text=out.get("text", ""),
                tool_calls=[ToolCall(**tc) for tc in out["tool_calls"]],
                usage=Usage(calls=1),
                model=self.name,
            )
        text = str(out)
        return ModelOutput(
            text=text,
            usage=Usage(
                calls=1,
                input_tokens=sum(len(m.content.split()) for m in messages),
                output_tokens=len(text.split()),
            ),
            model=self.name,
        )


class MockModel(Model):
    """Replays a fixed list of responses (cycling), or a constant response."""

    def __init__(self, responses: list[str | ModelOutput] | str = "OK", name: str = "mock"):
        self.responses = [responses] if isinstance(responses, (str, ModelOutput)) else list(responses)
        self.name = name
        self.i = 0
        self.calls: list[list[ChatMessage]] = []

    async def generate(self, messages, config=None, tools=None, sample=0) -> ModelOutput:
        self.calls.append(list(messages))
        r = self.responses[self.i % len(self.responses)]
        self.i += 1
        if isinstance(r, ModelOutput):
            return r
        return ModelOutput(text=r, usage=Usage(calls=1), model=self.name)


class RandomChoiceModel(Model):
    """Answers any question by emitting random text; distributions/choices are random.

    Deterministic given (messages, sample). Useful as a null baseline and for smoke tests.
    """

    def __init__(self, name: str = "random"):
        self.name = name

    async def generate(self, messages, config=None, tools=None, sample=0) -> ModelOutput:
        last = messages[-1].content if messages else ""
        rng = rng_for(self.name, last, sample)
        import re

        opts = re.findall(r"\(([A-Z])\)", last) or ["A", "B"]
        if "JSON" in last and "probab" in last.lower():
            ws = [rng.random() + 0.05 for _ in opts]
            s = sum(ws)
            body = ", ".join(f'"{o}": {w / s:.3f}' for o, w in zip(opts, ws))
            text = "{" + body + "}"
        elif "Answer with the letter" in last or "ANSWER:" in last:
            text = f"ANSWER: {rng.choice(opts)}"
        else:
            words = ["because", "the", "evidence", "shows", "clearly", "that", "this", "holds"]
            text = " ".join(rng.choice(words) for _ in range(20))
        return ModelOutput(text=text, usage=Usage(calls=1, output_tokens=len(text.split())), model=self.name)
