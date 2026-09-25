"""Programmatic agents (fixtures, simulated agents, engine-backed experts) and human agents."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable
from typing import Any

from ..core.util import stable_hash
from .base import Action, Agent, Observation


class ScriptedAgent(Agent):
    """Wrap ``fn(obs) -> Action | str | dict`` (sync or async) as an agent.

    A returned ``dict`` is treated as the parsed structured answer (``{"choice": "A"}``,
    ``{"probs": {...}}``, ``{"value": 0.3}``...); an optional ``"text"`` key sets the text.
    """

    def __init__(self, fn: Callable[[Observation], Any], id: str | None = None):
        self.fn = fn
        self.id = id or getattr(fn, "__name__", "scripted")

    def describe(self) -> dict[str, Any]:
        return {"type": "ScriptedAgent", "id": self.id, "fn": getattr(self.fn, "__qualname__", repr(self.fn))}

    async def act(self, obs: Observation) -> Action:
        out = self.fn(obs)
        if inspect.isawaitable(out):
            out = await out
        if isinstance(out, Action):
            return out
        if isinstance(out, dict):
            text = out.get("text")
            parsed = {k: v for k, v in out.items() if k not in ("text", "reasoning")}
            if text is None:
                text = json.dumps(parsed, default=str)
            return Action(text=text, parsed=parsed, reasoning=out.get("reasoning"))
        return Action(text=str(out))


class ConstantAgent(ScriptedAgent):
    """Always returns the same structured answer (e.g. a uniform judge baseline)."""

    def __init__(self, answer: dict[str, Any] | str, id: str | None = None):
        self.answer = answer
        super().__init__(lambda obs: answer, id=id or f"constant-{stable_hash(answer, length=6)}")


class HumanAgent(Agent):
    """Interactive console agent — lets a human play any role (e.g. a human judge)."""

    def __init__(self, id: str = "human"):
        self.id = id

    async def act(self, obs: Observation) -> Action:
        from .llm import format_instructions
        from .parsing import parse_choice, parse_distribution, parse_json, parse_scalar

        print("\n" + "=" * 80)
        print(f"You are playing: {obs.role_title}")
        if obs.brief:
            print(obs.brief)
        if obs.strategy:
            print("\n[Private instructions]\n" + obs.strategy)
        print("\n" + obs.task_text)
        if obs.transcript_text:
            print("\n--- Transcript ---\n" + obs.transcript_text)
        print("\n" + obs.prompt)
        print(format_instructions(obs.response))
        text = await asyncio.to_thread(input, "> ")
        spec = obs.response
        parsed: dict[str, Any] = {}
        if spec.kind == "choice":
            parsed = {"choice": parse_choice(text, spec.options or [])}
        elif spec.kind == "distribution":
            d = parse_distribution(text, spec.options or [])
            if d is None:
                c = parse_choice(text, spec.options or [])
                d = {o: (1.0 if o == c else 0.0) for o in spec.options or []}
            parsed = {"probs": d, "choice": max(d, key=d.get)}  # type: ignore[arg-type]
        elif spec.kind == "scalar":
            parsed = {spec.scalar_name: parse_scalar(text, spec.lo, spec.hi, spec.scalar_name)}
        elif spec.kind == "json":
            parsed = parse_json(text) or {}
        return Action(text=text, parsed=parsed)
