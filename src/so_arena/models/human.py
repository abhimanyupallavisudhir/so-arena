"""Human-in-the-loop models (e.g. human judges), with the time spent recorded as oversight effort."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence

from so_arena.core.types import Completion, Message, Usage
from so_arena.models.base import Model


class HumanModel(Model):
    """Asks a human at the terminal. ``time_budget_s`` is displayed (not enforced) so that evaluation
    budget can be an experimental variable (e.g. the ASD-vs-evaluation-time curve)."""

    def __init__(self, name: str = "human", *, time_budget_s: float | None = None):
        self.name = name
        self.time_budget_s = time_budget_s

    async def generate(self, messages, options=None, *, sample_index=0):
        def ask() -> tuple[str, float]:
            print("\n" + "=" * 80)
            for m in messages:
                print(f"[{m.role.upper()}]\n{m.content}\n")
            if self.time_budget_s:
                print(f"(Suggested time budget: {self.time_budget_s:.0f}s)")
            t0 = time.time()
            lines = []
            print("Your response (end with an empty line):")
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            return "\n".join(lines), time.time() - t0

        text, elapsed = await asyncio.to_thread(ask)
        return Completion(text=text, model=self.name, usage=Usage(calls=1, effort_seconds=elapsed))


class CallbackModel(Model):
    """Delegates to an async callback, e.g. a web form in a rating UI."""

    def __init__(self, callback: Callable[[Sequence[Message]], Awaitable[str]], name: str = "callback"):
        self.callback = callback
        self.name = name

    async def generate(self, messages, options=None, *, sample_index=0):
        t0 = time.time()
        text = await self.callback(list(messages))
        return Completion(text=text, model=self.name, usage=Usage(calls=1, effort_seconds=time.time() - t0))
