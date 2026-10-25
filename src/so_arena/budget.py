"""Budgets for experiments priced per token: sampling (APIs or Tinker) and RL training (Tinker).

A cost is episodes x tokens per episode x prices, summed over the roles of each episode. A role's
tokens per episode (:class:`Tokens`) are what it prefills (uncached and prompt-cached), samples and -
if it is trained - passes through forward-backward. Prompt sizes come from the library's dry run
(:func:`measure`, simulated models: no API calls). What a dry run cannot know - how long models
reason, how many tool turns an agent takes, how large tool outputs are - enters as explicit
assumptions with a (low, central, high) range, so a budget carries a range and a sensitivity table
(:meth:`Budget.sensitivity`) says which assumption the total depends on; a small pilot measures those.

Prices: Tinker's per-model prices (a dated snapshot of its ``models.json`` in ``data/tinker_prices.json``;
list prices by default, not limited-time discounts) and the model registry's API prices. Tinker bills
forward-backward over every token of a trained sequence (prompt included) at its train price.

Token profiles assume a role keeps its own reasoning in its context, so a trained conversation is one
sequence (train tokens = its final length); tokens appended by others or by the environment are
uncached prefill, everything earlier is a prompt-cache hit.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

_TINKER = Path(__file__).parent / "data" / "tinker_prices.json"


@dataclass(frozen=True)
class Tokens:
    """Tokens one role uses in one episode."""

    prefill: float = 0.0  # uncached input
    cached: float = 0.0  # input read from the prompt cache
    sample: float = 0.0  # generated, reasoning included
    train: float = 0.0  # forward-backward (trained roles only)

    def __add__(self, other: Tokens) -> Tokens:
        return Tokens(self.prefill + other.prefill, self.cached + other.cached, self.sample + other.sample,
                      self.train + other.train)

    def scale(self, k: float) -> Tokens:
        return Tokens(self.prefill * k, self.cached * k, self.sample * k, self.train * k)


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    prefill: float
    cached_prefill: float
    sample: float
    train: float | None = None
    provider: str = "api"

    def cost(self, t: Tokens) -> float:
        if t.train and self.train is None:
            raise ValueError(f"no training price ({self.provider}): this model cannot be trained there")
        return (t.prefill * self.prefill + t.cached * self.cached_prefill + t.sample * self.sample
                + t.train * (self.train or 0.0)) / 1e6


@lru_cache(maxsize=2)
def tinker_prices(list_prices: bool = True) -> dict[str, Price]:
    """Tinker's prices by model id (``list_prices``: undiscounted, the safe basis for a budget)."""
    data = json.loads(_TINKER.read_text())
    out = {}
    for mid, m in data["models"].items():
        p = m.get("list", m) if list_prices else m
        out[mid] = Price(p["prefill"], p["cached_prefill"], p["sample"], p["train"], provider="tinker")
    return out


def tinker_snapshot_date() -> str:
    return json.loads(_TINKER.read_text())["fetched"]


def price(model: str) -> Price:
    """A Tinker model id (``Qwen/Qwen3.5-9B``) or a model in the registry (API prices, not trainable)."""
    if model in tinker_prices():
        return tinker_prices()[model]
    from so_arena.models.registry import get_spec

    s = get_spec(model)
    if s is None or s.price_input_per_mtok is None or s.price_output_per_mtok is None:
        raise KeyError(f"no price for {model!r}: add it to the model registry or use a Tinker model id")
    cached = s.price_cached_input_per_mtok if s.price_cached_input_per_mtok is not None else s.price_input_per_mtok
    return Price(s.price_input_per_mtok, cached, s.price_output_per_mtok, provider="api")


# ------------------------------------------------------------------------------ token profiles


def conversation(prompt: float, calls: int, output: float, reasoning: float = 0.0, others: float = 0.0, *,
                 trained: bool = False) -> Tokens:
    """A role making ``calls`` sequential calls in one conversation: the first prefills ``prompt``;
    before each later call other roles add ``others`` tokens (uncached); each call samples
    ``reasoning + output``. Trained: forward-backward over the final sequence."""
    turn = reasoning + output
    contexts = [prompt + i * (turn + others) for i in range(calls)]
    uncached = prompt + (calls - 1) * others
    final = prompt + calls * turn + (calls - 1) * others
    return Tokens(prefill=uncached, cached=sum(contexts) - uncached, sample=calls * turn,
                  train=final if trained else 0.0)


def agentic(prompt: float, turns: float, reasoning: float, action: float, observation: float, final: float,
            final_reasoning: float = 0.0, *, trained: bool = False) -> Tokens:
    """A tool-using agent: ``turns`` tool calls (each ``reasoning + action`` sampled, answered by an
    ``observation`` from the environment), then a final answer (``final_reasoning + final``)."""
    step = reasoning + action + observation
    n = round(turns)
    contexts = [prompt + t * step for t in range(n + 1)]
    uncached = prompt + n * observation
    total = prompt + n * step + final_reasoning + final
    return Tokens(prefill=uncached, cached=sum(contexts) - uncached,
                  sample=n * (reasoning + action) + final_reasoning + final, train=total if trained else 0.0)


# ------------------------------------------------------------------------------ budgets

Assumptions = Mapping[str, float]
Profile = Mapping[str, tuple[str, Tokens]]  # role -> (model, tokens per episode)


@dataclass
class Item:
    """``episodes`` episodes, each using ``profile(assumptions)`` (role -> (model, tokens))."""

    name: str
    episodes: float | Callable[[Assumptions], float]
    profile: Callable[[Assumptions], Profile]
    group: str = ""
    note: str = ""

    def n(self, a: Assumptions) -> float:
        return self.episodes(a) if callable(self.episodes) else self.episodes

    def by_provider(self, a: Assumptions) -> dict[str, float]:
        out: dict[str, float] = {}
        n = self.n(a)
        for model, t in self.profile(a).values():
            p = price(model)
            out[p.provider] = out.get(p.provider, 0.0) + n * p.cost(t)
        return out

    def tokens(self, a: Assumptions) -> Tokens:
        tot = Tokens()
        for _, t in self.profile(a).values():
            tot = tot + t
        return tot.scale(self.n(a))


@dataclass
class Fixed:
    """A cost not priced per token (people, GPUs, CPUs): (low, central, high) USD."""

    name: str
    usd: tuple[float, float, float]
    provider: str = "other"
    group: str = ""
    note: str = ""


@dataclass
class Budget:
    items: Sequence[Item | Fixed]
    ranges: Mapping[str, tuple[float, float, float]]  # assumption -> (low, central, high)
    extra: dict[str, Any] = field(default_factory=dict)

    def scenario(self, which: str) -> dict[str, float]:
        i = {"low": 0, "central": 1, "high": 2}[which]
        return {k: v[i] for k, v in self.ranges.items()}

    def _cost(self, it: Item | Fixed, a: Assumptions, which: str) -> dict[str, float]:
        if isinstance(it, Fixed):
            return {it.provider: it.usd[{"low": 0, "central": 1, "high": 2}[which]]}
        return it.by_provider(a)

    def table(self) -> pd.DataFrame:
        """One row per item: episodes, central tokens (millions) and USD in the low/central/high scenarios."""
        rows = []
        sc = {w: self.scenario(w) for w in ("low", "central", "high")}
        for it in self.items:
            c = {w: self._cost(it, sc[w], w) for w in sc}
            row = {"group": it.group, "item": it.name}
            if isinstance(it, Item):
                t = it.tokens(sc["central"])
                row.update(episodes=it.n(sc["central"]), prefill_M=(t.prefill + t.cached) / 1e6,
                           sample_M=t.sample / 1e6, train_M=t.train / 1e6)
            row["provider"] = "+".join(sorted(c["central"]))
            row.update({f"usd_{w}": sum(c[w].values()) for w in sc})
            row["note"] = it.note
            rows.append(row)
        return pd.DataFrame(rows)

    def totals(self) -> pd.DataFrame:
        """USD per provider (what each kind of funding must cover) in each scenario."""
        sc = {w: self.scenario(w) for w in ("low", "central", "high")}
        acc: dict[str, dict[str, float]] = {}
        for it in self.items:
            for w, a in sc.items():
                for prov, usd in self._cost(it, a, w).items():
                    acc.setdefault(prov, {"low": 0.0, "central": 0.0, "high": 0.0})[w] += usd
        df = pd.DataFrame(acc).T[["low", "central", "high"]]
        df.loc["total"] = df.sum()
        return df

    def total(self, a: Assumptions | None = None, which: str = "central") -> float:
        a = a if a is not None else self.scenario(which)
        return sum(sum(self._cost(it, a, which).values()) for it in self.items)

    def interval(self, n: int = 2000, quantiles: Sequence[float] = (0.1, 0.5, 0.9), seed: int = 0) -> dict[float, float]:
        """Quantiles of the total with every assumption (and every fixed cost) drawn independently from a
        triangular distribution on its (low, central, high) - an interval, where the low and high
        scenarios (all assumptions at an end at once) are only bounds."""
        import random

        rng = random.Random(seed)
        totals = []
        for _ in range(n):
            a = {k: rng.triangular(lo, hi, c) for k, (lo, c, hi) in self.ranges.items()}
            t = 0.0
            for it in self.items:
                if isinstance(it, Fixed):
                    lo, c, hi = it.usd
                    t += rng.triangular(lo, hi, c) if hi > lo else c
                else:
                    t += sum(it.by_provider(a).values())
            totals.append(t)
        s = pd.Series(totals)
        return {q: float(s.quantile(q)) for q in quantiles}

    def sensitivity(self) -> pd.DataFrame:
        """Total USD with one assumption at its low / high value and the rest central, largest swing first."""
        base = self.scenario("central")
        rows = []
        for k, (lo, c, hi) in self.ranges.items():
            t_lo, t_hi = self.total({**base, k: lo}), self.total({**base, k: hi})
            rows.append({"assumption": k, "low": lo, "central": c, "high": hi, "total_at_low": t_lo,
                         "total_at_high": t_hi, "swing": abs(t_hi - t_lo)})
        return pd.DataFrame(rows).sort_values("swing", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------ measuring prompts


def measure(mechanism: Any, items: Sequence[Any], *, ctx: Any = None, output_tokens: Mapping[str, int] | None = None,
            default_output: int = 300) -> dict[str, dict[str, float]]:
    """Mean tokens per episode by role - ``{"input", "output", "calls", "first_input"}`` (the prompt of
    the role's first decision) - from a dry run of ``mechanism`` on ``items`` with simulated models
    producing ``output_tokens[role]`` tokens per call. Prompt sizes are the library's real prompts (~4
    characters per token); agentic loops are not simulated (a simulated model makes no tool calls), so
    tool-using roles show their first call."""
    import asyncio

    from so_arena.core.game import Player
    from so_arena.core.policy import LLMPolicy
    from so_arena.models.simulated import SimulatedModel

    outs = dict(output_tokens or {})
    players = {r: Player(policy=LLMPolicy(SimulatedModel(f"budget-sim/{r}", output_tokens=outs.get(r, default_output),
                                                          output_tokens_sd=0.01), label=r))
               for r in mechanism.roles()}
    acc: dict[str, list[float]] = {}
    for it in items:
        ep = asyncio.run(mechanism.run(it, players, ctx=ctx))
        for r, u in ep.usage.items():
            a = acc.setdefault(r, [0.0, 0.0, 0.0, 0.0])
            a[0] += u.input_tokens + u.cached_input_tokens
            a[1] += u.output_tokens
            a[2] += u.calls
            first = next((t.usage for t in ep.turns if t.role == r and t.usage.calls), None)
            a[3] += (first.input_tokens + first.cached_input_tokens) if first else 0.0
    return {r: {"input": v[0] / len(items), "output": v[1] / len(items), "calls": v[2] / len(items),
                "first_input": v[3] / len(items)} for r, v in acc.items()}
