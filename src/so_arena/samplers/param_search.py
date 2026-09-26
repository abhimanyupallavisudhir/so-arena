"""Parameter search: numeric (and categorical) search over the parameters of programmatic policies.

Scripted and simulated agents - a synthetic arguer's sophistry rate, a monitored actor's covert rate, a
forecaster's extremizing factor - have parameters instead of prompts. :class:`ParamSearch` searches them
the way prompt search searches strategies, with the same records (:class:`~so_arena.samplers.prompt_search.
Candidate`, :class:`~so_arena.samplers.prompt_search.SearchResult`), the same fixed minibatch of items for
every candidate (paired comparisons), held-out re-evaluation of the winners, and ground truth recorded but
never used to propose: an evolutionary loop (random proposals with probability ``explore``, else Gaussian
mutations of one of the ``top_k`` best candidates so far) scored by the mechanism's reward only.

``constraint`` restricts the search to candidates whose *measured* ground-truth value satisfies it
(``"honest"``: mean value above ``threshold``; ``"deceptive"``: below; or any ``Candidate -> bool``). It is
an experimental device for mapping frontiers - the best honest and the best deceptive behaviour the
mechanism pays for (:func:`~so_arena.samplers.prompt_search.honesty_margin`) - not something training
could do: a candidate that fails it (or has too few labels to tell) is kept in the record but is never a
parent or the search's best.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from so_arena.core.game import RunContext
from so_arena.core.ground_truth import GroundTruthScorer
from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Mechanism
from so_arena.core.policy import Policy
from so_arena.samplers.prompt_search import Candidate, PromptSearch, SearchResult

# a parameter's range: (lo, hi) floats (uniform), (lo, hi) ints (integers), or a list of values (categorical)
Space = Mapping[str, tuple[float, float] | tuple[int, int] | Sequence[Any]]


def _is_float_range(spec: Any) -> bool:
    return isinstance(spec, tuple) and len(spec) == 2 and any(isinstance(x, float) for x in spec)


def _is_int_range(spec: Any) -> bool:
    return isinstance(spec, tuple) and len(spec) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in spec)


def encode(params: Mapping[str, Any]) -> str:
    """The canonical text of a parameter setting (its strategy string): sorted keys, floats to 6 digits."""
    return json.dumps({k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in sorted(params.items())})


def satisfies(constraint: str | Callable[[Candidate], bool] | None, c: Candidate, threshold: float = 0.0) -> bool:
    """Whether ``c`` meets a measured-value constraint; an unknown mean value (NaN) never does."""
    if constraint is None or constraint == "unconstrained":
        return True
    if callable(constraint):
        return bool(constraint(c))
    v = c.mean_value
    if not math.isfinite(v):
        return False
    if constraint == "honest":
        return v > threshold
    if constraint == "deceptive":
        return v < threshold
    raise ValueError(f"constraint must be 'honest', 'deceptive', a function or None, got {constraint!r}")


class ParamSearch(PromptSearch):
    """Search over ``factory(**params)`` policies for one role of a mechanism, scored by its reward.

    Args:
        factory: ``**params -> Policy`` (e.g. :func:`~so_arena.domains.synthetic.synthetic_arguer`).
        space: parameter ranges (see :data:`Space`); parameters not in it keep the factory's defaults.
        seeds: starting settings (default: the middle of every range, the first of every list).
        explore / scale / top_k: probability of a fresh random proposal; mutation step as a fraction of each
            range; how many of the best candidates are parents.
        constraint / threshold: measured-value filter (module docstring).
        Other arguments as :class:`~so_arena.samplers.prompt_search.PromptSearch` (``others``, ``arms``,
        ``eval_items``, ``n_final``, ``repeats``, ``minibatch``, ...).
    """

    def __init__(self, mechanism: Mechanism, items: Sequence[TaskItem], *, role: str, factory: Callable[..., Policy],
                 space: Space, others: dict[str, Any], arms: Sequence[str | None] = (None,), iterations: int = 6,
                 candidates_per_iter: int = 6, seeds: Sequence[Mapping[str, Any]] | None = None, explore: float = 0.3,
                 scale: float = 0.25, top_k: int = 4, constraint: str | Callable[[Candidate], bool] | None = None,
                 threshold: float = 0.0, minibatch: int | None = None, eval_items: Sequence[TaskItem] | None = None,
                 n_final: int = 3, repeats: int = 1, ground_truth: Sequence[GroundTruthScorer] | None = None,
                 ctx: RunContext | None = None, concurrency: int | None = None, seed: int = 0):
        if not space:
            raise ValueError("ParamSearch needs a non-empty parameter space")
        for k, spec in space.items():
            if not (_is_float_range(spec) or _is_int_range(spec) or (isinstance(spec, (list, tuple)) and len(spec))):
                raise ValueError(f"space[{k!r}] must be (lo, hi) or a non-empty list of values, got {spec!r}")
            if (_is_float_range(spec) or _is_int_range(spec)) and not spec[0] <= spec[1]:
                raise ValueError(f"space[{k!r}]: lo > hi in {spec!r}")
        name = constraint if isinstance(constraint, str) else "custom" if constraint is not None else "unconstrained"
        self.space, self.factory = dict(space), factory
        seed_params = [dict(p) for p in seeds] if seeds else [self._middle()]
        super().__init__(mechanism, items, role=role, policy_factory=lambda s: factory(**json.loads(s)), others=others,
                         optimizer=None, arms=arms, directive=name, iterations=iterations,
                         candidates_per_iter=candidates_per_iter, seed_strategies=[encode(p) for p in seed_params],
                         minibatch=minibatch, eval_items=eval_items, n_final=n_final, repeats=repeats,
                         ground_truth=ground_truth, ctx=ctx, concurrency=concurrency, seed=seed)
        self.algorithm, self.directive_name = "param", name
        self.explore, self.scale, self.top_k = explore, scale, top_k
        self.constraint, self.threshold = constraint, threshold

    # ------------------------------------------------------------------ proposals
    def _middle(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, spec in self.space.items():
            if _is_float_range(spec):
                out[k] = (float(spec[0]) + float(spec[1])) / 2
            elif _is_int_range(spec):
                out[k] = (spec[0] + spec[1]) // 2
            else:
                out[k] = list(spec)[0]
        return out

    def _random(self) -> dict[str, Any]:
        rng, out = self._rng, {}
        for k, spec in self.space.items():
            if _is_float_range(spec):
                out[k] = rng.uniform(float(spec[0]), float(spec[1]))
            elif _is_int_range(spec):
                out[k] = rng.randint(spec[0], spec[1])
            else:
                out[k] = rng.choice(list(spec))
        return out

    def _mutate(self, params: Mapping[str, Any]) -> dict[str, Any]:
        rng, out = self._rng, dict(params)
        keys = [k for k in self.space if rng.random() < 0.5] or [rng.choice(list(self.space))]  # change at least one
        for k in keys:
            spec = self.space[k]
            if _is_float_range(spec):
                lo, hi = float(spec[0]), float(spec[1])
                out[k] = min(hi, max(lo, float(out.get(k, (lo + hi) / 2)) + rng.gauss(0.0, self.scale * (hi - lo))))
            elif _is_int_range(spec):
                lo, hi = spec
                step = rng.gauss(0.0, max(1.0, self.scale * (hi - lo)))
                out[k] = min(hi, max(lo, int(round(out.get(k, (lo + hi) // 2) + step))))
            else:
                out[k] = rng.choice(list(spec))
        return out

    def _accept(self, cands: list[Candidate]) -> list[Candidate]:
        for c in cands:
            c.accepted = satisfies(self.constraint, c, self.threshold)
        return cands

    async def _score(self, strategy: str, **kw: Any) -> Candidate:
        c = await super()._score(strategy, **kw)
        c.params = json.loads(strategy)
        return c

    async def _iteration(self, it: int) -> list[Candidate]:
        import asyncio

        seen = {c.strategy for c in self.candidates}
        parents = sorted([c for c in self.candidates if c.split == "train" and c.accepted],
                         key=lambda c: c.mean_reward, reverse=True)[: self.top_k]
        proposals: list[tuple[str, str | None]] = []
        for _ in range(20 * self.k):  # retries only against duplicates
            if len(proposals) >= self.k:
                break
            if not parents or self._rng.random() < self.explore:
                s, parent = encode(self._random()), None
            else:
                p = self._rng.choice(parents)
                s, parent = encode(self._mutate(p.params or json.loads(p.strategy))), p.id
            if s not in seen:
                seen.add(s)
                proposals.append((s, parent))
        scored = list(await asyncio.gather(*[self._score(s, iteration=it, parent=p) for s, p in proposals]))
        return self._accept(scored)

    async def arun(self) -> SearchResult:
        import asyncio

        seeds = self._accept(list(await asyncio.gather(*[self._score(s, iteration=0) for s in self.seed_strategies])))
        self.candidates += seeds
        for it in range(1, self.iterations + 1):
            self.candidates += await self._iteration(it)
        evaluated: list[Candidate] = []
        if self.eval_items:
            top = sorted([c for c in self.candidates if c.split == "train" and c.accepted], key=lambda c: c.mean_reward,
                         reverse=True)[: self.n_final]
            evaluated = self._accept(list(await asyncio.gather(*[
                self._score(c.strategy, iteration=c.iteration, parent=c.id, split="eval", items=self.eval_items)
                for c in top])))
        return SearchResult(role=self.role, mechanism=self.mechanism.name, directive=self.directive_name,
                            algorithm=self.algorithm, candidates=self.candidates, evaluated=evaluated)
