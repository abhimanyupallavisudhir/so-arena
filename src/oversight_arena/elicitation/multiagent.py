"""Multi-agent optimisation: PSRO / iterated best response / level-k with any strategy oracle.

In multi-agent mechanisms, what a strategy optimiser should do depends on what the *other*
optimisers do. Policy-Space Response Oracles (Lanctot et al. 2017) handles this soundly:
keep a population of strategies per role, estimate the empirical game between populations,
solve it (meta-strategy), and have each role's oracle (e.g. an LLM prompt optimiser)
best-respond to the others' meta-strategy; repeat. Special cases:

- ``meta_solver="last"`` → iterated best response (level-k thinking: level-k best-responds
  to level-(k-1));
- ``meta_solver="uniform"`` → fictitious play over the population;
- ``meta_solver="nash"`` → PSRO/double oracle.

The trace records the meta-equilibrium's *ground truth* at every iteration — i.e. whether the
strategic landscape the mechanism induces pulls toward honesty or deception as optimisers
adapt to each other.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..analysis.games import EmpiricalGame
from ..core.episode import EpisodeRecord
from ..core.strategy import Assignment, Profile, Strategy
from ..core.task import Task
from ..core.util import gather_limited
from ..domains.base import Domain
from ..experiment.results import Results
from ..experiment.runner import AgentTable, run_episode
from ..mechanisms.base import Mechanism
from .evaluator import Evaluator
from .optimize import PromptOptimizer, Proposer, ranked

Oracle = Callable[[Evaluator, list[Strategy]], Awaitable[Strategy | None]]


def optimizer_oracle(proposer: Proposer, iterations: int = 2, per_iter: int = 3, minibatch: int | None = None,
                     steering: str | None = None) -> Oracle:
    """Best-response oracle: short prompt optimisation seeded with the role's population."""

    async def oracle(ev: Evaluator, population: list[Strategy]) -> Strategy | None:
        po = PromptOptimizer(ev, proposer, seeds=population, iterations=iterations, per_iter=per_iter,
                             minibatch=minibatch, steering=steering, base=population[0])
        trace = await po.run()
        best = trace.best(1)
        return best[0].strategy if best else None

    return oracle


def pool_oracle(pool: Sequence[Strategy]) -> Oracle:
    """Best response chosen from a fixed candidate pool (cheap; exact within the pool)."""

    async def oracle(ev: Evaluator, population: list[Strategy]) -> Strategy | None:
        evals = await asyncio.gather(*[ev.evaluate(s) for s in pool])
        top = ranked(evals)
        return top[0].strategy if top else None

    return oracle


@dataclass
class PSROIteration:
    iteration: int
    populations: dict[str, list[str]]
    meta: dict[str, dict[str, float]]
    meta_payoffs: dict[str, float]
    meta_gt: dict[str, float]
    exploitability: float
    new: dict[str, str | None] = field(default_factory=dict)
    br_gain: dict[str, float] = field(default_factory=dict)


class PSRO:
    def __init__(
        self,
        domain: Domain,
        mechanism: Mechanism,
        agents: "AgentTable | dict[str, Any] | Any",
        initial: dict[str, list[Strategy]],
        oracles: dict[str, Oracle],
        iterations: int = 3,
        meta_solver: str = "nash",
        tasks: Sequence[Task] | None = None,
        seeds: int = 1,
        fixed: dict[str, Strategy] | None = None,
        gt_keys: Sequence[str] = ("correct",),
        concurrency: int = 8,
        clearances: dict[str, Sequence[str]] | None = None,
    ):
        self.domain = domain
        self.mechanism = mechanism
        self.agents = agents if isinstance(agents, AgentTable) else AgentTable(agents)
        self.pop = {r: list(v) for r, v in initial.items()}
        self.roles = list(initial)
        self.oracles = oracles
        self.iterations = iterations
        self.meta_solver = meta_solver
        self.tasks = list(tasks) if tasks is not None else domain.tasks()
        self.seeds = seeds
        self.fixed = fixed or {}
        self.gt_keys = list(gt_keys)
        self.concurrency = concurrency
        self.clearances = clearances
        self.records: dict[tuple, EpisodeRecord] = {}
        self.history: list[PSROIteration] = []

    async def _fill(self) -> None:
        import itertools

        jobs = []
        for combo in itertools.product(*[self.pop[r] for r in self.roles]):
            for t in self.tasks:
                for s in range(self.seeds):
                    key = (tuple(x.id for x in combo), t.id, s)
                    if key in self.records:
                        continue
                    asg = {r: Assignment(strategy=x, seed=s) for r, x in zip(self.roles, combo)}
                    asg.update({r: Assignment(strategy=x, seed=s) for r, x in self.fixed.items()})
                    jobs.append((key, t, Profile(assignments=asg), s))

        def make(key, t, prof, s):
            async def f():
                rec = await run_episode(self.mechanism, t, prof, self.agents, self.domain, seed=s, clearances=self.clearances)
                self.records[key] = rec
            return f

        await gather_limited([make(*j) for j in jobs], self.concurrency)

    def game(self) -> EmpiricalGame:
        g = EmpiricalGame.from_results(list(self.records.values()), self.roles, strategy_col="strategy",
                                        payoff_roles=[r for r in self.roles], gt_metrics=self.gt_keys)
        return g

    def _meta(self, g: EmpiricalGame) -> dict[str, np.ndarray]:
        if self.meta_solver == "uniform":
            return g.uniform()
        if self.meta_solver == "last":
            mix = {}
            for r in self.roles:
                v = np.zeros(len(g.strategies[r]))
                v[g.strategies[r].index(self.pop[r][-1].id)] = 1.0
                mix[r] = v
            return mix
        if self.meta_solver == "replicator":
            return g.replicator(iters=2000, lr=0.5)[-1]
        eqs = g.nash()
        if not eqs:
            return g.replicator(iters=2000, lr=0.5)[-1]
        return min(eqs, key=lambda e: e.regret).mix

    async def run(self) -> "PSROTrace":
        for it in range(self.iterations + 1):
            await self._fill()
            g = self.game()
            mix = self._meta(g)
            meta = {r: {s: float(p) for s, p in zip(g.strategies[r], mix[r])} for r in self.roles}
            rec = PSROIteration(
                iteration=it,
                populations={r: [s.name for s in self.pop[r]] for r in self.roles},
                meta={r: {self._name(r, sid): p for sid, p in m.items()} for r, m in meta.items()},
                meta_payoffs=g.expected_payoffs(mix),
                meta_gt=g.expected_gt(mix),
                exploitability=g.exploitability(mix),
            )
            if it == self.iterations:
                self.history.append(rec)
                break
            for r in self.roles:
                others = {
                    o: [(self._by_id(o, sid), p) for sid, p in meta[o].items() if p > 1e-6]
                    for o in self.roles if o != r
                }
                others.update(self.fixed)
                ev = Evaluator(self.domain, self.mechanism, self.agents, r, self.tasks, others=others,
                               seeds=self.seeds, gt_keys=self.gt_keys, clearances=self.clearances,
                               concurrency=self.concurrency)
                br = await self.oracles[r](ev, self.pop[r])
                if br is not None and all(br.id != s.id for s in self.pop[r]):
                    self.pop[r].append(br)
                    rec.new[r] = br.name
                    bre = await ev.evaluate(br)
                    rec.br_gain[r] = bre.reward - rec.meta_payoffs.get(r, float("nan"))
                else:
                    rec.new[r] = None
            self.history.append(rec)
        return PSROTrace(self.history, Results(list(self.records.values())), self.game(), self.pop)

    def _by_id(self, role: str, sid: str) -> Strategy:
        return next(s for s in self.pop[role] if s.id == sid)

    def _name(self, role: str, sid: str) -> str:
        return next((s.name for s in self.pop[role] if s.id == sid), sid)


@dataclass
class PSROTrace:
    iterations: list[PSROIteration]
    results: Results
    game: EmpiricalGame
    populations: dict[str, list[Strategy]]

    def df(self) -> pd.DataFrame:
        rows = []
        for it in self.iterations:
            row: dict[str, Any] = {"iteration": it.iteration, "exploitability": it.exploitability}
            row.update({f"u[{k}]": v for k, v in it.meta_payoffs.items()})
            row.update({f"gt[{k}]": v for k, v in it.meta_gt.items()})
            row.update({f"new[{k}]": v for k, v in it.new.items()})
            row.update({f"br_gain[{k}]": v for k, v in it.br_gain.items()})
            rows.append(row)
        return pd.DataFrame(rows)
