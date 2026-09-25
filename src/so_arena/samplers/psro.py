"""Policy-Space Response Oracles (Lanctot et al., 2017) with prompt search as the best-response oracle.

In a multi-agent mechanism, what is optimal for one agent depends on what the others do, and an
optimizer's proposals depend on how it models the others. PSRO makes this explicit: maintain a
population of strategies per role; estimate the empirical game among them; solve it for a
meta-strategy (e.g. a Nash equilibrium); then, for each role, search for a best response *to the
meta-strategy of the others* (the optimizer is told their strategies and probabilities) and add it to
the population. Repeat. The meta-strategy's NashConv (sum of regrets against the new best
responses) estimates how far the population is from equilibrium; the ground-truth value of the
meta-strategy is what the mechanism incentivizes at (approximate) equilibrium.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from so_arena.core.game import RunContext
from so_arena.core.ground_truth import GroundTruthScorer
from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Mechanism
from so_arena.core.policy import MixturePolicy, Policy
from so_arena.core.runner import run_sync
from so_arena.games.egta import EmpiricalGameExperiment
from so_arena.games.normal_form import NormalFormGame
from so_arena.samplers.prompt_search import PromptSearch, SearchResult


def solve_meta(game: NormalFormGame, solver: str = "nash", symmetric: Sequence[str] | None = None) -> list[np.ndarray]:
    """Meta-strategy for PSRO: ``nash`` (2p support enumeration, max-entropy equilibrium; n-p replicator
    from uniform), ``replicator``, ``fictitious`` or ``uniform``.

    Missing payoffs (profiles whose episodes all errored) are filled with each player's lowest observed
    payoff, with a warning (:meth:`NormalFormGame.imputed`) - never with 0, the best possible log-score
    reward, which would make a never-observed profile the equilibrium. ``symmetric`` players (one shared
    population) get one mixed strategy: a symmetric equilibrium, or replicator dynamics on one population.
    """
    if solver == "uniform":
        return game.uniform()
    clean = game.imputed()
    sym = [p for p in (symmetric or []) if p in game.players]
    sym = sym if len(sym) > 1 else []
    if solver == "fictitious":
        return _symmetrize(game, clean.fictitious_play(3000), sym)
    if solver == "nash" and game.n == 2:
        eqs = clean.support_enumeration()
        if sym:  # both players are one population: only symmetric equilibria
            eqs = [m for m in eqs if np.allclose(m[0], m[1], atol=1e-6)]
        if eqs:
            def ent(m):
                return -sum(float((x[x > 0] * np.log(x[x > 0])).sum()) for x in m)

            return max(eqs, key=ent)
    x, _ = clean.replicator(steps=3000, shared=[sym] if sym else None)
    return x


def _symmetrize(game: NormalFormGame, mixed: list[np.ndarray], sym: Sequence[str]) -> list[np.ndarray]:
    """Give the players of one population their average mixed strategy."""
    if not sym:
        return mixed
    idx = [game.players.index(p) for p in sym]
    mean = np.mean([mixed[a] for a in idx], axis=0)
    return [mean.copy() if a in idx else x for a, x in enumerate(mixed)]


class PSROIteration(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    iteration: int
    populations: dict[str, list[str]]
    meta_strategy: dict[str, list[float]]
    nash_conv_prev: float | None = None
    meta_value: dict[str, float] = Field(default_factory=dict)  # expected outcomes under the meta-strategy
    searches: dict[str, Any] = Field(default_factory=dict)


class PSRO:
    """PSRO over strategies for several roles.

    Args:
        roles: roles whose strategies evolve (others are ``fixtures``).
        initial: role -> {name: strategy text} initial populations.
        policy_factories: role -> ``strategy_text -> Policy``.
        search_kwargs: passed to each :class:`PromptSearch` (optimizer, iterations, algorithm, ...).
        symmetric: exchangeable roles, which share one population: their initial strategies (and
            stances) must agree, each iteration searches one best response for them (as the first of
            them, against a symmetric meta-strategy) and adds it to all of their populations under one
            name. The empirical game is then estimated on multisets of strategies - which is only valid
            because a name means the same strategy for every one of these roles.
    """

    def __init__(self, mechanism: Mechanism, items: Sequence[TaskItem], *, roles: Sequence[str],
                 initial: dict[str, dict[str, str]], policy_factories: dict[str, Callable[[str], Policy]],
                 fixtures: dict[str, Any], optimizer: Any, stances: dict[str, str | None] | None = None,
                 iterations: int = 2, meta_solver: str = "nash", search_kwargs: dict[str, Any] | None = None,
                 ground_truth: Sequence[GroundTruthScorer] | None = None, ctx: RunContext | None = None,
                 symmetric: Sequence[str] | None = None, repeats: int = 1, concurrency: int | None = None, seed: int = 0):
        self.mechanism, self.items, self.roles = mechanism, list(items), list(roles)
        self.populations = {r: dict(initial[r]) for r in self.roles}
        self.factories, self.fixtures, self.optimizer = policy_factories, dict(fixtures), optimizer
        self.stances = dict(stances or {})
        self.iterations, self.meta_solver = iterations, meta_solver
        self.search_kwargs = dict(search_kwargs or {})
        sym = list(symmetric or [])
        if [r for r in sym if r not in self.roles]:
            raise ValueError(f"symmetric roles {sym} must be among the evolving roles {self.roles}")
        for r in sym[1:]:
            if self.populations[r] != self.populations[sym[0]] or self.stances.get(r) != self.stances.get(sym[0]):
                raise ValueError(f"symmetric roles share one population: {r!r} must start with the same strategies "
                                 f"(names and texts) and stance as {sym[0]!r}")
        self.ground_truth, self.ctx, self.symmetric = ground_truth, ctx, sym if len(sym) > 1 else []
        self.repeats, self.concurrency, self.seed = repeats, concurrency, seed
        self.history: list[PSROIteration] = []
        self.games: list[NormalFormGame] = []
        self._policies: dict[tuple[str, str], Policy] = {}

    def _policy(self, role: str, name: str) -> Policy:
        key = (role, name)
        if key not in self._policies:
            self._policies[key] = self.factories[role](self.populations[role][name])
            self._policies[key].label = name
        return self._policies[key]

    async def _game(self) -> NormalFormGame:
        strategies = {r: {n: self._policy(r, n) for n in self.populations[r]} for r in self.roles}
        exp = EmpiricalGameExperiment(self.mechanism, self.items, strategies, fixtures=self.fixtures,
                                      stances=self.stances, repeats=self.repeats, ground_truth=self.ground_truth,
                                      ctx=self.ctx, concurrency=self.concurrency, seed=self.seed,
                                      symmetric=self.symmetric or None)
        await exp.arun()
        return exp.game(name=f"{self.mechanism.name}-psro{len(self.games)}")

    def _opponent_note(self, role: str, sigma: dict[str, np.ndarray]) -> str:
        lines = []
        for r in self.roles:
            if r == role:
                continue
            lines.append(f"Role '{r}' plays a mixture of strategies:")
            for (name, text), p in zip(self.populations[r].items(), sigma[r]):
                if p > 1e-3:
                    lines.append(f"  - with probability {p:.2f}: {text or '(default behaviour)'}")
        return "\n".join(lines)

    async def arun(self) -> list[PSROIteration]:
        prev_sizes: dict[str, int] | None = None
        prev_sigma: dict[str, np.ndarray] | None = None
        for it in range(self.iterations + 1):
            game = await self._game()
            self.games.append(game)
            clean = game.imputed()  # missing payoffs: each player's worst observed one (warns)
            # NashConv of the previous meta-strategy against the strategies added since (exploitability estimate)
            nash_conv_prev = None
            if prev_sigma is not None and prev_sizes is not None:
                padded = []
                for r in self.roles:
                    x = np.zeros(len(self.populations[r]))
                    x[: prev_sizes[r]] = prev_sigma[r]
                    padded.append(x)
                nash_conv_prev = clean.nash_conv(padded)
            sigma_list = solve_meta(clean, self.meta_solver, symmetric=self.symmetric)
            sigma = dict(zip(game.players, sigma_list))
            meta_value = {k: game.expected(sigma_list, key=k) for k in game.outcomes
                          if np.isfinite(game.outcomes[k]).all()}
            rec = PSROIteration(iteration=it, populations={r: list(self.populations[r]) for r in self.roles},
                                meta_strategy={r: sigma[r].tolist() for r in self.roles},
                                nash_conv_prev=nash_conv_prev, meta_value=meta_value)
            self.history.append(rec)
            if it == self.iterations:
                break
            prev_sizes = {r: len(self.populations[r]) for r in self.roles}
            prev_sigma = sigma
            for role in self.roles:
                if role in self.symmetric[1:]:
                    continue  # one population: its best response is searched once, as its first role
                others = dict(self.fixtures)
                for r in self.roles:
                    if r != role:
                        names = list(self.populations[r])
                        others[r] = MixturePolicy([self._policy(r, n) for n in names], sigma[r].tolist(), label=f"meta_{r}")
                search = PromptSearch(self.mechanism, self.items, role=role, policy_factory=self.factories[role],
                                      others={k: v for k, v in others.items()}, optimizer=self.optimizer,
                                      arms=[self.stances.get(role)], opponent_note=self._opponent_note(role, sigma),
                                      ground_truth=self.ground_truth, ctx=self.ctx, concurrency=self.concurrency,
                                      seed=self.seed + it, seed_strategies=[self.populations[role][n] for n in self.populations[role]],
                                      **self.search_kwargs)
                res: SearchResult = await search.arun()
                best = res.best
                name = f"br{it + 1}"
                info = {"best_reward": best.mean_reward, "best_value": best.mean_value, "strategy": best.strategy}
                for r in (self.symmetric if role in self.symmetric else [role]):
                    self.populations[r][name] = best.strategy
                    rec.searches[r] = info
        return self.history

    def run(self) -> list[PSROIteration]:
        return run_sync(self.arun())
