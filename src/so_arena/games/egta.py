"""Empirical game-theoretic analysis: estimate a normal-form game over strategy sets by simulation."""

from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from so_arena.core.game import RunContext
from so_arena.core.ground_truth import GroundTruthScorer
from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Episode, Mechanism
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.core.store import RunStore
from so_arena.games.normal_form import NormalFormGame


def gt_welfare(ep: Episode) -> float | None:
    """Mean ground-truth value over the episode's trainable roles (what the principal cares about)."""
    vals = [v for r, v in (ep.ground_truth.get("role_values") or {}).items() if r in ep.trainable_roles and v is not None]
    return float(np.mean(vals)) if vals else None


def reward_welfare(ep: Episode) -> float | None:
    vals = [v for r, v in ep.rewards.items() if r in ep.trainable_roles and v is not None]
    return float(np.mean(vals)) if vals else None


def _strategy_config(policy: Any) -> Any:
    """What a strategy is, for comparing it across roles: the policy's description without its label
    and id (naming), or the model name it is built from."""
    from so_arena.core.mechanism import describe_config
    from so_arena.core.policy import Policy

    if isinstance(policy, Policy):
        return describe_config({k: v for k, v in policy.describe().items() if k not in ("id", "label")})
    return describe_config(policy)


DEFAULT_OUTCOMES: dict[str, Callable[[Episode], float | None]] = {
    "gt_welfare": gt_welfare,
    "reward_welfare": reward_welfare,
    "outcome_value": lambda ep: ep.ground_truth.get("outcome_value"),
    "judge_correct": lambda ep: ep.ground_truth.get("judge_correct"),
}


class EmpiricalGameExperiment:
    """Simulate every strategy profile and assemble a :class:`NormalFormGame`.

    Args:
        strategies: role -> {strategy name -> policy (or model name)}.
        fixtures: policies for roles not being varied (judges, graders...).
        stances: optional fixed stance per role.
        symmetric: roles that are exchangeable (same strategy set, position-independent mechanism);
            only multisets of strategies are simulated and payoffs are shared by strategy type - so a
            strategy name must mean the same policy for each of these roles (checked on the policies'
            descriptions, labels aside), or unsimulated profiles would take another strategy's payoffs.
        outcomes: extra per-episode statistics to average per profile (defaults: ground-truth welfare,
            mean reward, outcome value, judge accuracy).
    """

    def __init__(self, mechanism: Mechanism, items: Sequence[TaskItem], strategies: dict[str, dict[str, Any]], *,
                 fixtures: dict[str, Any] | None = None, stances: dict[str, str | None] | None = None,
                 symmetric: Sequence[str] | None = None, repeats: int = 1,
                 ground_truth: Sequence[GroundTruthScorer] | None = None, ctx: RunContext | None = None,
                 store: RunStore | None = None, concurrency: int | None = None,
                 outcomes: dict[str, Callable[[Episode], float | None]] | None = None, seed: int = 0):
        self.mechanism, self.items = mechanism, list(items)
        self.strategies = {r: dict(s) for r, s in strategies.items()}
        self.players = list(self.strategies)
        self.fixtures = dict(fixtures or {})
        self.stances = dict(stances or {})
        self.symmetric = list(symmetric or [])
        if self.symmetric:
            names = {tuple(self.strategies[r]) for r in self.symmetric}
            if len(names) != 1:
                raise ValueError("symmetric roles must share the same strategy set")
            first = self.symmetric[0]
            for r in self.symmetric[1:]:
                for name, pol in self.strategies[r].items():
                    if _strategy_config(pol) != _strategy_config(self.strategies[first][name]):
                        raise ValueError(f"symmetric roles must share the same strategies: {name!r} is a different "
                                         f"policy for {r!r} than for {first!r}")
        self.repeats, self.ground_truth, self.ctx, self.store = repeats, ground_truth, ctx, store
        self.concurrency, self.seed = concurrency, seed
        self.outcome_fns = {**DEFAULT_OUTCOMES, **(outcomes or {})}
        self.episodes: list[Episode] = []

    def _profiles(self) -> list[Profile]:
        asym = [r for r in self.players if r not in self.symmetric]
        asym_choices = itertools.product(*[list(self.strategies[r]) for r in asym])
        sym_names = list(self.strategies[self.symmetric[0]]) if self.symmetric else []
        sym_choices = list(itertools.combinations_with_replacement(sym_names, len(self.symmetric))) if self.symmetric else [()]
        profiles = []
        for a in asym_choices:
            for s in sym_choices:
                assign = dict(zip(asym, a)) | dict(zip(self.symmetric, s))
                players = {r: PlayerSpec(policy=self.fixtures[r]) for r in self.fixtures}
                for r, name in assign.items():
                    players[r] = PlayerSpec(policy=self.strategies[r][name], stance=self.stances.get(r), label=name)
                pname = "|".join(f"{r}={assign[r]}" for r in self.players)
                profiles.append(Profile(name=pname, players=players, tags={"profile_assign": assign}))
        return profiles

    async def arun(self) -> list[Episode]:
        self.episodes = await run_episodes(self.mechanism, self.items, self._profiles(), ctx=self.ctx,
                                           ground_truth=self.ground_truth, repeats=self.repeats, store=self.store,
                                           concurrency=self.concurrency, seed=self.seed)
        return self.episodes

    def run(self) -> list[Episode]:
        return run_sync(self.arun())

    def game(self, episodes: Sequence[Episode] | None = None, name: str | None = None) -> NormalFormGame:
        eps = [e for e in (episodes if episodes is not None else self.episodes) if e.error is None]
        shape = tuple(len(self.strategies[r]) for r in self.players)
        index = {r: {n: i for i, n in enumerate(self.strategies[r])} for r in self.players}
        # collect per-profile samples
        pay: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        outc: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for ep in eps:
            assign = ep.tags.get("profile_assign") or {r: ep.players[r].label for r in self.players}
            by_type: dict[str, list[float]] = defaultdict(list)
            for r in self.players:
                v = ep.rewards.get(r)
                if v is not None:
                    if r in self.symmetric:
                        by_type[assign[r]].append(v)
            prof_samples: list[tuple[tuple, dict[str, float]]] = []
            if self.symmetric:
                sym_strats = [assign[r] for r in self.symmetric]
                for perm in set(itertools.permutations(sym_strats)):
                    a2 = dict(assign) | dict(zip(self.symmetric, perm))
                    prof = tuple(index[r][a2[r]] for r in self.players)
                    u = {}
                    for r in self.players:
                        if r in self.symmetric:
                            vals = by_type.get(a2[r])
                            u[r] = float(np.mean(vals)) if vals else np.nan
                        else:
                            u[r] = ep.rewards.get(r, np.nan)
                    prof_samples.append((prof, u))
            else:
                prof = tuple(index[r][assign[r]] for r in self.players)
                prof_samples.append((prof, {r: ep.rewards.get(r, np.nan) for r in self.players}))
            ovals = {k: fn(ep) for k, fn in self.outcome_fns.items()}
            for prof, u in prof_samples:
                for r, v in u.items():
                    if v is not None and np.isfinite(v):
                        pay[prof][r].append(float(v))
                for k, v in ovals.items():
                    if v is not None and np.isfinite(v):
                        outc[prof][k].append(float(v))
        U = {r: np.full(shape, np.nan) for r in self.players}
        SE = {r: np.full(shape, np.nan) for r in self.players}
        counts = np.zeros(shape, dtype=int)
        O = {k: np.full(shape, np.nan) for k in self.outcome_fns}
        for prof in itertools.product(*[range(s) for s in shape]):
            for r in self.players:
                vals = pay[prof][r]
                if vals:
                    U[r][prof] = np.mean(vals)
                    SE[r][prof] = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
            counts[prof] = max((len(pay[prof][r]) for r in self.players), default=0)
            for k in self.outcome_fns:
                if outc[prof][k]:
                    O[k][prof] = np.mean(outc[prof][k])
        missing = int(np.isnan(np.stack(list(U.values()))).sum())
        if missing:
            import logging

            logging.getLogger("so_arena").warning("%d payoff entries missing (unsimulated or errored profiles)", missing)
        return NormalFormGame(self.players, {r: list(self.strategies[r]) for r in self.players}, U, stderr=SE,
                              counts=counts, outcomes=O, name=name or self.mechanism.name)
