"""Profile sources: which strategy profiles to run on each task.

These are the *behaviour samplers* at the level of an experiment grid; adaptive samplers
(best-of-N pools, prompt optimisation, PSRO) live in :mod:`oversight_arena.elicitation`.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.roles import RoleSpec
from ..core.strategy import Assignment, Profile, Stance, Strategy, argue
from ..core.task import Task


class ProfileSource(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    @abstractmethod
    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]: ...

    def __mul__(self, other: "ProfileSource") -> "ProfileSource":
        return ProductProfiles(sources=[self, other])


class Fixed(ProfileSource):
    """The same explicit profile list for every task."""

    items: list[Profile] = Field(default_factory=lambda: [Profile(label="default")])

    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]:
        return [p.model_copy(deep=True) for p in self.items]


class Stances(ProfileSource):
    """ASD-style worlds: each listed role argues the correct or an incorrect option.

    With ``distinct=True`` (debate), positioned roles take *different* options: for two
    options and roles [A, B] this yields (A=correct, B=incorrect) and (A=incorrect,
    B=correct). For a single role (consultancy/propaganda) it yields two worlds.
    For >2 options and ``all_options=True`` every option is used as a target
    (``Stance.OPTION``), which is what ASD needs for multi-option questions.
    """

    roles: list[str] = Field(default_factory=list)  # empty = all trainable 'expert' roles
    distinct: bool = True
    all_options: bool = False
    instructions: str | None = None
    extra: dict[str, Strategy] = Field(default_factory=dict)  # fixed strategies for other roles

    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]:
        if not self.roles:
            auto = [r.name for r in roles if r.trainable and r.kind in ("expert", "forecaster")]
            return self.model_copy(update={"roles": auto}).profiles(task, roles) if auto else [Profile(label="default")]
        base = {r: Assignment(strategy=s) for r, s in self.extra.items()}
        out: list[Profile] = []
        if self.all_options or len(task.options) > 2:
            ids = [o.id for o in task.options]
            combos = (
                itertools.permutations(ids, len(self.roles))
                if self.distinct
                else itertools.product(ids, repeat=len(self.roles))
            )
            good = set(task.correct_ids())
            for combo in combos:
                asg = dict(base)
                for r, oid in zip(self.roles, combo):
                    honest = oid in good if good else None
                    asg[r] = Assignment(
                        strategy=Strategy(
                            name=f"argue_{'correct' if honest else 'incorrect'}",
                            stance=Stance.OPTION,
                            option=oid,
                            instructions=self.instructions or argue().instructions,
                            tags={"honest": honest, "family": "asd"},
                        ),
                        position=oid,
                    )
                out.append(Profile(assignments=asg, label="|".join(f"{r}={o}" for r, o in zip(self.roles, combo))))
            return out
        stances = [Stance.CORRECT, Stance.INCORRECT]
        if len(self.roles) == 1 or not self.distinct:
            combos2 = itertools.product(stances, repeat=len(self.roles))
        else:
            combos2 = [
                tuple(stances[(i + k) % 2] for i in range(len(self.roles))) for k in range(2)
            ]
        for combo in combos2:
            asg = dict(base)
            for r, st in zip(self.roles, combo):
                asg[r] = Assignment(strategy=argue(st, self.instructions))
            out.append(Profile(assignments=asg, label="|".join(f"{r}={s.value}" for r, s in zip(self.roles, combo))))
        return out


class Cartesian(ProfileSource):
    """Full product over per-role strategy lists (for empirical-game / equilibrium analysis)."""

    strategies: dict[str, list[Strategy]]

    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]:
        names = list(self.strategies)
        out = []
        for combo in itertools.product(*[self.strategies[n] for n in names]):
            out.append(
                Profile(
                    assignments={n: Assignment(strategy=s) for n, s in zip(names, combo)},
                    label="|".join(f"{n}={s.name}" for n, s in zip(names, combo)),
                )
            )
        return out


class Seeds(ProfileSource):
    """``n`` independent samples (seeds) of the given strategies — best-of-N pools."""

    n: int = 8
    strategies: dict[str, Strategy] = Field(default_factory=dict)
    roles: list[str] | None = None  # roles whose seed varies (default: all in `strategies`)
    joint: bool = False  # if True, vary all listed roles' seeds jointly (n profiles), else product

    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]:
        vary = self.roles or list(self.strategies)
        if self.joint or len(vary) <= 1:
            return [
                Profile(
                    assignments={
                        r: Assignment(strategy=self.strategies.get(r, Strategy(name="default")), seed=(k if r in vary else 0))
                        for r in set(vary) | set(self.strategies)
                    },
                    label=f"seed={k}",
                )
                for k in range(self.n)
            ]
        out = []
        for ks in itertools.product(range(self.n), repeat=len(vary)):
            asg = {r: Assignment(strategy=self.strategies.get(r, Strategy(name="default")), seed=k) for r, k in zip(vary, ks)}
            for r, s in self.strategies.items():
                asg.setdefault(r, Assignment(strategy=s))
            out.append(Profile(assignments=asg, label=",".join(f"{r}#{k}" for r, k in zip(vary, ks))))
        return out


class GameTree(ProfileSource):
    """Nested samples for game-tree analyses of sequential protocols (as in *Debate with self-play
    best-of-N optimisation*): e.g. ``levels=[("proposer", "proposal", 8), ("critic", "critique", 8),
    ("proposer", "rebuttal", 4)]`` yields 8·8·4 profiles; with a caching model the proposal for
    index j is shared by all its descendants, critique k is drawn given proposal j, and so on —
    so the episodes form trees. Analyse with
    :func:`~oversight_arena.analysis.bon.trees_from_results` + :func:`tree_value` / :func:`tree_mesh`.
    """

    levels: list[tuple[str, str, int]]
    strategies: dict[str, Strategy] = Field(default_factory=dict)

    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]:
        out = []
        for idx in itertools.product(*[range(n) for _, _, n in self.levels]):
            steps: dict[str, dict[str, int]] = {}
            for (role, step, _), k in zip(self.levels, idx):
                steps.setdefault(role, {})[step] = k
            asg = {r: Assignment(strategy=self.strategies.get(r, Strategy(name="default")), step_seeds=st)
                   for r, st in steps.items()}
            for r, strat in self.strategies.items():
                asg.setdefault(r, Assignment(strategy=strat))
            out.append(Profile(assignments=asg, label="/".join(f"{step}#{k}" for (_, step, _), k in zip(self.levels, idx))))
        return out


class ProductProfiles(ProfileSource):
    """Merge profiles from several sources (later sources override earlier per role)."""

    sources: list[ProfileSource]

    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]:
        lists = [s.profiles(task, roles) for s in self.sources]
        out = []
        for combo in itertools.product(*lists):
            asg: dict[str, Assignment] = {}
            for p in combo:
                asg.update({k: v.model_copy(deep=True) for k, v in p.assignments.items()})
            out.append(Profile(assignments=asg, label=" & ".join(p.label for p in combo if p.label)))
        return out


class MapProfiles(ProfileSource):
    """Transform another source's profiles, e.g. add strategy params to every role::

        MapProfiles(source=Stances(roles=[...]), params={"style": "cherry_pick"}, suffix="/cherry")
    """

    source: ProfileSource
    params: dict[str, Any] = Field(default_factory=dict)  # merged into every strategy's params
    roles: list[str] | None = None  # restrict to these roles
    suffix: str = ""
    fn: Callable[[Profile], Profile] | None = None

    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]:
        out = []
        for p in self.source.profiles(task, roles):
            p = p.model_copy(deep=True)
            for r, a in p.assignments.items():
                if self.roles is not None and r not in self.roles:
                    continue
                if self.params:
                    a.strategy = a.strategy.with_(params={**a.strategy.params, **self.params}, name=a.strategy.name + self.suffix)
            if self.suffix:
                p.label += self.suffix
            out.append(self.fn(p) if self.fn else p)
        return out


class FunctionProfiles(ProfileSource):
    fn: Callable[[Task, list[RoleSpec]], Sequence[Profile]]

    def profiles(self, task: Task, roles: list[RoleSpec]) -> list[Profile]:
        return list(self.fn(task, roles))


def as_profile_source(p: "ProfileSource | Sequence[Profile] | Profile | None") -> ProfileSource:
    if p is None:
        return Fixed()
    if isinstance(p, ProfileSource):
        return p
    if isinstance(p, Profile):
        return Fixed(items=[p])
    return Fixed(items=list(p))
