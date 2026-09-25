"""Instructed behaviour arms: the paired design behind ASD.

Honest and dishonest behaviour are simulated by assigning each agent role a stance - the true answer
in one arm, a false answer in the other - with ground truth known by construction. For two-agent
protocols (debate) the opponent takes a different answer; both side assignments are run, so each
debater is observed arguing for the truth and against it on every item.

Instructed arms are the cheapest sampler and a weak adversary: prompted liars may refuse, fail, or
lie clumsily. Use manipulation checks (:class:`~so_arena.core.ground_truth.PositionFollowed`), and
see :mod:`so_arena.samplers.pools` (natural behaviour under optimization) and
:mod:`so_arena.samplers.prompt_search` (searched-for behaviour) for stronger samplers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from so_arena.core.game import RunContext
from so_arena.core.ground_truth import GroundTruthScorer
from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Episode, Mechanism
from so_arena.core.policy import Policy
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.core.store import RunStore


def arm_profiles(mechanism: Mechanism, *, agent: Any, fixtures: dict[str, Any], arms: Sequence[str] = ("true", "false"),
                 agents: dict[str, Any] | None = None, all_false: bool = False,
                 items: Sequence[TaskItem] | None = None) -> list[Profile]:
    """Profiles that put every agent role on each arm.

    Args:
        agent: default policy (or model name) for agent roles.
        fixtures: policies for non-agent roles, keyed by role name or role kind (e.g. ``"judge"``).
        arms: stance specs for the evaluated role(s).
        agents: per-role agent policies overriding ``agent``.
        all_false: with more than two answers, add one arm per false answer (``false:0``, ``false:1``, ...)
            instead of a random false answer - needed for graded ASD over all answers. There are as
            many such arms as the most false answers of an item in ``items`` (8 without ``items``); on
            an item with fewer, the extra arms are skipped, so each false answer is argued exactly once.
        items: the items the profiles will run on (sizes the ``all_false`` arms).
    """
    from so_arena.mechanisms._common import NullPolicy

    specs = mechanism.role_specs()
    agent_roles = [r for r, s in specs.items() if s.kind == "agent" and s.trainable]
    base: dict[str, Any] = {}
    for r, s in specs.items():
        if r in agent_roles:
            continue
        pol = fixtures.get(r, fixtures.get(s.kind))
        if pol is None:
            if s.required:
                raise ValueError(f"no fixture policy for role {r!r} (kind {s.kind!r})")
            continue
        base[r] = PlayerSpec(policy=pol)

    def agent_policy(r: str) -> Any:
        if agents and r in agents:
            return agents[r]
        if not specs[r].required:
            return NullPolicy()
        return agent

    arm_list = list(arms)
    if all_false and "false" in arm_list:
        n_false = 8 if items is None else max(
            (len(it.false_labels) for it in items if it.true_label is not None), default=0)
        arm_list = [a for a in arm_list if a != "false"] + [f"false:{i}" for i in range(n_false)]

    profiles = []
    if len(agent_roles) <= 1:
        for arm in arm_list:
            players = dict(base)
            for r in agent_roles:
                players[r] = PlayerSpec(policy=agent_policy(r), stance=arm, label=f"argue_{arm.split(':')[0]}")
            profiles.append(Profile(name=f"arm={arm}", players=players, tags={"arm": arm}))
    else:
        # the first agent role takes each arm; the others argue for different answers. For binary
        # questions this yields both side assignments, so every agent role is observed on both arms.
        lead = agent_roles[0]
        for arm in arm_list:
            players = dict(base)
            players[lead] = PlayerSpec(policy=agent_policy(lead), stance=arm, label=f"argue_{arm.split(':')[0]}")
            for r in agent_roles[1:]:
                players[r] = PlayerSpec(policy=agent_policy(r), stance=f"opposite:{lead}", label="opponent")
            profiles.append(Profile(name=f"{lead}={arm}", players=players, tags={"arm": arm, "lead": lead}))
    return profiles


class ASDExperiment:
    """Run several mechanisms under instructed arms and compute ASD-family metrics.

    Example::

        exp = ASDExperiment([DirectJudge(), Consultancy(), Debate()], items,
                            agent=LLMPolicy("openai/gpt-4o"), fixtures={"judge": LLMPolicy("openai/gpt-4o-mini")})
        exp.run()
        exp.summary()
    """

    def __init__(self, mechanisms: Sequence[Mechanism], items: Sequence[TaskItem], *, agent: Any,
                 fixtures: dict[str, Any], arms: Sequence[str] = ("true", "false"), agents: dict[str, Any] | None = None,
                 ground_truth: Sequence[GroundTruthScorer] | None = None, ctx: RunContext | None = None,
                 repeats: int = 1, store: RunStore | str | None = None, concurrency: int | None = None,
                 all_false: bool = False, seed: int = 0):
        self.mechanisms = list(mechanisms)
        self.items = list(items)
        self.agent, self.fixtures, self.arms, self.agents = agent, fixtures, list(arms), agents
        self.ground_truth, self.ctx, self.repeats = ground_truth, ctx, repeats
        self.store = RunStore(store) if isinstance(store, str) else store
        self.concurrency, self.all_false, self.seed = concurrency, all_false, seed
        self.episodes: list[Episode] = []

    async def arun(self) -> list[Episode]:
        eps: list[Episode] = []
        for mech in self.mechanisms:
            profiles = arm_profiles(mech, agent=self.agent, fixtures=self.fixtures, arms=self.arms,
                                    agents=self.agents, all_false=self.all_false, items=self.items)
            eps += await run_episodes(mech, self.items, profiles, ctx=self.ctx, ground_truth=self.ground_truth,
                                      repeats=self.repeats, store=self.store, concurrency=self.concurrency,
                                      seed=self.seed)
        self.episodes = eps
        return eps

    def run(self) -> list[Episode]:
        return run_sync(self.arun())

    def frame(self) -> pd.DataFrame:
        from so_arena.analysis.frames import role_frame

        return role_frame(self.episodes)

    def summary(self, transforms: Sequence[str] = ("log", "brier")) -> pd.DataFrame:
        """ASD (per transform, re-scored post hoc), judge accuracy and IGR vs. the direct baseline."""
        from so_arena.analysis.metrics import asd_by_transform, incentive_gap_recovered, judge_accuracy

        a = asd_by_transform(self.episodes, transforms)
        ja = judge_accuracy(self.frame())
        out = a.merge(ja[["mechanism", "accuracy", "p_true"]], on="mechanism", how="left") if not ja.empty else a
        if "brier" in transforms and not a.empty:
            base = a[(a["transform"] == "brier") & (a["mechanism"] == "direct")]
            if not base.empty:
                b = float(base["asd"].iloc[0])
                out["igr_brier"] = [
                    incentive_gap_recovered(r.asd, b, 2.0) if r.transform == "brier" else float("nan")
                    for r in out.itertuples()
                ]
        return out


def _policy_label(p: Any) -> str:
    return p.label if isinstance(p, Policy) and p.label else str(p)
