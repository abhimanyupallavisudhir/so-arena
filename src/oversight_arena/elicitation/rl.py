"""Reinforcement learning under a mechanism.

Three levels of fidelity:

1. :class:`StrategyGradient` — *strategy-level* RL: each trainable role holds a softmax policy
   over a population of strategies (prompts / parameterised behaviours) and is updated from
   real episodes, by REINFORCE (vanilla policy gradient) or by natural policy gradient
   (≈ exponential weights, whose mean-field limit is the replicator dynamics of
   :meth:`EmpiricalGame.replicator`). Cheap (no weights), multi-agent, and it exhibits the
   learning dynamics that decide which equilibrium training selects.
2. :class:`MechanismEnv` — a text environment (PettingZoo-AEC-like) exposing any mechanism as
   an RL environment for one or more trainable roles: ``reset()`` → observation of the role to
   act; ``step(text)`` → next observation; final rewards from the mechanism's reward rule. Plug
   it into your RL stack (TRL, verl, OpenRLHF, verifiers...) for weight-level training.
3. :func:`reward_function` — a TRL-GRPO-style ``reward_funcs`` callable for single-turn roles,
   and :func:`preference_pairs` — DPO data from mechanism preferences (with GT agreement).
"""

from __future__ import annotations

import asyncio
import json
import math
import queue
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..agents.base import Action, Agent, Observation
from ..core.episode import EpisodeRecord
from ..core.strategy import Assignment, Profile, Strategy, argue
from ..core.task import Task
from ..core.transcript import Transcript
from ..core.util import gather_limited, rng_for
from ..domains.base import Domain
from ..experiment.runner import AgentTable, run_episode
from ..mechanisms.base import Mechanism


class StrategyGradient:
    r"""Independent softmax-policy learners over strategy populations, trained on real episodes.

    With ``natural=False`` (REINFORCE) the expected logit update is
    $\Delta\theta_a = \eta\,\pi_a (u_a - \bar u)$: rarely played strategies learn slowly. With
    ``natural=True`` (natural policy gradient for the softmax, i.e. exponential weights / Hedge)
    it is $\eta\,(u_a - \bar u)$, estimated without bias by importance weighting,
    $\hat g_a = \frac1B \sum_j \mathbb 1[a_j = a](r_j - b)/\pi_a$ (as in EXP3) — the stochastic
    version of the replicator dynamics. Both have the same rest points but can reach *different*
    equilibria from the same start, so equilibrium selection depends on the training algorithm.

    Args:
        populations: per trainable role, its strategy set (the policy's support).
        fixed: strategies for other roles.
        lr: learning rate on logits; ``baseline``: 'mean' (batch mean) or 'none'.
        batch: episodes per iteration (tasks sampled with replacement).
        init: optional initial probabilities per role (e.g. start near 'silent' to test
            path dependence / basins of attraction).
        natural: natural policy gradient (see above) instead of REINFORCE.
    """

    def __init__(
        self,
        domain: Domain,
        mechanism: Mechanism,
        agents: "AgentTable | dict[str, Any] | Any",
        populations: dict[str, list[Strategy]],
        fixed: dict[str, Strategy] | None = None,
        lr: float = 0.5,
        batch: int = 16,
        iterations: int = 20,
        baseline: str = "mean",
        init: dict[str, Sequence[float]] | None = None,
        tasks: Sequence[Task] | None = None,
        gt_keys: Sequence[str] = ("correct",),
        seed: int = 0,
        concurrency: int = 8,
        natural: bool = False,
    ):
        self.domain = domain
        self.mechanism = mechanism
        self.agents = agents if isinstance(agents, AgentTable) else AgentTable(agents)
        self.natural = natural
        self.pop = populations
        self.fixed = fixed or {}
        self.lr = lr
        self.batch = batch
        self.iterations = iterations
        self.baseline = baseline
        self.tasks = list(tasks) if tasks is not None else domain.tasks()
        self.gt_keys = list(gt_keys)
        self.seed = seed
        self.concurrency = concurrency
        self.logits = {
            r: (np.log(np.asarray(init[r], float) + 1e-12) if init and r in init else np.zeros(len(p)))
            for r, p in populations.items()
        }
        self.records: list[EpisodeRecord] = []

    def probs(self, role: str) -> np.ndarray:
        z = self.logits[role] - self.logits[role].max()
        e = np.exp(z)
        return e / e.sum()

    async def run(self) -> pd.DataFrame:
        rows = []
        for it in range(self.iterations + 1):
            rng = rng_for("sg", self.seed, it)
            jobs = []
            for b in range(self.batch):
                t = self.tasks[rng.randrange(len(self.tasks))]
                choice = {r: rng.choices(range(len(p)), weights=list(self.probs(r)))[0] for r, p in self.pop.items()}
                asg = {r: Assignment(strategy=self.pop[r][i], seed=it * 1000 + b) for r, i in choice.items()}
                asg.update({r: Assignment(strategy=s, seed=it * 1000 + b) for r, s in self.fixed.items()})
                jobs.append((t, Profile(assignments=asg), choice, it * 1000 + b))

            def make(t, prof, s):
                return lambda: run_episode(self.mechanism, t, prof, self.agents, self.domain, seed=s)

            recs = await gather_limited([make(t, p, s) for t, p, _, s in jobs], self.concurrency)
            self.records.extend(recs)
            row: dict[str, Any] = {"iteration": it}
            for r in self.pop:
                pr = self.probs(r)
                for i, s in enumerate(self.pop[r]):
                    row[f"p[{r}:{s.name}]"] = float(pr[i])
                rs = np.array([rec.rewards.get(r, np.nan) for rec in recs], float)
                row[f"reward[{r}]"] = float(np.nanmean(rs))
                for k in self.gt_keys:
                    vals = [rec.gt.get(k, {}).get(r) for rec in recs]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        row[f"gt_{k}[{r}]"] = float(np.mean(vals))
            for k in self.gt_keys:
                outs = [rec.gt.get(k, {}).get("_outcome") for rec in recs]
                outs = [v for v in outs if v is not None]
                if outs:
                    row[f"gt_{k}[_outcome]"] = float(np.mean(outs))
            rows.append(row)
            if it == self.iterations:
                break
            for r in self.pop:
                rs = np.array([rec.rewards.get(r, np.nan) for rec in recs], float)
                ok = ~np.isnan(rs)
                if not ok.any():
                    continue
                base = float(np.mean(rs[ok])) if self.baseline == "mean" else 0.0
                pr = self.probs(r)
                grad = np.zeros_like(pr)
                if self.natural:  # importance-weighted advantages: E[grad_a] = u_a - u_bar for every a, rare or not
                    picks = np.array([choice[r] for (_, _, choice, _) in jobs])
                    n = int(ok.sum())
                    for a in range(len(pr)):
                        m = ok & (picks == a)
                        if m.any():
                            grad[a] = float(np.sum(rs[m] - base)) / (n * pr[a])
                    self.logits[r] = self.logits[r] + self.lr * grad
                    continue
                for (_, _, choice, _), rv, good in zip(jobs, rs, ok):  # REINFORCE: (r - b) grad log pi(a)
                    if not good:
                        continue
                    g = -pr.copy()
                    g[choice[r]] += 1.0
                    grad += (rv - base) * g
                self.logits[r] = self.logits[r] + self.lr * grad / max(1, ok.sum())
        return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ text env
class _External(Agent):
    """Agent whose actions come from outside (the RL learner) via queues."""

    def __init__(self, role: str, out_q: "queue.Queue", in_q: "queue.Queue"):
        self.id = f"external:{role}"
        self.role = role
        self.out_q = out_q
        self.in_q = in_q

    async def act(self, obs: Observation) -> Action:
        self.out_q.put(("obs", self.role, obs))
        loop = asyncio.get_running_loop()
        act = await loop.run_in_executor(None, self.in_q.get)
        if isinstance(act, Action):
            return act
        if isinstance(act, dict):
            return Action(text=act.get("text", ""), parsed={k: v for k, v in act.items() if k != "text"})
        return Action(text=str(act))


class MechanismEnv:
    """Turn-based text environment for training roles under a mechanism.

    Example::

        env = MechanismEnv(domain, Debate(), fixtures={"judge": judge_agent}, trainable=["debater_a", "debater_b"])
        role, obs = env.reset(task)
        while role is not None:
            text = my_policy(env.render(obs))        # LLM generation
            role, obs = env.step(text)
        env.rewards   # {'debater_a': ..., 'debater_b': ...}; env.record has everything

    Structured replies (e.g. a judge's distribution) can be passed as dicts:
    ``env.step({"text": "...", "probs": {...}})``. ``render(obs)`` gives the default chat
    messages an LLM agent would see.
    """

    def __init__(self, domain: Domain, mechanism: Mechanism, fixtures: dict[str, Agent], trainable: Sequence[str],
                 profile: Profile | None = None, gt: Any = None):
        self.domain = domain
        self.mechanism = mechanism
        self.fixtures = fixtures
        self.trainable = list(trainable)
        self.profile = profile
        self.gt = gt
        self.record: EpisodeRecord | None = None
        self._thread: threading.Thread | None = None

    def reset(self, task: Task | None = None, profile: Profile | None = None, seed: int = 0) -> tuple[str | None, Observation | None]:
        task = task or self.domain.tasks()[0]
        self._out: queue.Queue = queue.Queue()
        self._ins = {r: queue.Queue() for r in self.trainable}
        agents = dict(self.fixtures)
        for r in self.trainable:
            agents[r] = _External(r, self._out, self._ins[r])
        prof = profile or self.profile or Profile()

        def runner():
            rec = asyncio.run(run_episode(self.mechanism, task, prof, agents, self.domain, gt=self.gt, seed=seed))
            self._out.put(("done", None, rec))

        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        return self._next()

    def _next(self) -> tuple[str | None, Observation | None]:
        kind, role, payload = self._out.get()
        if kind == "done":
            self.record = payload
            return None, None
        self._pending = role
        return role, payload

    def step(self, action: "str | dict | Action") -> tuple[str | None, Observation | None]:
        self._ins[self._pending].put(action)
        return self._next()

    @property
    def rewards(self) -> dict[str, float]:
        return dict(self.record.rewards) if self.record else {}

    @staticmethod
    def render(obs: Observation) -> list[dict[str, str]]:
        from ..agents.llm import render_observation

        return [{"role": m.role, "content": m.content} for m in render_observation(obs)]


def _row_assignment(i: int, kw: dict[str, Any]) -> Assignment | None:
    """The trainable role's assignment for completion ``i`` from per-row dataset columns:
    ``strategy`` (a Strategy or its dict), ``stance`` (+ ``option`` for OPTION) or ``position``."""

    def col(k: str) -> Any:
        v = kw.get(k)
        return v[i] if isinstance(v, (list, tuple)) and i < len(v) else None

    strat, stance, pos = col("strategy"), col("stance"), col("position")
    if strat is not None:
        s = strat if isinstance(strat, Strategy) else Strategy.model_validate(json.loads(strat) if isinstance(strat, str) else strat)
        return Assignment(strategy=s, position=pos)
    if stance is not None:
        s = argue(stance)
        if col("option") is not None:
            s = s.model_copy(update={"option": col("option")})
        return Assignment(strategy=s, position=pos)
    if pos is not None:
        return Assignment(position=pos)
    return None


def reward_function(domain: Domain, mechanism: Mechanism, role: str, fixtures: dict[str, Agent],
                    profile: Profile | None = None) -> Any:
    """A TRL-GRPO-compatible reward function for a *single-turn* trainable role.

    Returns ``fn(prompts, completions, task_id=[...], **columns) -> list[float]``: each completion
    is used as ``role``'s (only) move; the rest of the mechanism runs with ``fixtures`` and
    ``profile``. What the completion was asked to do comes from the same dataset row, so it is
    scored at the position it argued: pass a ``stance`` column (``"correct"`` / ``"incorrect"``,
    or ``"option"`` with an ``option`` column), a ``position`` column (option id) or a
    ``strategy`` column. Build prompts with the same strategy (e.g. ``oa.argue(stance)``).
    """
    tasks = {t.id: t for t in domain.tasks()}
    base = profile or Profile()

    def fn(prompts: list[Any], completions: list[Any], task_id: list[str] | None = None, **kw: Any) -> list[float]:
        assert task_id is not None, "pass task ids via the dataset column 'task_id'"

        async def one(i: int, c: Any, tid: str) -> float:
            text = c if isinstance(c, str) else (c[-1]["content"] if isinstance(c, list) else str(c))

            class _Fixed(Agent):
                id = "completion"

                async def act(self, obs: Observation) -> Action:
                    from ..agents.llm import LLMAgent

                    parsed, _ = LLMAgent._parse(None, text, obs.response)  # type: ignore[arg-type]
                    return Action(text=text, parsed=parsed)

            asg = _row_assignment(i, kw)
            prof = base if asg is None else base.model_copy(update={"assignments": {**base.assignments, role: asg}})
            agents = {**fixtures, role: _Fixed()}
            rec = await run_episode(mechanism, tasks[tid], prof, agents, domain)
            return float(rec.rewards.get(role, math.nan))

        async def all_():
            return await asyncio.gather(*[one(i, c, t) for i, (c, t) in enumerate(zip(completions, task_id))])

        return asyncio.run(all_())

    return fn


def _pair_context(r: EpisodeRecord, role: str) -> tuple:
    """What a preference pair must share: mechanism configuration, task, the role's assigned
    position, and everything about the other roles (strategy, agent, position, sample)."""
    me = r.bound.get(role)
    others = tuple(sorted((q, b.strategy_id, b.agent, b.target, b.seed) for q, b in r.bound.items() if q != role))
    return (r.mechanism_hash or r.mechanism, r.task_id, me.target if me else None, others)


def _prompt_text(r: EpisodeRecord, role: str) -> str:
    """The shared context of a pair: the role's position and what it saw before its first turn."""
    me = r.bound.get(role)
    first = next((i for i, e in enumerate(r.transcript.entries) if e.role == role), len(r.transcript.entries))
    before = Transcript(entries=[e for e in r.transcript.entries[:first] if e.visible(role)])
    head = f"Task {r.task_id} ({r.mechanism}); role {role}" + (f", arguing for {me.target}" if me and me.target else "")
    seen = before.render(for_role=role)
    return head + ("\n\n" + seen if seen else "")


def preference_pairs(results: Any, role: str, gt: str = "correct", min_gap: float = 0.0) -> pd.DataFrame:
    """(prompt, chosen, rejected) pairs ranked by mechanism reward, with a column saying whether
    the ground truth agrees — the quality of the preference data this mechanism would feed to
    DPO/RLHF. Only episodes with the same context are paired (:func:`_pair_context`: mechanism
    configuration, task, the role's position, the other roles' strategies and samples), so a pair
    differs only in how ``role`` behaved. For a multi-turn role the texts are its whole turns."""
    recs = results.records if hasattr(results, "records") else results
    groups: dict[tuple, list[EpisodeRecord]] = {}
    for r in recs:
        if r.error is None and role in r.rewards and not math.isnan(r.rewards[role]):
            groups.setdefault(_pair_context(r, role), []).append(r)
    rows = []
    for (mech, tid, target, _), rs in groups.items():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if abs(a.rewards[role] - b.rewards[role]) <= min_gap:
                    continue
                ch, rj = (a, b) if a.rewards[role] > b.rewards[role] else (b, a)
                gc, gr = ch.gt.get(gt, {}).get(role), rj.gt.get(gt, {}).get(role)
                rows.append({
                    "task": tid, "mechanism": ch.mechanism, "mechanism_hash": mech, "position": target,
                    "prompt": _prompt_text(ch, role),
                    "chosen": "\n".join(e.content for e in ch.transcript.entries if e.role == role),
                    "rejected": "\n".join(e.content for e in rj.transcript.entries if e.role == role),
                    "reward_gap": ch.rewards[role] - rj.rewards[role],
                    "gt_chosen": gc, "gt_rejected": gr,
                    "gt_agrees": None if gc is None or gr is None or gc == gr else bool(gc > gr),
                })
    return pd.DataFrame(rows)
