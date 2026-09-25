"""Running episodes and experiments."""

from __future__ import annotations

import asyncio
import fnmatch
import json
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..agents.base import Agent
from ..core.episode import BoundInfo, EpisodeRecord
from ..core.roles import RoleSpec
from ..core.strategy import Profile
from ..core.task import Task
from ..core.util import now_iso, read_jsonl, stable_hash, write_jsonl
from ..domains.base import Domain, Environment
from ..ground_truth.base import GTScorer, compute_gt, stale_scorers
from ..mechanisms.base import EpisodeContext, Mechanism
from .profiles import ProfileSource, as_profile_source
from .results import Results


class AgentTable:
    """Maps roles to agents. Keys may be exact role names, glob patterns (``debater_*``),
    role kinds (``kind:judge``) or ``"*"``. Assignments can pick an agent by key."""

    def __init__(self, agents: "dict[str, Agent] | Agent"):
        self.agents = agents if isinstance(agents, dict) else {"*": agents}

    def resolve(self, spec: RoleSpec, key: str | None = None) -> Agent:
        if key is not None:
            if key in self.agents:
                return self.agents[key]
            raise KeyError(f"agent key {key!r} not in agent table {list(self.agents)}")
        if spec.name in self.agents:
            return self.agents[spec.name]
        for pat, a in self.agents.items():
            if any(c in pat for c in "*?[") and pat != "*" and fnmatch.fnmatch(spec.name, pat):
                return a
        if f"kind:{spec.kind}" in self.agents:
            return self.agents[f"kind:{spec.kind}"]
        if "*" in self.agents:
            return self.agents["*"]
        raise KeyError(f"No agent for role {spec.name!r} (kind {spec.kind}); keys: {list(self.agents)}")

    def describe(self) -> dict[str, Any]:
        return {k: (a.describe() if hasattr(a, "describe") else repr(a)) for k, a in self.agents.items()}


def task_fingerprint(task: Task) -> str:
    """Content hash of a task (resources whose key starts with ``_`` — e.g. live checker
    objects — are excluded)."""
    d = task.model_dump(exclude={"resources"})
    d["resources"] = {k: v for k, v in task.resources.items() if not str(k).startswith("_")}
    return stable_hash(d, length=16)


def episode_key(mechanism: Mechanism, task: Task, profile: Profile, agents: dict[str, Agent], seed: int,
                domain: Domain | None = None, clearances: dict[str, Sequence[str]] | None = None) -> str:
    """Deterministic identity of an episode: everything that can change what happens in it
    (mechanism config, task content, strategies, agents, seed, domain config, clearances)."""
    return stable_hash(
        mechanism.config_hash(),
        task_fingerprint(task),
        profile.id,
        {r: a.describe() for r, a in sorted(agents.items())},
        seed,
        domain.describe() if domain is not None else None,
        {k: sorted(v) for k, v in sorted((clearances or {}).items())},
        length=20,
    )


def _error_record(mechanism: Mechanism, task: Task, profile: Profile, error: str, experiment: str | None, seed: int) -> EpisodeRecord:
    key = stable_hash("error", mechanism.config_hash(), task.id, profile.id, seed, length=20)
    return EpisodeRecord(
        id=stable_hash(key, time.time_ns(), length=16), key=key, experiment=experiment, mechanism=mechanism.display_name,
        mechanism_config=mechanism.config(), mechanism_hash=mechanism.config_hash(), reward_rule=mechanism.reward.name,
        task_id=task.id, domain=task.domain, profile=profile, roles=mechanism.roles(), error=error, seed=seed,
        created_at=now_iso(),
    )


async def run_episode(
    mechanism: Mechanism,
    task: Task,
    profile: Profile | None = None,
    agents: "AgentTable | dict[str, Agent] | Agent | None" = None,
    domain: Domain | None = None,
    *,
    gt: Sequence[GTScorer] | None = None,
    clearances: dict[str, Sequence[str]] | None = None,
    seed: int = 0,
    experiment: str | None = None,
) -> EpisodeRecord:
    """Run one episode of ``mechanism`` on ``task`` with strategy ``profile``."""
    profile = profile or Profile()
    table = agents if isinstance(agents, AgentTable) else AgentTable(agents or {})
    roles = mechanism.roles()
    bound = {}
    agent_map: dict[str, Agent] = {}
    try:
        for spec in roles:
            a = profile.get(spec.name)
            bound[spec.name] = a.strategy.bind(task, spec.name, seed=a.seed, position=a.position, step_seeds=a.step_seeds)
            agent_map[spec.name] = table.resolve(spec, a.agent)
    except Exception as e:  # a malformed strategy or missing agent must not abort a whole experiment
        return _error_record(mechanism, task, profile, f"setup failed: {type(e).__name__}: {e}", experiment, seed)
    key = episode_key(mechanism, task, profile, agent_map, seed, domain, clearances)
    clear: dict[str, set[str]] = {}
    for spec in roles:
        if clearances is not None and spec.name in clearances:
            clear[spec.name] = set(clearances[spec.name])
        elif clearances is not None and f"kind:{spec.kind}" in clearances:
            clear[spec.name] = set(clearances[f"kind:{spec.kind}"])
        elif domain is not None:
            clear[spec.name] = domain.default_clearance(spec)
        else:
            clear[spec.name] = set()
        clear[spec.name] |= set(bound[spec.name].clearance)  # granted by the simulated behaviour
    env: Environment = domain.make_env(task) if domain is not None else Environment()
    verifiers = domain.verifiers(task) if domain is not None else []
    ctx = EpisodeContext(
        mechanism=mechanism, task=task, agents=agent_map, bound=bound, clearances=clear,
        domain=domain, env=env, verifiers=verifiers, episode_key=key, seed=seed,
    )
    t0 = time.time()
    error = None
    try:
        await env.setup()
        await mechanism.run(ctx)
    except Exception as e:  # keep the partial record
        error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=6)}"
    finally:
        try:
            env_state = env.state()
        except Exception as e:  # pragma: no cover
            env_state = {"_error": str(e)}
        await env.teardown()
    rec = EpisodeRecord(
        id=stable_hash(key, now_iso(), time.time_ns(), length=16),
        key=key,
        experiment=experiment,
        mechanism=mechanism.display_name,
        mechanism_config=mechanism.config(),
        mechanism_hash=mechanism.config_hash(),
        reward_rule=mechanism.reward.name,
        task_id=task.id,
        domain=task.domain,
        profile=profile,
        bound={
            r: BoundInfo(
                agent=agent_map[r].id, strategy_id=b.strategy_id, strategy_name=b.name, stance=b.stance,
                target=b.target, tags=b.tags, seed=b.seed, step_seeds=b.step_seeds,
            )
            for r, b in bound.items()
        },
        roles=roles,
        transcript=ctx.transcript,
        outcome=ctx.outcome,
        channels=ctx.channel_uses,
        usage=ctx.usage,
        env_state=env_state,
        error=error,
        seed=seed,
        created_at=now_iso(),
        elapsed_s=round(time.time() - t0, 3),
        meta=ctx.meta,
    )
    if error is None and not type(mechanism.reward).batch:
        try:
            if type(mechanism.reward).delayed:
                if task.resolved:
                    rec.rewards = {k: float(v) for k, v in mechanism.reward.score(rec, task).items()}  # type: ignore[attr-defined]
                else:
                    rec.meta["rewards_pending"] = True
            else:
                rec.rewards = {k: float(v) for k, v in mechanism.reward(rec).items()}
        except Exception as e:
            rec.error = f"reward rule failed: {type(e).__name__}: {e}"
    scorers = list(gt) if gt is not None else (domain.gt_scorers() if domain is not None else [])
    if error is None:
        await compute_gt(task, rec, scorers)
    return rec


class Experiment:
    """A grid of episodes: tasks × mechanisms × profiles (× repeats), run concurrently.

    Results are appended to ``<out>/episodes.jsonl`` as they complete; re-running an
    experiment with the same ``out`` resumes (episodes with an existing key are skipped).
    """

    def __init__(
        self,
        domain: Domain,
        mechanisms: Mechanism | Sequence[Mechanism],
        agents: "AgentTable | dict[str, Agent] | Agent",
        profiles: "ProfileSource | Sequence[Profile] | Profile | dict[str, Any] | None" = None,
        *,
        gt: Sequence[GTScorer] | None = None,
        clearances: dict[str, Sequence[str]] | None = None,
        name: str | None = None,
        out: str | Path | None = None,
        concurrency: int = 8,
        repeats: int = 1,
        tasks: Sequence[Task] | None = None,
        seed: int = 0,
        progress: bool = True,
    ):
        self.domain = domain
        self.mechanisms = [mechanisms] if isinstance(mechanisms, Mechanism) else list(mechanisms)
        self.agents = agents if isinstance(agents, AgentTable) else AgentTable(agents)
        if isinstance(profiles, dict):
            self.profile_map = {k: as_profile_source(v) for k, v in profiles.items()}
            self.profiles = None
        else:
            self.profile_map = {}
            self.profiles = as_profile_source(profiles)
        self.gt = list(gt) if gt is not None else None
        self.clearances = clearances
        self.name = name or f"exp-{stable_hash([m.config_hash() for m in self.mechanisms], domain.name, length=8)}"
        self.out = Path(out) if out else None
        self.concurrency = concurrency
        self.repeats = repeats
        self._tasks = list(tasks) if tasks is not None else None
        self.seed = seed
        self.progress = progress

    def tasks(self) -> list[Task]:
        return self._tasks if self._tasks is not None else self.domain.tasks()

    def profiles_for(self, mech: Mechanism, task: Task) -> list[Profile]:
        src = self.profile_map.get(mech.display_name) or self.profile_map.get(mech.name) or self.profiles
        if src is None:
            src = as_profile_source(None)
        return src.profiles(task, mech.roles())

    def plan(self) -> list[tuple[Mechanism, Task, Profile, int]]:
        jobs = []
        for mech in self.mechanisms:
            for task in self.tasks():
                for prof in self.profiles_for(mech, task):
                    for rep in range(self.repeats):
                        jobs.append((mech, task, prof, self.seed + rep))
        return jobs

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.describe(),
            "mechanisms": [m.describe() for m in self.mechanisms],
            "agents": self.agents.describe(),
            "clearances": self.clearances,
            "repeats": self.repeats,
            "seed": self.seed,
        }

    async def arun(self) -> Results:
        jobs = self.plan()
        done: dict[str, EpisodeRecord] = {}
        path = None
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
            path = self.out / "episodes.jsonl"
            for row in read_jsonl(path):
                rec = EpisodeRecord.model_validate(row)
                if rec.error is None:
                    done[rec.key] = rec
            (self.out / "experiment.json").write_text(json.dumps(self.describe(), indent=2, default=str))
        sem = asyncio.Semaphore(self.concurrency)
        results: list[EpisodeRecord | None] = [None] * len(jobs)
        n_done = 0
        lock = asyncio.Lock()

        async def one(i: int, mech: Mechanism, task: Task, prof: Profile, seed: int) -> None:
            nonlocal n_done
            async with sem:
                try:
                    agent_map = {s.name: self.agents.resolve(s, prof.get(s.name).agent) for s in mech.roles()}
                    key = episode_key(mech, task, prof, agent_map, seed, self.domain, self.clearances)
                except Exception:
                    key = None
                if key is not None and key in done:
                    rec = done[key]
                    missing = stale_scorers(rec, list(self.gt if self.gt is not None else self.domain.gt_scorers()))
                    if missing and rec.error is None:  # scorers added, changed or failed since the episode ran
                        before = (dict(rec.gt), rec.gt_status)
                        await compute_gt(task, rec, missing)
                        if path is not None and (dict(rec.gt), rec.gt_status) != before:
                            async with lock:
                                write_jsonl(path, [rec], append=True)  # later lines supersede earlier ones
                    results[i] = rec
                else:
                    rec = await run_episode(
                        mech, task, prof, self.agents, self.domain, gt=self.gt,
                        clearances=self.clearances, seed=seed, experiment=self.name,
                    )
                    results[i] = rec
                    if path is not None:
                        async with lock:
                            write_jsonl(path, [rec], append=True)
                n_done += 1
                if tty and (n_done % max(1, len(jobs) // 20) == 0 or n_done == len(jobs)):
                    print(f"\r[{self.name}] {n_done}/{len(jobs)} episodes", end="", file=sys.stderr, flush=True)

        tty = self.progress and sys.stderr.isatty()
        await asyncio.gather(*[one(i, *job) for i, job in enumerate(jobs)])
        if tty:
            print(file=sys.stderr)
        elif self.progress:
            print(f"[{self.name}] {len(jobs)} episodes", file=sys.stderr)
        recs = [r for r in results if r is not None]
        res = Results(recs, tasks={t.id: t for t in self.tasks()})
        # batch reward rules (e.g. multi-task peer prediction) need all episodes
        for mech in self.mechanisms:
            if type(mech.reward).batch:
                res = res.rescore(mech.reward, mechanism_hash=mech.config_hash())
        return res

    def run(self) -> Results:
        from ..core.util import run_sync

        return run_sync(self.arun())
