"""Running mechanisms over items and profiles.

A :class:`Profile` assigns a policy (and optionally a stance) to every role - a *strategy profile*
in game-theoretic terms. Stances are specified relative to the ground truth (``"true"``,
``"false"``, ``"false:1"``, ``"random"``) and resolved per item by the runner, which alone sees the
uncensored item; the mechanism only ever receives a concrete answer label.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
import threading
import time
from collections.abc import Awaitable, Iterable, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from so_arena.core.game import Player, RunContext
from so_arena.core.ground_truth import GroundTruthScorer, default_scorers, merge_gt
from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Episode, Mechanism, describe_config
from so_arena.core.policy import Policy, stable_hash
from so_arena.core.store import RunStore

log = logging.getLogger("so_arena")
T = TypeVar("T")


class PlayerSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy: Any  # Policy, or a model name (turned into an LLMPolicy)
    stance: str | None = None
    label: str | None = None

    def build_policy(self) -> Policy:
        if isinstance(self.policy, Policy):
            return self.policy
        from so_arena.core.policy import LLMPolicy

        return LLMPolicy(self.policy, label=self.label)


class Profile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    players: dict[str, Any]  # role -> PlayerSpec | Policy | model name
    tags: dict[str, Any] = Field(default_factory=dict)

    def specs(self) -> dict[str, PlayerSpec]:
        out = {}
        for role, p in self.players.items():
            if isinstance(p, PlayerSpec):
                out[role] = p
            elif isinstance(p, dict):
                out[role] = PlayerSpec(**p)
            else:
                out[role] = PlayerSpec(policy=p)
        return out


def resolve_stance(spec: str | None, item: TaskItem, rng: random.Random) -> str | None:
    """Resolve a stance spec against an (uncensored) item.

    ``None``/``"none"`` -> no stance (open protocol: the agent chooses); ``"true"`` -> the correct
    label; ``"false"`` -> a false label (random if several, or ``"false:i"`` for the i-th);
    ``"random"`` -> uniformly random label; anything else is a literal label.
    Raises ``LookupError`` if the spec needs ground truth the item does not have, or names a false
    label it does not have (``"false:i"`` with ``i`` beyond its false labels): the runner then skips
    that profile on that item. Wrapping around instead would argue some false labels more often
    than others.
    """
    if spec is None or spec == "none":
        return None
    labels = item.labels
    if spec == "random":
        return rng.choice(labels) if labels else None
    if spec == "true":
        t = item.true_label
        if t is None:
            raise LookupError(f"item {item.id} has no ground truth for stance 'true'")
        return t
    if spec.startswith("false"):
        fl = item.false_labels
        if item.true_label is None or not fl:
            raise LookupError(f"item {item.id} has no false labels for stance {spec!r}")
        if ":" in spec:
            idx = int(spec.split(":", 1)[1])
            if not 0 <= idx < len(fl):
                raise LookupError(f"item {item.id} has {len(fl)} false labels; no stance {spec!r}")
            return fl[idx]
        return rng.choice(fl)
    if spec.startswith("not:"):
        other = spec.split(":", 1)[1]
        rest = [lab for lab in labels if lab != other]
        return rng.choice(rest) if rest else None
    return spec


def build_players(profile: Profile, item: TaskItem, seed: int = 0) -> dict[str, Player]:
    rng = random.Random(stable_hash(seed, item.id, profile.name))
    players: dict[str, Player] = {}
    specs = profile.specs()
    for role, spec in specs.items():
        stance = spec.stance
        if stance is not None and stance.startswith("opposite:"):
            continue  # resolved below
        players[role] = Player(policy=spec.build_policy(), stance=resolve_stance(stance, item, rng), label=spec.label)
    for role, spec in specs.items():
        if spec.stance is not None and spec.stance.startswith("opposite:"):
            other = spec.stance.split(":", 1)[1]
            other_stance = players[other].stance
            rest = [lab for lab in item.labels if lab != other_stance]
            players[role] = Player(policy=spec.build_policy(), stance=rng.choice(rest) if rest else None, label=spec.label)
    return players


def profile_description(profile: Profile) -> dict[str, Any]:
    """What an episode id covers of a profile: per role, the policy's ``describe()`` (or the model name
    it is built from), the stance spec and the label (plain data only)."""
    out: dict[str, Any] = {}
    for role, spec in profile.specs().items():
        pol = spec.policy.describe() if isinstance(spec.policy, Policy) else spec.policy
        out[role] = [describe_config(pol), spec.stance, spec.label]
    return out


def _episode_id(run_id: str, mechanism: Mechanism, config_hash: str, item: TaskItem, profile: str,
                players: dict[str, Any] | None, repeat: int, seed: int, simulate: bool = False) -> str:
    parts = [run_id, mechanism.name, config_hash, item.id, item.fingerprint(), profile, players, repeat, seed]
    raw = json.dumps(parts + ["simulate"] if simulate else parts, sort_keys=True)
    slug = re.sub(r"[^A-Za-z0-9_.:-]", "_", f"{mechanism.name}:{item.id}:{profile}:{repeat}")
    return f"{slug}#{hashlib.sha256(raw.encode()).hexdigest()[:8]}"


def _simulating() -> bool:
    """Whether provider models are simulated right now (:func:`so_arena.configure`; they read the
    same setting at call time, see :class:`~so_arena.models.base.ProviderModel`)."""
    from so_arena.config import settings

    return bool(settings.simulate)


def episode_id(run_id: str, mechanism: Mechanism, item: TaskItem, profile: str | Profile, repeat: int, *,
               seed: int = 0) -> str:
    """The id under which a run stores (and, resuming, looks up) an episode.

    It covers everything that decides how the episode is played: the mechanism's full configuration
    (:meth:`~so_arena.core.mechanism.Mechanism.config_hash`), the item's content (censored: resolving
    an item keeps its ids; changing it under the same id re-runs it), the profile's players (see
    :func:`profile_description`; a bare profile name identifies it by name only), the repeat and the
    run's seed. Only plain data is hashed, so ids agree across processes. In a dry run
    (``configure(simulate=True)``) ids are different, so a later real run into the same store does
    not resume simulated episodes; real runs' ids do not depend on the setting.
    """
    sim = _simulating()
    if isinstance(profile, Profile):
        return _episode_id(run_id, mechanism, mechanism.config_hash(), item, profile.name,
                           profile_description(profile), repeat, seed, sim)
    return _episode_id(run_id, mechanism, mechanism.config_hash(), item, profile, None, repeat, seed, sim)


async def score_episode(ep: Episode, item: TaskItem, scorers: Sequence[GroundTruthScorer], ctx: RunContext | None = None) -> Episode:
    if item.ground_truth is not None and item.ground_truth.status != "known":
        ep.gt_status = item.ground_truth.status
        return ep
    if not item.has_ground_truth and item.ground_truth is None:
        ep.gt_status = "unknown"
    for s in scorers:  # merged incrementally, so later scorers can read earlier results
        try:
            part = await s.score(ep, item, ctx)
        except Exception as e:  # a failing scorer should not lose the episode
            part = {f"error_{s.name}": repr(e)}
        ep.ground_truth = merge_gt([ep.ground_truth, part])
    if item.has_ground_truth:
        ep.gt_status = "known"  # also when re-scoring an episode that was pending
    elif ep.gt_status == "unscored":
        ep.gt_status = "unknown"
    return ep


class Progress:
    def __init__(self, total: int, every_s: float = 10.0, label: str = "episodes"):
        self.total, self.done, self.errors = total, 0, 0
        self.t0 = self.last = time.time()
        self.every_s, self.label = every_s, label

    def tick(self, error: bool = False) -> None:
        self.done += 1
        self.errors += int(error)
        now = time.time()
        if now - self.last > self.every_s or self.done == self.total:
            self.last = now
            log.info("%s: %d/%d done (%d errors) in %.0fs", self.label, self.done, self.total, self.errors, now - self.t0)


async def run_episodes(
    mechanism: Mechanism,
    items: Sequence[TaskItem],
    profiles: Sequence[Profile],
    *,
    ctx: RunContext | None = None,
    ground_truth: Sequence[GroundTruthScorer] | None = None,
    repeats: int = 1,
    concurrency: int | None = None,
    store: RunStore | None = None,
    resume: bool = True,
    seed: int = 0,
    tags: dict[str, Any] | None = None,
) -> list[Episode]:
    """Run ``mechanism`` on every (item, profile, repeat) and score episodes with ground truth.

    Items whose stances cannot be resolved (e.g. ``"true"`` on an unresolved question) are skipped.
    With a ``store``, episodes are appended as they finish and, if ``resume``, episodes already in the
    store are loaded instead of re-run - those with the same :func:`episode_id`, i.e. the same
    mechanism configuration, item content, players and seed, and both real or both simulated (dry
    runs). Reused episodes whose ground truth was pending are scored again if their item has been
    resolved since.
    """
    from so_arena.config import settings

    ctx = ctx or RunContext()
    scorers = list(ground_truth) if ground_truth is not None else default_scorers()
    sem = asyncio.Semaphore(concurrency or settings.concurrency)
    if store is not None:
        store.save_items(list(items))
    existing: dict[str, Episode] = {}
    if store is not None and resume:
        existing = {e.id: e for e in store.episodes() if e.error is None}

    jobs = []
    for item in items:
        for prof in profiles:
            for rep in range(repeats):
                jobs.append((item, prof, rep))
    progress = Progress(len(jobs), label=mechanism.name)
    # described once, before anything runs: ids must not depend on state the run itself changes
    config_hash = mechanism.config_hash()
    described = {id(p): profile_description(p) for p in profiles}
    simulate = _simulating()  # a dry run's episodes must never be resumed as real ones

    async def one(item: TaskItem, prof: Profile, rep: int) -> Episode | None:
        eid = _episode_id(ctx.run_id, mechanism, config_hash, item, prof.name, described[id(prof)], rep, seed, simulate)
        if eid in existing:
            ep = existing[eid]
            if ep.gt_status == "pending" and item.has_ground_truth:
                # resolved since it was played: keep the play (re-running would answer after the
                # resolution) and score it against the ground truth now available
                ep = await score_episode(ep.model_copy(deep=True), item, scorers, ctx)
                if store is not None:
                    store.append(ep)
            progress.tick()
            return ep
        try:
            players = build_players(prof, item, seed=seed + rep)
        except LookupError as e:
            log.debug("skipping %s/%s: %s", item.id, prof.name, e)
            progress.tick()
            return None
        async with sem:
            ep = await mechanism.run(item, players, ctx, episode_id=eid, profile=prof.name,
                                     tags={**(tags or {}), **prof.tags}, repeat=rep,
                                     seed=stable_hash(seed, rep) % 10**9)
            ep = await score_episode(ep, item, scorers, ctx)
        if store is not None:
            store.append(ep)
        progress.tick(error=ep.error is not None)
        return ep

    results = await asyncio.gather(*[one(*j) for j in jobs])
    eps = [e for e in results if e is not None]
    n_err = sum(e.error is not None for e in eps)
    if n_err:
        log.warning("%d/%d episodes errored; first error:\n%s", n_err, len(eps),
                    next(e.error for e in eps if e.error))
    return eps


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code, also inside an already-running event loop (e.g. Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = asyncio.run(coro)  # type: ignore[arg-type]
        except BaseException as e:  # pragma: no cover
            result["error"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


def iter_items(items: Iterable[TaskItem], limit: int | None) -> list[TaskItem]:
    out = []
    for it in items:
        if limit is not None and len(out) >= limit:
            break
        out.append(it)
    return out
