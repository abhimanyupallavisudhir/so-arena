"""Sampled game trees: natural behaviour pools for best-of-N / KL-regularized optimization analysis.

``expand_tree`` runs a mechanism with a :class:`~so_arena.core.game.BranchController` that samples a
pool of K candidate actions at every decision of the roles in ``pool_sizes`` and replays the protocol
under every combination of choices. The result is a :class:`~so_arena.analysis.optimization.GameTree`
whose leaves carry mechanism rewards and ground-truth values; :func:`evaluate_tree` /
:func:`optimization_grid` then compute what each role's behaviour converges to under Bo(n) or tilted
selection - a cheap proxy for one step of (self-play) RL.

Cost: a tree with pools (K_1, ..., K_d) along each path has prod(K) leaves (one judge call each)
and sum over depths of prod(K up to that depth) agent samples. The BoN self-play study used
(50, 50, 20) per question; start small.

Limitation (by design): Bo(n) only re-weights behaviour the base policy already produces. Behaviour
far from the base policy (e.g. rare deception strategies) needs prompt search or training - see
:mod:`so_arena.samplers.prompt_search`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Any

import pandas as pd

from so_arena.analysis.optimization import GameTree, TreeLeaf, TreeNode, optimization_grid
from so_arena.core.game import BranchController, NodeRecord, Player, RunContext
from so_arena.core.ground_truth import GroundTruthScorer, default_scorers
from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Episode, Mechanism
from so_arena.core.runner import Profile, build_players, score_episode
from so_arena.core.store import RunStore
from so_arena.core.types import Usage

log = logging.getLogger("so_arena")


def leaf_values(ep: Episode) -> dict[str, float]:
    gt = ep.ground_truth or {}
    out: dict[str, float] = {}
    for r, v in (gt.get("role_values") or {}).items():
        if v is not None:
            out[f"value_{r}"] = float(v)
    for k, v in gt.items():
        if isinstance(v, bool):
            out[k] = float(v)
        elif isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _pid(prefix: tuple) -> str:
    return hashlib.sha256(json.dumps(prefix).encode()).hexdigest()[:16]


async def expand_tree(
    mechanism: Mechanism,
    item: TaskItem,
    players: dict[str, Player],
    *,
    pool_sizes: dict[str, int],
    ctx: RunContext | None = None,
    ground_truth: Sequence[GroundTruthScorer] | None = None,
    concurrency: int = 8,
    max_leaves: int = 20000,
    keep_episodes: bool = False,
    seed: int = 0,
    profile: str = "pool",
) -> tuple[GameTree, list[Episode]]:
    ctx = ctx or RunContext()
    scorers = list(ground_truth) if ground_truth is not None else default_scorers()
    root_bc = BranchController(pool_sizes)
    sem = asyncio.Semaphore(concurrency)
    nodes: dict[str, TreeNode] = {}
    leaves: dict[str, TreeLeaf] = {}
    traces: dict[str, list[NodeRecord]] = {}  # leaf -> every decision of its play
    episodes: list[Episode] = []
    truncated = False

    async def run_plan(prefix: tuple):
        bc = root_bc.fork(dict(prefix))
        async with sem:
            ep = await mechanism.run(item, players, ctx, episode_id=f"{item.id}:{mechanism.name}:tree:{_pid(prefix)}",
                                     profile=profile, branch=bc, seed=seed)
            ep = await score_episode(ep, item, scorers, ctx)
        return bc.trace, ep

    async def expand(prefix: tuple) -> str:
        nonlocal truncated
        trace, ep = await run_plan(prefix)
        planned = dict(prefix)
        branching = sorted((n for n in trace if n.k > 1), key=lambda n: n.slot)
        nxt = next((n for n in branching if n.key not in planned), None)
        pid = _pid(prefix)
        if nxt is None or len(leaves) >= max_leaves:
            if nxt is not None:
                truncated = True
            leaves[pid] = TreeLeaf(id=pid, rewards=dict(ep.rewards), values=leaf_values(ep), episode_id=ep.id)
            traces[pid] = list(trace)
            if keep_episodes:
                episodes.append(ep)
            if ep.error:
                log.warning("leaf episode errored: %s", ep.error.splitlines()[-1])
            return pid
        node = TreeNode(id=pid, key=nxt.key, role=nxt.role, phase=nxt.phase, group=nxt.group)
        nodes[pid] = node
        node.candidates = [p.action.summary[:300] for p in root_bc.memo[nxt.key]]
        node.children = list(await asyncio.gather(*[expand(prefix + ((nxt.key, i),)) for i in range(nxt.k)]))
        return pid

    root = await expand(())
    if truncated:
        log.warning("tree for %s truncated at %d leaves", item.id, max_leaves)
    charged = _charge_pools(root_bc, traces)
    leaf_of = {leaf.episode_id: lid for lid, leaf in leaves.items()}
    for ep in episodes:
        ep.usage = charged.get(leaf_of.get(ep.id, ""), {})
    usage = {r: u.model_dump() for r, u in root_bc.usage.items()}
    tree = GameTree(item_id=item.id, mechanism=mechanism.name, root=root, nodes=nodes, leaves=leaves,
                    roles=list(mechanism.roles()), usage=usage)
    return tree, episodes


def _charge_pools(bc: BranchController, traces: dict[str, list[NodeRecord]]) -> dict[str, dict[str, Usage]]:
    """Charge each sampled pool to one canonical leaf, so a tree's leaf episodes sum to its total usage.

    A pool belongs to an information set, which plays that differ in moves its role cannot see all pass
    through; its canonical leaf is the one whose play, among those passing through it, takes the
    lexicographically smallest choices (in the order of the decisions) - deterministic, whichever replay
    sampled the pool first.
    """
    canon: dict[str, tuple[tuple[int, ...], str]] = {}
    for lid, trace in traces.items():
        order = tuple(n.choice for n in sorted(trace, key=lambda n: n.slot))
        for n in trace:
            if n.key not in canon or (order, lid) < canon[n.key]:
                canon[n.key] = (order, lid)
    out: dict[str, dict[str, Usage]] = {}
    for key, pool in bc.memo.items():
        if key not in canon:  # every sampled pool lies on some complete play; keep the total right regardless
            if not traces:
                continue
            log.warning("pool %s is on no leaf's play; charged to the first leaf", key)
            canon[key] = ((), min(traces))
        lid, role = canon[key][1], bc.roles[key]
        for p in pool:
            out.setdefault(lid, {})
            out[lid][role] = out[lid].get(role, Usage()) + p.usage
    return out


class OptimizationExperiment:
    """Best-of-N / tilted optimization pressure on sampled game trees, for any mechanism.

    Example (proposer-critic protocol, as in the BoN self-play study)::

        exp = OptimizationExperiment(ReviewedWork(critique_rounds=1), items, profile,
                                     pool_sizes={"worker": 8, "critic": 4})
        trees = exp.run()
        surface = exp.grid({"worker": [1, 2, 4, 8], "critic": [1, 2, 4]})

    With a ``store``, the trees go to its ``trees.jsonl``, replacing only this mechanism's earlier
    trees, so experiments on several mechanisms can share a store (:func:`load_trees` reads them back).
    """

    def __init__(self, mechanism: Mechanism, items: Sequence[TaskItem], profile: Profile, *,
                 pool_sizes: dict[str, int], ground_truth: Sequence[GroundTruthScorer] | None = None,
                 ctx: RunContext | None = None, store: RunStore | None = None, concurrency: int = 8,
                 max_leaves: int = 20000, seed: int = 0):
        self.mechanism, self.items, self.profile = mechanism, list(items), profile
        self.pool_sizes, self.ground_truth, self.ctx = pool_sizes, ground_truth, ctx
        self.store, self.concurrency, self.max_leaves, self.seed = store, concurrency, max_leaves, seed
        self.trees: list[GameTree] = []

    async def arun(self) -> list[GameTree]:
        async def one(item: TaskItem) -> GameTree | None:
            try:
                players = build_players(self.profile, item, seed=self.seed)
            except LookupError:
                return None
            tree, _ = await expand_tree(self.mechanism, item, players, pool_sizes=self.pool_sizes, ctx=self.ctx,
                                        ground_truth=self.ground_truth, concurrency=self.concurrency,
                                        max_leaves=self.max_leaves, seed=self.seed, profile=self.profile.name)
            return tree

        trees = await asyncio.gather(*[one(it) for it in self.items])
        self.trees = [t for t in trees if t is not None]
        if self.store is not None:
            save_trees(self.store.path, self.trees, mechanism=self.mechanism.name)
        return self.trees

    def run(self) -> list[GameTree]:
        from so_arena.core.runner import run_sync

        return run_sync(self.arun())

    def grid(self, grid: dict[str, Sequence[int | float]], *, kind: str = "bon", mode: str = "unbiased",
             min_coverage: float = 0.5) -> pd.DataFrame:
        return optimization_grid(self.trees, grid, kind=kind, mode=mode, min_coverage=min_coverage)


def save_trees(path: Any, trees: Sequence[GameTree], *, mechanism: str | None = None) -> None:
    """Write trees to ``trees.jsonl`` in a run directory (or to a file path).

    With ``mechanism``, only that mechanism's earlier trees are replaced and other mechanisms' are
    kept, so experiments on several mechanisms can share a store; without it the file is rewritten.
    """
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        p = p / "trees.jsonl"
    kept = [t for t in load_trees(p) if t.mechanism != mechanism] if mechanism is not None and p.exists() else []
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        for t in [*kept, *trees]:
            f.write(t.model_dump_json() + "\n")
    tmp.replace(p)


def load_trees(path: Any, mechanism: str | None = None) -> list[GameTree]:
    """Trees from ``trees.jsonl`` in a run directory (or a file path); ``mechanism`` selects one mechanism's."""
    from pathlib import Path

    p = Path(path)
    if p.is_dir():
        p = p / "trees.jsonl"
    trees = [GameTree.model_validate_json(line) for line in p.read_text().splitlines() if line.strip()]
    return [t for t in trees if mechanism is None or t.mechanism == mechanism]
