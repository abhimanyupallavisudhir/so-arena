"""Optimization pressure from sampled pools: best-of-N and KL-regularized (tilted) policies.

Given a pool of K sampled candidate actions with payoffs $w_1..w_K$ for the mover, a *selection
policy* maps the payoffs to a distribution over candidates:

* :class:`BestOfN` - the distribution of the argmax of n draws (ties split uniformly). ``mode="unbiased"``
  draws without replacement (the U-statistic estimator of BoN from a larger pool, needs n <= K);
  ``mode="plugin"`` draws with replacement from the empirical pool.
* :class:`Tilted` - $\\pi_\\beta(i) \\propto e^{\\beta w_i}$, the optimum of
  KL-regularized RL ($\\max_\\pi E_\\pi[w] - \\frac1\\beta KL(\\pi\\|\\pi_0)$) restricted to the pool.
* :class:`Uniform` - the base policy (Bo1).

For multi-agent mechanisms, :func:`evaluate_tree` performs backward induction on a *sampled game
tree* (see :mod:`so_arena.samplers.pools`): at each information set the mover's selection policy is
applied to its expected payoff under everyone else's policies. This generalizes the nested
Bo(n)-proposer / Bo(m)-critic procedure of "debate with self-play best-of-N optimization"
(Arcadia, 2026) to any protocol, any number of players and general-sum rewards. Moves a role cannot
see - simultaneous ones, private turns, hidden draws - put several nodes into one information set, which
gets one selection (refined by fictitious play), so no role selects on information it did not have.

Missing rewards (errored or pending leaves) are never counted as payoffs: selection treats such a
candidate as the worst in its pool, and expectations average over the finite rewards only. Missing
ground-truth labels are left out of expected values the same way, and every summary reports the
selection-weighted share of plays that have one (``coverage``): a value averaged over a sliver of what
selection favours is not a measurement of it.

The key identity connecting this to ASD is in :func:`first_order_gain`: the derivative at
$\\beta=0$ of the tilted policy's expected ground-truth value equals $\\mathrm{Cov}(u,v)$.
"""

from __future__ import annotations

import abc
import itertools
import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import comb

from pydantic import BaseModel, Field

log = logging.getLogger("so_arena")

# ------------------------------------------------------------------------------ selection policies

class Selection(abc.ABC):
    @abc.abstractmethod
    def probs(self, payoffs: np.ndarray) -> np.ndarray: ...

    def kl(self, payoffs: np.ndarray) -> float:
        """KL(selection || uniform) over the pool, in nats."""
        p = self.probs(payoffs)
        k = len(p)
        nz = p > 0
        return float((p[nz] * np.log(k * p[nz])).sum())


class Uniform(Selection):
    def probs(self, payoffs):
        return np.full(len(payoffs), 1.0 / len(payoffs))

    def __repr__(self):
        return "Uniform()"


def _clean(payoffs: np.ndarray) -> np.ndarray:
    w = np.asarray(payoffs, dtype=float).copy()
    bad = ~np.isfinite(w)
    if bad.all():
        return np.zeros_like(w)
    w[bad] = np.nanmin(w[~bad]) - 1e9
    return w


_WARNED_PLUGIN: set[tuple[int, int]] = set()


class BestOfN(Selection):
    def __init__(self, n: int, mode: str = "unbiased"):
        assert n >= 1 and mode in ("unbiased", "plugin")
        self.n, self.mode = n, mode

    def probs(self, payoffs):
        w = _clean(payoffs)
        k = len(w)
        n = self.n
        if n == 1:
            return np.full(k, 1.0 / k)
        mode = self.mode
        if mode == "unbiased" and n > k:
            # without replacement n <= K draws are possible: say so, since the plug-in (with replacement)
            # estimate is biased and a grid would otherwise mix estimators across pool sizes silently
            if (n, k) not in _WARNED_PLUGIN:
                _WARNED_PLUGIN.add((n, k))
                log.warning("BestOfN(%d, 'unbiased') on a pool of %d: more draws than candidates, so the plug-in "
                            "(with replacement) estimate is used; sample larger pools for an unbiased Bo(%d)", n, k, n)
            mode = "plugin"
        order = np.argsort(w, kind="stable")
        ws = w[order]
        probs_sorted = np.zeros(k)
        # group ties
        i = 0
        below = 0
        while i < k:
            j = i
            while j + 1 < k and abs(ws[j + 1] - ws[i]) < 1e-12:
                j += 1
            size = j - i + 1
            upto = below + size
            if mode == "unbiased":
                p_group = (comb(upto, n, exact=True) - comb(below, n, exact=True)) / comb(k, n, exact=True)
            else:
                p_group = (upto / k) ** n - (below / k) ** n
            probs_sorted[i : j + 1] = p_group / size
            below = upto
            i = j + 1
        out = np.zeros(k)
        out[order] = probs_sorted
        return out / out.sum()

    def __repr__(self):
        return f"BestOfN({self.n}, {self.mode!r})"


class Tilted(Selection):
    def __init__(self, beta: float):
        self.beta = beta

    def probs(self, payoffs):
        w = _clean(payoffs)
        if math.isinf(self.beta):
            return BestOfN(10**6, "plugin").probs(w)
        z = self.beta * (w - w.max())
        e = np.exp(z)
        return e / e.sum()

    def __repr__(self):
        return f"Tilted({self.beta:g})"


def bon_kl_continuous(n: int) -> float:
    """KL of best-of-n from the base policy for continuous rewards: log n - (n-1)/n."""
    return math.log(n) - (n - 1) / n


# ------------------------------------------------------------------------------ flat pools

def _weighted_mean(p: np.ndarray, x: np.ndarray) -> float:
    """Mean of the finite entries of ``x`` under weights ``p``, renormalized (NaN if none has weight)."""
    m = np.isfinite(x)
    den = float(p[m].sum())
    return float((p[m] * x[m]).sum() / den) if den > 0 else math.nan


def _nanmean(xs: Sequence[float]) -> float:
    arr = np.asarray(xs, dtype=float)
    return float(arr[np.isfinite(arr)].mean()) if np.isfinite(arr).any() else math.nan


def pool_curve(pool: pd.DataFrame, *, selections: Sequence[Selection], reward_col: str = "reward",
               value_cols: Sequence[str] = ("value",), group_col: str = "item_id",
               min_coverage: float = 0.5) -> pd.DataFrame:
    """Expected reward and values under each selection policy, averaged over groups (items).

    ``pool`` has one row per sampled candidate; each group is a pool for one decision. A candidate
    whose reward is missing is the worst for selection, and - like a missing value - is left out of
    the expectations (renormalized over the rest) rather than counted as a reward of 0.
    ``<value>_coverage`` is the selection-weighted share of candidates that have a value (averaged over
    groups); a group whose coverage is below ``min_coverage`` contributes no value (see
    :func:`optimization_grid`).
    """
    rows = []
    low = 0
    for sel in selections:
        er, ev, cv, kls = [], {c: [] for c in value_cols}, {c: [] for c in value_cols}, []
        for _, g in pool.groupby(group_col):
            w = g[reward_col].to_numpy(dtype=float)
            p = sel.probs(w)
            er.append(_weighted_mean(p, w))
            kls.append(sel.kl(w))
            for c in value_cols:
                x = g[c].to_numpy(dtype=float)
                cov = float(p[np.isfinite(x)].sum())
                cv[c].append(cov)
                ev[c].append(_weighted_mean(p, x) if cov >= min_coverage else math.nan)
                low += cov < min_coverage and np.isfinite(x).any()
        row = {"selection": repr(sel), "reward": _nanmean(er), "kl": float(np.mean(kls)), "n_groups": len(er)}
        if isinstance(sel, BestOfN):
            row["n"] = sel.n
        if isinstance(sel, Tilted):
            row["beta"] = sel.beta
        for c in value_cols:
            row[c] = _nanmean(ev[c])
            row[f"{c}_coverage"] = float(np.mean(cv[c])) if cv[c] else math.nan
        rows.append(row)
    if low:
        log.warning("pool_curve: %d group value(s) dropped: below %.0f%% of the selected mass has a label",
                    low, 100 * min_coverage)
    return pd.DataFrame(rows)


def first_order_gain(pool: pd.DataFrame, reward_col: str = "reward", value_col: str = "value",
                     group_col: str = "item_id") -> float:
    """$\\frac{d}{d\\beta}E_{\\pi_\\beta}[v]|_{\\beta=0} = \\mathrm{Cov}_{\\pi_0}(u, v)$, averaged over groups."""
    covs = []
    for _, g in pool.groupby(group_col):
        u = g[reward_col].to_numpy(dtype=float)
        v = g[value_col].to_numpy(dtype=float)
        m = np.isfinite(u) & np.isfinite(v)
        if m.sum() >= 2:
            covs.append(float(np.mean((u[m] - u[m].mean()) * (v[m] - v[m].mean()))))
    return float(np.mean(covs)) if covs else math.nan


# ------------------------------------------------------------------------------ game trees

class TreeNode(BaseModel):
    id: str
    key: str  # the role's information set; nodes with equal keys share one pool and one selection
    role: str
    phase: str = ""
    group: str | None = None
    children: list[str] = Field(default_factory=list)
    candidates: list[str] = Field(default_factory=list)  # short text of each candidate


class TreeLeaf(BaseModel):
    id: str
    rewards: dict[str, float | None] = Field(default_factory=dict)
    values: dict[str, float] = Field(default_factory=dict)
    episode_id: str | None = None


class GameTree(BaseModel):
    item_id: str
    mechanism: str
    config: str = ""  # the mechanism's config hash: trees of different configurations are kept apart
    root: str
    nodes: dict[str, TreeNode] = Field(default_factory=dict)
    leaves: dict[str, TreeLeaf] = Field(default_factory=dict)
    roles: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    def value_keys(self) -> list[str]:
        keys: set[str] = set()
        for leaf in self.leaves.values():
            keys |= set(leaf.values)
        return sorted(keys)


class TreeValue(BaseModel):
    rewards: dict[str, float]
    values: dict[str, float]
    # The share of plays - weighted by how often the selection policies reach them - that have a finite
    # reward / ground-truth label. ``rewards`` and ``values`` average over those plays only, so they say
    # nothing about the rest: best-of-N may put most of its mass on candidates nobody could label.
    reward_coverage: dict[str, float] = Field(default_factory=dict)
    coverage: dict[str, float] = Field(default_factory=dict)
    # In reward units: the largest change, over information sets, in the mover's expected reward there if it
    # re-applied its selection policy to everyone's final behaviour - 0 at a fixed point (always, in trees
    # without hidden moves); a large gap means fictitious play has not settled (e.g. it cycles).
    gap: float = 0.0


class _Tree:
    """A game tree indexed for solving: information sets, depths, and value vectors (rewards, then values)."""

    def __init__(self, tree: GameTree):
        self.tree = tree
        self.rkeys = sorted({r for leaf in tree.leaves.values() for r in leaf.rewards})
        self.vkeys = sorted({k for leaf in tree.leaves.values() for k in leaf.values})
        self.ridx = {r: i for i, r in enumerate(self.rkeys)}
        m = len(self.rkeys) + len(self.vkeys)
        self.x: dict[str, np.ndarray] = {}  # expected components (NaN: no play below has one)
        self.cov: dict[str, np.ndarray] = {}  # probability that the play has each component
        for lid, leaf in tree.leaves.items():
            x = np.full(m, math.nan)
            for r, v in leaf.rewards.items():
                if v is not None and math.isfinite(v):
                    x[self.ridx[r]] = float(v)
            for i, k in enumerate(self.vkeys):
                v = leaf.values.get(k)
                if v is not None and math.isfinite(v):
                    x[len(self.rkeys) + i] = float(v)
            self.x[lid], self.cov[lid] = x, np.isfinite(x).astype(float)
        self.parent: dict[str, str] = {}
        self.depth = {tree.root: 0}
        self.order: list[str] = []  # decision nodes, parents before children
        queue = [tree.root]
        while queue:
            nid = queue.pop()
            if nid not in tree.nodes:
                continue
            self.order.append(nid)
            for c in tree.nodes[nid].children:
                self.parent[c], self.depth[c] = nid, self.depth[nid] + 1
                queue.append(c)
        self.order.sort(key=self.depth.__getitem__)
        self.sets: dict[str, list[str]] = {}
        for nid in self.order:
            self.sets.setdefault(tree.nodes[nid].key, []).append(nid)
        for key, ids in self.sets.items():
            if len({(tree.nodes[n].role, len(tree.nodes[n].children)) for n in ids}) > 1:
                raise ValueError(f"information set {key}: its nodes differ in role or number of candidates")
        self.roles = sorted({tree.nodes[n].role for n in self.order})
        self.sigma = {key: np.full(len(tree.nodes[ids[0]].children), 1.0 / len(tree.nodes[ids[0]].children))
                      for key, ids in self.sets.items()}
        for nid in reversed(self.order):
            self.combine(nid)

    def combine(self, nid: str) -> None:
        """A node's expected components under its information set's selection, each averaged over the
        children that have it (a missing reward or label is left out, not counted as 0)."""
        node = self.tree.nodes[nid]
        p = self.sigma[node.key][:, None]
        kids = np.stack([self.x[c] for c in node.children])
        fin = np.isfinite(kids)
        num, den = (p * np.where(fin, kids, 0.0)).sum(0), (p * fin).sum(0)
        self.x[nid] = np.where(den > 0, num / np.where(den > 0, den, 1.0), math.nan)
        self.cov[nid] = (p * np.stack([self.cov[c] for c in node.children])).sum(0)

    def refresh(self, ids: list[str]) -> None:
        """Recompute the nodes of an information set whose selection changed, and all their ancestors."""
        todo: set[str] = set()
        for nid in ids:
            while nid is not None and nid not in todo:
                todo.add(nid)
                nid = self.parent.get(nid)
        for nid in sorted(todo, key=self.depth.__getitem__, reverse=True):
            self.combine(nid)

    def reach(self) -> dict[str, np.ndarray]:
        """Counterfactual reach of every node for every role: the probability that everyone else's
        selections (and the fixtures' draws) lead there - the weight of a node within its role's
        information set (the role's own moves are the same at every node of the set)."""
        roles = np.array(self.roles)
        out = {self.tree.root: np.ones(len(roles))}
        for nid in self.order:
            node = self.tree.nodes[nid]
            mine = roles == node.role
            for i, c in enumerate(node.children):
                out[c] = out[nid] * np.where(mine, 1.0, self.sigma[node.key][i])
        return out

    def payoffs(self, key: str, reach: dict[str, np.ndarray]) -> np.ndarray:
        """Expected reward of each candidate of an information set: over its nodes, weighted by reach."""
        ids = self.sets[key]
        role = self.tree.nodes[ids[0]].role
        k = len(self.sigma[key])
        j = self.ridx.get(role)
        if j is None:
            return np.full(k, math.nan)
        w = np.array([reach[n][self.roles.index(role)] for n in ids])
        if not w.sum() > 0:  # an information set nothing leads to: weigh its nodes equally
            w = np.ones(len(ids))
        u = np.array([[self.x[c][j] for c in self.tree.nodes[n].children] for n in ids])
        fin = np.isfinite(u)
        num, den = (w[:, None] * np.where(fin, u, 0.0)).sum(0), (w[:, None] * fin).sum(0)
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), math.nan)


def evaluate_tree(tree: GameTree, policies: dict[str, Selection] | None = None, *,
                  fp_iters: int = 200, tol: float = 1e-12) -> TreeValue:
    """Expected rewards (per role) and values when each role uses its selection policy, by backward
    induction over *information sets*.

    A node's key is its role's information set (see :meth:`~so_arena.core.game.Game._infoset_key`): the
    nodes that differ only in moves the role cannot see - a simultaneous partner's move, another role's
    private turn, a dealer's hidden card - share one pool and get one selection, applied to each
    candidate's expected reward over the set's nodes weighted by how likely everyone else's behaviour
    makes each of them (counterfactual reach). A role therefore never selects on information it did not
    have.

    Selections are computed deepest information sets first, each against the current behaviour below it
    (exact backward induction when every information set is a single node), then refined by fictitious
    play: for ``t = 2..fp_iters`` each selection moves to $(1-1/t)\,\sigma + \frac1t\,\mathrm{sel}(u)$
    until nothing changes by more than ``tol``. Simultaneous stages and hidden moves are solved this way;
    the result's ``gap`` says, in reward units, how far from a fixed point it ended (0 without hidden
    moves). Fictitious play need not converge (it cycles in some games): then the gap stays large and the
    values depend on ``fp_iters`` - :func:`optimization_grid` warns.

    Roles without a policy act as their base (uniform over their pool). A missing reward (an errored or
    pending leaf) or ground-truth label is never counted as a payoff or a value: selection treats a
    candidate without one as the worst, expectations average over the plays that have one, and
    ``coverage`` / ``reward_coverage`` report the (selection-weighted) share of plays that do.
    """
    policies = policies or {}

    def sel(role: str, phase: str = "") -> Selection:
        # a "role:phase" policy (e.g. optimize only the rebuttal) overrides the role-level one
        return policies.get(f"{role}:{phase}", policies.get(role, Uniform()))

    t_ = _Tree(tree)
    if tree.root in tree.leaves:
        return _tree_value(t_, tree.root, 0.0)
    # deepest information sets first: backward induction when each set is a single node
    order = sorted(t_.sets, key=lambda k: (-max(t_.depth[n] for n in t_.sets[k]), k))

    def response(key: str, reach: dict[str, np.ndarray], u: np.ndarray | None = None) -> np.ndarray:
        node = tree.nodes[t_.sets[key][0]]
        return sel(node.role, node.phase).probs(t_.payoffs(key, reach) if u is None else u)

    for it in range(1, max(1, fp_iters) + 1):
        reach = t_.reach()
        change = 0.0
        for key in order:
            br = response(key, reach)
            new = br if it == 1 else (1 - 1.0 / it) * t_.sigma[key] + br / it
            d = float(np.abs(new - t_.sigma[key]).sum())
            if d > 0:
                t_.sigma[key] = new
                t_.refresh(t_.sets[key])
            change = max(change, d)
        if it > 1 and change <= tol:
            break
    reach = t_.reach()
    gap = 0.0
    for k in order:  # what re-applying each selection would change, in reward (not strategy) units: pure
        u = t_.payoffs(k, reach)  # responses to a near-equilibrium mixture move far in strategy space for nothing
        fin = np.isfinite(u)
        if fin.any():
            gap = max(gap, abs(float(((response(k, reach, u) - t_.sigma[k])[fin] * u[fin]).sum())))
    return _tree_value(t_, tree.root, gap)


def _tree_value(t_: _Tree, nid: str, gap: float) -> TreeValue:
    x, cov, nr = t_.x[nid], t_.cov[nid], len(t_.rkeys)
    return TreeValue(rewards={r: float(x[i]) for i, r in enumerate(t_.rkeys)},
                     values={k: float(x[nr + i]) for i, k in enumerate(t_.vkeys)},
                     reward_coverage={r: float(cov[i]) for i, r in enumerate(t_.rkeys)},
                     coverage={k: float(cov[nr + i]) for i, k in enumerate(t_.vkeys)}, gap=gap)


def optimization_grid(trees: Sequence[GameTree], grid: dict[str, Sequence[int | float]], *,
                      kind: str = "bon", mode: str = "unbiased", min_coverage: float = 0.5) -> pd.DataFrame:
    """Evaluate every combination of per-role optimization levels, averaged over trees (items).

    ``grid`` maps ``"role"`` or ``"role:phase"`` -> list of n (``kind="bon"``) or beta (``kind="tilt"``).
    Returns one row per combination with ``level_<role>``, ``reward_<role>`` and every value key, plus
    bootstrap CIs over items for each value (``<key>_ci_low/high``), the selection-weighted label coverage
    (``<key>_coverage``, the mean over items of the share of selected plays that have a label) and the
    largest fixed-point gap of the trees' solutions (``gap``, see :func:`evaluate_tree`).

    An item whose label coverage for a key is below ``min_coverage`` contributes no value for it (NaN, and
    a warning): its average would describe only a sliver of what selection favours - e.g. best-of-N
    moving onto the candidates nobody could label.
    """
    from so_arena.analysis.metrics import bootstrap_mean_ci

    roles = list(grid)
    rows = []
    low = 0
    for levels in itertools.product(*[grid[r] for r in roles]):
        pols: dict[str, Selection] = {}
        for r, lv in zip(roles, levels):
            pols[r] = BestOfN(int(lv), mode) if kind == "bon" else Tilted(float(lv))
        per_tree = [evaluate_tree(t, pols) for t in trees]
        row: dict[str, Any] = {f"level_{r}": lv for r, lv in zip(roles, levels)}
        all_roles = sorted(set().union(*[set(tv.rewards) for tv in per_tree]))
        for r in all_roles:
            row[f"reward_{r}"] = _nanmean([tv.rewards.get(r, math.nan) for tv in per_tree])
        keys = sorted(set().union(*[set(tv.values) for tv in per_tree]))
        for k in keys:
            cov = np.array([tv.coverage.get(k, 0.0) for tv in per_tree], dtype=float)
            arr = np.array([tv.values.get(k, math.nan) for tv in per_tree], dtype=float)
            dropped = (cov < min_coverage) & np.isfinite(arr)
            low += int(dropped.sum())
            arr = np.where(dropped, math.nan, arr)
            row[k] = _nanmean(arr)
            lo, hi = bootstrap_mean_ci(arr, n_boot=500)
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = lo, hi
            row[f"{k}_coverage"] = float(cov.mean()) if len(cov) else math.nan
        row["gap"] = max((tv.gap for tv in per_tree), default=0.0)
        row["n_items"] = len(per_tree)
        rows.append(row)
    span = max((float(np.ptp(r)) for r in (np.array([v for leaf in t.leaves.values() for v in leaf.rewards.values()
                                                      if v is not None and math.isfinite(v)]) for t in trees) if r.size),
               default=0.0)
    unsettled = [row["gap"] for row in rows if row["gap"] > 1e-2 * max(span, 1e-9)]
    if unsettled:
        log.warning("optimization_grid: fictitious play did not settle on %d row(s) (gap up to %.3g, rewards span %.3g): "
                    "their values depend on fp_iters", len(unsettled), max(unsettled), span)
    if low:
        log.warning("optimization_grid: %d item value(s) dropped: below %.0f%% of the selected plays have a label",
                    low, 100 * min_coverage)
    return pd.DataFrame(rows)
