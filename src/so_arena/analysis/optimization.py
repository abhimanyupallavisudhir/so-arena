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
tree* (see :mod:`so_arena.samplers.pools`): at each node the mover's selection policy is applied to
its expected payoff under everyone else's policies downstream. This generalizes the nested
Bo(n)-proposer / Bo(m)-critic procedure of "debate with self-play best-of-N optimization"
(Arcadia, 2026) to any protocol, any number of players and general-sum rewards. Simultaneous moves
(nodes in the same information set) are solved as stage games by damped fixed-point iteration.

The key identity connecting this to ASD is in :func:`first_order_gain`: the derivative at
$\\beta=0$ of the tilted policy's expected ground-truth value equals $\\mathrm{Cov}(u,v)$.
"""

from __future__ import annotations

import abc
import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import comb

from pydantic import BaseModel, Field


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
            mode = "plugin"  # cannot draw more than K without replacement
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

def pool_curve(pool: pd.DataFrame, *, selections: Sequence[Selection], reward_col: str = "reward",
               value_cols: Sequence[str] = ("value",), group_col: str = "item_id") -> pd.DataFrame:
    """Expected reward and values under each selection policy, averaged over groups (items).

    ``pool`` has one row per sampled candidate; each group is a pool for one decision.
    """
    rows = []
    for sel in selections:
        er, ev, kls = [], {c: [] for c in value_cols}, []
        for _, g in pool.groupby(group_col):
            w = g[reward_col].to_numpy(dtype=float)
            p = sel.probs(w)
            er.append(float(np.nansum(p * np.nan_to_num(w))))
            kls.append(sel.kl(w))
            for c in value_cols:
                vals = g[c].to_numpy(dtype=float)
                m = np.isfinite(vals)
                ev[c].append(float((p[m] * vals[m]).sum() / p[m].sum()) if m.any() else math.nan)
        row = {"selection": repr(sel), "reward": float(np.mean(er)), "kl": float(np.mean(kls)), "n_groups": len(er)}
        if isinstance(sel, BestOfN):
            row["n"] = sel.n
        if isinstance(sel, Tilted):
            row["beta"] = sel.beta
        for c in value_cols:
            row[c] = float(np.nanmean(ev[c]))
        rows.append(row)
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
    key: str  # decision key; nodes with equal keys share one pool (an information set)
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


def _combine(parts: list[TreeValue], weights: np.ndarray) -> TreeValue:
    rewards: dict[str, float] = {}
    values: dict[str, float] = {}
    for dict_name, target in (("rewards", rewards), ("values", values)):
        keys = set().union(*[getattr(p, dict_name).keys() for p in parts]) if parts else set()
        for k in keys:
            num, den = 0.0, 0.0
            for p, w in zip(parts, weights):
                x = getattr(p, dict_name).get(k)
                if x is not None and np.isfinite(x) and w > 0:
                    num += w * x
                    den += w
            target[k] = num / den if den > 0 else math.nan
    return TreeValue(rewards=rewards, values=values)


def evaluate_tree(tree: GameTree, policies: dict[str, Selection] | None = None, *,
                  fp_iters: int = 200) -> TreeValue:
    """Backward induction: expected rewards (per role) and values when each role uses its selection policy.

    Roles without a policy act as their base (uniform over their pool).
    """
    policies = policies or {}

    def sel(role: str) -> Selection:
        return policies.get(role, Uniform())

    def ev(nid: str) -> TreeValue:
        if nid in tree.leaves:
            leaf = tree.leaves[nid]
            return TreeValue(rewards={k: (math.nan if v is None else float(v)) for k, v in leaf.rewards.items()},
                             values=dict(leaf.values))
        node = tree.nodes[nid]
        # simultaneous group: this node's children are nodes of the same group with a shared key
        if node.group is not None:
            members = _group_chain(tree, node)
            if len(members) > 1:
                return _solve_stage(tree, members, sel, ev, fp_iters)
        kids = [ev(c) for c in node.children]
        w = np.array([k.rewards.get(node.role, math.nan) for k in kids], dtype=float)
        return _combine(kids, sel(node.role).probs(w))

    return ev(tree.root)


def _group_chain(tree: GameTree, node: TreeNode) -> list[TreeNode]:
    """Roles moving simultaneously with ``node``: successive levels in the same group."""
    chain = [node]
    cur = node
    while cur.children:
        nxt = tree.nodes.get(cur.children[0])
        if nxt is None or nxt.group != node.group or nxt.role in [c.role for c in chain]:
            break
        chain.append(nxt)
        cur = nxt
    return chain


def _marginal(T: np.ndarray, pis: list[np.ndarray], a: int) -> np.ndarray:
    """Expected payoff of each of player ``a``'s candidates, contracting all other axes with ``pis``."""
    out = T
    for b in reversed(range(len(pis))):  # descending order keeps lower axis indices valid
        if b != a:
            out = np.tensordot(out, pis[b], axes=([b], [0]))
    return out


def _solve_stage(tree: GameTree, members: list[TreeNode], sel, ev, fp_iters: int) -> TreeValue:
    """Solve a simultaneous-move stage by damped fixed-point iteration of the selection policies.

    Each member selects from its (shared) pool using its expected payoff against the others' current
    selection distributions; iterates are averaged (fictitious-play style) to damp cycling.
    """
    sizes = [len(m.children) for m in members]
    outcomes: dict[tuple[int, ...], TreeValue] = {}

    def walk(nid: str, depth: int, idx: tuple[int, ...]) -> None:
        if depth == len(members):
            outcomes[idx] = ev(nid)
            return
        for i, c in enumerate(tree.nodes[nid].children):
            walk(c, depth + 1, idx + (i,))

    walk(members[0].id, 0, ())
    roles = [m.role for m in members]
    R = {r: np.full(tuple(sizes), -1e9) for r in roles}
    for idx, tv in outcomes.items():
        for r in roles:
            x = tv.rewards.get(r, math.nan)
            if x is not None and np.isfinite(x):
                R[r][idx] = x
    pis = [np.full(s, 1.0 / s) for s in sizes]
    for t in range(1, fp_iters + 1):
        new = [sel(r).probs(_marginal(R[r], pis, a)) for a, r in enumerate(roles)]
        step = 1.0 / (t + 1)
        pis = [(1 - step) * p + step * q for p, q in zip(pis, new)]
    weights, parts = [], []
    for idx in itertools.product(*[range(s) for s in sizes]):
        weights.append(float(np.prod([pis[a][i] for a, i in enumerate(idx)])))
        parts.append(outcomes[idx])
    return _combine(parts, np.array(weights))


def optimization_grid(trees: Sequence[GameTree], grid: dict[str, Sequence[int | float]], *,
                      kind: str = "bon", mode: str = "unbiased") -> pd.DataFrame:
    """Evaluate every combination of per-role optimization levels, averaged over trees (items).

    ``grid`` maps role -> list of n (``kind="bon"``) or beta (``kind="tilt"``). Returns one row per
    combination with ``level_<role>``, ``reward_<role>`` and every value key, plus bootstrap CIs over
    items for each value (``<key>_ci_low/high``).
    """
    roles = list(grid)
    rows = []
    for levels in itertools.product(*[grid[r] for r in roles]):
        pols: dict[str, Selection] = {}
        for r, lv in zip(roles, levels):
            pols[r] = BestOfN(int(lv), mode) if kind == "bon" else Tilted(float(lv))
        per_tree = [evaluate_tree(t, pols) for t in trees]
        row: dict[str, Any] = {f"level_{r}": lv for r, lv in zip(roles, levels)}
        all_roles = sorted(set().union(*[set(tv.rewards) for tv in per_tree]))
        for r in all_roles:
            row[f"reward_{r}"] = float(np.nanmean([tv.rewards.get(r, math.nan) for tv in per_tree]))
        keys = sorted(set().union(*[set(tv.values) for tv in per_tree]))
        from so_arena.analysis.metrics import bootstrap_mean_ci

        for k in keys:
            arr = np.array([tv.values.get(k, math.nan) for tv in per_tree], dtype=float)
            row[k] = float(np.nanmean(arr)) if np.isfinite(arr).any() else math.nan
            lo, hi = bootstrap_mean_ci(arr, n_boot=500)
            row[f"{k}_ci_low"], row[f"{k}_ci_high"] = lo, hi
        row["n_items"] = len(per_tree)
        rows.append(row)
    return pd.DataFrame(rows)
