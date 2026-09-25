"""Best-of-N analyses: exact order-statistics selection from empirical pools.

"One step of RL": draw ``n`` samples from the base policy and keep the one the mechanism
rewards most. Given a finite pool of $N$ sampled behaviours with rewards $r_i$, the
probability that Bo($n$) (with replacement) selects a behaviour whose reward is the $k$-th
smallest distinct value $v_k$ is $F(v_k)^n - F(v_{k-1})^n$ (split evenly among ties), where
$F$ is the pool's empirical CDF. We compute expectations exactly — no resampling noise.
Minimisation (e.g. an adversarial critic) uses $-r$.

For sequential protocols (proposal → critique → rebuttal) :func:`tree_value` does backward
induction with Bo($k$) selection at each level ("worm plots" and 2-D meshes as in *Debate
with self-play best-of-N optimisation*).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .stats import bootstrap_ci


def bon_weights(scores: Sequence[float] | np.ndarray, n: float, maximize: bool = True) -> np.ndarray:
    """Selection probabilities of each pool element under best-of-``n`` (``n`` may be fractional)."""
    s = np.asarray(scores, dtype=float)
    if not maximize:
        s = -s
    N = len(s)
    if N == 0:
        return s
    if n <= 0:
        return np.full(N, 1.0 / N)
    vals, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    cdf = np.cumsum(counts) / N
    prev = np.concatenate([[0.0], cdf[:-1]])
    p_val = cdf**n - prev**n
    return p_val[inv] / counts[inv]


def bon_kl(n: float) -> float:
    r"""$\log n - (n-1)/n$ nats: KL(best-of-n || base policy) for a *continuous* reward
    distribution (Stiennon et al. 2020), the standard x-axis for optimisation-pressure curves. For a
    finite pool it is only an upper bound (Beirami et al. 2024); :func:`pool_kl` is exact."""
    import math

    return math.log(n) - (n - 1) / n if n >= 1 else 0.0


def pool_kl(scores: Sequence[float] | np.ndarray, n: float, maximize: bool = True) -> float:
    r"""Exact KL(best-of-n || uniform) over a finite pool of samples, $\sum_i w_i \log(N w_i)$.
    At most :func:`bon_kl` ``(n)``, and at most $\log N$ however large $n$ is."""
    w = bon_weights(scores, n, maximize)
    w = w[w > 0]
    return float(np.sum(w * np.log(len(scores) * w))) if len(w) else 0.0


def bon_expectation(scores, values, n: float, maximize: bool = True) -> float:
    w = bon_weights(scores, n, maximize)
    return float(np.dot(w, np.asarray(values, dtype=float)))


def bon_curve(
    results,
    roles: str | Sequence[str] | None = None,
    gt: str = "correct",
    n_values: Sequence[float] = (1, 2, 4, 8, 16, 32, 64),
    by: Sequence[str] = ("mechanism",),
    pool_key: str = "task",
    n_boot: int = 1000,
) -> pd.DataFrame:
    r"""Expected reward and GT under Bo(n) selection by the mechanism's reward, per pool.

    Each task is a pool (the policy's samples for that task). Returns one row per
    (group, n) with task-bootstrap CIs — plot ``reward`` vs ``gt`` parametrically in ``n``
    for the 2-D "optimisation response" plots. ``kl`` is the continuous-case axis
    $\log n - (n-1)/n$ (an upper bound here); ``kl_pool`` is the exact KL, averaged over pools.
    """
    d = results if isinstance(results, pd.DataFrame) else results.df()
    col = gt if gt.startswith("gt_") else f"gt_{gt}"
    if roles is not None:
        roles = [roles] if isinstance(roles, str) else list(roles)
        d = d[d["role"].isin(roles)]
    d = d.dropna(subset=["reward", col])
    rows = []
    for keys, g in d.groupby(list(by)) if by else [((), d)]:
        pools = [(p["reward"].to_numpy(float), p[col].to_numpy(float)) for _, p in g.groupby(pool_key)]
        for n in n_values:
            rs = [bon_expectation(r, r, n) for r, _ in pools]
            gs = [bon_expectation(r, y, n) for r, y in pools]
            r_est, r_lo, r_hi = bootstrap_ci(rs, n_boot=n_boot)
            g_est, g_lo, g_hi = bootstrap_ci(gs, n_boot=n_boot)
            row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
            row.update(n=n, kl=bon_kl(n), kl_pool=float(np.mean([pool_kl(r, n) for r, _ in pools])), reward=r_est,
                       reward_lo=r_lo, reward_hi=r_hi, gt=g_est, gt_lo=g_lo, gt_hi=g_hi, pools=len(pools))
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class Node:
    """A node in a sampled game tree.

    Leaves carry ``payoff`` (to the root player, e.g. judge P(accept)) and optionally ``gt``.
    Internal nodes' children are the sampled continuations of the player to move at that level.
    ``gt`` at an internal node (e.g. proposal correctness) is inherited by its subtree.
    """

    children: list["Node"] = field(default_factory=list)
    payoff: float | None = None
    gt: float | None = None
    label: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def tree_value(node: Node, ks: Sequence[float], maximize: Sequence[bool], depth: int = 0, gt: float | None = None) -> tuple[float, float]:
    """Backward induction with Bo(k) selection at each level.

    ``ks[d]`` / ``maximize[d]``: optimisation pressure and direction of the player choosing
    among the children at depth ``d`` (e.g. proposer: max, critic: min of proposer payoff).
    Returns (expected root-player payoff, expected GT).
    """
    gt = node.gt if node.gt is not None else gt
    if not node.children:
        return float(node.payoff if node.payoff is not None else float("nan")), float(gt if gt is not None else float("nan"))
    vals = [tree_value(c, ks, maximize, depth + 1, gt) for c in node.children]
    pay = np.array([v[0] for v in vals])
    gts = np.array([v[1] for v in vals])
    k = ks[depth] if depth < len(ks) else 1
    mx = maximize[depth] if depth < len(maximize) else True
    w = bon_weights(pay, k, maximize=mx)
    ok = ~np.isnan(gts)  # GT averaged over the children where it is defined
    g = float(np.dot(w[ok], gts[ok]) / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")
    return float(np.dot(w, pay)), g


def tree_mesh(
    roots: Sequence[Node], ks_grid: Sequence[Sequence[float]], maximize: Sequence[bool]
) -> pd.DataFrame:
    """Evaluate :func:`tree_value` over many roots (tasks) for a grid of per-level pressures."""
    rows = []
    for ks in ks_grid:
        vals = [tree_value(r, ks, maximize) for r in roots]
        pay = [v[0] for v in vals]
        gts = [v[1] for v in vals]
        p, plo, phi = bootstrap_ci(pay)
        g, glo, ghi = bootstrap_ci(gts)
        row = {f"k{d}": k for d, k in enumerate(ks)}
        row.update(payoff=p, payoff_lo=plo, payoff_hi=phi, gt=g, gt_lo=glo, gt_hi=ghi, n=len(roots))
        rows.append(row)
    return pd.DataFrame(rows)


def trees_from_results(results, levels: Sequence[str | tuple[str, str]], payoff: str = "accept_prob",
                       gt: tuple[str, str] | None = ("correct", "proposer")) -> list[Node]:
    """Build game trees from episodes whose profiles vary role seeds jointly.

    ``levels`` are the moves in order, as role names or (role, step) pairs — e.g.
    ``[("proposer", "proposal"), ("critic", "critique"), ("proposer", "rebuttal")]`` for episodes
    generated with :class:`~oversight_arena.experiment.profiles.GameTree`. A node at each level
    is identified by that role's (strategy, sample index for the step). Episodes sharing the
    same task and the same prefix are siblings. ``payoff`` is read from ``record.outcome``.
    """
    recs = results.records if hasattr(results, "records") else results
    by_task: dict[str, dict] = {}
    for r in recs:
        if r.error:
            continue
        path = []
        for lv in levels:
            role, step = (lv, None) if isinstance(lv, str) else lv
            b = r.bound.get(role)
            path.append((b.strategy_id, b.seed_for(step)) if b is not None else ("", 0))
        path = tuple(path)
        val = r.outcome.get(payoff)
        g = r.gt.get(gt[0], {}).get(gt[1]) if gt else None
        tree = by_task.setdefault(r.task_id, {})
        node = tree
        for s in path[:-1]:
            node = node.setdefault(s, {})
        node[path[-1]] = (val, g)

    def build(d: dict | tuple) -> Node:
        if isinstance(d, tuple):
            return Node(payoff=d[0], gt=d[1])
        return Node(children=[build(v) for _, v in sorted(d.items())])

    return [build(t) for t in by_task.values()]
