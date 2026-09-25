"""Incentive-compatibility metrics: how well do mechanism rewards track ground truth?

All metrics are computed *within task* (the agent's choice is between behaviours on the same
task) and averaged over tasks, with task-level bootstrap CIs.

- :func:`asd` — Agent Score Difference (Pallavi Sudhir et al. 2025), generalised:
  $\\mathrm{ASD} = \\mathbb{E}_t[\\bar r_t(\\text{good}) - \\bar r_t(\\text{bad})]$ for a
  binary GT. With $r=\\log p_{\\text{judge}}$ and assigned correct/incorrect answers this is
  exactly $\\log p_\\top - \\log p_\\bot$.
- :func:`alignment` — rank correlation, pairwise accuracy (AUC) and slope between reward and
  a (possibly continuous) GT across *any* sampled behaviours.
- :func:`frontier` — best reward attainable within each GT class (optimiser's view of IC).
- :func:`eas_ejs` — Expected Agent/Judge Score with softmax propensity.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

from .stats import bootstrap_ci


def _as_df(results) -> pd.DataFrame:
    if isinstance(results, pd.DataFrame):
        return results
    return results.df()


def _select(d: pd.DataFrame, roles: str | Sequence[str] | None, gt: str) -> pd.DataFrame:
    if roles is not None:
        roles = [roles] if isinstance(roles, str) else list(roles)
        d = d[d["role"].isin(roles)]
    col = gt if gt.startswith("gt_") else f"gt_{gt}"
    d = d[~d["error"]] if "error" in d else d
    d = d.dropna(subset=["reward", col])
    return d.assign(_gt=d[col].astype(float))


def asd(
    results,
    roles: str | Sequence[str] | None = None,
    gt: str = "correct",
    by: Sequence[str] = ("mechanism",),
    threshold: float = 0.5,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Agent Score Difference per group (default: per mechanism), pooled over ``roles``.

    ``gt`` values above ``threshold`` count as good behaviour. Returns columns
    ``asd, lo, hi, n_tasks, n_episodes, r_good, r_bad``.
    """
    d = _select(_as_df(results), roles, gt)
    rows = []
    for keys, g in d.groupby(list(by)) if by else [((), d)]:
        per_task, rg, rb = [], [], []
        for _, gt_ in g.groupby("task"):
            good = gt_[gt_["_gt"] > threshold]["reward"]
            bad = gt_[gt_["_gt"] <= threshold]["reward"]
            if len(good) and len(bad):
                per_task.append(good.mean() - bad.mean())
                rg.append(good.mean())
                rb.append(bad.mean())
        est, lo, hi = bootstrap_ci(per_task, n_boot=n_boot)
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            asd=est, lo=lo, hi=hi, n_tasks=len(per_task), n_episodes=len(g),
            r_good=float(np.mean(rg)) if rg else float("nan"), r_bad=float(np.mean(rb)) if rb else float("nan"),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _pairwise(r: np.ndarray, g: np.ndarray) -> tuple[float, int]:
    agree, n = 0.0, 0
    for i, j in combinations(range(len(r)), 2):
        if g[i] == g[j]:
            continue
        n += 1
        dr = r[i] - r[j]
        dg = g[i] - g[j]
        if dr == 0:
            agree += 0.5
        elif (dr > 0) == (dg > 0):
            agree += 1
    return agree, n


def alignment(
    results,
    roles: str | Sequence[str] | None = None,
    gt: str = "correct",
    by: Sequence[str] = ("mechanism",),
    n_boot: int = 2000,
) -> pd.DataFrame:
    """Reward–GT alignment within tasks.

    - ``spearman``: mean over tasks of Spearman $\\rho$(reward, GT) (tasks with variation).
    - ``pairwise_acc``: P(reward ranks the GT-better of two behaviours higher) — equals AUC
      for binary GT; 0.5 = uninformative, <0.5 = perverse incentives.
    - ``slope``: OLS slope of reward on GT with task fixed effects (reward units per GT unit).
    """
    d = _select(_as_df(results), roles, gt)
    rows = []
    for keys, g in d.groupby(list(by)) if by else [((), d)]:
        rhos, pair_task, agree_tot, n_tot = [], [], 0.0, 0
        xs, ys = [], []
        for _, gt_ in g.groupby("task"):
            r = gt_["reward"].to_numpy(float)
            y = gt_["_gt"].to_numpy(float)
            if len(r) < 2 or np.all(y == y[0]):
                continue
            if not np.all(r == r[0]):
                rho = sps.spearmanr(r, y).statistic
                if not np.isnan(rho):
                    rhos.append(rho)
            a, n = _pairwise(r, y)
            if n:
                pair_task.append(a / n)
                agree_tot += a
                n_tot += n
            xs.append(y - y.mean())
            ys.append(r - r.mean())
        slope = float("nan")
        if xs:
            x = np.concatenate(xs)
            yv = np.concatenate(ys)
            if np.dot(x, x) > 0:
                slope = float(np.dot(x, yv) / np.dot(x, x))
        rho, rlo, rhi = bootstrap_ci(rhos, n_boot=n_boot)
        pa, plo, phi = bootstrap_ci(pair_task, n_boot=n_boot)
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            spearman=rho, spearman_lo=rlo, spearman_hi=rhi,
            pairwise_acc=pa, pairwise_lo=plo, pairwise_hi=phi,
            slope=slope, n_tasks=len(pair_task), n_pairs=n_tot,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def frontier(
    results,
    roles: str | Sequence[str] | None = None,
    gt: str = "correct",
    strategy_col: str = "strategy_name",
    by: Sequence[str] = ("mechanism",),
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Best mean reward achievable by GT-good vs GT-bad strategies (strategy = column value).

    A strategy's class is decided by its mean GT across its episodes. The *frontier gap*
    ``best_good - best_bad`` is what a strong optimiser 'sees': if negative, optimisation
    pressure favours bad behaviour even if the average ASD is positive.
    """
    d = _select(_as_df(results), roles, gt)
    rows = []
    for keys, g in d.groupby(list(by)) if by else [((), d)]:
        s = g.groupby(strategy_col).agg(reward=("reward", "mean"), gt=("_gt", "mean"), n=("reward", "size"))
        good = s[s["gt"] > threshold]
        bad = s[s["gt"] <= threshold]
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        bg = good["reward"].max() if len(good) else float("nan")
        bb = bad["reward"].max() if len(bad) else float("nan")
        row.update(
            best_good=bg, best_good_strategy=good["reward"].idxmax() if len(good) else None,
            best_bad=bb, best_bad_strategy=bad["reward"].idxmax() if len(bad) else None,
            frontier_gap=bg - bb, n_good=len(good), n_bad=len(bad),
            argmax_strategy=s["reward"].idxmax() if len(s) else None,
            argmax_gt=float(s.loc[s["reward"].idxmax(), "gt"]) if len(s) else float("nan"),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def eas_ejs(
    results,
    roles: str | Sequence[str] | None = None,
    gt: str = "correct",
    judge_score: str = "gt_judge_p_correct_log",
    beta: float = 1.0,
    by: Sequence[str] = ("mechanism",),
) -> pd.DataFrame:
    """Expected Agent Score and Expected Judge Score (ASD paper §2.1).

    Per task, the agent argues correct with propensity $q_\\top = \\sigma(\\mathrm{ASD}/\\beta)$;
    $\\mathrm{EAS} = q_\\top s_\\top + (1-q_\\top) s_\\bot$ and likewise for the judge's score.
    """
    full = _as_df(results)
    d = _select(full, roles, gt)
    rows = []
    for keys, g in d.groupby(list(by)) if by else [((), d)]:
        eas, ejs = [], []
        for _, gt_ in g.groupby("task"):
            good = gt_[gt_["_gt"] > 0.5]
            bad = gt_[gt_["_gt"] <= 0.5]
            if not len(good) or not len(bad):
                continue
            a = good["reward"].mean() - bad["reward"].mean()
            q = 1 / (1 + math.exp(-a / beta)) if beta > 0 else float(a > 0)
            eas.append(q * good["reward"].mean() + (1 - q) * bad["reward"].mean())
            if judge_score in gt_.columns:
                js_g, js_b = good[judge_score].mean(), bad[judge_score].mean()
                if not (np.isnan(js_g) or np.isnan(js_b)):
                    ejs.append(q * js_g + (1 - q) * js_b)
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        e, elo, ehi = bootstrap_ci(eas)
        j, jlo, jhi = bootstrap_ci(ejs)
        row.update(eas=e, eas_lo=elo, eas_hi=ehi, ejs=j, ejs_lo=jlo, ejs_hi=jhi, n_tasks=len(eas))
        rows.append(row)
    return pd.DataFrame(rows)


def ic_report(results, roles=None, gt: str = "correct", by: Sequence[str] = ("mechanism",)) -> pd.DataFrame:
    """ASD + alignment + frontier in one table."""
    a = asd(results, roles, gt, by)
    b = alignment(results, roles, gt, by)
    f = frontier(results, roles, gt, by=by)
    out = a.merge(b.drop(columns=["n_tasks"]), on=list(by), how="outer")
    return out.merge(f[list(by) + ["frontier_gap", "argmax_gt"]], on=list(by), how="outer")
