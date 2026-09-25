"""Statistics helpers: cluster (task-level) bootstrap and summaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of ``stat`` over i.i.d. units (e.g. per-task values)."""
    x = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))], dtype=float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    est = float(stat(x))
    if len(x) == 1:
        return (est, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (est, float(lo), float(hi))


def cluster_bootstrap(
    df: pd.DataFrame,
    cluster: str,
    fn: Callable[[pd.DataFrame], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap resampling whole clusters (tasks) — the right unit for per-task correlation."""
    est = float(fn(df))
    groups = [g for _, g in df.groupby(cluster)]
    if len(groups) < 2:
        return (est, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in pick], ignore_index=True)
        try:
            boots.append(fn(sample))
        except Exception:
            continue
    if not boots:
        return (est, float("nan"), float("nan"))
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (est, float(lo), float(hi))


def fmt_ci(est: float, lo: float, hi: float, digits: int = 3) -> str:
    if np.isnan(lo):
        return f"{est:.{digits}f}"
    return f"{est:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
