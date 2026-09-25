"""Plotting (matplotlib). Consistent, quiet style: fixed categorical order, thin marks,
hairline grids, one y-axis, legends for ≥2 series plus sparse direct labels.

Main figures:
- :func:`optimization_frontier` — the 2-D "reward vs ground truth under optimisation pressure"
  plot (best-of-N or prompt-optimisation trajectories), one line per mechanism/steering.
- :func:`asd_bars` — ASD (or any metric) with CIs per mechanism.
- :func:`payoff_heatmap` — empirical-game payoffs / GT.
- :func:`replicator_field` — 2×2 learning dynamics with basins.
- :func:`regime_map` — e.g. the swarm whistleblowing phase diagram.
- :func:`learning_curves` — strategy probabilities during (strategy-level) RL.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e0"
NEUTRAL = "#f0efec"
DIVERGE = ("#2a78d6", "#e34948")


def set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9.5, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 2.0, "lines.markersize": 6,
        "font.size": 9.5, "axes.prop_cycle": matplotlib.cycler(color=SERIES), "text.color": INK,
    })


set_style()


def color_map(keys: Sequence[str]) -> dict[str, str]:
    """Stable entity → colour assignment (fixed order; never recycled past 8)."""
    keys = list(dict.fromkeys(keys))
    if len(keys) > len(SERIES):
        raise ValueError("more than 8 series: facet or fold into 'Other'")
    return {k: SERIES[i] for i, k in enumerate(keys)}


def save(fig: Any, path: str | Path, svg: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if svg:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    return path


def optimization_frontier(
    df: pd.DataFrame, x: str = "reward", y: str = "gt", group: str = "mechanism", pressure: str = "n",
    title: str = "Optimisation pressure: mechanism reward vs ground truth", xlabel: str = "expected mechanism reward",
    ylabel: str = "expected ground truth", label_points: Sequence[Any] | None = None, ax: Any = None,
    ci: bool = True,
) -> tuple[Any, Any]:
    """Parametric curves (x(reward), y(GT)) traced as optimisation pressure grows.

    Healthy mechanisms move up-and-right (more reward ⇒ better behaviour); Goodharting
    mechanisms bend down (reward keeps rising while ground truth falls)."""
    fig, ax = (plt.subplots(figsize=(6.2, 4.2)) if ax is None else (ax.figure, ax))
    groups = list(dict.fromkeys(df[group])) if group in df else [None]
    cmap = color_map([str(g) for g in groups])
    for g in groups:
        d = df if g is None else df[df[group] == g]
        d = d.sort_values(pressure)
        c = cmap[str(g)]
        ax.plot(d[x], d[y], color=c, lw=2, solid_capstyle="round", label=str(g), zorder=2)
        ax.scatter(d[x], d[y], s=26, color=c, edgecolor=SURFACE, linewidth=1.6, zorder=3)
        if ci and f"{y}_lo" in d and d[f"{y}_lo"].notna().any():
            ax.fill_between(d[x], d[f"{y}_lo"], d[f"{y}_hi"], color=c, alpha=0.10, lw=0, zorder=1)
        pts = d if label_points is None else d[d[pressure].isin(label_points)]
        for _, r in pts.iloc[[0, -1]].iterrows() if label_points is None else pts.iterrows():
            ax.annotate(f"{pressure}={r[pressure]:g}", (r[x], r[y]), xytext=(4, 4), textcoords="offset points",
                        fontsize=7.5, color=INK2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if len(groups) > 1:
        ax.legend(loc="best")
    return fig, ax


def asd_bars(df: pd.DataFrame, metric: str = "asd", label: str = "mechanism", title: str = "Agent Score Difference",
             xlabel: str = "ASD (reward for truth − reward for falsehood)") -> tuple[Any, Any]:
    d = df.sort_values(metric)
    fig, ax = plt.subplots(figsize=(5.6, 0.34 * len(d) + 1.0))
    y = np.arange(len(d))
    vals = d[metric].to_numpy(float)
    ax.barh(y, vals, height=0.5, color=SERIES[0], zorder=2)
    ends = vals.copy()
    if "lo" in d and d["lo"].notna().any():
        lo = d["lo"].to_numpy(float)
        hi = d["hi"].to_numpy(float)
        err = np.vstack([vals - lo, hi - vals])
        ax.errorbar(vals, y, xerr=np.nan_to_num(err), fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5, zorder=3)
        ends = np.where(vals >= 0, np.nan_to_num(np.maximum(hi, vals), nan=0) , np.nan_to_num(np.minimum(lo, vals), nan=0))
    for yi, v, e in zip(y, vals, ends):
        ax.annotate(f"{v:.2f}", (e, yi), xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                    fontsize=7.5, color=INK2, ha="left" if v >= 0 else "right", va="center")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, d[label].astype(str))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    return fig, ax


def payoff_heatmap(game: Any, role: str | None = None, gt_key: str | None = None, title: str | None = None,
                   annotate: bool = True) -> tuple[Any, Any]:
    """Heatmap of a 2-role empirical game's payoff (for ``role``) or GT tensor (``gt_key``)."""
    from matplotlib.colors import LinearSegmentedColormap

    r0, r1 = game.roles[:2]
    M = game.gt[gt_key] if gt_key else game.payoffs[role or r0]
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * M.shape[1], 0.8 + 0.6 * M.shape[0]))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(M.shape[1]), game.strategies[r1], rotation=30, ha="right")
    ax.set_yticks(range(M.shape[0]), game.strategies[r0])
    ax.set_xlabel(r1)
    ax.set_ylabel(r0)
    ax.grid(False)
    if annotate:
        lo, hi = np.nanmin(M), np.nanmax(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if np.isnan(v):
                    continue
                dark = (v - lo) / (hi - lo + 1e-12) > 0.55
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if dark else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title or (f"GT: {gt_key}" if gt_key else f"payoff to {role or r0}"), loc="left")
    return fig, ax


def replicator_field(game: Any, grid: int = 15, iters: int = 300, lr: float = 0.5, gt_key: str | None = None,
                     labels: tuple[str, str] | None = None, title: str = "Learning dynamics (replicator)") -> tuple[Any, Any]:
    """For two roles with ≥2 strategies each: vector field over (P[role0 plays s0], P[role1 plays s0])
    and final ground truth reached from each start (sequential shading)."""
    from matplotlib.colors import LinearSegmentedColormap

    r0, r1 = game.roles[:2]
    ps = np.linspace(0.02, 0.98, grid)
    U = np.zeros((grid, grid))
    V = np.zeros((grid, grid))
    F = np.full((grid, grid), np.nan)
    for i, q in enumerate(ps):
        for j, p in enumerate(ps):
            x0 = {r0: _mix(p, len(game.strategies[r0])), r1: _mix(q, len(game.strategies[r1]))}
            step = game.replicator(x0, iters=1, lr=lr)[-1]
            U[i, j] = step[r0][0] - p
            V[i, j] = step[r1][0] - q
            if gt_key:
                fin = game.replicator(x0, iters=iters, lr=lr)[-1]
                F[i, j] = game.expected_gt(fin).get(gt_key, np.nan)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    if gt_key:
        cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
        im = ax.imshow(F, origin="lower", extent=(0, 1, 0, 1), cmap=cmap, aspect="auto", alpha=0.9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"final {gt_key}")
    ax.quiver(ps, ps, U, V, color=INK2, angles="xy", width=0.003, headwidth=4)
    s0 = game.strategies[r0][0]
    s1 = game.strategies[r1][0]
    ax.set_xlabel(labels[0] if labels else f"P({r0} plays {s0})")
    ax.set_ylabel(labels[1] if labels else f"P({r1} plays {s1})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(False)
    ax.set_title(title, loc="left")
    return fig, ax


def _mix(p: float, k: int) -> np.ndarray:
    if k == 1:
        return np.array([1.0])
    return np.array([p] + [(1 - p) / (k - 1)] * (k - 1))


def regime_map(df: pd.DataFrame, x: str, y: str, regime: str = "regime", contour: str | None = None,
               title: str = "", xlabel: str | None = None, ylabel: str | None = None,
               order: Sequence[str] | None = None, contour_in: str | None = None,
               contour_label: str = "{:.0%}") -> tuple[Any, Any]:
    """Categorical regime map over a 2-D parameter grid (e.g. bounty × audit rate)."""
    from matplotlib.colors import ListedColormap

    cats = list(order or dict.fromkeys(df[regime]))
    cmap = color_map(cats)
    xs = np.sort(df[x].unique())
    ys = np.sort(df[y].unique())
    Z = np.full((len(ys), len(xs)), np.nan)
    idx = {c: k for k, c in enumerate(cats)}
    for _, r in df.iterrows():
        Z[np.searchsorted(ys, r[y]), np.searchsorted(xs, r[x])] = idx[r[regime]]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    alpha = 0.35
    ax.imshow(Z, origin="lower", aspect="auto", extent=(xs[0], xs[-1], ys[0], ys[-1]),
              cmap=ListedColormap([cmap[c] for c in cats]), vmin=-0.5, vmax=len(cats) - 0.5, alpha=alpha,
              interpolation="nearest")
    if contour:
        C = np.full((len(ys), len(xs)), np.nan)
        for _, r in df.iterrows():
            if contour_in is None or r[regime] == contour_in:
                C[np.searchsorted(ys, r[y]), np.searchsorted(xs, r[x])] = r[contour]
        cs = ax.contour(xs, ys, C, levels=[0.25, 0.5, 0.75], colors=INK2, linewidths=1)
        ax.clabel(cs, fmt=lambda v: contour_label.format(v), fontsize=7.5)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(facecolor=cmap[c], alpha=alpha, label=c) for c in cats], loc="upper right")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(False)
    ax.set_title(title, loc="left")
    return fig, ax


def learning_curves(df: pd.DataFrame, prefix: str = "p[", x: str = "iteration", title: str = "Policy during training",
                    ylabel: str = "probability") -> tuple[Any, Any]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cmap = color_map(cols[:8])
    for c in cols[:8]:
        ax.plot(df[x], df[c], color=cmap[c], lw=2, label=c[len(prefix):-1])
        ax.annotate(c[len(prefix):-1].split(":")[-1], (df[x].iloc[-1], df[c].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=7.5, color=INK2, va="center")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.12, 0.5))
    return fig, ax


def line_compare(df: pd.DataFrame, x: str, y: str, group: str, title: str = "", xlabel: str | None = None,
                 ylabel: str | None = None, band: tuple[str, str] | None = None) -> tuple[Any, Any]:
    """Simple multi-series line chart (one y-axis) with end labels and a legend."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    groups = list(dict.fromkeys(df[group]))
    cmap = color_map([str(g) for g in groups])
    for g in groups:
        d = df[df[group] == g].sort_values(x)
        ax.plot(d[x], d[y], color=cmap[str(g)], lw=2, label=str(g))
        ax.scatter(d[x], d[y], s=22, color=cmap[str(g)], edgecolor=SURFACE, linewidth=1.4, zorder=3)
        if band and band[0] in d:
            ax.fill_between(d[x], d[band[0]], d[band[1]], color=cmap[str(g)], alpha=0.10, lw=0)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title, loc="left")
    if len(groups) > 1:
        ax.legend(loc="best")
    return fig, ax
