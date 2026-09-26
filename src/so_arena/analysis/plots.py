"""Plots for incentive-compatibility analyses.

Every chart is drawn with one validated palette (categorical slots in fixed order; first three used
for all-pairs forms such as parametric curves) and one set of mark specs (thin marks, 2px lines, 8px
markers with a surface ring, hairline solid grid). Charts render for a light or dark surface, save
as PNG/SVG for papers, and export as SVG with a native tooltip on every mark for the HTML report.

Chart forms:

* :func:`asd_bars` - ASD (or any signed metric) per mechanism: diverging bars from zero with CIs
  (blue = rewards honesty, red = rewards deception).
* :func:`parametric_curves` - mechanism reward (x) vs ground-truth value (y) as optimization pressure
  grows (best-of-n, tilting beta, prompt-search iterations, training steps); one line per series.
* :func:`heatmap` - a value over a 2D grid (e.g. ground truth over proposer n x critic n).
* :func:`threshold_curves` - basin boundaries (e.g. the snitching threshold p* vs bounty ratio).
* :func:`line_chart` - generic multi-series lines (e.g. learning trajectories).
"""

from __future__ import annotations

import io
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "base": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "pos": "#2a78d6", "neg": "#e34948", "mid": "#f0efec",
        "seq": ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "base": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "pos": "#3987e5", "neg": "#e66767", "mid": "#383835",
        "seq": ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7",
                "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"],
    },
}
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
MAX_ALLPAIRS_SERIES = 3


@dataclass
class Chart:
    """A rendered figure plus per-mark tooltips (gid -> text) and a table view of its data."""

    fig: plt.Figure
    tooltips: dict[str, str] = field(default_factory=dict)
    table: pd.DataFrame | None = None
    title: str = ""

    def save(self, path: str | Path, dpi: int = 200) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=dpi, facecolor=self.fig.get_facecolor())
        return path

    def svg(self) -> str:
        return figure_svg(self.fig, self.tooltips)

    def close(self) -> None:
        plt.close(self.fig)


def _setup(mode: str, figsize=(6.4, 3.6)):
    t = THEMES[mode]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none",
        "axes.titlesize": 10, "axes.labelsize": 9,
    })
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(t["surface"])
    _style(ax, t)
    return fig, ax, t


def _style(ax, t) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["base"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], length=0, pad=4)
    ax.grid(True, color=t["grid"], linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])


def _title(ax, t, title: str, subtitle: str | None = None, width: int = 84) -> None:
    import textwrap

    lines = textwrap.wrap(subtitle, width) if subtitle else []
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, color=t["ink"], fontweight="bold",
                     pad=8 + 12 * len(lines))
    if lines:
        ax.text(0, 1.02, "\n".join(lines), transform=ax.transAxes, fontsize=8.5, color=t["ink2"], va="bottom",
                ha="left", linespacing=1.3)


def figure_svg(fig: plt.Figure, tooltips: dict[str, str]) -> str:
    """SVG text of a figure with a native ``<title>`` tooltip on every mark that has a gid."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", facecolor=fig.get_facecolor())
    raw = buf.getvalue()
    raw = re.sub(r"<\?xml[^>]*\?>\s*", "", raw)
    raw = re.sub(r"<!DOCTYPE[^>]*>\s*", "", raw)
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(raw)
    for el in root.iter():
        gid = el.get("id")
        if gid and gid in tooltips:
            title = ET.Element(f"{{{ns}}}title")
            title.text = tooltips[gid]
            el.insert(0, title)
            el.set("class", "mark")
            el.set("tabindex", "0")
    root.set("width", "100%")
    root.attrib.pop("height", None)
    root.set("role", "img")
    out = ET.tostring(root, encoding="unicode")
    return out.replace("font-family:'DejaVu Sans'", f"font-family:{FONT}").replace("font-family: 'DejaVu Sans'", f"font-family:{FONT}")


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "–"
    a = abs(v)
    return f"{v:.3f}" if a < 10 else f"{v:,.1f}"


# ------------------------------------------------------------------------------------ bars


def _rounded_hbar(ax, y: float, v: float, thick: float, color: str, gid: str, r_pt: float = 3.0):
    """Horizontal bar from 0 to v: rounded data-end (radius ~4px), square at the baseline.

    Call after limits and layout are final: the radius is made circular in display space.
    """
    if v == 0:
        return None
    fig = ax.figure
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    ppd_x, ppd_y = bb.width / (x1 - x0), bb.height / (y1 - y0)
    aspect = ppd_x / ppd_y
    r = min(r_pt * fig.dpi / 72 / ppd_x, abs(v) / 2, thick / 2 / aspect)
    left = min(0.0, v)
    bar = FancyBboxPatch((left, y - thick / 2), abs(v), thick, boxstyle=f"round,pad=0,rounding_size={r}",
                         mutation_aspect=aspect, linewidth=0, facecolor=color, zorder=2)
    ax.add_patch(bar)
    base = Rectangle((0.0 if v > 0 else -r, y - thick / 2), r, thick, linewidth=0, facecolor=color, zorder=2)
    ax.add_patch(base)
    bar.set_gid(gid)
    return bar


def asd_bars(table: pd.DataFrame, *, value: str = "asd", label: str = "mechanism", lo: str = "ci_low",
             hi: str = "ci_high", title: str = "Agent score difference", subtitle: str | None = None,
             xlabel: str = "reward(arguing for truth) − reward(arguing for falsehood)", mode: str = "light") -> Chart:
    """Signed metric per category with CI whiskers; blue when positive (honesty pays), red when negative."""
    d = table.reset_index(drop=True)
    n = len(d)
    fig, ax, t = _setup(mode, figsize=(6.4, max(1.6, 0.42 * n + 1.2)))
    tips: dict[str, str] = {}
    vals = d[value].to_numpy(dtype=float)
    los = d[lo].to_numpy(dtype=float) if lo in d else vals.copy()
    his = d[hi].to_numpy(dtype=float) if hi in d else vals.copy()
    allv = np.concatenate([vals, los, his])
    allv = allv[np.isfinite(allv)]
    span = float(np.max(np.abs(allv))) if len(allv) else 1.0
    span = span or 1.0
    ax.set_yticks(range(n))
    ax.set_yticklabels(list(d[label])[::-1])
    ax.set_ylim(-0.7, n - 0.3)
    xmin = min(0.0, float(np.min(allv)) if len(allv) else 0.0)
    xmax = max(0.0, float(np.max(allv)) if len(allv) else 0.0)
    ax.set_xlim(xmin - 0.28 * span, xmax + 0.28 * span)
    ax.axvline(0, color=t["base"], linewidth=0.8, zorder=1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    _title(ax, t, title, subtitle)
    for i, (v, a, b) in enumerate(zip(vals, los, his)):
        y = n - 1 - i
        if np.isfinite(a) and np.isfinite(b):
            ax.plot([a, b], [y, y], color=t["ink2"], linewidth=1.0, solid_capstyle="butt", zorder=3)
            for e in (a, b):
                ax.plot([e, e], [y - 0.09, y + 0.09], color=t["ink2"], linewidth=1.0, zorder=3)
        tip_x = (max(v, b) if np.isfinite(b) else v) if v >= 0 else (min(v, a) if np.isfinite(a) else v)
        ax.text(tip_x + (0.02 if v >= 0 else -0.02) * span, y, _fmt(v), va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=t["ink"], zorder=4)
    fig.tight_layout()
    thick = 0.5
    for i, v in enumerate(vals):
        if not np.isfinite(v):
            continue
        gid = f"bar{i}"
        _rounded_hbar(ax, n - 1 - i, float(v), thick, t["pos"] if v >= 0 else t["neg"], gid)
        tips[gid] = f"{d[label].iloc[i]}: {_fmt(v)}" + (
            f" (95% CI {_fmt(los[i])} to {_fmt(his[i])})" if lo in d and np.isfinite(los[i]) else "")
    cols = [c for c in (label, value, lo, hi) if c in d]
    return Chart(fig, tips, d[cols], title)


# ------------------------------------------------------------------------------------ curves


def parametric_curves(df: pd.DataFrame, *, x: str, y: str, series: str | None = None, level: str | None = None,
                      title: str = "Optimization pressure", subtitle: str | None = None,
                      xlabel: str = "mechanism reward", ylabel: str = "ground-truth value",
                      level_name: str = "n", mode: str = "light") -> Chart:
    """Parametric curves: each series traces (x, y) as its optimization level increases.

    The start and end of each series are labelled with their level; every point carries a tooltip.
    More than three series are split into panels (all-pairs colour safety).
    """
    groups = [(None, df)] if series is None else list(df.groupby(series, sort=False))
    panels = [groups[i:i + MAX_ALLPAIRS_SERIES] for i in range(0, len(groups), MAX_ALLPAIRS_SERIES)] or [[]]
    t = THEMES[mode]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, len(panels), figsize=(min(6.4 * len(panels), 12.8), 4.0), squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    tips: dict[str, str] = {}
    k = 0
    for pi, (ax, panel) in enumerate(zip(axes[0], panels)):
        _style(ax, t)
        for si, (name, g) in enumerate(panel):
            g = g.sort_values(level) if level else g
            c = t["series"][si]
            ax.plot(g[x], g[y], color=c, linewidth=2, solid_joinstyle="round", solid_capstyle="round", zorder=2,
                    label=str(name) if name is not None else None)
            for _, row in g.iterrows():
                (pt,) = ax.plot([row[x]], [row[y]], marker="o", markersize=7, color=c, markeredgecolor=t["surface"],
                                markeredgewidth=1.5, zorder=3, linestyle="none")
                gid = f"pt{k}"
                k += 1
                pt.set_gid(gid)
                lv = f"{level_name}={row[level]:g}, " if level else ""
                tips[gid] = f"{name + ': ' if name is not None else ''}{lv}{xlabel} {_fmt(row[x])}, {ylabel} {_fmt(row[y])}"
            if level and len(g) >= 2:
                for row, ha in ((g.iloc[0], "right"), (g.iloc[-1], "left")):
                    ax.annotate(f"{level_name}={row[level]:g}", (row[x], row[y]), xytext=(6 if ha == "left" else -6, 4),
                                textcoords="offset points", fontsize=7.5, color=t["ink2"], ha=ha)
        ax.set_xlabel(xlabel)
        if pi == 0:
            ax.set_ylabel(ylabel)
        if series is not None and len(panel) >= 2:
            leg = ax.legend(frameon=False, fontsize=8, loc="best", labelcolor=t["ink2"])
            for line in leg.get_lines():
                line.set_linewidth(2)
        elif series is not None and len(panel) == 1:
            ax.set_title(str(panel[0][0]), loc="left", fontsize=9, color=t["ink2"])
        ax.margins(0.12)
    _title(axes[0][0], t, title, subtitle)
    fig.tight_layout()
    cols = [c for c in (series, level, x, y) if c]
    return Chart(fig, tips, df[cols].reset_index(drop=True), title)


def line_chart(df: pd.DataFrame, *, x: str, ys: Sequence[str], labels: Sequence[str] | None = None,
               title: str = "", subtitle: str | None = None, xlabel: str = "", ylabel: str = "",
               ylim: tuple[float, float] | None = None, bands: dict[str, tuple[str, str]] | None = None,
               vlines: Sequence[tuple[float, str]] = (), dashed: Sequence[str] = (), mode: str = "light") -> Chart:
    """Multi-series lines (<= 4 per chart), direct-labelled at the right end, with a legend.

    ``bands`` maps a series to its (low, high) columns, drawn as a light band in the series' colour (e.g. a
    95% CI across seeds); ``vlines`` are labelled reference lines ``(x, label)`` (e.g. a theoretical threshold);
    ``dashed`` series are drawn dashed (e.g. a theoretical prediction next to its simulation).
    """
    if len(ys) > 4:
        raise ValueError("at most 4 series per line chart; facet into several charts")
    fig, ax, t = _setup(mode, figsize=(6.4, 3.6))
    labels = list(labels or ys)
    bands = dict(bands or {})
    tips: dict[str, str] = {}
    step = max(1, len(df) // 40)
    for xv, lab in vlines:
        ax.axvline(xv, color=t["base"], linewidth=0.8, zorder=1)
        ax.text(xv, 0.98, " " + lab, transform=ax.get_xaxis_transform(), fontsize=7.5, color=t["ink2"], ha="left",
                va="top", rotation=90)
    for i, (col, lab) in enumerate(zip(ys, labels)):
        c = t["series"][i]
        if col in bands:
            lo, hi = bands[col]
            ax.fill_between(df[x], df[lo], df[hi], color=c, alpha=0.18, linewidth=0, zorder=1)
        ax.plot(df[x], df[col], color=c, linewidth=2, solid_joinstyle="round", solid_capstyle="round", label=lab,
                linestyle="--" if col in dashed else "-")
        for j in range(0, len(df), step):
            (pt,) = ax.plot([df[x].iloc[j]], [df[col].iloc[j]], marker="o", markersize=10, alpha=0.0, linestyle="none")
            gid = f"l{i}_{j}"
            pt.set_gid(gid)
            ci = (f" (95% CI {_fmt(df[bands[col][0]].iloc[j])} to {_fmt(df[bands[col][1]].iloc[j])})"
                  if col in bands else "")
            tips[gid] = f"{lab}: {_fmt(df[col].iloc[j])}{ci} at {x}={df[x].iloc[j]:g}"
        last = df.iloc[-1]
        (end,) = ax.plot([last[x]], [last[col]], marker="o", markersize=7, color=c, markeredgecolor=t["surface"],
                         markeredgewidth=1.5, linestyle="none")
    if len(ys) >= 2:
        ax.legend(frameon=False, fontsize=8, loc="best", labelcolor=t["ink2"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    _title(ax, t, title, subtitle)
    fig.tight_layout()
    cols = [x, *ys, *(c for col in ys if col in bands for c in bands[col])]
    return Chart(fig, tips, df[cols].reset_index(drop=True), title)


def threshold_curves(df: pd.DataFrame, *, x: str, y: str, series: str, title: str = "", subtitle: str | None = None,
                     xlabel: str = "", ylabel: str = "", below: str = "", above: str = "", vline: float | None = None,
                     vline_label: str = "", mode: str = "light") -> Chart:
    """Boundary curves (e.g. the unstable threshold p* separating two basins), one per series (<= 3)."""
    groups = list(df.groupby(series, sort=True))[:MAX_ALLPAIRS_SERIES]
    fig, ax, t = _setup(mode, figsize=(6.4, 3.8))
    tips: dict[str, str] = {}
    k = 0
    for i, (name, g) in enumerate(groups):
        g = g.sort_values(x)
        c = t["series"][i]
        ax.plot(g[x], g[y], color=c, linewidth=2, label=f"{series} = {name}", solid_capstyle="round")
        for _, row in g.iterrows():
            if not np.isfinite(row[y]):
                continue
            (pt,) = ax.plot([row[x]], [row[y]], marker="o", markersize=9, alpha=0.0, linestyle="none")
            gid = f"t{k}"
            k += 1
            pt.set_gid(gid)
            tips[gid] = f"{series}={name}: {ylabel or y} {_fmt(row[y])} at {xlabel or x} {row[x]:g}"
    if vline is not None:
        ax.axvline(vline, color=t["base"], linewidth=0.8)
        ax.text(vline, 0.98, vline_label + " ", transform=ax.get_xaxis_transform(), fontsize=7.5, color=t["ink2"],
                ha="right", va="top")
    # region meanings go in the subtitle: in-plot annotations collide with the curves
    region = "; ".join(x for x in (below, above) if x)
    if region:
        subtitle = f"{subtitle}. {region}" if subtitle else region
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(groups) >= 2:
        ax.legend(frameon=False, fontsize=8, loc="center right", labelcolor=t["ink2"])
    _title(ax, t, title, subtitle)
    fig.tight_layout()
    return Chart(fig, tips, df[[series, x, y]].reset_index(drop=True), title)


# ------------------------------------------------------------------------------------ heatmap


def _to_oklab(color: str) -> np.ndarray:
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (lin(c) for c in matplotlib.colors.to_rgb(color))
    lms = np.cbrt([0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b,
                   0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b,
                   0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b])
    return np.array([0.2104542553 * lms[0] + 0.7936177850 * lms[1] - 0.0040720468 * lms[2],
                     1.9779984951 * lms[0] - 2.4285922050 * lms[1] + 0.4505937099 * lms[2],
                     0.0259040371 * lms[0] + 0.7827717662 * lms[1] - 0.8086757660 * lms[2]])


def _from_oklab(lab: np.ndarray) -> str:
    l_, m_, s_ = (lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2],
                  lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2],
                  lab[0] - 0.0894841775 * lab[1] - 1.2914855480 * lab[2])
    lin = np.array([4.0767416621 * l_**3 - 3.3077115913 * m_**3 + 0.2309699292 * s_**3,
                    -1.2684380046 * l_**3 + 2.6097574011 * m_**3 - 0.3413193965 * s_**3,
                    -0.0041960863 * l_**3 - 0.7034186147 * m_**3 + 1.7076147010 * s_**3])

    def enc(c: float) -> float:
        c = min(1.0, max(0.0, c))
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    return matplotlib.colors.to_hex([enc(c) for c in lin])


def diverging_color(v: float, span: float, t: dict, steps: int = 6) -> str:
    """Diverging encoding: the neutral midpoint at 0, blue (positive) and red (negative) poles at
    ``+-span``, interpolated in OKLab in ``steps`` equal steps per arm (monotone lightness per arm)."""
    frac = 0.0 if span <= 0 else min(1.0, abs(v) / span)
    k = round(frac * steps) / steps
    mid, pole = _to_oklab(t["mid"]), _to_oklab(t["pos"] if v >= 0 else t["neg"])
    return _from_oklab(mid + k * (pole - mid))


def heatmap(df: pd.DataFrame, *, x: str, y: str, value: str, title: str = "", subtitle: str | None = None,
            xlabel: str = "", ylabel: str = "", value_label: str = "", vmin: float | None = None,
            vmax: float | None = None, center: float | None = None, order_x: Sequence | None = None,
            order_y: Sequence | None = None, tip_extra: dict[tuple, str] | None = None, decimals: int | None = None,
            mode: str = "light") -> Chart:
    """Heatmap with 2px surface gaps and in-cell labels: sequential one-hue by default; with ``center``
    (e.g. 0 for ASD) diverging - blue above, red below, neutral gray at the center.

    ``order_x``/``order_y`` fix the column/row order (rows are drawn top to bottom in ``order_y``);
    ``tip_extra[(x, y)]`` appends text to a cell's tooltip (e.g. a confidence interval); ``decimals``
    fixes the in-cell number format. The figure is sized from the row labels and the number of columns."""
    import textwrap

    piv = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    if order_x is not None:
        piv = piv.reindex(columns=[c for c in order_x if c in piv.columns])
    if order_y is not None:
        piv = piv.reindex(index=[r for r in order_y if r in piv.index][::-1])
    label_in = 0.068 * max((len(str(r)) for r in piv.index), default=4) + 0.5
    width = max(3.6, label_in + 0.95 * piv.shape[1] + 0.6, 0.075 * len(title) + 0.6)
    fig, ax, t = _setup(mode, figsize=(width, max(2.8, 0.42 * piv.shape[0] + 1.9)))
    cell = (lambda v: f"{v:.{decimals}f}") if decimals is not None else _fmt
    ramp = t["seq"]
    lo = np.nanmin(piv.values) if vmin is None else vmin
    hi = np.nanmax(piv.values) if vmax is None else vmax
    span = max(abs(lo - center), abs(hi - center)) if center is not None else 0.0
    tips: dict[str, str] = {}
    gap = 0.04
    for i, yv in enumerate(piv.index):
        for j, xv in enumerate(piv.columns):
            v = piv.loc[yv, xv]
            if not np.isfinite(v):
                continue
            if center is not None:
                color = diverging_color(v - center, span, t)
            else:
                frac = 0.0 if hi == lo else (v - lo) / (hi - lo)
                color = ramp[min(len(ramp) - 1, int(round(frac * (len(ramp) - 1))))]
            r = Rectangle((j + gap, i + gap), 1 - 2 * gap, 1 - 2 * gap, facecolor=color, linewidth=0)
            ax.add_patch(r)
            gid = f"c{i}_{j}"
            r.set_gid(gid)
            tips[gid] = f"{xlabel or x}={xv}, {ylabel or y}={yv}: {value_label or value} {_fmt(v)}" + (
                f" {tip_extra[(xv, yv)]}" if tip_extra and (xv, yv) in tip_extra else "")
            rgb = matplotlib.colors.to_rgb(color)
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(j + 0.5, i + 0.5, cell(v), ha="center", va="center", fontsize=8,
                    color="#0b0b0b" if lum > 0.55 else "#ffffff")
    ax.set_xlim(0, piv.shape[1])
    ax.set_ylim(0, piv.shape[0])
    ax.set_xticks(np.arange(piv.shape[1]) + 0.5)
    ax.set_xticklabels([f"{c:g}" if isinstance(c, (int, float)) else "\n".join(textwrap.wrap(str(c), 12))
                        for c in piv.columns])
    ax.set_yticks(np.arange(piv.shape[0]) + 0.5)
    ax.set_yticklabels([f"{c:g}" if isinstance(c, (int, float)) else str(c) for c in piv.index])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if center is not None:
        sub = subtitle or (f"{value_label or value}: blue above {_fmt(center)}, red below, gray near {_fmt(center)}")
    else:
        sub = subtitle or (f"{value_label or value}: lighter = lower ({_fmt(lo)}), darker = higher ({_fmt(hi)})")
    # title block anchored to the figure (not the axes), so long row labels never push it off the canvas
    w_in, h_in = fig.get_size_inches()
    lines = textwrap.wrap(sub, max(30, int((w_in - 0.3) / 0.066))) if sub else []
    top_in = (0.42 if title else 0.12) + 0.2 * len(lines)
    fig.tight_layout(rect=(0, 0, 1, 1 - top_in / h_in))
    if title:
        fig.text(0.1 / w_in, 1 - 0.12 / h_in, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=t["ink"])
    if lines:
        fig.text(0.1 / w_in, 1 - (0.4 if title else 0.1) / h_in, "\n".join(lines), ha="left", va="top", fontsize=8.5,
                 color=t["ink2"], linespacing=1.3)
    return Chart(fig, tips, piv.reset_index(), title)


def dual_mode(fn, *args, **kwargs) -> dict[str, Chart]:
    """Render a chart for both surfaces (the HTML report shows the one matching the viewer's theme)."""
    return {m: fn(*args, mode=m, **kwargs) for m in ("light", "dark")}
