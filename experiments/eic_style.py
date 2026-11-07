"""Figure style for the incentive-compatibility report (static figures for the paper and docs).

Follows the data-viz method used across so-arena's reports:
- palette: the validated reference palette. All-pairs forms (dot plots, scatter) use at most its first three
  slots, which pass every all-pairs check on the light surface. Aqua is below 3:1 contrast, so every series
  is also direct-labeled.
- marks: hairline solid grid, 2px lines, >= 8px markers with a 2px surface ring, a 10% wash for intervals.
- text: always in text tokens, never in series colors.
- one axis per plot; small multiples rather than dual axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SERIES = [BLUE, ORANGE, AQUA]
CONTEXT = "#c9c7c0"  # de-emphasis gray for context marks

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 4, "ytick.major.pad": 4,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 7.5, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round", "axes.titlelocation": "left", "axes.titleweight": "bold",
    "axes.titlepad": 8,
})


def axes(ax, *, grid: str = "both"):
    """Recessive frame: no top/right spines, hairline grid behind the data."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if grid == "x" else "y" if grid == "y" else "both")
    if grid == "none":
        ax.grid(False)
    return ax


def zero(ax, orient: str = "v", label: str | None = None, pos=None):
    """The reference line where the protocol gives no incentive either way."""
    if orient == "v":
        ax.axvline(0, color=INK2, lw=1, zorder=1)
        if label:
            ax.text(0, pos if pos is not None else 1.0, f" {label}", transform=ax.get_xaxis_transform(),
                    color=INK2, fontsize=7, va="top", ha="left")
    else:
        ax.axhline(0, color=INK2, lw=1, zorder=1)
        if label:
            ax.text(pos if pos is not None else 0.0, 0, label, transform=ax.get_yaxis_transform(), color=INK2,
                    fontsize=7, va="bottom", ha="left")


def dot_ci(ax, x, y, lo, hi, color, *, size=7, label=None, horizontal=True):
    """A point estimate with its 95% interval: a hairline range and a ringed dot."""
    if horizontal:
        ax.hlines(y, lo, hi, color=color, lw=1.6, zorder=2)
    else:
        ax.vlines(x, lo, hi, color=color, lw=1.6, zorder=2)
    ax.plot(x, y, "o", color=color, ms=size, mec=SURFACE, mew=1.5, zorder=3, label=label)


def line_ci(ax, x, y, lo, hi, color, *, label=None, ls="-", marker="o"):
    """A 2px line with a 10% wash for its interval and ringed markers."""
    ax.fill_between(x, lo, hi, color=color, alpha=0.10, lw=0, zorder=1)
    ax.plot(x, y, ls=ls, marker=marker, color=color, lw=2, ms=6.5, mec=SURFACE, mew=2, zorder=3, label=label)


def end_label(ax, x, y, text, *, dx=4, dy=0, ha="left"):
    """Direct label at a series' end, in text ink (identity comes from the mark beside it)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK2, fontsize=7.2, va="center",
                ha=ha)


def spread_labels(ax, x: float, ys, texts, *, min_gap: float = 0.07, dx: int = 4):
    """End labels for several series at one x, pushed apart vertically (in axes fraction) so none overlap,
    each joined to its line end by a hairline leader when moved."""
    lo, hi = ax.get_ylim()
    frac = [(y - lo) / (hi - lo) for y in ys]
    order = np.argsort(frac)
    placed = list(np.array(frac)[order])
    for i in range(1, len(placed)):
        placed[i] = max(placed[i], placed[i - 1] + min_gap)
    shift = max(0.0, placed[-1] - 0.98) if placed else 0.0
    placed = [p - shift for p in placed]
    for k, i in enumerate(order):
        y_lab = lo + placed[k] * (hi - lo)
        ax.annotate(texts[i], (x, ys[i]), xytext=(x, y_lab), textcoords="data", color=INK2, fontsize=7.2,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(y_lab - ys[i]) > 1e-9 else None)


PAPER = False  # paper figures: no in-figure title or subtitle (the LaTeX caption carries them)


class paper_mode:
    """Render figures for the paper: figure-level titles and notes are dropped and files get a ``_paper``
    suffix; axes titles (panel names) stay."""

    def __enter__(self):
        global PAPER
        from matplotlib.figure import Figure

        PAPER = True
        self._sup, self._text = Figure.suptitle, Figure.text
        Figure.suptitle = lambda self_, *a, **k: None
        Figure.text = lambda self_, *a, **k: None
        return self

    def __exit__(self, *exc):
        global PAPER
        from matplotlib.figure import Figure

        PAPER = False
        Figure.suptitle, Figure.text = self._sup, self._text


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    if PAPER:
        name = f"{name}_paper"
        for ax in fig.axes:  # single-panel figures carry their headline in the axes title: drop it too
            if len(fig.axes) == 1 or (len(fig.axes) == 2 and ax.get_label() == "<colorbar>"):
                ax.set_title("")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()


def nice(v: float, pct: bool = False) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{100 * v:+.0f}" if pct else f"{v:+.2f}"
