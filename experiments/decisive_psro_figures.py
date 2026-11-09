"""Figures for the PSRO self-play stage and the instant-judge replay (``docs/results/eic/fig10*``, ``fig11*``).

* fig10_psro_equilibrium: held-out ASD at each PSRO iteration's equilibrium, sequential and simultaneous debate;
  the three judges pooled (blue, 95% item-bootstrap band) and each judge (gray); the blind judge on the same items
  (dashed).
* fig11_instant_best_of_n: ASD along best-of-n with the liar alone or both sides optimized, one-round consultancy
  and debate: instant judges (no reasoning, near-deterministic; in-sample = held-out) against reasoning judges
  scored on independent samples.

    python experiments/decisive_psro_figures.py --run runs/decisive --out docs/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import eic_style as st
from eic_style import BLUE, CONTEXT, INK2, ORANGE

JUDGE_NAMES = {"gpt-4o-mini": "gpt-4o-mini", "llama-3.1-8b": "Llama-3.1-8B", "gemma-3-12b": "Gemma-3-12B"}
NAMES = {"debate_seq": "Sequential debate", "debate_sim": "Simultaneous debate",
         "consultancy_1r": "Consultancy, 1 round", "debate_1r": "Debate, 1 round"}


def psro_blind(run: Path) -> float | None:
    """Blind-judge ASD (log score, judges pooled) on the PSRO held-out items."""
    from decisive_eic import blind_judge

    states = sorted((run / "psro").glob("*/*/state.json"))
    if not states:
        return None
    test = set(json.loads(states[0].read_text())["test"])
    b = blind_judge(run, {"psro held-out": test})
    r = b[(b.items_set == "psro held-out") & (b.unit == "log score") & (b.judge == "all")]
    return float(r.asd.iloc[0]) if len(r) else None


def fig_psro(t: pd.DataFrame, blind: float | None, out: Path):
    mechs = [m for m in ("debate_seq", "debate_sim") if m in set(t.protocol)]
    fig, axes = plt.subplots(1, len(mechs), figsize=(4.4 * len(mechs), 3.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], mechs):
        st.axes(ax)
        ends = []
        for j in JUDGE_NAMES:
            g = t[(t.protocol == m) & (t.judge == j)].sort_values("iteration")
            if g.empty:
                continue
            ax.plot(g.iteration, g.asd, color=CONTEXT, lw=1.2, zorder=1)
            ends.append((g.iteration.iloc[-1], g.asd.iloc[-1], JUDGE_NAMES[j]))
        g = t[(t.protocol == m) & (t.judge == "pooled")].sort_values("iteration")
        st.line_ci(ax, g.iteration, g.asd, g.asd_lo, g.asd_hi, BLUE, label="3 judges pooled")
        st.zero(ax, "h")
        if blind is not None:
            ax.axhline(blind, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.text(0.0, blind, " blind judge", color=INK2, fontsize=7, va="bottom", ha="left")
        ax.set_title(NAMES[m])
        ax.set_xticks(sorted(set(g.iteration)))
        ax.set_xlabel("PSRO iteration (0 = unprompted debaters)")
        ax.set_xlim(-0.3, g.iteration.max() + 1.6)
        if ends:
            st.spread_labels(ax, ends[0][0] + 0.15, [e[1] for e in ends], [e[2] for e in ends])
    axes[0][0].set_ylabel("held-out ASD at equilibrium (log score)")
    seq = t[(t.protocol == "debate_seq") & (t.judge == "pooled")].sort_values("iteration")
    sim = t[(t.protocol == "debate_sim") & (t.judge == "pooled")].sort_values("iteration")
    title = "Self-play: honesty pays at equilibrium in simultaneous debate"
    if len(seq) and len(sim) and seq.asd_lo.iloc[-1] <= 0 < sim.asd_lo.iloc[-1]:
        title += "; sequential debate's incentive is optimized away"
    fig.suptitle(title.replace("; ", ";\n"), x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.17)
    fig.text(0.005, 0.925, "Honest and lying gpt-5-mini strategies best-respond to each other's Nash mixture (prompt "
             "search on 100 training items,\njudge = mean of 3 samples); ASD = E[honest - liar reward] at each "
             "equilibrium on 390 held-out items.\nBlue: judges pooled, 95% item-bootstrap band; gray: each judge.",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    st.save(fig, out, "fig10_psro_equilibrium")


def fig_instant(s: pd.DataFrame, out: Path):
    s = s[s.judge == "pooled"]
    mechs = ["consultancy_1r", "debate_1r"]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), sharey=True)
    panels = [(m, mode) for m in mechs for mode in ("liar", "both")]
    for ax, (m, mode) in zip(axes, panels):
        st.axes(ax)
        for judging, color, name in (("reasoning (held out)", BLUE, "reasoning judge"),
                                     ("instant", ORANGE, "instant judge")):
            g = s[(s.judging == judging) & (s.mechanism == m) & (s["mode"] == mode)].sort_values("level")
            if g.empty:
                continue
            st.line_ci(ax, g.level, g.asd, g.asd_lo, g.asd_hi, color, label=name)
            st.end_label(ax, g.level.iloc[-1], g.asd.iloc[-1], name.split()[0])
        st.zero(ax, "h")
        st.log2_axis(ax, [1, 2, 4, 8, 16])
        ax.set_xlim(0.8, 40)
        ax.set_title(f"{NAMES[m]}: {'liar optimized' if mode == 'liar' else 'both optimized'}")
        ax.set_xlabel("best-of-n against the judge's reward")
    axes[0].set_ylabel("ASD (log score)")
    fig.suptitle("Without reasoning noise the judge is easier to exploit: best-of-16 liars drive the honesty "
                 "incentive far below zero", x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.04)
    fig.text(0.005, 0.955, "Orange: weak judges answering at once (no reasoning, temperature 0; two calls agree at "
             "r = 0.78-0.95), so the payoff that selects is the payoff reported. Blue: the same judges reasoning first, "
             "scored on samples independent of selection. gpt-5-mini, 150 items, 3 judges pooled, 95% bands.",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    st.save(fig, out, "fig11_instant_best_of_n")


def make_all(run: Path, res: Path) -> None:
    out = res / "eic"
    p, b = res / "psro.csv", res / "instant" / "bon.csv"
    blind = psro_blind(run) if (run / "psro").exists() else None
    for paper in (False, True):
        ctx = st.paper_mode() if paper else _null()
        with ctx:
            if p.exists():
                fig_psro(pd.read_csv(p), blind, out)
            if b.exists():
                fig_instant(pd.read_csv(b), out)


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results")
    a = ap.parse_args(argv)
    make_all(Path(a.run), Path(a.out))


if __name__ == "__main__":
    main()
