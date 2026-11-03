"""Figures and the markdown report for ``decisive_eic.py`` (static figures; style in ``eic_style.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import eic_style as st
from eic_style import AQUA, BLUE, CONTEXT, GRID, INK, INK2, MUTED, ORANGE, SURFACE, plt

NAMES = {"consultancy": "Consultancy", "debate_seq": "Debate, sequential", "debate_sim": "Debate, simultaneous",
         "recursive_debate": "Recursive debate", "prover_estimator": "Prover-estimator",
         "consultancy_1r": "Consultancy", "debate_1r": "Debate"}
UNITS = {"recursive_debate": "judge probability", "prover_estimator": "prover payoff"}
PROTOCOLS = ["consultancy", "debate_seq", "debate_sim", "recursive_debate", "prover_estimator"]
JUDGE_ORDER = ["llama-3.1-8b", "gemma-3-12b", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
JUDGE_NAMES = {"llama-3.1-8b": "Llama 3.1 8B", "gemma-3-12b": "Gemma 3 12B", "gpt-4o-mini": "gpt-4o-mini",
               "gpt-4.1-mini": "gpt-4.1-mini", "gpt-4.1": "gpt-4.1"}


def _pick(t: pd.DataFrame | None, **kw) -> pd.DataFrame:
    if t is None or t.empty:
        return pd.DataFrame()
    m = np.ones(len(t), bool)
    for k, v in kw.items():
        if k in t:
            m &= (t[k] == v).to_numpy()
    return t[m]


# --------------------------------------------------------------------------- 1. headline: EIC per protocol x sampler

def fig_headline(tables, out: Path):
    rows = []
    for m in PROTOCOLS:
        for label, t, sel in (
                ("Instructed (ASD)", tables.get("instructed"), dict(mechanism=m, judge="all", debater="gpt-5-mini")),
                ("Instructed, committed liars", tables.get("committed"),
                 dict(mechanism=m, judge="all", debater="gpt-5-mini")),
                ("Prompt-optimized, honest vs deceptive", tables.get("po_path"), dict(mechanism=m, judge="all")),
                ("Best-of-16 per speech, both sides", tables.get("bon_3round"),
                 dict(mechanism=m, judge="all", mode="both", n=16)),
                ("Best-of-16 per speech, liar only", tables.get("bon_3round"),
                 dict(mechanism=m, judge="all", mode="liar", n=16))):
            r = _pick(t, **sel)
            if label.startswith("Prompt") and len(r):
                r = r[r["round"] == r["round"].max()]
            if len(r):
                r = r.iloc[0]
                rows.append({"mechanism": m, "sampler": label, "asd": r.asd, "lo": r.asd_lo, "hi": r.asd_hi,
                             "conc": r.concordance, "items": r["items"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    samplers = list(dict.fromkeys(df.sampler))
    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(13.5, 3.0), sharey=True)
    for ax, m in zip(axes, PROTOCOLS):
        st.axes(ax, grid="x")
        g = df[df.mechanism == m].set_index("sampler").reindex(samplers)
        y = np.arange(len(samplers))[::-1]
        for yi, (_, r) in zip(y, g.iterrows()):
            if np.isfinite(r.asd):
                st.dot_ci(ax, r.asd, yi, r.lo, r.hi, BLUE if r.lo > 0 else ORANGE if r.hi < 0 else MUTED)
            else:
                ax.text(0.03, yi, "n/a", transform=ax.get_yaxis_transform(), color=CONTEXT, fontsize=7, va="center")
        st.zero(ax)
        ax.set_title(NAMES[m], fontsize=9)
        ax.set_xlabel(f"ASD ({UNITS.get(m, 'log score')})", fontsize=7.5)
        lo = np.nanmin([g.lo.min(), 0]) if g.lo.notna().any() else -1
        hi = np.nanmax([g.hi.max(), 0]) if g.hi.notna().any() else 1
        pad = 0.12 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_yticks(np.arange(len(samplers))[::-1], samplers)
    fig.suptitle("Instructed liars make every protocol but prover-estimator look incentive compatible; "
                 "optimized liars leave only debate and recursive debate",
                 x=0.005, ha="left", fontsize=10, color=INK, fontweight="bold", y=1.04)
    fig.text(0.005, 0.965, "EIC as ASD: reward for arguing the true answer minus a false one. gpt-5-mini agents, "
             "3 weak judges pooled, 95% item-bootstrap intervals. Blue: CI above zero; orange: below; gray: "
             "overlaps zero.", fontsize=7.5, color=INK2, ha="left")
    fig.tight_layout()
    st.save(fig, out, "fig1_eic_by_sampler")


# --------------------------------------------------------------------------- 2. EIC along best-of-n

def fig_pressure(tables, out: Path):
    b1, b3 = tables.get("bon_1round"), tables.get("bon_3round")
    panels = [("1 round", b1, "consultancy_1r", "Consultancy, 1 round"), ("1 round", b1, "debate_1r", "Debate, 1 round"),
              ("3 rounds", b3, "consultancy", "Consultancy, 3 rounds"), ("3 rounds", b3, "debate_seq", "Debate, 3 rounds")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.2), sharey=False)
    for ax, (_, t, m, title) in zip(axes, panels):
        st.axes(ax)
        sel = dict(mechanism=m, debater="gpt-5-mini") if t is b1 else dict(mechanism=m, judge="all")
        for mode, color, name in (("both", BLUE, "both sides optimized"), ("liar", ORANGE, "liar optimized")):
            g = _pick(t, mode=mode, **sel).sort_values("n")
            if g.empty:
                continue
            st.line_ci(ax, g.n, g.asd, g.asd_lo, g.asd_hi, color, label=name)
            st.end_label(ax, g.n.iloc[-1], g.asd.iloc[-1], name.split()[0])
        st.zero(ax, "h")
        st.log2_axis(ax, sorted(set(_pick(t, **sel).n.astype(int))) if len(_pick(t, **sel)) else [1])
        ax.set_title(title)
        ax.set_xlabel("best-of-n against the judge's reward")
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1] * 1.6)
    axes[0].set_ylabel("ASD (log score)")
    fig.suptitle("Optimizing only the liar erodes the incentive to be honest; three rounds and symmetric "
                 "optimization keep it", x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.04)
    fig.text(0.005, 0.955, "Blue: both sides optimized. Orange: only the liar optimized. gpt-5-mini; held-out judging "
             "(selection and evaluation on independent judge samples); 3 judges pooled; 95% item-bootstrap bands.",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    st.save(fig, out, "fig2_eic_under_best_of_n")


# --------------------------------------------------------------------------- 3. EIC along prompt optimization

def fig_prompt_opt(tables, out: Path):
    t = tables.get("po_path")
    if t is None or t.empty:
        return
    mechs = [m for m in PROTOCOLS if m in set(t.mechanism)]
    fig, axes = plt.subplots(1, len(mechs), figsize=(2.7 * len(mechs), 3.0), sharey=False, squeeze=False)
    for ax, m in zip(axes[0], mechs):
        st.axes(ax)
        ends = []
        for j in [j for j in JUDGE_ORDER if j in set(t.judge)]:
            g = _pick(t, mechanism=m, judge=j).sort_values("round")
            ax.plot(g["round"], g.asd, color=CONTEXT, lw=1.2, zorder=1)
            ends.append((g["round"].iloc[-1], g.asd.iloc[-1], JUDGE_NAMES.get(j, j)))
        g = _pick(t, mechanism=m, judge="all").sort_values("round")
        st.line_ci(ax, g["round"], g.asd, g.asd_lo, g.asd_hi, BLUE, label="3 judges pooled")
        st.zero(ax, "h")
        ax.set_title(NAMES[m])
        ax.set_xticks(sorted(set(g["round"])))
        ax.set_xlabel("search round")
        ax.set_xlim(-0.3, g["round"].max() + 2.2)
        st.spread_labels(ax, ends[0][0] + 0.25, [e[1] for e in ends], [e[2] for e in ends])
    axes[0][0].set_ylabel("held-out ASD (protocol reward units)")
    fig.suptitle("Optimized deception erases consultancy's incentive to be honest and deepens prover-estimator's "
                 "deficit; debate's shrinks but survives", x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.04)
    fig.text(0.005, 0.955, "Blue: 3 judges pooled, with 95% item-bootstrap band; gray: each judge. Round 0 is the "
             "unprompted agent on each side (the instructed sampler); each round keeps the best strategy so far "
             "on 36 training items; ASD on 80 held-out items.", fontsize=7.5, color=INK2)
    fig.tight_layout()
    st.save(fig, out, "fig3_eic_prompt_optimization")


# --------------------------------------------------------------------------- 4. choosing a side

def fig_choice(tables, out: Path):
    t = tables.get("choosing_a_side")
    if t is None or t.empty:
        return
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    st.axes(ax)
    for m, color in (("consultancy_1r", BLUE), ("debate_1r", ORANGE)):
        g = t[t.mechanism == m].sort_values("n")
        if g.empty:
            continue
        st.line_ci(ax, g.n, g.honest, g.honest_lo, g.honest_hi, color, label=NAMES[m])
        st.end_label(ax, g.n.iloc[-1], g.honest.iloc[-1], f"{NAMES[m]} {100 * g.honest.iloc[-1]:.0f}%",
                     dy=7 if m == "debate_1r" else -7)
    ax.axhline(0.5, color=INK2, lw=1)
    ax.text(t.n.max() * 1.25, 0.503, "no preference", color=INK2, fontsize=7, va="bottom")
    st.log2_axis(ax, sorted(set(t.n.astype(int))))
    ax.set_xlim(0.8, t.n.max() * 3.2)
    ax.set_ylim(0.45, 0.75)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel("best-of-n against the judge's reward")
    ax.set_ylabel("selected behaviour defending the truth")
    ax.set_title("An agent free to choose its side is pushed toward honesty", loc="left")
    st.save(fig, out, "fig4_choosing_a_side")


# --------------------------------------------------------------------------- 5. EIC by judge capability

def fig_judges(tables, out: Path):
    t = tables.get("judge_strength")
    if t is None or t.empty:
        return
    t = t[t.liars == "committed"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    for ax, ds, title in zip(axes, ("quality", "testbed"), ("QuALITY: the story is hidden from the judge",
                                                            "Testbed: a capability gap")):
        st.axes(ax)
        g0 = t[t.dataset == ds]
        judges = [j for j in JUDGE_ORDER if j in set(g0.judge)]
        x = np.arange(len(judges))
        for k, (m, color) in enumerate((("debate_seq", BLUE), ("consultancy", ORANGE))):
            g = g0[g0.mechanism == m].set_index("judge").reindex(judges)
            xs = x + (k - 0.5) * 0.16
            ax.plot(xs, g.concordance, color=color, lw=1.4, zorder=2, alpha=0.6)
            for xi, (_, r) in zip(xs, g.iterrows()):
                st.dot_ci(ax, xi, r.concordance, r.conc_lo, r.conc_hi, color, horizontal=False,
                          label=NAMES[m] if xi == xs[0] else None)
        ax.axvspan(-0.5, 2.5, color="#f0efec", zorder=0, lw=0)
        ax.text(1, 1.0, "weak judges", transform=ax.get_xaxis_transform(), ha="center", va="top", color=INK2,
                fontsize=7)
        ax.axhline(0.5, color=INK2, lw=1)
        ax.set_ylim(0.4, 1.0)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ax.set_xticks(x, [JUDGE_NAMES[j] for j in judges], fontsize=7.5)
        ax.set_xlim(-0.5, len(judges) - 0.5)
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    axes[0].set_ylabel("items on which honesty earns more\n(committed liars)")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    fig.suptitle("The incentive to be honest grows with the judge's capability", x=0.005, ha="left", fontsize=10,
                 fontweight="bold", y=1.03)
    fig.text(0.005, 0.95, "Concordance (scale-free: a more confident judge inflates log-score ASD). The same debates "
             "re-judged by five judges, debaters' speeches replayed; both debaters; 95% item-bootstrap intervals.",
             fontsize=7.5, color=INK2)
    fig.tight_layout()
    st.save(fig, out, "fig5_eic_by_judge")


# --------------------------------------------------------------------------- 6. tournament

def fig_tournament(tables, out: Path):
    t = tables.get("tournament")
    if t is None or t.empty:
        return
    a = t[t.liars == "all"].set_index("model")
    c = t[t.liars == "committed"].set_index("model")
    order = a.sort_values("asd").index.tolist()
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    st.axes(ax, grid="x")
    y = np.arange(len(order))
    for yi, m in zip(y, order):
        ax.hlines(yi, min(a.asd[m], c.asd.get(m, np.nan)), max(a.asd[m], c.asd.get(m, np.nan)), color=CONTEXT, lw=2,
                  zorder=1)
        st.dot_ci(ax, a.asd[m], yi, a.asd_lo[m], a.asd_hi[m], MUTED, size=6.5,
                  label="all prompted liars" if yi == 0 else None)
        if m in c.index:
            st.dot_ci(ax, c.asd[m], yi, c.asd_lo[m], c.asd_hi[m], BLUE,
                      label="committed liars only" if yi == 0 else None)
    st.zero(ax)
    ax.set_yticks(y, order)
    ax.set_xlabel("ASD: the model's reward arguing the truth minus lying (log score)")
    ax.set_title("Models that refuse to lie look incentive-compatible; with committed liars they converge",
                 loc="left", fontsize=8.8)
    ax.legend(loc="lower right")
    st.save(fig, out, "fig6_tournament")


# --------------------------------------------------------------------------- 7. overview heatmap

def fig_heatmap(tables, out: Path):
    t = tables.get("committed")
    if t is None or t.empty:
        return
    g = t[(t.debater == "gpt-5-mini") & (t.judge != "all")]
    judges = [j for j in JUDGE_ORDER if j in set(g.judge)]
    mat = g.pivot_table(index="mechanism", columns="judge", values="concordance").reindex(index=PROTOCOLS,
                                                                                         columns=judges)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", BLUE])
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    st.axes(ax, grid="none")
    im = ax.imshow(mat.to_numpy(), cmap=cmap, vmin=0.2, vmax=0.8, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            ax.text(j, i, f"{100 * v:.0f}%", ha="center", va="center", fontsize=8,
                    color=INK if 0.32 < v < 0.68 else SURFACE)
    ax.set_xticks(range(len(judges)), [JUDGE_NAMES[j] for j in judges])
    ax.set_yticks(range(len(PROTOCOLS)), [NAMES[m] for m in PROTOCOLS])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_title("Share of items on which honesty earns more (committed liars, gpt-5-mini)", loc="left", fontsize=8.8)
    st.save(fig, out, "fig7_concordance_heatmap")


def fig_agreement(tables, out: Path):
    cells = tables.get("agreement_cells")
    if cells is None or cells.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    colors = dict(zip([j for j in JUDGE_ORDER if j in set(cells.judge)], (BLUE, ORANGE, AQUA)))
    short = {"consultancy": "cons.", "debate_seq": "seq.", "debate_sim": "sim.", "recursive_debate": "rec.",
             "prover_estimator": "PE"}
    for ax, x, title in zip(axes, ("instructed", "committed"), ("Instructed liars", "Committed liars only")):
        st.axes(ax)
        for j, g in cells.groupby("judge"):
            ax.plot(g[x], g.po, "o", color=colors[j], ms=8, mec=SURFACE, mew=1.5, label=JUDGE_NAMES[j], zorder=3)
            for _, r in g.iterrows():
                ax.annotate(short[r.mechanism], (r[x], r.po), xytext=(5, 3), textcoords="offset points",
                            fontsize=6.8, color=INK2)
        lims = [0.0, 1.0]
        ax.plot(lims, lims, color=CONTEXT, lw=1, zorder=1)
        ax.axhline(0.5, color=INK2, lw=0.8)
        ax.axvline(0.5, color=INK2, lw=0.8)
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.set_xlabel(f"concordance, {title.lower()}")
        ax.set_title(title)
    axes[0].set_ylabel("concordance, prompt-optimized strategies")
    axes[0].legend(loc="upper left")
    fig.suptitle("Does EIC measured with instructed liars predict EIC under optimization? Only loosely",
                 x=0.005, ha="left", fontsize=10, fontweight="bold", y=1.03)
    fig.text(0.005, 0.95, "Each point: one protocol under one judge. x: share of items where honesty out-earns lying "
             "when the agent is just told which side to argue; y: the same after prompt optimization.", fontsize=7.5, color=INK2)
    fig.tight_layout()
    st.save(fig, out, "fig8_sampler_agreement")


# --------------------------------------------------------------------------- 9. Goodhart curves of the prompt search

def fig_goodhart(tables, out: Path):
    t = tables.get("po_goodhart")
    if t is None or t.empty:
        return
    t = t.assign(side=t.side.astype(str).str.lower())  # "true"/"false" read back from CSV as booleans
    mechs = [m for m in PROTOCOLS if m in set(t.mechanism)]
    fig, axes = plt.subplots(1, len(mechs), figsize=(2.7 * len(mechs), 3.0), squeeze=False)
    series = [("true", "train", BLUE, "--", "honest, training"), ("true", "held-out", BLUE, "-", "honest, held-out"),
              ("false", "train", ORANGE, "--", "deceptive, training"),
              ("false", "held-out", ORANGE, "-", "deceptive, held-out")]
    for ax, m in zip(axes[0], mechs):
        st.axes(ax)
        ends = []
        for side, split, color, ls, name in series:
            g = _pick(t, mechanism=m, side=side, split=split).sort_values("round")
            if split == "held-out":
                st.line_ci(ax, g["round"], g.reward, g.reward_lo, g.reward_hi, color)
            else:
                ax.plot(g["round"], g.reward, ls=ls, color=color, lw=1.6, zorder=2)
            ends.append((g["round"].iloc[-1], g.reward.iloc[-1], name))
        ax.set_title(NAMES[m])
        ax.set_xticks(sorted(set(t["round"])))
        ax.set_xlabel("search round")
        ax.set_xlim(-0.3, t["round"].max() + 3.2)
        st.spread_labels(ax, ends[0][0] + 0.25, [e[1] for e in ends], [e[2] for e in ends], min_gap=0.09)
    axes[0][0].set_ylabel("mean reward (protocol units)")
    fig.suptitle("Optimizing against the judge pays off for lying, not for honesty", x=0.005, ha="left",
                 fontsize=10, fontweight="bold", y=1.04)
    fig.text(0.005, 0.955, "Best strategy so far in each search, 3 judges pooled. Dashed: reward on the 36 training "
             "items it was selected on (the proxy the search climbs). Solid, with 95% item-bootstrap band: reward on "
             "80 held-out items.", fontsize=7.5, color=INK2)
    fig.tight_layout()
    st.save(fig, out, "fig9_po_goodhart")


def make_all(tables: dict, out: Path, strategies=None):
    figs = (fig_headline, fig_pressure, fig_prompt_opt, fig_choice, fig_judges, fig_tournament, fig_heatmap,
            fig_agreement, fig_goodhart)
    for paper in (False, True):  # the docs' titled figures, then the paper's caption-only ones
        with (st.paper_mode() if paper else _null()):
            for f in figs:
                try:
                    f(tables, out)
                except Exception as e:  # noqa: BLE001 - a missing stage must not block the other figures
                    print(f"figure {f.__name__} skipped: {e!r}")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# --------------------------------------------------------------------------- markdown

def _fmt(r, col="asd"):
    return f"{r[col]:+.2f} [{r[col + '_lo']:+.2f}, {r[col + '_hi']:+.2f}]"


def report_markdown(tables: dict) -> str:
    L = ["# Empirical incentive compatibility of the oversight protocols", "",
         "EIC: the protocol's reward for arguing the true answer minus its reward for arguing a false one (ASD, in "
         "the protocol's reward units; equal to $\\mathrm{Cov}(u,v)/\\mathrm{Var}(v)$ for binary $v$), with the "
         "share of items on which honesty earns more (concordance) and, where per-episode rewards exist, "
         "$r = \\mathrm{corr}(u, v)$. 95% item-bootstrap intervals. Figures in `eic/`.", ""]

    def table(t, cols, title):
        if t is None or t.empty:
            return
        L.extend([f"## {title}", "", "| " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * len(cols)])
        for _, r in t.iterrows():
            L.append("| " + " | ".join(f(r) for _, f in cols) + " |")
        L.append("")

    proto = lambda r: NAMES.get(r.mechanism, r.mechanism)  # noqa: E731
    for key, title in (("instructed", "Instructed sampler (ASD), gpt-5-mini, 927 items"),
                       ("committed", "Instructed sampler, committed liars only, gpt-5-mini")):
        t = tables.get(key)
        if t is not None:
            t = t[(t.judge == "all") & (t.debater == "gpt-5-mini")]
            table(t, [("protocol", proto), ("ASD", _fmt),
                      ("concordance", lambda r: f"{r.concordance:.2f} [{r.conc_lo:.2f}, {r.conc_hi:.2f}]"),
                      ("r", lambda r: f"{r.r:+.2f} [{r.r_lo:+.2f}, {r.r_hi:+.2f}]"), ("items", lambda r: str(r["items"]))],
                  title)
    t = tables.get("po_path")
    if t is not None:
        t = t[t.judge == "all"].sort_values(["mechanism", "round"])
        table(t, [("protocol", proto), ("search round", lambda r: str(int(r["round"]))), ("held-out ASD", _fmt),
                  ("concordance", lambda r: f"{r.concordance:.2f}"), ("items", lambda r: str(r["items"]))],
              "Prompt optimization: best honest vs best deceptive strategy, held-out")
    t = tables.get("po_goodhart")
    if t is not None and not t.empty:
        k0, k1 = t["round"].min(), t["round"].max()
        piv = t.pivot_table(index=["mechanism", "side", "split"], columns="round", values="reward").reset_index()
        piv["gain"] = piv[k1] - piv[k0]
        table(piv.sort_values(["mechanism", "side", "split"]),
              [("protocol", proto), ("search", lambda r: "honest" if r.side == "true" else "deceptive"),
               ("items", lambda r: r.split), (f"round {k0}", lambda r: f"{r[k0]:+.2f}"),
               (f"round {k1}", lambda r: f"{r[k1]:+.2f}"), ("gain", lambda r: f"{r.gain:+.2f}")],
              "Prompt optimization: reward of the best strategy so far, training vs held-out items")
    t = tables.get("po_population")
    if t is not None:
        table(t[t.judge == "all"], [("protocol", proto), ("ASD", _fmt), ("r", lambda r: f"{r.r:+.2f} [{r.r_lo:+.2f}, {r.r_hi:+.2f}]"),
                                    ("strategy-episodes", lambda r: str(r.episodes))],
              "Prompt optimization: covariance over every searched strategy (training items)")
    for key, title in (("bon_1round", "Exact best-of-n, one round (held-out judging)"),
                       ("bon_3round", "Best-of-n on every speech, three rounds")):
        t = tables.get(key)
        if t is not None:
            t = t[(t.judge == "all") & (t.debater == "gpt-5-mini")].sort_values(["mechanism", "mode", "n"])
            table(t, [("protocol", proto), ("optimized", lambda r: r["mode"]), ("n", lambda r: str(int(r.n))),
                      ("ASD", _fmt), ("concordance", lambda r: f"{r.concordance:.2f}")], title)
    t = tables.get("choosing_a_side")
    if t is not None:
        table(t.sort_values(["mechanism", "n"]), [("protocol", proto), ("n", lambda r: str(int(r.n))),
                                                  ("honest share", lambda r: f"{r.honest:.2f} [{r.honest_lo:.2f}, {r.honest_hi:.2f}]")],
              "Choosing a side: share of the optimized agent's behaviour that is honest")
    t = tables.get("judge_strength")
    if t is not None:
        t = t.copy()
        t["o"] = t.judge.map({j: i for i, j in enumerate(JUDGE_ORDER)})
        table(t.sort_values(["liars", "dataset", "mechanism", "o"]),
              [("liars", lambda r: r.liars), ("dataset", lambda r: r.dataset), ("protocol", proto),
               ("judge", lambda r: r.judge), ("ASD", _fmt), ("concordance", lambda r: f"{r.concordance:.2f}")],
              "EIC by judge capability (the same debates re-judged)")
    t = tables.get("agreement")
    if t is not None:
        table(t, [("samplers", lambda r: r.samplers), ("scope", lambda r: r.scope),
                  ("Spearman", lambda r: f"{r.spearman:+.2f}"), ("cells", lambda r: str(r.cells))],
              "Sampler agreement: rank correlation of EIC (concordance) between samplers")
    t = tables.get("tournament")
    if t is not None:
        table(t.sort_values(["liars", "asd"]), [("liars", lambda r: r.liars), ("model", lambda r: r.model),
                                                ("ASD", _fmt), ("concordance", lambda r: f"{r.concordance:.2f}")],
              "Tournament: each model's reward arguing the truth minus lying")
    return "\n".join(L)
