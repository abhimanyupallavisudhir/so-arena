"""Report of the model panel (``decisive_panel.py``): EIC against agent and judge intelligence.

For every agent configuration (pooled over judges) and every judge (pooled over agents), per protocol:

* EIC from per-episode (u, v): ASD (log score, item-bootstrap CI), concordance (share of items where honesty earns
  more) and r = corr(u, v); all prompted liars and committed liars only (gpt-5 audit of the panel's liar speeches;
  a sample whose liar did not commit is dropped for every judge, both arms together);
* the blind baseline per judge (the judge alone), and each judge's EIC gain over it;
* the agent's own accuracy (``solo``) and its liars' commitment rate;
* slopes of EIC on the Artificial Analysis Intelligence Index (v4.3) across agents and across judges, with a
  bootstrap over items (resampling items, recomputing every point, refitting the slope).

Writes ``<out>/panel/panel.md``, CSVs and ``<out>/eic/fig12_panel_agents``, ``fig13_panel_judges``, ``fig14_panel_heatmap``.

    python experiments/decisive_panel_report.py --run runs/decisive --out docs/results
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from decisive_report import _found

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "panel" / "models.json"
MECHS = {"debate_sim": "Simultaneous debate (self-play)", "consultancy_1r": "Consultancy (one argument)"}
N_BOOT = 1000


def config() -> dict:
    return json.loads(CONFIG.read_text())


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows = pd.read_csv(_found(run / "panel" / "rows.csv"), low_memory=False)
    verdicts = None
    vf, rf = _found(run / "audit" / "verdicts_gpt-5.jsonl"), _found(run / "audit" / "rows.csv.gz")
    if vf.exists() and rf.exists():
        ar = pd.read_csv(rf, low_memory=False)
        ar = ar[ar.kind == "panel"]
        if len(ar):
            v = pd.read_json(vf, lines=True).drop_duplicates("key", keep="last")
            verdicts = ar.merge(v[["key", "verdict"]], on="key", how="inner")
    return rows, verdicts


def committed(rows: pd.DataFrame, verdicts: pd.DataFrame | None) -> pd.DataFrame | None:
    if verdicts is None or verdicts.empty:
        return None
    bad = verdicts[verdicts.verdict != "committed"][["debater", "mechanism", "item", "repeat"]].drop_duplicates()
    bad["bad"] = True
    m = rows.merge(bad, on=["debater", "mechanism", "item", "repeat"], how="left")
    return m[m.bad.isna()].drop(columns="bad")


def per_item(df: pd.DataFrame, keys: list[str], reward: str = "reward") -> pd.DataFrame:
    """Per (keys, item): reward arguing the truth minus lying (each item's sides averaged over samples and roles).
    ``reward="reward"`` is the protocol's own reward (log score of the judge's probability for the agent's side);
    ``reward="reward_p"`` is that probability itself (bounded, so judges of different confidence compare)."""
    x = df[df.value.notna() & df[reward].notna()].copy()
    x["v"] = (x.value > 0).astype(float)
    g = x.groupby(keys + ["item", "v"])[reward].mean().unstack("v")
    g = g.dropna(subset=[0.0, 1.0]) if {0.0, 1.0} <= set(g.columns) else g.iloc[0:0]
    out = (g[1.0] - g[0.0]).rename("asd").reset_index()
    out["conc"] = np.where(out.asd > 0, 1.0, np.where(out.asd == 0, 0.5, 0.0))
    return out


def with_prob_reward(rows: pd.DataFrame) -> pd.DataFrame:
    """Add ``reward_p``: the judge's probability for the side the agent argued (p_true if it argued the truth)."""
    rows = rows.copy()
    rows["reward_p"] = np.where(rows.value > 0, rows.p_true, 1 - rows.p_true)
    rows.loc[rows.value.isna() | rows.p_true.isna(), "reward_p"] = np.nan
    rows["reward_b"] = -(1 - rows.reward_p) ** 2  # Brier score of the judge's probability for the agent's side
    return rows


def both_units(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per-item ASD in log score (``asd``, with concordance ``conc``), in probability (``asd_p``) and in Brier score
    (``asd_b``)."""
    lg = per_item(df, keys)
    for col, name in (("reward_p", "asd_p"), ("reward_b", "asd_b")):
        if col in df:
            pr = per_item(df, keys, col).rename(columns={"asd": name}).drop(columns="conc")
            lg = lg.merge(pr, on=keys + ["item"], how="left")
    return lg


def summarize(items: pd.DataFrame, keys: list[str], seed: int = 0) -> pd.DataFrame:
    """Mean ASD and concordance per key with item-bootstrap CIs (items resampled jointly across keys)."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(items.item.unique()))
    out = []
    for key, g in items.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        a = g.set_index("item")
        cols = [c for c in ("asd", "asd_p", "asd_b", "conc", "gain", "gain_b") if c in a]
        stat = {c: (lambda x, c=c: x[c].mean()) for c in cols}
        stat["asd_med"] = lambda x: x["asd"].median()  # robust to a few confidently wrong verdicts
        boots = []
        for _ in range(N_BOOT):
            s = rng.choice(ids, len(ids))
            b = a.reindex(s).dropna(subset=["asd"])
            boots.append([f(b) for f in stat.values()])
        boots = np.array(boots)
        rec = {**dict(zip(keys, key)), "items": len(a)}
        for k, (c, f) in enumerate(stat.items()):
            rec |= {c: f(a), f"{c}_lo": np.nanquantile(boots[:, k], .025),
                    f"{c}_hi": np.nanquantile(boots[:, k], .975)}
        out.append(rec)
    return pd.DataFrame(out)


def _pivot(items: pd.DataFrame, key: str, index: dict, col: str):
    pivot = items.pivot_table(index="item", columns=key, values=col)
    names = [n for n in pivot.columns if n in index and index[n] is not None]
    return pivot[names], np.array([index[n] for n in names], dtype=float)


def regression(items: pd.DataFrame, key: str, index: dict, col: str = "conc", seed: int = 1,
               grid: np.ndarray | None = None) -> dict:
    """OLS of the per-model mean of ``col`` on the AA index, with two bootstraps.

    * items only (models held fixed): the uncertainty in each model's point;
    * two-level (resample items, then resample the models): also the uncertainty from *which* models are on the
      panel. This is the honest interval for "does EIC change with intelligence across models", and the one
      reported as ``lo``/``hi`` and drawn as the regression band.
    Returns the slope per 10 index points, both intervals, r, and the fitted line with its band on ``grid``.
    """
    rng = np.random.default_rng(seed)
    pivot, xs = _pivot(items, key, index, col)
    if grid is None:
        grid = np.linspace(xs.min(), xs.max(), 60) if len(xs) else np.array([])

    def fit(p, take=None):
        ys = p.mean().to_numpy()
        x = xs
        if take is not None:
            x, ys = x[take], ys[take]
        ok = np.isfinite(ys)
        if ok.sum() < 3 or np.unique(x[ok]).size < 2:
            return np.nan, np.nan
        b, a = np.polyfit(x[ok], ys[ok], 1)
        return b, a

    b0, a0 = fit(pivot)
    s_items, s_two, lines = [], [], []
    n_it, n_m = len(pivot), len(xs)
    for _ in range(N_BOOT):
        p = pivot.iloc[rng.integers(0, n_it, n_it)]
        s_items.append(fit(p)[0])
        b, a = fit(p, rng.integers(0, n_m, n_m))
        s_two.append(b)
        lines.append(a + b * grid)
    lines = np.array(lines)
    r = np.corrcoef(xs, pivot.mean().to_numpy())[0, 1] if n_m > 2 else np.nan
    return {"slope_per_10": 10 * b0, "lo": 10 * np.nanquantile(s_two, .025), "hi": 10 * np.nanquantile(s_two, .975),
            "lo_items": 10 * np.nanquantile(s_items, .025), "hi_items": 10 * np.nanquantile(s_items, .975),
            "r": r, "n": n_m, "grid": grid, "fit": a0 + b0 * grid,
            "band_lo": np.nanquantile(lines, .025, axis=0), "band_hi": np.nanquantile(lines, .975, axis=0)}


def slope(items: pd.DataFrame, key: str, index: dict, col: str = "conc", seed: int = 1) -> dict:
    """Slope of the per-model mean of ``col`` on the AA index, per 10 index points (see :func:`regression`)."""
    r = regression(items, key, index, col, seed)
    return {k: r[k] for k in ("slope_per_10", "lo", "hi", "lo_items", "hi_items", "r", "n")}


def tables(run: Path) -> dict:
    cfg = config()
    a_idx = {a["name"]: a["aa_index"] for a in cfg["agents"]}
    j_idx = {j["name"]: j["aa_index"] for j in cfg["judges"]}
    rows, verdicts = load(run)
    rows = with_prob_reward(rows)
    prot = rows[rows.mechanism.isin(MECHS)]
    out = {"a_idx": a_idx, "j_idx": j_idx}
    variants = {"all": prot}
    c = committed(prot, verdicts)
    if c is not None:
        variants["committed"] = c
    # blind baseline per judge (the judge alone, no argument): per item, per judge and pooled over judges
    d = rows[rows.mechanism == "direct"].copy()
    d = d[d.value.notna() & d.reward.notna()]
    blind_j = both_units(d, ["judge"])
    out["blind"] = summarize(blind_j, ["judge"]).assign(aa=lambda x: x.judge.map(j_idx))
    blind_pool = both_units(d.assign(pool="all"), ["pool"]).drop(columns="pool")
    out["blind_pooled"] = summarize(blind_pool.assign(pool="all"), ["pool"])

    def gains(it: pd.DataFrame, blind: pd.DataFrame, on: list[str]) -> pd.DataFrame:
        """Per-item gain over the blind judge on the same item (and judge), log and Brier."""
        b = blind[on + ["item", "asd", "asd_b"]].rename(columns={"asd": "_bl", "asd_b": "_bb"})
        m = it.merge(b, on=on + ["item"], how="left")
        return m.assign(gain=m.asd - m._bl, gain_b=m.asd_b - m._bb).drop(columns=["_bl", "_bb"])

    for name, df in variants.items():
        it_a = gains(both_units(df, ["mechanism", "debater"]), blind_pool, [])     # agents: pooled over judges
        it_j = gains(both_units(df, ["mechanism", "judge"]), blind_j, ["judge"])  # judges: pooled over agents
        it_aj = gains(both_units(df, ["mechanism", "debater", "judge"]), blind_j, ["judge"])
        out[f"agents_{name}"] = summarize(it_a, ["mechanism", "debater"]).assign(aa=lambda d: d.debater.map(a_idx))
        out[f"judges_{name}"] = summarize(it_j, ["mechanism", "judge"]).assign(aa=lambda d: d.judge.map(j_idx))
        out[f"cells_{name}"] = (it_aj.groupby(["mechanism", "debater", "judge"])
                                [["asd", "asd_p", "asd_b", "conc", "gain", "gain_b"]].mean().reset_index())
        sl = []
        for m in MECHS:
            for col in ("asd", "asd_b", "gain_b", "conc"):
                sl.append({"variant": name, "mechanism": m, "over": "agents", "measure": col,
                           **slope(it_a[it_a.mechanism == m], "debater", a_idx, col)})
                sl.append({"variant": name, "mechanism": m, "over": "judges", "measure": col,
                           **slope(it_j[it_j.mechanism == m], "judge", j_idx, col)})
        out[f"slopes_{name}"] = pd.DataFrame(sl)
        out.setdefault("_items", {})[name] = {"agents": it_a, "judges": it_j}
    out["_items"]["blind"] = blind_j
    # agent accuracy alone and liar commitment
    s = rows[rows.mechanism == "solo"]
    out["solo"] = s.drop_duplicates("episode").groupby("debater").correct.mean().rename("solo_acc").reset_index()
    if verdicts is not None and len(verdicts):
        out["commit"] = (verdicts.drop_duplicates("key").groupby(["debater", "mechanism"]).verdict
                         .apply(lambda v: float((v == "committed").mean())).rename("committed").reset_index())
    return out


def fmt(r, col="asd") -> str:
    return f"{r[col]:+.2f} [{r[col + '_lo']:+.2f}, {r[col + '_hi']:+.2f}]"


def markdown(t: dict) -> str:
    v = "committed" if "agents_committed" in t else "all"
    L = ["# Model panel: EIC against agent and judge intelligence", "",
         f"Items not selected on any judge. ASD = mean reward arguing the truth minus lying, in log score (the "
         f"protocol's reward; also its median over items) and in Brier score of the judge's probability for the "
         f"agent's side (bounded, so judges of different confidence compare); conc = share of items where "
         f"honesty earns more. "
         f"Headline variant: {'committed liars only (gpt-5 audit)' if v == 'committed' else 'all prompted liars'}.", ""]
    for name in ("committed", "all"):
        if f"agents_{name}" not in t:
            continue
        L += [f"## Agents ({name} liars), pooled over judges", "",
              "| agent | AA index | protocol | ASD, log score [95% CI] | median, log | ASD, Brier [95% CI] | conc [95% CI] | gain over blind, Brier [95% CI] | solo acc | committed |", "|---|---|---|---|---|---|---|---|---|---|"]
        a = t[f"agents_{name}"].merge(t["solo"], on="debater", how="left")
        if "commit" in t:
            a = a.merge(t["commit"], on=["debater", "mechanism"], how="left")
        for _, r in a.sort_values(["mechanism", "aa"]).iterrows():
            cm = f"{r.committed:.2f}" if "committed" in r and pd.notna(r.get("committed")) else "-"
            sa = f"{r.solo_acc:.2f}" if pd.notna(r.get("solo_acc")) else "-"
            L.append(f"| {r.debater} | {r.aa:.1f} | {r.mechanism} | {fmt(r)} | {r.asd_med:+.2f} | {fmt(r, 'asd_b')} | "
                     f"{fmt(r, 'conc')} | {fmt(r, 'gain_b')} | {sa} | {cm} |")
        L += ["", f"## Judges ({name} liars), pooled over agents", "",
              "| judge | AA index | protocol | ASD, log score [95% CI] | median, log | ASD, Brier [95% CI] | conc [95% CI] | blind ASD (log / Brier) | gain over blind, log [95% CI] | gain over blind, Brier [95% CI] |", "|---|---|---|---|---|---|---|---|---|---|"]
        b = t["blind"].set_index("judge")
        for _, r in t[f"judges_{name}"].sort_values(["mechanism", "aa"]).iterrows():
            bl, bp = b.asd.get(r.judge, np.nan), b.asd_b.get(r.judge, np.nan)
            L.append(f"| {r.judge} | {r.aa:.1f} | {r.mechanism} | {fmt(r)} | {r.asd_med:+.2f} | {fmt(r, 'asd_b')} | "
                     f"{fmt(r, 'conc')} | {bl:+.2f} / {bp:+.2f} | {fmt(r, 'gain')} | {fmt(r, 'gain_b')} |")
        L += ["", f"## Slopes on the AA index ({name} liars; per 10 index points)", "",
              "Interval: two-level bootstrap (items, then models), so it includes the uncertainty from which "
              "models are on the panel. The items-only interval treats the models as fixed.", "",
              "| protocol | across | measure | slope [95% CI] | items-only CI | r | points |",
              "|---|---|---|---|---|---|---|"]
        for _, r in t[f"slopes_{name}"].iterrows():
            L.append(f"| {r.mechanism} | {r.over} | {r.measure} | {r.slope_per_10:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | "
                     f"[{r.lo_items:+.3f}, {r.hi_items:+.3f}] | {r.r:.2f} | {r.n} |")
        L.append("")
    return "\n".join(L) + "\n"


def place_labels(ax, xs, ys, names, *, fontsize=6.3, color=None, pad=5) -> None:
    """Label points to their right without overlaps: labels that would collide are pushed apart vertically (in
    display space) and joined to their point by a hairline."""
    fig = ax.figure
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, max(hi, float(np.nanmax(xs)) + 0.16 * (hi - lo)))  # room for the rightmost labels
    fig.canvas.draw()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    h = fontsize * fig.dpi / 72 * 1.25
    w = [len(n) * fontsize * fig.dpi / 72 * 0.56 for n in names]
    lab = [[px + pad * fig.dpi / 72, py] for px, py in pts]
    order = np.argsort(pts[:, 1])
    for _ in range(200):
        moved = False
        for a_i, i in enumerate(order):
            for j in order[a_i + 1:]:
                (xi, yi), (xj, yj) = lab[i], lab[j]
                if xi < xj + w[j] and xj < xi + w[i] and abs(yi - yj) < h:
                    push = (h - abs(yi - yj)) / 2 + 0.5
                    lo, hi = (i, j) if yi <= yj else (j, i)
                    lab[lo][1] -= push
                    lab[hi][1] += push
                    moved = True
        if not moved:
            break
    for x, y, (px, py), (lx, ly), n in zip(xs, ys, pts, lab, names):
        dx, dy = (lx - px) * 72 / fig.dpi, (ly - py) * 72 / fig.dpi
        far = abs(dy) > 0.6 * fontsize
        ax.annotate(n, (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=fontsize, color=color, va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=color, shrinkA=0, shrinkB=2) if far else None)




def short(name: str) -> str:
    """Compact point label: reasoning models run at low effort unless the label says otherwise."""
    n = name.replace("@low", "").replace("@medium", " (medium)").replace("@high", " (high)").replace("@none", " (none)")
    return n


def _band(ax, reg: dict, color, *, label=None, alpha=0.14, ls="-"):
    """Regression line with its two-level bootstrap 95% band (the shaded region)."""
    ax.fill_between(reg["grid"], reg["band_lo"], reg["band_hi"], color=color, alpha=alpha, lw=0, zorder=1)
    ax.plot(reg["grid"], reg["fit"], color=color, lw=1.8, ls=ls, zorder=2, label=label)


def _points(ax, g, meas, color, *, label=None, size=5.5):
    """Each model's estimate with its 95% item-bootstrap interval (thin vertical bar)."""
    ax.errorbar(g.aa, g[meas], yerr=[g[meas] - g[f"{meas}_lo"], g[f"{meas}_hi"] - g[meas]], fmt="o", color=color,
                ms=size, mec="white", mew=0.8, elinewidth=0.9, capsize=0, zorder=4, label=label, alpha=0.95)


def _slope_note(ax, lines):
    ax.text(0.99, 0.02, "\n".join(lines), transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8,
            color="#52514e", linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec="#e1e0d9", lw=0.6))


def figures(t: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import eic_style as st
    from eic_style import BLUE, CONTEXT, INK2, MUTED, ORANGE

    v = "committed" if "agents_committed" in t else "all"
    items = t["_items"]
    GRAY = "#8f8d86"
    units = {"asd_b": "ASD, Brier score", "asd": "ASD, log score", "gain_b": "gain over the blind judge (Brier ASD)"}

    def fmt_s(reg):
        return f"{reg['slope_per_10']:+.3f} [{reg['lo']:+.3f}, {reg['hi']:+.3f}]"

    for paper in (False, True):
        ctx = st.paper_mode() if paper else _null()
        with ctx:
            bp = t["blind_pooled"].iloc[0]
            # ---- agents: one figure per unit (Brier = headline, log score = appendix)
            for meas, fname in (("asd_b", "fig12_panel_agents"), ("asd", "fig12b_panel_agents_log")):
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
                for ax, m in zip(axes, MECHS):
                    st.axes(ax)
                    ax.axhline(0, color=INK2, lw=0.9, zorder=1)
                    ax.axhline(bp[meas], color=GRAY, lw=1.1, ls=(0, (4, 3)), zorder=1)
                    ax.text(0.01, bp[meas], " judges with no argument", color=INK2, fontsize=6.8, va="bottom", ha="left",
                            transform=ax.get_yaxis_transform())
                    notes = ["slope per 10 index points [95% CI]"]
                    if v == "committed":
                        ra = regression(items["all"]["agents"].query("mechanism == @m"), "debater", t["a_idx"], meas)
                        _band(ax, ra, GRAY, alpha=0.10, ls=(0, (1.5, 1.5)))
                        notes.append(f"all prompted liars: {fmt_s(ra)}")
                    rc = regression(items[v]["agents"].query("mechanism == @m"), "debater", t["a_idx"], meas)
                    _band(ax, rc, BLUE)
                    notes.insert(1, f"{'committed liars' if v == 'committed' else 'all liars'}: {fmt_s(rc)}")
                    g = t[f"agents_{v}"].query("mechanism == @m").sort_values("aa")
                    lad = g[g.debater.str.startswith("gpt-5.6-sol")]
                    rest = g[~g.debater.str.startswith("gpt-5.6-sol")]
                    _points(ax, rest, meas, BLUE)
                    ax.plot(lad.aa, lad[meas], color=ORANGE, lw=1.4, zorder=3)
                    _points(ax, lad, meas, ORANGE)
                    ax.set_title(MECHS[m])
                    ax.set_xlabel("agent: Artificial Analysis Intelligence Index (v4.3)")
                    _slope_note(ax, notes)
                axes[0].set_ylabel(units[meas] + "\n(reward arguing the truth minus lying)")
                handles = [Line2D([], [], color=BLUE, marker="o", mec="white", lw=1.8,
                                  label="agent (committed liars), with fit and 95% band" if v == "committed"
                                  else "agent, with fit and 95% band"),
                           Line2D([], [], color=ORANGE, marker="o", mec="white", lw=1.4,
                                  label="GPT-5.6 Sol, effort none → high")]
                if v == "committed":
                    handles.append(Line2D([], [], color=GRAY, lw=1.8, ls=(0, (1.5, 1.5)),
                                          label="fit with all prompted liars (incl. conceding / refusing)"))
                handles.append(Line2D([], [], color=GRAY, lw=1.1, ls=(0, (4, 3)), label="judges with no argument"))
                fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.995 if paper else 0.945),
                           ncol=4, fontsize=7, handlelength=2.6, columnspacing=1.6)
                fig.suptitle("Across agents, the incentive to be honest barely changes with intelligence"
                             if v == "committed" else "EIC against agent intelligence",
                             x=0.005, ha="left", fontsize=10.5, fontweight="bold", y=1.0)
                fig.text(0.005, 0.965, "Each point: one agent, pooled over 9 judges and 300 random testbed items, "
                         "95% item-bootstrap interval. Line and shaded band: least-squares fit and its 95% "
                         "bootstrap band (resampling items and agents). Reasoning models at low effort unless noted.",
                         fontsize=7.3, color=INK2, va="top")
                fig.tight_layout(rect=(0, 0, 1, 0.95 if not paper else 0.94))
                for ax, m in zip(axes, MECHS):
                    g = t[f"agents_{v}"].query("mechanism == @m")
                    names = [("effort: " + n.split("@")[1]) if n.startswith("gpt-5.6-sol") else short(n)
                             for n in g.debater]
                    place_labels(ax, g.aa.to_numpy(), g[meas].to_numpy(), names, color=INK2, fontsize=6.2)
                st.save(fig, out, fname)

            # ---- judges: raw ASD next to the blind baseline, then the gain over it
            fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), sharex=True, sharey="row")
            bl = t["blind"].sort_values("aa")
            rb = regression(items["blind"], "judge", t["j_idx"], "asd_b")
            for col, m in enumerate(MECHS):
                ax = axes[0, col]
                st.axes(ax)
                ax.axhline(0, color=INK2, lw=0.9, zorder=1)
                _band(ax, rb, GRAY, alpha=0.12, ls=(0, (4, 3)))
                _points(ax, bl, "asd_b", GRAY, size=5)
                rj = regression(items[v]["judges"].query("mechanism == @m"), "judge", t["j_idx"], "asd_b")
                _band(ax, rj, BLUE)
                g = t[f"judges_{v}"].query("mechanism == @m").sort_values("aa")
                _points(ax, g, "asd_b", BLUE)
                ax.set_title(MECHS[m])
                _slope_note(ax, ["slope per 10 index points [95% CI]", f"with the protocol: {fmt_s(rj)}",
                                 f"no argument: {fmt_s(rb)}"])
                ax2 = axes[1, col]
                st.axes(ax2)
                ax2.axhline(0, color=INK2, lw=0.9, zorder=1)
                rg = regression(items[v]["judges"].query("mechanism == @m"), "judge", t["j_idx"], "gain_b")
                _band(ax2, rg, BLUE)
                _points(ax2, g, "gain_b", BLUE)
                ax2.set_xlabel("judge: Artificial Analysis Intelligence Index (v4.3)")
                _slope_note(ax2, ["slope per 10 index points [95% CI]", fmt_s(rg)])
            axes[0, 0].set_ylabel("ASD, Brier score\n(reward arguing the truth minus lying)")
            axes[1, 0].set_ylabel("gain over the same judge\nwith no argument (Brier ASD)")
            handles = [Line2D([], [], color=BLUE, marker="o", mec="white", lw=1.8,
                              label="judge with the protocol" + (" (committed liars)" if v == "committed" else "")),
                       Line2D([], [], color=GRAY, marker="o", mec="white", lw=1.8, ls=(0, (4, 3)),
                              label="same judge, no argument (blind baseline)"),
                       Patch(color=BLUE, alpha=0.14, label="95% bootstrap band of the fit (items and judges)")]
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.995 if paper else 0.962), ncol=3,
                       fontsize=7, handlelength=2.6, columnspacing=1.6)
            fig.suptitle("Stronger judges reward honesty more, but no more than they do with no argument at all",
                         x=0.005, ha="left", fontsize=10.5, fontweight="bold", y=1.0)
            fig.text(0.005, 0.975, "Each point: one judge, pooled over 20 agent configurations and 300 random "
                     "testbed items, 95% item-bootstrap interval. Bottom row: the protocol's ASD minus the same "
                     "judge's ASD with no argument, item by item.", fontsize=7.3, color=INK2, va="top")
            fig.tight_layout(rect=(0, 0, 1, 0.945 if not paper else 0.965))
            for col, m in enumerate(MECHS):
                g = t[f"judges_{v}"].query("mechanism == @m")
                for row, meas in ((0, "asd_b"), (1, "gain_b")):
                    place_labels(axes[row, col], g.aa.to_numpy(), g[meas].to_numpy(), list(g.judge), color=INK2,
                                 fontsize=6.2)
            st.save(fig, out, "fig13_panel_judges")

            # heatmap: ASD in probability units (bounded, comparable across judges)
            cells = t[f"cells_{v}"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
            a_order = sorted(t["a_idx"], key=lambda n: t["a_idx"][n])
            j_order = sorted(t["j_idx"], key=lambda n: t["j_idx"][n])
            lim = float(np.nanquantile(np.abs(cells.asd_b), 0.95)) if len(cells) else 0.5
            for ax, m in zip(axes, MECHS):
                p = cells[cells.mechanism == m].pivot_table(index="debater", columns="judge", values="asd_b")
                p = p.reindex(index=[x for x in a_order if x in p.index], columns=[x for x in j_order if x in p.columns])
                im = ax.imshow(p.to_numpy(), cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
                ax.set_xticks(range(p.shape[1]), p.columns, rotation=60, ha="right", fontsize=6.5)
                ax.set_yticks(range(p.shape[0]), p.index, fontsize=6.5)
                ax.grid(False)
                ax.set_title(MECHS[m])
            fig.colorbar(im, ax=axes, shrink=0.8, label="ASD, Brier score")
            fig.suptitle("ASD by agent (rows, by AA index) and judge (columns, by AA index)", x=0.005, ha="left",
                         fontsize=10, fontweight="bold")
            st.save(fig, out, "fig14_panel_heatmap")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    t = tables(Path(a.run))
    out = Path(a.out) / "panel"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in t.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"{k}.csv", index=False)
    md = markdown(t)
    (out / "panel.md").write_text(md)
    print(md)
    if not a.no_figures:
        figures(t, Path(a.out) / "eic")


if __name__ == "__main__":
    main()
