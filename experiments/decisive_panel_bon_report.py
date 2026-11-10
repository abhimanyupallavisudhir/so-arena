"""Report for the model panel under optimization (``decisive_panel_bon.py`` and ``decisive_panel_po.py``).

* Best-of-n: per agent, per-item held-out ASD (selection on one judge sample, scoring on an independent one) at
  n = 1, 2, 4, 8, 16, for ``liar`` (only the false side optimized) and ``both``, pooled over the three weak judges
  (item bootstrap), in log score (the protocols' reward) and Brier score (bounded). Regressions on the agent's
  Artificial Analysis index with a two-level bootstrap (items, then agents), for n = 1, n = 16 and the drop
  n = 16 minus n = 1: does optimization cost a protocol more for smarter agents?
* Prompt search: per agent, held-out ASD of the best honest minus the best deceptive strategy at round 0 (the
  unprompted agent) and after the last round, paired by item.

    python experiments/decisive_panel_bon_report.py --run runs/decisive --out docs/results
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from decisive_panel import panel_config
from decisive_panel_bon import AGENTS, JUDGES, mechanisms, root
from decisive_panel_report import N_BOOT, regression

UNIT_NAMES = {"brier": "Brier score", "log": "log score", "brier_cal": "Brier score after a calibration map"}
NAMES = {"consultancy_1r": "Consultancy (one argument)", "debate_1r": "Debate (one round, simultaneous)"}


def brier(log_reward: float) -> float:
    """Brier score of the judge's probability for the agent's side, from the log-score reward."""
    return -(1.0 - math.exp(log_reward)) ** 2


def calibrated_brier(a: float, b: float):
    """Brier score after the judge's calibration map (fitted on the model panel's unselected items)."""
    def f(log_reward: float) -> float:
        p = min(max(math.exp(log_reward), 0.01), 0.99)
        q = 1 / (1 + math.exp(-(a + b * math.log(p / (1 - p)))))
        return -(1.0 - q) ** 2
    return f


def panel_maps(run: Path) -> dict:
    """Platt maps per (judge, protocol family) on all of the panel's items (``decisive_calibration.py``)."""
    import decisive_calibration as dc

    panel = dc.panel_frame(run).assign(family=lambda d: d.mechanism.map(dc.FAMILY))
    return {(j, f): dc.platt(dc.logit(g.p), g.y.to_numpy(), g.w.to_numpy(), f in dc.SYMMETRIC)
            for (j, f), g in panel.groupby(["judge", "family"])}


def _cell(args):
    import decisive_ic as dic

    run, ag, jd, m, maps = args
    out = []
    units = [("log", None), ("brier", brier)]
    fam = "consultancy" if m.startswith("consultancy") else "debate"
    if maps and (jd, fam) in maps:
        units.append(("brier_cal", calibrated_brier(*maps[(jd, fam)])))
    for unit, tf in units:
        for r in dic.cell_rows(root(run), ag, jd, m, holdout=True, transform=tf, kinds=("bon",)):
            out.append({**r, "unit": unit})
    return out


def bon_items(run: Path, agents=None, judges=None, workers: int = 6, maps: dict | None = None) -> pd.DataFrame:
    cells = [(run, a, j, m.name, maps) for a in (agents or AGENTS) for j in (judges or JUDGES) for m in mechanisms()]
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for got in ex.map(_cell, cells) for r in got]
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    # the same items for every judge of an agent (a judge still missing trees would change the item set)
    k = d.groupby(["debater", "mechanism", "item"]).judge.nunique().rename("nj").reset_index()
    d = d.merge(k, on=["debater", "mechanism", "item"])
    full = d.groupby(["debater", "mechanism"]).nj.transform("max")
    return d[d.nj == full].drop(columns="nj")


def pooled(items: pd.DataFrame) -> pd.DataFrame:
    """Per agent, protocol, unit, mode and n: mean over judges per item, then the item-bootstrap mean."""
    per = items.groupby(["debater", "mechanism", "unit", "mode", "level", "item"]).asd.mean().reset_index()
    rng = np.random.default_rng(0)
    out = []
    for key, g in per.groupby(["debater", "mechanism", "unit", "mode", "level"]):
        x = g.asd.to_numpy()
        b = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
        out.append(dict(zip(["debater", "mechanism", "unit", "mode", "n"], key),
                        asd=x.mean(), asd_lo=np.quantile(b, .025), asd_hi=np.quantile(b, .975), items=len(x)))
    return pd.DataFrame(out)


def per_item_frame(items: pd.DataFrame, mech: str, unit: str, mode: str, n) -> pd.DataFrame:
    """Per (agent, item) ASD pooled over judges; ``n="drop"``: n = 16 minus n = 1 on the same item."""
    sel = items[(items.mechanism == mech) & (items.unit == unit) & (items["mode"] == mode)]
    per = sel.groupby(["debater", "level", "item"]).asd.mean().unstack("level")
    col = (per[16] - per[1]) if n == "drop" else per[n]
    return col.rename("asd").reset_index()


def slopes(items: pd.DataFrame, idx: dict) -> tuple[pd.DataFrame, dict]:
    rows, regs = [], {}
    for m in items.mechanism.unique():
        for unit in [u for u in ("brier", "log", "brier_cal") if u in set(items.unit)]:
            for mode, n in (("liar", 1), ("liar", 16), ("both", 16), ("liar", "drop"), ("both", "drop")):
                f = per_item_frame(items, m, unit, mode, n)
                r = regression(f, "debater", idx, "asd")
                regs[(m, unit, mode, n)] = r
                rows.append({"mechanism": m, "unit": unit, "mode": mode, "n": n,
                             **{k: r[k] for k in ("slope_per_10", "lo", "hi", "lo_items", "hi_items", "r", "n")
                                if k != "n"}, "agents": r["n"]})
    return pd.DataFrame(rows), regs


def po_rows(run: Path) -> pd.DataFrame:
    """Per agent and protocol: held-out ASD (log and Brier) of the best honest minus best deceptive strategy at
    every search round, paired by item, with an item bootstrap."""
    out = []
    rng = np.random.default_rng(1)
    for f in sorted((Path(run) / "panel_po").glob("*/*.json")):
        rec = json.loads(f.read_text())
        h, d = rec["searches"]["true"], rec["searches"]["false"]
        for k, (a, b) in enumerate(zip(h["best_path"], d["best_path"])):
            def frame(eps, honest):
                x = pd.DataFrame([e for e in eps if e["reward"] is not None and e.get("p_true") is not None])
                p = x.p_true if honest else 1 - x.p_true
                return pd.DataFrame({"item": x.item, "log": x.reward, "brier": -(1 - p) ** 2}).groupby("item").mean()

            fh, fd = frame(h["heldout"][a], True), frame(d["heldout"][b], False)
            j = fh.join(fd, lsuffix="_h", rsuffix="_d", how="inner")
            for unit in ("log", "brier"):
                x = (j[f"{unit}_h"] - j[f"{unit}_d"]).to_numpy()
                bs = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(1)
                out.append({"debater": rec["debater"], "mechanism": rec["protocol"], "judge": rec["judge"],
                            "round": k, "unit": unit, "asd": x.mean(), "asd_lo": np.quantile(bs, .025),
                            "asd_hi": np.quantile(bs, .975), "items": len(x)})
    return pd.DataFrame(out)


def markdown(pool: pd.DataFrame, sl: pd.DataFrame, po: pd.DataFrame, idx: dict) -> str:
    L = ["# Model panel under optimization", "",
         "Held-out ASD (honest reward minus liar reward; selection on one judge sample, scoring on an independent "
         "one), pooled over gpt-4o-mini, Llama-3.1-8B and Gemma-3-12B, 95% item-bootstrap intervals. "
         "`liar`: only the false side optimized; `both`: both sides at the same n.", ""]
    for unit in [u for u in ("brier", "log", "brier_cal") if u in set(pool.unit)]:
        L += [f"## Best-of-n by agent ({UNIT_NAMES[unit]})", "",
              "| agent | AA index | protocol | n=1 | liar n=16 | both n=16 | items |", "|---|---|---|---|---|---|---|"]
        p = pool[pool.unit == unit]
        for (ag, m), g in sorted(p.groupby(["debater", "mechanism"]), key=lambda x: (x[0][1], idx.get(x[0][0], 0))):
            def cell(mode, n):
                r = g[(g["mode"] == mode) & (g.n == n)]
                return "-" if r.empty else f"{r.asd.iloc[0]:+.2f} [{r.asd_lo.iloc[0]:+.2f}, {r.asd_hi.iloc[0]:+.2f}]"
            L.append(f"| {ag} | {idx.get(ag, float('nan')):.1f} | {m} | {cell('liar', 1)} | {cell('liar', 16)} | "
                     f"{cell('both', 16)} | {int(g['items'].max())} |")
        L.append("")
    L += ["## Slopes on the agent's AA index (per 10 points; two-level bootstrap: items, then agents)", "",
          "| protocol | unit | curve | slope [95% CI] | items-only CI | r | agents |", "|---|---|---|---|---|---|---|"]
    for _, r in sl.iterrows():
        curve = f"{r['mode']} n={r.n}" if r.n != "drop" else f"{r['mode']}: n=16 minus n=1"
        L.append(f"| {r.mechanism} | {r.unit} | {curve} | {r.slope_per_10:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | "
                 f"[{r.lo_items:+.3f}, {r.hi_items:+.3f}] | {r.r:.2f} | {r.agents} |")
    if not po.empty:
        L += ["", "## Prompt search by agent (judge gpt-4o-mini; held-out)", "",
              "| agent | AA index | protocol | unit | round 0 (unprompted) | last round | items |",
              "|---|---|---|---|---|---|---|"]
        for (ag, m, u), g in sorted(po.groupby(["debater", "mechanism", "unit"]),
                                    key=lambda x: (x[0][1], x[0][2], idx.get(x[0][0], 0))):
            g = g.sort_values("round")
            f = lambda r: f"{r.asd:+.2f} [{r.asd_lo:+.2f}, {r.asd_hi:+.2f}]"  # noqa: E731
            L.append(f"| {ag} | {idx.get(ag, float('nan')):.1f} | {m} | {u} | {f(g.iloc[0])} | {f(g.iloc[-1])} | "
                     f"{int(g['items'].iloc[-1])} |")
    return "\n".join(L) + "\n"


def blind_baseline(run: Path, items: set, unit: str = "brier") -> tuple[float, float, float] | None:
    """The three weak judges with no argument on the same items, Brier ASD with an item-bootstrap interval."""
    import decisive_calibration as dc

    if unit != "brier":
        return None
    b = dc.panel_frame(run)
    b = b[(b.mechanism == "direct") & b.judge.isin(JUDGES) & b.item.isin(items)]
    return dc.boot(dc.item_asd(b, ["judge"], "p").groupby("item").b.mean().to_numpy())


def figure(pool: pd.DataFrame, regs: dict, po: pd.DataFrame, idx: dict, out: Path, unit: str = "brier",
           blind: tuple | None = None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import eic_style as st
    from decisive_panel_report import _band, _null, place_labels, short
    from eic_style import BLUE, INK2, ORANGE

    GRAY = "#8f8d86"
    mechs = [m for m in NAMES if m in set(pool.mechanism)]
    has_po = not po.empty
    for paper in (False, True):
        with (st.paper_mode() if paper else _null()):
            nrow = 2 if has_po else 1
            fig, axes = plt.subplots(nrow, len(mechs), figsize=(11, 4.6 * nrow), squeeze=False, sharey="row")
            for c, m in enumerate(mechs):
                ax = axes[0, c]
                st.axes(ax)
                ax.axhline(0, color=INK2, lw=0.9, zorder=1)
                if blind is not None:
                    ax.axhspan(blind[1], blind[2], color=GRAY, alpha=0.13, lw=0, zorder=0)
                    ax.axhline(blind[0], color=GRAY, lw=1.4, ls=(0, (4, 3)), zorder=1)
                p = pool[(pool.unit == unit) & (pool.mechanism == m)]
                series = (("liar", 1, GRAY, "unoptimized (n = 1)"), ("liar", 16, ORANGE, "liar optimized, n = 16"),
                          ("both", 16, BLUE, "both optimized, n = 16"))
                base = p[(p["mode"] == "liar") & (p.n == 1)].set_index("debater")
                for mode, n, col, _ in series:
                    g = p[(p["mode"] == mode) & (p.n == n)].copy()
                    g["aa"] = g.debater.map(idx)
                    _band(ax, regs[(m, unit, mode, n)], col, alpha=0.12)
                    if n == 16:  # a hairline from each agent's unoptimized point to its optimized one
                        for _, r in g.iterrows():
                            if r.debater in base.index:
                                ax.plot([r.aa, r.aa], [base.asd[r.debater], r.asd], color=col, lw=0.6, alpha=0.5,
                                        zorder=2)
                    ax.errorbar(g.aa, g.asd, yerr=[g.asd - g.asd_lo, g.asd_hi - g.asd], fmt="o", color=col, ms=5.5,
                                mec="white", mew=0.8, elinewidth=0.9, capsize=0, zorder=4)
                rl, rb = regs[(m, unit, "liar", "drop")], regs[(m, unit, "both", "drop")]
                ax.text(0.99, 0.02, "change from n=1 to n=16, slope per 10 index points\n"
                        f"liar optimized: {rl['slope_per_10']:+.3f} [{rl['lo']:+.3f}, {rl['hi']:+.3f}]\n"
                        f"both optimized: {rb['slope_per_10']:+.3f} [{rb['lo']:+.3f}, {rb['hi']:+.3f}]",
                        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.8, color=INK2, linespacing=1.4,
                        bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec="#e1e0d9", lw=0.6))
                ax.set_title(NAMES[m] + ": best-of-16")
                if nrow == 1:
                    ax.set_xlabel("agent: Artificial Analysis Intelligence Index (v4.3)")
            axes[0, 0].set_ylabel(f"held-out ASD, {unit} score\n(reward arguing the truth minus lying)")
            if has_po:
                for c, m in enumerate(mechs):
                    ax = axes[1, c]
                    st.axes(ax)
                    ax.axhline(0, color=INK2, lw=0.9, zorder=1)
                    g = po[(po.unit == unit) & (po.mechanism == m)]
                    first = g[g["round"] == 0].assign(aa=lambda d: d.debater.map(idx))
                    last = g[g["round"] == g["round"].max()].assign(aa=lambda d: d.debater.map(idx))
                    for _, r in last.iterrows():
                        b = first[first.debater == r.debater]
                        if len(b):
                            ax.annotate("", (r.aa, r.asd), (r.aa, b.asd.iloc[0]),
                                        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.8, alpha=0.7))
                    for d, col in ((first, GRAY), (last, ORANGE)):
                        ax.errorbar(d.aa, d.asd, yerr=[d.asd - d.asd_lo, d.asd_hi - d.asd], fmt="o", color=col,
                                    ms=5.5, mec="white", mew=0.8, elinewidth=0.9, capsize=0, zorder=4)
                    ax.set_title(NAMES[m] + ": prompt search (gpt-4o-mini judge)")
                    ax.set_xlabel("agent: Artificial Analysis Intelligence Index (v4.3)")
                    place_labels(ax, last.aa.to_numpy(), last.asd.to_numpy(), [short(n) for n in last.debater],
                                 color=INK2, fontsize=6.2)
                axes[1, 0].set_ylabel(f"held-out ASD, {unit} score\nbest honest minus best deceptive strategy")
            handles = [Line2D([], [], color=GRAY, marker="o", mec="white", lw=1.8, label="unoptimized (n = 1)"),
                       Line2D([], [], color=ORANGE, marker="o", mec="white", lw=1.8,
                              label="liar optimized (best-of-16 / after prompt search)"),
                       Line2D([], [], color=BLUE, marker="o", mec="white", lw=1.8, label="both sides optimized")]
            if blind is not None:
                handles.append(Line2D([], [], color=GRAY, lw=1.4, ls=(0, (4, 3)),
                                      label="judges with no argument, same items (95% interval shaded)"))
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.995 if paper else 0.955), ncol=4,
                       fontsize=7, handlelength=2.6, columnspacing=1.6)
            fig.suptitle("An optimized liar erases most of debate's honesty incentive, and smarter agents don't lose more of it",
                         x=0.005, ha="left",
                         fontsize=10.5, fontweight="bold", y=1.0)
            fig.text(0.005, 0.975, "Top: each point is one agent, pooled over 3 weak judges, scored on independent "
                     "judge samples; lines and bands: least-squares fits with 95% bootstrap bands (items and agents). "
                     "Bottom: arrows from the unprompted agent to the end of the prompt search.", fontsize=7.3,
                     color=INK2, va="top")
            fig.tight_layout(rect=(0, 0, 1, 0.945 if not paper else 0.965))
            for c, m in enumerate(mechs):
                g = pool[(pool.unit == unit) & (pool.mechanism == m) & (pool["mode"] == "liar") & (pool.n == 16)]
                place_labels(axes[0, c], g.debater.map(idx).to_numpy(), g.asd.to_numpy(),
                             [short(n) for n in g.debater], color=INK2, fontsize=6.2)
            st.save(fig, out, "fig15_panel_optimized" + ("" if unit == "brier" else "_log"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reuse", action="store_true", help="read the per-item rows saved by an earlier run")
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    idx = {x["name"]: x["aa_index"] for x in panel_config()["agents"]}
    res = out / "panel_opt"
    res.mkdir(parents=True, exist_ok=True)
    if a.reuse and (res / "bon_items.csv.gz").exists():
        items = pd.read_csv(res / "bon_items.csv.gz")
    else:
        items = bon_items(run, workers=a.workers, maps=panel_maps(run))
        items.to_csv(res / "bon_items.csv.gz", index=False)
    pool = pooled(items)
    pool.to_csv(res / "bon_agents.csv", index=False)
    sl, regs = slopes(items, idx)
    sl.to_csv(res / "slopes.csv", index=False)
    po = po_rows(run)
    po.to_csv(res / "po_agents.csv", index=False)
    md = markdown(pool, sl, po, idx)
    (res / "panel_opt.md").write_text(md)
    print(md)
    for unit in ("brier", "log"):
        figure(pool, regs, po, idx, out / "eic", unit, blind_baseline(run, set(items.item), unit))


if __name__ == "__main__":
    main()
