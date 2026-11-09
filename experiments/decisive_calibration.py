"""Does the judge's scale explain the results? ASD after a simple calibration map.

Debate may lower a judge's confidence without lowering what it knows. A Brier or log-score ASD then penalizes debate
against the judge with no argument, even at equal accuracy. To separate the two, each judge's stated probability is
passed through a Platt map, logit p' = a + b * logit p (b >= 0), and every ASD is recomputed from p'.

* ``p`` is the judge's probability for the side the agent argued, ``y`` is 1 if that side is true.
* A map is fitted per judge and condition (protocol, or the judge with no argument), so each condition gets its own
  best scale. In debate and for the judge alone the two sides are mirror images, so the map is symmetric (a = 0). In
  consultancy the intercept can discount the side the judge has just heard argued.
* Maps are cross-fitted over items (5 folds): no item is scored by a map fitted on it.
* b is floored at 0, so a map can erase a judge's signal but cannot invert it.

The gap-selected experiments need a second variant. Those items were chosen because the judges fail alone, so a map
fitted on them learns the selection. ``transfer`` fits the map on the panel's unselected items instead, per judge and
protocol family, and applies it unchanged.

Stages: the model panel (300 unselected items), the protocol grid (927 gap-selected), prompt optimization (480 held
out), and self-play (390 held out). Recursive and prover-estimator debate are left out, because their rewards are not
a score of the judge's probability for the agent's answer.

    python experiments/decisive_calibration.py --run runs/decisive --out docs/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from decisive_panel_report import config as panel_config
from decisive_report import _found

EPS = 0.01  # stated probabilities of exactly 0 or 1 are read as 0.01 and 0.99 by the map
LOG_EPS = 1e-4  # the protocols' own log-score clip
B_MAX = 8.0
FOLDS = 5
N_BOOT = 2000
FAMILY = {"debate_sim": "debate", "debate_seq": "debate", "debate": "debate", "consultancy": "consultancy",
          "consultancy_1r": "consultancy", "direct": "direct"}
SYMMETRIC = {"debate", "direct"}
NAMES = {"debate_sim": "debate, simultaneous", "debate_seq": "debate, sequential", "consultancy": "consultancy",
         "consultancy_1r": "consultancy (one argument)", "direct": "judge, no argument"}


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def platt(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, symmetric: bool = False) -> tuple[float, float]:
    """Weighted logistic fit of y on logit-probability ``x``; returns (a, b), with 0 <= b <= B_MAX and a = 0 if
    ``symmetric``."""
    w = np.ones_like(x, dtype=float) if w is None else np.asarray(w, dtype=float)
    w = w / w.sum()

    def loss(t):
        a, b = (0.0, t[0]) if symmetric else (t[0], t[1])
        z = a + b * x
        return float(np.sum(w * (np.logaddexp(0, z) - y * z)))

    start, bounds = ([1.0], [(0, B_MAX)]) if symmetric else ([0.0, 1.0], [(-5, 5), (0, B_MAX)])
    t = minimize(loss, start, method="L-BFGS-B", bounds=bounds).x
    return (0.0, float(t[0])) if symmetric else (float(t[0]), float(t[1]))


def folds_for(items, seed: int = 0, k: int = FOLDS) -> dict:
    names = np.array(sorted(set(items)))
    order = np.random.default_rng(seed).permutation(len(names))
    return dict(zip(names[order], np.arange(len(names)) % k))


def crossfit(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Add ``pc``: each row's calibrated probability, from a map fitted on the other folds' items of its group."""
    df = df.copy()
    df["pc"] = np.nan
    fold = df.item.map(folds_for(df.item))
    for _, g in df.groupby(by):
        sym = FAMILY[g.mechanism.iloc[0]] in SYMMETRIC
        for f in range(FOLDS):
            te = g.index[fold[g.index] == f]
            tr = g.index[fold[g.index] != f]
            if len(te) == 0 or len(tr) == 0:
                continue
            a, b = platt(logit(g.loc[tr, "p"]), g.loc[tr, "y"].to_numpy(), g.loc[tr, "w"].to_numpy(), sym)
            df.loc[te, "pc"] = expit(a + b * logit(g.loc[te, "p"]))
    return df


def maps_for(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Whole-sample maps per group, with the judge's confidence and accuracy."""
    out = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        a, b = platt(logit(g.p), g.y.to_numpy(), g.w.to_numpy(), FAMILY[g.mechanism.iloc[0]] in SYMMETRIC)
        honest = g[g.y == 1]
        out.append({**dict(zip(by, key)), "a": a, "b": b, "confidence": float(np.abs(g.p - .5).mean()),
                    "accuracy": float(((honest.p > .5) + .5 * (honest.p == .5)).mean()), "rows": len(g)})
    return pd.DataFrame(out)


def item_asd(df: pd.DataFrame, keys: list[str], col: str) -> pd.DataFrame:
    """Per (keys, item): reward arguing the truth minus lying, Brier (``b``) and log score (``l``). Rows are weighted
    by ``w`` (the equilibrium mixture in self-play)."""
    x = df[keys + ["item", "y", "w"]].copy()
    x["b"] = -(1 - df[col]) ** 2 * df.w
    x["l"] = np.log(np.clip(df[col], LOG_EPS, 1)) * df.w
    g = x.groupby(keys + ["item", "y"])[["b", "l", "w"]].sum()
    g["b"] /= g.w
    g["l"] /= g.w
    u = g[["b", "l"]].unstack("y").dropna()
    return pd.DataFrame({"b": u[("b", 1.0)] - u[("b", 0.0)], "l": u[("l", 1.0)] - u[("l", 0.0)]}).reset_index()


def boot(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def rows_frame(df: pd.DataFrame, **extra) -> pd.DataFrame:
    """Standard columns from rows with ``value`` and ``p_true``: p is the judge's probability for the agent's side."""
    x = df[df.value.notna() & df.p_true.notna()].copy()
    x["y"] = (x.value > 0).astype(float)
    x["p"] = np.where(x.y == 1, x.p_true, 1 - x.p_true)
    x["w"] = 1.0
    return x.assign(**extra)


def panel_frame(run: Path) -> pd.DataFrame:
    r = pd.read_csv(_found(run / "panel" / "rows.csv"), low_memory=False)
    return rows_frame(r[r.mechanism.isin(["debate_sim", "consultancy_1r", "direct"])], condition="all")


def panel_committed_frame(run: Path) -> pd.DataFrame:
    """The panel with samples whose liar did not commit dropped (gpt-5 audit); the judge alone is unaffected."""
    from decisive_panel_report import committed, load

    rows, verdicts = load(run)
    c = committed(rows[rows.mechanism.isin(["debate_sim", "consultancy_1r", "direct"])], verdicts)
    return rows_frame(c, condition="all")


def grid_frame(run: Path) -> pd.DataFrame:
    r = pd.read_csv(_found(run / "protocols" / "rows.csv"), low_memory=False)
    x = r[(r.mechanism == "direct") | ((r.debater == "gpt-5-mini") & r.mechanism.isin(["consultancy", "debate_seq", "debate_sim"]))]
    return rows_frame(x, condition="all")


def po_frame(run: Path) -> pd.DataFrame:
    """Held-out rows of the best honest and best deceptive strategy after the first (seed) and last search round."""
    from decisive_eic import po_records

    out = []
    for rec in po_records(run):
        if rec["protocol"] not in FAMILY:
            continue
        h, d = rec["searches"]["true"], rec["searches"]["false"]
        last = len(h["best_path"]) - 1
        for label, k in (("before search", 0), ("after search", last)):
            for s, cid in ((h, h["best_path"][k]), (d, d["best_path"][k])):
                e = pd.DataFrame(s["heldout"][cid])
                if "error" in e:
                    e = e[~e.error.fillna(False).astype(bool)]
                e = e[e.reward.notna() & e.p_true.notna()]
                out.append(pd.DataFrame({"judge": rec["judge"], "mechanism": rec["protocol"], "item": e["item"].to_numpy(),
                                         "value": e.value.to_numpy(), "p_true": e.p_true.to_numpy(), "condition": label}))
    return rows_frame(pd.concat(out, ignore_index=True))


def psro_frame(run: Path) -> pd.DataFrame:
    """Held-out episodes at the first and last equilibrium, weighted by the mixture, with the 3-sample mean judge."""
    from decisive_psro_report import load_cell

    out = []
    for proto in ("debate_seq", "debate_sim"):
        for d in sorted((_found(run / "psro") / proto).glob("*")):
            if not (d / "state.json").exists():
                continue
            state, pairs = load_cell(d)
            its = [r for r in state["iterations"] if "sigma" in r]
            for label, rec in (("unprompted", its[0]), ("equilibrium", its[-1])):
                pops, sig = rec["populations"], rec["sigma"]
                for hs, wh in zip(pops["honest"], sig["honest"]):
                    for ls, wl in zip(pops["liar"], sig["liar"]):
                        df = pairs.get(("test", hs, ls))
                        if df is None or wh * wl < 1e-6:
                            continue
                        p = [float(np.mean([v for v in ps if v is not None])) if isinstance(ps, list) and any(v is not None for v in ps)
                             else np.nan for ps in df.p_true_samples]
                        base = pd.DataFrame({"item": df.index.to_numpy(), "p_true": p, "w": wh * wl})
                        base = base[base.p_true.notna()]
                        for v, pside in ((1.0, base.p_true), (-1.0, 1 - base.p_true)):
                            out.append(base.assign(judge=state["judge"], mechanism=proto, condition=label, value=v,
                                                   y=float(v > 0), p=pside))
    return pd.concat(out, ignore_index=True)


def apply_transfer(target: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Calibrate ``target`` with maps fitted on all of the panel's (unselected) items, per judge and protocol family."""
    fit = panel.assign(family=panel.mechanism.map(FAMILY))
    maps = {}
    for (j, fam), g in fit.groupby(["judge", "family"]):
        maps[(j, fam)] = platt(logit(g.p), g.y.to_numpy(), g.w.to_numpy(), fam in SYMMETRIC)
    target = target.copy()
    target["pt"] = np.nan
    for (j, m), g in target.groupby(["judge", "mechanism"]):
        if (j, FAMILY[m]) in maps:
            a, b = maps[(j, FAMILY[m])]
            target.loc[g.index, "pt"] = expit(a + b * logit(g.p))
    return target


def summarize(df: pd.DataFrame, cols: dict[str, str], baseline: pd.DataFrame | None, by: list[str],
              pool_label: str = "all") -> pd.DataFrame:
    """Brier and log ASD per ``by`` and pooled over judges (a judge's items averaged first), under each probability
    column of ``cols``. Gains are per item against the same judge's no-argument baseline, under the same treatment."""
    rows = []
    for name, col in cols.items():
        it = item_asd(df, by + ["judge"], col)
        bl = None
        if baseline is not None:
            bl = item_asd(baseline, ["judge"], col).rename(columns={"b": "bb", "l": "bl"})
            it = it.merge(bl, on=["judge", "item"], how="left")
        else:
            it = it.assign(bb=np.nan, bl=np.nan)
        it["gain_b"], it["gain_l"] = it.b - it.bb, it.l - it.bl
        for j, part in [(pool_label, it.groupby(by + ["item"], as_index=False).mean(numeric_only=True).assign(judge=pool_label))] + list(it.groupby("judge")):
            for key, g in part.groupby(by):
                key = key if isinstance(key, tuple) else (key,)
                rec = {**dict(zip(by, key)), "judge": j, "calibration": name, "items": len(g)}
                for c in ("b", "l", "gain_b", "gain_l", "bb", "bl"):
                    rec[c], rec[f"{c}_lo"], rec[f"{c}_hi"] = boot(g[c].to_numpy())
                rows.append(rec)
    return pd.DataFrame(rows)


def agent_table(panel: pd.DataFrame, a_idx: dict) -> pd.DataFrame:
    """Per agent (pooled over judges): gain over the no-argument judge, raw and calibrated, and the slope on AA index."""
    base = panel[panel.mechanism == "direct"]
    prot = panel[panel.mechanism != "direct"]
    out = []
    for name, col in (("raw", "p"), ("calibrated", "pc")):
        bl = item_asd(base, ["judge"], col).groupby("item", as_index=False).b.mean().rename(columns={"b": "bb"})
        it = item_asd(prot, ["mechanism", "debater", "judge"], col).groupby(["mechanism", "debater", "item"], as_index=False).b.mean()
        it = it.merge(bl, on="item", how="left")
        it["gain_b"] = it.b - it.bb
        for (m, a), g in it.groupby(["mechanism", "debater"]):
            mean, lo, hi = boot(g.gain_b.to_numpy())
            out.append({"mechanism": m, "debater": a, "aa": a_idx.get(a), "calibration": name, "b": g.b.mean(),
                        "gain_b": mean, "gain_b_lo": lo, "gain_b_hi": hi})
    return pd.DataFrame(out)


def selection_bridge(panel: pd.DataFrame, run: Path) -> pd.DataFrame:
    """The panel's own data, split by whether an item was in the gap-selected set. Selection used separate samples of
    the same weak judges, so the panel's verdicts on these items are fresh."""
    sel = set(json.loads(_found(run / "select" / "selected.json").read_text())["items"])
    weak = ["gpt-4o-mini", "llama-3.1-8b", "gemma-3-12b"]
    out = []
    p = panel[panel.judge.isin(weak)]
    for label, x in (("gap-selected", p[p.item.isin(sel)]), ("not selected", p[~p.item.isin(sel)])):
        base = x[x.mechanism == "direct"]
        bl = item_asd(base, ["judge"], "p").groupby("item").b.mean()
        for m in ("debate_sim", "consultancy_1r"):
            it = item_asd(x[x.mechanism == m], ["debater", "judge"], "p").groupby("item").b.mean()
            gain = (it - bl).dropna()
            a = boot(gain.to_numpy())
            out.append({"items_set": label, "mechanism": m, "items": len(gain), "blind_b": float(bl.mean()),
                        "protocol_b": float(it.mean()), "gain_b": a[0], "gain_b_lo": a[1], "gain_b_hi": a[2]})
    return pd.DataFrame(out)


def fmt(r, c) -> str:
    return f"{r[c]:+.3f} [{r[c + '_lo']:+.3f}, {r[c + '_hi']:+.3f}]"


def markdown(t: dict) -> str:
    L = ["# ASD after a calibration map", "",
         "Each judge's stated probability goes through a Platt map (logit p' = a + b logit p, b >= 0), fitted per judge "
         "and protocol and cross-fitted over items. `b` above 1 means the judge was underconfident, below 1 overconfident. "
         "All prompted liars; Brier ASD unless stated.", "",
         "## Model panel (300 unselected items): is the judge less confident with an argument?", "",
         "| judge | condition | accuracy | mean abs(p - 0.5) | a | b |", "|---|---|---|---|---|---|"]
    for _, r in t["panel_maps"].iterrows():
        L.append(f"| {r.judge} | {NAMES[r.mechanism]} | {r.accuracy:.3f} | {r.confidence:.3f} | {r.a:+.2f} | {r.b:.2f} |")
    L += ["", "## Model panel: gain over the same judge with no argument", "",
          "| judge | protocol | raw ASD | raw gain | calibrated ASD | calibrated gain |", "|---|---|---|---|---|---|"]
    s = t["panel"]
    for (j, m), g in s.groupby(["judge", "mechanism"]):
        r, c = g[g.calibration == "raw"].iloc[0], g[g.calibration == "calibrated"].iloc[0]
        L.append(f"| {j} | {NAMES[m]} | {r.b:+.3f} | {fmt(r, 'gain_b')} | {c.b:+.3f} | {fmt(c, 'gain_b')} |")
    L += ["", "## Appendix: the panel with committed liars only (gpt-5 audit)", "",
          "| judge | protocol | raw gain | calibrated gain |", "|---|---|---|---|"]
    for (j, m), g in t["panel_committed"].groupby(["judge", "mechanism"]):
        r, c = g[g.calibration == "raw"].iloc[0], g[g.calibration == "calibrated"].iloc[0]
        L.append(f"| {j} | {NAMES[m]} | {fmt(r, 'gain_b')} | {fmt(c, 'gain_b')} |")
    for name, key, cond in (("Protocol grid (927 gap-selected items, gpt-5-mini)", "grid", False),
                            ("Prompt optimization (480 held-out items, gpt-5-mini)", "po", True),
                            ("Self-play (390 held-out items, gpt-5-mini; mean of 3 judge samples)", "psro", True)):
        L += ["", f"## {name}", "",
              "`own map`: fitted on these items, so it learns the item selection. `panel map`: fitted on the panel's "
              "unselected items and applied here.", "",
              "| protocol |" + (" stage |" if cond else "") + " judge | treatment | ASD, Brier | ASD, log | gain over no argument, Brier |",
              "|---|" + ("---|" if cond else "") + "---|---|---|---|---|"]
        for _, r in t[key].sort_values(["mechanism"] + (["condition"] if cond else []) + ["judge", "calibration"]).iterrows():
            L.append(f"| {NAMES.get(r.mechanism, r.mechanism)} |" + (f" {r.condition} |" if cond else "") +
                     f" {r.judge} | {r.calibration} | {fmt(r, 'b')} | {r.l:+.3f} | {fmt(r, 'gain_b') if pd.notna(r.gain_b) else '-'} |")
    L += ["", "## The panel split by gap selection (weak judges, fresh verdicts)", "",
          "| items | protocol | n | judge alone | with the protocol | gain [95% CI] |", "|---|---|---|---|---|---|"]
    for _, r in t["bridge"].iterrows():
        L.append(f"| {r.items_set} | {NAMES[r.mechanism]} | {r['items']} | {r.blind_b:+.3f} | {r.protocol_b:+.3f} | {fmt(r, 'gain_b')} |")
    return "\n".join(L) + "\n"


def figure(t: dict, out: Path) -> None:
    """Raw against calibrated: per judge on the panel, then per protocol on the optimized stages."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import eic_style as st
    from eic_style import BLUE, INK2

    GRAY = "#8f8d86"
    j_idx = {j["name"]: j["aa_index"] for j in panel_config()["judges"]}
    panel = t["panel"][t["panel"].judge != "all"]
    order = sorted(j_idx, key=j_idx.get)

    def pair(ax, y, r, c, col):
        ax.plot([r[col], c[col]], [y, y], color="#c9c7bf", lw=1.2, zorder=1)
        for rec, color, dy in ((r, GRAY, 0.09), (c, BLUE, -0.09)):
            ax.errorbar(rec[col], y + dy, xerr=[[rec[col] - rec[f"{col}_lo"]], [rec[f"{col}_hi"] - rec[col]]], fmt="o",
                        color=color, ms=5.5, mec="white", mew=0.8, elinewidth=1, capsize=0, zorder=3)

    for paper in (False, True):
        with st.paper_mode() if paper else _Null():
            fig, axes = plt.subplots(1, 4, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1, .85, .85]})
            for ax, m in zip(axes[:2], ("debate_sim", "consultancy_1r")):
                st.axes(ax)
                ax.axvline(0, color=INK2, lw=0.9, zorder=1)
                for y, j in enumerate(order):
                    g = panel[(panel.judge == j) & (panel.mechanism == m)]
                    pair(ax, y, g[g.calibration == "raw"].iloc[0], g[g.calibration == "calibrated"].iloc[0], "gain_b")
                ax.set_yticks(range(len(order)), order if m == "debate_sim" else [])
                ax.set_title(NAMES[m])
                ax.set_xlabel("gain over the same judge with no argument\n(Brier ASD; 300 unselected items)")
            stages = [("po", "after search", "prompt-optimized"), ("psro", "equilibrium", "self-play equilibrium")]
            for ax, (key, cond, title) in zip(axes[2:], stages):
                st.axes(ax)
                ax.axvline(0, color=INK2, lw=0.9, zorder=1)
                d = t[key][(t[key].judge == "all") & (t[key].condition == cond)]
                ms = [m for m in ("consultancy", "debate_seq", "debate_sim") if m in set(d.mechanism)]
                for y, m in enumerate(ms):
                    g = d[d.mechanism == m]
                    pair(ax, y, g[g.calibration == "raw"].iloc[0], g[g.calibration == "panel map"].iloc[0], "b")
                ax.set_yticks(range(len(ms)), [NAMES[m] for m in ms])
                ax.set_ylim(-0.6, len(ms) - 0.4)
                ax.set_title(title)
                ax.set_xlabel("ASD, Brier score\n(gap-selected items, held out)")
            handles = [Line2D([], [], color=GRAY, marker="o", mec="white", lw=0, ms=6, label="stated probabilities"),
                       Line2D([], [], color=BLUE, marker="o", mec="white", lw=0, ms=6, label="after a calibration map")]
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.06, 0.945), ncol=2, fontsize=7.5)
            fig.suptitle("Calibration shrinks the ASDs, and strong judges still gain nothing from either protocol",
                         x=0.005, ha="left", fontsize=10.5, fontweight="bold", y=1.0)
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            st.save(fig, out, "fig15_calibration")


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def tables(run: Path) -> dict:
    panel = crossfit(panel_frame(run), ["judge", "mechanism"])
    t = {"panel_maps": maps_for(panel, ["judge", "mechanism"])}
    base = panel[panel.mechanism == "direct"]
    t["panel"] = summarize(panel[panel.mechanism != "direct"], {"raw": "p", "calibrated": "pc"}, base, ["mechanism"])
    pc = crossfit(panel_committed_frame(run), ["judge", "mechanism"])
    t["panel_committed"] = summarize(pc[pc.mechanism != "direct"], {"raw": "p", "calibrated": "pc"},
                                     pc[pc.mechanism == "direct"], ["mechanism"])
    cfg = panel_config()
    t["panel_agents"] = agent_table(panel, {a["name"]: a["aa_index"] for a in cfg["agents"]})
    t["bridge"] = selection_bridge(panel, run)
    treat = {"raw": "p", "own map": "pc", "panel map": "pt"}

    grid = apply_transfer(crossfit(grid_frame(run), ["judge", "mechanism"]), panel)
    t["grid_maps"] = maps_for(grid, ["judge", "mechanism"])
    t["grid"] = summarize(grid[grid.mechanism != "direct"], treat, grid[grid.mechanism == "direct"], ["mechanism"])

    def staged(frame: pd.DataFrame, map_base: pd.DataFrame | None = None) -> pd.DataFrame:
        f = apply_transfer(crossfit(frame, ["judge", "mechanism", "condition"]), panel)
        return summarize(f, treat, map_base, ["mechanism", "condition"])

    bl = grid[grid.mechanism == "direct"]
    t["po"] = staged(po_frame(run), bl)
    t["psro"] = staged(psro_frame(run), bl)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    t = tables(Path(a.run))
    out = Path(a.out) / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    for k, v in t.items():
        v.to_csv(out / f"{k}.csv", index=False)
    md = markdown(t)
    (out / "calibration.md").write_text(md)
    print(md)
    if not a.no_figures:
        figure(t, Path(a.out) / "eic")


if __name__ == "__main__":
    main()
