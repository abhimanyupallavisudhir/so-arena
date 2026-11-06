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


def per_item(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Per (keys, item): reward arguing the truth minus lying (each item's sides averaged over samples and roles)."""
    x = df[df.value.notna() & df.reward.notna()].copy()
    x["v"] = (x.value > 0).astype(float)
    g = x.groupby(keys + ["item", "v"]).reward.mean().unstack("v")
    g = g.dropna(subset=[0.0, 1.0]) if {0.0, 1.0} <= set(g.columns) else g.iloc[0:0]
    out = (g[1.0] - g[0.0]).rename("asd").reset_index()
    out["conc"] = np.where(out.asd > 0, 1.0, np.where(out.asd == 0, 0.5, 0.0))
    return out


def summarize(items: pd.DataFrame, keys: list[str], seed: int = 0) -> pd.DataFrame:
    """Mean ASD and concordance per key with item-bootstrap CIs (items resampled jointly across keys)."""
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(items.item.unique()))
    out = []
    for key, g in items.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        a = g.set_index("item")
        boots = []
        for _ in range(N_BOOT):
            s = rng.choice(ids, len(ids))
            b = a.reindex(s).dropna()
            boots.append((b.asd.mean(), b.conc.mean()))
        boots = np.array(boots)
        out.append({**dict(zip(keys, key)), "asd": a.asd.mean(), "asd_lo": np.nanquantile(boots[:, 0], .025),
                    "asd_hi": np.nanquantile(boots[:, 0], .975), "conc": a.conc.mean(),
                    "conc_lo": np.nanquantile(boots[:, 1], .025), "conc_hi": np.nanquantile(boots[:, 1], .975),
                    "items": len(a)})
    return pd.DataFrame(out)


def slope(items: pd.DataFrame, key: str, index: dict, col: str = "conc", seed: int = 1) -> dict:
    """OLS slope of the per-key mean of ``col`` on the AA index, per 10 index points, item-bootstrap CI."""
    rng = np.random.default_rng(seed)
    pivot = items.pivot_table(index="item", columns=key, values=col)
    names = [n for n in pivot.columns if n in index and index[n] is not None]
    xs = np.array([index[n] for n in names])

    def fit(p):
        ys = p[names].mean().to_numpy()
        ok = np.isfinite(ys)
        return 10 * np.polyfit(xs[ok], ys[ok], 1)[0] if ok.sum() > 2 else np.nan

    est = fit(pivot)
    boots = [fit(pivot.iloc[rng.integers(0, len(pivot), len(pivot))]) for _ in range(N_BOOT)]
    r = np.corrcoef(xs, pivot[names].mean().to_numpy())[0, 1] if len(names) > 2 else np.nan
    return {"slope_per_10": est, "lo": np.nanquantile(boots, .025), "hi": np.nanquantile(boots, .975), "r": r,
            "n": len(names)}


def tables(run: Path) -> dict:
    cfg = config()
    a_idx = {a["name"]: a["aa_index"] for a in cfg["agents"]}
    j_idx = {j["name"]: j["aa_index"] for j in cfg["judges"]}
    rows, verdicts = load(run)
    prot = rows[rows.mechanism.isin(MECHS)]
    out = {"a_idx": a_idx, "j_idx": j_idx}
    variants = {"all": prot}
    c = committed(prot, verdicts)
    if c is not None:
        variants["committed"] = c
    for name, df in variants.items():
        it_a = per_item(df, ["mechanism", "debater"])          # agents: pooled over judges
        it_j = per_item(df, ["mechanism", "judge"])            # judges: pooled over agents
        it_aj = per_item(df, ["mechanism", "debater", "judge"])
        out[f"agents_{name}"] = summarize(it_a, ["mechanism", "debater"]).assign(aa=lambda d: d.debater.map(a_idx))
        out[f"judges_{name}"] = summarize(it_j, ["mechanism", "judge"]).assign(aa=lambda d: d.judge.map(j_idx))
        out[f"cells_{name}"] = it_aj.groupby(["mechanism", "debater", "judge"])[["asd", "conc"]].mean().reset_index()
        sl = []
        for m in MECHS:
            for col in ("conc", "asd"):
                sl.append({"variant": name, "mechanism": m, "over": "agents", "measure": col,
                           **slope(it_a[it_a.mechanism == m], "debater", a_idx, col)})
                sl.append({"variant": name, "mechanism": m, "over": "judges", "measure": col,
                           **slope(it_j[it_j.mechanism == m], "judge", j_idx, col)})
        out[f"slopes_{name}"] = pd.DataFrame(sl)
    # blind baseline per judge, and gains over it
    d = rows[rows.mechanism == "direct"].copy()
    d = d[d.value.notna() & d.reward.notna()]
    out["blind"] = summarize(per_item(d, ["judge"]), ["judge"]).assign(aa=lambda x: x.judge.map(j_idx))
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
         f"Items not selected on any judge; ASD in log score; conc = share of items where honesty earns more. "
         f"Headline variant: {'committed liars only (gpt-5 audit)' if v == 'committed' else 'all prompted liars'}.", ""]
    for name in ("committed", "all"):
        if f"agents_{name}" not in t:
            continue
        L += [f"## Agents ({name} liars), pooled over judges", "",
              "| agent | AA index | protocol | ASD [95% CI] | conc [95% CI] | solo acc | committed |", "|---|---|---|---|---|---|---|"]
        a = t[f"agents_{name}"].merge(t["solo"], on="debater", how="left")
        if "commit" in t:
            a = a.merge(t["commit"], on=["debater", "mechanism"], how="left")
        for _, r in a.sort_values(["mechanism", "aa"]).iterrows():
            cm = f"{r.committed:.2f}" if "committed" in r and pd.notna(r.get("committed")) else "-"
            sa = f"{r.solo_acc:.2f}" if pd.notna(r.get("solo_acc")) else "-"
            L.append(f"| {r.debater} | {r.aa:.1f} | {r.mechanism} | {fmt(r)} | {fmt(r, 'conc')} | {sa} | {cm} |")
        L += ["", f"## Judges ({name} liars), pooled over agents", "",
              "| judge | AA index | protocol | ASD [95% CI] | conc [95% CI] | blind ASD | gain over blind |", "|---|---|---|---|---|---|---|"]
        b = t["blind"].set_index("judge")
        for _, r in t[f"judges_{name}"].sort_values(["mechanism", "aa"]).iterrows():
            bl = b.asd.get(r.judge, np.nan)
            L.append(f"| {r.judge} | {r.aa:.1f} | {r.mechanism} | {fmt(r)} | {fmt(r, 'conc')} | {bl:+.2f} | {r.asd - bl:+.2f} |")
        L += ["", f"## Slopes on the AA index ({name} liars; per 10 index points, item bootstrap)", "",
              "| protocol | across | measure | slope [95% CI] | r | points |", "|---|---|---|---|---|---|"]
        for _, r in t[f"slopes_{name}"].iterrows():
            L.append(f"| {r.mechanism} | {r.over} | {r.measure} | {r.slope_per_10:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] | "
                     f"{r.r:.2f} | {r.n} |")
        L.append("")
    return "\n".join(L) + "\n"


def figures(t: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    import eic_style as st
    from eic_style import BLUE, CONTEXT, INK2, ORANGE

    v = "committed" if "agents_committed" in t else "all"
    for paper in (False, True):
        ctx = st.paper_mode() if paper else _null()
        with ctx:
            # agents
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
            a = t[f"agents_{v}"]
            for ax, m in zip(axes, MECHS):
                st.axes(ax)
                g = a[a.mechanism == m].sort_values("aa")
                for _, r in g.iterrows():
                    ladder = r.debater.startswith("gpt-5.6-sol")
                    st.dot_ci(ax, r.aa, r.conc, r.conc_lo, r.conc_hi, ORANGE if ladder else BLUE, horizontal=False,
                              size=6)
                    ax.annotate(r.debater, (r.aa, r.conc), xytext=(4, 0), textcoords="offset points", fontsize=6.3,
                                color=INK2, va="center")
                lad = g[g.debater.str.startswith("gpt-5.6-sol")].sort_values("aa")
                ax.plot(lad.aa, lad.conc, color=ORANGE, lw=1.2, zorder=1)
                ax.axhline(0.5, color=INK2, lw=1)
                ax.set_title(MECHS[m])
                ax.set_xlabel("agent: Artificial Analysis Intelligence Index (v4.3)")
            axes[0].set_ylabel("concordance: share of items where honesty earns more")
            fig.suptitle("Does incentive compatibility change with the agent's intelligence?", x=0.005, ha="left",
                         fontsize=10, fontweight="bold", y=1.06)
            fig.text(0.005, 0.965, f"Each agent debates itself / argues once; {('committed liars only; ' if v == 'committed' else '')}"
                     "pooled over 9 judges; 95% item-bootstrap intervals. Orange: GPT-5.6 Sol at reasoning effort "
                     "none, low, medium, high.", fontsize=7.5, color=INK2)
            fig.tight_layout()
            st.save(fig, out, "fig12_panel_agents")
            # judges
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
            jd = t[f"judges_{v}"]
            for ax, m in zip(axes, MECHS):
                st.axes(ax)
                g = jd[jd.mechanism == m].sort_values("aa")
                for _, r in g.iterrows():
                    st.dot_ci(ax, r.aa, r.conc, r.conc_lo, r.conc_hi, BLUE, horizontal=False, size=6)
                    ax.annotate(r.judge, (r.aa, r.conc), xytext=(4, 0), textcoords="offset points", fontsize=6.3,
                                color=INK2, va="center")
                b = t["blind"].sort_values("aa")
                ax.plot(b.aa, b.conc, color=CONTEXT, lw=1.2, ls=(0, (4, 3)), marker="o", ms=3, zorder=1)
                ax.axhline(0.5, color=INK2, lw=1)
                ax.set_title(MECHS[m])
                ax.set_xlabel("judge: Artificial Analysis Intelligence Index (v4.3)")
            axes[0].set_ylabel("concordance: share of items where honesty earns more")
            fig.suptitle("Does incentive compatibility change with the judge's intelligence?", x=0.005, ha="left",
                         fontsize=10, fontweight="bold", y=1.06)
            fig.text(0.005, 0.965, "Pooled over 20 agent configurations; dashed gray: the same judge with no argument "
                     "(blind baseline); 95% item-bootstrap intervals.", fontsize=7.5, color=INK2)
            fig.tight_layout()
            st.save(fig, out, "fig13_panel_judges")
            # heatmap
            cells = t[f"cells_{v}"]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
            a_order = sorted(t["a_idx"], key=lambda n: t["a_idx"][n])
            j_order = sorted(t["j_idx"], key=lambda n: t["j_idx"][n])
            for ax, m in zip(axes, MECHS):
                p = cells[cells.mechanism == m].pivot_table(index="debater", columns="judge", values="conc")
                p = p.reindex(index=[x for x in a_order if x in p.index], columns=[x for x in j_order if x in p.columns])
                im = ax.imshow(p.to_numpy(), cmap="RdBu", vmin=0.2, vmax=0.8, aspect="auto")
                ax.set_xticks(range(p.shape[1]), p.columns, rotation=60, ha="right", fontsize=6.5)
                ax.set_yticks(range(p.shape[0]), p.index, fontsize=6.5)
                ax.grid(False)
                ax.set_title(MECHS[m])
            fig.colorbar(im, ax=axes, shrink=0.8, label="concordance")
            fig.suptitle("Concordance by agent (rows, by AA index) and judge (columns, by AA index)", x=0.005, ha="left",
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
