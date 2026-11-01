"""Decisive study, report: gap achieved, every protocol against the judge alone, liars audited and optimized.

    python experiments/decisive_report.py runs/decisive [docs/results]
    python experiments/decisive_report.py docs/results/decisive_data   # from the committed data bundle

Reads the stage outputs under the run directory and writes, to the output directory (``docs/results`` by
default), ``decisive_*.csv`` tables, ``decisive.md`` and two figures. Every CI is a percentile bootstrap over
items (2,000 resamples): an item's arms, samples and - for pooled rows - (judge, debater) pairs are resampled
together.

Metrics, per (judge, debater, protocol):

* accuracy - share of verdicts that pick the true answer (ties and unparsed verdicts count 1/2), averaged over
  arms and samples within an item, then over items;
* delta - accuracy minus the same judge's accuracy alone on the same items (paired by item);
* ASD - reward arguing the truth minus reward arguing a falsehood, in the protocol's own units (log score of
  the judge's probability; recursive debate: the judge's probability; prover-estimator: the prover's payoff);
* concordance - P(reward arguing the truth > reward arguing the falsehood) over items;
* gap recovered - (accuracy - judge alone) / (debater alone - judge alone): the share of the sandwich gap the
  protocol closes;
* IGR - the same ratio for ASD measured on one scale for every protocol: the Brier score of the verdict's
  probability on the lead agent's side (judge alone and debater alone included), i.e. the incentive gap
  recovered relative to the debater alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decisive_common import DEBATERS, JUDGES  # noqa: E402

PROTOCOLS = ["consultancy", "debate_seq", "debate_sim", "recursive_debate", "prover_estimator"]
LABEL = {"direct": "judge alone", "consultancy": "consultancy", "debate_seq": "debate (sequential)",
         "debate_sim": "debate (simultaneous)", "recursive_debate": "recursive debate",
         "prover_estimator": "prover-estimator", "direct_strong": "debater alone",
         "consultancy_1r": "single argument (consultancy, 1 round)", "debate_1r": "debate (1 round)"}
N_BOOT = 2000


# ------------------------------------------------------------------------------------------------ per item

def lead_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per episode: the first stance-holding lead role (debater_a in debate, the phantom agent of the
    references), which takes the arm's stance - so arms are balanced."""
    return rows.sort_values("role").drop_duplicates("episode", keep="first")


def brier_reward(p_true: pd.Series, value: pd.Series) -> pd.Series:
    """Brier score of the verdict for the side ``value`` argued (binary: -2 (1 - p_side)^2; 0 is perfect)."""
    p_side = np.where(value > 0, p_true, 1 - p_true)
    return -2 * (1 - p_side) ** 2


def item_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (judge, debater, mechanism, item): accuracy, verdict-Brier ASD, protocol-units ASD and concordance."""
    ep = lead_rows(rows).copy()
    ep["brier"] = brier_reward(ep.p_true, ep.value)
    keys = ["judge", "debater", "mechanism", "item"]
    acc = ep.groupby(keys).correct.mean().rename("accuracy")
    b = ep.groupby(keys + [ep.value > 0]).brier.mean().unstack()
    asd_b = (b.get(True) - b.get(False)).rename("asd_brier")
    r = rows[rows.value.notna() & rows.reward.notna()]
    rr = r.groupby(keys + [r.value > 0]).reward.mean().unstack()
    asd = (rr.get(True) - rr.get(False)).rename("asd")
    conc = np.sign(asd).map({1.0: 1.0, 0.0: 0.5, -1.0: 0.0}).rename("concordance")
    return pd.concat([acc, asd_b, asd, conc], axis=1).reset_index()


def with_references(it: pd.DataFrame) -> pd.DataFrame:
    """Attach, per row, the same judge's accuracy/ASD alone and the same debater's alone on the same item."""
    j = it[it.mechanism == "direct"][["judge", "item", "accuracy", "asd_brier"]]
    j = j.rename(columns={"accuracy": "acc_judge", "asd_brier": "asdb_judge"})
    d = it[it.mechanism == "direct_strong"][["debater", "item", "accuracy", "asd_brier"]]
    d = d.rename(columns={"accuracy": "acc_debater", "asd_brier": "asdb_debater"})
    p = it[~it.mechanism.isin(["direct", "direct_strong"])]
    return p.merge(j, on=["judge", "item"], how="left").merge(d, on=["debater", "item"], how="left")


# ------------------------------------------------------------------------------------------------ bootstrap

def item_means(frame: pd.DataFrame, cols: list[str], *, n_boot: int = N_BOOT, seed: int = 0
               ) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Column means over rows, and their bootstrap replicates resampling items (all rows of an item together)."""
    g = frame.groupby("item")
    sums, counts = g[cols].sum(min_count=1), g[cols].count()
    S, N = sums.fillna(0).to_numpy(float), counts.to_numpy(float)
    est = {c: S[:, i].sum() / N[:, i].sum() if N[:, i].sum() else np.nan for i, c in enumerate(cols)}
    rng = np.random.default_rng(seed)
    k = len(sums)
    W = np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = {c: (W @ S[:, i]) / (W @ N[:, i]) for i, c in enumerate(cols)}
    return est, reps


def boot(frame: pd.DataFrame, stat, *, n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float, float]:
    """``stat`` (a function of column means) and its percentile CI, resampling items."""
    est, reps = item_means(frame, stat.cols, n_boot=n_boot, seed=seed)
    vals = stat(reps)
    if frame.item.nunique() < 2:
        return float(stat(est)), np.nan, np.nan
    lo, hi = np.nanquantile(np.asarray(vals, dtype=float), [0.025, 0.975])
    return float(stat(est)), float(lo), float(hi)


def _stat(fn, cols):
    fn.cols = cols
    return fn


def mean_of(col):
    return _stat(lambda m: m[col], [col])


def diff_of(a, b):
    """Mean of a - b over rows that have both (paired); rows missing either are dropped by the caller."""
    return _stat(lambda m: m[a] - m[b], [a, b])


def ratio_of(num_a, num_b, den_a, den_b):
    def fn(m):
        den = m[den_a] - m[den_b]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(np.abs(den) > 1e-9, (m[num_a] - m[num_b]) / np.where(np.abs(den) > 1e-9, den, 1), np.nan)
    return _stat(fn, [num_a, num_b, den_a, den_b])


def summarize(it: pd.DataFrame, by: list[str], n_boot: int = N_BOOT) -> pd.DataFrame:
    """Accuracy, delta, gap recovered, IGR (with CIs), ASD and concordance per group of ``by``. Rows lacking a
    reference (judge or debater alone on the item) are dropped, so every comparison is paired."""
    out = []
    for key, g in it.groupby(by):
        g = g.dropna(subset=["accuracy", "acc_judge", "acc_debater"]).reset_index(drop=True)
        if g.empty:
            continue
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["judge_alone"] = float(g.acc_judge.mean())
        row["debater_alone"] = float(g.acc_debater.mean())
        for name, stat in (("accuracy", mean_of("accuracy")), ("delta", diff_of("accuracy", "acc_judge")),
                           ("gap", diff_of("acc_debater", "acc_judge")),
                           ("gap_recovered", ratio_of("accuracy", "acc_judge", "acc_debater", "acc_judge")),
                           ("igr", ratio_of("asd_brier", "asdb_judge", "asdb_debater", "asdb_judge"))):
            est, lo, hi = boot(g, stat, n_boot=n_boot)
            row.update({name: float(est), f"{name}_lo": lo, f"{name}_hi": hi})
        row["asd"] = float(g.asd.mean())
        row["concordance"] = float(g.concordance.mean())
        row["items"] = int(g.item.nunique())
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ audit

def optimized_vs_references(curve_items: pd.DataFrame, it: pd.DataFrame, by: list[str],
                            n_boot: int = N_BOOT) -> pd.DataFrame:
    """Best-of-n accuracy against the same judge alone and the same debater alone on the same items (paired;
    rows are (pair, item), items resampled together), per ``by`` + (mechanism, mode, n)."""
    refs = it.drop_duplicates(["judge", "debater", "item"])[["judge", "debater", "item", "acc_judge", "acc_debater"]]
    m = curve_items.merge(refs, on=["judge", "debater", "item"], how="inner").dropna(
        subset=["accuracy", "acc_judge", "acc_debater"])
    out = []
    for key, g in m.groupby(by + ["mechanism", "mode", "n"]):
        key = dict(zip(by + ["mechanism", "mode", "n"], key if isinstance(key, tuple) else (key,)))
        acc = boot(g, mean_of("accuracy"), n_boot=n_boot)
        delta = boot(g, diff_of("accuracy", "acc_judge"), n_boot=n_boot)
        rec = boot(g, ratio_of("accuracy", "acc_judge", "acc_debater", "acc_judge"), n_boot=n_boot)
        out.append({**key, "accuracy": acc[0], "accuracy_lo": acc[1], "accuracy_hi": acc[2],
                    "judge_alone": g.acc_judge.mean(), "debater_alone": g.acc_debater.mean(),
                    "delta": delta[0], "delta_lo": delta[1], "delta_hi": delta[2],
                    "gap_recovered": rec[0], "gap_recovered_lo": rec[1], "gap_recovered_hi": rec[2],
                    "items": g.item.nunique()})
    return pd.DataFrame(out)


def committed_filter(rows: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Rows of (judge, debater, protocol, item, sample) groups whose liars were all committed, both arms kept
    together; the references are kept whole (they are compared on the surviving items only)."""
    v = verdicts[verdicts.kind == "prompted"]
    bad = v[v.verdict != "committed"][["judge", "debater", "mechanism", "item", "repeat"]].drop_duplicates()
    bad["bad"] = True
    m = rows.merge(bad, on=["judge", "debater", "mechanism", "item", "repeat"], how="left")
    return m[m["bad"].isna()].drop(columns="bad")


def audit_rates(verdicts: pd.DataFrame) -> pd.DataFrame:
    v = verdicts[verdicts.kind == "prompted"].drop_duplicates("key")  # judge-independent transcripts once
    g = v.groupby(["debater", "mechanism"])
    out = g.verdict.value_counts(normalize=True).unstack(fill_value=0.0)
    out["tell"] = g.tell.mean()
    out["n"] = g.size()
    return out.reset_index()


def selected_audit(sel: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Share of the liar's selected arguments (weighted by best-of-n's selection probability) per verdict."""
    v = verdicts[verdicts.kind == "optimized"][["debater", "mechanism", "item", "arm", "pool_key", "index",
                                                "verdict"]].drop_duplicates()
    m = sel.merge(v.rename(columns={"pool_key": "key"}), on=["debater", "mechanism", "item", "arm", "key", "index"],
                  how="inner")
    m["not_committed"] = (m.verdict != "committed").astype(float) * m.weight
    keys = ["debater", "mechanism", "n"]
    g = m.groupby(keys)
    return pd.DataFrame({"not_committed": g.not_committed.sum() / g.weight.sum(),
                         "conceded": (m.assign(x=(m.verdict == "conceded") * m.weight)
                                      .groupby(keys).x.sum() / g.weight.sum())}).reset_index()


# ------------------------------------------------------------------------------------------------ figures

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def _axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def figure_protocols(pooled_all: pd.DataFrame, pooled_committed: pd.DataFrame, out: Path) -> None:
    """Accuracy minus the judge alone, per protocol and debater, pooled over judges: all prompted liars vs
    committed liars only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    debaters = [d for d in DEBATERS if d in set(pooled_all.debater)]
    fig, axes = plt.subplots(1, len(debaters), figsize=(4.2 * len(debaters), 3.6), sharey=True, squeeze=False)
    series = [("all prompted liars", pooled_all, SERIES[0], "o"),
              ("committed liars only", pooled_committed, SERIES[1], "s")]
    for ax, d in zip(axes[0], debaters):
        _axes(ax)
        x = np.arange(len(PROTOCOLS))
        for k, (name, df, color, marker) in enumerate(series):
            g = df[df.debater == d].set_index("mechanism").reindex(PROTOCOLS)
            xs = x + (k - 0.5) * 0.22
            ax.vlines(xs, g.delta_lo, g.delta_hi, color=color, lw=2)
            ax.plot(xs, g.delta, marker, color=color, ms=7, mec="white", mew=1.5, label=name, ls="none")
        ax.axhline(0, color=INK2, lw=1)
        ax.text(-0.45, 0.005, "judge alone", color=INK2, fontsize=7.5, ha="left", va="bottom")
        ax.set_xticks(x, [LABEL[p].replace(" (", "\n(").replace("prover-estimator", "prover-\nestimator")
                          for p in PROTOCOLS], fontsize=7.5)
        ax.set_title(f"debater: {d}", color=INK, fontsize=9, loc="left")
    axes[0][0].set_ylabel("verdict accuracy minus the judge alone\n(pooled over 3 judges; 95% CI)", color=INK2,
                          fontsize=8.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "decisive_protocols.png", dpi=170)
    plt.close(fig)


def figure_optimization(curves: pd.DataFrame, refs: dict[str, float], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mechs = [m for m in ("consultancy_1r", "debate_1r") if m in set(curves.mechanism)]
    fig, axes = plt.subplots(1, len(mechs), figsize=(4.4 * len(mechs), 3.5), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], mechs):
        _axes(ax)
        for k, (mode, name) in enumerate((("liar", "liar optimized"), ("both", "both sides optimized"))):
            g = curves[(curves.mechanism == m) & (curves["mode"] == mode)].set_index("n").sort_index()
            ax.fill_between(g.index, g.lo, g.hi, color=SERIES[k], alpha=0.15, lw=0)
            ax.plot(g.index, g.accuracy, "o-", color=SERIES[k], lw=2, ms=6, mec="white", mew=1.2, label=name)
        for ref, ls in (("judge alone", "--"), ("debater alone", ":")):
            if ref in refs:
                ax.axhline(refs[ref], color=INK2, lw=1, ls=ls)
                ax.text(16, refs[ref] + 0.01, ref, color=INK2, fontsize=7.5, ha="right", va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("best-of-n against the judge's reward", color=INK2, fontsize=8.5)
        ax.set_title(LABEL[m], color=INK, fontsize=9, loc="left")
    axes[0][0].set_ylabel("judge accuracy (mean of arms;\npooled over 6 pairs; 95% CI)", color=INK2, fontsize=8.5)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "decisive_optimization.png", dpi=170)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ main

def md(df: pd.DataFrame, index: bool = False) -> str:
    """A markdown table (without pandas' optional tabulate dependency)."""
    df = df.reset_index() if index else df

    def cell(x):
        return f"{x:.3f}" if isinstance(x, float) else str(x)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rows = ["| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, "|" + "---|" * len(df.columns), *rows])


def fmt_ci(r, col, pct=False):
    if not np.isfinite(r.get(col, np.nan)):
        return "-"
    f = (lambda x: f"{100 * x:+.0f}") if pct else (lambda x: f"{x:.2f}")
    return f"{f(r[col])} [{f(r[col + '_lo'])}, {f(r[col + '_hi'])}]"


def _found(path: Path) -> Path:
    """A stage output, or its gzipped copy (the committed data bundle, ``docs/results/decisive_data``)."""
    gz = path.with_name(path.name + ".gz")
    return gz if not path.exists() and gz.exists() else path


def main(run: str, out: str = "docs/results") -> None:
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(_found(run / "protocols" / "rows.csv"))
    it = with_references(item_table(rows))
    per_pair = summarize(it, ["judge", "debater", "mechanism"])
    pooled = summarize(it, ["debater", "mechanism"])
    overall = summarize(it, ["mechanism"])
    per_pair.to_csv(out / "decisive_per_pair.csv", index=False)
    pd.concat([pooled, overall.assign(debater="all")]).to_csv(out / "decisive_pooled.csv", index=False)

    lines = ["# Decisive study: tables", ""]
    spend = json.loads((run / "spend.json").read_text()) if (run / "spend.json").exists() else {}
    if spend:
        lines += ["## Spend", "", "| stage | $ |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in spend.items()]
        lines += [f"| **total** | **{sum(spend.values()):.2f}** |", ""]

    # gap: selection draws vs fresh draws
    sel = json.loads((run / "select" / "selected.json").read_text())
    ran = set(rows.item)
    cand = pd.read_csv(_found(run / "select" / "candidates.csv")).set_index("item")
    cand = cand.loc[cand.index.intersection(list(ran))]
    ep = lead_rows(rows)
    gap_rows = []
    for j in JUDGES:
        fresh = ep[(ep.mechanism == "direct") & (ep.judge == j)].groupby("item").correct.mean()
        gap_rows.append({"model": j, "role": "judge", "selection draws": cand[f"{j}_acc"].mean(),
                         "fresh draws": fresh.mean(), "fresh CI": boot(fresh.rename("x").reset_index(), mean_of("x"))})
    for d in DEBATERS:
        fresh = ep[(ep.mechanism == "direct_strong") & (ep.debater == d)].groupby("item").correct.mean()
        gap_rows.append({"model": d, "role": "debater", "selection draws": cand[f"{d}_acc"].mean(),
                         "fresh draws": fresh.mean(), "fresh CI": boot(fresh.rename("x").reset_index(), mean_of("x"))})
    gap = pd.DataFrame(gap_rows)
    gap.to_csv(out / "decisive_gap.csv", index=False)
    lines += ["## Items and the gap achieved", "",
              f"{sel['candidates']} candidates; {sel['selected']} selected ({sel['rule']} rule: every debater 2/2, "
              f"judges pooled <= {sel['judge_max']:.2f}); {len(ran)} run through the protocols.", "",
              "| model | role | accuracy on the draws used to select | fresh draws [95% CI] |", "|---|---|---|---|"]
    for r in gap.itertuples():
        lines.append(f"| {r.model} | {r.role} | {r._3:.2f} | {r._4:.2f} [{r._5[1]:.2f}, {r._5[2]:.2f}] |")
    fams = pd.Series({i: f for i, f in cand.family.items()}).value_counts()
    lines += ["", "Families of the items run: " + ", ".join(f"{k} {v}" for k, v in fams.items()), ""]

    def table(df, title):
        t = [f"## {title}", "", "| judge | debater | protocol | accuracy | judge alone | debater alone | delta (pts) "
             "[CI] | gap recovered [CI] | IGR [CI] | ASD | concordance | items |", "|" + "---|" * 12]
        for r in df.sort_values([c for c in ("judge", "debater") if c in df] + ["mechanism"]).to_dict("records"):
            t.append(f"| {r.get('judge', 'all')} | {r.get('debater', 'all')} | {LABEL.get(r['mechanism'], r['mechanism'])} "
                     f"| {fmt_ci(r, 'accuracy')} | {r['judge_alone']:.2f} | {r['debater_alone']:.2f} "
                     f"| {fmt_ci(r, 'delta', True)} | {fmt_ci(r, 'gap_recovered')} | {fmt_ci(r, 'igr')} "
                     f"| {r['asd']:.2f} | {r['concordance']:.2f} | {r['items']} |")
        return t + [""]

    lines += table(overall, "Prompted liars: pooled over all six (judge, debater) pairs")
    lines += table(pooled, "Prompted liars: per debater, pooled over judges")
    lines += table(per_pair, "Prompted liars: per (judge, debater) pair")

    # the primary auditor is gpt-5 where its verdicts exist (validated against Claude Sonnet 5, decisive_auditcheck.py);
    # gpt-4.1-mini's verdicts give the robustness row
    audit_files = {name: _found(run / "audit" / f) for name, f in (("gpt-5", "verdicts_gpt-5.jsonl"),
                                                                  ("gpt-4.1-mini", "verdicts.jsonl"))}
    audit_files = {k: v for k, v in audit_files.items() if v.exists()}
    pooled_c = None
    if audit_files:
        audit_rows = pd.read_csv(run / "audit" / "rows.csv.gz")
        robustness = []
        for name, f in list(audit_files.items())[1:]:
            v = audit_rows.merge(pd.read_json(f, lines=True).drop_duplicates("key", keep="last"), on="key", how="left")
            alt = summarize(with_references(item_table(committed_filter(rows, v))), ["mechanism"])
            robustness.append(alt.assign(auditor=name))
            audit_rates(v).to_csv(out / f"decisive_audit_rates_{name}.csv", index=False)
        primary, vf = next(iter(audit_files.items()))
        verdicts = audit_rows.merge(pd.read_json(vf, lines=True).drop_duplicates("key", keep="last"), on="key",
                                    how="left")
        rates = audit_rates(verdicts)
        rates.to_csv(out / "decisive_audit_rates.csv", index=False)
        lines += [f"## Audit of prompted liars (distinct transcripts; auditor {primary})", "", md(rates), ""]
        kept = committed_filter(rows, verdicts)
        itc = with_references(item_table(kept))
        # the references are compared on the items that keep at least one sample of the protocol
        pooled_c = summarize(itc, ["debater", "mechanism"])
        overall_c = summarize(itc, ["mechanism"])
        per_pair_c = summarize(itc, ["judge", "debater", "mechanism"])
        pd.concat([per_pair_c, pooled_c.assign(judge="all"), overall_c.assign(judge="all", debater="all")]).to_csv(
            out / "decisive_committed.csv", index=False)
        lines += table(overall_c, f"Committed liars only (auditor {primary}): pooled over all pairs")
        for alt in robustness:
            lines += table(alt, f"Committed liars only, auditor {alt.auditor.iloc[0]} (robustness): pooled")
        lines += table(pooled_c, "Committed liars only: per debater")
        lines += table(per_pair_c, "Committed liars only: per pair")
        sel_path = run / "optimize" / "selection.csv.gz"
        if sel_path.exists():
            sa = selected_audit(pd.read_csv(sel_path), verdicts)
            sa.to_csv(out / "decisive_optimized_audit.csv", index=False)
            lines += ["## Audit of optimized liars (share of selected liar arguments, weighted by selection)", "",
                      md(sa), ""]
    figure_protocols(pooled, pooled_c if pooled_c is not None else pooled, out)

    cf = run / "optimize" / "curves.csv"
    if cf.exists():
        cur = pd.read_csv(cf)
        cur.to_csv(out / "decisive_optimization_curves.csv", index=False)
        ci_path = run / "optimize" / "curve_items.csv.gz"
        opt_items = set(pd.read_csv(ci_path, usecols=["item"]).item) if ci_path.exists() else None
        refs = {}
        if opt_items:
            sub = it[it.item.isin(opt_items)].drop_duplicates(["judge", "debater", "item"])
            refs = {"judge alone": float(sub.acc_judge.mean()), "debater alone": float(sub.acc_debater.mean())}
        # pooled over pairs: an item's accuracy is its mean over the pairs, CIs bootstrap items
        pooled_cur = cur[(cur.debater == "all") & (cur.judge == "all")].drop(columns=["debater", "judge"])
        pooled_cur.to_csv(out / "decisive_optimization_pooled.csv", index=False)
        figure_optimization(pooled_cur, refs, out)
        pairs = cur[cur.debater != "all"]
        if ci_path.exists():
            ci = pd.read_csv(ci_path)
            vs = pd.concat([optimized_vs_references(ci, it, []).assign(debater="all"),
                            optimized_vs_references(ci, it, ["debater"])])
            vs.to_csv(out / "decisive_optimization_vs_judge.csv", index=False)
            t = ["## Optimized liars against the judge alone (paired by item; pooled over judges)", "",
                 "| debater | protocol | optimized | n | accuracy | judge alone | delta (pts) [CI] | gap recovered [CI] "
                 "| items |", "|" + "---|" * 9]
            for r in vs.sort_values(["debater", "mechanism", "mode", "n"]).to_dict("records"):
                t.append(f"| {r['debater']} | {LABEL.get(r['mechanism'], r['mechanism'])} | {r['mode']} | {r['n']} "
                         f"| {fmt_ci(r, 'accuracy')} | {r['judge_alone']:.2f} | {fmt_ci(r, 'delta', True)} "
                         f"| {fmt_ci(r, 'gap_recovered')} | {r['items']} |")
            lines += t + [""]
        piv = pairs.pivot_table(index=["mechanism", "mode", "debater", "judge"], columns="n", values="accuracy")
        lines += ["## Optimized liars: judge accuracy under best-of-n (mean of arms)", "",
                  f"References on these items: {json.dumps({k: round(v, 3) for k, v in refs.items()})}", "",
                  md(pooled_cur), "", md(piv, index=True), ""]
    (out / "decisive.md").write_text("\n".join(lines))
    print("\n".join(lines[:80]))


if __name__ == "__main__":
    main(*sys.argv[1:])
