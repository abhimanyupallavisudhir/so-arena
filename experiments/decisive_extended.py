"""Decisive study, extended report: the stages added after the first write-up.

Reads whatever stage outputs exist under the run directory and writes ``docs/results/decisive_extended.md`` plus
CSVs and figures:

* multi-round best-of-n (``decisive_multiround.py``): judge accuracy and ASD at n = 1 (the protocol grid), 4 and
  16, liar-only and both optimized, paired against the judge alone on the same items;
* held-out best-of-n (``decisive_ic.py --holdout``): the one-round curves and incentives with selection and
  evaluation on independent judge samples, against the in-sample ones;
* prompt search (``decisive_promptsearch.py``): honesty margins and the best-response ladder;
* QuALITY (``decisive_quality.py``): the protocol comparison under information asymmetry;
* the auditor check (``decisive_auditcheck.py``) and the tournament (``decisive_tournament_report.py``).

    python experiments/decisive_extended.py runs/decisive [docs/results]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decisive_report import (INK, INK2, LABEL, SERIES, _axes, _found, boot, diff_of, fmt_ci, item_table,  # noqa: E402
                             mean_of, md, ratio_of, with_references)

N_BOOT = 2000


def references(run: Path) -> pd.DataFrame:
    """Per (judge, debater, item): the judge alone, the debater alone, and every grid protocol's accuracy."""
    rows = pd.read_csv(_found(run / "protocols" / "rows.csv"))
    return with_references(item_table(rows))


def multiround_items(run: Path, ref: pd.DataFrame) -> pd.DataFrame:
    """Per (judge, debater, protocol, mode, n, item): accuracy (mean over arms and samples) and ASD (log score:
    reward arguing the truth minus arguing the falsehood). n = 1 is the protocol grid on the same items;
    consultancy's liar mode takes its true arm from the grid."""
    mr = pd.read_csv(run / "multiround" / "rows.csv")
    grid = pd.read_csv(_found(run / "protocols" / "rows.csv"))
    for f in (mr, grid):  # pandas reads a column of "true"/"false" as booleans
        f["arm"] = f.arm.astype(str).str.lower()
    grid = grid[grid.mechanism.isin(["consultancy", "debate_seq"]) & grid.debater.isin(mr.debater.unique())
                & grid.item.isin(mr.item.unique())]
    out = []
    # n = 1: the grid (both modes coincide)
    for (j, d, m, it), g in grid.groupby(["judge", "debater", "mechanism", "item"]):
        lead = g.sort_values("role").drop_duplicates("episode")
        hon = g[g.value > 0].reward.mean()
        lie = g[g.value <= 0].reward.mean()
        for mode in ("liar", "both"):
            out.append({"judge": j, "debater": d, "mechanism": m, "mode": mode, "n": 1, "item": it,
                        "accuracy": lead.correct.mean(), "asd": hon - lie})
    for (j, d, m, mode, n, it), g in mr.groupby(["judge", "debater", "mechanism", "mode", "n", "item"]):
        acc = list(g.correct)
        hon, lie = list(g.honest_reward.dropna()), list(g.liar_reward.dropna())
        if m == "consultancy" and mode == "liar":  # the unoptimized true arm is the grid's
            gg = grid[(grid.judge == j) & (grid.debater == d) & (grid.mechanism == m) & (grid.item == it)
                      & (grid.arm == "true")]
            acc = [float(np.mean(acc)), float(gg.correct.mean())]
            hon = list(gg.reward.dropna())
        elif m == "consultancy":
            hon = list(g[g.arm == "true"].honest_reward.dropna())
            lie = list(g[g.arm == "false"].liar_reward.dropna())
        out.append({"judge": j, "debater": d, "mechanism": m, "mode": mode, "n": int(n), "item": it,
                    "accuracy": float(np.mean(acc)), "asd": (np.mean(hon) - np.mean(lie)) if hon and lie else np.nan})
    df = pd.DataFrame(out)
    refs = ref.drop_duplicates(["judge", "debater", "item"])[["judge", "debater", "item", "acc_judge", "acc_debater"]]
    return df.merge(refs, on=["judge", "debater", "item"], how="inner")


def audit_verdicts(run: Path) -> pd.DataFrame | None:
    """Every audited liar row with its gpt-5 verdict."""
    vf, rf = run / "audit" / "verdicts_gpt-5.jsonl", run / "audit" / "rows.csv.gz"
    if not (vf.exists() and rf.exists()):
        return None
    v = pd.read_json(vf, lines=True).drop_duplicates("key", keep="last")[["key", "verdict"]]
    r = pd.read_csv(rf, low_memory=False)
    if "arm" in r:
        r["arm"] = r.arm.astype(str).str.lower()
    return r.merge(v, on="key", how="inner")


def multiround_audit(run: Path, ref: pd.DataFrame) -> pd.DataFrame | None:
    """Liar commitment under per-speech best-of-n, and accuracy on the episodes whose liar stayed committed."""
    v = audit_verdicts(run)
    if v is None or not (v.kind == "multiround").any():
        return None
    mr = pd.read_csv(run / "multiround" / "rows.csv")
    vm = v[v.kind == "multiround"][["episode", "verdict"]].drop_duplicates("episode")
    m = mr.merge(vm, on="episode", how="inner")
    refs = ref.drop_duplicates(["judge", "debater", "item"])[["judge", "debater", "item", "acc_judge"]]
    out = []
    for (mech, mode, n), g in m.groupby(["mechanism", "mode", "n"]):
        c = g[g.verdict == "committed"].merge(refs, on=["judge", "debater", "item"])
        out.append({"mechanism": mech, "mode": mode, "n": n, "liar_committed": float((g.verdict == "committed").mean()),
                    "liar_conceded": float((g.verdict == "conceded").mean()), "episodes": len(g),
                    "false_arm_accuracy_committed": float(c[c.arm.astype(str).str.lower() == "false"].correct.mean())
                    if mech == "consultancy" else float(c.correct.mean()),
                    "judge_alone_same_items": float(c.acc_judge.mean())})
    return pd.DataFrame(out)


def summarize_multiround(items: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = []
    for key, g in items.groupby(by + ["mechanism", "mode", "n"]):
        key = dict(zip(by + ["mechanism", "mode", "n"], key if isinstance(key, tuple) else (key,)))
        g = g.dropna(subset=["accuracy", "acc_judge"])
        row = dict(key)
        for name, stat in (("accuracy", mean_of("accuracy")), ("delta", diff_of("accuracy", "acc_judge")),
                           ("gap_recovered", ratio_of("accuracy", "acc_judge", "acc_debater", "acc_judge")),
                           ("asd", mean_of("asd"))):
            est, lo, hi = boot(g.dropna(subset=["asd"]) if name == "asd" else g, stat, n_boot=N_BOOT)
            row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
        row["judge_alone"] = float(g.acc_judge.mean())
        row["items"] = int(g.item.nunique())
        out.append(row)
    return pd.DataFrame(out)


def figure_multiround(pooled: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, m in zip(axes, ("consultancy", "debate_seq")):
        _axes(ax)
        for k, (mode, name) in enumerate((("liar", "liar optimized"), ("both", "both sides optimized"))):
            g = pooled[(pooled.mechanism == m) & (pooled["mode"] == mode)].sort_values("n")
            ax.fill_between(g.n, g.delta_lo, g.delta_hi, color=SERIES[k], alpha=0.15, lw=0)
            ax.plot(g.n, g.delta, "o-", color=SERIES[k], lw=2, ms=6, mec="white", label=name)
        ax.axhline(0, color=INK2, lw=1)
        ax.text(1, 0.005, "judge alone", color=INK2, fontsize=7.5, va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 16], ["1", "4", "16"])
        ax.set_xlabel("best-of-n on every speech (fresh preview judge)", color=INK2, fontsize=8.5)
        ax.set_title(f"{LABEL[m]}, 3 rounds", color=INK, fontsize=9, loc="left")
    axes[0].set_ylabel("accuracy minus the judge alone\n(pooled over 3 judges; 95% CI)", color=INK2, fontsize=8.5)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "decisive_multiround.png", dpi=170)
    plt.close(fig)


def figure_holdout(ins: pd.DataFrame, out_: pd.DataFrame, judge_alone: float, out: Path) -> None:
    """One-round best-of-n: judge accuracy selected and scored on the same judge sample vs on independent ones."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    for ax, m in zip(axes, ("consultancy_1r", "debate_1r")):
        _axes(ax)
        for k, (mode, name) in enumerate((("liar", "liar optimized"), ("both", "both optimized"))):
            for df, ls, tag in ((ins, ":", "in-sample"), (out_, "-", "held-out")):
                g = df[(df.debater == "all") & (df.judge == "all") & (df.kind == "bon") & (df.mechanism == m)
                       & (df["mode"] == mode)].sort_values("level")
                ax.plot(g.level, g.accuracy, "o" + ls, color=SERIES[k], lw=2 if tag == "held-out" else 1.2, ms=5,
                        mec="white", alpha=1 if tag == "held-out" else 0.6, label=f"{name}, {tag}")
                if tag == "held-out":
                    ax.fill_between(g.level, g.accuracy_lo, g.accuracy_hi, color=SERIES[k], alpha=0.12, lw=0)
        ax.axhline(judge_alone, color=INK2, lw=1, ls="--")
        ax.text(16, judge_alone + 0.005, "judge alone", color=INK2, fontsize=7.5, ha="right", va="bottom")
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
        ax.set_xlabel("best-of-n against the judge's reward", color=INK2, fontsize=8.5)
        ax.set_title(LABEL[m], color=INK, fontsize=9, loc="left")
    axes[0].set_ylabel("judge accuracy (pooled; 95% CI held-out)", color=INK2, fontsize=8.5)
    axes[0].legend(fontsize=6.8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "decisive_holdout.png", dpi=170)
    plt.close(fig)


def figure_quality(perj: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mechs = ["consultancy", "debate_seq", "debate_seq_noquotes"]
    names = {"consultancy": "consultancy", "debate_seq": "debate, quotes verified", "debate_seq_noquotes": "debate, no verification"}
    debs = sorted(perj.debater.unique())
    judges = sorted(perj.judge.unique())
    fig, axes = plt.subplots(1, len(debs), figsize=(4.6 * len(debs), 3.4), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], debs):
        _axes(ax)
        for k, m in enumerate(mechs):
            g = perj[(perj.debater == d) & (perj.mechanism == m)].set_index("judge").reindex(judges)
            x = np.arange(len(judges)) + (k - 1) * 0.25
            ax.bar(x, g.delta, width=0.23, color=SERIES[k], label=names[m])
            ax.vlines(x, g.delta_lo, g.delta_hi, color=INK2, lw=1)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(np.arange(len(judges)), judges, fontsize=7.5)
        ax.set_title(f"QuALITY, debater {d}", color=INK, fontsize=9, loc="left")
    axes[0][0].set_ylabel("accuracy minus the judge alone (95% CI)", color=INK2, fontsize=8.5)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "decisive_quality.png", dpi=170)
    plt.close(fig)


def judge_strength(run: Path, out: Path) -> pd.DataFrame | None:
    """Every judge (weak and strong) on the same protocols, per dataset: accuracy, the judge alone, and delta."""
    from decisive_report import summarize

    f = run / "judges" / "rows.csv"
    if not f.exists():
        return None
    strong = pd.read_csv(f)
    res = []
    for ds, src in (("testbed", run / "protocols" / "rows.csv"), ("quality", run / "quality" / "rows.csv")):
        base = pd.read_csv(_found(src))
        st = strong[strong.dataset == ds].drop(columns="dataset")
        if st.empty:
            continue
        items = set(st.item)
        mechs = set(st.mechanism) - {"direct"}
        weak = base[base.item.isin(items) & (base.mechanism.isin(mechs | {"direct"}))]
        refs = base[(base.mechanism == "direct_strong") & base.item.isin(items)]
        frame = pd.concat([weak, st, refs], ignore_index=True)
        t = summarize(with_references(item_table(frame)), ["judge", "debater", "mechanism"], n_boot=1000)
        res.append(t.assign(dataset=ds))
    if not res:
        return None
    df = pd.concat(res, ignore_index=True)
    df.to_csv(out / "decisive_judge_strength.csv", index=False)
    figure_judge_strength(df, out)
    return df


def figure_judge_strength(df: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [d for d in ("testbed", "quality") if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), sharey=True, squeeze=False)
    names = {"consultancy": "consultancy", "debate_seq": "debate (sequential)", "debate_sim": "debate (simultaneous)",
             "debate_seq_noquotes": "debate, no verification"}
    for ax, ds in zip(axes[0], datasets):
        _axes(ax)
        g = df[df.dataset == ds]
        for k, m in enumerate(sorted(set(g.mechanism))):
            h = g[g.mechanism == m].groupby("judge")[["judge_alone", "delta"]].mean().sort_values("judge_alone")
            ax.plot(h.judge_alone, h.delta, "o-", color=SERIES[k % len(SERIES)], lw=1.5, ms=6, mec="white",
                    label=names.get(m, m))
            if k == 0:
                for j, r in h.iterrows():
                    ax.annotate(j, (r.judge_alone, r.delta), fontsize=6.5, color=INK2, xytext=(3, 4),
                                textcoords="offset points")
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xlabel("judge alone accuracy on these items", color=INK2, fontsize=8.5)
        ax.set_title("testbed (capability gap)" if ds == "testbed" else "QuALITY (hidden story)", color=INK,
                     fontsize=9, loc="left")
    axes[0][0].set_ylabel("accuracy minus the judge alone\n(mean over debaters; prompted liars)", color=INK2,
                          fontsize=8.5)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "decisive_judge_strength.png", dpi=170)
    plt.close(fig)


def table(df: pd.DataFrame, cols: list[str]) -> str:
    t = df.copy()
    for c in ("accuracy", "delta", "gap_recovered", "asd"):
        if c in t and f"{c}_lo" in t:
            t[c] = [fmt_ci(r, c, c == "delta") for r in t.to_dict("records")]
    return md(t[cols])


def promptsearch(run: Path) -> tuple[list[str], dict]:
    base = run / "promptsearch"
    lines, data = [], {}
    if not base.exists():
        return lines, data
    margins, ladders, cands = [], [], []
    for jdir in sorted(p for p in base.iterdir() if p.is_dir()):
        for f in sorted(jdir.glob("open-*.margin.json")):
            m = json.loads(f.read_text())
            mech = f.name.split("open-")[1].split(".margin")[0]
            for basis in ("measured", "directive"):
                r = m[basis]
                margins.append({"judge": jdir.name, "mechanism": mech, "basis": basis, "margin": r["margin"],
                                "ci_low": r["ci_low"], "ci_high": r["ci_high"], "honest_reward": r["honest_reward"],
                                "deceptive_reward": r["deceptive_reward"], "honest_value": r["honest_value"],
                                "deceptive_value": r["deceptive_value"], "n_items": r["n_items"]})
        for f in sorted(jdir.glob("ladder-*.ladder.csv")):
            ladders.append(pd.read_csv(f))
        for f in sorted(jdir.glob("*.candidates.csv")):
            cands.append(pd.read_csv(f))
    if margins:
        mg = pd.DataFrame(margins)
        data["margins"] = mg
        lines += ["## Prompt search: honesty margin (agent chooses its side; held-out items)", "",
                  "margin = held-out reward of the best honest strategy minus the best deceptive one (log score); "
                  "value = 2 x share of episodes defending the true answer - 1.", "",
                  md(mg.round(3)), ""]
    if ladders:
        ld = pd.concat(ladders, ignore_index=True)
        data["ladder"] = ld
        lines += ["## Prompt search: best-response ladder (assigned sides; held-out items)", "", md(ld.round(3)), ""]
    if cands:
        data["candidates"] = pd.concat(cands, ignore_index=True)
    return lines, data


def main(run: str, out: str = "docs/results") -> None:
    run, out = Path(run), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Decisive study: extended results", ""]
    spend = json.loads((run / "spend.json").read_text()) if (run / "spend.json").exists() else {}
    if spend:
        lines += ["## Spend", "", "| stage | $ |", "|---|---|"] + [f"| {k} | {v:.2f} |" for k, v in sorted(spend.items())]
        lines += [f"| **total** | **{sum(spend.values()):.2f}** |", ""]
    ref = references(run)

    if (run / "multiround" / "rows.csv").exists():
        mi = multiround_items(run, ref)
        mi.to_csv(run / "multiround_items.csv.gz", index=False)
        pooled = summarize_multiround(mi, [])
        per_judge = summarize_multiround(mi, ["judge"])
        pooled.to_csv(out / "decisive_multiround.csv", index=False)
        per_judge.to_csv(out / "decisive_multiround_by_judge.csv", index=False)
        figure_multiround(pooled, out)
        cols = ["mechanism", "mode", "n", "accuracy", "judge_alone", "delta", "gap_recovered", "asd", "items"]
        ma = multiround_audit(run, ref)
        if ma is not None:
            ma.to_csv(out / "decisive_multiround_audit.csv", index=False)
        lines += ["## Multi-round best-of-n (3 rounds; preview judge on fresh samples)", "",
                  "Pooled over judges; delta = accuracy minus the judge alone (points); ASD in log score.", "",
                  table(pooled, cols), "", "Per judge:", "", table(per_judge, ["judge"] + cols), ""]
        if ma is not None:
            lines += ["Liars under per-speech best-of-n (gpt-5 audit); accuracy on episodes with a committed liar "
                      "(consultancy: its false arm):", "", md(ma.round(3)), ""]

    for suffix, title in (("", "in-sample"), ("_holdout", "held-out")):
        f = out / f"decisive_ic{suffix}.csv"
        if f.exists():
            ic = pd.read_csv(f)
            p = ic[(ic.debater == "all") & (ic.kind == "bon")][["mechanism", "mode", "level", "accuracy", "asd",
                                                                "asd_lo", "asd_hi", "concordant"]]
            lines += [f"## One-round best-of-n incentives ({title} judging)", "", md(p.round(3)), ""]
            o = out / f"decisive_open_choice{suffix}.csv"
            if o.exists():
                oc = pd.read_csv(o)
                oc = oc[(oc.judge == "all") & (oc.kind == "bon")][["debater", "level", "honest", "honest_lo",
                                                                     "honest_hi", "accuracy"]]
                lines += [f"Consultant choosing its side ({title}): share of selected arguments defending the truth", "",
                          md(oc.round(3)), ""]

    if (out / "decisive_ic.csv").exists() and (out / "decisive_ic_holdout.csv").exists():
        refs = ref[ref.item.isin(pd.read_csv(run / "ic_items.csv.gz", usecols=["item"]).item.unique())]
        figure_holdout(pd.read_csv(out / "decisive_ic.csv"), pd.read_csv(out / "decisive_ic_holdout.csv"),
                       float(refs.drop_duplicates(["judge", "item"]).acc_judge.mean()), out)

    ps_lines, _ = promptsearch(run)
    lines += ps_lines

    if (run / "quality" / "rows.csv").exists():
        q = pd.read_csv(run / "quality" / "rows.csv")
        qi = with_references(item_table(q))
        from decisive_report import summarize

        pooled_q = summarize(qi, ["debater", "mechanism"])
        overall_q = summarize(qi, ["mechanism"]).assign(debater="all")
        perj_q = summarize(qi, ["judge", "debater", "mechanism"])
        qq = pd.concat([overall_q, pooled_q])
        qq.to_csv(out / "decisive_quality.csv", index=False)
        perj_q.to_csv(out / "decisive_quality_by_judge.csv", index=False)
        figure_quality(perj_q, out)
        cols = ["debater", "mechanism", "accuracy", "judge_alone", "debater_alone", "delta", "gap_recovered", "asd",
                "concordance", "items"]
        lines += ["## QuALITY (information asymmetry, verified quotes)", "", table(qq, cols), ""]
        verdicts = audit_verdicts(run)
        if verdicts is not None and (verdicts.kind == "quality").any():
            from decisive_report import committed_filter

            qc = with_references(item_table(committed_filter(q, verdicts, kind="quality")))
            qqc = pd.concat([summarize(qc, ["mechanism"]).assign(debater="all"), summarize(qc, ["debater", "mechanism"])])
            qqc.to_csv(out / "decisive_quality_committed.csv", index=False)
            rates = (verdicts[verdicts.kind == "quality"].drop_duplicates("key").groupby(["debater", "mechanism"])
                     .verdict.apply(lambda v: float((v == "committed").mean())).rename("committed").reset_index())
            rates.to_csv(out / "decisive_quality_audit.csv", index=False)
            lines += ["QuALITY liars committed (gpt-5 audit):", "", md(rates.round(3)), "",
                      "### QuALITY, committed liars only", "", table(qqc, cols), ""]

    js = judge_strength(run, out)
    if js is not None:
        cols = ["dataset", "judge", "debater", "mechanism", "accuracy", "judge_alone", "debater_alone", "delta",
                "gap_recovered", "items"]
        lines += ["## Judge strength: the same debates, weak and strong judges (prompted liars)", "",
                  table(js.sort_values(["dataset", "mechanism", "judge_alone"]), cols), ""]

    for name, title in (("decisive_auditcheck_agreement.csv", "Auditor agreement"),
                        ("decisive_auditcheck_rates.csv", "Share of liar texts not committed, by auditor")):
        if (out / name).exists():
            lines += [f"## {title}", "", md(pd.read_csv(out / name).round(3)), ""]
    if (out / "decisive_tournament.csv").exists():
        t = pd.read_csv(out / "decisive_tournament.csv")
        keep = [c for c in ("model", "elo", "elo_lo", "elo_hi", "alone", "self_play", "consultancy", "win_as_honest",
                            "win_as_liar") if c in t]
        lines += ["## Tournament", "", md(t[keep].round(3)), ""]
        s = out / "decisive_tournament_summary.txt"
        if s.exists():
            lines += [s.read_text(), ""]
    (out / "decisive_extended.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
