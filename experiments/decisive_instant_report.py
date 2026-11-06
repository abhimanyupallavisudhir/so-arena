"""Report of the instant-judge replay (``decisive_instant.py``): EIC and accuracy when the weak judges answer at once.

* ``grid``: instructed EIC (ASD, concordance, r) of consultancy and sequential and simultaneous debate for gpt-5-mini
  on the 927 items, each weak judge answering instantly vs reasoning first (the protocol grid), on the same
  transcripts; all prompted liars and committed liars only (gpt-5 audit: debates' liars are the stored ones,
  consultancy's those of the reasoning judge whose questions were replayed); the blind baseline (judge alone).
* ``trees``: best-of-n (n = 1 ... 16) under each instant judge, liar only and both sides optimized: ASD and judge
  accuracy along n. The instant judge is (near-)deterministic, so the payoff that selects is the one reported
  (``reliability`` measures how near); the reasoning judge's held-out curve is shown alongside.
* ``reliability``: test-retest agreement of two independent instant verdicts on the same transcript.

    python experiments/decisive_instant_report.py --run runs/decisive --out docs/results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from decisive_eic import eic_items, eic_rows, instructed
from decisive_report import committed_filter

PROTOS = ["consultancy", "debate_seq", "debate_sim"]
WEAK = ["gpt-4o-mini", "llama-3.1-8b", "gemma-3-12b"]
DEBATER = "gpt-5-mini"


def instant_rows(run: Path) -> pd.DataFrame:
    frames = [pd.read_csv(f).assign(judge=f.parent.name) for f in sorted((run / "instant" / "grid").glob("*/*.csv"))]
    if not frames:
        return pd.DataFrame()
    x = pd.concat(frames, ignore_index=True)
    return x


def uv(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows[rows.value.notna() & rows.reward.notna()].copy()
    x["u"], x["v"] = x.reward, (x.value > 0).astype(float)
    return x


def committed_instant(rows: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Instant rows whose (replayed) liars were committed: verdicts are keyed by the transcript's source judge."""
    src = np.where(rows.mechanism == "consultancy", rows.judge.str.removesuffix("-lp"), "gpt-4o-mini")
    kept = committed_filter(rows.assign(judge_out=rows.judge, judge=src), verdicts)
    return kept.assign(judge=kept.judge_out).drop(columns="judge_out")


def grid_tables(run: Path) -> dict[str, pd.DataFrame]:
    inst = instant_rows(run)
    if inst.empty:
        return {}
    reason, committed = instructed(run)
    reason = reason[(reason.debater == DEBATER) & reason.mechanism.isin(PROTOS)]
    items = set(inst.item) & set(reason.item)
    out = {}
    pooled = lambda df, by: pd.concat([eic_rows(df[df.judge.isin(WEAK)], by).assign(judge="pooled"),  # noqa: E731
                                       eic_rows(df, ["judge"] + by)])
    li = uv(inst[inst.mechanism.isin(PROTOS) & inst.item.isin(items)])
    lr = reason[reason.item.isin(items)]
    out["eic"] = pd.concat([pooled(li, ["mechanism"]).assign(judging="instant"),
                            pooled(lr, ["mechanism"]).assign(judging="reasoning")])
    if committed is not None:
        from decisive_eic import _found  # noqa: F401  (same verdict file as the EIC report)
        import json  # noqa: F401
        vf = run / "audit" / "verdicts_gpt-5.jsonl"
        ar = pd.read_csv(run / "audit" / "rows.csv.gz", low_memory=False)
        v = ar.merge(pd.read_json(vf, lines=True).drop_duplicates("key", keep="last"), on="key", how="left")
        ci = committed_instant(li, v)
        cr = committed[(committed.debater == DEBATER) & committed.mechanism.isin(PROTOS) & committed.item.isin(items)]
        out["eic_committed"] = pd.concat([pooled(ci, ["mechanism"]).assign(judging="instant"),
                                          pooled(cr, ["mechanism"]).assign(judging="reasoning")])
    # blind baselines and accuracies
    blind_i = uv(inst[(inst.mechanism == "direct") & inst.item.isin(items)])
    blind_r = uv(pd.read_csv(run / "protocols" / "rows.csv").query("mechanism == 'direct'"))
    blind_r = blind_r[blind_r.item.isin(items)]
    out["blind"] = pd.concat([pooled(blind_i, []).assign(judging="instant"),
                              pooled(blind_r, []).assign(judging="reasoning")])
    acc = []
    for judging, df, blind in (("instant", inst[inst.item.isin(items)], blind_i), ("reasoning", lr, blind_r)):
        for j, g in df.groupby("judge"):
            b = blind[blind.judge == j]
            row = {"judging": judging, "judge": j, "alone": g[g.mechanism == "direct"].correct.mean()
                   if (g.mechanism == "direct").any() else b.correct.mean()}
            for m in PROTOS:
                row[m] = g[g.mechanism == m].drop_duplicates("episode").correct.mean()
            acc.append(row)
    out["accuracy"] = pd.DataFrame(acc)
    return out


def tree_tables(run: Path) -> pd.DataFrame:
    """Per-item ASD and accuracy along best-of-n under each instant judge (and the reasoning judges' held-out
    curve for gpt-5-mini)."""
    from decisive_ic import cell_rows
    from decisive_instant import INSTANT

    rows = []
    for j in INSTANT:
        for m in ("consultancy_1r", "debate_1r"):
            rows += [dict(r, judging="instant") for r in cell_rows(run / "instant", DEBATER, j, m)]
    df = pd.DataFrame(rows)
    hold = run / "ic_items_holdout.csv.gz"
    if hold.exists():
        h = pd.read_csv(hold)
        h = h[(h.debater == DEBATER)].assign(judging="reasoning (held out)")
        df = pd.concat([df, h], ignore_index=True)
    return df[df.kind == "bon"] if not df.empty else df


def reliability(run: Path) -> pd.DataFrame:
    out = []
    for f in sorted((run / "instant" / "reliability").glob("*.csv")):
        d = pd.read_csv(f).drop_duplicates(["episode", "draw"])
        w = d.pivot_table(index="episode", columns="draw", values="p_true")
        if w.shape[1] < 2:
            continue
        w = w.dropna()
        out.append({"judge": f.stem, "transcripts": len(w), "identical": float((w[0] - w[1]).abs().lt(1e-9).mean()),
                    "r": float(np.corrcoef(w[0], w[1])[0, 1])})
    return pd.DataFrame(out)


def fmt(r, a="asd") -> str:
    return f"{r[a]:+.2f} [{r[a + '_lo']:+.2f}, {r[a + '_hi']:+.2f}]"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results")
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out) / "instant"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Instant judges: the same transcripts judged without reasoning", ""]
    g = grid_tables(run)
    for name, df in g.items():
        df.to_csv(out / f"{name}.csv", index=False)
    if g:
        for key, title in (("eic", "all prompted liars"), ("eic_committed", "committed liars only")):
            if key not in g:
                continue
            lines += [f"## Instructed EIC, {title} (gpt-5-mini, 927 items)", "",
                      "| judge | protocol | reasoning ASD | instant ASD | reasoning conc | instant conc | reasoning r | instant r |",
                      "|---|---|---|---|---|---|---|---|"]
            e = g[key]
            for (j, m), grp in e.groupby(["judge", "mechanism"], sort=False):
                r = grp.set_index("judging")
                if "instant" not in r.index or "reasoning" not in r.index:
                    continue
                ri, rr = r.loc["instant"], r.loc["reasoning"]
                lines.append(f"| {j} | {m} | {fmt(rr)} | {fmt(ri)} | {rr.concordance:.2f} | {ri.concordance:.2f} | "
                             f"{rr.r:.2f} | {ri.r:.2f} |")
            for j, grp in e.groupby("judge"):
                if j.endswith("-lp"):
                    for _, ri in grp.iterrows():
                        lines.append(f"| {j} | {ri.mechanism} | - | {fmt(ri)} | - | {ri.concordance:.2f} | - | {ri.r:.2f} |")
            lines.append("")
        b = g["blind"]
        lines += ["## Blind baseline (judge alone, log score)", "", "| judge | reasoning ASD | instant ASD |", "|---|---|---|"]
        for j, grp in b.groupby("judge", sort=False):
            r = grp.set_index("judging")
            cells = [fmt(r.loc[k]) if k in r.index else "-" for k in ("reasoning", "instant")]
            lines.append(f"| {j} | {cells[0]} | {cells[1]} |")
        lines += ["", "## Judge accuracy", "", "| judging | judge | alone | " + " | ".join(PROTOS) + " |",
                  "|---|---|---|" + "---|" * len(PROTOS)]
        for _, r in g["accuracy"].iterrows():
            lines.append(f"| {r.judging} | {r.judge} | {r.alone:.2f} | " + " | ".join(f"{r[m]:.2f}" for m in PROTOS) + " |")
        lines.append("")
    t = tree_tables(run)
    if not t.empty:
        t.to_csv(out / "bon_items.csv.gz", index=False)
        s = pd.concat([eic_items(t, ["judging", "judge", "mechanism", "mode", "level"]),
                       eic_items(t[t.judge.isin(WEAK)], ["judging", "mechanism", "mode", "level"]).assign(judge="pooled")])
        accs = t.groupby(["judging", "judge", "mechanism", "mode", "level"]).accuracy.mean().rename("accuracy")
        accp = t[t.judge.isin(WEAK)].groupby(["judging", "mechanism", "mode", "level"]).accuracy.mean().rename(
            "accuracy").reset_index().assign(judge="pooled")
        s = s.merge(pd.concat([accs.reset_index(), accp]), on=["judging", "judge", "mechanism", "mode", "level"])
        s.to_csv(out / "bon.csv", index=False)
        lines += ["## Best-of-n under instant judges (pooled over the three weak judges)", "",
                  "| judging | protocol | optimized | n=1 ASD | n=16 ASD | n=1 acc | n=16 acc |", "|---|---|---|---|---|---|---|"]
        for (jd, m, mode), grp in s[s.judge == "pooled"].groupby(["judging", "mechanism", "mode"]):
            r1, r16 = grp[grp.level == 1], grp[grp.level == 16]
            if r1.empty or r16.empty:
                continue
            r1, r16 = r1.iloc[0], r16.iloc[0]
            lines.append(f"| {jd} | {m} | {mode} | {fmt(r1)} | {fmt(r16)} | {r1.accuracy:.2f} | {r16.accuracy:.2f} |")
        lines.append("")
    rel = reliability(run)
    if not rel.empty:
        rel.to_csv(out / "reliability.csv", index=False)
        lines += ["## Test-retest of instant verdicts (two independent calls, same transcript)", "",
                  "| judge | transcripts | identical | r |", "|---|---|---|---|"]
        for _, r in rel.iterrows():
            lines.append(f"| {r.judge} | {r.transcripts} | {r.identical:.2f} | {r.r:.2f} |")
    (out / "instant.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
