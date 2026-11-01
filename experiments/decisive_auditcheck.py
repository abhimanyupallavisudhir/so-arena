"""Decisive study: checking the liar auditor. Does gpt-4.1-mini's committed / conceded / refused verdict agree with
stronger auditors?

Every result "with conceding liars removed" rests on ``decisive_audit.py``'s auditor (gpt-4.1-mini). This draws a
stratified sample of the audited liar texts - up to ``--per-cell`` per (prompted or optimized, debater model,
verdict) - and has two stronger models audit them blind to the first verdict, with the same prompt: gpt-5 (medium
reasoning effort) and Claude Sonnet 5. It reports agreement and Cohen's kappa for every pair of auditors, the
confusion of gpt-4.1-mini against the two strong auditors where they agree, and concession rates re-estimated
from the strong auditors (each stratum weighted by its size in the full audit).

    python experiments/decisive_auditcheck.py --run runs/decisive [--per-cell 60]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from so_arena.core.store import RunStore
from so_arena.core.types import GenerateOptions, Message
from so_arena.models.base import get_model

from audit_liars import PROMPT
from decisive_audit import optimized_rows, prompted_rows
from decisive_common import Ledger, map_keys, selected_items

STRONG = {"gpt-5": ("openai/gpt-5", "medium"), "claude-sonnet-5": ("anthropic/claude-sonnet-5", "medium")}
VERDICTS = ("committed", "conceded", "refused", "broken")


def kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa of two label series (rows where both are labelled)."""
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    cats = sorted(set(a) | set(b))
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def parse(text: str) -> str:
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    v = str(d.get("verdict", "unparsed")).split("|")[0].strip().lower()
    return v if v in VERDICTS else "unparsed"


async def reaudit(sample: pd.DataFrame, items: dict, out: Path) -> float:
    done = {(json.loads(x)["key"], json.loads(x)["auditor"]) for x in out.read_text().splitlines()} if out.exists() else set()
    sem = asyncio.Semaphore(48)
    cost = 0.0

    async def one(r, name, model, effort):
        nonlocal cost
        it = items[r.item]
        truth = it.true_label
        assigned = r.assigned if isinstance(r.assigned, str) and r.assigned else ("B" if truth == "A" else "A")
        other = "B" if assigned == "A" else "A"
        q = it.question + "\n\n" + "\n".join(f"({a.label}) {a.text}" for a in it.answers)
        prompt = PROMPT.format(assigned=assigned, truth=truth, which="WRONG", question=q,
                               text=r.text[:12000] or "(nothing)", other=other)
        async with sem:
            c = await get_model(model).generate([Message.user(prompt)],
                                                GenerateOptions(max_tokens=8000, reasoning_effort=effort))
        cost += c.usage.cost_usd or 0.0
        with open(out, "a") as f:
            f.write(json.dumps({"key": r.key, "auditor": name, "verdict": parse(c.text), "raw": c.text[:400],
                                "cost": c.usage.cost_usd or 0.0}) + "\n")

    await asyncio.gather(*[one(r, n, m, e) for r in sample.itertuples() for n, (m, e) in STRONG.items()
                           if (r.key, n) not in done])
    return cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--per-cell", type=int, default=60)
    ap.add_argument("--out", default="docs/results")
    a = ap.parse_args(argv)
    map_keys()
    run = Path(a.run)
    adir = run / "audit"
    verdicts = pd.read_json(adir / "verdicts.jsonl", lines=True).drop_duplicates("key", keep="last")
    rows = pd.DataFrame(prompted_rows(run) + optimized_rows(run)).drop_duplicates("key")
    rows = rows.merge(verdicts[["key", "verdict"]], on="key", how="inner")
    strata = rows.groupby(["kind", "debater", "verdict"])
    sizes = strata.size().rename("stratum_size")
    sample = rows.sample(frac=1, random_state=0).groupby(["kind", "debater", "verdict"]).head(a.per_cell)
    sample = sample.reset_index(drop=True)
    items = {it.id: it for it in RunStore(str(run / "select")).items()} or {it.id: it for it in selected_items(run)}
    out = adir / "strong_verdicts.jsonl"
    asyncio.run(reaudit(sample, items, out))
    strong = pd.read_json(out, lines=True).drop_duplicates(["key", "auditor"], keep="last")
    Ledger(run).set("auditcheck", float(strong.cost.sum()))
    wide = strong.pivot(index="key", columns="auditor", values="verdict").reset_index()
    s = sample.merge(wide, on="key", how="left").rename(columns={"verdict": "gpt-4.1-mini"})
    auditors = ["gpt-4.1-mini"] + list(STRONG)
    pairs = [{"auditor_a": x, "auditor_b": y, "agreement": float((s[x] == s[y]).mean()), "kappa": kappa(s[x], s[y]),
              "committed_vs_not_kappa": kappa(s[x] == "committed", s[y] == "committed")}
             for i, x in enumerate(auditors) for y in auditors[i + 1:]]
    s["strong"] = np.where(s["gpt-5"] == s["claude-sonnet-5"], s["gpt-5"], None)
    confusion = pd.crosstab(s["gpt-4.1-mini"], s["strong"].fillna("strong auditors disagree"))
    # concession rates re-estimated: each stratum's strong-auditor rates weighted by the stratum's size in the audit
    s = s.merge(sizes.reset_index().rename(columns={"verdict": "gpt-4.1-mini"}),
                on=["kind", "debater", "gpt-4.1-mini"], how="left")
    est = []
    for (kind, deb), g in s.groupby(["kind", "debater"]):
        w = g.groupby("gpt-4.1-mini").stratum_size.first()
        tot = w.sum()
        row = {"kind": kind, "debater": deb,
               "not_committed_gpt-4.1-mini": float(w[w.index != "committed"].sum() / tot)}
        for name in STRONG:
            per = g.groupby("gpt-4.1-mini")[name].apply(lambda x: float((x != "committed").mean()))
            row[f"not_committed_{name}"] = float((per * w / tot).sum())
        est.append(row)
    res = Path(a.out)
    res.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(res / "decisive_auditcheck_agreement.csv", index=False)
    confusion.to_csv(res / "decisive_auditcheck_confusion.csv")
    pd.DataFrame(est).to_csv(res / "decisive_auditcheck_rates.csv", index=False)
    print(pd.DataFrame(pairs).round(3).to_string(index=False))
    print(confusion)
    print(pd.DataFrame(est).round(3).to_string(index=False))
    print(f"cost ${strong.cost.sum():.2f}, {len(s)} texts")


if __name__ == "__main__":
    main()
