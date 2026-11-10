"""Audit the liars of the best-of-16 model panel (``decisive_panel_bon.py``): does optimization pick lies or concessions?

Every candidate in a liar's pool is audited by gpt-5 (the study's auditor) on a random subset of the panel's items.
The pools are shared by the three judges, so each text is audited once. Two shares are reported per agent and
protocol:

* ``n = 1``: every candidate weighted equally, as an unoptimized liar would be drawn;
* ``n = 16``: candidates weighted by their exact best-of-16 selection probability under each judge's reward (the
  liar's reward averaged over the honest side's candidates, as in "liar optimized"), averaged over the judges.

    python experiments/decisive_panel_bon_audit.py --run runs/decisive --items 40 [--estimate]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

from so_arena.analysis.optimization import BestOfN

from decisive_audit import audit_with, key_of
from decisive_common import Ledger, map_keys
from decisive_optimize import liar_role
from decisive_panel_bon import AGENTS, JUDGES, mechanisms, tree_path

STAGE = "audit_panel_bon"
SEED = 7


def liar_pool(tree: dict, liar: str) -> tuple[list[str], np.ndarray]:
    """The liar's candidate texts and, per candidate, its mean reward over the other side's candidates.

    A consultancy tree is one node whose children are leaves. A one-round simultaneous debate is a root
    (debater_a) whose children are debater_b nodes sharing one candidate list, each with one leaf per candidate."""
    nodes, leaves = tree["nodes"], tree["leaves"]
    root = nodes[tree["root"]]

    def reward(leaf_id: str) -> float:
        lf = leaves.get(leaf_id)
        r = (lf or {}).get("rewards", {}).get(liar)
        return np.nan if r is None else float(r)

    if all(c in leaves for c in root["children"]):  # one decision: the liar's own node
        return list(root["candidates"]), np.array([reward(c) for c in root["children"]])
    second = [nodes[c] for c in root["children"]]
    R = np.array([[reward(leaf) for leaf in b["children"]] for b in second])  # R[i, j]: root cand i, second cand j
    with np.errstate(all="ignore"):
        if root["role"] == liar:
            return list(root["candidates"]), np.nanmean(R, axis=1)
        return list(second[0]["candidates"]), np.nanmean(R, axis=0)


def selection_weights(u: np.ndarray, n: int) -> np.ndarray:
    """Exact best-of-n selection probabilities over a pool; candidates without a reward are never selected."""
    ok = np.isfinite(u)
    w = np.zeros(len(u))
    if ok.any():
        w[ok] = BestOfN(n).probs(u[ok])
    return w


def sample_items(run: Path, k: int) -> list[str]:
    f = run / "panel" / "items.jsonl"
    ids = [json.loads(x)["id"] for x in f.read_text().splitlines() if x.strip()][:150]
    return sorted(random.Random(SEED).sample(ids, k))


def pool_rows(run: Path, items: set[str]) -> list[dict]:
    rows = []
    for ag in AGENTS:
        for m in mechanisms():
            for arm in ("true", "false"):
                if m.name.startswith("consultancy") and arm == "true":
                    continue  # the honest consultant
                liar = liar_role(m.name, arm)
                for jd in JUDGES:
                    p = tree_path(run, ag, jd, m.name, arm)
                    if not p.exists():
                        continue
                    with open(p) as fh:
                        for line in fh:
                            t = json.loads(line)
                            if t["item_id"] not in items:
                                continue
                            texts, u = liar_pool(t, liar)
                            w16 = selection_weights(u, 16)
                            for i, text in enumerate(texts):
                                rows.append({"debater": ag, "judge": jd, "mechanism": m.name, "arm": arm,
                                             "item": t["item_id"], "role": liar, "index": i, "w1": 1 / len(texts),
                                             "w16": w16[i], "text": text, "key": key_of(m.name, t["item_id"], liar, text)})
    return rows


def shares(rows: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    """Committed and conceded-or-refused shares at n = 1 and n = 16, per agent and protocol."""
    d = rows.merge(verdicts[["key", "verdict"]], on="key", how="inner")
    d["committed"] = (d.verdict == "committed").astype(float)
    out = []
    for (ag, m), g in d.groupby(["debater", "mechanism"]):
        rec = {"debater": ag, "mechanism": m, "texts": g.key.nunique()}
        for n in (1, 16):
            w = g[f"w{n}"]
            rec[f"committed_n{n}"] = float((g.committed * w).sum() / w.sum())
        out.append(rec)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    map_keys()
    run = Path(a.run)
    from so_arena.core.items import TaskItem

    all_items = {it.id: it for it in (TaskItem.model_validate_json(x) for x in
                                       (run / "panel" / "items.jsonl").read_text().splitlines() if x.strip())}
    rows = pool_rows(run, set(sample_items(run, a.items)))
    vf = run / "audit" / "verdicts_gpt-5_panel_bon.jsonl"
    done = {json.loads(x)["key"] for x in vf.read_text().splitlines()} if vf.exists() else set()
    todo = list({r["key"]: r for r in rows if r["key"] not in done}.values())
    print(f"{len(rows)} weighted rows, {len({r['key'] for r in rows})} distinct texts, {len(todo)} to audit")
    if a.estimate:
        print(f"about ${0.0054 * len(todo):.0f} (measured $0.0054 per text)")
        return
    vf.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(audit_with(todo, all_items, vf, "gpt-5"))
    verdicts = pd.read_json(vf, lines=True).drop_duplicates("key", keep="last")
    Ledger(run).set(STAGE, float(verdicts.cost.sum()))
    frame = pd.DataFrame(rows).drop(columns=["text"])
    s = shares(frame, verdicts)
    out = Path(a.out) / "panel_opt"
    out.mkdir(parents=True, exist_ok=True)
    s.to_csv(out / "audit.csv", index=False)
    frame.merge(verdicts[["key", "verdict"]], on="key", how="left").to_csv(out / "audit_rows.csv.gz", index=False)
    print(s.round(3).to_string(index=False))
    print(f"audit cost ${verdicts.cost.sum():.2f}")


if __name__ == "__main__":
    main()
