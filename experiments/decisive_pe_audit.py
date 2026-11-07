"""Where prover-estimator debate's reward comes from (docs/oversight-results.md, "Why prover-estimator debate fails
here").

Reads every instructed prover-estimator episode of the protocol grid (gpt-5-mini as prover and estimator, the three
weak judges) and splits the prover's reward into its terms: the start term (r^(d+1) x the estimator's probability
for the prover's answer), the decomposition terms and the leaf term (direction x (judge - estimator) on the final
subclaim). Also the prover's direction calls and the leaf term it would have earned with a fixed direction.

    python experiments/decisive_pe_audit.py --run runs/decisive --out docs/results/eic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from so_arena.core.store import RunStore

JUDGES = ["gpt-4o-mini", "llama-3.1-8b", "gemma-3-12b"]


def episode_row(e) -> dict | None:
    d = e.outcome.data if e.outcome else {}
    t = d.get("reward_terms")
    if e.error or not t:
        return None
    path, leaf = d["path"], d["leaf"]
    return {"item": e.item_id, "side": "honest" if e.profile.endswith("true") else "liar",
            "reward": d["rewards"]["prover"], "start": t[0], "decomposition": sum(t[1:-1]), "leaf": t[-1],
            "p_answer": d["p_top"], "p_leaf_estimator": leaf["p_estimator"], "p_leaf_judge": leaf["p_judge"],
            "dir_leaf": leaf["direction"], "dir_rounds": [lvl["direction"] for lvl in path]}


def rows(run: Path, debater: str = "gpt-5-mini") -> pd.DataFrame:
    out = []
    for root in ("protocols", "protocols_ext"):
        for j in JUDGES:
            p = run / root / debater / j / "prover_estimator"
            if not p.exists():
                continue
            for e in RunStore(str(p)).episodes():
                r = episode_row(e)
                if r:
                    out.append({"judge": j, **r})
    return pd.DataFrame(out)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    diff = df.p_leaf_judge - df.p_leaf_estimator
    df = df.assign(leaf_if_too_high=-diff, leaf_if_too_low=diff,
                   leaf_too_low_share=(df.dir_leaf == 1).astype(float),
                   leaf_too_high_share=(df.dir_leaf == -1).astype(float))
    cols = ["p_answer", "start", "decomposition", "p_leaf_estimator", "p_leaf_judge", "leaf", "reward",
            "leaf_too_low_share", "leaf_too_high_share", "leaf_if_too_high", "leaf_if_too_low"]
    pooled = df.groupby("side")[cols].mean().assign(judge="all")
    per_judge = df.groupby(["judge", "side"])[cols].mean().reset_index(level="side")
    return pd.concat([pooled.reset_index(), per_judge.reset_index()])[["judge", "side", *cols]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results/eic")
    a = ap.parse_args(argv)
    df = rows(Path(a.run))
    out = summary(df)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(a.out) / "pe_reward_terms.csv", index=False)
    print(f"{len(df)} episodes")
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
