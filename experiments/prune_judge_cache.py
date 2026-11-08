"""Free disk under a running stage by deleting judge verdicts from its response cache, keeping the spend exact.

A stage's response cache holds every paid completion. The agents' candidate arguments must stay (a held-out tree
replays them to rebuild the same leaves), but a judge verdict is only ever replayed when an interrupted tree is
regrown. Judge verdicts end with the verdict JSON (``{"A": 0.4, "B": 0.6}``); arguments never do. This deletes those
rows in small transactions (so the running job is not locked out) and adds their recorded cost to
``spend.json[<stage>_pruned]``, so ``spend[stage] + spend[stage_pruned]`` stays the stage's exact spend. SQLite reuses
the freed pages, so the cache stops growing until they are used up.

    python experiments/prune_judge_cache.py --run runs/decisive --stage panel_bon
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path

from decisive_common import Ledger

VERDICT = re.compile(r'\{\s*"[A-Z]"\s*:\s*[0-9.]+\s*(,\s*"[A-Z]"\s*:\s*[0-9.]+\s*)+\}\s*$')


def is_verdict(text: str | None) -> bool:
    return bool(text) and bool(VERDICT.search(text))


def prune(db: Path, *, batch: int = 5000, pause: float = 0.2) -> tuple[int, float]:
    con = sqlite3.connect(db, timeout=60)
    con.create_function("is_verdict", 1, lambda t: int(is_verdict(t)))
    deleted, cost, last = 0, 0.0, ""
    while True:
        rows = con.execute("SELECT key, json_extract(value, '$.text'), json_extract(value, '$.usage.cost_usd') "
                           "FROM cache WHERE key > ? ORDER BY key LIMIT ?", (last, batch)).fetchall()
        if not rows:
            break
        last = rows[-1][0]
        drop = [(k, c or 0.0) for k, t, c in rows if is_verdict(t)]
        if drop:
            with con:
                con.executemany("DELETE FROM cache WHERE key = ?", [(k,) for k, _ in drop])
            deleted += len(drop)
            cost += sum(c for _, c in drop)
        time.sleep(pause)
    con.close()
    return deleted, cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--stage", required=True)
    a = ap.parse_args(argv)
    run = Path(a.run)
    n, usd = prune(run / "cache" / a.stage / "responses.sqlite")
    ledger = Ledger(run)
    ledger._reload()
    key = f"{a.stage}_pruned"
    ledger.set(key, ledger.data.get(key, 0.0) + usd)
    print(f"deleted {n} judge verdicts costing ${usd:.2f}; {key} = ${ledger.data[key]:.2f}")


if __name__ == "__main__":
    main()
