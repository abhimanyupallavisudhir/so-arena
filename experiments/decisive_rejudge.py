"""Decisive study: held-out judging of the best-of-n trees (winner's-curse check).

``decisive_optimize.py`` selects each candidate with the judge's verdict on it and reports that same verdict, so
best-of-n is partly selecting judge noise (a winner's curse that overstates what optimization buys). This regrows
every tree with the same agent candidates (replayed from the optimization cache at no cost) and a judge drawing
*fresh* samples (``FreshSamples``), writing the trees to ``<run>/optimize_holdout/``. ``decisive_ic.py --holdout``
and ``decisive_optimize.py --holdout`` then select on the original verdicts and evaluate on the fresh ones.

    python experiments/decisive_rejudge.py --run runs/decisive [--estimate]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import so_arena as soa
from so_arena.core.runner import build_players, run_sync
from so_arena.samplers.arms import arm_profiles
from so_arena.samplers.pools import expand_tree, load_trees

from decisive_common import (DEBATERS, JUDGES, FreshSamples, Ledger, chunks, debater_policy, judge_policy, map_keys,
                             selected_items)
from decisive_optimize import GT, mechanisms, tree_path


def holdout_path(run: Path, d: str, j: str, mech: str, arm: str) -> Path:
    return run / "optimize_holdout" / d / j / f"{mech}-{arm}.jsonl"


async def regrow(run: Path, items, d: str, j: str, mech, *, pool: int, concurrency: int, estimate: bool) -> float:
    cost = 0.0
    sem = asyncio.Semaphore(concurrency)
    sizes = {"consultant": pool} if mech.name.startswith("consultancy") else {"debater_a": pool, "debater_b": pool}
    profs = arm_profiles(mech, agent=debater_policy(DEBATERS[d]),
                         fixtures={"judge": FreshSamples(judge_policy(JUDGES[j]))})

    async def grow_arm(prof):
        arm = prof.tags["arm"]
        src = tree_path(run, d, j, mech.name, arm)
        have = {t.item_id for t in load_trees(src)} if src.exists() else set()
        path = holdout_path(run, d, j, mech.name, arm)
        done = {t.item_id for t in load_trees(path)} if path.exists() else set()
        path.parent.mkdir(parents=True, exist_ok=True)

        async def one(item):
            nonlocal cost
            async with sem:
                tree, _ = await expand_tree(mech, item, build_players(prof, item), pool_sizes=sizes,
                                            ground_truth=GT, concurrency=16, profile=prof.name)
            cost += sum((u.get("cost_usd") or 0.0) for u in tree.usage.values())
            if not estimate:
                with open(path, "a") as f:
                    f.write(tree.model_dump_json() + "\n")

        await asyncio.gather(*[one(it) for it in items if it.id in have and it.id not in done])

    await asyncio.gather(*[grow_arm(p) for p in profs])
    return cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--items", type=int, default=150)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    # the agents' candidates replay from the optimization cache; only the fresh judge samples are new calls
    soa.configure(cache_dir=str(run / "cache" / "optimize"), simulate=a.estimate or None)
    items = selected_items(run, a.items)
    ledger = Ledger(run)
    cost_path = run / "optimize_holdout" / "cost.json"
    cost = json.loads(cost_path.read_text()) if cost_path.exists() else {}
    log = logging.getLogger("decisive")
    for batch in chunks(items, a.chunk):
        jobs = [(d, j, m) for d in DEBATERS for j in JUDGES for m in mechanisms()]
        async def gather(batch=batch, jobs=jobs):
            return await asyncio.gather(*[regrow(run, batch, d, j, m, pool=a.pool, concurrency=a.concurrency,
                                                 estimate=a.estimate) for d, j, m in jobs])

        got = run_sync(gather())
        for (d, j, m), c in zip(jobs, got):
            cost[f"{d}|{j}|{m.name}"] = cost.get(f"{d}|{j}|{m.name}", 0.0) + c
        if not a.estimate:
            cost_path.parent.mkdir(parents=True, exist_ok=True)
            cost_path.write_text(json.dumps(cost, indent=1))
            ledger.set("rejudge", sum(cost.values()))
        log.info("batch done, stage $%.2f", sum(cost.values()))
    print(json.dumps({k: round(v, 2) for k, v in cost.items()}, indent=1), f"total ${sum(cost.values()):.2f}")


if __name__ == "__main__":
    main()
