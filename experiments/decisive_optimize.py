"""Decisive study, stage 3: optimized liars. Best-of-n arguments selected against each weak judge's reward.

For a single-round consultancy (the consultant writes one argument for its assigned answer; the judge reads it)
and a single-round simultaneous debate, the debater model samples a pool of 16 candidate speeches for every
agent decision, the judge scores every combination (16 leaves for consultancy, 16 x 16 for debate), and best-of-n
selection against the judge's log-score reward is computed exactly over the pools (``samplers.pools``:
``expand_tree`` and ``optimization_grid``, the machinery of ``OptimizationExperiment``). Curves over n = 1, 2,
4, 8, 16:

* ``liar``: only the side arguing the false answer is optimized (the honest side stays at n = 1);
* ``both``: both sides are optimized at the same n (consultancy: the honest consultant on the true arm too).

Each point is the judge's accuracy averaged over the two arms of an item, with a bootstrap CI over items. Pools
do not depend on the judge (the agents never see it), so the first judge pays for them and the others replay
them from the cache. Every candidate speech is kept (``candidates.jsonl``) and the liar's selection weights at
each n are written to ``selection.csv`` for the audit (``decisive_audit.py``).

    python experiments/decisive_optimize.py --run runs/decisive --items 150 --estimate
    python experiments/decisive_optimize.py --run runs/decisive --items 150
    python experiments/decisive_optimize.py --run runs/decisive --curves-only   # recompute the curves
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import so_arena as soa
from so_arena.analysis.metrics import bootstrap_mean_ci
from so_arena.analysis.optimization import BestOfN, GameTree, evaluate_tree, selection_policies
from so_arena.core.ground_truth import JudgeCorrectness, StanceValue
from so_arena.core.runner import build_players, run_sync
from so_arena.mechanisms import Consultancy, Debate
from so_arena.samplers.arms import arm_profiles
from so_arena.samplers.pools import expand_tree, load_trees

from decisive_common import (DEBATERS, JUDGES, Ledger, cache_spend, chunks, debater_policy, judge_policy,
                             map_keys, selected_items)

GT = [StanceValue(), JudgeCorrectness()]
NS = [1, 2, 4, 8, 16]
WORDS = 200


def mechanisms():
    return [Consultancy(rounds=1, word_limit=WORDS, name="consultancy_1r"),
            Debate(rounds=1, simultaneous=True, word_limit=WORDS, name="debate_1r")]


def liar_role(mech: str, arm: str) -> str:
    """The role arguing the false answer on an arm (instructed arms: the lead takes the arm's stance)."""
    if mech.startswith("consultancy"):
        return "consultant"
    return "debater_b" if arm == "true" else "debater_a"


def honest_role(mech: str, arm: str) -> str | None:
    if mech.startswith("consultancy"):
        return None if arm == "false" else "consultant"
    return "debater_a" if arm == "true" else "debater_b"


def policies(mech: str, arm: str, mode: str, n: int) -> dict:
    """Selection policies of one curve point. ``liar``: only the false side at n; ``both``: every agent at n."""
    if mode == "both":
        roles = ["consultant"] if mech.startswith("consultancy") else ["debater_a", "debater_b"]
        return {r: BestOfN(n) for r in roles}
    liar = liar_role(mech, arm)
    if mech.startswith("consultancy") and arm == "true":
        return {}  # the true arm has no liar: its consultant stays at n = 1
    return {liar: BestOfN(n)}


def tree_path(run: Path, d: str, j: str, mech: str, arm: str) -> Path:
    return run / "optimize" / d / j / f"{mech}-{arm}.jsonl"


def candidate_texts(tree: GameTree, episodes) -> list[dict]:
    """Full text of every candidate of every pool: the acting role's turn in a leaf below each child."""
    by_id = {e.id: e for e in episodes}
    out, seen = [], set()

    def first_leaf(nid: str) -> str:
        while nid in tree.nodes:
            nid = tree.nodes[nid].children[0]
        return nid

    for nid, node in tree.nodes.items():
        for i, child in enumerate(node.children):
            if (node.key, i) in seen:
                continue
            ep = by_id.get(tree.leaves[first_leaf(child)].episode_id)
            if ep is None:
                continue
            text = "\n\n".join(t.text for t in ep.turns if t.role == node.role and t.phase == node.phase and t.text)
            seen.add((node.key, i))
            out.append({"key": node.key, "role": node.role, "phase": node.phase, "index": i, "text": text})
    return out


async def grow(run: Path, items, d: str, j: str, mech, *, pool: int, concurrency: int, estimate: bool) -> float:
    """Trees for one (debater, judge, mechanism), both arms; resumes from saved trees. Returns the new cost."""
    cost = 0.0
    sem = asyncio.Semaphore(concurrency)
    sizes = {"consultant": pool} if mech.name.startswith("consultancy") else {"debater_a": pool, "debater_b": pool}
    profs = arm_profiles(mech, agent=debater_policy(DEBATERS[d]), fixtures={"judge": judge_policy(JUDGES[j])})
    async def grow_arm(prof):
        arm = prof.tags["arm"]
        path = tree_path(run, d, j, mech.name, arm)
        done = {t.item_id for t in load_trees(path)} if path.exists() else set()
        cand_path = path.with_name(f"{mech.name}-{arm}.candidates.jsonl")
        if not estimate:
            path.parent.mkdir(parents=True, exist_ok=True)

        async def one(item):
            nonlocal cost
            async with sem:
                tree, eps = await expand_tree(mech, item, build_players(prof, item), pool_sizes=sizes,
                                              ground_truth=GT, concurrency=16, keep_episodes=True,
                                              profile=prof.name)
            cost += sum((u.get("cost_usd") or 0.0) for u in tree.usage.values())  # every sample, judges included
            if estimate:
                return
            # appended as each tree completes (trees.jsonl format), so an interrupted run resumes
            with open(cand_path, "a") as f:
                for row in candidate_texts(tree, eps):
                    f.write(json.dumps({"item": item.id, **row}) + "\n")
            with open(path, "a") as f:
                f.write(tree.model_dump_json() + "\n")

        await asyncio.gather(*[one(it) for it in items if it.id not in done])

    await asyncio.gather(*[grow_arm(prof) for prof in profs])  # the arms share the semaphore
    return cost


def _curve_cell(run: Path, d: str, j: str, mech: str) -> tuple[list[dict], list[dict]]:
    """Per-item accuracies and the liar's selection weights for one (debater, judge, mechanism)."""
    per_item, sel = [], []
    per_arm = {arm: {t.item_id: t for t in load_trees(tree_path(run, d, j, mech, arm))}
               for arm in ("true", "false") if tree_path(run, d, j, mech, arm).exists()}
    if len(per_arm) < 2:
        return per_item, sel
    for mode in ("liar", "both"):
        for n in NS:
            for it in sorted(set(per_arm["true"]) & set(per_arm["false"])):
                vals = []
                for arm in ("true", "false"):
                    tree = per_arm[arm][it]
                    pols = policies(mech, arm, mode, n)
                    v = evaluate_tree(tree, pols).values
                    vals.append((v.get("judge_correct", np.nan), v.get("judge_p_true", np.nan)))
                    if mode == "liar" and (arm == "false" or not mech.startswith("consultancy")):
                        liar = liar_role(mech, arm)
                        for key, s in selection_policies(tree, pols).items():
                            if s["role"] == liar:
                                for i, p in enumerate(s["probs"]):
                                    sel.append({"debater": d, "judge": j, "mechanism": mech, "item": it, "arm": arm,
                                                "n": n, "key": key, "index": i, "weight": float(p)})
                per_item.append({"debater": d, "judge": j, "mechanism": mech, "mode": mode, "n": n, "item": it,
                                 "accuracy": np.mean([x[0] for x in vals]), "p_true": np.mean([x[1] for x in vals])})
    return per_item, sel


def curves(run: Path, n_boot: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Judge accuracy per (debater, judge, mechanism, mode, n), mean of the arms per item, with bootstrap CIs
    over items, plus rows pooled over every (debater, judge) pair (``debater == judge == "all"``: an item's
    accuracy is its mean over the pairs, and items are resampled); the per-item accuracies; and the liar's
    selection weights over its candidates at each n (for the audit)."""
    from concurrent.futures import ProcessPoolExecutor

    cells = [(run, d, j, m.name) for d in DEBATERS for j in JUDGES for m in mechanisms()]
    per_item, sel = [], []
    with ProcessPoolExecutor(max_workers=min(len(cells), os.cpu_count() or 1)) as pool:
        for a, b in pool.map(_curve_cell, *zip(*cells)):
            per_item += a
            sel += b
    items = pd.DataFrame(per_item)
    return summarize_curves(items, n_boot=n_boot), items, pd.DataFrame(sel)


def summarize_curves(items: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    """Per-pair rows and rows pooled over pairs (on the items every pair has), bootstrap CIs over items."""
    if items.empty:
        return pd.DataFrame()
    keys = ["mechanism", "mode", "n"]
    npairs = items.groupby(keys + ["item"]).size()
    full = npairs[npairs == npairs.groupby(keys).transform("max")].reset_index()[keys + ["item"]]
    pooled = (items.merge(full, on=keys + ["item"]).groupby(keys + ["item"])[["accuracy", "p_true"]].mean()
              .reset_index().assign(debater="all", judge="all"))
    rows = []
    for (d, j, m, mode, n), g in pd.concat([items, pooled]).groupby(["debater", "judge"] + keys, sort=False):
        acc = g.accuracy.to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(acc, n_boot=n_boot)
        rows.append({"debater": d, "judge": j, "mechanism": m, "mode": mode, "n": n,
                     "accuracy": float(np.nanmean(acc)), "lo": lo, "hi": hi,
                     "p_true": float(np.nanmean(g.p_true)), "items": int(np.isfinite(acc).sum())})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--items", type=int, default=150, help="random subset of the selected items")
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=6, help="trees grown at once per (debater, judge, protocol)")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--curves-only", action="store_true")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--budget", type=float, default=None, help="stop before a chunk would take the stage over this")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    out = run / "optimize"
    if not a.curves_only:
        soa.configure(cache_dir=str(run / "cache" / "optimize"), simulate=a.estimate or None)
        items = selected_items(run, a.items)
        ledger = Ledger(run)
        prev = json.loads((out / "cost.json").read_text()) if (out / "cost.json").exists() else {}
        cost = dict(prev)

        async def judge_pass(batch, js):
            jobs = [(d, j, m) for j in js for d in DEBATERS for m in mechanisms()]
            got = await asyncio.gather(*[grow(run, batch, d, j, m, pool=a.pool, concurrency=a.concurrency,
                                              estimate=a.estimate) for d, j, m in jobs])
            for (d, j, m), c in zip(jobs, got):
                k = f"{d}|{j}|{m.name}"
                cost[k] = cost.get(k, 0.0) + c
            if not a.estimate:
                out.mkdir(parents=True, exist_ok=True)
                (out / "cost.json").write_text(json.dumps(cost, indent=1))
                ledger.set("optimize", cache_spend(run, "optimize"))  # exact; cost.json misses interrupted trees

        async def batch_run(batch):
            first, *others = JUDGES
            await judge_pass(batch, [first])  # pays for the pools
            await judge_pass(batch, others)  # replays them; pays only its own judge calls

        log = logging.getLogger("decisive")
        per_item, done = None, 0
        for batch in chunks(items, a.chunk):
            spent = sum(cost.values()) if a.estimate else cache_spend(run, "optimize")
            if a.budget is not None and per_item is not None and spent + per_item * len(batch) > a.budget:
                log.warning("stopping: $%.0f spent, next chunk ~$%.0f, budget $%.0f", spent, per_item * len(batch),
                            a.budget)
                break
            before = spent
            run_sync(batch_run(batch))  # items with saved trees are skipped (resume)
            done += len(batch)
            now = sum(cost.values()) if a.estimate else cache_spend(run, "optimize")
            if now - before > 0:
                per_item = max(per_item or 0.0, (now - before) / len(batch))
            log.info("%d items done, stage $%.2f", done, now)
        total = sum(cost.values()) - sum(prev.values())
        print(json.dumps({k: round(v, 2) for k, v in cost.items()}, indent=1), f"\nthis run ${total:.2f}")
        if a.estimate:
            return
    cur, per_item, sel = curves(run)
    cur.to_csv(out / "curves.csv", index=False)
    per_item.to_csv(out / "curve_items.csv.gz", index=False)
    sel.to_csv(out / "selection.csv.gz", index=False)
    print(cur.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
