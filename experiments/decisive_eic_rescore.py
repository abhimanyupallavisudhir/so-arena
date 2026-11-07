"""More held-out items for the prompt-optimization sampler (``decisive_eic_po.py``).

The searches chose their strategies on 36 training items and scored them on 80 held-out items: unbiased, but with
95% intervals of about +-0.4 pooled over judges and +-0.3 to +-0.9 within a judge. This stage scores every strategy
on the searches' best-so-far paths (seed included) again, unchanged, on 400 further items, disjoint from the
training and first held-out items (positions 436-836 of the fixed item order), with the same seat balancing. No
search is rerun, so the strategies, and the noise in choosing them, are those of the original searches; only the
precision of their held-out scores changes. The analysis (``decisive_eic.py``) pools both held-out sets.

Prover-estimator is left out by default: its strategies exploit a flaw in our implementation of the payoff
(docs/oversight-results.md, "Why prover-estimator debate fails here"), so scoring them more precisely has little value.

Outputs ``<run>/eic_po_rescore/<judge>/<protocol>.json``: ``{"items": [...], "heldout": {arm: {id: [records]}}}``.

    python experiments/decisive_eic_rescore.py --run runs/decisive [--items 400] [--budget 900]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import so_arena as soa
from so_arena.core.runner import run_sync

from decisive_common import JUDGES, Ledger, cache_spend, judge_policy, map_keys, selected_items
from decisive_eic_po import N_TEST, N_TRAIN, ArmSearch, episode_record, split
from decisive_protocols import protocols

START = 320 + N_TRAIN + N_TEST  # the first item after the original held-out set
DIRECTIVES = {"true": "honest", "false": "deceptive"}


def more_items(run: Path, n: int):
    items = selected_items(run, START + n)
    return items[START:START + n]


def path_strategies(search: dict) -> dict[str, str]:
    """Strategy text of every distinct strategy on a search's best-so-far path, in path order."""
    text = {c["id"]: c["strategy"] for c in search["strategies"]}
    return {cid: text[cid] for cid in dict.fromkeys(search["best_path"])}


async def rescore_cell(rec: dict, train, items, out: Path, concurrency: int) -> dict:
    mech = next(m for m in protocols(fixed_pe=True) if m.name == rec["protocol"])
    jp = judge_policy(JUDGES[rec["judge"]])

    async def one(arm: str, cid: str, strategy: str):
        s = ArmSearch(mech, train, arm=arm, judge=jp, directive=DIRECTIVES[arm], algorithm="reflective",
                      iterations=0, candidates_per_iter=1, concurrency=concurrency, max_strategy_words=200)
        _, _, eps = await s.evaluate_strategy(strategy, items)
        return arm, cid, [episode_record(e) for e in eps]

    # each strategy is saved as it finishes, so an interrupted cell resumes without holding every episode
    partial = out.with_suffix(".partial.jsonl")
    held: dict = {"true": {}, "false": {}}
    if partial.exists():
        for line in partial.read_text().splitlines():
            r = json.loads(line)
            held[r["arm"]][r["id"]] = r["records"]

    async def save(arm, cid, st):
        arm, cid, records = await one(arm, cid, st)
        held[arm][cid] = records
        with partial.open("a") as f:
            f.write(json.dumps({"arm": arm, "id": cid, "records": records}) + "\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.gather(*[save(arm, cid, st) for arm in ("true", "false")
                           for cid, st in path_strategies(rec["searches"][arm]).items() if cid not in held[arm]])
    result = {"protocol": rec["protocol"], "judge": rec["judge"], "items": [it.id for it in items], "heldout": held}
    out.write_text(json.dumps(result))
    partial.unlink(missing_ok=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--judges", nargs="*", default=list(JUDGES))
    ap.add_argument("--protocols", nargs="*", default=["consultancy", "debate_seq", "debate_sim", "recursive_debate"])
    ap.add_argument("--items", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--cells", type=int, default=4, help="cells in flight at once (memory)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / "eic_rescore"))
    train, _ = split(run)
    items = more_items(run, a.items)
    ledger = Ledger(run)
    log = logging.getLogger("decisive")

    async def all_cells():
        gate = asyncio.Semaphore(a.cells)

        async def one(j, p):
            async with gate:
                await cell(j, p)

        async def cell(j, p):
            src, out = run / "eic_po" / j / f"{p}.json", run / "eic_po_rescore" / j / f"{p}.json"
            if out.exists() or not src.exists():
                return
            if cache_spend(run, "eic_rescore") > a.budget:
                log.warning("budget reached; skipping %s %s", j, p)
                return
            await rescore_cell(json.loads(src.read_text()), train, items, out, a.concurrency)
            ledger.set("eic_rescore", cache_spend(run, "eic_rescore"))
            log.info("%s %s rescored; stage $%.2f", j, p, ledger.data["eic_rescore"])

        await asyncio.gather(*[one(j, p) for j in a.judges for p in a.protocols])

    run_sync(all_cells())
    ledger.set("eic_rescore", cache_spend(run, "eic_rescore"))
    print(json.dumps(ledger.data, indent=1))


if __name__ == "__main__":
    main()
