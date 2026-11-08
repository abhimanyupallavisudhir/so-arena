"""Model panel under optimization: does a smarter liar beat the protocols more? Best-of-n across panel agents.

The model panel (``decisive_panel.py``) measures EIC with prompted liars only. Here every panel agent is optimized
against each weak judge by exact best-of-n over sampled pools, as in the one-round best-of-n stage
(``decisive_optimize.py``), and scored on independent judge samples (``decisive_rejudge.py``), so selection and
scoring never share a judge sample:

* protocols: a one-round consultancy (one 200-word argument, no judge questions) and a one-round simultaneous
  debate, both instructed arms;
* pools of 16 candidate speeches per agent decision; the judge scores every leaf (16 for consultancy, 16 x 16 for
  debate) once to *select*, and the same leaves are re-judged with fresh samples to *score*;
* curves over n = 1, 2, 4, 8, 16 for ``liar`` (only the false side optimized) and ``both`` (both sides);
* agents: a spread of the panel's agents over the Artificial Analysis index; judges: the study's three weak
  judges; items: the first ``--items`` of the panel's uniformly drawn (unselected) testbed items.

Outputs go under ``<run>/panel_bon/`` in the layout of the main best-of-n stage (``optimize/<agent>/<judge>/`` and
``optimize_holdout/...``), so ``decisive_ic.cell_rows`` computes the per-item ASD. Analysis:
``decisive_panel_bon_report.py``.

    python experiments/decisive_panel_bon.py --run runs/decisive --items 4 --estimate
    python experiments/decisive_panel_bon.py --run runs/decisive --items 150 --budget 3200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import so_arena as soa
from so_arena.core.runner import build_players, run_sync
from so_arena.mechanisms import Consultancy, Debate
from so_arena.samplers.arms import arm_profiles
from so_arena.samplers.pools import expand_tree, load_trees

from decisive_common import FreshSamples, Ledger, cache_spend, chunks, map_keys
from decisive_optimize import GT
from decisive_panel import agent_policy, judge_policy, panel_config, panel_items, register_prices

STAGE = "panel_bon"
WORDS = 200
AGENTS = ["llama-3.1-8b", "gpt-4o-mini", "mistral-small-4", "gpt-4.1", "claude-haiku-4.5", "deepseek-v3.2",
          "claude-sonnet-4.5", "gpt-5-mini@medium", "claude-sonnet-5@low", "gpt-5.6-sol@low",
          "claude-opus-5@low", "gpt-6-astra@low"]
JUDGES = ["gpt-4o-mini", "llama-3.1-8b", "gemma-3-12b"]


def mechanisms():
    return [Consultancy(rounds=1, word_limit=WORDS, judge_questions=False, name="consultancy_1r"),
            Debate(rounds=1, simultaneous=True, word_limit=WORDS, name="debate_1r")]


def root(run: Path) -> Path:
    """The stage's own 'run' directory: ``decisive_ic`` reads ``<root>/optimize`` and ``<root>/optimize_holdout``."""
    return Path(run) / STAGE


def tree_path(run: Path, agent: str, judge: str, mech: str, arm: str, *, holdout: bool = False) -> Path:
    return root(run) / ("optimize_holdout" if holdout else "optimize") / agent / judge / f"{mech}-{arm}.jsonl"


def done_items(run: Path, agent: str, judge: str, mech: str) -> set:
    """Items whose selection and held-out trees exist on both arms."""
    sets = []
    for arm in ("true", "false"):
        for h in (False, True):
            p = tree_path(run, agent, judge, mech, arm, holdout=h)
            sets.append({t.item_id for t in load_trees(p)} if p.exists() else set())
    return set.intersection(*sets)


async def grow_cell(run: Path, items, agent: dict, judge: dict, mech, *, pool: int, concurrency: int,
                    estimate: bool) -> float:
    """Selection and held-out trees for one (agent, judge, protocol) on both arms. The held-out tree has the same
    leaves (the agents' candidates replay from the response cache) and fresh judge samples."""
    cost = 0.0
    sem = asyncio.Semaphore(concurrency)
    sizes = {"consultant": pool} if mech.name.startswith("consultancy") else {"debater_a": pool, "debater_b": pool}
    jp = judge_policy(judge)
    sel = arm_profiles(mech, agent=agent_policy(agent), fixtures={"judge": jp})
    held = {p.tags["arm"]: p for p in arm_profiles(mech, agent=agent_policy(agent), fixtures={"judge": FreshSamples(jp)})}

    async def arm_run(prof):
        arm = prof.tags["arm"]
        paths = {h: tree_path(run, agent["name"], judge["name"], mech.name, arm, holdout=h) for h in (False, True)}
        have = {h: ({t.item_id for t in load_trees(p)} if p.exists() else set()) for h, p in paths.items()}
        if not estimate:
            for p in paths.values():
                p.parent.mkdir(parents=True, exist_ok=True)

        async def one(item, holdout):
            nonlocal cost
            p = held[arm] if holdout else prof
            async with sem:
                tree, _ = await expand_tree(mech, item, build_players(p, item), pool_sizes=sizes, ground_truth=GT,
                                            concurrency=16, profile=p.name)
            cost += sum((u.get("cost_usd") or 0.0) for u in tree.usage.values())
            if not estimate:
                with open(paths[holdout], "a") as f:  # appended per tree, so an interrupted run resumes
                    f.write(tree.model_dump_json() + "\n")

        async def both(item):
            if item.id not in have[False]:
                await one(item, False)  # the selection tree pays for the candidates; the held-out one replays them
            if item.id not in have[True]:
                await one(item, True)

        await asyncio.gather(*[both(it) for it in items])

    await asyncio.gather(*[arm_run(p) for p in sel])
    return cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--items", type=int, default=150)
    ap.add_argument("--agents", nargs="*", default=AGENTS)
    ap.add_argument("--judges", nargs="*", default=JUDGES)
    ap.add_argument("--pool", type=int, default=16)
    ap.add_argument("--concurrency", type=int, default=8, help="trees grown at once per (agent, judge, protocol)")
    ap.add_argument("--cells", type=int, default=12, help="(agent, judge, protocol) cells grown at once")
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--budget", type=float, default=3200.0)
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("decisive")
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / STAGE), simulate=a.estimate or None, cache_reasoning=False)
    register_prices()
    cfg = panel_config()
    agents = [next(x for x in cfg["agents"] if x["name"] == n) for n in a.agents]
    judges = [next(x for x in cfg["judges"] if x["name"] == n) for n in a.judges]
    items = panel_items(run, a.items)
    ledger = Ledger(run)

    async def batch_run(batch, estimate):
        sem = asyncio.Semaphore(a.cells)
        costs = {}

        async def cell(ag, jd, m):
            todo = [it for it in batch if estimate or it.id not in done_items(run, ag["name"], jd["name"], m.name)]
            if not todo:
                return
            async with sem:
                costs[(ag["name"], jd["name"], m.name)] = await grow_cell(
                    run, todo, ag, jd, m, pool=a.pool, concurrency=a.concurrency, estimate=estimate)

        await asyncio.gather(*[cell(ag, jd, m) for ag in agents for jd in judges for m in mechanisms()])
        return costs

    if a.estimate:
        costs = run_sync(batch_run(items, True))
        by_agent = {}
        for (ag, _, _), c in costs.items():
            by_agent[ag] = by_agent.get(ag, 0.0) + c
        print(json.dumps({k: round(v, 2) for k, v in by_agent.items()}, indent=1))
        print(f"estimated ${sum(costs.values()):.2f} for {len(items)} items")
        return
    for batch in chunks(items, a.chunk):
        spent = cache_spend(run, STAGE)
        if spent > a.budget:
            log.warning("budget reached ($%.2f of $%.2f); stopping", spent, a.budget)
            break
        run_sync(batch_run(batch, False))
        ledger.set(STAGE, cache_spend(run, STAGE))
        log.info("chunk of %d items done; stage $%.2f", len(batch), ledger.data[STAGE])
    ledger.set(STAGE, cache_spend(run, STAGE))
    print(json.dumps({STAGE: ledger.data.get(STAGE)}, indent=1))


if __name__ == "__main__":
    main()
