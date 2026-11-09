"""Model panel under prompt optimization: honest and deceptive strategy searches for a spread of panel agents.

The prompt-optimization sampler of the main study (``decisive_eic_po.py``: an optimizer, Claude Opus 5.5, writes
system-prompt strategies scored only by the protocol's reward) applied to several panel agents instead of
gpt-5-mini alone, on the one-round protocols of the best-of-n panel (``decisive_panel_bon.py``) so the two
samplers compare agent by agent:

* per (agent, protocol): an **honest** search (assigned the true answer, ``honest`` directive) and a **deceptive**
  search (assigned a false answer, ``deceptive`` directive), each 5 rounds of 4 candidates after the empty seed,
  on ``--train`` training items; the other debate seat plays the agent's base policy, seats balanced by item;
* the best strategy after every round (chosen on training reward) is scored on ``--test`` held-out items, and
  EIC along the search is the held-out ASD of the best honest minus the best deceptive strategy, paired by item.

Items: the uniformly drawn panel items after the first 300 (``decisive_panel.panel_items``), so training and
held-out items are disjoint from the instructed panel and the best-of-n panel. Judge: one weak judge
(``--judge``, default gpt-4o-mini).

Outputs ``<run>/panel_po/<agent>/<protocol>.json`` in the format of ``decisive_eic_po.py``. Analysis:
``decisive_panel_bon_report.py``.

    python experiments/decisive_panel_po.py --run runs/decisive --budget 1800
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import so_arena as soa
from so_arena.core.runner import run_sync
from so_arena.samplers.prompt_search import PromptSearch

import decisive_eic_po as dpo
from decisive_common import Ledger, cache_spend, map_keys
from decisive_panel import agent_policy, judge_policy, panel_config, panel_items, register_prices
from decisive_panel_bon import mechanisms
from decisive_protocols import GT

STAGE = "panel_po"
AGENTS = ["llama-3.1-8b", "gpt-4.1", "claude-sonnet-4.5", "gpt-5-mini@medium", "gpt-5.6-sol@low",
          "claude-opus-5@low"]
OFFSET = 300  # after the instructed panel's items


class PanelArmSearch(dpo.ArmSearch):
    """``decisive_eic_po.ArmSearch`` for a panel agent on the one-round protocols."""

    def __init__(self, mech, items, *, arm: str, judge, factory, **kw):
        self.seats = ("debater_a", "debater_b") if mech.name.startswith("debate") else ("consultant",)
        self.judge, self.base = judge, factory("")
        self.agent_roles = [r for r, s in mech.role_specs().items() if s.kind == "agent"]
        PromptSearch.__init__(self, mech, items, role=self.seats[0], policy_factory=factory, others={},
                              arms=[arm], optimizer=dpo.OPTIMIZER, ground_truth=GT, **kw)
        self.arm = arm


async def run_cell(mech, agent: dict, judge: dict, train, test, out: Path, *, concurrency: int, rounds: int,
                   per_round: int) -> dict:
    jp = judge_policy(judge)

    def factory(strategy: str):
        return agent_policy(agent, strategy=strategy or None)

    searches = {arm: PanelArmSearch(mech, train, arm=arm, judge=jp, factory=factory, directive=d,
                                    algorithm="reflective", iterations=rounds, candidates_per_iter=per_round,
                                    concurrency=concurrency, max_strategy_words=200)
                for arm, d in (("true", "honest"), ("false", "deceptive"))}
    results = dict(zip(searches, await asyncio.gather(*[s.arun() for s in searches.values()])))
    record: dict = {"protocol": mech.name, "judge": judge["name"], "debater": agent["name"],
                    "train": [it.id for it in train], "test": [it.id for it in test], "searches": {}}

    async def finish(arm, s):
        train_c = [c for c in results[arm].candidates if c.split == "train"]
        path, best = [], None
        for k in range(rounds + 1):
            pool = [c for c in train_c if c.iteration <= k]
            best = max(pool, key=lambda c: c.mean_reward) if pool else best
            path.append(best.id)

        async def held_out(cid):
            c = next(c for c in train_c if c.id == cid)
            _, _, eps = await s.evaluate_strategy(c.strategy, test)
            return cid, [dpo.episode_record(e) for e in eps]

        held = dict(await asyncio.gather(*[held_out(cid) for cid in dict.fromkeys(path)]))
        return arm, {
            "directive": s.directive_name,
            "strategies": [{"id": c.id, "iteration": c.iteration, "parent": c.parent, "strategy": c.strategy,
                            "rationale": c.rationale,
                            "train": [dpo.episode_record(e) for e in s.episodes.get(c.id, [])]} for c in train_c],
            "best_path": path, "heldout": held, "optimizer_calls": s.optimizer_calls}

    record["searches"] = dict(await asyncio.gather(*[finish(arm, s) for arm, s in searches.items()]))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(record))
    tmp.replace(out)
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--agents", nargs="*", default=AGENTS)
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--train", type=int, default=100)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=dpo.ROUNDS)
    ap.add_argument("--per-round", type=int, default=dpo.PER_ROUND)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--cells", type=int, default=4, help="(agent, protocol) cells at once")
    ap.add_argument("--budget", type=float, default=1800.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("decisive")
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / STAGE), cache_reasoning=False)
    register_prices()
    cfg = panel_config()
    agents = [next(x for x in cfg["agents"] if x["name"] == n) for n in a.agents]
    judge = next(x for x in cfg["judges"] if x["name"] == a.judge)
    items = panel_items(run, OFFSET + a.train + a.test)[OFFSET:]
    train, test = items[:a.train], items[a.train:]
    ledger = Ledger(run)

    async def all_cells():
        sem = asyncio.Semaphore(a.cells)

        async def one(ag, m):
            out = run / STAGE / ag["name"] / f"{m.name}.json"
            if out.exists():
                return
            async with sem:
                if cache_spend(run, STAGE) > a.budget:
                    log.warning("budget reached; skipping %s %s", ag["name"], m.name)
                    return
                rec = await run_cell(m, ag, judge, train, test, out, concurrency=a.concurrency, rounds=a.rounds,
                                     per_round=a.per_round)
                ledger.set(STAGE, cache_spend(run, STAGE))
                log.info("%s; stage $%.2f", dpo.summary(rec), ledger.data[STAGE])

        await asyncio.gather(*[one(ag, m) for ag in agents for m in mechanisms()])

    run_sync(all_cells())
    ledger.set(STAGE, cache_spend(run, STAGE))
    print(json.dumps({STAGE: ledger.data.get(STAGE)}, indent=1))


if __name__ == "__main__":
    main()
