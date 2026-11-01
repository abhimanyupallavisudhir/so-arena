"""Decisive study, stage 2: every protocol, (judge, debater) pair, arm and sample on the gap-selected items.

Protocols (all with 200-word messages):

* ``direct``: the judge alone (fresh samples, not the ones the items were selected on);
* ``consultancy``: 3 rounds, the judge asks a question between rounds;
* ``debate_seq`` / ``debate_sim``: 3 rounds, sequential or simultaneous speeches;
* ``recursive_debate``: depth 2, width 3;
* ``prover_estimator``: depth 2, width 3, rho 2;
* ``direct_strong``: the debater model alone (fresh samples), the top of the sandwich.

Each runs on both instructed arms with 3 samples (repeats) per item and arm. Only consultancy's agents see
the judge (its questions): every other protocol's transcript is independent of the judge until the verdict, so
its agents' calls are made once and replayed from the response cache for the other judges (identical prompt
and sample index). Judges therefore run one after another, the first paying for the transcripts.

Items run in chunks; after each chunk the stage's spend is recomputed from its stores, and the stage stops
before a chunk would take it over ``--budget`` (what is left of the study's budget by default). Compact
per-episode rows go to ``<run>/protocols/rows.csv`` for the report.

    python experiments/decisive_protocols.py --run runs/decisive --estimate --limit 20
    python experiments/decisive_protocols.py --run runs/decisive
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import pandas as pd

import so_arena as soa
from so_arena.core.ground_truth import JudgeCorrectness, StanceValue
from so_arena.core.runner import run_sync
from so_arena.mechanisms import Consultancy, Debate, DirectJudge, ProverEstimatorDebate, RecursiveDebate
from so_arena.samplers.arms import ASDExperiment

from decisive_common import (DEBATERS, JUDGES, Ledger, chunks, debater_policy, episode_cost, judge_policy,
                             map_keys, selected_items, solo_policy)

GT = [StanceValue(), JudgeCorrectness()]
WORDS = 200
LEAD_ROLES = {"direct": ["agent"], "direct_strong": ["agent"], "consultancy": ["consultant"],
              "debate_seq": ["debater_a", "debater_b"], "debate_sim": ["debater_a", "debater_b"],
              "recursive_debate": ["prover"], "prover_estimator": ["prover"]}


def protocols():
    return [Consultancy(rounds=3, word_limit=WORDS),
            Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq"),
            Debate(rounds=3, simultaneous=True, word_limit=WORDS, name="debate_sim"),
            RecursiveDebate(depth=2, width=3, word_limit=WORDS),
            ProverEstimatorDebate(depth=2, width=3, rho=2.0, word_limit=WORDS)]


def episode_rows(eps, *, judge: str, debater: str) -> list[dict]:
    """One row per (episode, stance-holding lead role): what the report needs, without the transcripts."""
    rows = []
    for e in eps:
        if e.error is not None:
            continue
        gt = e.ground_truth or {}
        values = gt.get("role_values") or {}
        base = {"judge": judge, "debater": debater, "mechanism": e.mechanism, "item": e.item_id,
                "arm": e.tags.get("arm"), "repeat": e.repeat, "episode": e.id,
                "correct": gt.get("judge_correct"), "p_true": gt.get("judge_p_true"),
                "parse_ok": (e.outcome.data or {}).get("judge_parse_ok") if e.outcome else None,
                "cost": episode_cost(e)}
        roles = [r for r in LEAD_ROLES.get(e.mechanism, []) if values.get(r) is not None]
        for i, r in enumerate(roles or [None]):
            rows.append({**base, "role": r, "value": values.get(r) if r else None,
                         "reward": e.rewards.get(r) if r else None, "cost": base["cost"] if i == 0 else 0.0})
    return rows


def experiment(mechs, items, *, agent, judge, store, repeats, concurrency) -> ASDExperiment:
    return ASDExperiment(mechs, items, agent=agent, fixtures={"judge": judge}, ground_truth=GT, repeats=repeats,
                         store=store, concurrency=concurrency)


async def run_chunk(items, run: Path, *, repeats: int, concurrency: int, estimate: bool,
                    debaters: list[str] | None = None, store_root: str = "protocols") -> list[dict]:
    """Phase A: the references, every protocol for the first judge and consultancy (whose agents see the judge)
    for the others, all concurrently. Phase B: the other judges on the judge-independent protocols, whose agent
    calls replay phase A's from the cache."""
    st = (lambda *p: None) if estimate else (lambda *p: str(run / store_root / Path(*p)))
    rows: list[dict] = []

    async def go(mech, j, d, judge, agent, store):
        exp = experiment([mech], items, agent=agent, judge=judge, store=store, repeats=repeats,
                         concurrency=concurrency)
        rows.extend(episode_rows(await exp.arun(), judge=j, debater=d))

    def pair(mech, j, d):
        return go(mech, j, d, judge_policy(JUDGES[j]), debater_policy(DEBATERS[d]), st(d, j, mech.name))

    debaters = debaters or list(DEBATERS)
    first, *others = JUDGES
    phase_a = [go(DirectJudge(), j, "-", judge_policy(m), None, st("direct", j)) for j, m in JUDGES.items()]
    phase_a += [go(DirectJudge(name="direct_strong"), "-", d, solo_policy(m), None, st("strong", d))
                for d, m in DEBATERS.items() if d in debaters]
    phase_a += [pair(mech, first, d) for d in debaters for mech in protocols()]
    phase_a += [pair(mech, j, d) for j in others for d in debaters for mech in protocols()
                if mech.name == "consultancy"]
    await asyncio.gather(*phase_a)
    await asyncio.gather(*[pair(mech, j, d) for j in others for d in debaters for mech in protocols()
                           if mech.name != "consultancy"])
    return rows


def stage_cost(run: Path) -> float:
    f = run / "protocols" / "rows.csv"
    return float(pd.read_csv(f).cost.sum()) if f.exists() else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="first N selected items only")
    ap.add_argument("--budget", type=float, default=None, help="this stage's cap in $ (default: what is left)")
    ap.add_argument("--reserve", type=float, default=0.0, help="keep this much of the study budget for later stages")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--concurrency", type=int, default=32, help="per (protocol, judge, debater)")
    ap.add_argument("--debaters", nargs="*", default=None, help="a subset of the debater models (default: all)")
    ap.add_argument("--offset", type=int, default=0, help="skip the first N selected items (already run)")
    ap.add_argument("--store-root", default="protocols",
                    help="directory under the run for the episode stores (a new one keeps memory low: resuming "
                         "reads every episode of a store); rows always go to <run>/protocols/rows.csv")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / "protocols"), simulate=a.estimate or None, concurrency=a.concurrency)
    items = selected_items(run, a.limit)[a.offset:]
    ledger = Ledger(run)
    done_before = stage_cost(run)
    budget = a.budget if a.budget is not None else ledger.remaining() + ledger.data.get("protocols", 0.0) - a.reserve
    log = logging.getLogger("decisive")
    out = run / "protocols"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.csv"
    all_rows: list[dict] = []
    per_item = None
    for i, chunk in enumerate(chunks(items, a.chunk)):
        spent = sum(r["cost"] for r in all_rows) if a.estimate else done_before
        if per_item is not None and spent + per_item * len(chunk) > budget:
            log.warning("stopping before chunk %d: $%.0f spent, next chunk ~$%.0f, budget $%.0f", i, spent,
                        per_item * len(chunk), budget)
            break
        rows = run_sync(run_chunk(chunk, run, repeats=a.repeats, concurrency=a.concurrency, estimate=a.estimate,
                                   debaters=a.debaters, store_root=a.store_root))
        new_cost = sum(r["cost"] for r in rows)
        per_item = max(per_item or 0.0, new_cost / len(chunk)) if new_cost > 0 else per_item
        if a.estimate:
            all_rows += rows
            continue
        # rows are rebuilt from the stores each chunk (resumed episodes carry their original cost)
        prev = pd.read_csv(rows_path) if rows_path.exists() else pd.DataFrame()
        df = pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates(["episode", "role"], keep="last")
        df.to_csv(rows_path, index=False)
        done_before = float(df.cost.sum())
        ledger.set("protocols", done_before)
        log.info("chunk %d done: %d items, stage $%.2f, study $%.2f", i, len(chunk), done_before, ledger.total)
    if a.estimate:
        df = pd.DataFrame(all_rows)
        by = df.groupby(["debater", "mechanism"]).cost.sum().round(2)
        n = len(items)
        print(by.to_string(), f"\n{n} items: ${df.cost.sum():.2f} (${df.cost.sum() / n:.2f} per item)")
        return
    print(json.dumps(ledger.data, indent=1), f"total ${ledger.total:.2f}")


if __name__ == "__main__":
    main()
