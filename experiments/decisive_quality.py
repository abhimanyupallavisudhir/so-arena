"""Decisive study, replication under information asymmetry: QuALITY, as in Khan et al. (2024) and Michael et al. (2023).

The rest of the study gives the debaters a *capability* gap over the judge. Here the gap is *information*: the
debaters read a long story (QuALITY v1.0.1 dev, the ``difficult`` questions that every untimed annotator answered
correctly; binary, the answer against the annotators' favourite distractor), the judge sees only the question and
the debate, and quotes from the story are checked against it (``<claim kind="quote">``, verified or flagged).

Same judges, debaters, arms, rounds and word limits as ``decisive_protocols.py``:

* ``direct``: the judge alone, without the story;
* ``direct_strong``: the debater model alone, with the story;
* ``consultancy``: 3 rounds, quotes verified, the judge asks a question between rounds;
* ``debate_seq``: 3 sequential rounds, quotes verified;
* ``debate_seq_noquotes``: the same debate with no verification (does verification carry the effect?).

Rows go to ``<run>/quality/rows.csv`` in ``decisive_protocols.py``'s format; ``decisive_report.py``'s summaries
apply unchanged.

    python experiments/decisive_quality.py --run runs/decisive --items 300 [--debaters gpt-5-mini] [--budget 900]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

import pandas as pd

import so_arena as soa
from so_arena.core.runner import run_sync
from so_arena.domains import get_domain
from so_arena.mechanisms import Consultancy, Debate, DirectJudge
from so_arena.samplers.arms import ASDExperiment

from decisive_common import DEBATERS, JUDGES, Ledger, judge_set, chunks, debater_policy, judge_policy, map_keys, solo_policy
from decisive_protocols import GT, episode_rows

WORDS = 200
PASSAGE = {"agents": ["passage"]}


def quote_policy():
    return soa.VerificationPolicy(verifiers=["quote"])


def protocols():
    return [Consultancy(rounds=3, word_limit=WORDS, affordances=PASSAGE, verification=quote_policy()),
            Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq", affordances=PASSAGE,
                   verification=quote_policy()),
            Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq_noquotes", affordances=PASSAGE)]


def quality_items(n: int, seed: int = 0):
    dom = get_domain("quality", hard_only=True, min_untimed_accuracy=1.0)
    items = dom.load(split="dev")
    return dom, random.Random(seed).sample(items, len(items))[:n]  # a fixed order: smaller runs are prefixes


async def run_chunk(items, run: Path, ctx, *, debaters: list[str], repeats: int, concurrency: int,
                    judges: dict[str, str] | None = None) -> list[dict]:
    st = lambda *p: str(run / "quality" / Path(*p))  # noqa: E731
    rows: list[dict] = []

    async def go(mech, j, d, judge, agent, store):
        exp = ASDExperiment([mech], items, agent=agent, fixtures={"judge": judge}, ground_truth=GT, repeats=repeats,
                            store=store, concurrency=concurrency, ctx=ctx)
        rows.extend(episode_rows(await exp.arun(), judge=j, debater=d))

    judges = judges or dict(JUDGES)
    first, *others = judges
    phase_a = [go(DirectJudge(), j, "-", judge_policy(m), None, st("direct", j)) for j, m in judges.items()]
    phase_a += [go(DirectJudge(name="direct_strong", affordances={"judge": ["passage"]}), "-", d,
                   solo_policy(DEBATERS[d]), None, st("strong", d)) for d in debaters]
    for d in debaters:
        for mech in protocols():
            js = list(judges) if mech.name == "consultancy" else [first]  # only consultancy's agents see the judge
            phase_a += [go(mech, j, d, judge_policy(judges[j]), debater_policy(DEBATERS[d]), st(d, j, mech.name))
                        for j in js]
    await asyncio.gather(*phase_a)
    await asyncio.gather(*[go(mech, j, d, judge_policy(judges[j]), debater_policy(DEBATERS[d]), st(d, j, mech.name))
                           for j in others for d in debaters for mech in protocols() if mech.name != "consultancy"])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--items", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--debaters", nargs="*", default=list(DEBATERS))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--judges", nargs="*", default=None, help="judges by name (weak or STRONG_JUDGES; default: weak)")
    ap.add_argument("--stage", default="quality", help="ledger key and output directory (rows.csv)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / "quality"), concurrency=a.concurrency)
    dom, items = quality_items(a.items)
    ctx = dom.context()
    ledger = Ledger(run)
    out = run / a.stage
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.csv"
    log = logging.getLogger("decisive")
    spent = float(pd.read_csv(rows_path).cost.sum()) if rows_path.exists() else 0.0
    per_item = None
    for i, chunk in enumerate(chunks(items, a.chunk)):
        if per_item is not None and spent + per_item * len(chunk) > a.budget:
            log.warning("stopping before chunk %d: $%.0f spent, next ~$%.0f", i, spent, per_item * len(chunk))
            break
        before = spent
        rows = run_sync(run_chunk(chunk, run, ctx, debaters=a.debaters, repeats=a.repeats, concurrency=a.concurrency,
                                  judges=judge_set(a.judges)))
        prev = pd.read_csv(rows_path) if rows_path.exists() else pd.DataFrame()
        df = pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates(["episode", "role"], keep="last")
        df.to_csv(rows_path, index=False)
        spent = float(df.cost.sum())
        if spent > before:
            per_item = max(per_item or 0.0, (spent - before) / len(chunk))
        ledger.set(a.stage, spent)
        log.info("chunk %d done: %d items, stage $%.2f, study $%.2f", i, len(chunk), spent, ledger.total)
    print(json.dumps(ledger.data, indent=1), f"total ${ledger.total:.2f}")


if __name__ == "__main__":
    main()
