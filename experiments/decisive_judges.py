"""Decisive study: do the protocols help more as the judge gets stronger? Stored debates judged by stronger judges.

The weak judges (gpt-4o-mini, Llama-3.1-8B, Gemma-3-12B) gain little from debate under information asymmetry and
about a fifth of the gap under a capability gap. Here the same debates are judged by stronger judges
(``STRONG_JUDGES``: gpt-4.1-mini, gpt-4.1). Debate transcripts do not depend on the judge, so the debaters are
*replayed* from the stored episodes (``ReplayPolicy`` returns the stored speech for the same item, sample, role,
round and stance; verification of quotes reruns on it) and only the new judge is called. Each judge is also run
alone. Consultancy, whose consultant answers the judge's questions, is run fresh (one sample) with the new judge.

Sources: the protocol grid (``protocols/``, testbed, 320 items; sequential and simultaneous debate) and QuALITY
(``quality/``; quoted and unquoted debate). Rows go to ``<run>/judges/rows.csv`` with a ``dataset`` column.

    python experiments/decisive_judges.py --run runs/decisive [--judges gpt-4.1-mini gpt-4.1] [--datasets testbed quality]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import pandas as pd

import so_arena as soa
from so_arena.core.actions import Action
from so_arena.core.policy import Policy
from so_arena.core.runner import run_sync
from so_arena.core.store import RunStore
from so_arena.mechanisms import Consultancy, Debate, DirectJudge
from so_arena.samplers.arms import ASDExperiment

from decisive_common import DEBATERS, STRONG_JUDGES, Ledger, debater_policy, judge_policy, judge_set, map_keys, selected_items
from decisive_protocols import GT, episode_rows

WORDS = 200


class ReplayPolicy(Policy):
    """Replays an agent's stored speeches: the text it produced for (item, sample, role, phase, stance)."""

    def __init__(self, episodes, source: str, label: str = "replay"):
        super().__init__(label=label)
        self.source = source  # part of the episode ids: replays of different debaters must not collide
        self.speech: dict[tuple, str] = {}
        for ep in episodes:
            if ep.error:
                continue
            for t in ep.turns:
                if t.kind == "text" and t.role in ep.positions:
                    self.speech[(ep.item_id, ep.repeat, t.role, t.phase, ep.positions[t.role])] = t.text
        self.missing = 0

    def describe(self) -> dict:
        return {**super().describe(), "replay_of": self.source, "speeches": len(self.speech)}

    async def act(self, request, ctx):
        g = ctx.game
        key = (g.item.id, g.repeat, ctx.role, request.phase, g.stance(ctx.role))
        if key not in self.speech:
            self.missing += 1
            raise LookupError(f"no stored speech for {key}")
        return Action(text=self.speech[key])


def sources(run: Path, dataset: str, debater: str):
    """(mechanism, source store) for the judge-independent protocols of a dataset; the stored first judge's runs."""
    if dataset == "testbed":
        root = run / "protocols" / debater / "gpt-4o-mini"
        mechs = [Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq"),
                 Debate(rounds=3, simultaneous=True, word_limit=WORDS, name="debate_sim")]
    else:
        from decisive_quality import PASSAGE, quote_policy

        root = run / "quality" / debater / "gpt-4o-mini"
        mechs = [Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq", affordances=PASSAGE,
                        verification=quote_policy()),
                 Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq_noquotes", affordances=PASSAGE)]
    return [(m, root / m.name) for m in mechs]


def consultancy(dataset: str):
    if dataset == "testbed":
        return Consultancy(rounds=3, word_limit=WORDS)
    from decisive_quality import PASSAGE, quote_policy

    return Consultancy(rounds=3, word_limit=WORDS, affordances=PASSAGE, verification=quote_policy())


def dataset_items(run: Path, dataset: str):
    if dataset == "testbed":
        return selected_items(run, 320), None
    from decisive_quality import quality_items

    dom, items = quality_items(300)
    return items, dom.context()


async def run_dataset(run: Path, dataset: str, judges: dict[str, str], *, debaters: list[str], repeats: int,
                      concurrency: int, limit: int | None = None) -> list[dict]:
    items, ctx = dataset_items(run, dataset)
    items = items[:limit]
    out = run / "judges" / dataset
    rows: list[dict] = []

    async def go(mech, j, d, judge, agent, store, its, reps):
        exp = ASDExperiment([mech], its, agent=agent, fixtures={"judge": judge}, ground_truth=GT, repeats=reps,
                            store=str(store), concurrency=concurrency, ctx=ctx)
        got = episode_rows(await exp.arun(), judge=j, debater=d)
        rows.extend({**r, "dataset": dataset} for r in got)

    replays = {}  # (debater, protocol) -> (mechanism, replay policy, items, repeats); loaded once for all judges
    wanted = {it.id for it in items}
    for d in debaters:
        for mech, src in sources(run, dataset, d):
            st = RunStore(str(src))
            if not st.has_episodes():
                continue
            eps = [e for e in st.episodes() if e.item_id in wanted]
            done = {e.item_id for e in eps if not e.error}
            reps = 1 + max((e.repeat for e in eps), default=0)
            replays[(d, mech.name)] = (mech, ReplayPolicy(eps, source=str(src.relative_to(run))),
                                       [it for it in items if it.id in done], reps)
            del eps
    jobs = []
    for j, model in judges.items():
        jp = judge_policy(model)
        jobs.append(go(DirectJudge(), j, "-", jp, None, out / "direct" / j, items, 3 if dataset == "testbed" else 2))
        for (d, name), (mech, replay, its, reps) in replays.items():
            jobs.append(go(mech, j, d, jp, replay, out / d / j / name, its, reps))
        for d in debaters:
            jobs.append(go(consultancy(dataset), j, d, jp, debater_policy(DEBATERS[d]), out / d / j / "consultancy",
                           items, 1))
    await asyncio.gather(*jobs)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--judges", nargs="*", default=list(STRONG_JUDGES))
    ap.add_argument("--datasets", nargs="*", default=["testbed", "quality"])
    ap.add_argument("--debaters", nargs="*", default=list(DEBATERS))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N items of each dataset (a pilot)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / "judges"))
    judges = judge_set(a.judges)
    rows = []
    for ds in a.datasets:
        rows += run_sync(run_dataset(run, ds, judges, debaters=a.debaters, repeats=1, concurrency=a.concurrency,
                                     limit=a.limit))
    path = run / "judges" / "rows.csv"
    prev = pd.read_csv(path) if path.exists() else pd.DataFrame()
    df = pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates(["episode", "role", "judge"], keep="last")
    df.to_csv(path, index=False)
    Ledger(run).set("judges", float(df.cost.sum()))
    print(df.groupby(["dataset", "judge", "debater", "mechanism"]).correct.mean().round(3).to_string())
    print(json.dumps(Ledger(run).data, indent=1))


if __name__ == "__main__":
    main()
