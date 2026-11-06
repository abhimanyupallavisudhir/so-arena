"""Decisive study: the same transcripts judged by *instant* judges, which answer without reasoning.

The study's judges reason before answering (testbed questions need working out), so every verdict ends a sampled
chain of thought: two samples of one judge on one argument correlate only about 0.2 within an item. That noise
attenuates reward-truth covariances and made in-sample best-of-n select judge noise (a winner's curse). Prior
best-of-n work (Khan et al. 2024; the debate best-of-n self-play post) scored arguments with a deterministic judge
instead: the answer probability read from one token, with no reasoning.

Here each weak judge model answers *instantly* - no chain of thought, temperature 0, a verbalized probability
(``instant``); gpt-4o-mini also by answer-token logprobs (``gpt-4o-mini-lp``; OpenRouter serves no logprobs for
Gemma-3-12B). Nothing but the judge is called: the agents are replayed from stored transcripts.

* ``grid``: gpt-5-mini's instructed consultancy and sequential and simultaneous debate on all 927 gap-selected
  items (3 samples per item and arm), plus the instant judge alone (the blind baseline). Debate transcripts do not
  depend on the judge and are replayed from the first judge's runs; consultancy's include the judge's questions,
  so judge *J*'s instant verdict is on the consultation that judge *J* (reasoning) conducted - its questions
  replayed, only the final verdict instant. Rows: ``<run>/instant/grid/<judge>/<protocol>.csv``.
* ``trees``: every best-of-16 pool of gpt-5-mini (one-round consultancy and simultaneous debate, 150 items, both
  arms) regrown with the instant judge: every candidate (and, in debate, every one of the 256 pairings) is judged
  again. With a (near-)deterministic judge the payoff that selects is the payoff that is reported, as in the prior
  work. Trees: ``<run>/instant/optimize/gpt-5-mini/<judge>/<mech>-<arm>.jsonl`` (the layout ``decisive_ic.py``
  reads with ``--root instant/optimize``).
* ``reliability``: 300 grid transcripts per judge judged a second time (an independent call), for test-retest.

    python experiments/decisive_instant.py --run runs/decisive --parts grid trees reliability
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
from so_arena.core.runner import build_players, run_sync
from so_arena.core.store import RunStore
from so_arena.mechanisms import Consultancy, Debate, DirectJudge
from so_arena.samplers.arms import ASDExperiment, arm_profiles
from so_arena.samplers.pools import expand_tree, load_trees, save_trees

from decisive_common import FreshSamples, JUDGES, Ledger, cache_spend, map_keys, selected_items
from decisive_optimize import mechanisms as tree_mechanisms
from decisive_optimize import tree_path
from decisive_protocols import GT, episode_rows

WORDS = 200
DEBATER = "gpt-5-mini"
STAGE = "instant"
INSTANT = {**{j: (m, "ask_now") for j, m in JUDGES.items()}, "gpt-4o-mini-lp": (JUDGES["gpt-4o-mini"], "logprobs")}


def instant_judge(model: str, elicitation: str = "ask_now") -> soa.LLMPolicy:
    """A judge that answers at once: no chain of thought, temperature 0 (as near deterministic as the provider
    allows), its probabilities verbalized at once (``ask_now``) or read from the answer token (``logprobs``)."""
    return soa.LLMPolicy(model, cot=False, temperature=0.0, max_tokens=60, elicitation=elicitation)


def grid_mechanisms():
    return [Consultancy(rounds=3, word_limit=WORDS),
            Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq"),
            Debate(rounds=3, simultaneous=True, word_limit=WORDS, name="debate_sim")]


def _positions(get, roles) -> tuple:
    return tuple(sorted((r, get(r)) for r in roles if r != "judge"))


class TranscriptReplay(Policy):
    """Replays every stored text turn (agents' speeches and the judge's questions) for the same item, sample, role,
    phase and agents' stances; probability requests (the verdict) go to ``verdict`` if one is given."""

    def __init__(self, episodes=(), verdict: Policy | None = None, label: str = "replay"):
        super().__init__(label=label)
        self.verdict = verdict
        self.text: dict[tuple, str] = {}
        self.items: set[str] = set()
        self.repeats = 0
        self.add(episodes)

    def add(self, episodes) -> "TranscriptReplay":
        """Index the text turns of (an iterator of) episodes; errored episodes are skipped."""
        for ep in episodes:
            if ep.error:
                continue
            self.items.add(ep.item_id)
            self.repeats = max(self.repeats, ep.repeat + 1)
            pos = _positions(ep.positions.get, ep.positions)
            for t in ep.turns:
                if t.kind == "text":
                    self.text[(ep.item_id, ep.repeat, t.role, t.phase, pos)] = t.text
        return self

    def describe(self) -> dict:
        return {**super().describe(), "turns": len(self.text),
                "verdict": self.verdict.describe() if self.verdict else None}

    async def act(self, request, ctx):
        if request.kind != "text":
            if self.verdict is None:
                raise LookupError("replay has no verdict policy")
            return await self.verdict.act(request, ctx)
        g = ctx.game
        key = (g.item.id, g.repeat, ctx.role, request.phase, _positions(g.stance, g.positions))
        if key not in self.text:
            raise LookupError(f"no stored turn for {key}")
        return Action(text=self.text[key])


class PoolReplay(Policy):
    """Replays the stored candidates of best-of-n pools: the i-th sample of (item, role, phase) is candidate i."""

    def __init__(self, path: Path, label: str = "pool-replay"):
        super().__init__(label=label)
        self.cands: dict[tuple, str] = {}
        with open(path) as f:
            for line in f:
                c = json.loads(line)
                self.cands[(c["item"], c["role"], c["phase"], c["index"])] = c["text"]

    async def act(self, request, ctx):
        key = (ctx.game.item.id, ctx.role, request.phase, ctx.sample_index % 10_000)
        if key not in self.cands:
            raise LookupError(f"no stored candidate for {key}")
        return Action(text=self.cands[key])


def grid_sources(run: Path, mech: str, judge: str) -> list[Path]:
    """Stores holding gpt-5-mini's transcripts: the 320-item grid and the 607-item extension. Debates come from
    the first judge's runs (transcripts are judge-independent); consultancy from the same judge's."""
    j = judge if mech == "consultancy" else "gpt-4o-mini"
    return [run / root / DEBATER / j / mech for root in ("protocols", "protocols_ext")]


async def grid(run: Path, items, judges: dict, *, concurrency: int, log) -> None:
    out = run / STAGE / "grid"

    async def go(mech, name, judge, agent, its, repeats, path, extra=None):
        exp = ASDExperiment([mech], its, agent=agent, fixtures={"judge": judge}, ground_truth=GT, repeats=repeats,
                            store=None, concurrency=concurrency)
        rows = episode_rows(await exp.arun(), judge=name, debater=DEBATER if agent else "-")
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{**r, **(extra or {})} for r in rows]).to_csv(path, index=False)
        log.info("%s: %d rows", path.relative_to(run), len(rows))

    replays: dict[tuple, TranscriptReplay] = {}  # loaded once per source, streamed (stores are large)

    def replay_of(mech: str, judge: str) -> TranscriptReplay:
        srcs = tuple(grid_sources(run, mech, judge))
        if srcs not in replays:
            r = TranscriptReplay()
            for src in srcs:
                if RunStore(str(src)).has_episodes():
                    r.add(RunStore(str(src)).episodes())
            replays[srcs] = r
        return replays[srcs]

    jobs = []
    for name, (model, how) in judges.items():
        jp = instant_judge(model, how)
        p = out / name / "direct.csv"
        if not p.exists():
            jobs.append(go(DirectJudge(), name, jp, None, items, 1, p))
        for mech in grid_mechanisms():
            p = out / name / f"{mech.name}.csv"
            if p.exists():
                continue
            replay = replay_of(mech.name, name.removesuffix("-lp"))
            if mech.name == "consultancy":  # the judge's questions replayed, only its verdict instant
                judge = TranscriptReplay(verdict=jp, label=f"instant-{name}")
                judge.text = replay.text
            else:
                judge = jp
            its = [it for it in items if it.id in replay.items]
            jobs.append(go(mech, name, judge, replay, its, replay.repeats, p))
    await asyncio.gather(*jobs)


async def trees(run: Path, items, judges: dict, *, concurrency: int, log) -> None:
    sem = asyncio.Semaphore(concurrency)
    for mech in tree_mechanisms():
        sizes = {"consultant": 16} if mech.name.startswith("consultancy") else {"debater_a": 16, "debater_b": 16}
        for arm in ("true", "false"):
            src = tree_path(run, DEBATER, "gpt-4o-mini", mech.name, arm)
            agent = PoolReplay(src.with_name(f"{mech.name}-{arm}.candidates.jsonl"))
            have = {t.item_id for t in load_trees(src)}

            async def regrow(name, model, how, mech=mech, arm=arm, agent=agent, have=have):
                path = run / STAGE / "optimize" / DEBATER / name / f"{mech.name}-{arm}.jsonl"
                if path.exists():
                    return
                prof = next(p for p in arm_profiles(mech, agent=agent, fixtures={"judge": instant_judge(model, how)})
                            if p.tags["arm"] == arm)

                async def one(item):
                    async with sem:
                        tree, _ = await expand_tree(mech, item, build_players(prof, item), pool_sizes=sizes,
                                                    ground_truth=GT, concurrency=64, profile=prof.name)
                    return tree

                got = await asyncio.gather(*[one(it) for it in items if it.id in have])
                path.parent.mkdir(parents=True, exist_ok=True)
                save_trees(path, got)
                log.info("%s: %d trees", path.relative_to(run), len(got))

            await asyncio.gather(*[regrow(n, m, h) for n, (m, h) in judges.items()])


async def reliability(run: Path, judges: dict, *, n: int, concurrency: int, log) -> None:
    """A second, independent instant verdict (a different sample index, so not the cached one) on ``n`` debate
    transcripts per judge (sequential debate, first sample, items in the fixed order)."""
    items = selected_items(run, n)
    mech = grid_mechanisms()[1]
    replay, wanted = TranscriptReplay(), {it.id for it in items}
    for src in grid_sources(run, mech.name, "gpt-4o-mini"):
        if RunStore(str(src)).has_episodes():
            replay.add(e for e in RunStore(str(src)).episodes() if e.repeat == 0 and e.item_id in wanted)
    for name, (model, how) in judges.items():
        path = run / STAGE / "reliability" / f"{name}.csv"
        if path.exists():
            continue
        rows = []
        for k, jp in enumerate([instant_judge(model, how), FreshSamples(instant_judge(model, how), 70_000)]):
            exp = ASDExperiment([mech], items, agent=replay, fixtures={"judge": jp}, ground_truth=GT, repeats=1,
                                store=None, concurrency=concurrency)
            rows += [{**r, "draw": k} for r in episode_rows(await exp.arun(), judge=name, debater=DEBATER)]
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("%s: %d rows", path.relative_to(run), len(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--parts", nargs="*", default=["grid", "trees", "reliability"])
    ap.add_argument("--judges", nargs="*", default=list(INSTANT))
    ap.add_argument("--items", type=int, default=927)
    ap.add_argument("--tree-items", type=int, default=150)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--estimate", action="store_true", help="dry run with simulated models")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / STAGE), simulate=a.estimate or None)
    log = logging.getLogger("decisive")
    judges = {n: INSTANT[n] for n in a.judges}
    if "grid" in a.parts:
        run_sync(grid(run, selected_items(run, a.items), judges, concurrency=a.concurrency, log=log))
    if "trees" in a.parts:
        run_sync(trees(run, selected_items(run, a.tree_items), judges, concurrency=a.concurrency, log=log))
    if "reliability" in a.parts:
        run_sync(reliability(run, judges, n=300, concurrency=a.concurrency, log=log))
    if not a.estimate:
        Ledger(run).set(STAGE, cache_spend(run, STAGE))
        print(json.dumps(Ledger(run).data, indent=1))


if __name__ == "__main__":
    main()
