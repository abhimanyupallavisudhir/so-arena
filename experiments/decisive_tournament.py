"""Decisive study: a debater tournament. Does judge accuracy rise with debater persuasiveness (Khan et al. 2024)
when the gap between debaters and judge is a capability gap rather than hidden information?

Eight debater models, from gpt-4.1-mini to Claude Opus 5.5, play 3-round sequential debates (200 words a
speech, as in ``decisive_protocols.py``) on the gap-selected items: self-play on every item, and every ordered
cross (honest, liar) pairing on a fixed half of the items (both directions of a pair on the same half). Positions are balanced: on each item and pairing, the honest debater speaks first or second
by a fixed hash. Debate transcripts do not depend on the judge, so the first judge's run pays for them and the
other two judges replay them from the cache. Each debater is also run

* alone (``solo``: 2 samples per item, its knowledge of the answer), and
* as a consultant (3 rounds; the judge questions it; both arms, every judge).

Rows go to ``<run>/tournament/rows.csv`` (one per episode). ``decisive_tournament_report.py`` fits Elo ratings
from the judges' verdicts and plots accuracy against persuasiveness.

    python experiments/decisive_tournament.py --run runs/decisive --items 200 [--estimate] [--budget 1500]
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
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.core.store import RunStore
from so_arena.mechanisms import Consultancy, Debate, DirectJudge
from so_arena.core.policy import stable_hash

from decisive_common import (JUDGES, Ledger, chunks, debater_policy, episode_cost, judge_policy, map_keys,
                             selected_items, solo_policy)

GT = [StanceValue(), JudgeCorrectness()]
WORDS = 200
TOURNAMENT = {
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-5-mini": "openai/gpt-5-mini",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5",
    "claude-sonnet-5": "anthropic/claude-sonnet-5",
    "gpt-5": "openai/gpt-5",
    "claude-opus-5-5": "anthropic/claude-opus-5-5",
}


def debate():
    return Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq")


def consultancy():
    return Consultancy(rounds=3, word_limit=WORDS)


def honest_first(item_id: str, honest: str, liar: str) -> bool:
    """Whether the honest debater speaks first (debater_a) on this item and pairing: balanced by a fixed hash."""
    return stable_hash("position", item_id, honest, liar) % 2 == 0


def plays(item_id: str, honest: str, liar: str, share: float) -> bool:
    """Whether a pairing is played on an item. Self-play always; a cross pairing on a fixed ``share`` of the
    items, the same items for both directions of the pair (so its role-specific ratings are paired)."""
    if honest == liar or share >= 1:
        return True
    a, b = sorted((honest, liar))
    return (stable_hash("pair", item_id, a, b) % 1000) < share * 1000


def pair_profiles(honest: str, liar: str, judge) -> list[Profile]:
    """The two seatings of one (honest, liar) pairing. Stances are explicit ('true' / 'false'): with more than
    two answers the opponent of a liar must argue the true answer, not another wrong one."""
    h, li = debater_policy(TOURNAMENT[honest]), debater_policy(TOURNAMENT[liar])
    tags = {"honest": honest, "liar": liar}
    return [Profile(name=f"{honest}>{liar}|honest=a", tags={**tags, "honest_role": "debater_a"},
                    players={"debater_a": PlayerSpec(policy=h, stance="true", label="honest"),
                             "debater_b": PlayerSpec(policy=li, stance="false", label="liar"),
                             "judge": PlayerSpec(policy=judge)}),
            Profile(name=f"{honest}>{liar}|honest=b", tags={**tags, "honest_role": "debater_b"},
                    players={"debater_a": PlayerSpec(policy=li, stance="false", label="liar"),
                             "debater_b": PlayerSpec(policy=h, stance="true", label="honest"),
                             "judge": PlayerSpec(policy=judge)})]


def consult_profiles(model: str, judge) -> list[Profile]:
    pol = debater_policy(TOURNAMENT[model])
    return [Profile(name=f"{model}|{arm}", tags={"consultant": model, "arm": arm},
                    players={"consultant": PlayerSpec(policy=pol, stance=arm, label=f"argue_{arm}"),
                             "judge": PlayerSpec(policy=judge)}) for arm in ("true", "false")]


def rows_of(eps, *, judge: str, kind: str) -> list[dict]:
    out = []
    for e in eps:
        if e.error is not None:
            continue
        gt = e.ground_truth or {}
        values = gt.get("role_values") or {}
        row = {"kind": kind, "judge": judge, "item": e.item_id, "episode": e.id, "repeat": e.repeat,
               "correct": gt.get("judge_correct"), "p_true": gt.get("judge_p_true"), "cost": episode_cost(e),
               "parse_ok": (e.outcome.data or {}).get("judge_parse_ok") if e.outcome else None, **e.tags}
        if kind == "debate":
            hr = e.tags["honest_role"]
            lr = "debater_b" if hr == "debater_a" else "debater_a"
            row.update({"honest_reward": e.rewards.get(hr), "liar_reward": e.rewards.get(lr),
                        "honest_value": values.get(hr), "liar_value": values.get(lr)})
        elif kind == "consultancy":
            row.update({"reward": e.rewards.get("consultant"), "value": values.get("consultant")})
        out.append(row)
    return out


async def run_chunk(items, run: Path, *, estimate: bool, concurrency: int, models: list[str],
                    cross_share: float = 0.5) -> list[dict]:
    st = (lambda *p: None) if estimate else (lambda *p: RunStore(str(run / "tournament" / Path(*p))))
    rows: list[dict] = []

    async def debates(j: str, h: str, li: str):
        profs = pair_profiles(h, li, judge_policy(JUDGES[j]))
        mine = [it for it in items if plays(it.id, h, li, cross_share)]
        first = [it for it in mine if honest_first(it.id, h, li)]
        second = [it for it in mine if not honest_first(it.id, h, li)]
        for prof, its in ((profs[0], first), (profs[1], second)):
            if its:
                eps = await run_episodes(debate(), its, [prof], ground_truth=GT, concurrency=concurrency,
                                         store=st(j, "debate", prof.name.replace("|", "_").replace(">", "__")))
                rows.extend(rows_of(eps, judge=j, kind="debate"))

    async def consults(j: str, m: str):
        eps = await run_episodes(consultancy(), items, consult_profiles(m, judge_policy(JUDGES[j])), ground_truth=GT,
                                 concurrency=concurrency, store=st(j, "consultancy", m))
        rows.extend(rows_of(eps, judge=j, kind="consultancy"))

    async def solo(m: str):
        prof = Profile(name=f"solo|{m}", tags={"model": m},
                       players={"judge": PlayerSpec(policy=solo_policy(TOURNAMENT[m]))})
        eps = await run_episodes(DirectJudge(name="direct_strong"), items, [prof], ground_truth=GT, repeats=2,
                                 concurrency=concurrency, store=st("solo", m))
        rows.extend(rows_of(eps, judge="-", kind="solo"))

    first, *others = JUDGES
    pairs = [(h, li) for h in models for li in models]
    # phase A pays for every debate transcript (first judge), every consultancy and every solo answer
    await asyncio.gather(*[debates(first, h, li) for h, li in pairs], *[consults(j, m) for j in JUDGES for m in models],
                         *[solo(m) for m in models])
    # phase B: the other judges on the same debates (agent calls replay from the cache)
    await asyncio.gather(*[debates(j, h, li) for j in others for h, li in pairs])
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--items", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--concurrency", type=int, default=6, help="per (pairing, judge)")
    ap.add_argument("--models", nargs="*", default=list(TOURNAMENT))
    ap.add_argument("--cross-share", type=float, default=0.5, help="share of items each cross pairing plays")
    ap.add_argument("--budget", type=float, default=1500.0, help="this stage's cap in $")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / "tournament"), simulate=a.estimate or None)
    items = selected_items(run, a.items)
    ledger = Ledger(run)
    out = run / "tournament"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.csv"
    log = logging.getLogger("decisive")
    spent = float(pd.read_csv(rows_path).cost.sum()) if rows_path.exists() and not a.estimate else 0.0
    per_item, est_rows = None, []
    for i, chunk in enumerate(chunks(items, a.chunk)):
        if per_item is not None and spent + per_item * len(chunk) > a.budget:
            log.warning("stopping before chunk %d: $%.0f spent, next ~$%.0f, budget $%.0f", i, spent,
                        per_item * len(chunk), a.budget)
            break
        rows = run_sync(run_chunk(chunk, run, estimate=a.estimate, concurrency=a.concurrency, models=a.models,
                                   cross_share=a.cross_share))
        new = sum(r["cost"] for r in rows)
        if new > 0:
            per_item = max(per_item or 0.0, new / len(chunk))
        if a.estimate:
            est_rows += rows
            spent += new
            break  # one chunk is enough to extrapolate
        prev = pd.read_csv(rows_path) if rows_path.exists() else pd.DataFrame()
        df = pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates(["episode", "judge"], keep="last")
        df.to_csv(rows_path, index=False)
        spent = float(df.cost.sum())
        ledger.set("tournament", spent)
        log.info("chunk %d done: %d items, stage $%.2f, study $%.2f", i, len(chunk), spent, ledger.total)
    if a.estimate:
        df = pd.DataFrame(est_rows)
        n = min(a.chunk, len(items))
        print(df.groupby("kind").cost.sum().round(2).to_string(),
              f"\n{n} items: ${df.cost.sum():.2f}; {len(items)} items: ~${df.cost.sum() / n * len(items):.0f}")
        return
    print(json.dumps(ledger.data, indent=1), f"total ${ledger.total:.2f}")


if __name__ == "__main__":
    main()
