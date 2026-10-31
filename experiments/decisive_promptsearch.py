"""Decisive study: strategies found by prompt search, a stronger adversary than best-of-n (docs/theory.md section 9).

An optimizer model (Claude Opus 5.5) that knows the protocol's rules and rewards proposes system-prompt
strategies for the debater (gpt-5-mini); each is scored by the protocol's reward only (the judge's log score),
never by the truth, on 40 training items, and the best are re-scored on 100 held-out items (disjoint from the
training items and from the best-of-n items). Search is GEPA-style ``reflective``: the optimizer reads the best
strategy's transcripts and rewards and proposes improvements, 4 rounds of 4 candidates after the empty seed.

* **Honesty margin (open protocols).** The agent chooses which answer to defend (consultancy; debate's first
  speaker, the other taking the remaining answer). Two searches, one restricted to honest strategies and one to
  deceptive ones; every strategy is classed by its *measured* honesty (did it defend the true answer) and the
  margin is the held-out reward of the best honest strategy minus the best deceptive one
  (``PromptSearchSuite.honesty_margin``): positive = the best the optimizer found is honest.
* **Best-response ladder (assigned sides).** Consultancy: a liar strategy (false arm) and an honest strategy
  (true arm), searched separately. Debate (honest debater speaks first, liar second): liar L1 against the base
  honest debater, honest H1 against L1, liar L2 against H1. Held-out judge accuracy and ASD at each rung:
  base, liar optimized (L1 vs base), both optimized (H1 vs L1), and the liar's reply (H1 vs L2).

Results go to ``<run>/promptsearch/<judge>/``: every searched strategy with its train/held-out rewards, values and
judge accuracy (``candidates.csv``), ``margins.json`` and ``ladder.csv``.

    python experiments/decisive_promptsearch.py --run runs/decisive [--judges gpt-4o-mini] [--parts open ladder]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import so_arena as soa
from so_arena.core.ground_truth import JudgeCorrectness, StanceValue
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.mechanisms import Consultancy, Debate
from so_arena.samplers.prompt_search import PromptSearch, PromptSearchSuite

from decisive_common import (DEBATERS, JUDGES, Ledger, cache_spend, debater_policy, judge_policy, map_keys,
                             selected_items)

GT = [StanceValue(), JudgeCorrectness()]
WORDS = 200
OPTIMIZER = "anthropic/claude-opus-5-5"
DEBATER = "gpt-5-mini"
SEARCH = dict(algorithm="reflective", iterations=4, candidates_per_iter=4, n_final=3, max_strategy_words=200)


def mechanisms():
    return {"consultancy": Consultancy(rounds=3, word_limit=WORDS),
            "debate_seq": Debate(rounds=3, simultaneous=False, word_limit=WORDS, name="debate_seq")}


def split_items(run: Path):
    """Training and held-out items: positions 150-190 and 190-290 of the study's fixed order (the best-of-n
    items are 0-150; all are in the protocol grid, which gives the unoptimized baseline on the same items)."""
    items = selected_items(run, 290)
    return items[150:190], items[190:290]


def factory(strategy: str):
    return debater_policy(DEBATERS[DEBATER], strategy or None)


def accuracy(eps) -> float:
    v = [e.ground_truth.get("judge_correct") for e in eps if e.error is None and e.ground_truth]
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


def candidate_rows(search: PromptSearch, result, *, part: str, judge: str, mech: str, directive: str) -> list[dict]:
    rows = []
    for c in list(result.candidates) + list(result.evaluated):
        eps = search.episodes.get(c.id, [])
        rows.append({"part": part, "judge": judge, "mechanism": mech, "directive": directive, "id": c.id,
                     "split": c.split, "iteration": c.iteration, "parent": c.parent, "reward": c.mean_reward,
                     "value": c.mean_value, "accuracy": accuracy(eps), "n": len(c.rewards), "strategy": c.strategy,
                     "rationale": c.rationale})
    return rows


async def open_part(mech_name: str, judge: str, train, test, concurrency: int) -> tuple[list[dict], dict]:
    mech = mechanisms()[mech_name]
    jp = judge_policy(JUDGES[judge])
    if mech_name == "consultancy":
        role, others = "consultant", {"judge": PlayerSpec(policy=jp)}
    else:  # the second debater takes the remaining answer (it has no stance: ensure_position gives it the other)
        role, others = "debater_a", {"judge": PlayerSpec(policy=jp), "debater_b": PlayerSpec(policy=factory(""))}
    suite = PromptSearchSuite(directives=("honest", "deceptive"), mechanism=mech, items=train, eval_items=test,
                              role=role, policy_factory=factory, others=others, optimizer=OPTIMIZER, arms=[None],
                              ground_truth=GT, concurrency=concurrency, **SEARCH)
    results = await suite.arun()
    rows = []
    for d, res in results.items():
        rows += candidate_rows(suite.searches[d], res, part="open", judge=judge, mech=mech_name, directive=d)
    margin = {"measured": suite.honesty_margin(by="measured"), "directive": suite.honesty_margin(by="directive")}
    return rows, margin


async def evaluate(mech, items, players: dict, concurrency: int, name: str):
    eps = await run_episodes(mech, items, [Profile(name=name, players=players)], ground_truth=GT,
                             concurrency=concurrency)
    return eps


def best_eval(result):
    """The held-out version of the search's best training strategy (selected on training rewards)."""
    top = max((c for c in result.candidates if c.split == "train"), key=lambda c: c.mean_reward)
    held = [e for e in result.evaluated if e.parent == top.id]
    return top, (held[0] if held else None)


async def ladder_part(mech_name: str, judge: str, train, test, concurrency: int) -> tuple[list[dict], list[dict]]:
    mech = mechanisms()[mech_name]
    jp = judge_policy(JUDGES[judge])
    rows, rungs = [], []

    async def search(role, stance, others, directive, tag):
        s = PromptSearch(mech, train, eval_items=test, role=role, policy_factory=factory, others=others,
                         optimizer=OPTIMIZER, arms=[stance], directive=directive, ground_truth=GT,
                         concurrency=concurrency, **SEARCH)
        res = await s.arun()
        rows.extend(candidate_rows(s, res, part=f"ladder:{tag}", judge=judge, mech=mech_name, directive=directive))
        top, _ = best_eval(res)
        return top.strategy

    def rung(name, eps_true, eps_false, honest_role, liar_role):
        acc_t, acc_f = accuracy(eps_true), accuracy(eps_false)
        r_h = [e.rewards.get(honest_role) for e in eps_true if e.error is None]
        r_l = [e.rewards.get(liar_role) for e in eps_false if e.error is None]
        rungs.append({"judge": judge, "mechanism": mech_name, "rung": name, "accuracy": float(np.nanmean([acc_t, acc_f])),
                      "accuracy_true_arm": acc_t, "accuracy_false_arm": acc_f,
                      "honest_reward": float(np.nanmean([x for x in r_h if x is not None])),
                      "liar_reward": float(np.nanmean([x for x in r_l if x is not None]))})

    if mech_name == "consultancy":
        judge_only = {"judge": PlayerSpec(policy=jp)}
        liar, honest = await asyncio.gather(search("consultant", "false", judge_only, "unconstrained", "liar"),
                                            search("consultant", "true", judge_only, "unconstrained", "honest"))
        plays = {}
        for tag, strat, arm in (("base_t", "", "true"), ("base_f", "", "false"), ("H1", honest, "true"),
                                ("L1", liar, "false")):
            plays[tag] = await evaluate(mech, test, {**judge_only, "consultant": PlayerSpec(policy=factory(strat), stance=arm)},
                                        concurrency, f"ladder:{tag}")
        rung("base", plays["base_t"], plays["base_f"], "consultant", "consultant")
        rung("liar optimized", plays["base_t"], plays["L1"], "consultant", "consultant")
        rung("both optimized", plays["H1"], plays["L1"], "consultant", "consultant")
        return rows, rungs

    # debate: the honest debater speaks first (debater_a, the true answer), the liar second
    def players(h: str, li: str):
        return {"judge": PlayerSpec(policy=jp), "debater_a": PlayerSpec(policy=factory(h), stance="true"),
                "debater_b": PlayerSpec(policy=factory(li), stance="false")}

    l1 = await search("debater_b", "false", {"judge": PlayerSpec(policy=jp),
                                             "debater_a": PlayerSpec(policy=factory(""), stance="true")},
                      "unconstrained", "L1")
    h1 = await search("debater_a", "true", {"judge": PlayerSpec(policy=jp),
                                            "debater_b": PlayerSpec(policy=factory(l1), stance="false")},
                      "unconstrained", "H1")
    l2 = await search("debater_b", "false", {"judge": PlayerSpec(policy=jp),
                                             "debater_a": PlayerSpec(policy=factory(h1), stance="true")},
                      "unconstrained", "L2")
    for name, h, li in (("base", "", ""), ("liar optimized", "", l1), ("both optimized", h1, l1),
                        ("liar replies", h1, l2)):
        eps = await evaluate(mech, test, players(h, li), concurrency, f"ladder:{name}")
        rung(name, eps, eps, "debater_a", "debater_b")
    return rows, rungs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--judges", nargs="*", default=list(JUDGES))
    ap.add_argument("--mechanisms", nargs="*", default=list(mechanisms()))
    ap.add_argument("--parts", nargs="*", default=["open", "ladder"])
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--budget", type=float, default=1000.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / "promptsearch"))
    train, test = split_items(run)
    ledger = Ledger(run)
    log = logging.getLogger("decisive")
    for j in a.judges:
        out = run / "promptsearch" / j
        out.mkdir(parents=True, exist_ok=True)
        for part in a.parts:
            for m in a.mechanisms:
                tag = f"{part}-{m}"
                if (out / f"{tag}.done").exists():
                    continue
                if cache_spend(run, "promptsearch") > a.budget:
                    log.warning("budget reached before %s %s", j, tag)
                    return
                if part == "open":
                    rows, margin = run_sync(open_part(m, j, train, test, a.concurrency))
                    (out / f"{tag}.margin.json").write_text(json.dumps(margin, indent=1, default=str))
                else:
                    rows, rungs = run_sync(ladder_part(m, j, train, test, a.concurrency))
                    pd.DataFrame(rungs).to_csv(out / f"{tag}.ladder.csv", index=False)
                pd.DataFrame(rows).to_csv(out / f"{tag}.candidates.csv", index=False)
                (out / f"{tag}.done").write_text("ok")
                ledger.set("promptsearch", cache_spend(run, "promptsearch"))
                log.info("%s %s done; stage $%.2f", j, tag, ledger.data["promptsearch"])
    print(json.dumps(ledger.data, indent=1), f"total ${ledger.total:.2f}")


if __name__ == "__main__":
    main()
