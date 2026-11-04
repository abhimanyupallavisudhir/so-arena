"""Empirical incentive compatibility by prompt optimization.

ASD samples two behaviours per item by instruction: argue the true answer, argue a false one. Prompt optimization
samples further: an optimizer model (Claude Opus 5.5) that knows the protocol's rules and rewards writes
system-prompt strategies for the agent (gpt-5-mini), scored only by the protocol's reward (what training would
optimize), never by the truth. Two searches per (protocol, judge), as in the OversightArena proposal:

* **honest**: strategies for the agent assigned the true answer, under the ``honest`` directive (argue sincerely);
* **deceptive**: strategies for the agent assigned a false answer, under the ``deceptive`` directive (win anyway:
  misleading framing, selective or fabricated evidence, exploiting the judge).

Each search runs GEPA-style reflection (the optimizer reads the best strategy's transcripts and rewards) for 5
rounds of 4 candidates after the empty seed, on 36 training items. Every searched strategy is a behaviour with
ground truth $v$ by construction (1 on the honest side, 0 on the deceptive side) and reward $u$. EIC is computed
on 80 held-out items, never the training items, so selection noise does not inflate it:

* **EIC along the optimization path**: at search round $k$, the best honest and best deceptive strategy so far
  (chosen on training reward) are scored on the held-out items; ASD$_k$ = mean held-out reward of the honest
  minus the deceptive one, paired by item;
* **population EIC**: $\\mathrm{Cov}(u, v)$ over every strategy the two searches produced (training items).

Debaters' seats are balanced: on each item the searched agent speaks first or second by a fixed hash (weak judges
favour the last speaker by 25-35 points). The other agents play the base policy on the other side.

Outputs ``<run>/eic_po/<judge>/<protocol>.json``: every strategy with per-item rewards, values and verdicts on
training and held-out items. Analysis: ``decisive_eic.py``.

    python experiments/decisive_eic_po.py --run runs/decisive [--judges gpt-4o-mini] [--protocols debate_seq]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np

import so_arena as soa
from so_arena.core.policy import stable_hash
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.samplers.prompt_search import CapturePolicy, PromptSearch, _cid

from decisive_common import (DEBATERS, JUDGES, Ledger, cache_spend, debater_policy, judge_policy, map_keys,
                             selected_items)
from decisive_protocols import GT, LEAD_ROLES, protocols

OPTIMIZER = "anthropic/claude-opus-5-5"
DEBATER = "gpt-5-mini"
ROUNDS, PER_ROUND, N_TRAIN, N_TEST = 5, 4, 36, 80
OTHER = {"true": "false", "false": "true"}


def split(run: Path, n_train: int = N_TRAIN, n_test: int = N_TEST):
    """Training and held-out items from the 927-item extension range (positions 320-436 of the fixed order),
    disjoint from the 320-item grid and its optimization subsets."""
    items = selected_items(run, 320 + n_train + n_test)
    return items[320:320 + n_train], items[320 + n_train:]


def factory(strategy: str):
    return debater_policy(DEBATERS[DEBATER], strategy or None)


class ArmSearch(PromptSearch):
    """Prompt search for the agent on one side (``arm``) of a protocol. The other agents play the base policy on
    the other side; in debate the searched agent's seat alternates by item."""

    def __init__(self, mech, items, *, arm: str, judge, **kw):
        self.seats = LEAD_ROLES[mech.name] if mech.name.startswith("debate") else LEAD_ROLES[mech.name][:1]
        self.judge, self.base = judge, factory("")
        self.agent_roles = [r for r, s in mech.role_specs().items() if s.kind == "agent"]
        super().__init__(mech, items, role=self.seats[0], policy_factory=factory, others={}, arms=[arm],
                         optimizer=OPTIMIZER, ground_truth=GT, **kw)
        self.arm = arm

    def seat_map(self, items) -> dict[str, str]:
        """Seats balanced exactly within an item set: items ranked by a fixed hash, seats alternating."""
        ranked = sorted(items, key=lambda it: stable_hash("eic-seat", it.id))
        return {it.id: self.seats[i % len(self.seats)] for i, it in enumerate(ranked)}

    async def evaluate_strategy(self, strategy, items=None, *, capture=False):
        items = list(items if items is not None else self.items)
        seat_of = self.seat_map(items)
        policy = self.policy_factory(strategy)
        cap = CapturePolicy(policy) if capture else None
        async def one_seat(seat):
            its = [it for it in items if seat_of[it.id] == seat]
            if not its:
                return []
            players = {"judge": PlayerSpec(policy=self.judge)}
            for r in self.agent_roles:
                players[r] = (PlayerSpec(policy=cap or policy, stance=self.arm, label=_cid(strategy)) if r == seat
                              else PlayerSpec(policy=self.base, stance=OTHER[self.arm], label="base"))
            prof = Profile(name=f"{_cid(strategy)}:{self.arm}:{seat}", players=players,
                           tags={"strategy_id": _cid(strategy), "directive": self.directive_name, "seat": seat})
            return await run_episodes(self.mechanism, its, [prof], ground_truth=self.ground_truth,
                                      concurrency=self.concurrency, seed=self.seed)

        eps_all = [e for eps in await asyncio.gather(*[one_seat(seat) for seat in self.seats]) for e in eps]
        if cap is not None and cap.captured is not None and self.example_prompt is None:
            self.example_prompt = "\n\n".join(f"[{m.role}]\n{m.content}" for m in cap.captured.prompt)[:6000]
        # the reflective optimizer reads episodes from the searched agent's first seat: put those first
        eps_all.sort(key=lambda e: e.tags.get("seat") != self.seats[0])
        rewards = [e.rewards.get(e.tags["seat"]) if e.error is None else None for e in eps_all]
        values = [e.value(e.tags["seat"]) for e in eps_all]
        return rewards, values, eps_all


def episode_record(e) -> dict:
    gt = e.ground_truth or {}
    seat = e.tags.get("seat")
    return {"item": e.item_id, "seat": seat, "reward": e.rewards.get(seat) if e.error is None else None,
            "value": e.value(seat) if e.error is None else None, "judge_correct": gt.get("judge_correct"),
            "p_true": gt.get("judge_p_true"), "error": bool(e.error)}


async def run_cell(mech, j: str, train, test, out: Path, concurrency: int, rounds: int = ROUNDS,
                   per_round: int = PER_ROUND) -> dict:
    jp = judge_policy(JUDGES[j])
    searches = {arm: ArmSearch(mech, train, arm=arm, judge=jp, directive=d, algorithm="reflective",
                               iterations=rounds, candidates_per_iter=per_round, concurrency=concurrency,
                               max_strategy_words=200)
                for arm, d in (("true", "honest"), ("false", "deceptive"))}
    results = dict(zip(searches, await asyncio.gather(*[s.arun() for s in searches.values()])))
    record: dict = {"protocol": mech.name, "judge": j, "debater": DEBATER, "train": [it.id for it in train],
                    "test": [it.id for it in test], "searches": {}}

    async def finish(arm, s):
        res = results[arm]
        train_c = [c for c in res.candidates if c.split == "train"]
        # the best strategy (on training reward) after each round, scored on the held-out items
        path, best = [], None
        for k in range(rounds + 1):
            pool = [c for c in train_c if c.iteration <= k]
            best = max(pool, key=lambda c: c.mean_reward) if pool else best
            path.append(best.id)
        distinct = list(dict.fromkeys(path))

        async def held_out(cid):
            c = next(c for c in train_c if c.id == cid)
            _, _, eps = await s.evaluate_strategy(c.strategy, test)
            return cid, [episode_record(e) for e in eps]

        held = dict(await asyncio.gather(*[held_out(cid) for cid in distinct]))
        return arm, {
            "directive": s.directive_name,
            "strategies": [{"id": c.id, "iteration": c.iteration, "parent": c.parent, "strategy": c.strategy,
                            "rationale": c.rationale, "train": [episode_record(e) for e in s.episodes.get(c.id, [])]}
                           for c in train_c],
            "best_path": path, "heldout": held, "optimizer_calls": s.optimizer_calls}

    record["searches"] = dict(await asyncio.gather(*[finish(arm, s) for arm, s in searches.items()]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    return record


def summary(rec: dict) -> str:
    def mean_r(eps):
        r = [e["reward"] for e in eps if e["reward"] is not None]
        return float(np.mean(r)) if r else float("nan")

    h, d = rec["searches"]["true"], rec["searches"]["false"]
    asd = [mean_r(h["heldout"][a]) - mean_r(d["heldout"][b]) for a, b in zip(h["best_path"], d["best_path"])]
    return f"{rec['judge']} {rec['protocol']}: held-out ASD along the search {np.round(asd, 3).tolist()}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--judges", nargs="*", default=list(JUDGES))
    ap.add_argument("--protocols", nargs="*", default=[m.name for m in protocols()])
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--budget", type=float, default=1100.0)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--per-round", type=int, default=PER_ROUND)
    ap.add_argument("--train", type=int, default=N_TRAIN)
    ap.add_argument("--test", type=int, default=N_TEST)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "openai", "anthropic", "inspect_ai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    map_keys()
    run = Path(a.run)
    soa.configure(cache_dir=str(run / "cache" / "eic_po"))
    train, test = split(run, a.train, a.test)
    ledger = Ledger(run)
    log = logging.getLogger("decisive")
    mechs = [m for m in protocols(fixed_pe=True) if m.name in a.protocols]

    async def all_cells():
        async def one(m, j):
            out = run / "eic_po" / j / f"{m.name}.json"
            if out.exists():
                return
            if cache_spend(run, "eic_po") > a.budget:
                log.warning("budget reached; skipping %s %s", j, m.name)
                return
            rec = await run_cell(m, j, train, test, out, a.concurrency, a.rounds, a.per_round)
            ledger.set("eic_po", cache_spend(run, "eic_po"))
            log.info("%s; stage $%.2f", summary(rec), ledger.data["eic_po"])

        await asyncio.gather(*[one(m, j) for j in a.judges for m in mechs])

    run_sync(all_cells())
    ledger.set("eic_po", cache_spend(run, "eic_po"))
    print(json.dumps(ledger.data, indent=1))


if __name__ == "__main__":
    main()
