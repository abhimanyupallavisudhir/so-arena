"""Oversight protocols on the conceptual-reasoning testbed: does argument help a weak judge find the truth?

Every protocol runs on the same items under instructed arms (the agent argues the true answer on one arm and the
false one on the other), with a weak judge and stronger agents, plus two references: the weak judge alone and
the strong model answering alone (the sandwich gap a protocol should close). Reported per protocol and family:

* judge accuracy - is the protocol's final verdict right (both arms pooled)?
* ASD - mean reward when arguing the truth minus when arguing a falsehood, in the protocol's own units;
* concordance - P(reward when arguing the truth > reward when arguing the falsehood) over items: ASD made
  scale-free, so protocols that pay in different units (log scores, prover-estimator payoffs) compare.

    python experiments/oversight_testbed.py --estimate                  # dry run: $ per protocol
    python experiments/oversight_testbed.py --out runs/oversight-v1     # run (cached and resumable)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import so_arena as soa
from so_arena.analysis.frames import role_frame
from so_arena.analysis.metrics import asd, judge_accuracy, with_parse_status
from so_arena.core.ground_truth import JudgeCorrectness, StanceValue
from so_arena.domains import get_domain
from so_arena.mechanisms import (Consultancy, Debate, DirectJudge, ProverEstimatorDebate, RecursiveDebate)
from so_arena.samplers.arms import ASDExperiment

FAMILIES = ["decisions/newcomb", "decisions/lesion", "decisions/twin", "decisions/beauty", "socialchoice/winners",
            "socialchoice/rankings", "numbers/counts", "numbers/collatz", "rlhack/sensor", "agents/auction"]
LEAD_ROLES = {"direct": ["agent"], "consultancy": ["consultant"], "debate": ["debater_a", "debater_b"],
              "prover_estimator": ["prover"], "recursive_debate": ["prover"], "direct_strong": ["agent"]}


def mechanisms():
    return [DirectJudge(), Consultancy(rounds=2), Debate(rounds=2), RecursiveDebate(depth=2, width=3),
            ProverEstimatorDebate(depth=2, width=3, rho=2.0)]


def agent_policy(model: str, judge: bool = False):
    """Reasoning models (gpt-5*, o*) spend max_tokens on hidden reasoning; give them room, at low effort."""
    reasoning = any(t in model for t in ("gpt-5", "/o3", "/o4"))
    if reasoning:
        return soa.LLMPolicy(model, temperature=None, max_tokens=8000, reasoning_effort="low", cot=False)
    if judge:
        return soa.LLMPolicy(model, cot=True, temperature=0, max_tokens=2000)
    return soa.LLMPolicy(model, temperature=0.7, max_tokens=1500)


def concordance(frame: pd.DataFrame, roles: list[str]) -> tuple[float, int]:
    d = frame[frame.role.isin(roles) & frame.value.notna()]
    wins = []
    for _, g in d.groupby("item_id"):
        t, f = g[g.value > 0].reward, g[g.value < 0].reward
        if len(t) and len(f):
            diff = t.mean() - f.mean()
            wins.append(1.0 if diff > 0 else 0.5 if diff == 0 else 0.0)
    return (float(np.mean(wins)) if wins else float("nan")), len(wins)


def summarize(episodes, by_family: bool) -> pd.DataFrame:
    frame = with_parse_status(role_frame(episodes), episodes)
    frame["family"] = frame.item_id.str.rsplit("-s", n=1).str[0]
    rows = []
    groups = frame.groupby(["mechanism", "family"]) if by_family else frame.groupby(["mechanism"])
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        mech = key[0]
        roles = LEAD_ROLES.get(mech.split("(")[0], None) or list(g.role.unique())
        a = asd(g, roles=roles, by=(), parse_warn=None)
        ja = judge_accuracy(g, by=())
        conc, n = concordance(g, roles)
        rows.append({"mechanism": mech, **({"family": key[1]} if by_family else {}),
                     "accuracy": float(ja.accuracy.iloc[0]) if not ja.empty else np.nan,
                     "asd": float(a.asd.iloc[0]) if not a.empty else np.nan,
                     "concordance": conc, "items": n,
                     "parse_fail": float((g.drop_duplicates("episode_id").judge_parse_ok == False).mean())})  # noqa: E712
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--per-family", type=int, default=12)
    ap.add_argument("--agents", nargs="+", default=["openai/gpt-4.1-mini"],
                    help="agent models; several give a debater-strength sweep with the same judge")
    ap.add_argument("--judge", default="openai/gpt-4.1-nano")
    ap.add_argument("--out", default="runs/oversight-testbed")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--concurrency", type=int, default=24)
    a = ap.parse_args(argv)
    soa.configure(cache_dir=".cache/so_arena", simulate=a.estimate or None, concurrency=a.concurrency)
    items = get_domain("testbed", families=a.families, per_family=a.per_family).load()
    out = Path(a.out)
    # judges reason before answering: testbed items need working out, and an instant answer is mostly position bias
    judge = soa.LLMPolicy(a.judge, cot=True, temperature=0, max_tokens=2000)
    gt = [StanceValue(), JudgeCorrectness()]
    eps = []
    for model in a.agents:
        tag = model.split("/")[-1]
        agent = agent_policy(model)
        exp = ASDExperiment(mechanisms(), items, agent=agent, fixtures={"judge": judge}, ground_truth=gt,
                            store=None if a.estimate else str(out / tag / "protocols"), concurrency=a.concurrency)
        got = exp.run()
        # reference: the agent model answering alone (as the judge of a direct protocol)
        strong = ASDExperiment([DirectJudge(name="direct_strong")], items, agent=agent,
                               fixtures={"judge": agent_policy(model, judge=True)},
                               ground_truth=gt, store=None if a.estimate else str(out / tag / "strong"),
                               concurrency=a.concurrency)
        got += strong.run()
        for e in got:
            e.tags["agent_model"] = tag
        eps += got
    if a.estimate:
        cost: dict[str, float] = {}
        for e in eps:
            cost[e.mechanism] = cost.get(e.mechanism, 0.0) + sum(u.cost_usd for u in e.usage.values())
        print(json.dumps({k: round(v, 2) for k, v in cost.items()}, indent=1), "total", round(sum(cost.values()), 2))
        return
    overall = pd.concat([summarize([e for e in eps if e.tags["agent_model"] == t], by_family=False).assign(agent=t)
                         for t in dict.fromkeys(e.tags["agent_model"] for e in eps)])
    fam = pd.concat([summarize([e for e in eps if e.tags["agent_model"] == t], by_family=True).assign(agent=t)
                     for t in dict.fromkeys(e.tags["agent_model"] for e in eps)])
    out.mkdir(parents=True, exist_ok=True)
    overall.to_csv(out / "summary.csv", index=False)
    fam.to_csv(out / "summary_by_family.csv", index=False)
    cost = sum(sum(u.cost_usd for u in e.usage.values()) for e in eps)
    print(overall.round(3).to_string(index=False))
    print(f"cost ${cost:.2f}")


if __name__ == "__main__":
    main()
