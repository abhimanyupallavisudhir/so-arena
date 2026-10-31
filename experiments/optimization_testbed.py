"""Optimization pressure on the testbed: does a protocol stay truthful when agents are optimized against it?

A protocol can look fine with prompted agents and still reward the wrong thing once agents are optimized for
its reward. This experiment samples pools of candidate arguments and computes, exactly, what best-of-n
selection against the protocol's reward converges to (``samplers.pools``), for

* ``propaganda``: one agent argues an assigned answer; the weak judge reads the single argument;
* ``debate``: two debaters argue opposite answers (one round, simultaneous); the weak judge decides.

Positions are assigned (instructed arms), so the dishonest side is always optimized to mislead, and n is how
hard. The curves report the judge's accuracy and its probability on the truth as n grows - for propaganda
separately for the true and false arms, for debate with the dishonest debater optimized alone and with both
debaters optimized.

    python experiments/optimization_testbed.py --out runs/optimization-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import so_arena as soa
from so_arena.analysis.optimization import optimization_grid
from so_arena.core.ground_truth import JudgeCorrectness, StanceValue
from so_arena.domains import get_domain
from so_arena.mechanisms import Debate, Propaganda
from so_arena.samplers.arms import arm_profiles
from so_arena.samplers.pools import OptimizationExperiment

FAMILIES = ["decisions/newcomb", "decisions/lesion", "decisions/twin", "decisions/beauty", "socialchoice/winners",
            "socialchoice/rankings", "numbers/counts", "numbers/collatz", "rlhack/sensor", "agents/auction"]


def tree_cost(t) -> float:
    def c(u):
        return (u.get("cost_usd", 0.0) if isinstance(u, dict) else getattr(u, "cost_usd", 0.0)) or 0.0
    return float(sum(c(u) for u in t.usage.values()))


def run(mech, items, agent, judge, pools, gt):
    trees = {}
    for prof in arm_profiles(mech, agent=agent, fixtures={"judge": judge}):
        exp = OptimizationExperiment(mech, items, prof, pool_sizes=pools, ground_truth=gt, concurrency=16)
        trees[prof.tags["arm"]] = exp.run()
    return trees


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--agent", default="openai/gpt-4.1-mini")
    ap.add_argument("--judge", default="openai/gpt-4.1-nano")
    ap.add_argument("--pool", type=int, default=8, help="candidate arguments per decision (propaganda)")
    ap.add_argument("--debate-pool", type=int, default=4, help="candidate speeches per debater (k^2 leaves)")
    ap.add_argument("--out", default="runs/optimization-testbed")
    ap.add_argument("--estimate", action="store_true")
    a = ap.parse_args(argv)
    soa.configure(cache_dir=".cache/so_arena", simulate=a.estimate or None)
    items = get_domain("testbed", families=a.families, per_family=a.per_family).load()
    agent = soa.LLMPolicy(a.agent, temperature=1.0, max_tokens=1200)
    judge = soa.LLMPolicy(a.judge, cot=True, temperature=0, max_tokens=2000)
    gt = [StanceValue(), JudgeCorrectness()]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cost = 0.0

    prop = run(Propaganda(word_limit=200), items, agent, judge, {"agent": a.pool}, gt)
    ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= a.pool]
    for arm, trees in prop.items():
        g = optimization_grid(trees, {"agent": ns})
        for r in g.itertuples():
            rows.append({"protocol": "propaganda", "arm": arm, "optimized": "agent", "n": r.level_agent,
                         "judge_correct": r.judge_correct, "judge_p_true": getattr(r, "judge_p_true", np.nan),
                         "lo": getattr(r, "judge_correct_ci_low", np.nan), "hi": getattr(r, "judge_correct_ci_high", np.nan)})
        cost += sum(tree_cost(t) for t in trees)

    k = a.debate_pool  # debate trees have k^2 leaves
    deb = run(Debate(rounds=1, word_limit=200), items, agent, judge, {"debater_a": k, "debater_b": k}, gt)
    nd = [n for n in (1, 2, 4, 8) if n <= k]
    for arm, trees in deb.items():
        liar = "debater_b" if arm == "true" else "debater_a"
        honest = "debater_a" if liar == "debater_b" else "debater_b"
        for mode, grid in (("liar", {liar: nd, honest: [1]}), ("both", None)):
            if grid is None:
                gg = optimization_grid(trees, {"debater_a": nd, "debater_b": nd})
                gg = gg[gg.level_debater_a == gg.level_debater_b]
                for r in gg.itertuples():
                    rows.append({"protocol": "debate", "arm": arm, "optimized": "both", "n": r.level_debater_a,
                                 "judge_correct": r.judge_correct, "judge_p_true": getattr(r, "judge_p_true", np.nan),
                         "lo": getattr(r, "judge_correct_ci_low", np.nan), "hi": getattr(r, "judge_correct_ci_high", np.nan)})
            else:
                gg = optimization_grid(trees, grid)
                for r in gg.itertuples():
                    rows.append({"protocol": "debate", "arm": arm, "optimized": "liar",
                                 "n": getattr(r, f"level_{liar}"), "judge_correct": r.judge_correct,
                                 "judge_p_true": getattr(r, "judge_p_true", np.nan),
                                 "lo": getattr(r, "judge_correct_ci_low", np.nan),
                                 "hi": getattr(r, "judge_correct_ci_high", np.nan)})
        cost += sum(tree_cost(t) for t in trees)

    df = pd.DataFrame(rows)
    if a.estimate:
        print(f"estimated cost ${cost:.2f}")
        return
    df.to_csv(out / "curves.csv", index=False)
    (out / "meta.json").write_text(json.dumps({**vars(a), "items": len(items), "cost_usd": cost}, indent=1))
    print(df.round(3).to_string(index=False))
    print(f"cost ${cost:.2f}")




def plot(run: str) -> None:
    """Pooled accuracy (mean of the two arms) against selection pressure: runs/<run>/optimization.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(run)
    df = pd.read_csv(out / "curves.csv")
    series = [("propaganda", "agent", "single argument, liar optimized", "#2a78d6"),
              ("debate", "liar", "debate, liar optimized", "#eb6834"),
              ("debate", "both", "debate, both optimized", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for proto, opt, label, color in series:
        g = df[(df.protocol == proto) & (df.optimized == opt)].groupby("n")[["judge_correct", "lo", "hi"]].mean()
        ax.fill_between(g.index, g.lo, g.hi, color=color, alpha=0.15, lw=0)
        ax.plot(g.index, g.judge_correct, "o-", color=color, lw=2, ms=5, mec="white", mew=1.2, label=label)
    ax.axhline(0.5, color="#52514e", lw=0.8, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("best-of-n selection against the judge's reward", color="#52514e", fontsize=9)
    ax.set_ylabel("weak judge's accuracy\n(mean of both arms; band: mean of arm CIs)", color="#52514e", fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "optimization.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["plot"]:
        plot(sys.argv[2])
    else:
        main()
