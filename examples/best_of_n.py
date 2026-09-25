"""One step of optimisation: which of N sampled behaviours does the mechanism reward, and are
they better? Plots expected ground truth against expected reward as n grows (x: reward, y: GT;
KL(Bo(n) || base) = log n - (n-1)/n reported), per protocol. With ``--tree`` it builds nested
proposal -> critique -> rebuttal trees and evaluates min-max best-of-n (proposer maximises,
critic minimises), as in "Debate with self-play best-of-N optimization".

    OA_EXPERT_MODEL=... OA_JUDGE_MODEL=... python examples/best_of_n.py --domain gsm8k --tasks 30 --n 8
"""

import argparse
from pathlib import Path

import oversight_arena as oa
from oversight_arena.analysis import bon_curve, plots, tree_mesh, trees_from_results
from oversight_arena.channels import EvidencePolicy
from oversight_arena.experiment.profiles import GameTree
from oversight_arena.mechanisms import OpenConsultancy, OpenDebate, ProposerCritic
from oversight_arena.registry import build

FREE = oa.Strategy(name="base", instructions="", params={"temperature": 1.0})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="gsm8k")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--tree", action="store_true")
    ap.add_argument("--out", default="runs/best_of_n")
    args = ap.parse_args()
    out = Path(args.out)
    domain = build("domain", {"type": args.domain, "limit": args.tasks})
    agents = oa.llm_agents(temperature=1.0)
    if not args.tree:
        runs = {"open consultancy": (OpenConsultancy(evidence=EvidencePolicy()), "consultant"),
                "open debate": (OpenDebate(rounds=2, evidence=EvidencePolicy()), "debater_a")}
        curves = []
        for label, (mech, role) in runs.items():
            src = oa.Seeds(n=args.n, strategies={role: FREE}, roles=[role])
            res = oa.Experiment(domain, mech.model_copy(update={"label": label}), agents, src, out=out / label.replace(" ", "_")).run()
            curves.append(bon_curve(res, roles=role, gt="correct", n_values=[1, 2, 4, 8, 16][: 1 + args.n.bit_length() - 1]))
        import pandas as pd

        cur = pd.concat(curves)
        print(cur.round(3).to_string(index=False))
        fig, _ = plots.optimization_frontier(cur, ylabel="P(chosen answer is correct)", xlabel="expected reward (log judge prob.)")
        print(plots.save(fig, out / "bon_frontier.png"))
    else:
        src = GameTree(levels=[("proposer", "proposal", args.n), ("critic", "critique", args.n), ("proposer", "rebuttal", 2)],
                       strategies={"proposer": FREE, "critic": FREE})
        res = oa.Experiment(domain, ProposerCritic(critiques=1, rebuttal=True, evidence=EvidencePolicy()), agents, src,
                            out=out / "tree").run()
        trees = trees_from_results(res, [("proposer", "proposal"), ("critic", "critique"), ("proposer", "rebuttal")])
        ks = sorted({1, 2, args.n})
        grid = [[k, m, 1] for k in ks for m in ks]
        mesh = tree_mesh(trees, grid, maximize=[True, False, True])
        print(mesh.round(3).to_string(index=False))  # payoff = P(accept) for the proposer; gt = P(proposal correct)
