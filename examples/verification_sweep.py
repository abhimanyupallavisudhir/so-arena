"""How verification capability shapes incentive compatibility (no LLMs needed).

HiddenBits: experts see 15 hidden bits; the judge sees none; the question is whether at least
half are 1. Advocates can make checkable claims about individual bits. We sweep the verification
budget and the judge's credulity toward unverified claims, for four protocols, with exact Bayesian
judges and programmatic advocates that lie when true favourable bits run out.

    python examples/verification_sweep.py            # writes runs/verification_sweep/
"""

from pathlib import Path

import oversight_arena as oa
from oversight_arena.analysis import plots
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Consultancy, Debate, NaiveJudge, Propaganda
from oversight_arena.sim import BayesianBitJudge, BitAdvocate

OUT = Path("runs/verification_sweep")


def make(budget: int, trust: float) -> oa.Experiment:
    pol = EvidencePolicy(budget=budget)
    mechanisms = [NaiveJudge(), Propaganda(evidence=pol), Consultancy(rounds=2, evidence=pol), Debate(rounds=2, evidence=pol)]
    agents = {"kind:judge": BayesianBitJudge(trust=trust), "*": BitAdvocate(claims=3, lie_rate=0.5)}
    return oa.Experiment(HiddenBits(n_tasks=60), mechanisms, agents, oa.Stances(), progress=False)


if __name__ == "__main__":
    table = oa.sweep(make, {"budget": [0, 2, 4, 6, 8], "trust": [0.5, 0.8]})
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "table.csv", index=False)
    print(table[["trust", "budget", "mechanism", "asd", "lo", "hi", "accuracy"]].round(3).to_string(index=False))
    for trust, d in table.groupby("trust"):
        fig, _ = plots.line_compare(d, "budget", "asd", "mechanism", band=("lo", "hi"), xlabel="verification budget",
                                    ylabel="ASD (log score)", title=f"ASD vs verification budget (judge trust {trust})")
        print(plots.save(fig, OUT / f"asd_trust{trust}.png"))
