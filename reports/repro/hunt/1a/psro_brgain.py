"""PSRO br_gain is measured on the same tasks/seeds the oracle used to pick the best response."""
import asyncio
import numpy as np
import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Debate
from oversight_arena.sim import BayesianBitJudge, BitAdvocate
from oversight_arena.elicitation.multiagent import PSRO, pool_oracle
from oversight_arena.elicitation.evaluator import Evaluator
AG = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate(claims=3)}
dom = HiddenBits(n_tasks=60)
tasks = dom.tasks()
mech = Debate(rounds=1, evidence=EvidencePolicy(budget=3))
S = lambda n, **p: oa.Strategy(name=n, stance=oa.Stance.INCORRECT, params=p)
# a pool of near-identical best-response candidates: different lie rates, same claims
pool = [S(f"c{c}l{r:.1f}", claims=c, lie_rate=float(r)) for c in (2, 3, 4) for r in (0.2, 0.5, 0.8)]
opp = [oa.Strategy(name="honest_b", stance=oa.Stance.CORRECT, params={"claims": 3})]
ps = PSRO(dom, mech, AG, initial={"debater_a": [S("seed_a", claims=3, lie_rate=0.5)], "debater_b": opp},
          oracles={"debater_a": pool_oracle(pool), "debater_b": pool_oracle(opp)}, iterations=1, tasks=tasks[:4], seeds=1)
tr = asyncio.run(ps.run())
it0 = tr.iterations[0]
print("PSRO iter0 new BR:", it0.new, " reported br_gain:", {k: round(v, 3) for k, v in it0.br_gain.items()})
br = next(s for s in pool if s.name == it0.new["debater_a"])
seed = ps.pop["debater_a"][0]
fresh = Evaluator(dom, mech, AG, "debater_a", tasks=tasks[4:], others={"debater_b": opp[0]}, seeds=1)
a, b = asyncio.run(fresh.evaluate(br)), asyncio.run(fresh.evaluate(seed))
print(f"on 56 fresh tasks: BR reward {a.reward:.3f} vs incumbent {b.reward:.3f} -> gain {a.reward - b.reward:.3f}")
allr = [asyncio.run(fresh.evaluate(s)).reward for s in pool]
print("fresh-task rewards of whole pool:", np.round(allr, 3))
