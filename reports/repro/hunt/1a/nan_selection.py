import asyncio, math
import numpy as np
from oversight_arena.core.strategy import Strategy
from oversight_arena.elicitation.optimize import Candidate, OptimizationTrace, ParamProposer, ProposalContext
from oversight_arena.elicitation.multiagent import pool_oracle
from oversight_arena.elicitation.evaluator import Evaluation

def cand(name, it, r):
    return Candidate(id=name, strategy=Strategy(name=name, params={"x": r}), iteration=it, parent=None, reward=r,
                     reward_se=0.0, gt={"correct": 0.0 if math.isnan(r) else r}, per_task={}, n=0)

# 1) trajectory(): a candidate whose episodes all errored (reward NaN) is 'best so far' forever
tr = OptimizationTrace([cand("broken", 0, float("nan")), cand("good", 1, 0.9), cand("better", 2, 0.95)])
print(tr.trajectory()[["iteration", "best_reward", "best"]].to_string(index=False))
print("best():", [c.id for c in tr.best(1)])

# 2) pool_oracle: max() with NaN first -> returns the broken strategy as the best response
class FakeEv:
    async def evaluate(self, s):
        r = {"broken": float("nan"), "weak": 0.1, "strong": 0.9}[s.name]
        return Evaluation(strategy=s, reward=r, reward_se=0.0, gt={}, per_task={})
pool = [Strategy(name=n) for n in ("broken", "weak", "strong")]
br = asyncio.run(pool_oracle(pool)(FakeEv(), []))
print("pool_oracle best response:", br.name)

# 3) ParamProposer parent ranking: sorted() with a NaN reward gives a wrong top-k
hist = [cand("c0", 0, 0.2), cand("nan", 0, float("nan")), cand("c1", 0, 0.9), cand("c2", 0, 0.5), cand("c3", 0, 0.1)]
top = sorted([c for c in hist if c.accepted], key=lambda c: c.reward, reverse=True)[:2]
print("ParamProposer top-2 parents (should be c1,c2):", [c.id for c in top])

pp = ParamProposer({"x": (0.0, 1.0)}, explore=0.0, top_k=2)
ctx = ProposalContext(role="a", brief="", domain="d", history=hist, iteration=1, base=Strategy(name="base"))
props = asyncio.run(pp.propose(ctx, 8))
print("ParamProposer.propose parents used:", sorted({p.origin["parent"] for p in props}))
