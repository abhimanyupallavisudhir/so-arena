from oversight_arena.elicitation.optimize import OptimizationTrace, Candidate
from oversight_arena.core.strategy import Strategy
cs=[Candidate(id=str(i),strategy=Strategy(name=f"s{i}"),iteration=0,parent=None,reward=r,reward_se=0,gt={},per_task={},n=1) for i,r in enumerate([-1.0,0.0])]
t=OptimizationTrace(cs)
print("best():", [(c.id,c.reward) for c in t.best(2)])
