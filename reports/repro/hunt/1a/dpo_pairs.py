import oversight_arena as oa
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Debate
from oversight_arena.sim import BayesianBitJudge, BitAdvocate
from oversight_arena.elicitation.rl import preference_pairs
AGENTS = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate(claims=3)}
base = oa.Strategy(name="base", params={"sample": {"claims": (1, 5), "lie_rate": (0.0, 1.0), "side_error": 0.3}})
dom = HiddenBits(n_tasks=3)
m1 = Debate(rounds=1, evidence=EvidencePolicy(budget=3), label="debate-verified")
m2 = Debate(rounds=1, label="debate-cheap-talk")
res = oa.Experiment(dom, [m1, m2], AGENTS, oa.Seeds(n=4, strategies={"debater_a": base}, roles=["debater_a"]), progress=False).run()
pp = preference_pairs(res, "debater_a")
print("episodes per (task, mechanism): 4 -> within-mechanism pairs per task at most 2*C(4,2)=12")
print("pairs per task:", pp.groupby("task").size().to_dict())
print("columns:", list(pp.columns))
# which mechanism produced each side? match by text+task
key = {}
for r in res.records:
    txt = "\n".join(e.content for e in r.transcript.entries if e.role == "debater_a")
    key.setdefault((r.task_id, txt), set()).add(r.mechanism)
cross = sum(key[(t, c)] != key[(t, j)] for t, c, j in zip(pp.task, pp.chosen, pp.rejected))
print("pairs whose chosen/rejected come from different mechanisms:", cross, "of", len(pp))
print("pairs where chosen text == rejected text:", int((pp.chosen == pp.rejected).sum()))
