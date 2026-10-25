import oversight_arena as oa
from collections import Counter
from oversight_arena.analysis.bon import tree_value, trees_from_results
from oversight_arena.experiment.profiles import GameTree
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import ProposerCritic
from oversight_arena.sim import BayesianBitJudge, BitAdvocate
AGENTS = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate(claims=3)}
base = oa.Strategy(name="base", params={"sample": {"claims": (1, 5), "lie_rate": (0.0, 1.0), "side_error": 0.4}})
src = GameTree(levels=[("proposer", "proposal", 3), ("critic", "critique", 2), ("proposer", "rebuttal", 2)],
               strategies={"proposer": base, "critic": base})
dom = HiddenBits(n_tasks=2)
lv = [("proposer", "proposal"), ("critic", "critique"), ("proposer", "rebuttal")]
def leaves(n):
    return 1 if not n.children else sum(leaves(c) for c in n.children)
# (a) repeats=2: two independent samples per profile
res = oa.Experiment(dom, ProposerCritic(critiques=1, rebuttal=True, evidence=EvidencePolicy(budget=3)), AGENTS, src,
                    repeats=2, progress=False).run()
trees = trees_from_results(res, lv)
print("(a) repeats=2: episodes", len(res), "tasks", len(trees), "leaves", sum(leaves(t) for t in trees))
print("    distinct episode outcomes per path differ?",
      len({(r.task_id, r.seed, r.outcome['accept_prob']) for r in res.records}))
# (b) two mechanisms in one Results
m1 = ProposerCritic(critiques=1, rebuttal=True, evidence=EvidencePolicy(budget=3), label="pc-verified")
m2 = ProposerCritic(critiques=1, rebuttal=True, label="pc-cheap-talk")
res2 = oa.Experiment(dom, [m1, m2], AGENTS, src, progress=False).run()
trees2 = trees_from_results(res2, lv)
print("(b) two mechanisms: episodes", len(res2), "trees", len(trees2), "leaves", sum(leaves(t) for t in trees2),
      Counter(r.mechanism for r in res2.records))
