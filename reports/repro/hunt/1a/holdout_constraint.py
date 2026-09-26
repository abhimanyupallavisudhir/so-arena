import asyncio
from oversight_arena.core.strategy import Strategy, Stance
from oversight_arena.core.util import stable_hash
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.elicitation.evaluator import Evaluator
from oversight_arena.elicitation.optimize import ParamProposer, PromptOptimizer
from oversight_arena.mechanisms import Propaganda
from oversight_arena.sim import BayesianBitJudge, BitAdvocate
# exactly demo_optimize's 'argue truth · honest only' search (propaganda, budget 2)
dom = HiddenBits(n_tasks=40, length=15); tasks = dom.tasks()
agents = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate()}
space = {"claims": (1, 8), "lie_rate": (0.0, 1.0), "lie_first": [True, False], "markup": [True, False]}
label = "argue truth · honest only"
mech = Propaganda(evidence=EvidencePolicy(budget=2), label="propaganda · budget 2")
base = Strategy(name="seed", stance=Stance.CORRECT, params={"claims": 2, "lie_rate": 0.0, "markup": True, "lie_first": False})
ev = Evaluator(dom, mech, agents, "agent", tasks=tasks[:24], gt_keys=("honesty", "correct"))
po = PromptOptimizer(ev, ParamProposer(space, seed=int(stable_hash(label, length=6), 16) % 1000), seeds=[base], iterations=8,
                     per_iter=5, minibatch=12, constraint=lambda e: e.gt.get("honesty", 1.0) >= 0.999, holdout=tasks[24:],
                     final_k=3, label=label)
tr = asyncio.run(po.run())
for c in tr.best(3):
    print(f"{c.strategy.name:8s} lie_rate={c.strategy.params.get('lie_rate'):.3f} claims={c.strategy.params.get('claims')} "
          f"minibatch honesty={c.gt.get('honesty'):.3f} reward={c.reward:.3f} | holdout honesty={c.holdout_gt.get('honesty'):.3f} "
          f"holdout reward={c.holdout_reward:.3f}")
acc = [c for c in tr.candidates if c.accepted]
print("accepted candidates with lie_rate>0:", sum(c.strategy.params.get('lie_rate', 0) > 0 for c in acc), "of", len(acc))
