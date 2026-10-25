from collections import Counter
from oversight_arena.core.strategy import Strategy
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.elicitation.evaluator import Evaluator
from oversight_arena.mechanisms import Debate
dom = HiddenBits(n_tasks=6)
A, B = Strategy(name="opp_A"), Strategy(name="opp_B")
ev = Evaluator(dom, Debate(), {}, "debater_a", others={"debater_b": [(A, 0.5), (B, 0.5)]}, seeds=1)
for cand in (Strategy(name="cand1"), Strategy(name="cand2", params={"claims": 5})):
    c = Counter(ev.profile(cand, t, 0).get("debater_b").strategy.name for t in ev.tasks)
    print(cand.name, "faces", dict(c), "over", len(ev.tasks), "episodes (meta-strategy is 50/50)")
