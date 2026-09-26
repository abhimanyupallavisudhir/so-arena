import asyncio, warnings
warnings.filterwarnings("ignore")
from so_arena.core.mechanism import Mechanism, Outcome, RoleSpec
from so_arena.core.rewards import RewardRule
from so_arena.core.items import TaskItem, AnswerOption, GroundTruth
from so_arena.core.policy import FunctionPolicy
from so_arena.core.game import Player, RunContext
from so_arena.samplers.pools import expand_tree
from so_arena.analysis.optimization import evaluate_tree, BestOfN
item = TaskItem(id="c0", domain="coin", question="guess", answers=[AnswerOption(label="A", text="H"), AnswerOption(label="B", text="T")], ground_truth=GroundTruth(correct="A"))

# Variant 1: the hidden move is a TRAINABLE opponent's private turn (sequential matching pennies)
class Pennies(Mechanism):
    name = "pennies"
    def roles(self):
        return {"hider": RoleSpec(name="hider"), "guesser": RoleSpec(name="guesser")}
    async def protocol(self, g):
        await g.act("hider", kind="text", phase="hide", prompt="hide", visible_to=["hider"])
        await g.act("guesser", kind="text", phase="guess", prompt="guess")
        return Outcome(decision="x")
class PR(RewardRule):
    def compute(self, ep):
        t = {t.role: t.text.strip() for t in ep.turns}
        m = float(t["hider"] == t["guesser"]); return {"guesser": m, "hider": 1 - m}
    def describe(self): return "pennies"
ht = lambda req, ctx: "HT"[ctx.sample_index % 2]
players = {"hider": Player(policy=FunctionPolicy(ht)), "guesser": Player(policy=FunctionPolicy(ht))}
tree, _ = asyncio.run(expand_tree(Pennies(reward=PR()), item, players, pool_sizes={"hider": 2, "guesser": 2}, ctx=RunContext()))
print("V1 guesser keys:", len({n.key for n in tree.nodes.values() if n.role == "guesser"}))
for pol in ({"guesser": BestOfN(4)}, {"hider": BestOfN(4)}, {"hider": BestOfN(4), "guesser": BestOfN(4)}):
    tv = evaluate_tree(tree, pol)
    print("V1", pol, {k: round(v, 3) for k, v in tv.rewards.items()}, "gap", tv.gap)

# Variant 2: noisy signal. Dealer flips privately; a noise role privately says same/flip (3:1);
# the guesser is shown coin XOR flip in its PROMPT. Optimal guesser: 0.75. Full-history keys: 1.0.
class Noisy(Mechanism):
    name = "noisy"
    def roles(self):
        return {"dealer": RoleSpec(name="dealer", trainable=False), "noise": RoleSpec(name="noise", trainable=False),
                "guesser": RoleSpec(name="guesser")}
    async def protocol(self, g):
        d = await g.act("dealer", kind="text", phase="flip", prompt="flip", visible_to=["dealer"])
        n = await g.act("noise", kind="text", phase="noise", prompt="noise", visible_to=["noise"])
        sig = d.text if n.text == "same" else {"H": "T", "T": "H"}[d.text]
        await g.act("guesser", kind="text", phase="guess", prompt=f"signal: {sig}")
        return Outcome(decision="x")
class M(RewardRule):
    def compute(self, ep):
        t = {t.role: t.text.strip() for t in ep.turns}
        return {"guesser": float(t["dealer"] == t["guesser"])}
    def describe(self): return "match"
players = {"dealer": Player(policy=FunctionPolicy(ht)),
           "noise": Player(policy=FunctionPolicy(lambda r, c: "flip" if c.sample_index % 4 == 3 else "same")),
           "guesser": Player(policy=FunctionPolicy(ht))}
tree, _ = asyncio.run(expand_tree(Noisy(reward=M()), item, players, pool_sizes={"dealer": 2, "noise": 4, "guesser": 2}, ctx=RunContext()))
print("V2 guesser keys:", len({n.key for n in tree.nodes.values() if n.role == "guesser"}), "leaves", len(tree.leaves))
for n in (1, 2, 8):
    tv = evaluate_tree(tree, {"guesser": BestOfN(n)})
    print(f"V2 Bo{n}", round(tv.rewards["guesser"], 4), "gap", tv.gap)
