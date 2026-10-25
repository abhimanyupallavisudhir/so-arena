"""Dealer privately flips a coin (visible only to itself); a blind guesser then guesses.
Guesser reward 1 if match. Best achievable by any policy that can't see the coin: 0.5."""
import asyncio
from so_arena.core.mechanism import Mechanism, Outcome, RoleSpec
from so_arena.core.rewards import RewardRule
from so_arena.core.items import TaskItem, AnswerOption, GroundTruth
from so_arena.core.policy import FunctionPolicy
from so_arena.core.game import Player
from so_arena.core.game import RunContext
from so_arena.samplers.pools import expand_tree
from so_arena.analysis.optimization import evaluate_tree, BestOfN

class Coin(Mechanism):
    name = "coin"
    def roles(self):
        return {"dealer": RoleSpec(name="dealer", trainable=False),
                "guesser": RoleSpec(name="guesser")}
    async def protocol(self, g):
        d = await g.act("dealer", kind="text", phase="flip", prompt="flip", visible_to=["dealer"])
        seen = [t.text for t in g.visible_turns("guesser")]
        assert not any("H" == s or "T" == s for s in seen), seen   # guesser really is blind
        x = await g.act("guesser", kind="text", phase="guess", prompt="guess")
        return Outcome(decision=x.text, metadata={"coin": d.text})

class Match(RewardRule):
    def compute(self, ep):
        t = {t.role: t.text for t in ep.turns}
        return {"guesser": float(t["dealer"].strip() == t["guesser"].strip())}
    def describe(self): return "1 if guess matches coin"

def dealer(req, ctx):  return "HHT"[ctx.sample_index % 3]
def guesser(req, ctx): return "HT"[ctx.sample_index % 2]

item = TaskItem(id="c0", domain="coin", question="guess", answers=[AnswerOption(label="A", text="H"), AnswerOption(label="B", text="T")], ground_truth=GroundTruth(correct="A"))
players = {"dealer": Player(policy=FunctionPolicy(dealer)), "guesser": Player(policy=FunctionPolicy(guesser))}
tree, _ = asyncio.run(expand_tree(Coin(reward=Match()), item, players, pool_sizes={"dealer": 3, "guesser": 2}, ctx=RunContext()))
print("leaves:", {k: v.rewards for k, v in tree.leaves.items()})
print("guesser keys distinct per coin branch:", len({n.key for n in tree.nodes.values() if n.role == "guesser"}))
for n in (1, 2, 8):
    print(f"Bo{n} guesser success:", evaluate_tree(tree, {"guesser": BestOfN(n)}).rewards["guesser"])
