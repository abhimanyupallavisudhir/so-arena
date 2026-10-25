"""The hider's private choice H triggers one extra PRIVATE turn (e.g. a private follow-up / tool round); T does not.
Then a public 'announce' turn by the hider (same text in both branches), then the guesser guesses."""
import asyncio, warnings; warnings.filterwarnings("ignore")
from so_arena.core.mechanism import Mechanism, Outcome, RoleSpec
from so_arena.core.rewards import RewardRule
from so_arena.core.items import TaskItem, AnswerOption, GroundTruth
from so_arena.core.policy import FunctionPolicy
from so_arena.core.game import Player, RunContext
from so_arena.samplers.pools import expand_tree
from so_arena.analysis.optimization import evaluate_tree, BestOfN
item = TaskItem(id="c0", domain="coin", question="guess", answers=[AnswerOption(label="A", text="H"), AnswerOption(label="B", text="T")], ground_truth=GroundTruth(correct="A"))
seen = {}
class P(Mechanism):
    name = "count"
    def roles(self): return {"hider": RoleSpec(name="hider", trainable=False), "guesser": RoleSpec(name="guesser")}
    async def protocol(self, g):
        h = await g.act("hider", kind="text", phase="hide", prompt="hide", visible_to=["hider"])
        if h.text == "H":
            await g.act("hider", kind="text", phase="extra", prompt="extra", visible_to=["hider"])
        await g.act("hider", kind="text", phase="announce", prompt="announce")
        return Outcome(decision="x")
class R(RewardRule):
    def compute(self, ep):
        h = [t.text for t in ep.turns if t.phase == "hide"][0]; x = [t.text for t in ep.turns if t.role == "guesser"]
        return {"guesser": float(bool(x) and x[0] == h)}
    def describe(self): return "m"
class P2(P):
    async def protocol(self, g):
        await P.protocol(self, g)
        return Outcome(decision="x")
async def guess_proto(self, g):
    h = await g.act("hider", kind="text", phase="hide", prompt="hide", visible_to=["hider"])
    if h.text == "H":
        await g.act("hider", kind="text", phase="extra", prompt="extra", visible_to=["hider"])
    await g.act("hider", kind="text", phase="announce", prompt="announce")
    await g.act("guesser", kind="text", phase="guess", prompt="guess")
    return Outcome(decision="x")
P.protocol = guess_proto
def hider(req, c):
    return {"hide": "HT"[c.sample_index % 2], "extra": "ok", "announce": "done"}[req.phase]
def guesser(req, c):
    seen.setdefault("prompts", set()).add(str([m.content for m in req.prompt]))
    return "HT"[c.sample_index % 2]
players = {"hider": Player(policy=FunctionPolicy(hider)), "guesser": Player(policy=FunctionPolicy(guesser))}
tree, _ = asyncio.run(expand_tree(P(reward=R()), item, players, pool_sizes={"hider": 2, "guesser": 2}, ctx=RunContext()))
print("guesser keys:", len({n.key for n in tree.nodes.values() if n.role == "guesser"}), "| distinct guesser prompts:", len(seen["prompts"]))
for n in (1, 2):
    print(f"Bo{n}", evaluate_tree(tree, {"guesser": BestOfN(n)}).rewards["guesser"])
dumps = []
def guesser2(req, c):
    import json; dumps.append(json.dumps(req.model_dump(mode="json"), sort_keys=True, default=str)); return "H"
players = {"hider": Player(policy=FunctionPolicy(hider)), "guesser": Player(policy=FunctionPolicy(guesser2))}
asyncio.run(expand_tree(P(reward=R()), item, players, pool_sizes={"hider": 2, "guesser": 1}, ctx=RunContext()))
a, b = sorted(set(dumps))[:2]
import difflib
for i, (x, y) in enumerate(zip(a, b)):
    if x != y: print("first diff at", i, "|", a[max(0,i-120):i+40], "\n vs\n", b[max(0,i-120):i+40]); break
