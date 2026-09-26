"""A mover privately records H/T only in the environment's HIDDEN state (as firm's refund books,
tampered_payments or sent-email records do). An observer with read access to the (identical) visible
files then guesses. No policy of the observer can beat 0.5, but its infoset key includes the snapshot
content id, which hashes ws.hidden too."""
import asyncio, warnings, tempfile
warnings.filterwarnings("ignore")
import so_arena as soa
from so_arena.core.mechanism import Mechanism, Outcome, RoleSpec
from so_arena.core.rewards import RewardRule
from so_arena.core.policy import FunctionPolicy
from so_arena.core.game import Player, RunContext
from so_arena.core.state import StateStore, FilesEnvironment, WorkspaceTool
from so_arena.core.tools import ToolResult
from so_arena.samplers.pools import expand_tree
from so_arena.analysis.optimization import evaluate_tree, BestOfN

class Book(WorkspaceTool):
    name = "book"; description = "record a booking"
    async def run(self, args, ws, item, game=None):
        pass
        return ToolResult(output="ok")

class Hidden(Mechanism):
    name = "hidden_state_coin"
    def roles(self):
        return {"mover": RoleSpec(name="mover", trainable=False, state_access="write"),
                "observer": RoleSpec(name="observer", state_access="read")}
    async def protocol(self, g):
        m = await g.act("mover", kind="text", phase="move", prompt="move", visible_to=["mover"])
        await g.act("observer", kind="text", phase="guess", prompt="guess")
        return Outcome(decision="x", metadata={"coin": m.text})
class R(RewardRule):
    def compute(self, ep):
        t = {t.role: t.text.strip() for t in ep.turns}
        return {"observer": float(t["mover"] == t["observer"])}
    def describe(self): return "match"

async def mover(req, c):
    x = "HT"[c.sample_index % 2]
    await c.call_tool("book", x)
    return x
observer = lambda req, c: "HT"[c.sample_index % 2]
item = soa.TaskItem(id="t", question="q", context={"workspace": {"files": {"a.txt": "same\n"}}})
env = FilesEnvironment(extra_tools={"book": Book()})
d = tempfile.mkdtemp(dir="/tmp/fixrev2")
ctx = RunContext(environment=env, states=StateStore(d + "/states"))
players = {"mover": Player(policy=FunctionPolicy(mover)), "observer": Player(policy=FunctionPolicy(observer))}
tree, eps = asyncio.run(expand_tree(Hidden(reward=R(), tools={"mover": ["book"]}), item, players,
                                    pool_sizes={"mover": 2, "observer": 2}, ctx=ctx))
print("errors:", [e.error for e in eps if e.error][:1])
print("leaves:", {k: v.rewards for k, v in tree.leaves.items()})
print("observer infoset keys:", len({n.key for n in tree.nodes.values() if n.role == "observer"}))
for n in (1, 2):
    print(f"Bo{n} observer success:", evaluate_tree(tree, {"observer": BestOfN(n)}).rewards["observer"])
