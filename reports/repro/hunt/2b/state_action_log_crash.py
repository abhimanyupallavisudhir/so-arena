"""Repro: with an action log configured, an agent that makes its workspace read-only (chmod -R a-w .) makes
Game._produce raise after its turn (the log append is outside the tools' error handling): the episode errors and
role_frame (used by ASD summaries) silently drops it."""
import tempfile, pathlib
import so_arena as soa
from so_arena.core.game import Player
from so_arena.core.runner import run_sync
from so_arena.domains.repo import RepoDomain
from so_arena.mechanisms import ReviewedWork
from so_arena.analysis.frames import role_frame

dom = RepoDomain(action_log="logs/actions.jsonl")
item = dom.load(limit=1)[0]
ctx = dom.context(states=pathlib.Path(tempfile.mkdtemp()) / "states")

def make(lock):
    async def worker(req, c):
        if not req.phase.startswith("work"):
            return "ok"
        await c.call_tool("shell", "echo hi > notes.txt" + (" && chmod -R a-w ." if lock else ""))
        return "The task is complete."
    return worker

eps = []
for lock in (False, True):
    mech = ReviewedWork(dossier="diff", tools={"worker": ["shell"]})
    players = {"worker": Player(policy=soa.FunctionPolicy(make(lock), label=f"lock={lock}")),
               "reviewer": Player(policy=soa.FunctionPolicy(lambda r, c: {"accept": 0.5, "reject": 0.5}))}
    ep = run_sync(mech.run(item, players, ctx))
    print(f"lock={lock}: error =", (ep.error or "None").strip().splitlines()[-1])
    eps.append(ep)
df = role_frame(eps)
print("role_frame rows per worker label:", df[df.role == "worker"]["label"].tolist())
