"""Repro: RepoAudit grades solution.py in isolation, so correct work split across repository modules scores -1."""
import tempfile, pathlib
import so_arena as soa
from so_arena.core.game import Player
from so_arena.core.runner import run_sync
from so_arena.domains.repo import MODULE, RepoDomain, RepoAudit
from so_arena.mechanisms import ReviewedWork

dom = RepoDomain()
item = dom.load(limit=1)[0]
ctx = dom.context(states=pathlib.Path(tempfile.mkdtemp()) / "states")
entry = item.context["entry_point"]

def worker(req, c):
    if not req.phase.startswith("work"):
        return "ok"
    c.workspace.write_text("impl/__init__.py", "")
    c.workspace.write_text("impl/core.py", req.view.item.private["solutions"]["reference"])
    c.workspace.write_text(MODULE, f"from impl.core import {entry}\n")
    run = c.workspace.run("python run_tests.py")
    print("worker sees run_tests:", run.stdout.strip().splitlines(), "exit", run.returncode)
    return "Implemented in impl/core.py and re-exported from solution.py; 1 passed, 0 failed. The task is complete."

mech = ReviewedWork(dossier="diff", affordances={"worker": ["solutions"]})
players = {"worker": Player(policy=soa.FunctionPolicy(worker)),
           "reviewer": Player(policy=soa.FunctionPolicy(lambda r, c: {"accept": 0.5, "reject": 0.5}))}
ep = run_sync(mech.run(item, players, ctx))
assert ep.error is None, ep.error
gt = run_sync(RepoAudit().score(ep, item, ctx))
print("GT:", {k: gt[k] for k in ("hidden_pass_rate", "visible_pass", "tampered", "misreport", "work_ok", "role_values")})
