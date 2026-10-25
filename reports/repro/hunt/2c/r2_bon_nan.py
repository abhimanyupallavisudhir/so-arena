"""BestOfNPolicy picks max((score, -i)); a NaN score on the first candidate wins against every later one."""
import math
from so_arena.core.actions import ActionRequest
from so_arena.core.policy import ActContext, BestOfNPolicy, FunctionPolicy
from so_arena.core.runner import run_sync

base = FunctionPolicy(lambda req, ctx: f"candidate {ctx.sample_index}")
scores = [math.nan, 0.9, 0.5, 0.99]  # e.g. a preview judge whose parse failed on candidate 0
def scorer(req, a, ctx):
    return scores[int(a.text.split()[-1])]
pol = BestOfNPolicy(base, 4, scorer)
a = run_sync(pol.act(ActionRequest(kind="text"), ActContext(role="agent")))
print("chosen:", a.text, "scores:", a.metadata["bon_scores"])
scores[:] = [0.2, math.nan, 0.1, 0.05]
a = run_sync(pol.act(ActionRequest(kind="text"), ActContext(role="agent")))
print("chosen:", a.text, "scores:", a.metadata["bon_scores"])
