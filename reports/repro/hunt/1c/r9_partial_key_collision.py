"""ScriptedAgent.describe() -> code_hash(fn); for functools.partial (no __code__, args not in
__dict__) the hash ignores the bound arguments, so agents built as partial(policy, p=...) share
episode keys: a resumed Experiment (same out dir) silently returns the other agent's episodes."""
import tempfile
from functools import partial
import oversight_arena as oa
from oversight_arena.core.util import code_hash
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Propaganda

def judge_policy(obs, p_first):
    o = obs.task.option_ids
    return {"probs": {o[0]: p_first, o[1]: 1 - p_first}} if obs.response.kind == "distribution" else "argument"

a3, a9 = oa.ScriptedAgent(partial(judge_policy, p_first=0.3)), oa.ScriptedAgent(partial(judge_policy, p_first=0.9))
print("code_hash equal:", code_hash(a3.fn) == code_hash(a9.fn), "| describe equal:", a3.describe() == a9.describe())
out = tempfile.mkdtemp()
dom = HiddenBits(n_tasks=3)
for a in (a3, a9):
    res = oa.Experiment(dom, Propaganda(), {"*": a}, oa.Stances(), out=out, progress=False).run()
    print("judge P(first option) recorded:", sorted({round(r.outcome["probs"][dom.tasks()[0].options[0].id] if False else list(r.outcome["probs"].values())[0], 2) for r in res.records}))

# same for a closure whose captured cell is itself a closure (only its __qualname__ is hashed)
def make(p):
    def inner(obs):
        return p
    return lambda obs: inner(obs)
print("nested-closure code_hash equal for p=0.3 vs 0.9:", code_hash(make(0.3)) == code_hash(make(0.9)))
