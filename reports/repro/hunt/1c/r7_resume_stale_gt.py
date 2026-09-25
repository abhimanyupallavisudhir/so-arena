"""Experiment resume only recomputes GT scorers whose *name* is absent from the stored record.
A scorer whose configuration changed (FunctionGT with a fixed fn, LLMGroundTruth with another
model/rubric, a scorer with a different threshold...) keeps the stale values silently."""
import tempfile
import oversight_arena as oa
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.ground_truth.common import FunctionGT
from oversight_arena.mechanisms import Propaganda

out = tempfile.mkdtemp()
dom = HiddenBits(n_tasks=4)
agent = oa.ScriptedAgent(lambda obs: {"probs": {o: 1 / len(obs.task.option_ids) for o in obs.task.option_ids}}
                         if obs.response.kind == "distribution" else "arg", id="s")
def run(gt_fn):
    return oa.Experiment(dom, Propaganda(), agent, oa.Stances(), gt=[FunctionGT(fn=gt_fn)], out=out, progress=False).run()
r1 = run(lambda task, rec: {"agent": 0.0})        # buggy GT
r2 = run(lambda task, rec: {"agent": 1.0})        # fixed GT, same output dir
print("GT after fix (should be 1.0):", sorted({r.gt["custom"]["agent"] for r in r2.records}))
r3 = oa.Experiment(dom, Propaganda(), agent, oa.Stances(), gt=[FunctionGT(fn=lambda t, r: {"agent": 1.0})], progress=False).run()
print("GT on a fresh run:", sorted({r.gt["custom"]["agent"] for r in r3.records}))
