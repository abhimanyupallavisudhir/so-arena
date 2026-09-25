"""A ground-truth scorer that fails transiently (sandbox/docker/network error) leaves
ground_truth={'error_<scorer>': ...} with gt_status='known'. Resuming reuses the episode (only
'pending' GT is re-scored), so the missing value is baked in for good and ASD silently drops the item."""
import tempfile
import so_arena as soa
from so_arena.core.ground_truth import GroundTruthScorer, StanceValue
from so_arena.core.items import binary_item
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.core.store import RunStore
from so_arena.analysis.frames import role_frame
from so_arena.analysis.metrics import asd
from so_arena.mechanisms.qa import Propaganda

state = {"broken": True}
class FlakyStance(GroundTruthScorer):
    name = "stance_value"
    async def score(self, ep, item, ctx=None):
        if state["broken"] and item.id == "q0":
            raise RuntimeError("sandbox unavailable")
        return await StanceValue().score(ep, item, ctx)

items = [binary_item(f"q{i}", f"Q{i}", "right", "wrong", shuffle_seed=i) for i in range(4)]
judge = soa.ScriptedPolicy('{"A": 0.7, "B": 0.3}')
profs = [Profile(name=s, players={"agent": PlayerSpec(policy=soa.ScriptedPolicy("argument"), stance=s), "judge": judge})
         for s in ("true", "false")]
store = RunStore(tempfile.mkdtemp())
for attempt in ("first run (scorer down for q0)", "resume (scorer fixed)"):
    eps = run_sync(run_episodes(Propaganda(), items, profs, store=store, ground_truth=[FlakyStance()]))
    q0 = [e for e in eps if e.item_id == "q0"][0]
    print(f"{attempt}: q0 gt={q0.ground_truth} status={q0.gt_status}; ASD n_items =",
          asd(role_frame(eps), n_boot=20)["n_items"].tolist())
    state["broken"] = False
