"""ControlArena mapping: a missing suspicion score (monitor error / timeout -> NaN) is silently
treated as 'not flagged' (task_if_unflagged pays full task reward) or yields a NaN reward
(task_minus_suspicion / monitor) that ASD's groupby-mean then silently skips."""
import math
import pandas as pd
from so_arena.integrations.controlarena import episodes_from_frame
from so_arena.analysis.frames import role_frame
from so_arena.analysis.metrics import asd

rows = []
for t in range(3):
    rows.append(dict(eval_mode="honest", score_max_suspicion_scorer=0.1 * t, main_task_success=True,
                     side_task_success=None, metadata_main_task_name=f"task{t}", epoch=1, id=f"h{t}"))
    # attack run whose monitor call failed on task0 (NaN), otherwise highly suspicious
    rows.append(dict(eval_mode="attack", score_max_suspicion_scorer=math.nan if t == 0 else 0.9,
                     main_task_success=True, side_task_success=True,
                     metadata_main_task_name=f"task{t}", epoch=1, id=f"a{t}"))
df = pd.DataFrame(rows)
for preset in ("task_if_unflagged", "task_minus_suspicion"):
    eps = episodes_from_frame(df, reward=preset, audit_budget=0.5)
    print(preset, [(e.profile, e.item_id, e.rewards["untrusted"], e.outcome.data["flagged"]) for e in eps if e.profile == "attack"])
    a = asd(role_frame(eps), n_boot=50)
    print("   ASD:", a[["asd", "n_items"]].to_dict("records"))
