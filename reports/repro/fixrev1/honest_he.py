# Honest canonical HumanEval+ solutions under the new isolated harness: pass rate and time.
import json, time, sys
from oversight_arena.domains.code import passes_hidden
rows = json.load(open("/tmp/hunt/1b/oa_data/hf_evalplus__humanevalplus_default_test_None.json"))
lo, hi = int(sys.argv[1]), int(sys.argv[2])
fails = []; t0 = time.time(); nospec_fail = []
for r in rows[lo:hi]:
    code = r["prompt"] + r["canonical_solution"]
    ok = passes_hidden(code, r["test"], r["entry_point"], spec=r["prompt"])
    if not ok: fails.append(r["task_id"])
print(f"rows {lo}-{hi}: honest fails {len(fails)} {fails}  time {time.time()-t0:.0f}s")
