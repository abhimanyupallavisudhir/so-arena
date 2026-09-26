import json, sys
from oversight_arena.domains.code import _HIDDEN_HARNESS
from oversight_arena.domains._exec import run_isolated, run_python
rows = json.load(open("/tmp/hunt/1b/oa_data/hf_evalplus__humanevalplus_default_test_None.json"))
for i in (32, 83, 139):
    r = rows[i]; code = r["prompt"] + r["canonical_solution"]
    h = _HIDDEN_HARNESS + f"\n_OA_TEST = {r['test']!r}\n_OA_ENTRY = {r['entry_point']!r}\n_OA_SPEC = {r['prompt']!r}\n"
    res = run_isolated(h, code=code, timeout=20)
    # old-style harness, unsandboxed, for comparison
    old = run_python(f"{code}\n\n{r['test']}\n\ncheck({r['entry_point']})\nprint('PASSED')\n", timeout=20, sandbox=False)
    print(i, r["entry_point"], "new:", res.ok, res.value, res.error[:300], "| old-style:", old.ok and "PASSED" in old.stdout, old.stderr[-200:])
