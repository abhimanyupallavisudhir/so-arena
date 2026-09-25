# Load HumanEval+ pair tasks and count mutants carrying a textual tell.
import collections, re
from oversight_arena.domains.code import HiddenTestsCode
ts = HiddenTestsCode(dataset="humanevalplus").tasks()
c = collections.Counter(t.gt["mutation"] for t in ts)
tell = [t.id for t in ts if "(lambda _x: _x)" in "".join(t.resources["impls"].values())]
print("n tasks", len(ts), dict(c))
print("tasks whose wrong impl contains '(lambda _x: _x)':", len(tell), tell[:8])
