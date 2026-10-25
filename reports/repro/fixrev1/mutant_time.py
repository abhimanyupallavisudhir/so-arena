import time, collections
t0 = time.time()
from oversight_arena.domains.code import HiddenTestsCode
ts = HiddenTestsCode(dataset="humanevalplus").tasks()
c = collections.Counter(t.gt["mutation"] for t in ts)
print("n tasks", len(ts), dict(c), "cold-cache build time %.0fs" % (time.time() - t0))
print("lambda tell:", sum("(lambda _x: _x)" in "".join(t.resources["impls"].values()) for t in ts))
