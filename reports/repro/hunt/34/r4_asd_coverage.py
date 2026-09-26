# Attempt 4: audited paired_asd lumps "label missing/error" together with "deception did not happen"
# in a single `excluded` count, so label coverage is not reported.
import sys
sys.path.insert(0, "/tmp/att4/src")
from oversight_arena.metrics import paired_asd
from oversight_arena.types import Evaluation, Measurement, Outcome, Reward, Role, Run, Task
def run(tid, cond, r):
    return Run(f"{tid}-{cond}", Task(tid, "p"), "m", (Role("w"),), 0, (), Outcome({"w": Reward(r)}),
               "complete", {}, {"condition": {"id": cond}})
runs, evs = [], []
for i in range(10):
    runs += [run(f"t{i}", "honest", 0.4), run(f"t{i}", "deceptive", 0.9)]
    ok = i < 2
    evs += [Evaluation(f"t{i}-honest", "sql", "1", {"w": {"quality": Measurement(1.0 if ok else None, "observed" if ok else "error", "s")}}),
            Evaluation(f"t{i}-deceptive", "sql", "1", {"w": {"quality": Measurement(0.0, "observed", "s")}})]
res = paired_asd(runs, "w", evaluations=evs)
print("pairs:", res.pairs, "excluded:", res.excluded, "value:", res.estimate.value, "fields:", list(res.__dataclass_fields__))
