import numpy as np, pandas as pd
from oversight_arena.analysis.diagnostics import outcome_metrics
from oversight_arena.analysis.stats import bootstrap_ci
# 4 tasks, 50 episodes each (seeds/profiles); judge correctness is almost entirely task-determined
rng = np.random.default_rng(0)
rows = []
for t, p in enumerate([0.95, 0.9, 0.1, 0.05]):
    for s in range(50):
        rows.append({"mechanism": "debate", "task": f"t{t}", "gt_decision_correct[_outcome]": float(rng.random() < p)})
ep = pd.DataFrame(rows)
om = outcome_metrics(ep)
print("outcome_metrics accuracy CI (episode bootstrap):", om[["accuracy", "accuracy_lo", "accuracy_hi"]].round(3).to_dict("records"))
per_task = ep.groupby("task")["gt_decision_correct[_outcome]"].mean()
print("task-clustered bootstrap CI:", tuple(round(x, 3) for x in bootstrap_ci(per_task.to_numpy())))
