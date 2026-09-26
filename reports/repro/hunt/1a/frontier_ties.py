import pandas as pd
from oversight_arena.analysis.ic import frontier
# accept/reject style reward: both strategies always get reward 1.0; one is honest (gt 1), one lies (gt 0)
for hon, lie in [("honest", "lie"), ("zz_honest", "lie")]:
    rows = []
    for t in range(5):
        rows += [dict(mechanism="m", task=f"t{t}", role="a", reward=1.0, gt_correct=1.0, strategy_name=hon),
                 dict(mechanism="m", task=f"t{t}", role="a", reward=1.0, gt_correct=0.0, strategy_name=lie)]
    f = frontier(pd.DataFrame(rows))
    print(f"names ({hon!r}, {lie!r}):", f[["frontier_gap", "argmax_strategy", "argmax_gt", "gt_regret"]].to_dict("records"))
