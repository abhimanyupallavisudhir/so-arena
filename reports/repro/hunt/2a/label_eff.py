import numpy as np, pandas as pd
from so_arena.analysis.metrics import incentive_alignment
from so_arena.theory.audits import label_efficiency, optimal_lambda
rng=np.random.default_rng(0)
rows=[]
# natural pools: per item, a base rate of good behaviour that varies a lot across items;
# the reward only ranks behaviours *within* an item (plus an item offset the judge applies)
for i in range(400):
    base=rng.choice([0.05,0.95]); off=rng.normal(0,3)
    for j in range(10):
        v=1.0 if rng.random()<base else -1.0
        u=off+0.5*v+rng.normal(0,0.3)
        rows.append(dict(item_id=f"i{i}",role="agent",kind="agent",trainable=True,mechanism="m",reward=u,value=v))
df=pd.DataFrame(rows)
ia=incentive_alignment(df,n_boot=10)
print("incentive_alignment within_item_corr=%.3f label_efficiency=%.2f"%(ia.within_item_corr[0],ia.label_efficiency[0]))
print("theory.audits.label_efficiency (raw corr)=%.2f"%label_efficiency(df.reward,df.value))
# best achievable with item-demeaned control variate: 1/(1-corr(v, u-ubar_i)^2)
uw=df.reward-df.groupby("item_id").reward.transform("mean")
r=np.corrcoef(uw,df.value)[0,1]; print("achievable with item-demeaned u as control variate: %.2f"%(1/(1-r**2)))
