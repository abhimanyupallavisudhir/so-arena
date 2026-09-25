import itertools, numpy as np, math
from so_arena.analysis.optimization import BestOfN, Tilted, pool_curve
import pandas as pd

def brute(w, n, replace):
    k=len(w); p=np.zeros(k)
    draws = itertools.product(range(k), repeat=n) if replace else itertools.combinations(range(k), n)
    draws=list(draws)
    for d in draws:
        m=max(w[i] for i in d)
        winners=[i for i in d if w[i]==m]
        # ties split uniformly among tied *candidates* (distinct)
        uniq=sorted(set(winners))
        for i in uniq: p[i]+=1/len(uniq)/len(draws)
    return p

rng=np.random.default_rng(0)
worst=0
for trial in range(300):
    k=rng.integers(2,7); n=rng.integers(1,k+1)
    w=rng.integers(0,3,size=k).astype(float)
    for mode,rep in (("unbiased",False),("plugin",True)):
        got=BestOfN(n,mode).probs(w); exp=brute(list(w),n,rep)
        worst=max(worst,np.abs(got-exp).max())
print("max abs error vs brute force (ties incl.):", worst)
# n > K fallback
w=np.array([0.,0.,1.])
print("K=3, BoN(5,'unbiased') probs:", BestOfN(5,'unbiased').probs(w), "repr:", repr(BestOfN(5,'unbiased')))
print("K=3, BoN(3,'unbiased') probs:", BestOfN(3,'unbiased').probs(w))
# pool_curve mixing pool sizes: item A K=3, item B K=10
pool=pd.DataFrame({"item_id":["A"]*3+["B"]*10,
                   "reward":[0,0,1]+[0]*9+[1],"value":[0,0,1]+[0]*9+[1]})
print(pool_curve(pool, selections=[BestOfN(4),BestOfN(5)]))
