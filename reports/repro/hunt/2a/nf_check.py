import numpy as np, itertools
from so_arena.games.normal_form import NormalFormGame, zero_sum_value
from so_arena.samplers.psro import solve_meta

def G(A,B): 
    m,k=A.shape
    return NormalFormGame(["r","c"],{"r":[f"r{i}" for i in range(m)],"c":[f"c{j}" for j in range(k)]},{"r":A,"c":B})

# degenerate game typical of empirical games with binary rewards (ties)
A=np.array([[1,1,0],[1,0,1]],float); B=np.array([[1,0,1],[0,1,1]],float)
g=G(A,B)
print("pure NE:", g.pure_nash())
eqs=g.support_enumeration(); print("support enum found", len(eqs), [[np.round(x,3) for x in e] for e in eqs])
# zero-sum LP check vs brute on random
rng=np.random.default_rng(1)
for _ in range(3):
    A=rng.normal(size=(3,4)); v,x,y=zero_sum_value(A)
    gg=G(A,-A); print("zs value",round(v,4),"nashconv",round(gg.nash_conv([x,y]),8))
# CE: chicken game: welfare max CE
A=np.array([[6,2],[7,0]],float);B=A.T
g=G(A,B); g.outcomes["w"]=A+B
mu,val=g.correlated_equilibrium("w"); print("chicken max-welfare CE", round(val,4),"(known 10.5)")
mu,val=g.correlated_equilibrium("w",coarse=True); print("chicken max-welfare CCE", round(val,4))
