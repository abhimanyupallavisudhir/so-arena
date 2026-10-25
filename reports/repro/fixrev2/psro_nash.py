import numpy as np, time, logging
logging.basicConfig(level=logging.WARNING)
from so_arena.games.normal_form import NormalFormGame
from so_arena.samplers.psro import solve_meta
import inspect; print(inspect.signature(solve_meta))
# 3-player matching pennies (Jordan): p1 wants to match p2, p2 match p3, p3 mismatch p1
P = ["a","b","c"]; S = {p: ["H","T"] for p in P}
pay = {p: np.zeros((2,2,2)) for p in P}
for i in range(2):
  for j in range(2):
    for k in range(2):
      pay["a"][i,j,k] = 1 if i==j else -1
      pay["b"][i,j,k] = 1 if j==k else -1
      pay["c"][i,j,k] = 1 if k!=i else -1
g = NormalFormGame(P, S, pay, name="pennies3")
t=time.time(); x = solve_meta(g, "nash"); print("pennies3", [np.round(v,3) for v in x], "NC", g.nash_conv(x), f"{time.time()-t:.1f}s")
rng = np.random.default_rng(0)
worst = 0
for trial in range(15):
    shape = tuple(rng.integers(2, 5, size=3))
    Pn = ["p0","p1","p2"]; Sn = {p: [str(i) for i in range(s)] for p, s in zip(Pn, shape)}
    payn = {p: rng.normal(size=shape) for p in Pn}
    gn = NormalFormGame(Pn, Sn, payn)
    x = solve_meta(gn, "nash")
    nc = gn.nash_conv(x) / gn.payoff_scale()
    worst = max(worst, nc)
print("random 3p games worst relative NashConv:", worst)
# 4 players
Pn = [f"p{i}" for i in range(4)]; shape=(3,3,3,3)
payn = {p: rng.normal(size=shape) for p in Pn}
gn = NormalFormGame(Pn, {p:[str(i) for i in range(3)] for p in Pn}, payn)
t=time.time(); x = solve_meta(gn, "nash"); print("4p NC rel", gn.nash_conv(x)/gn.payoff_scale(), f"{time.time()-t:.1f}s")
