import numpy as np, itertools
from nash_degenerate import all_eq_payoffs
from oversight_arena.analysis.games import EmpiricalGame
for shape in [(2, 2), (2, 3)]:
    for vals in itertools.product([0., 1.], repeat=2 * shape[0] * shape[1]):
        A = np.array(vals[:shape[0]*shape[1]]).reshape(shape); B = np.array(vals[shape[0]*shape[1]:]).reshape(shape)
        g = EmpiricalGame.from_matrices(A, B, row="agent", col="overseer", row_strats=["honest", "lie"][:shape[0]],
                                        col_strats=["audit", "trust", "skim"][:shape[1]])
        eqs = g.nash()
        used = {i for e in eqs for i in np.flatnonzero(e.mix["agent"] > 1e-9)}
        truth = all_eq_payoffs(A, B)
        if 1 in {i for *_, I, J in truth for i in I} and 1 not in used:
            print(shape, "A (agent)=", A.tolist(), " B (overseer)=", B.tolist())
            for e in eqs: print("   nash() returns", {k: v.round(3).tolist() for k, v in e.mix.items()}, e.payoffs)
            for u, v, I, J in truth:
                if 1 in I: print("   missed: agent support", I, "overseer support", J, "payoffs", u, v)
            # verify one missed equilibrium explicitly
            raise SystemExit
print("none")
