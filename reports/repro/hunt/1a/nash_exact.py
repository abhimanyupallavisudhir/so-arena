"""Exact check: for each support pair (I,J), LP-maximise x_k (k in I) over Nash equilibria whose supports lie in I x J."""
import itertools, numpy as np
from scipy.optimize import linprog
from oversight_arena.analysis.games import EmpiricalGame

def max_weight_in_NE(A, B, k):
    m, n = A.shape; best, arg = 0.0, None
    for ki in range(1, m + 1):
        for I in itertools.combinations(range(m), ki):
            if k not in I: continue
            for kj in range(1, n + 1):
                for J in itertools.combinations(range(n), kj):
                    nv = m + n + 2; Aeq, beq, Aub, bub = [], [], [], []
                    for i in range(m):
                        row = np.zeros(nv); row[m:m+n] = A[i]; row[m+n] = -1
                        (Aeq if i in I else Aub).append(row); (beq if i in I else bub).append(0)
                    for j in range(n):
                        row = np.zeros(nv); row[:m] = B[:, j]; row[m+n+1] = -1
                        (Aeq if j in J else Aub).append(row); (beq if j in J else bub).append(0)
                    r = np.zeros(nv); r[:m] = 1; Aeq.append(r); beq.append(1)
                    r = np.zeros(nv); r[m:m+n] = 1; Aeq.append(r); beq.append(1)
                    bounds = [(0, None) if i in I else (0, 0) for i in range(m)] + [(0, None) if j in J else (0, 0) for j in range(n)] + [(None, None)] * 2
                    c = np.zeros(nv); c[k] = -1
                    res = linprog(c, A_ub=np.array(Aub) if Aub else None, b_ub=bub or None, A_eq=np.array(Aeq), b_eq=beq, bounds=bounds)
                    if res.success and -res.fun > best + 1e-9:
                        best, arg = -res.fun, (res.x[:m].round(4), res.x[m:m+n].round(4))
    return best, arg

rng = np.random.default_rng(1); bad = 0; shown = False
for trial in range(300):
    A = rng.integers(0, 2, (3, 3)).astype(float); B = rng.integers(0, 2, (3, 3)).astype(float)
    g = EmpiricalGame.from_matrices(A, B)
    eqs = g.nash()
    for k in range(3):
        found = max((e.mix["row"][k] for e in eqs), default=0.0)
        true, arg = max_weight_in_NE(A, B, k)
        if true > 1e-6 and found < 1e-9:
            bad += 1
            if not shown:
                shown = True
                x, y = arg
                print("A =", A.tolist(), "B =", B.tolist())
                print("nash() returns:", [{r: v.round(3).tolist() for r, v in e.mix.items()} for e in eqs])
                print(f"missed exact NE with row strategy {k} played: x={x.tolist()} y={y.tolist()} regret={g.regret({'row': x, 'col': y})}")
            break
print("games with a row strategy that is used in some exact NE but in none returned by nash():", bad, "/ 300")
