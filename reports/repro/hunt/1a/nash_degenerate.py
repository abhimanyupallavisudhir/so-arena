import itertools, numpy as np
from scipy.optimize import linprog
from oversight_arena.analysis.games import EmpiricalGame

def all_eq_payoffs(A, B):
    """Brute force: for every support pair (any sizes), LP-feasibility of an equilibrium; collect payoff pairs."""
    m, n = A.shape; out = []
    for ki in range(1, m+1):
        for I in itertools.combinations(range(m), ki):
            for kj in range(1, n+1):
                for J in itertools.combinations(range(n), kj):
                    # variables x(m), y(n), u, v ; x zero off I, y zero off J
                    # A y = u on I, A y <= u off I ; x^T B = v on J, <= v off J
                    nv = m+n+2
                    Aeq, beq, Aub, bub = [], [], [], []
                    for i in range(m):
                        row = np.zeros(nv); row[m:m+n] = A[i]; row[m+n] = -1
                        (Aeq if i in I else Aub).append(row); (beq if i in I else bub).append(0)
                    for j in range(n):
                        row = np.zeros(nv); row[:m] = B[:, j]; row[m+n+1] = -1
                        (Aeq if j in J else Aub).append(row); (beq if j in J else bub).append(0)
                    r = np.zeros(nv); r[:m] = 1; Aeq.append(r); beq.append(1)
                    r = np.zeros(nv); r[m:m+n] = 1; Aeq.append(r); beq.append(1)
                    bounds = [(1e-4, None) if i in I else (0, 0) for i in range(m)] + [(1e-4, None) if j in J else (0, 0) for j in range(n)] + [(None, None)]*2
                    res = linprog(np.zeros(nv), A_ub=np.array(Aub) if Aub else None, b_ub=bub or None, A_eq=np.array(Aeq), b_eq=beq, bounds=bounds)
                    if res.success:
                        out.append((round(res.x[m+n], 6), round(res.x[m+n+1], 6), I, J))
    return out

rng = np.random.default_rng(1)
found_missing = 0
for trial in range(400):
    A = rng.integers(0, 2, (3, 3)).astype(float); B = rng.integers(0, 2, (3, 3)).astype(float)
    g = EmpiricalGame.from_matrices(A, B)
    eqs = g.nash()
    got = {(round(e.payoffs["row"], 6), round(e.payoffs["col"], 6)) for e in eqs}
    truth = {(u, v) for u, v, _, _ in all_eq_payoffs(A, B)}
    if not eqs or (truth - got):
        found_missing += 1
        if found_missing <= 2:
            print("A=\n", A, "\nB=\n", B, "\n nash() payoff outcomes:", got, "\n all equilibrium payoff outcomes:", truth)
print("games where nash() misses an equilibrium payoff outcome (or finds none):", found_missing, "/ 400")
