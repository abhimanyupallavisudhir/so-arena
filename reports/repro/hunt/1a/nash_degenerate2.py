import numpy as np, itertools
from nash_degenerate import all_eq_payoffs
from oversight_arena.analysis.games import EmpiricalGame
rng = np.random.default_rng(1)
miss = 0; example = None
for trial in range(400):
    A = rng.integers(0, 2, (3, 3)).astype(float); B = rng.integers(0, 2, (3, 3)).astype(float)
    g = EmpiricalGame.from_matrices(A, B)
    eqs = g.nash()
    used_found = {i for e in eqs for i in np.flatnonzero(e.mix["row"] > 1e-9)}
    used_true = {i for *_, I, J in all_eq_payoffs(A, B) for i in I}
    if used_true - used_found:
        miss += 1
        if example is None: example = (A, B, used_true - used_found, eqs)
print("games where some row strategy is played in an equilibrium but in none returned by nash():", miss, "/ 400")
A, B, m, eqs = example
print("A=", A.tolist(), "B=", B.tolist(), "row strategies missed:", m)
for e in eqs: print("  found", {k: v.round(3).tolist() for k, v in e.mix.items()})
for u, v, I, J in all_eq_payoffs(A, B):
    if set(I) & m: print("  missed support I=", I, "J=", J, "payoffs", u, v)
