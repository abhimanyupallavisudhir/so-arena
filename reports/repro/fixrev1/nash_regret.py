# Every equilibrium nash() returns should have ~0 regret (degenerate binary games, and 4x4 ints).
import numpy as np
from oversight_arena.analysis.games import EmpiricalGame
rng = np.random.default_rng(7); worst = 0; empty = 0; n = 0
for shape, hi in [((3, 3), 2), ((4, 4), 3), ((2, 5), 2)]:
    for _ in range(150):
        A = rng.integers(0, hi, shape).astype(float); B = rng.integers(0, hi, shape).astype(float)
        g = EmpiricalGame.from_matrices(A, B); eqs = g.nash(); n += 1
        if not eqs: empty += 1
        for e in eqs:
            worst = max(worst, max(g.regret(e.mix).values()))
print("games", n, "with no equilibrium returned", empty, "max regret of returned equilibria", worst)
# NaN payoff cell
A = np.array([[1.0, np.nan], [0.0, 0.5]]); B = np.array([[1.0, 0.0], [0.0, 0.5]])
try:
    g = EmpiricalGame.from_matrices(A, B); print("NaN cell ->", [{k: v.round(3).tolist() for k, v in e.mix.items()} for e in g.nash()])
except Exception as ex: print("NaN cell -> raises", type(ex).__name__, ex)
