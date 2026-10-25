import numpy as np, itertools
from oversight_arena.analysis.games import EmpiricalGame
rng = np.random.default_rng(1)
games = [(rng.integers(0, 2, (3, 3)).astype(float), rng.integers(0, 2, (3, 3)).astype(float)) for _ in range(400)]
# brute-force: grid over row mixes with col pure / 2-support; check exact equilibria via regret on a fine simplex grid
grid = [np.array(p) / 20 for p in itertools.product(range(21), repeat=3) if sum(p) == 20]
bad = 0
for A, B in games:
    g = EmpiricalGame.from_matrices(A, B)
    found_rows = {i for e in g.nash() for i in np.flatnonzero(e.mix["row"] > 1e-9)}
    true_rows = set()
    for x in grid:
        for y in grid:
            m = {"row": x, "col": y}
            if g.exploitability(m) < 1e-12:
                true_rows |= set(np.flatnonzero(x > 0))
    if true_rows - found_rows:
        bad += 1
        if bad == 1:
            print("A =", A.tolist(), "\nB =", B.tolist())
            print("nash():", [{k: v.round(3).tolist() for k, v in e.mix.items()} for e in g.nash()])
            for x in grid:
                for y in grid:
                    m = {"row": x, "col": y}
                    if g.exploitability(m) < 1e-12 and set(np.flatnonzero(x > 0)) & (true_rows - found_rows):
                        print("exact NE missed:", x.tolist(), y.tolist(), "regret", g.regret(m)); break
                else: continue
                break
print("games (of 400 random binary 3x3) with a strategy used in an exact NE but in none returned by nash():", bad)
