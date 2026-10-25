# Brute-force check of exact with-replacement BoN (uniform tie-break) vs library formulas
import itertools, sys, random
which = sys.argv[1]
sys.path.insert(0, f"/tmp/att{which}/src")
if which == "3":
    from oversight_arena.optimization import best_of_n_weights as f
    call = lambda v, n, b: f(v, n, base=b)
else:
    from oversight_arena.optimization import best_of_n as f
    call = lambda v, n, b: f(v, n, b)
def brute(v, n, b):
    out = [0.0]*len(v)
    for draw in itertools.product(range(len(v)), repeat=n):
        p = 1.0
        for i in draw: p *= b[i]
        m = max(v[i] for i in draw); winners = [i for i in draw if v[i] == m]
        for i in winners: out[i] += p / len(winners)
    return out
rng = random.Random(0); worst = 0
for _ in range(300):
    k = rng.randint(1, 5); v = [rng.choice([0, 1, 2, 0.5]) for _ in range(k)]
    b = [rng.choice([0, 1, 2, 3]) for _ in range(k)]
    if sum(b) == 0: b[0] = 1
    b = [x / sum(b) for x in b]; n = rng.randint(1, 4)
    worst = max(worst, max(abs(a - c) for a, c in zip(call(v, n, b), brute(v, n, b))))
print(f"att{which}: max abs error vs brute force = {worst:.2e}")
