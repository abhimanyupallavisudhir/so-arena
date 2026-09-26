# Check swarm theory claims numerically: b_R(n=3,o=0.8)=g/3; single crossing of E[Delta](q) (unique mixed eq);
# b_R against brute-force; regime labels at the demo bounties.
import numpy as np, itertools
from dataclasses import replace
from oversight_arena.theory.swarm_game import SwarmParams, report_equilibrium_bounty, expected_advantage, equilibria, regime, basin_of_reporting
p = SwarmParams(n=3, o=0.8, g=0.3, a=0.0, c=0.0)
print("b_R n=3 o=0.8 g=0.3:", report_equilibrium_bounty(p), "g/3 =", 0.1)
multi = 0; checked = 0; bad_bR = 0
rng = np.random.default_rng(0)
for _ in range(3000):
    q = SwarmParams(n=int(rng.integers(3, 9)), g=float(rng.uniform(0.05, 1)), b=float(rng.uniform(0, 1)), a=float(rng.uniform(0, 0.9)),
                    c=float(rng.uniform(0, 0.5)) * (rng.random() < 0.5), o=float(rng.uniform(0.05, 1)))
    qs = np.linspace(0, 1, 2001); f = np.array([expected_advantage(x, q) for x in qs])
    s = np.sign(f[np.abs(f) > 1e-12]); changes = int((np.diff(s) != 0).sum()); checked += 1
    if changes > 1: multi += 1
    bR = report_equilibrium_bounty(q)
    if bR > 1e-9:  # E[Delta](1) at b_R should be ~0
        if abs(expected_advantage(1.0, replace(q, b=bR))) > 1e-9: bad_bR += 1
print("random param sets", checked, "with >1 sign change of E[Delta](q):", multi, "| b_R not a root:", bad_bR)
for b in (0.05, 0.2, 0.45):
    pp = replace(SwarmParams(n=3, o=0.8, g=0.3, P=0.3), b=b)
    print(f"b={b}: regime {regime(pp)}, eqs {[(round(e['q'],3), e['type'], e['stable']) for e in equilibria(pp)]}, basin {basin_of_reporting(pp):.3f}")
