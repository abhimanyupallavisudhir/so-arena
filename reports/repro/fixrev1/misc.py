import math, numpy as np
from oversight_arena.analysis import bon as B
from oversight_arena.elicitation.optimize import ranked, OptimizationTrace, Candidate, is_score
for n in (0, 0.5, -1):
    try: print("bon_kl", n, "->", B.bon_kl(n))
    except Exception as e: print("bon_kl", n, "-> raises", type(e).__name__, e)
# NaN variants in ranked(): numpy float32 NaN, Decimal NaN, string
class C:
    def __init__(s, r): s.reward = r
xs = [C(0.0), C(np.float32("nan")), C(-1.0), C(0.5)]
print("ranked with float32 NaN:", [x.reward for x in ranked(xs)], "| is_score(np.float32(nan)) =", is_score(np.float32("nan")))
