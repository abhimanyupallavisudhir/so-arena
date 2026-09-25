"""DMI divides every population's payments by that population's max |payment| -> payoffs of
different profiles are on different, profile-dependent scales (not the Kong 2020 rule)."""
import random
from oversight_arena.core.episode import EpisodeRecord
from oversight_arena.core.roles import RoleSpec
from oversight_arena.mechanisms.peer_prediction import DMI

rng = random.Random(0)
N = 40
truth = [rng.choice("AB") for _ in range(N)]
roles = [RoleSpec(name=f"reporter_{i}", kind="expert") for i in (1, 2, 3)]

def pop(policy):
    recs = []
    for t in range(N):
        ans = {r.name: policy(r.name, t) for r in roles}
        recs.append(EpisodeRecord(id=str(t), key=str(t), mechanism="reporters", task_id=f"t{t:02d}", domain="d",
                                  roles=roles, outcome={"answers": ans}))
    return recs

def noisy(p):  # report truth w.p. p, else flip
    def f(r, t):
        x = truth[t]
        return x if random.Random(f"{r}{t}{p}").random() < p else ("B" if x == "A" else "A")
    return f

truthful = pop(noisy(1.0))
weak = pop(noisy(0.7))             # everybody much less informative
dev = pop(lambda r, t: noisy(1.0)(r, t) if r != "reporter_1" else noisy(0.7)(r, t))  # reporter_1 deviates

rule = DMI()
for name, P in [("all truthful", truthful), ("all 70%-informative", weak), ("reporter_1 deviates to 70%", dev)]:
    rw = rule.compute_batch(P)[0]
    # the documented (unnormalised) payment:
    import numpy as np
    raw = {}
    tasks = sorted(r.task_id for r in P); half = N // 2
    tab = {r.name: {rec.task_id: rec.outcome["answers"][r.name] for rec in P} for r in roles}
    for r in tab:
        vals = []
        for q in tab:
            if q == r: continue
            ms = []
            for part in (tasks[:half], tasks[half:]):
                M = np.zeros((2, 2))
                for t in part:
                    M["AB".index(tab[r][t]), "AB".index(tab[q][t])] += 1
                ms.append(np.linalg.det(M))
            vals.append(ms[0] * ms[1])
        raw[r] = float(np.mean(vals))
    print(f"{name:28s} DMI() reward r1={rw['reporter_1']:.3f}  documented det*det r1={raw['reporter_1']:.1f}")
