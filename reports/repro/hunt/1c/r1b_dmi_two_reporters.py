"""With 2 reporters DMI() pays both the same value (det M = det M^T), and the per-population
normalisation maps it to +-1: truthful reporting and pure noise earn the same reward."""
import random
from oversight_arena.core.episode import EpisodeRecord
from oversight_arena.core.roles import RoleSpec
from oversight_arena.mechanisms.peer_prediction import DMI

N = 40
truth = [random.Random(t).choice("AB") for t in range(N)]
roles = [RoleSpec(name="reporter_1", kind="expert"), RoleSpec(name="reporter_2", kind="expert")]
def pop(policy):
    return [EpisodeRecord(id=str(t), key=str(t), mechanism="reporters", task_id=f"t{t:02d}", domain="d", roles=roles,
                          outcome={"answers": {r.name: policy(r.name, t) for r in roles}}) for t in range(N)]
policies = {
    "both truthful": lambda r, t: truth[t],
    "r1 truthful, r2 random": lambda r, t: truth[t] if r == "reporter_1" else random.Random(f"x{t}").choice("AB"),
    "both random (seed 3)": lambda r, t: random.Random(f"{r}{t}3").choice("AB"),
    "both random (seed 5)": lambda r, t: random.Random(f"{r}{t}5").choice("AB"),
}
for name, pol in policies.items():
    print(f"{name:24s}", DMI().compute_batch(pop(pol))[0])
