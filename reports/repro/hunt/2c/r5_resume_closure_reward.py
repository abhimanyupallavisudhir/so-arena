"""Episode ids hash the reward rule with describe_config, which describes a function by its
qualified name only (unlike policies, whose closures are hashed by describe_callable). Changing a
closure parameter of a FunctionReward / RandomAudit oracle keeps the id, so resuming a store
silently returns episodes rewarded under the OLD rule."""
import tempfile
import so_arena as soa
from so_arena.core.rewards import FunctionReward, JudgeScore, RandomAudit
from so_arena.core.runner import Profile, run_episodes, run_sync, PlayerSpec
from so_arena.core.store import RunStore
from so_arena.core.items import binary_item
from so_arena.mechanisms.qa import Propaganda

def length_penalty(lam):
    return FunctionReward(lambda ep: {"agent": -lam * len(ep.turns[-2].text)}, name="len_pen")

item = binary_item("q1", "2+2?", "4", "5", shuffle_seed=1)
prof = Profile(name="p", players={"agent": PlayerSpec(policy=soa.ScriptedPolicy("It is clearly this one."), stance="true"),
                                  "judge": soa.ScriptedPolicy('{"A": 0.6, "B": 0.4}')})
store = RunStore(tempfile.mkdtemp())
for lam in (1.0, 100.0):
    mech = Propaganda(reward=length_penalty(lam))
    ep = run_sync(run_episodes(mech, [item], [prof], store=store, ground_truth=[]))[0]
    print(f"lam={lam}: config_hash={mech.config_hash()} episode={ep.id} reward={ep.rewards['agent']}")

def oracle_with_noise(noise):
    return lambda ep: {"agent": -noise}
for noise in (0.0, 5.0):
    mech = Propaganda(reward=RandomAudit(JudgeScore("log"), oracle_with_noise(noise), p=1.0, mode="ipw"))
    print(f"audit oracle noise={noise}: config_hash={mech.config_hash()}")
