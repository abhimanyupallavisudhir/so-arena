"""GT-channel randomness (Auditor, SimulatedProbe, SwarmAudit, verification noise) is seeded by the
episode key, which hashes the profile (strategies/stances), the mechanism config (incl. the reward
rule), the agents and the task's GT. So arms being compared get *independent* audit/probe draws:
common random numbers are lost, and e.g. a TeamReward-vs-Whistleblower comparison with identical
behaviour sees different audits."""
import oversight_arena as oa
from oversight_arena.channels.gt_channels import SimulatedProbe
from oversight_arena.core.strategy import Profile, Assignment, Strategy
from oversight_arena.domains.monitoring_sim import AbstractMonitoring
from oversight_arena.domains.swarm import AbstractSwarm
from oversight_arena.mechanisms.monitoring import Monitoring
from oversight_arena.mechanisms.swarm import Swarm, SwarmAudit, TeamReward, Whistleblower
from oversight_arena.sim.monitoring_agents import ActionActor
from oversight_arena.sim.swarm_agents import SwarmWorker

# 1) swarm: identical workers, only the reward rule differs
dom = AbstractSwarm(n_tasks=200)
aud = {}
for rule in (TeamReward(), Whistleblower()):
    res = oa.Experiment(dom, Swarm(audit=SwarmAudit(prob=0.3), reward=rule), SwarmWorker(cheat=1.0), progress=False).run()
    aud[type(rule).__name__] = {r.task_id: r.outcome["audit"]["audited"] for r in res.records}
same = sum(aud["TeamReward"][t] == aud["Whistleblower"][t] for t in aud["TeamReward"])
print(f"swarm: audited under TeamReward {sum(aud['TeamReward'].values())}, under Whistleblower {sum(aud['Whistleblower'].values())};"
      f" audit decision agrees on {same}/200 tasks (CRN would give 200/200)")

# 2) monitoring probe: two *identical-behaviour* strategies that differ only in name
mdom = AbstractMonitoring(n_tasks=200)
mech = Monitoring(monitor="probe", grader=False, probe=SimulatedProbe(auroc=0.8))
p = lambda n: Profile(assignments={"actor": Assignment(strategy=Strategy(name=n, params={"action": "overt"}))}, label=n)
res = oa.Experiment(mdom, mech, {"actor": ActionActor()}, [p("overt_v1"), p("overt_v2")], progress=False).run()
by = {}
for r in res.records:
    by.setdefault(r.profile.label, {})[r.task_id] = r.outcome["alarm"]
diff = sum(by["overt_v1"][t] != by["overt_v2"][t] for t in by["overt_v1"])
print(f"probe: same action, strategy renamed -> alarm differs on {diff}/200 tasks;"
      f" alarm rates {sum(by['overt_v1'].values())/200:.2f} vs {sum(by['overt_v2'].values())/200:.2f}")
