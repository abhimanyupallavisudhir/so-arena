"""Monitoring(monitor='both'): alarm = max(LLM monitor, probe) > threshold, and MonitoredReward
pays the *monitor role* for that alarm -> the trainable monitor is credited/penalised for the
probe's (GT-simulated) decisions, independent of what it said."""
import oversight_arena as oa
from oversight_arena.agents.base import Action
from oversight_arena.channels.gt_channels import Auditor, SimulatedProbe
from oversight_arena.core.strategy import Profile, Assignment, Strategy
from oversight_arena.domains.monitoring_sim import AbstractMonitoring
from oversight_arena.mechanisms.monitoring import Monitoring
from oversight_arena.sim.monitoring_agents import ActionActor

class SilentMonitor(oa.Agent):  # always reports suspicion 0
    id = "silent"
    async def act(self, obs):
        return Action(text="SUSPICION: 0", parsed={"suspicion": 0.0})

dom = AbstractMonitoring(n_tasks=200)
mech = Monitoring(monitor="both", monitor_trainable=True, grader=False, probe=SimulatedProbe(auroc=0.9),
                  audit=Auditor(prob=1.0))
prof = lambda a: Profile(assignments={"actor": Assignment(strategy=Strategy(name=a, params={"action": a}))}, label=a)
res = oa.Experiment(dom, mech, {"actor": ActionActor(), "monitor": SilentMonitor()}, [prof("honest"), prof("overt")],
                    progress=False).run()
d = res.df()
m = d[d.role == "monitor"]
print("monitor always said 0; its rewards by actor action:")
print(m.groupby("profile_label").reward.value_counts().unstack(fill_value=0))
print("mean monitor reward:", m.reward.mean())
