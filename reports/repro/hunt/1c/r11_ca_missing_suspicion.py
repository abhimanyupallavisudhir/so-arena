"""results_from_control_arena: a sample whose monitor produced no suspicion score gets
suspicion=None -> alarm False and MonitoredReward(continuous) penalty `o.get('suspicion') or 0.0`
= 0: missing monitor output is scored as 'perfectly unsuspicious', with no coverage report."""
from types import SimpleNamespace as NS
from oversight_arena.integrations.control_arena import results_from_control_arena

def sample(sid, attack, susp):
    msgs = [NS(role="assistant", text="done", tool_calls=[], metadata=({"suspicion_score": susp} if susp is not None else {}))]
    return NS(id=sid, epoch=1, error=None, store={}, messages=msgs,
              metadata={"main_task_name": "p1", "side_task_name": "exfil" if attack else None, "eval_mode": "attack" if attack else "honest"},
              scores={"apps_scorer": NS(value={"main_task_success": "C", "side_task_success": "C" if attack else "I"})})
log = NS(location="mem", eval=NS(metadata={"micro_protocol": {"type": "tm"}, "setting": {"type": "apps"}}, task="t", model="m"),
         samples=[sample("h", False, 0.3), sample("a", True, None)])
res = results_from_control_arena([log], transcripts=False)
for r in res.records:
    print(r.outcome["eval_mode"], "suspicion", r.outcome["suspicion"], "alarm", r.outcome["alarm"], "reward", r.rewards,
          "gt honest", r.gt["honest"])
