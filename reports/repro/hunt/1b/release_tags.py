# create_release(names="auto") publishes strategy names/profile labels whenever no stance is set,
# even if the strategies carry GT-relevant tags (StrategyTag is a GT scorer: tags["honest"]).
from types import SimpleNamespace as NS
from oversight_arena.release.release import _item
from oversight_arena.core.strategy import Strategy
from oversight_arena.core.task import Task
t = Task(id="q1", domain="forecasting", question="?")
from oversight_arena.core.episode import BoundInfo
s1, s2 = Strategy(name="sandbag_forecaster", tags={"honest": False}), Strategy(name="honest_forecaster", tags={"honest": True})
b = {"forecaster_1": BoundInfo(agent="m", strategy_id=s1.id, strategy_name=s1.name, tags=s1.tags),
     "forecaster_2": BoundInfo(agent="m", strategy_id=s2.id, strategy_name=s2.name, tags=s2.tags)}
rec = NS(outcome={"forecasts": {"forecaster_1": 0.2, "forecaster_2": 0.7}}, bound=b, id="e1", task_id="q1",
         mechanism="Forecast", mechanism_hash="h", reward_rule="r", profile=NS(label="honest vs sandbagger", id="p-123"),
         rewards={}, meta={}, created_at="now", transcript=NS(entries=[]))
it = _item(rec, transcripts=False)
print("published profile:", it["profile"], "| strategies:", it["strategies"])
