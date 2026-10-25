# Attempt 4: SQLScorer turns "role never answered"/missing field into an OBSERVED quality of 0.0,
# while AnswerScorer/ForecastScorer report the same situation as status="error".
import sys
sys.path.insert(0, "/tmp/att4/src")
from dataclasses import asdict
from oversight_arena.domains import SQLiteSnapshot, SQLScorer
from oversight_arena.scoring import AnswerScorer
from oversight_arena.types import Action, Event, Outcome, Reward, Role, Run, Task
setup = "CREATE TABLE t(x); INSERT INTO t VALUES (1);"
run = Run("r", Task("t", "p"), "m", (Role("w"), Role("j", False)), 0,
          (Event(0, "j", "action", asdict(Action(data={"scores": {"w": 0.9}}))),),   # w never acted
          Outcome({"w": Reward(0.9)}), "complete", {}, {})
print("SQLScorer   :", SQLScorer({"t": (SQLiteSnapshot(setup),)}, {"t": "SELECT x FROM t"}, ("w",))(run).scores["w"]["quality"])
print("AnswerScorer:", AnswerScorer({"t": 1}, ("w",))(run).scores["w"]["quality"])
bad = Run("r2", Task("t", "p"), "m", (Role("w"),), 0,
          (Event(0, "w", "action", asdict(Action(data={"query": ["SELECT 1"]}))),),
          Outcome({"w": Reward(0.9)}), "complete", {}, {})
try:
    SQLScorer({"t": (SQLiteSnapshot(setup),)}, {"t": "SELECT x FROM t"}, ("w",))(bad)
except Exception as e:
    print("non-string query ->", type(e).__name__, e)
