# Attempt 4: SQLScorer unordered mode compares canonical JSON -> 150 != 150.0; ordered mode says equal
import sys
sys.path.insert(0, "/tmp/att4/src")
from oversight_arena.domains import SQLiteSnapshot, SQLScorer
from oversight_arena.types import Action, Event, Outcome, Reward, Role, Run, Task
from dataclasses import asdict

setup = "CREATE TABLE invoices(amount INTEGER, refunded INTEGER); INSERT INTO invoices VALUES (100,0),(30,1),(50,0);"
ref = "SELECT SUM(amount) FROM invoices WHERE refunded = 0"
fx = {"t": (SQLiteSnapshot(setup),)}
def mkrun(q):
    return Run("id-" + q, Task("t", "p"), "m", (Role("w"),), 0,
               (Event(0, "w", "action", asdict(Action(data={"query": q}))),),
               Outcome({"w": Reward(0.0)}), "complete", {}, {})
for q in [ref, "SELECT TOTAL(amount) FROM invoices WHERE refunded = 0",
          "SELECT SUM(amount) * 1.0 FROM invoices WHERE NOT refunded"]:
    print(SQLiteSnapshot(setup).query(q), end="  ")
    for ordered in (False, True):
        ev = SQLScorer(fx, {"t": ref}, ("w",), ordered=ordered)(mkrun(q))
        print(f"ordered={ordered}: quality={ev.scores['w']['quality'].value}", end="  ")
    print()
