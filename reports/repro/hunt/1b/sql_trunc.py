# SQLArtifactGT compares only the first 1001 fetched rows (run_query(max_rows=1000) fetches
# max_rows+1 and never flags truncation). A wrong query that returns a SUPERSET of a large gold
# result, with the same prefix, is scored correct. The same comparison filters distractors at load.
import sqlite3, os, tempfile
from oversight_arena.domains.sql import SQLArtifactGT, run_query, same_result, is_ordered
from oversight_arena.core.task import Task
from oversight_arena.core.episode import EpisodeRecord

path = os.path.join(tempfile.mkdtemp(), "db.sqlite")
con = sqlite3.connect(path)
con.execute("create table Track(TrackId integer primary key, Milliseconds int)")
con.executemany("insert into Track values (?, ?)", [(i, 1000 * i) for i in range(1, 3504)])
con.commit(); con.close()

gold = "SELECT TrackId FROM Track WHERE Milliseconds > 1000000"            # 2503 rows
wrong = "SELECT TrackId FROM Track WHERE Milliseconds > 1000000 OR TrackId > 3000"  # same rows here (subset check)
wrong2 = "SELECT TrackId FROM Track WHERE Milliseconds >= 0 ORDER BY TrackId DESC"  # nothing like gold
wrong3 = "SELECT TrackId FROM Track WHERE Milliseconds > 1000000 AND TrackId < 2400"  # drops 1103 gold rows
task = Task(id="t", domain="sql", question="q", answer_type="sql", resources={"db_path": path}, gt={"gold_sql": gold})
for name, q in [("gold", gold), ("drops 1103 rows", wrong3)]:
    rec = EpisodeRecord.model_construct(outcome={"artifacts": {"proposer": q}})
    n = len(run_query(path, q, max_rows=100000)[1])
    print(f"{name:16} true rows={n:5}  SQLArtifactGT={SQLArtifactGT().score(task, rec)}")
