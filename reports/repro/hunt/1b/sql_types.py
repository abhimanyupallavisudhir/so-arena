# SQLArtifactGT uses exact equality (no float tolerance, int vs float, no ROUND awareness) and
# is_ordered() treats ANY "ORDER BY" (e.g. inside a subquery / window) as an ordered comparison.
import json
from pathlib import Path
from oversight_arena.domains.sql import PrivateSQL, SQLArtifactGT, run_query, same_result, is_ordered
from oversight_arena.core.episode import EpisodeRecord
import oversight_arena.domains.sql as S

ts = {t.metadata["qid"]: t for t in PrivateSQL(format="open").tasks()}
qs = {q["id"]: q for q in json.loads((Path(S.__file__).parent / "assets" / "chinook_questions.json").read_text())}
def gt(qid, q):
    rec = EpisodeRecord.model_construct(outcome={"artifacts": {"proposer": q}})
    return SQLArtifactGT().score(ts[qid], rec)["proposer"]
db = ts["q01"].resources["db_path"]
for qid, alt in [
    ("q06", "SELECT ROUND(SUM(Total) / COUNT(*), 2) FROM Invoice"),
    ("q06", "SELECT ROUND(AVG(Total) * 100) / 100 FROM Invoice"),
    ("q24", "SELECT ROUND(AVG(t.Milliseconds / 1000.0)) FROM Track t JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = 'Metal'"),
    ("q30", "SELECT SUM(t.Milliseconds) / 60000.0 FROM Track t JOIN Album a ON a.AlbumId = t.AlbumId WHERE a.Title = 'Let There Be Rock'"),
]:
    print(qid, "|", qs[qid]["question"])
    print("   gold:", run_query(db, qs[qid]["gold"])[1], " alt:", run_query(db, alt)[1], " GT:", gt(qid, alt))
