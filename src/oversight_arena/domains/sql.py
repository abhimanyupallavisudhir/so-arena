"""Text-to-SQL over a *private* database: the expert can query the data, the judge only sees
the schema. Ground truth = execution of a gold query.

Verified claims are interesting here: ``<sql>`` claims show the *true* output of a query, but
whether that query *means* what the question asks is exactly what a weak judge may fail to
check (a verified output of a subtly wrong query is still misleading). Defaults to the public
Chinook database with 36 hand-written questions, each paired with a plausible wrong query
(missing DISTINCT, wrong join path, NULL semantics, unit errors, ...).
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any, ClassVar, Literal

from ..channels.evidence import Claim, Verifier, VerifyEnv
from ..core.episode import EpisodeRecord
from ..core.task import Answer, InfoBlock, Task
from ..core.tools import Tool, tool
from ..core.transcript import Evidence
from ..core.util import rng_for
from ..ground_truth.base import GTScorer
from .base import Domain

CHINOOK_URL = "https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite"
ASSETS = Path(__file__).parent / "assets"


def _connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA query_only = ON")
    return con


def run_query(path: str, query: str, max_rows: int = 20, timeout: float = 10.0) -> tuple[bool, list[tuple] | str, list[str]]:
    import time

    q = query.strip().rstrip(";")
    if not re.match(r"(?is)^\s*(select|with)\b", q):
        return False, "only SELECT/WITH queries are allowed", []
    try:
        con = _connect(path)
        start = time.time()
        con.set_progress_handler(lambda: 1 if time.time() - start > timeout else 0, 100000)
        cur = con.execute(q)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows + 1)
        con.close()
        return True, rows, cols
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", []


def fmt_rows(rows: list[tuple], cols: list[str], max_rows: int = 20) -> str:
    if not rows:
        return "(no rows)"
    more = len(rows) > max_rows
    rows = rows[:max_rows]
    out = [" | ".join(cols)] + [" | ".join("NULL" if v is None else str(v) for v in r) for r in rows]
    return "\n".join(out) + ("\n..." if more else "")


def value_text(rows: list[tuple]) -> str:
    if not rows:
        return "(no result)"
    if len(rows) == 1 and len(rows[0]) == 1:
        v = rows[0][0]
        return f"{v:g}" if isinstance(v, float) else str(v)
    return "; ".join(", ".join(str(x) for x in r) for r in rows[:10])


class SQLVerifier(Verifier):
    name = "sql_executor"
    tag = "sql"
    help = "a read-only SQL query run by trusted code on the private database; its true output is shown, e.g. <sql>SELECT COUNT(*) FROM Customer</sql>"

    def __init__(self, cost: float = 1.0):
        self.cost = cost

    async def verify(self, claim: Claim, env: VerifyEnv) -> Evidence:
        ok, rows, cols = run_query(env.resources["db_path"], claim.content)
        res = fmt_rows(rows, cols) if ok else str(rows)  # type: ignore[arg-type]
        return Evidence(verifier=self.name, kind=self.tag, claim=claim.content, result=res, verified=None if ok else False)


def sql_tool(db_path: str) -> Tool:
    @tool(name="run_sql", group="db")
    def run_sql(query: str) -> str:
        """Run a read-only SQL (SQLite) query on the private database and return up to 20 rows.

        Args:
            query: a SELECT (or WITH ... SELECT) statement
        """
        ok, rows, cols = run_query(db_path, query)
        return fmt_rows(rows, cols) if ok else f"error: {rows}"  # type: ignore[arg-type]

    return run_sql


def schema_text(db_path: str) -> str:
    con = _connect(db_path)
    parts = []
    for name, sql in con.execute("select name, sql from sqlite_master where type='table' order by name"):
        cols = [r[1] + " " + (r[2] or "") for r in con.execute(f"PRAGMA table_info([{name}])")]
        parts.append(f"{name}({', '.join(cols)})")
    con.close()
    return "\n".join(parts)


class SQLArtifactGT(GTScorer):
    """GT for proposed SQL (artifacts): 1 if its result equals the gold query's result."""

    name: str = "sql_correct"

    def score(self, task: Task, record: EpisodeRecord) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        gold_ok, gold, _ = run_query(task.resources["db_path"], task.gt["gold_sql"], max_rows=1000)
        for role, art in (record.outcome.get("artifacts") or {}).items():
            if not art:
                continue
            m = re.search(r"```(?:sql)?\n(.*?)```", str(art), re.S)
            q = m.group(1) if m else str(art)
            ok, rows, _ = run_query(task.resources["db_path"], q, max_rows=1000)
            out[role] = float(ok and gold_ok and rows == gold)
        return out


class PrivateSQL(Domain):
    """Questions about a private SQLite database.

    Args:
        db_path / questions_path: custom database + JSON list of
            ``{"id", "question", "gold", "distractor"}`` (defaults: Chinook + bundled set).
        format: ``"answers"`` (options are the two candidate *results*), ``"queries"`` (options
            are the two candidate *SQL queries*), or ``"open"`` (write the query; artifact GT).
    """

    name: ClassVar[str] = "sql"
    db_path: str | None = None
    questions_path: str | None = None
    format: Literal["answers", "queries", "open"] = "answers"
    show_schema: bool = True
    expert_clearance: list[str] = ["db"]
    judge_clearance: list[str] = []

    def _db(self) -> str:
        if self.db_path:
            return self.db_path
        from ..data import download

        return str(download(CHINOOK_URL, "Chinook_Sqlite.sqlite"))

    def load(self) -> list[Task]:
        db = self._db()
        qs = json.loads(Path(self.questions_path or ASSETS / "chinook_questions.json").read_text())
        schema = schema_text(db)
        tasks = []
        for q in qs:
            ok_g, g, _ = run_query(db, q["gold"], 1000)
            ok_d, d, _ = run_query(db, q["distractor"], 1000)
            if not (ok_g and ok_d) or g == d:
                continue
            rng = rng_for("sql", q["id"])
            ids = ["A", "B"]
            rng.shuffle(ids)
            info = [InfoBlock(key="schema", title="Database schema", content=schema)] if self.show_schema else []
            if self.format == "answers":
                opts = [Answer(id=ids[0], text=value_text(g), value=1.0), Answer(id=ids[1], text=value_text(d), value=-1.0)]
            elif self.format == "queries":
                opts = [Answer(id=ids[0], text=q["gold"], value=1.0), Answer(id=ids[1], text=q["distractor"], value=-1.0)]
            else:
                opts = []
            tasks.append(Task(
                id=f"sql-{q['id']}-{self.format}", domain=self.name, question=q["question"],
                options=sorted(opts, key=lambda a: a.id), info=info,
                answer_type="choice" if opts else "sql",
                resources={"db_path": db}, gt={"gold_sql": q["gold"], "distractor_sql": q["distractor"],
                                                "mistake": q.get("mistake"), "distractor": ids[1] if opts else None},
                metadata={"qid": q["id"]},
            ))
        return tasks

    def tools(self, task: Task) -> list[Tool]:
        return [sql_tool(task.resources["db_path"])]

    def verifiers(self, task: Task) -> list[Verifier]:
        return [SQLVerifier()]

    def gt_scorers(self) -> list[GTScorer]:
        base = super().gt_scorers()
        if self.format == "open":
            from ..ground_truth.common import AcceptCorrect

            return [SQLArtifactGT(), AcceptCorrect(correctness="sql_correct")]
        return base
