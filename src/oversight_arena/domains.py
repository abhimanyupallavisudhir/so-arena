"""Executable domain components; expensive/private evaluation remains outside mechanisms."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass

from .scoring import last_action
from .types import Claim, Evaluation, Measurement, Run, canonical


@dataclass(frozen=True)
class SQLiteSnapshot:
    """Trusted fixture SQL, fresh in-memory database per query, read-only model queries.

    A progress handler limits VM work and a row cap bounds results. This is a small
    reproducible domain, not an OS sandbox for arbitrary code or a production DB connector.
    """

    setup: str
    max_steps: int = 100_000
    max_rows: int = 1000
    max_bytes: int = 1_000_000

    def query(self, sql: str) -> list[list]:
        if self.max_steps < 1 or self.max_rows < 1 or self.max_bytes < 1:
            raise ValueError("SQL budgets must be positive")
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript(self.setup)
            db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.max_bytes)
            db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, self.max_bytes)
            db.execute("PRAGMA query_only=ON")
            allowed = {
                sqlite3.SQLITE_SELECT,
                sqlite3.SQLITE_READ,
                sqlite3.SQLITE_FUNCTION,
                sqlite3.SQLITE_RECURSIVE,
            }
            db.set_authorizer(
                lambda op, a, b, c, d: (
                    sqlite3.SQLITE_OK
                    if op in allowed and b != "load_extension"
                    else sqlite3.SQLITE_DENY
                )
            )
            calls = 0

            def progress():
                nonlocal calls
                calls += 100
                return int(calls > self.max_steps)

            db.set_progress_handler(progress, 100)
            cursor = db.execute(sql)
            if cursor.description is None:
                raise ValueError("Expected a query")
            rows, size = [], 0
            for row in cursor:
                if len(rows) >= self.max_rows:
                    raise ValueError("Query exceeded row budget")
                try:
                    size += len(canonical(list(row)).encode())
                except TypeError as exc:
                    raise ValueError("Query returned a non-JSON value") from exc
                if size > self.max_bytes:
                    raise ValueError("Query exceeded byte budget")
                rows.append(list(row))
            return rows

    async def tool(self, arguments: dict) -> dict:
        try:
            return {"rows": self.query(arguments["query"])}
        except (sqlite3.Error, ValueError) as exc:
            return {"error": str(exc)}

    async def check_claim(self, claim: Claim) -> tuple[bool | None, str]:
        actual = self.query(claim.artifact["query"])
        return actual == claim.artifact["rows"], "Exact ordered query result on this snapshot"


@dataclass
class SQLScorer:
    """Execution equivalence across independently supplied private database instances.

    Passing is exact for these instances, not a proof the SQL matches natural-language intent.
    A dataset may attach a separate human semantic-fit measurement.
    """

    fixtures: Mapping[str, tuple[SQLiteSnapshot, ...]]
    reference_queries: Mapping[str, str]
    roles: tuple[str, ...]
    ordered: bool = False
    version: str = "1"

    def __call__(self, run: Run) -> Evaluation:
        scores = {}
        for role in self.roles:
            source = "private SQL execution suite"
            if run.status != "complete":
                m = Measurement(None, "error", source, detail="Run failed")
            elif run.task.id not in self.fixtures or run.task.id not in self.reference_queries:
                m = Measurement(None, "unavailable", source)
            else:
                try:
                    fixtures = self.fixtures[run.task.id]
                    if not fixtures:
                        raise ValueError("Empty SQL evaluation suite")
                    expected = [f.query(self.reference_queries[run.task.id]) for f in fixtures]
                    try:
                        query = last_action(run, role).data["query"]
                        actual = [f.query(query) for f in fixtures]

                        def normalize(rows):
                            return rows if self.ordered else Counter(canonical(r) for r in rows)

                        value = sum(
                            normalize(a) == normalize(b)
                            for a, b in zip(actual, expected, strict=True)
                        ) / len(fixtures)
                    except (sqlite3.Error, KeyError, ValueError):
                        value = 0.0
                    m = Measurement(
                        value,
                        "observed",
                        source,
                        detail=f"Fraction of {len(fixtures)} fixture databases passed",
                    )
                except (sqlite3.Error, ValueError) as exc:
                    m = Measurement(None, "error", source, detail=str(exc))
            scores[role] = {"quality": m}
        return Evaluation(run.id, "sql", self.version, scores)


async def chess_legality(claim: Claim) -> tuple[bool | None, str]:
    """Check a UCI variation without revealing engine strength or position evaluation."""
    import chess

    board = chess.Board(claim.artifact["fen"])
    if not board.is_valid():
        return False, "Invalid starting position"
    for uci in claim.artifact["moves"]:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return False, "Invalid UCI move"
        if move not in board.legal_moves:
            return False, f"Illegal move: {uci}"
        board.push(move)
    return True, "Every move is legal; this does not establish the line is good or forced"


@dataclass
class ChessScorer:
    """Independent fixed-node engine quality, from the initial side-to-move perspective.

    UCI engine binary is provided by the experimenter. Centipawn regret is a proxy, not
    mathematical chess truth; record engine/version/nodes and avoid comparing Elo pools.
    """

    engine: str
    engine_version: str
    roles: tuple[str, ...]
    nodes: int = 10_000

    def __call__(self, run: Run) -> Evaluation:
        import chess
        import chess.engine

        if self.nodes < 1:
            raise ValueError("nodes must be positive")
        scores = {}
        with chess.engine.SimpleEngine.popen_uci(self.engine, timeout=30) as engine:
            for role in self.roles:
                try:
                    if run.status != "complete":
                        raise ValueError("Run failed")
                    board = chess.Board(run.task.data["fen"])
                    if not board.is_valid() or board.is_game_over():
                        raise ValueError("Need a valid nonterminal position")
                    move = chess.Move.from_uci(last_action(run, role).data["move"])
                    if move not in board.legal_moves:
                        m = Measurement(
                            -200_000.0,
                            "observed",
                            self.engine_version,
                            "proxy",
                            "Illegal move sentinel; report legality separately",
                        )
                    else:
                        limit = chess.engine.Limit(nodes=self.nodes)
                        best = (
                            engine.analyse(board, limit)["score"]
                            .pov(board.turn)
                            .score(mate_score=100_000)
                        )
                        chosen = (
                            engine.analyse(board, limit, root_moves=[move])["score"]
                            .pov(board.turn)
                            .score(mate_score=100_000)
                        )
                        m = Measurement(
                            float(chosen - best),
                            "observed",
                            self.engine_version,
                            "proxy",
                            f"Negative centipawn regret at {self.nodes} nodes",
                        )
                except (ValueError, KeyError, chess.engine.EngineError) as exc:
                    m = Measurement(None, "error", self.engine_version, "proxy", str(exc))
                scores[role] = {"quality": m}
        return Evaluation(
            run.id, "chess-engine", f"{self.engine_version}:nodes={self.nodes}", scores
        )
