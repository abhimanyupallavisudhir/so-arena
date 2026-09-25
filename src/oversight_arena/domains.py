"""Reusable ground-truth scorers and task manifests, independent of protocols."""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from .analysis import binary_score
from .core import JSON, Episode, Evaluation, PublicTask, Task, finite


def last_content(episode: Episode, role: str) -> JSON:
    for event in reversed(episode.events):
        if event.actor == role and event.kind == "message":
            return event.content
    raise ValueError(f"No message from {role}")


@dataclass(frozen=True)
class ExactMatch:
    roles: tuple[str, ...]
    answer_key: str = "answer"
    target_key: str = "answer"
    name: str = "exact_match"
    version: str = "1"

    async def __call__(self, task: Task, episode: Episode) -> Evaluation:
        if self.target_key not in task.evaluator_data:
            return Evaluation(self.name, self.version, status="unavailable")
        scores = {}
        for role in self.roles:
            content = last_content(episode, role)
            answer = content.get(self.answer_key) if isinstance(content, dict) else content
            scores[role] = {"quality": float(answer == task.evaluator_data[self.target_key])}
        return Evaluation(self.name, self.version, scores)


@dataclass(frozen=True)
class Forecast:
    roles: tuple[str, ...]
    name: str = "forecast_brier"
    version: str = "1"

    async def __call__(self, task: Task, episode: Episode) -> Evaluation:
        outcome = task.evaluator_data.get("outcome")
        if outcome is None:
            return Evaluation(self.name, self.version, status="pending")
        if outcome not in (0, 1):
            raise ValueError("Forecast outcome must be binary")
        scores = {}
        for role in self.roles:
            probability = last_content(episode, role)["probability"]
            score = binary_score(probability, outcome)
            scores[role] = {
                "quality": 1 + score,
                "negative_brier": score,
                "probability": probability,
            }
        return Evaluation(
            self.name,
            self.version,
            scores,
            evidence={
                "resolution_source": task.evaluator_data.get("source"),
                "resolved_at": task.evaluator_data.get("resolved_at"),
            },
        )


@dataclass(frozen=True)
class SQLExecution:
    """Read-only SQLite execution against a private, researcher-supplied fixture.

    The reference query and database define result equivalence, not semantic intent.
    Progress/row limits bound this fixture runner; use an OS sandbox for hostile SQL.
    """

    roles: tuple[str, ...]
    ordered: bool = False
    max_rows: int = 10000
    vm_steps: int = 100000
    max_bytes: int = 1000000
    name: str = "sql_execution"
    version: str = "1"

    def __post_init__(self):
        if self.max_rows < 1 or self.vm_steps < 1 or self.max_bytes < 1:
            raise ValueError("Execution limits must be positive")

    def execute(self, connection: sqlite3.Connection, query: str) -> list[tuple]:
        if not isinstance(query, str) or len(query) > 100000:
            raise ValueError("SQL query must be a bounded string")
        allowed = {
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_FUNCTION,
            sqlite3.SQLITE_RECURSIVE,
        }

        def authorize(action, arg1, arg2, database, source):
            if action not in allowed or (
                action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() == "load_extension"
            ):
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        previous_length = connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, min(previous_length, self.max_bytes))
        connection.set_authorizer(authorize)
        calls = 0

        def progress():
            nonlocal calls
            calls += 1
            return int(calls * 100 >= self.vm_steps)

        connection.set_progress_handler(progress, 100)
        try:
            rows = []
            used_bytes = 0
            for row in connection.execute(query):
                used_bytes += sum(
                    len(cell) if isinstance(cell, bytes) else len(str(cell).encode())
                    for cell in row
                )
                if len(rows) >= self.max_rows or used_bytes > self.max_bytes:
                    raise ValueError("SQL result budget exceeded")
                rows.append(row)
            return rows
        finally:
            connection.set_authorizer(None)
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, previous_length)
            connection.set_progress_handler(None, 0)

    async def __call__(self, task: Task, episode: Episode) -> Evaluation:
        # Fixture script is trusted evaluator input, never an agent-produced script.
        if not {"fixture_sql", "reference_sql"} <= task.evaluator_data.keys():
            return Evaluation(self.name, self.version, status="unavailable")
        connection = sqlite3.connect(":memory:")
        errors = {}
        try:
            connection.executescript(task.evaluator_data["fixture_sql"])
            expected = self.execute(connection, task.evaluator_data["reference_sql"])
            scores = {}
            for role in self.roles:
                query = last_content(episode, role)["sql"]
                try:
                    actual = self.execute(connection, query)
                    equal = (
                        actual == expected if self.ordered else Counter(actual) == Counter(expected)
                    )
                    scores[role] = {"quality": float(equal), "execution_success": 1.0}
                except (sqlite3.Error, ValueError) as error:
                    scores[role] = {"quality": 0.0, "execution_success": 0.0}
                    errors[role] = str(error)
        finally:
            connection.close()
        return Evaluation(self.name, self.version, scores, evidence={"errors": errors})


@dataclass(frozen=True)
class ArtifactScorer:
    """Adapter for hidden code tests, Lean kernels, chess engines or rubric evaluators.

    The callback owns sandboxing/tool dependencies. It receives oracle data here,
    unlike mechanism-time verification tools. Record engine/version in `version`.
    """

    roles: tuple[str, ...]
    evaluate: Callable[[JSON, dict[str, JSON]], Awaitable[dict[str, float]]]
    name: str
    version: str

    async def __call__(self, task: Task, episode: Episode) -> Evaluation:
        results = {}
        for role in self.roles:
            metrics = await self.evaluate(last_content(episode, role), task.evaluator_data)
            if "quality" not in metrics:
                raise ValueError("Artifact evaluator must provide quality")
            results[role] = {k: finite(v) for k, v in metrics.items()}
        return Evaluation(self.name, self.version, results)


def tasks_from_rows(
    rows: Sequence[dict], *, dataset: str, version: str, split: str, gap: str, license: str
) -> list[Task]:
    """Explicit data provenance and gap classification; no implicit oracle exposure."""
    if gap not in {"capability", "information", "tool_access", "cost", "temporal", "mixed"}:
        raise ValueError("Unknown gap type")
    if split not in {"train", "validation", "test", "unspecified"}:
        raise ValueError("Unknown split")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task ids")
    return [
        Task(
            PublicTask(str(row["id"]), row["prompt"], row.get("public", {})),
            row.get("oracle", {}),
            row.get("role_data", {}),
            split,
            {"dataset": dataset, "version": version, "gap": gap, "license": license},
        )
        for row in rows
    ]
