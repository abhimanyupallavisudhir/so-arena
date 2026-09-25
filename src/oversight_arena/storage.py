"""Transactional, append-only episode/evaluation storage using SQLite."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

from .core import Episode, Evaluation, canonical, digest, episode_from_dict


class RunStore:
    def __init__(self, path: str | Path):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS evaluations (
                episode_id TEXT NOT NULL REFERENCES episodes(id),
                ordinal INTEGER NOT NULL, payload TEXT NOT NULL, checksum TEXT NOT NULL,
                recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                PRIMARY KEY (episode_id, ordinal));
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def save(self, episode: Episode) -> str:
        episode.validate()
        payload = canonical({**asdict(episode), "evaluations": []})
        with self.connection:
            self.connection.execute(
                "INSERT OR IGNORE INTO episodes VALUES (?, ?)", (episode.id, payload)
            )
            existing = self.connection.execute(
                "SELECT payload FROM episodes WHERE id=?", (episode.id,)
            ).fetchone()[0]
            if existing != payload:
                raise ValueError("Conflicting episode payload")
            previous = self.connection.execute(
                "SELECT payload FROM evaluations WHERE episode_id=? ORDER BY ordinal", (episode.id,)
            ).fetchall()
            revisions = [canonical(asdict(e)) for e in episode.evaluations]
            # Allow idempotent saves of an older prefix, never replacement of a revision.
            if any(a[0] != b for a, b in zip(previous, revisions, strict=False)):
                raise ValueError("Cannot replace evaluation history; append a new revision")
            for ordinal in range(len(previous), len(revisions)):
                revision = revisions[ordinal]
                self.connection.execute(
                    "INSERT INTO evaluations "
                    "(episode_id, ordinal, payload, checksum) VALUES (?, ?, ?, ?)",
                    (episode.id, ordinal, revision, digest(json.loads(revision))),
                )
        return episode.id

    def get(self, episode_id: str) -> Episode:
        row = self.connection.execute(
            "SELECT payload FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()
        if row is None:
            raise KeyError(episode_id)
        data = json.loads(row[0])
        revisions = []
        for payload, checksum in self.connection.execute(
            "SELECT payload, checksum FROM evaluations WHERE episode_id=? ORDER BY ordinal",
            (episode_id,),
        ):
            value = json.loads(payload)
            if digest(value) != checksum:
                raise ValueError("Evaluation checksum mismatch")
            revisions.append(value)
        data["evaluations"] = revisions
        return episode_from_dict(data)

    def records(self) -> list[Episode]:
        return [
            self.get(row[0])
            for row in self.connection.execute("SELECT id FROM episodes ORDER BY id")
        ]

    def export_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w") as file:
            for record in self.records():
                file.write(canonical(asdict(record)) + "\n")

    def import_jsonl(self, path: str | Path) -> int:
        # Parse and validate everything before beginning the write transaction.
        records = [
            episode_from_dict(json.loads(line))
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
        for record in records:
            self.save(record)
        return len(records)


def public_result(episode: Episode) -> dict:
    """Publication view: public transcript, scalar scores, no private channels/oracle evidence.

    Review public task data and public agent text before publication; automatic
    projection cannot detect secrets an agent voluntarily included in its text.
    Source id refers to the original full record, not a hash of this projection.
    """
    latest: dict[str, Evaluation] = {}
    for evaluation in episode.evaluations:
        latest[evaluation.scorer] = evaluation
    return {
        "source_episode_id": episode.id,
        "schema": "oversight-arena-public-v1",
        "task": asdict(episode.task),
        "mechanism": episode.mechanism,
        "config": episode.config,
        "seed": episode.seed,
        "roles": [asdict(r) for r in episode.roles],
        "rewards": {r: reward.value for r, reward in episode.rewards.items()},
        "output": episode.output,
        "usage": episode.usage,
        "transcript": [
            {"actor": e.actor, "kind": e.kind, "content": e.content}
            for e in episode.events
            if e.recipients is None
        ],
        "evaluations": [
            {
                "scorer": e.scorer,
                "version": e.version,
                "status": e.status,
                "per_agent": e.per_agent,
                "outcomes": e.outcomes,
            }
            for e in latest.values()
        ],
    }
