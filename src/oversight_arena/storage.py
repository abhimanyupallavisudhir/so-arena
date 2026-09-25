"""Immutable run files and append-only, versioned evaluation sidecars."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from .types import Evaluation, Run, canonical, digest, evaluation_from_dict, run_from_dict


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def _put(self, relative: str, data: dict):
        path = self.path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = canonical(data) + "\n"
        # Complete temporary file + atomic no-clobber link: readers never see half a run.
        fd, temp = tempfile.mkstemp(dir=path.parent)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            try:
                os.link(temp, path)
            except FileExistsError:
                if path.read_text() != payload:
                    raise ValueError(f"Conflicting immutable record: {relative}") from None
        finally:
            os.unlink(temp)

    def save_run(self, run: Run):
        self._put(f"runs/{digest(run.id)}.json", asdict(run))

    def save_evaluation(self, evaluation: Evaluation):
        if not (self.path / "runs" / f"{digest(evaluation.run_id)}.json").exists():
            raise ValueError("Evaluation must reference a stored run")
        key = digest([evaluation.run_id, evaluation.scorer, evaluation.version])
        self._put(f"evaluations/{key}.json", asdict(evaluation))

    def runs(self) -> list[Run]:
        return [
            run_from_dict(json.loads(p.read_text()))
            for p in sorted((self.path / "runs").glob("*.json"))
        ]

    def evaluations(self) -> list[Evaluation]:
        return [
            evaluation_from_dict(json.loads(p.read_text()))
            for p in sorted((self.path / "evaluations").glob("*.json"))
        ]

    def export_public(self, path: str | Path):
        """Mechanism-only publication: no evaluator sidecars, manifest or private events.

        Public action text is still model-generated content; review it before publication.
        """
        rows = []
        for run in self.runs():
            rows.append(
                {
                    "schema_version": 1,
                    "id": run.id,
                    "task_id": run.task.id,
                    "prompt": run.task.prompt,
                    "mechanism": run.mechanism,
                    "status": run.status,
                    "usage": run.usage,
                    "events": [asdict(e) for e in run.events if e.audience is None],
                    "outcome": asdict(run.outcome) if run.outcome else None,
                }
            )
        Path(path).write_text("".join(canonical(row) + "\n" for row in rows))
