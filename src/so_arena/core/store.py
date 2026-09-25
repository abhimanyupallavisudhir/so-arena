"""Run directories: config, append-only episode log, metrics and figures.

Layout::

    runs/<run_id>/
        run.json          # configuration, versions, timestamps
        episodes.jsonl    # one Episode per line, appended as episodes finish (crash-safe, resumable)
        metrics.json      # computed metrics
        figures/          # plots
        report.html       # self-contained report
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from so_arena.core.mechanism import Episode


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def episodes_path(self) -> Path:
        return self.path / "episodes.jsonl"

    @property
    def figures_dir(self) -> Path:
        d = self.path / "figures"
        d.mkdir(exist_ok=True)
        return d

    def write_meta(self, meta: dict[str, Any]) -> None:
        self.save_json("run.json", meta)

    def meta(self) -> dict[str, Any]:
        return self.load_json("run.json") or {}

    def append(self, ep: Episode) -> None:
        line = ep.model_dump_json()
        with self._lock, open(self.episodes_path, "a") as f:
            f.write(line + "\n")
            f.flush()

    def episodes(self) -> list[Episode]:
        if not self.episodes_path.exists():
            return []
        eps: dict[str, Episode] = {}
        with open(self.episodes_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    ep = Episode.model_validate_json(line)
                    eps[ep.id] = ep  # later lines (e.g. re-scored) override earlier ones
        return list(eps.values())

    def ids(self) -> set[str]:
        return {e.id for e in self.episodes() if e.error is None}

    def rewrite(self, episodes: list[Episode]) -> None:
        tmp = self.episodes_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for ep in episodes:
                f.write(ep.model_dump_json() + "\n")
        tmp.replace(self.episodes_path)

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.path / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def load_json(self, name: str) -> Any:
        p = self.path / name
        return json.loads(p.read_text()) if p.exists() else None
