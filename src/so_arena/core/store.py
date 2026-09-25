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
import logging
import os
import threading
from pathlib import Path
from typing import Any

from so_arena.core.items import TaskItem
from so_arena.core.mechanism import Episode

log = logging.getLogger("so_arena")


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
        """Append one episode line. If a crash left a partial last line, it is terminated first, so the
        new record stays readable (the partial one is skipped by :meth:`episodes`)."""
        line = (ep.model_dump_json() + "\n").encode("utf-8")
        with self._lock, open(self.episodes_path, "ab+") as f:
            if f.seek(0, os.SEEK_END) > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    line = b"\n" + line
            f.write(line)  # append mode: always at the end
            f.flush()

    def episodes(self) -> list[Episode]:
        """All stored episodes (a later line with the same id overrides an earlier one). Lines that do
        not parse - e.g. one half-written when a run was killed - are skipped with a warning, so the
        run can still be resumed (the lost episodes simply run again)."""
        if not self.episodes_path.exists():
            return []
        eps: dict[str, Episode] = {}
        bad = []
        with open(self.episodes_path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = Episode.model_validate_json(line)
                except ValueError:  # pydantic's ValidationError (also for malformed JSON) is a ValueError
                    bad.append(n)
                    continue
                eps[ep.id] = ep  # later lines (e.g. re-scored) override earlier ones
        if bad:
            log.warning("%s: skipped %d unreadable line(s) (e.g. line %d), probably cut off by a crash",
                        self.episodes_path, len(bad), bad[0])
        return list(eps.values())

    def ids(self) -> set[str]:
        return {e.id for e in self.episodes() if e.error is None}

    def rewrite(self, episodes: list[Episode]) -> None:
        tmp = self.episodes_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            for ep in episodes:
                f.write(ep.model_dump_json() + "\n")
        tmp.replace(self.episodes_path)

    def save_items(self, items: list[TaskItem]) -> Path:
        """Store the (uncensored) items; merges with items already stored. Written to a temporary file
        and moved into place, so a crash never leaves a truncated item file behind."""
        path = self.path / "items.jsonl"
        existing = {it.id: it for it in self.items()}
        for it in items:
            existing[it.id] = it
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with self._lock:
            with open(tmp, "w") as f:
                for it in existing.values():
                    f.write(it.model_dump_json() + "\n")
            tmp.replace(path)
        return path

    def items(self) -> list[TaskItem]:
        path = self.path / "items.jsonl"
        if not path.exists():
            return []
        return [TaskItem.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]

    def save_json(self, name: str, obj: Any) -> Path:
        p = self.path / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def load_json(self, name: str) -> Any:
        p = self.path / name
        return json.loads(p.read_text()) if p.exists() else None
