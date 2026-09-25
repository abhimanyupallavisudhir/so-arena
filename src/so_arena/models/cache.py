"""On-disk response cache (sqlite) for model calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import ClassVar

from so_arena.core.types import Completion, GenerateOptions, Message
from so_arena.models.base import Model


def cache_key(model_name: str, messages, options: GenerateOptions, sample_index: int) -> str:
    payload = {
        "model": model_name,
        "messages": [m.model_dump() for m in messages],
        "options": options.model_dump(exclude_none=True),
        "sample_index": sample_index,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class ResponseCache:
    """A tiny sqlite key-value store of completions."""

    _instances: ClassVar[dict[str, "ResponseCache"]] = {}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()

    @classmethod
    def at(cls, directory: str | Path) -> "ResponseCache":
        p = str(Path(directory) / "responses.sqlite")
        if p not in cls._instances:
            cls._instances[p] = cls(p)
        return cls._instances[p]

    def get(self, key: str) -> Completion | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return Completion.model_validate_json(row[0]) if row else None

    def put(self, key: str, completion: Completion) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
                (key, completion.model_dump_json()),
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


class CachedModel(Model):
    """Wraps a model with a :class:`ResponseCache`. Cached hits report zero new cost."""

    def __init__(self, inner: Model, cache: ResponseCache):
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.supports_logprobs = inner.supports_logprobs

    async def generate(self, messages, options=None, *, sample_index=0):
        options = options or GenerateOptions()
        key = cache_key(getattr(self.inner, "identity", self.name), messages, options, sample_index)
        hit = self.cache.get(key)
        if hit is not None:
            usage = hit.usage.model_copy(update={"cost_usd": 0.0, "cached_calls": 1, "calls": 0})
            return hit.model_copy(update={"cached": True, "usage": usage})
        out = await self.inner.generate(list(messages), options, sample_index=sample_index)
        self.cache.put(key, out)
        return out


def _messages(messages) -> list[Message]:
    return [m if isinstance(m, Message) else Message(**m) for m in messages]
