"""Content-addressed SQLite cache for model calls (distinct per sample index)."""

from __future__ import annotations

import asyncio
import os
import sqlite3
import threading
from pathlib import Path

from ..core.util import stable_hash
from .base import GenConfig, Model, ModelOutput


def default_cache_dir() -> Path:
    return Path(os.environ.get("OA_CACHE_DIR", Path.cwd() / ".oa_cache"))


class _Store:
    _instances: dict[str, "_Store"] = {}
    _lock = threading.Lock()

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS calls (k TEXT PRIMARY KEY, v TEXT)")
        self.mu = threading.Lock()

    @classmethod
    def get(cls, path: Path) -> "_Store":
        key = str(path.resolve())
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = _Store(path)
            return cls._instances[key]

    def read(self, k: str) -> str | None:
        with self.mu:
            row = self.conn.execute("SELECT v FROM calls WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def write(self, k: str, v: str) -> None:
        with self.mu:
            self.conn.execute("INSERT OR REPLACE INTO calls (k, v) VALUES (?, ?)", (k, v))


class CachedModel(Model):
    """Wraps a model with a persistent cache keyed by (model, messages, config, tools, sample)."""

    def __init__(self, inner: Model, path: str | Path | None = None):
        self.inner = inner
        self.name = inner.name
        self.store = _Store.get(Path(path) if path else default_cache_dir() / "llm_cache.sqlite")
        self._inflight: dict[str, asyncio.Future] = {}
        self.hits = 0
        self.misses = 0

    def key(self, messages, config, tools, sample) -> str:
        return stable_hash(
            self.inner.describe(),
            [m.model_dump(exclude_none=True) for m in messages],
            (config or GenConfig()).model_dump(exclude_none=True),
            [t.model_dump() for t in (tools or [])],
            sample,
            length=32,
        )

    async def generate(self, messages, config=None, tools=None, sample=0) -> ModelOutput:
        k = self.key(messages, config, tools, sample)
        hit = self.store.read(k)
        if hit is not None:
            self.hits += 1
            return _replayed(ModelOutput.model_validate_json(hit))
        if k in self._inflight:  # de-duplicate concurrent identical calls
            return _replayed(await asyncio.shield(self._inflight[k]))
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[k] = fut
        try:
            out = await self.inner.generate(messages, config, tools, sample)
            self.misses += 1
            self.store.write(k, out.model_dump_json())
            fut.set_result(out)
            return out
        except BaseException as e:
            fut.set_exception(e)
            fut.exception()  # mark retrieved
            raise
        finally:
            self._inflight.pop(k, None)

    def describe(self) -> str:
        return self.inner.describe()


def _replayed(out: ModelOutput) -> ModelOutput:
    """A cached output: tokens are kept (they measure the protocol's size), but no API call or
    cost was incurred, so ``calls`` and ``cost_usd`` are zeroed to avoid double counting."""
    out = out.model_copy(deep=True)
    out.cached = True
    out.usage = out.usage.model_copy(update={"calls": 0, "cost_usd": 0.0})
    return out
