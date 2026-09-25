"""Small shared utilities: stable hashing, deterministic RNG, JSON helpers, async helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from collections.abc import Awaitable, Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


def _default(o: Any) -> Any:
    """JSON fallback that is *stable across processes* (never embeds memory addresses)."""
    if isinstance(o, BaseModel):
        return o.model_dump(mode="json")
    if isinstance(o, (set, frozenset)):
        return sorted(o, key=repr)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, bytes):
        return hashlib.sha256(o).hexdigest()
    if hasattr(o, "describe") and callable(o.describe):
        return o.describe()
    if callable(o) and hasattr(o, "__code__"):
        return f"<fn {getattr(o, '__module__', '')}.{getattr(o, '__qualname__', '')}:{code_hash(o)}>"
    if hasattr(o, "__name__"):
        return f"<{getattr(o, '__module__', '')}.{o.__name__}>"
    fields = {k: v for k, v in getattr(o, "__dict__", {}).items() if not k.startswith("_")}
    if fields:
        return {"__type__": f"{type(o).__module__}.{type(o).__qualname__}", **fields}
    return f"<{type(o).__module__}.{type(o).__qualname__}>"


def code_hash(fn: Any) -> str:
    """Hash of a function's code (bytecode, constants, names), e.g. to key scripted agents."""
    code = getattr(fn, "__code__", None)
    if code is None:
        return ""
    parts = [code.co_code.hex(), repr(code.co_consts), code.co_names, getattr(fn, "__qualname__", "")]
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:10]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialisation (sorted keys, no whitespace) used for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default, ensure_ascii=False)


def stable_hash(*parts: Any, length: int = 16) -> str:
    """Stable content hash of arbitrary (JSON-able) objects."""
    h = hashlib.sha256()
    for p in parts:
        h.update(canonical_json(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()[:length]


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def rng_for(*parts: Any) -> random.Random:
    """A `random.Random` seeded deterministically from arbitrary parts."""
    return random.Random(int(stable_hash(*parts, length=16), 16))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_log(p: float, eps: float = 1e-6) -> float:
    return math.log(clamp(p, eps, 1.0))


def normalize(d: dict[str, float], eps: float = 0.0) -> dict[str, float]:
    vals = {k: max(float(v), 0.0) + eps for k, v in d.items()}
    s = sum(vals.values())
    if s <= 0:
        n = len(vals)
        return {k: 1.0 / n for k in vals} if n else {}
    return {k: v / s for k, v in vals.items()}


def write_jsonl(path: str | Path, rows: Iterable[Any], append: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            if isinstance(r, BaseModel):
                f.write(r.model_dump_json() + "\n")
            else:
                f.write(json.dumps(r, default=_default, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


async def gather_limited(
    coros: Iterable[Callable[[], Awaitable[T]]], limit: int = 8
) -> list[T]:
    """Run zero-arg coroutine factories with bounded concurrency, preserving order."""
    sem = asyncio.Semaphore(max(1, limit))
    factories = list(coros)

    async def _run(f: Callable[[], Awaitable[T]]) -> T:
        async with sem:
            return await f()

    return await asyncio.gather(*[_run(f) for f in factories])


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine from sync code (works in plain scripts; in notebooks use `await`)."""
    try:
        asyncio.get_running_loop()
        running = True
    except RuntimeError:
        running = False
    if running:
        if hasattr(coro, "close"):
            coro.close()  # type: ignore[union-attr]
        raise RuntimeError(
            "run_sync() called inside a running event loop (e.g. Jupyter). Use the async API: `await ...arun()`."
        )
    return asyncio.run(coro)  # type: ignore[arg-type]


def truncate(s: str, n: int = 400) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def word_count(s: str) -> int:
    return len(s.split())
