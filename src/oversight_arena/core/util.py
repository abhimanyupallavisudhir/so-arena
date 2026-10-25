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


def _code_parts(code: Any) -> list[Any]:
    consts = [_code_parts(c) if hasattr(c, "co_code") else repr(c) for c in code.co_consts]
    return [code.co_code.hex(), consts, list(code.co_names), list(code.co_varnames)]


_PLAIN = (str, int, float, bool, type(None))


def _value_key(v: Any, seen: set[int], depth: int) -> Any:
    """A JSON-able stand-in for a value a function depends on (functions by their code hash)."""
    if isinstance(v, _PLAIN):
        return v
    if callable(v) and depth < 8:
        return {"fn": code_hash(v, _seen=seen, _depth=depth + 1)}
    if isinstance(v, (list, tuple)):
        return [_value_key(x, seen, depth + 1) for x in v]
    if isinstance(v, dict):
        return {str(k): _value_key(x, seen, depth + 1) for k, x in v.items()}
    try:
        canonical_json(v)
        return v
    except Exception:
        return f"<{type(v).__module__}.{type(v).__qualname__}>"


def code_hash(fn: Any, *, _seen: set[int] | None = None, _depth: int = 0) -> str:
    """Hash of what a function does: bytecode (nested code objects included, never memory
    addresses), names, defaults, closure contents (recursively), the plain values of globals it
    reads, the bound arguments of ``functools.partial`` and the instance of a bound method — e.g.
    to key scripted agents, so that ``make(0.3)`` and ``make(0.7)`` from one factory, or
    ``partial(policy, p=0.3)`` and ``partial(policy, p=0.9)``, hash differently. Stable across
    processes."""
    import functools

    seen = set() if _seen is None else _seen
    if id(fn) in seen:
        return "<recursive>"
    seen = seen | {id(fn)}
    if isinstance(fn, functools.partial):
        return stable_hash("partial", code_hash(fn.func, _seen=seen, _depth=_depth),
                           _value_key(list(fn.args), seen, _depth), _value_key(dict(fn.keywords), seen, _depth), length=10)
    owner = getattr(fn, "__self__", None)
    if owner is not None and hasattr(fn, "__func__"):  # bound method: the instance's state matters
        return stable_hash("method", code_hash(fn.__func__, _seen=seen, _depth=_depth),
                           f"<{type(owner).__qualname__}>", _value_key(getattr(owner, "__dict__", {}), seen, _depth), length=10)
    code = getattr(fn, "__code__", None)
    if code is None:  # a callable object
        call = type(fn).__call__ if callable(fn) else None
        return stable_hash(f"<{type(fn).__module__}.{type(fn).__qualname__}>",
                           code_hash(call, _seen=seen, _depth=_depth) if hasattr(call, "__code__") else "",
                           _value_key(getattr(fn, "__dict__", {}), seen, _depth), length=10)
    cells = []
    for c in getattr(fn, "__closure__", None) or ():
        try:
            cells.append(_value_key(c.cell_contents, seen, _depth))
        except ValueError:  # empty cell
            cells.append(None)
    glob = getattr(fn, "__globals__", {}) or {}
    reads = {n: _value_key(glob[n], seen, _depth) for n in sorted(_code_names(code)) if n in glob
             and (isinstance(glob[n], _PLAIN + (list, tuple, dict)) or (callable(glob[n]) and hasattr(glob[n], "__code__")
                                                                          and getattr(glob[n], "__module__", None) == getattr(fn, "__module__", None)))}
    parts = [_code_parts(code), getattr(fn, "__qualname__", ""), _value_key(getattr(fn, "__defaults__", None), seen, _depth),
             _value_key(getattr(fn, "__kwdefaults__", None), seen, _depth)]
    try:
        return stable_hash(parts, cells, reads, length=10)
    except Exception:  # unserialisable contents
        return stable_hash(parts, [type(c).__name__ for c in cells], sorted(reads), length=10)


def _code_names(code: Any) -> set[str]:
    names = set(code.co_names)
    for c in code.co_consts:
        if hasattr(c, "co_names"):
            names |= _code_names(c)
    return names


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
