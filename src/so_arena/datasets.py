"""Dataset caching helpers.

Large or restrictively licensed data is never committed: loaders download it on demand into the cache
directory (``$SO_ARENA_DATA`` or ``~/.cache/so_arena``). Small, permissively licensed samples used by
tests and demos live in ``so_arena/data/samples``.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

SAMPLES_DIR = Path(__file__).resolve().parent / "data" / "samples"


def cache_dir() -> Path:
    d = Path(os.environ.get("SO_ARENA_DATA", Path.home() / ".cache" / "so_arena"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, name: str | None = None, *, force: bool = False, timeout: float = 120.0,
             max_bytes: int | None = None) -> Path:
    """Download ``url`` into the cache (once) and return the local path.

    ``max_bytes`` fetches only a prefix (useful for streaming-decompressable archives).
    """
    name = name or hashlib.sha256(url.encode()).hexdigest()[:16] + "_" + url.rsplit("/", 1)[-1]
    path = cache_dir() / name
    if path.exists() and not force:
        return path
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "so-arena/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        read = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            read += len(chunk)
            if max_bytes is not None and read >= max_bytes:
                break
    tmp.replace(path)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def sample_path(name: str) -> Path:
    return SAMPLES_DIR / name
