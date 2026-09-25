"""Dataset download/caching helpers (no heavy dependencies)."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def data_dir() -> Path:
    d = Path(os.environ.get("OA_DATA_DIR", Path.home() / ".cache" / "oversight_arena"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, name: str | None = None, max_bytes: int | None = None, timeout: int = 120) -> Path:
    """Download ``url`` into the data dir (cached). ``max_bytes`` fetches only a prefix."""
    name = name or urllib.parse.urlparse(url).path.rsplit("/", 1)[-1]
    path = data_dir() / name
    if path.exists() and path.stat().st_size > 0:
        return path
    req = urllib.request.Request(url, headers={"User-Agent": "oversight-arena/0.1"})
    if max_bytes:
        req.add_header("Range", f"bytes=0-{max_bytes - 1}")
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(path)
    return path


def get_json(url: str, timeout: int = 60) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "oversight-arena/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def hf_rows(dataset: str, split: str, config: str = "default", limit: int | None = None, cache: bool = True) -> list[dict]:
    """Rows of a Hugging Face dataset via the datasets-server JSON API (cached locally)."""
    key = f"hf_{dataset.replace('/', '__')}_{config}_{split}_{limit}.json"
    path = data_dir() / key
    if cache and path.exists():
        return json.loads(path.read_text())
    rows: list[dict] = []
    offset = 0
    while limit is None or len(rows) < limit:
        n = 100 if limit is None else min(100, limit - len(rows))
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": n})
        page = get_json(f"https://datasets-server.huggingface.co/rows?{q}")
        batch = [r["row"] for r in page.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < n or offset >= page.get("num_rows_total", offset):
            break
    if cache:
        path.write_text(json.dumps(rows))
    return rows
