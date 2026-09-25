"""Stateful work: forkable workspaces for agentic tasks (the paired-branch design).

Oversight of real work is oversight of *state changes*: an agent edits a repository, writes to a
database, sends mail, moves money. To compare an honest and a deceptive agent on the same task, both
must start from the same state $S_0$ and act in isolated branches; to optimize (best-of-N), every
sampled candidate needs its own branch as well. This module provides that.

* A **snapshot** is an immutable, content-addressed state held by a :class:`StateStore`: a tree of
  agent-visible files (repository, database files, mailbox, ledger - the working directory of shell
  tools) plus ``hidden`` JSON state that only the environment reads (simulated counterparties'
  private preferences, which card is the company's, ...). Identical states have identical ids.
* A :class:`Workspace` is a mutable working copy forked from a snapshot. The game forks one for each
  decision of a role with state access (one per sampled candidate in branch mode) and freezes it
  afterwards. With ``write`` access the frozen result becomes the game's current state; with ``read``
  access it is discarded, so a reviewer can run the tests without changing the work under review.
* An :class:`Environment` builds $S_0$ for a task item and supplies the tools that act on workspaces;
  :func:`diff_states` renders what changed (text diffs, database rows, appended mail or ledger
  records) for reviewers - never the hidden part.

Isolation is by copying directories, which suits small workspaces. It is NOT a security sandbox:
shell tools run commands as the current user (with a timeout and a memory limit). Use a
container-backed environment for untrusted agents.
"""

from __future__ import annotations

import abc
import contextlib
import contextvars
import difflib
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from so_arena.core.items import TaskItem
from so_arena.core.tools import Tool, ToolResult
from so_arena.core.verification import Verification, Verifier

if TYPE_CHECKING:
    from so_arena.core.game import Game
    from so_arena.core.mechanism import Episode

StateAccess = Literal["none", "read", "write"]
STATE_ACCESS = ("none", "read", "write")
# caches that tools leave behind; removed at freeze so they never change a state's identity
IGNORED_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
DB_SUFFIXES = (".db", ".sqlite", ".sqlite3")

_CURRENT: contextvars.ContextVar["WorkspaceSlot | None"] = contextvars.ContextVar("so_arena_workspace", default=None)


class WorkspaceSlot:
    """The working copy of one decision, forked from ``parent`` on first use - so decisions that never
    touch the state (e.g. an LLM reviewer that calls no tools) cost no copy."""

    def __init__(self, store: "StateStore", parent: str, access: StateAccess):
        self.store, self.parent, self.access = store, parent, access
        self.ws: Workspace | None = None

    @classmethod
    def of(cls, ws: "Workspace") -> "WorkspaceSlot":
        slot = cls(ws.store, ws.parent or "", ws.access)
        slot.ws = ws
        return slot

    def get(self) -> "Workspace":
        if self.ws is None:
            self.ws = self.store.fork(self.parent, access=self.access)
        return self.ws

    def close(self, keep: bool) -> str | None:
        """Freeze the working copy (``keep``: returns the resulting state, the parent if it was never
        touched) or discard it (returns None)."""
        if self.ws is None:
            return self.parent if keep else None
        if keep:
            return self.store.freeze(self.ws)
        self.store.discard(self.ws)
        return None


def current_workspace() -> "Workspace | None":
    """The working copy of the decision being made (while a role with state access acts), else None."""
    slot = _CURRENT.get()
    return slot.get() if slot is not None else None


@contextlib.contextmanager
def using_workspace(ws: "Workspace | WorkspaceSlot | None") -> Iterator["WorkspaceSlot | None"]:
    slot = WorkspaceSlot.of(ws) if isinstance(ws, Workspace) else ws
    token = _CURRENT.set(slot)
    try:
        yield slot
    finally:
        _CURRENT.reset(token)


class CommandResult(BaseModel):
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def render(self, max_chars: int = 4000) -> str:
        parts = []
        if self.timed_out:
            parts.append("(timed out)")
        if self.stdout.strip():
            parts.append(self.stdout.rstrip())
        if self.stderr.strip():
            parts.append("[stderr]\n" + self.stderr.rstrip())
        parts.append(f"[exit code {self.returncode}]")
        return _clip("\n".join(parts), max_chars)


def _clip(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    head = text[: n * 2 // 3]
    tail = text[-(n // 3 - 40):] if n > 120 else ""
    return f"{head}\n... [{len(text) - len(head) - len(tail)} characters omitted] ...\n{tail}"


def _limits(memory_mb: int) -> Callable[[], None] | None:
    if sys.platform == "win32":  # pragma: no cover
        return None

    def apply() -> None:  # pragma: no cover - runs in the child
        try:
            import resource

            lim = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except Exception:
            pass

    return apply


# ============================================================================== workspaces


class Workspace:
    """A mutable working copy of a snapshot: ``root`` (agent-visible files) and ``hidden`` (environment state).

    Paths are relative to ``root`` and may not escape it. ``events`` collects environment-level
    records a decision produced (e.g. messages sent), for tools that want to report them.
    """

    def __init__(self, store: "StateStore", work_dir: Path, hidden: dict[str, Any], parent: str | None,
                 access: StateAccess = "write"):
        self.store = store
        self.work_dir = work_dir
        self.root = work_dir / "files"
        self.hidden = hidden
        self.parent = parent
        self.access = access
        self.closed = False
        self.events: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------- files
    def path(self, rel: str | Path) -> Path:
        rel = str(rel).strip()
        p = (self.root / rel.lstrip("/")).resolve() if not os.path.isabs(rel) else Path(rel).resolve()
        root = self.root.resolve()
        if p != root and root not in p.parents:
            raise ValueError(f"path {rel!r} is outside the workspace")
        return p

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def read_text(self, rel: str) -> str:
        return self.path(rel).read_text(encoding="utf-8", errors="replace")

    def write_text(self, rel: str, text: str) -> None:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def delete(self, rel: str) -> None:
        p = self.path(rel)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()

    def files(self, sub: str = "") -> list[str]:
        """Relative paths of all files (sorted), excluding ignored caches and ``.git`` internals."""
        return _list_files(self.path(sub) if sub else self.root, self.root)

    def append_jsonl(self, rel: str, record: dict[str, Any]) -> None:
        p = self.path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def read_jsonl(self, rel: str) -> list[dict[str, Any]]:
        p = self.path(rel)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    @contextlib.contextmanager
    def db(self, rel: str, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        """A connection to a SQLite database in the workspace (committed and closed on exit)."""
        p = self.path(rel)
        if not readonly:
            p.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True) if readonly else sqlite3.connect(p)
        try:
            yield con
            if not readonly:
                con.commit()
        finally:
            con.close()

    # -------------------------------------------------------------------------- commands
    def run(self, command: str | Sequence[str], *, timeout: float = 30.0, env: Mapping[str, str] | None = None,
            input: str | None = None, memory_mb: int = 1024) -> CommandResult:
        """Run a command in ``root`` (a string runs under ``bash -c``) with a timeout and memory limit.

        The environment is minimal (PATH, HOME=root, no bytecode files, fixed hash seed), so
        commands do not see the caller's credentials.
        """
        base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.root), "LANG": "C.UTF-8",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0", "PYTHONPATH": str(self.root),
                "GIT_TERMINAL_PROMPT": "0", "SO_ARENA_WORKSPACE": "1"}
        args = ["bash", "-c", command] if isinstance(command, str) else list(command)
        try:
            proc = subprocess.run(args, cwd=self.root, capture_output=True, text=True, timeout=timeout,
                                  input=input, env={**base, **(env or {})}, preexec_fn=_limits(memory_mb))
            return CommandResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
            return CommandResult(returncode=-1, stdout=out, stderr=err, timed_out=True)
        except OSError as e:
            return CommandResult(returncode=127, stderr=str(e))

    # -------------------------------------------------------------------------- lifecycle
    def diff(self, against: str | None = None, **kw: Any) -> str:
        """What changed in this working copy relative to snapshot ``against`` (default: the parent)."""
        base = against or self.parent
        if base is None:
            raise ValueError("no snapshot to diff against")
        return diff_trees(self.store.files_dir(base), self.root, **kw)

    def __repr__(self) -> str:
        return f"Workspace(parent={self.parent!r}, access={self.access!r}, root={str(self.root)!r})"


class SnapshotView:
    """Read-only access to a stored snapshot (for scorers, verifiers and reports)."""

    def __init__(self, store: "StateStore", sid: str):
        self.store, self.id = store, sid
        self.root = store.files_dir(sid)
        if not self.root.exists():
            raise KeyError(f"no snapshot {sid!r} in {store.root}")

    @property
    def hidden(self) -> dict[str, Any]:
        p = self.store.snapshot_dir(self.id) / "hidden.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root.resolve() and self.root.resolve() not in p.parents:
            raise ValueError(f"path {rel!r} is outside the snapshot")
        return p

    def exists(self, rel: str) -> bool:
        return self.path(rel).exists()

    def read_text(self, rel: str) -> str:
        return self.path(rel).read_text(encoding="utf-8", errors="replace")

    def files(self) -> list[str]:
        return _list_files(self.root, self.root)

    def read_jsonl(self, rel: str) -> list[dict[str, Any]]:
        p = self.path(rel)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]

    def query(self, db: str, sql: str, params: Sequence[Any] = ()) -> list[tuple]:
        con = sqlite3.connect(f"file:{self.path(db)}?mode=ro", uri=True)
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()


def _list_files(start: Path, root: Path) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_NAMES and d != ".git")
        for fn in sorted(filenames):
            out.append(str((Path(dirpath) / fn).relative_to(root)))
    return sorted(out)


def _scrub(root: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(root, topdown=True):
        for d in list(dirnames):
            if d in IGNORED_NAMES:
                shutil.rmtree(Path(dirpath) / d, ignore_errors=True)
                dirnames.remove(d)
        for fn in filenames:
            if fn.endswith((".pyc", "-journal", "-wal", "-shm")):
                with contextlib.suppress(OSError):
                    (Path(dirpath) / fn).unlink()


def content_id(root: Path, hidden: Mapping[str, Any]) -> str:
    """Content address of a state: every entry's relative path, type, executable bit and bytes, plus ``hidden``."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel_dir = Path(dirpath).relative_to(root)
        if not dirnames and not filenames and rel_dir != Path("."):
            h.update(b"D\0" + str(rel_dir).encode() + b"\0")
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            rel = str(p.relative_to(root)).encode()
            if p.is_symlink():
                h.update(b"L\0" + rel + b"\0" + os.readlink(p).encode() + b"\0")
                continue
            mode = b"x" if os.access(p, os.X_OK) else b"-"
            fh = hashlib.sha256(p.read_bytes()).digest()
            h.update(b"F\0" + rel + b"\0" + mode + fh)
    h.update(b"H\0" + json.dumps(hidden, sort_keys=True, default=str).encode())
    return "s-" + h.hexdigest()[:24]


# ============================================================================== store


class StateStore:
    """Content-addressed snapshots on disk.

    Layout: ``<root>/snapshots/<id>/files/...`` and ``<root>/snapshots/<id>/hidden.json``; working
    copies live in ``<root>/work/`` until frozen. Default root: ``<cache_dir>/states`` if a cache
    directory is configured, else a fresh temporary directory. Give experiments whose episodes are
    kept a persistent root (e.g. ``<run dir>/states``) so final states can be inspected and re-scored.
    """

    def __init__(self, root: str | Path | None = None):
        if root is None:
            from so_arena.config import settings

            root = settings.cache_dir / "states" if settings.cache_dir else tempfile.mkdtemp(prefix="so_arena_states_")
        self.root = Path(root).resolve()
        (self.root / "snapshots").mkdir(parents=True, exist_ok=True)
        (self.root / "work").mkdir(parents=True, exist_ok=True)

    def snapshot_dir(self, sid: str) -> Path:
        if not sid or "/" in sid or sid.startswith("."):
            raise ValueError(f"bad snapshot id {sid!r}")
        return self.root / "snapshots" / sid

    def files_dir(self, sid: str) -> Path:
        return self.snapshot_dir(sid) / "files"

    def has(self, sid: str | None) -> bool:
        return bool(sid) and self.files_dir(sid).exists()  # type: ignore[arg-type]

    def _new_work_dir(self) -> Path:
        d = self.root / "work" / uuid.uuid4().hex
        (d / "files").mkdir(parents=True)
        return d

    def create(self, files: Mapping[str, str | bytes] | None = None, *, hidden: Mapping[str, Any] | None = None,
               source_dir: str | Path | None = None, build: Callable[[Workspace], None] | None = None) -> str:
        """Build a snapshot from a directory tree, a ``{path: content}`` mapping and/or a ``build(ws)`` callback."""
        wd = self._new_work_dir()
        ws = Workspace(self, wd, dict(hidden or {}), parent=None, access="write")
        if source_dir is not None:
            shutil.copytree(source_dir, ws.root, symlinks=True, dirs_exist_ok=True)
        for rel, content in (files or {}).items():
            p = ws.path(rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                p.write_bytes(content)
            else:
                p.write_text(content, encoding="utf-8")
        if build is not None:
            build(ws)
        return self.freeze(ws)

    def fork(self, sid: str, *, access: StateAccess = "write") -> Workspace:
        """A fresh working copy of snapshot ``sid``."""
        src = self.snapshot_dir(sid)
        if not (src / "files").exists():
            raise KeyError(f"no snapshot {sid!r} in {self.root}")
        wd = self.root / "work" / uuid.uuid4().hex
        wd.mkdir(parents=True)
        shutil.copytree(src / "files", wd / "files", symlinks=True)
        hidden = json.loads((src / "hidden.json").read_text()) if (src / "hidden.json").exists() else {}
        return Workspace(self, wd, hidden, parent=sid, access=access)

    def freeze(self, ws: Workspace) -> str:
        """Store a working copy as a snapshot (content-addressed: an unchanged fork maps back to its parent)."""
        if ws.closed:
            raise RuntimeError("workspace already frozen or discarded")
        _scrub(ws.root)
        sid = content_id(ws.root, ws.hidden)
        target = self.snapshot_dir(sid)
        if target.exists():
            shutil.rmtree(ws.work_dir, ignore_errors=True)
        else:
            (ws.work_dir / "hidden.json").write_text(json.dumps(ws.hidden, sort_keys=True, indent=1, default=str))
            try:
                os.rename(ws.work_dir, target)
            except OSError:  # another task froze identical content first
                if not target.exists():
                    raise
                shutil.rmtree(ws.work_dir, ignore_errors=True)
        ws.closed = True
        return sid

    def discard(self, ws: Workspace) -> None:
        if not ws.closed:
            shutil.rmtree(ws.work_dir, ignore_errors=True)
            ws.closed = True

    def view(self, sid: str) -> SnapshotView:
        return SnapshotView(self, sid)

    def diff(self, a: str, b: str, **kw: Any) -> str:
        """What changed from snapshot ``a`` to snapshot ``b`` (see :func:`diff_trees`)."""
        return diff_trees(self.files_dir(a), self.files_dir(b), **kw)

    @contextlib.contextmanager
    def scratch(self, sid: str) -> Iterator[Workspace]:
        """A throwaway working copy (e.g. to run hidden tests on a final state), discarded on exit."""
        ws = self.fork(sid, access="read")
        try:
            yield ws
        finally:
            self.discard(ws)


# ============================================================================== diffs


def _is_text(data: bytes) -> bool:
    if b"\0" in data[:8192]:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _sqlite_rows(path: Path) -> dict[str, tuple[list[str], dict[Any, tuple]]]:
    """table -> (columns, {key: row}) keyed by rowid (or the full row for WITHOUT ROWID tables)."""
    out: dict[str, tuple[list[str], dict[Any, tuple]]] = {}
    if not path.exists():
        return out
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                                            "ORDER BY name")]
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            try:
                rows = con.execute(f'SELECT rowid, * FROM "{t}"').fetchall()
                out[t] = (cols, {r[0]: tuple(r[1:]) for r in rows})
            except sqlite3.OperationalError:  # WITHOUT ROWID
                rows = con.execute(f'SELECT * FROM "{t}"').fetchall()
                out[t] = (cols, {r: r for r in rows})
    finally:
        con.close()
    return out


def _fmt_row(cols: Sequence[str], row: tuple) -> str:
    return "{" + ", ".join(f"{c}: {v!r}" for c, v in zip(cols, row)) + "}"


def diff_sqlite(a: Path, b: Path, *, max_rows: int = 15) -> str:
    """Row-level differences between two SQLite databases (added, deleted and changed rows per table)."""
    ta, tb = _sqlite_rows(a), _sqlite_rows(b)
    lines = []
    for t in sorted(set(ta) | set(tb)):
        if t not in ta:
            cols, rows = tb[t]
            lines.append(f"table {t}: created with {len(rows)} rows")
            lines += [f"  + {_fmt_row(cols, r)}" for r in list(rows.values())[:max_rows]]
            continue
        if t not in tb:
            lines.append(f"table {t}: dropped ({len(ta[t][1])} rows)")
            continue
        (ca, ra), (cb, rb) = ta[t], tb[t]
        added = [k for k in rb if k not in ra]
        deleted = [k for k in ra if k not in rb]
        changed = [k for k in rb if k in ra and rb[k] != ra[k]]
        if ca != cb:
            lines.append(f"table {t}: columns changed from {ca} to {cb}")
        if not (added or deleted or changed):
            continue
        lines.append(f"table {t}: {len(added)} added, {len(deleted)} deleted, {len(changed)} changed rows")
        lines += [f"  + {_fmt_row(cb, rb[k])}" for k in added[:max_rows]]
        lines += [f"  - {_fmt_row(ca, ra[k])}" for k in deleted[:max_rows]]
        for k in changed[:max_rows]:
            before, after = dict(zip(ca, ra[k])), dict(zip(cb, rb[k]))
            ch = ", ".join(f"{c}: {before.get(c)!r} -> {after.get(c)!r}" for c in cb if before.get(c) != after.get(c))
            lines.append(f"  ~ (rowid {k}) {ch}")
        hidden = max(0, len(added) - max_rows) + max(0, len(deleted) - max_rows) + max(0, len(changed) - max_rows)
        if hidden:
            lines.append(f"  ... ({hidden} more rows)")
    return "\n".join(lines) if lines else "(no row changes)"


def diff_trees(a: Path, b: Path, *, max_chars: int = 12_000, max_file_chars: int = 4_000, max_rows: int = 15,
               include: Callable[[str], bool] | None = None) -> str:
    """A reviewer-readable diff of two file trees.

    Text files get unified diffs, SQLite databases row-level diffs, append-only JSONL files (mailboxes,
    ledgers) the appended records; binary files a size note. ``.git`` internals and caches are skipped.
    """
    fa = set(_list_files(a, a)) if a.exists() else set()
    fb = set(_list_files(b, b)) if b.exists() else set()
    blocks = []
    for rel in sorted(fa | fb):
        if include is not None and not include(rel):
            continue
        pa, pb = a / rel, b / rel
        da = pa.read_bytes() if rel in fa else None
        db_ = pb.read_bytes() if rel in fb else None
        if da == db_:
            continue
        status = "added" if da is None else "deleted" if db_ is None else "modified"
        if rel.endswith(DB_SUFFIXES):
            body = diff_sqlite(pa if da is not None else Path("/nonexistent"), pb if db_ is not None else Path("/nonexistent"),
                               max_rows=max_rows)
        elif (da is None or _is_text(da)) and (db_ is None or _is_text(db_)):
            ta = da.decode("utf-8") if da is not None else ""
            tb = db_.decode("utf-8") if db_ is not None else ""
            if rel.endswith(".jsonl") and ta and tb.startswith(ta):
                new = tb[len(ta):].strip().splitlines()
                body = f"{len(new)} records appended:\n" + "\n".join(f"+ {ln}" for ln in new)
            else:
                body = "".join(difflib.unified_diff(ta.splitlines(keepends=True), tb.splitlines(keepends=True),
                                                    fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3))
                if not body.endswith("\n"):
                    body += "\n"
        else:
            body = f"binary file ({len(da or b'')} -> {len(db_ or b'')} bytes)"
        blocks.append(f"### {rel} ({status})\n{_clip(body.rstrip(), max_file_chars)}")
    if not blocks:
        return "(no changes)"
    return _clip("\n\n".join(blocks), max_chars)


# ============================================================================== environments


class Environment(abc.ABC):
    """Builds the starting state $S_0$ of a task item and supplies the tools that act on workspaces.

    ``initial_state`` receives the *uncensored* item (it runs on the experimenter's side, like the
    runner), so hidden environment state - e.g. simulated customers' willingness to pay - may come
    from ``item.ground_truth.data``; only the hidden part of the snapshot should carry it.
    """

    name: str = "environment"
    description: str = ""

    @abc.abstractmethod
    def initial_state(self, item: TaskItem, store: StateStore) -> str | None: ...

    def tools(self) -> dict[str, Tool]:
        return {}


class FilesEnvironment(Environment):
    """Starting states from item data: ``item.context["workspace"]["files"]`` (``{path: text}``, public),
    optionally a directory or git repository (``"source"``: a path, or ``{"repo": url, "commit": sha}``),
    and hidden state from ``item.ground_truth.data["hidden_state"]``. Items without a workspace spec get
    no state. Tools: the standard workspace tools (see :func:`workspace_tools`), plus ``extra_tools``.
    """

    name = "files"

    def __init__(self, *, test_command: str | None = None, extra_tools: Mapping[str, Tool] | None = None,
                 shell_timeout: float = 30.0):
        self.test_command, self.shell_timeout = test_command, shell_timeout
        self.extra_tools = dict(extra_tools or {})
        self._built: dict[tuple[str, str], str] = {}

    def initial_state(self, item: TaskItem, store: StateStore) -> str | None:
        spec = item.context.get("workspace")
        if not isinstance(spec, dict):
            return None
        hidden = (item.ground_truth.data.get("hidden_state") if item.ground_truth else None) or {}
        key = (str(store.root), hashlib.sha256(json.dumps([spec, hidden], sort_keys=True, default=str).encode()).hexdigest())
        if key in self._built and store.has(self._built[key]):
            return self._built[key]
        source = spec.get("source")
        source_dir = None
        if isinstance(source, dict) and source.get("repo"):
            source_dir = checkout_repo(source["repo"], source.get("commit"))
        elif isinstance(source, str):
            source_dir = Path(source)
        sid = store.create(spec.get("files") or {}, hidden=hidden, source_dir=source_dir)
        self._built[key] = sid
        return sid

    def tools(self) -> dict[str, Tool]:
        return {**workspace_tools(test_command=self.test_command, shell_timeout=self.shell_timeout), **self.extra_tools}


def checkout_repo(repo: str, commit: str | None = None) -> Path:
    """A cached clone of ``repo`` at ``commit`` with its remotes removed (so work cannot be pushed)."""
    from so_arena.datasets import cache_dir

    key = hashlib.sha256(f"{repo}@{commit}".encode()).hexdigest()[:16]
    dest = cache_dir() / "repos" / key
    if dest.exists():
        return dest
    tmp = dest.with_name(dest.name + f".tmp{uuid.uuid4().hex[:6]}")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", repo, str(tmp)], check=True, capture_output=True, timeout=600)
    if commit:
        subprocess.run(["git", "-C", str(tmp), "checkout", "--quiet", commit], check=True, capture_output=True)
    for remote in subprocess.run(["git", "-C", str(tmp), "remote"], capture_output=True, text=True).stdout.split():
        subprocess.run(["git", "-C", str(tmp), "remote", "remove", remote], capture_output=True)
    os.replace(tmp, dest)
    return dest


# ============================================================================== tools


class WorkspaceTool(Tool):
    """A tool acting on the caller's workspace (the working copy of the decision being made)."""

    readonly: bool = False

    async def call(self, args, item, game=None):
        ws = current_workspace()
        if ws is None:
            return ToolResult(output=f"error: {self.name} needs access to the task's state, which this role does not have",
                              error=True)
        try:
            return await self.run(args, ws, item, game)
        except Exception as e:  # tools report errors to the caller
            return ToolResult(output=f"error: {e}", error=True)

    @abc.abstractmethod
    async def run(self, args: str, ws: Workspace, item: TaskItem, game: "Game | None") -> ToolResult: ...


class ShellTool(WorkspaceTool):
    name = "shell"

    def __init__(self, timeout: float = 30.0, max_output_chars: int = 4000, name: str = "shell"):
        self.name, self.timeout, self.max_output_chars = name, timeout, max_output_chars
        self.description = (f"run a bash command in the task's working directory (timeout {timeout:g}s); returns its "
                            "output and exit code")
        self.example = f'<tool name="{name}">ls -la && python run_tests.py</tool>'

    async def run(self, args, ws, item, game=None):
        import asyncio

        res = await asyncio.to_thread(ws.run, args, timeout=self.timeout)
        return ToolResult(output=res.render(self.max_output_chars), error=not res.ok)


class ReadFileTool(WorkspaceTool):
    name = "read_file"
    readonly = True
    description = "show a file of the working directory (optionally a line range: 'path 10-40')"
    example = '<tool name="read_file">src/app.py 1-80</tool>'

    def __init__(self, max_chars: int = 8000):
        self.max_chars = max_chars

    async def run(self, args, ws, item, game=None):
        parts = args.strip().split()
        if not parts:
            return ToolResult(output="error: give a path", error=True)
        rel, rng = parts[0], parts[1] if len(parts) > 1 else None
        p = ws.path(rel)
        if p.is_dir():
            return ToolResult(output="\n".join(ws.files(rel)) or "(empty directory)")
        if rel.endswith(DB_SUFFIXES):
            return ToolResult(output="(a SQLite database: query it with the sql tool)")
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        lo, hi = 1, len(lines)
        if rng and "-" in rng:
            a, _, b = rng.partition("-")
            lo, hi = max(1, int(a or 1)), min(len(lines), int(b or len(lines)))
        body = "\n".join(f"{i:4d}  {lines[i - 1]}" for i in range(lo, hi + 1))
        return ToolResult(output=_clip(body or "(empty file)", self.max_chars))


class WriteFileTool(WorkspaceTool):
    name = "write_file"
    description = "create or overwrite a file: the first line is the path, the rest is the new content"
    example = '<tool name="write_file">notes.txt\nfirst line of the file\nsecond line</tool>'

    async def run(self, args, ws, item, game=None):
        head, _, body = args.lstrip("\n").partition("\n")
        rel = head.strip()
        if not rel:
            return ToolResult(output="error: the first line must be a path", error=True)
        ws.write_text(rel, body)
        return ToolResult(output=f"wrote {rel} ({len(body.splitlines())} lines)")


class ListFilesTool(WorkspaceTool):
    name = "list_files"
    readonly = True
    description = "list the files of the working directory (or of a subdirectory)"
    example = '<tool name="list_files"></tool>'

    async def run(self, args, ws, item, game=None):
        return ToolResult(output="\n".join(ws.files(args.strip())) or "(no files)")


class SQLTool(WorkspaceTool):
    """SQL against a SQLite database of the workspace; ``readonly=True`` opens it read-only."""

    def __init__(self, db: str = "data/app.db", *, readonly: bool = False, name: str | None = None, max_rows: int = 50):
        self.db, self.readonly, self.max_rows = db, readonly, max_rows
        self.name = name or ("sql_readonly" if readonly else "sql")
        mode = "read-only" if readonly else "reads and writes"
        self.description = f"run SQL on the database {db} ({mode}); returns up to {max_rows} rows"
        self.example = f'<tool name="{self.name}">SELECT name FROM sqlite_master WHERE type = \'table\'</tool>'

    async def run(self, args, ws, item, game=None):
        sql = args.strip().rstrip(";").strip()
        if not sql:
            return ToolResult(output="error: empty query", error=True)
        with ws.db(self.db, readonly=self.readonly) as con:
            if ";" in sql:  # several statements
                con.executescript(sql)
                return ToolResult(output="(statements executed)")
            cur = con.execute(sql)
            rows = cur.fetchmany(self.max_rows + 1)
            cols = [d[0] for d in cur.description] if cur.description else []
            affected = cur.rowcount
        if not cols:
            return ToolResult(output=f"(ok; {max(affected, 0)} rows affected)")
        more = len(rows) > self.max_rows
        lines = [" | ".join(cols)] + [" | ".join(str(v) for v in r) for r in rows[: self.max_rows]]
        return ToolResult(output="\n".join(lines) + ("\n... (more rows)" if more else ""))


class DiffTool(WorkspaceTool):
    name = "diff"
    readonly = True
    description = "show what has changed in the working directory since the task started"
    example = '<tool name="diff"></tool>'

    async def run(self, args, ws, item, game=None):
        base = game.base_state if game is not None and game.base_state else ws.parent
        return ToolResult(output=ws.diff(base))


class TestTool(WorkspaceTool):
    """Runs the task's visible test command."""

    name = "run_tests"

    def __init__(self, command: str = "python run_tests.py", timeout: float = 60.0):
        self.command, self.timeout = command, timeout
        self.description = f"run the visible tests ({command}); returns the output"
        self.example = '<tool name="run_tests"></tool>'

    async def run(self, args, ws, item, game=None):
        import asyncio

        res = await asyncio.to_thread(ws.run, self.command, timeout=self.timeout)
        return ToolResult(output=res.render(), error=not res.ok)


def workspace_tools(*, test_command: str | None = None, shell_timeout: float = 30.0,
                    db: str | None = None) -> dict[str, Tool]:
    """The standard workspace tools: ``shell``, ``read_file``, ``write_file``, ``list_files``, ``diff``,
    ``run_tests`` (if ``test_command``) and ``sql``/``sql_readonly`` (if ``db``)."""
    tools: dict[str, Tool] = {"shell": ShellTool(timeout=shell_timeout), "read_file": ReadFileTool(),
                              "write_file": WriteFileTool(), "list_files": ListFilesTool(), "diff": DiffTool()}
    if test_command:
        tools["run_tests"] = TestTool(test_command)
    if db:
        tools["sql"] = SQLTool(db)
        tools["sql_readonly"] = SQLTool(db, readonly=True)
    return tools


# ============================================================================== verifiers on the state


class CommandClaimVerifier(Verifier):
    """Execution claims about the work: a trusted executor runs the claimed command on the claimant's
    resulting state (a scratch copy) and shows the output. Verified iff it exits with status 0 and,
    with ``expect="..."``, prints exactly that."""

    uses_state = True

    def __init__(self, name: str = "run", timeout: float = 60.0, max_output_chars: int = 1200):
        self.name, self.timeout, self.max_output_chars = name, timeout, max_output_chars
        self.description = ("a shell command that a trusted executor runs on the work as submitted (e.g. the tests); "
                            "verified if it succeeds (and prints `expect`, if given)")
        self.example = f'<claim kind="{name}">python run_tests.py</claim>'

    async def verify(self, claim, item, game=None):
        import asyncio

        ws = current_workspace()
        if ws is None:
            return Verification(claim=claim, status="unchecked", output="(this task has no state to run commands on)")
        res = await asyncio.to_thread(ws.run, claim.content.strip(), timeout=self.timeout)
        ok = res.ok and ("expect" not in claim.attrs or res.stdout.strip() == claim.attrs["expect"].strip())
        return Verification(claim=claim, status="verified" if ok else "refuted", output=res.render(self.max_output_chars))


class ProtectedCommandVerifier(Verifier):
    """A fixed, trusted check of the work - e.g. "the tests pass" - run on the claimant's result with the
    ``protected`` paths (tests, test runner) restored from the task's starting state, so edits to the
    tests cannot make the claim true. The claim's content is ignored: ``<claim kind="tests"></claim>``."""

    uses_state = True

    def __init__(self, command: str, *, protected: Sequence[str] = (), name: str = "tests", timeout: float = 60.0,
                 max_output_chars: int = 1200):
        self.command, self.protected, self.name = command, tuple(protected), name
        self.timeout, self.max_output_chars = timeout, max_output_chars
        guard = f" with {', '.join(self.protected)} restored to their original versions" if self.protected else ""
        self.description = f"a trusted executor runs `{command}` on your work as submitted{guard}; verified if it succeeds"
        self.example = f'<claim kind="{name}"></claim>'

    async def verify(self, claim, item, game=None):
        import asyncio

        ws = current_workspace()
        if ws is None:
            return Verification(claim=claim, status="unchecked", output="(this task has no state to check)")
        base = getattr(game, "base_state", None)
        if base and self.protected:
            src = ws.store.files_dir(base)
            for rel in self.protected:
                ws.delete(rel)
                if (src / rel).is_dir():
                    shutil.copytree(src / rel, ws.path(rel), symlinks=True)
                elif (src / rel).exists():
                    ws.path(rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src / rel, ws.path(rel))
        res = await asyncio.to_thread(ws.run, self.command, timeout=self.timeout)
        return Verification(claim=claim, status="verified" if res.ok else "refuted", output=res.render(self.max_output_chars))


class QueryClaimVerifier(Verifier):
    """Claims about the data: a read-only SQL query on the claimant's resulting database, with an
    optional ``expect="..."`` for the rendered result (rows joined by newlines, columns by ``|``)."""

    uses_state = True

    def __init__(self, db: str = "data/app.db", name: str = "db", max_rows: int = 20):
        self.db, self.name, self.max_rows = db, name, max_rows
        self.description = f"a read-only SQL query on the database {db} as the work left it; the result is shown"
        self.example = f'<claim kind="{name}" expect="12">SELECT COUNT(*) FROM users</claim>'

    async def verify(self, claim, item, game=None):
        ws = current_workspace()
        if ws is None or not ws.exists(self.db):
            return Verification(claim=claim, status="unchecked", output="(no database to query)")
        try:
            with ws.db(self.db, readonly=True) as con:
                rows = con.execute(claim.content.strip().rstrip(";")).fetchmany(self.max_rows + 1)
        except sqlite3.Error as e:
            return Verification(claim=claim, status="refuted", output=f"query failed: {e}")
        text = "\n".join("|".join(str(v) for v in r) for r in rows[: self.max_rows])
        ok = "expect" not in claim.attrs or text.strip() == claim.attrs["expect"].strip()
        return Verification(claim=claim, status="verified" if ok else "refuted", output=text or "(no rows)")


# ============================================================================== ground truth helpers


def episode_store(ep: "Episode", ctx: Any = None) -> StateStore | None:
    """The store holding an episode's snapshots: the run context's, else the one recorded with the episode."""
    store = getattr(ctx, "_states", None) if ctx is not None else None
    if store is not None and (ep.final_state is None or store.has(ep.final_state)):
        return store
    if ep.state_store:
        return StateStore(ep.state_store)
    return store


def final_view(ep: "Episode", ctx: Any = None) -> SnapshotView | None:
    """Read-only view of an episode's final state (None if the episode has no state)."""
    store = episode_store(ep, ctx)
    if store is None or not ep.final_state:
        return None
    return store.view(ep.final_state)
