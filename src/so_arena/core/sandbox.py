"""Filesystem isolation for agent-controlled commands and untrusted code.

The ground truth an experiment grades agents against lives on the machine the agents act on: the hidden
environment state of every snapshot (``hidden.json`` in each :class:`~so_arena.core.state.StateStore`),
hidden tests in the bundled samples and the dataset cache, run directories with their items. A process of
the same user can read all of it - a shell tool's ``cat``, a candidate program's ``open`` - and an agent
that reads its answer key makes the experiment meaningless. So every agent-controlled command (workspace
shells and test runs, candidate programs, executable claims) runs through :func:`wrap`, in a private view
of the file system:

* ``/tmp``, ``/var/tmp``, ``/dev/shm`` and the user's home are empty, fresh scratch space - except the
  command's own working directory, which stays at its path and writable;
* nothing else can be changed: the rest of the file system is read-only;
* the interpreter's installation (``sys.prefix``, ``sys.base_prefix``) is visible, so Python runs, but
  the ``so_arena`` package (its bundled samples), the dataset cache (``$SO_ARENA_DATA``), the response
  cache, the experimenter's working directory (where runs, configs and data usually live) and every
  directory registered with :func:`hide` - state stores, work roots and run directories register
  themselves - are not;
* host processes are invisible (a private process namespace: ``/proc/<pid>/root`` of the parent cannot
  reach the unsandboxed view), there is no network by default, and the command has no capabilities, so it
  cannot unmount what hides the rest.

Backends: bubblewrap (``bwrap``) when it is installed, else util-linux ``unshare`` with unprivileged user,
mount, process and network namespaces plus ``setpriv``. Where neither works (macOS, containers without
user namespaces), agent code would see everything, so it is refused unless the experimenter opts in with
``SO_ARENA_ALLOW_UNSANDBOXED=1`` or :func:`allow_unsandboxed` (results may then be contaminated).
``SO_ARENA_SANDBOX`` chooses a backend (``bwrap``, ``unshare``; ``off`` disables sandboxing).

This is filesystem isolation for honest experiments with capable agents, not a security boundary against
a determined attacker (same kernel; see the design notes on container-backed environments).
"""

from __future__ import annotations

import functools
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

log = logging.getLogger("so_arena")

ALLOW_ENV = "SO_ARENA_ALLOW_UNSANDBOXED"
BACKEND_ENV = "SO_ARENA_SANDBOX"
BACKENDS = ("bwrap", "unshare")

_HIDDEN: set[str] = set()
_LOCK = threading.Lock()
_ALLOW: bool | None = None
_WARNED = False


class SandboxUnavailable(RuntimeError):
    """Agent code would run without filesystem isolation, and the experimenter has not allowed it."""


def hide(path: str | os.PathLike[str]) -> None:
    """Keep a directory out of sandboxed commands' view (state stores and run directories call this)."""
    with _LOCK:
        _HIDDEN.add(os.path.realpath(path))


def unhide(path: str | os.PathLike[str]) -> None:
    with _LOCK:
        _HIDDEN.discard(os.path.realpath(path))


def allow_unsandboxed(allow: bool | None = True) -> None:
    """Run agent code without isolation where no sandbox is available (None: back to the environment
    variable). Agents can then read hidden state and hidden tests."""
    global _ALLOW
    _ALLOW = allow


def _allowed() -> bool:
    if _ALLOW is not None:
        return _ALLOW
    return os.environ.get(ALLOW_ENV, "").strip().lower() in ("1", "true", "yes")


def _existing_dir(p: str | os.PathLike[str]) -> str | None:
    try:
        r = os.path.realpath(p)
    except OSError:
        return None
    return r if os.path.isdir(r) else None


def _under(p: str, roots: Sequence[str]) -> bool:
    return any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots)


_PSEUDO_FS = ("proc|sysfs|cgroup2?|devpts|devtmpfs|mqueue|tracefs|debugfs|securityfs|selinuxfs|pstore|bpf|autofs|"
              "binfmt_misc|fusectl|hugetlbfs|configfs|efivarfs|rpc_pipefs|nsfs")
# system directories never hidden, even when they are the experimenter's working directory
_SYSTEM = ("/", "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/etc", "/dev", "/proc", "/sys", "/run", "/var")


def _plan(workdir: str, visible: Sequence[str]) -> list[tuple[str, str]]:
    """The view's mounts, in the order to apply them: ``(path, op)`` with ``op`` one of ``"scratch"`` (an
    empty, writable tmpfs), ``"cover"`` (an empty tmpfs over a secret), ``"ro"`` (the real directory,
    read-only) and ``"rw"`` (the working directory).

    Operations apply shallowest first, so nesting resolves: a project directory is covered, the virtual
    environment inside it re-exposed read-only, the ``so_arena`` package inside that covered again, and the
    working directory - whatever holds it - exposed last among its ancestors.
    """
    import so_arena
    from so_arena.config import settings
    from so_arena.datasets import cache_dir

    ops: dict[str, str] = {}
    for d in (tempfile.gettempdir(), "/tmp", "/var/tmp", "/dev/shm", f"/run/user/{os.getuid()}", Path.home()):
        if (r := _existing_dir(d)) and r not in _SYSTEM:
            ops[r] = "scratch"
    with _LOCK:
        registered = set(_HIDDEN)
    # secrets: the package (bundled samples with hidden tests), the dataset and response caches, every
    # registered store and run directory, and the experimenter's working directory (runs, configs, data)
    for d in (Path(so_arena.__file__).parent, cache_dir(), settings.cache_dir, os.getcwd(), *registered):
        if d is not None and (r := _existing_dir(d)) and r not in _SYSTEM and ops.get(r) != "scratch":
            ops[r] = "cover"
    for d in (sys.prefix, sys.base_prefix, sys.exec_prefix, os.path.dirname(os.path.realpath(sys.executable)),
              *visible):
        if r := _existing_dir(d):
            ops[r] = "ro"
    ops[workdir] = "rw"
    rank = {"scratch": 0, "cover": 1, "ro": 2, "rw": 3}
    return sorted(ops.items(), key=lambda kv: (kv[0].rstrip("/").count("/"), rank[kv[1]], kv[0]))


@functools.lru_cache(maxsize=None)
def _probe(kind: str) -> bool:
    d = tempfile.mkdtemp(prefix="so_arena_sbx_")
    try:
        argv = _wrap_with(kind, ["true"], d, network=False, visible=())
        return subprocess.run(argv, cwd=d, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def backend() -> str | None:
    """The sandbox backend this machine supports (``"bwrap"`` or ``"unshare"``), or None."""
    want = os.environ.get(BACKEND_ENV, "auto").strip().lower()
    if want in ("off", "none", "0"):
        return None
    for kind in BACKENDS if want in ("auto", "") else (want,):
        if kind in BACKENDS and shutil.which(kind) and (kind != "unshare" or shutil.which("setpriv")) and _probe(kind):
            return kind
    return None


def available() -> bool:
    return backend() is not None


def _wrap_with(kind: str, argv: Sequence[str], workdir: str, *, network: bool, visible: Sequence[str],
               chdir: str | None = None) -> list[str]:
    ops = _plan(workdir, visible)
    start = chdir or workdir
    if kind == "bwrap":
        out = ["bwrap", "--die-with-parent", "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
               "--unshare-cgroup-try", *(() if network else ("--unshare-net",)), "--ro-bind", "/", "/",
               "--dev", "/dev", "--proc", "/proc"]
        for d, op in ops:
            out += {"scratch": ["--tmpfs", d], "cover": ["--tmpfs", d], "ro": ["--ro-bind", d, d],
                    "rw": ["--bind", d, d]}[op]
        return [*out, "--chdir", start, "--", *argv]
    # unshare: a setup script (as root of a fresh user namespace) makes every existing mount read-only - the
    # root included, so nothing outside the scratch space and the working directory can be changed -, applies
    # the view's mounts, then drops every capability before running the command, so it cannot undo them.
    # Directories it exposes are opened first and bound from their descriptors: their paths may be covered
    # by then.
    keep = [d for d, op in ops if op in ("ro", "rw")]
    q = shlex.quote
    lines = ["set -e"]
    lines += [f"exec {10 + i}<{q(d)}" for i, d in enumerate(keep)]
    # (kernel pseudo-filesystems hold no files to change, so only real file systems are remounted)
    lines += ["for m in $(awk '{for (i = 7; i <= NF; i++) if ($i == \"-\") {t = $(i + 1); break} "
              f"if (t !~ /^({_PSEUDO_FS})$/) print $5}}' /proc/self/mountinfo | sort -u); do "
              'mount -o remount,bind,ro "$m" 2>/dev/null || [ "$m" != / ] || exit 97; done']
    for d, op in ops:
        if op == "scratch":
            lines.append(f"mount -t tmpfs -o mode=1777 tmpfs {q(d)}")
        elif op == "cover":
            lines.append(f"if [ -d {q(d)} ]; then mount -t tmpfs -o mode=755 tmpfs {q(d)}; fi")
        else:
            i = keep.index(d)
            lines += [f"mkdir -p {q(d)}", f"mount --no-canonicalize --bind /proc/self/fd/{10 + i} {q(d)}",
                      f"mount -o remount,bind,{op} {q(d)}"]
    lines += [f"exec {10 + i}<&-" for i in range(len(keep))]
    lines += [f"cd {q(start)}",
              'exec setpriv --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs -- "$@"']
    return ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--ipc", "--fork", "--kill-child", "--mount-proc",
            *(() if network else ("--net",)), "--", "bash", "--norc", "--noprofile", "-c", "\n".join(lines),
            "so-arena-sandbox", *argv]


def wrap(argv: Sequence[str], workdir: str | os.PathLike[str], *, network: bool = False,
         visible: Sequence[str | os.PathLike[str]] = (), chdir: str | os.PathLike[str] | None = None) -> list[str]:
    """``argv`` as a command that runs sandboxed, in ``workdir`` (visible at its own path, writable).

    ``visible`` lists further directories the command may read (e.g. a toolchain under the home
    directory); ``chdir`` is where it starts (default ``workdir``; e.g. a project among ``visible``).
    Raises :class:`SandboxUnavailable` where no backend works, unless unsandboxed runs are allowed -
    then ``argv`` is returned unchanged (with a warning, once).
    """
    global _WARNED
    wd = os.path.realpath(workdir)
    kind = backend()
    if kind is not None:
        return _wrap_with(kind, list(argv), wd, network=network, visible=[os.fspath(v) for v in visible],
                          chdir=os.path.realpath(chdir) if chdir is not None else None)
    if not _allowed():
        raise SandboxUnavailable(
            "no sandbox is available for agent code on this system (bubblewrap, or util-linux unshare with "
            "unprivileged user namespaces, plus setpriv): the agents could read hidden state and hidden tests. "
            f"Install bubblewrap or enable user namespaces, or set {ALLOW_ENV}=1 to run unsandboxed anyway.")
    if not _WARNED:
        _WARNED = True
        log.warning("running agent code WITHOUT a sandbox (%s): agents can read hidden state and hidden tests",
                    ALLOW_ENV)
    return list(argv)
