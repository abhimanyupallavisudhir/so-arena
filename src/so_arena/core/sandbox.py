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
* the interpreter's installation (``sys.prefix``, ``sys.base_prefix``) is visible read-only, so Python
  runs, but the ``so_arena`` package (its bundled samples), the dataset cache (``$SO_ARENA_DATA``) and
  every directory registered with :func:`hide` (state stores, work roots, run directories) are not;
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


def _plan(workdir: str, visible: Sequence[str]) -> tuple[list[str], list[str], list[str], list[str]]:
    """(scratch dirs replaced by empty tmpfs, read-only exposures, secret dirs to cover, read-only extras)."""
    scratch = [d for d in (_existing_dir(tempfile.gettempdir()), _existing_dir("/tmp"), _existing_dir("/var/tmp"),
                           _existing_dir("/dev/shm"), _existing_dir(Path.home())) if d]
    scratch = sorted(set(scratch), key=len)
    scratch = [d for i, d in enumerate(scratch) if not _under(d, scratch[:i]) and d != "/"]
    ro = sorted({d for d in (_existing_dir(sys.prefix), _existing_dir(sys.base_prefix), _existing_dir(sys.exec_prefix),
                             _existing_dir(os.path.dirname(os.path.realpath(sys.executable)))) if d}, key=len)
    ro = [d for i, d in enumerate(ro) if not _under(d, ro[:i])]
    import so_arena
    from so_arena.datasets import cache_dir

    with _LOCK:
        registered = set(_HIDDEN)
    secrets = {d for d in (_existing_dir(Path(so_arena.__file__).parent), _existing_dir(cache_dir()),
                           *(_existing_dir(h) for h in registered)) if d}
    # only secrets that would otherwise be visible need covering (the working directory is re-exposed after
    # them, so a work root holding it is still covered: other working copies stay out of sight); a secret
    # must not contain the interpreter's installation, which would then be hidden too
    secrets_l = sorted((s for s in secrets if (not _under(s, scratch) or _under(s, ro)) and s != "/"
                        and not any(_under(r, [s]) for r in ro)), key=len)
    extras = sorted({d for d in (_existing_dir(v) for v in visible) if d}, key=len)
    return scratch, ro, secrets_l, extras


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
    scratch, ro, secrets, extras = _plan(workdir, visible)
    start = chdir or workdir
    if kind == "bwrap":
        out = ["bwrap", "--die-with-parent", "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
               "--unshare-cgroup-try", *(() if network else ("--unshare-net",)), "--ro-bind", "/", "/",
               "--dev", "/dev", "--proc", "/proc"]
        for d in scratch:
            out += ["--tmpfs", d]
        for d in ro:
            out += ["--ro-bind", d, d]
        for d in secrets:
            out += ["--tmpfs", d]
        out += ["--bind", workdir, workdir]
        for d in extras:
            out += ["--ro-bind", d, d]
        return [*out, "--chdir", start, "--", *argv]
    # unshare: a setup script (as root of a fresh user namespace) builds the view, then drops every
    # capability before running the command, so it cannot undo the mounts. Directories it re-exposes are
    # opened first and bound from their descriptors: their paths are covered by then.
    keep = [workdir, *ro, *extras]
    q = shlex.quote
    lines = ["set -e"]
    lines += [f"exec {10 + i}<{q(d)}" for i, d in enumerate(keep)]
    lines += [f"mount -t tmpfs -o mode=1777 tmpfs {q(d)}" for d in scratch]

    def expose(i: int, d: str, *, readonly: bool) -> list[str]:
        cmds = [f"mkdir -p {q(d)}", f"mount --no-canonicalize --bind /proc/self/fd/{10 + i} {q(d)}"]
        return cmds + ([f"mount -o remount,bind,ro {q(d)}"] if readonly else [])

    for d in ro:
        lines += expose(keep.index(d), d, readonly=True)
    lines += [f"if [ -d {q(d)} ]; then mount -t tmpfs -o mode=755 tmpfs {q(d)}; fi" for d in secrets]
    lines += expose(0, workdir, readonly=False)
    for d in extras:
        lines += expose(keep.index(d), d, readonly=True)
    lines += [f"exec {10 + i}<&-" for i in range(len(keep))]
    lines += [f"cd {q(start)}",
              'exec setpriv --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs -- "$@"']
    return ["unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child", "--mount-proc",
            *(() if network else ("--net",)), "--", "bash", "-c", "\n".join(lines), "so-arena-sandbox", *argv]


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
