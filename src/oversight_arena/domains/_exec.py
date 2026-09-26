"""Run model-written Python: confined single-process runs, and isolated grading.

- :func:`run_python` runs code in a subprocess. By default the process confines itself before the
  code runs (``_sandbox_lib.confine``): no reads outside the Python installation, system directories
  and ``allow_read``; no writes outside its own directory; no new processes, sockets, signals to
  other processes or exec; time and memory limits. Model code therefore cannot read the parent's
  environment, data caches (e.g. which implementation is the mutant) or hidden tests.
- :func:`run_isolated` is for *grading*: trusted harness code (tests, checks) runs in one process,
  the untrusted code in a separate confined process, and only plain data crosses between them.
  The verdict comes from the harness, tagged with a per-run nonce, so untrusted code cannot print
  a fake verdict, exit early, or return objects with a forged ``__eq__``.

The kernel layers (Landlock, seccomp) need Linux; elsewhere only the rlimits and a Python-level
audit hook apply. For adversarial workloads at scale, also run inside a container, or plug in your
own executor with :func:`set_executor`.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import _sandbox_lib as _lib

_LIB_SRC = Path(_lib.__file__).read_text()


@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str
    timeout: bool = False

    def short(self, n: int = 800) -> str:
        s = self.stdout.strip()
        if not self.ok:
            s = (s + "\n" + self.stderr.strip().splitlines()[-1] if self.stderr.strip() else s).strip()
        return s if len(s) <= n else s[: n - 3] + "..."


def _limits(mem_mb: int, cpu_s: int) -> Callable[[], None]:
    def f() -> None:
        try:
            import resource

            resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 2**20, mem_mb * 2**20))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 2**20, 16 * 2**20))  # bound output files
        except Exception:
            pass

    return f


_CHILD_ENV = {
    "PATH": "/usr/bin:/bin",
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "LANG": "C.UTF-8",
}
# opt-outs a caller may set; forwarded to the confined child (which otherwise gets a minimal env)
_PASSTHROUGH = ("OA_ALLOW_NO_KERNEL_SANDBOX",)


def _passthrough_env() -> dict:
    return {k: os.environ[k] for k in _PASSTHROUGH if k in os.environ}
_MAX_OUT = 1 << 20  # bytes of stdout/stderr kept (output goes to files, so the parent never blows up)


def _tail(path: str, n: int) -> str:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > n:
            f.seek(size - n)
        return f.read().decode("utf-8", errors="replace")


def _default_exec(code: str, timeout: float = 10.0, mem_mb: int = 384, *, sandbox: bool = True,
                  allow_read: Sequence[str] = (), stdin: str | None = None, require_kernel: bool = True) -> ExecResult:
    with tempfile.TemporaryDirectory(prefix="oa_exec_") as d:
        d = os.path.realpath(d)
        src = os.path.join(d, "main.py")
        out_p, err_p = os.path.join(d, "out"), os.path.join(d, "err")
        if sandbox:
            cfg = {"read": _lib.default_read_roots(allow_read), "write": [d], "mem_mb": mem_mb,
                   "cpu_s": int(timeout) + 1, "require_kernel": require_kernel}
            code = f"{_LIB_SRC}\nrun_confined({cfg!r}, {code!r})\n"
        with open(src, "w") as f:
            f.write(code)
        timed_out = False
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            p = subprocess.Popen(
                [sys.executable, "-I", "-B", src], stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
                stdout=fo, stderr=fe, cwd=d, env={**_CHILD_ENV, **_passthrough_env(), "HOME": d, "TMPDIR": d},
                start_new_session=True,
                preexec_fn=_limits(mem_mb, int(timeout) + 1) if os.name == "posix" else None,
            )
            try:
                p.communicate(stdin.encode() if stdin is not None else None, timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    p.kill()
                p.wait()
            rc = p.returncode
        out, err = _tail(out_p, _MAX_OUT), _tail(err_p, 20000)
        if timed_out:
            err = (err + "\ntimeout").strip()
        return ExecResult(rc == 0 and not timed_out, out, err, timed_out)


_EXECUTOR: Callable[..., ExecResult] = _default_exec


def set_executor(fn: Callable[..., ExecResult]) -> None:
    """Replace the executor (e.g. with a Docker/gVisor/E2B-backed one).

    Contract: ``fn(code, timeout=..., mem_mb=..., sandbox=..., allow_read=..., stdin=...)`` runs
    ``code`` with ``python -I`` and returns an :class:`ExecResult`. ``stdin`` (text or None) must
    be delivered to the process; isolated grading relies on it. An executor that provides its own
    isolation may ignore ``sandbox`` and ``allow_read``.
    """
    global _EXECUTOR
    _EXECUTOR = fn


def run_python(code: str, timeout: float = 10.0, mem_mb: int = 384, *, sandbox: bool = True,
               allow_read: Sequence[str] = (), stdin: str | None = None, require_kernel: bool = True) -> ExecResult:
    """Run code in a fresh process. ``sandbox=True`` (default) confines it first; ``allow_read``
    lists extra readable paths. Only trusted code may run with ``sandbox=False``. With
    ``require_kernel`` (default), a sandboxed run refuses to proceed on Linux if the kernel
    filesystem sandbox (Landlock) is unavailable, rather than relying on the audit hook alone."""
    kw = {"sandbox": sandbox, "allow_read": tuple(allow_read), "stdin": stdin}
    if _EXECUTOR is _default_exec:
        kw["require_kernel"] = require_kernel
    return _EXECUTOR(code, timeout=timeout, mem_mb=mem_mb, **kw)


def run_json(code: str, timeout: float = 10.0) -> tuple[bool, object]:
    """Run code that prints a single JSON value on its last stdout line."""
    r = run_python(code, timeout)
    if not r.ok:
        return False, r.stderr[-500:]
    try:
        return True, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return False, f"bad output: {e}"


@dataclass
class IsolatedResult:
    ok: bool
    value: Any = None
    error: str = ""


def run_isolated(harness: str, *, code: str | None = None, path: str | None = None, allow_read: Sequence[str] = (),
                 timeout: float = 20.0, mem_mb: int = 512) -> IsolatedResult:
    """Run trusted ``harness`` code against untrusted code in a separate confined process.

    ``harness`` must define ``main(u)`` returning a JSON-serialisable value; ``u`` is a
    :class:`~oversight_arena.domains._sandbox_lib.Untrusted` handle (``u.function(name)``,
    ``u.eval(expr)``, ``u.truth(expr)``, ``u.lookup(name)``, ``u.importer(package)``). ``code`` is
    loaded as the untrusted module ``candidate``; ``path`` is put on the untrusted side's
    ``sys.path`` (e.g. a project directory). The result counts only if the harness itself reports
    it, tagged with this run's nonce, and exits cleanly.
    """
    nonce = secrets.token_hex(16)
    budget = max(1.0, timeout - 2.0)
    paths = [os.path.realpath(path)] if path else []
    script = "\n".join([
        _LIB_SRC,
        f"_OA_LIB_SRC = {_LIB_SRC!r}",
        harness,
        f"harness_main({code!r}, {list(allow_read)!r}, {paths!r}, {budget!r}, {mem_mb!r}, main)",
    ])
    r = run_python(script, timeout=timeout, mem_mb=mem_mb + 256, sandbox=False, stdin=nonce + "\n")
    lines = r.stdout.rstrip("\n").splitlines()
    last = lines[-1] if lines else ""
    head = f"OA-RESULT {nonce} "
    if r.ok and last.startswith(head):
        try:
            return IsolatedResult(True, json.loads(last[len(head):]))
        except ValueError:
            return IsolatedResult(False, error="malformed harness result")
    if last.startswith("OA-ERROR "):
        try:
            return IsolatedResult(False, error=str(json.loads(last[len("OA-ERROR "):])))
        except ValueError:
            pass
    if r.timeout:
        return IsolatedResult(False, error="timeout")
    tail = r.stderr.strip().splitlines()
    return IsolatedResult(False, error=tail[-1] if tail else "harness failed")


def sandbox_layers() -> dict[str, bool]:
    """Which confinement layers are active on this machine (runs a tiny confined probe)."""
    probe = f"{_LIB_SRC}\nimport json\nprint(json.dumps(confine(default_read_roots(), ['.'], 256, 5)))\n"
    r = run_python(probe, timeout=20, sandbox=False)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return {"landlock": False, "seccomp": False, "rlimits": False, "audit": False}
