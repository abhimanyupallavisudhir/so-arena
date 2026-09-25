"""Run untrusted Python in a subprocess with time/memory limits.

NOTE: this is *not* a security sandbox (no filesystem/network isolation). For untrusted model
code at scale, run OversightArena inside a container or plug a proper sandbox in via
``set_executor``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass


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
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "LANG": "C.UTF-8",
}
_MAX_OUT = 1 << 20  # bytes of stdout/stderr kept (output goes to files, so the parent never blows up)


def _tail(path: str, n: int) -> str:
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if size > n:
            f.seek(size - n)
        return f.read().decode("utf-8", errors="replace")


def _default_exec(code: str, timeout: float = 10.0, mem_mb: int = 384) -> ExecResult:
    with tempfile.TemporaryDirectory(prefix="oa_exec_") as d:
        src = os.path.join(d, "main.py")
        out_p, err_p = os.path.join(d, "out"), os.path.join(d, "err")
        with open(src, "w") as f:
            f.write(code)
        timed_out = False
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            try:
                p = subprocess.run(
                    [sys.executable, "-I", src], stdout=fo, stderr=fe, timeout=timeout, cwd=d, env=_CHILD_ENV,
                    preexec_fn=_limits(mem_mb, int(timeout) + 1) if os.name == "posix" else None,
                )
                rc = p.returncode
            except subprocess.TimeoutExpired:
                rc, timed_out = -9, True
        out, err = _tail(out_p, _MAX_OUT), _tail(err_p, 20000)
        if timed_out:
            err = (err + "\ntimeout").strip()
        return ExecResult(rc == 0, out, err, timed_out)


_EXECUTOR: Callable[..., ExecResult] = _default_exec


def set_executor(fn: Callable[..., ExecResult]) -> None:
    """Replace the executor (e.g. with a Docker/gVisor/E2B-backed one)."""
    global _EXECUTOR
    _EXECUTOR = fn


def run_python(code: str, timeout: float = 10.0, mem_mb: int = 384) -> ExecResult:
    return _EXECUTOR(code, timeout=timeout, mem_mb=mem_mb)


def run_json(code: str, timeout: float = 10.0) -> tuple[bool, object]:
    """Run code that prints a single JSON value on its last stdout line."""
    r = run_python(code, timeout)
    if not r.ok:
        return False, r.stderr[-500:]
    try:
        return True, json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:
        return False, f"bad output: {e}"
