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
        except Exception:
            pass

    return f


def _default_exec(code: str, timeout: float = 10.0, mem_mb: int = 512) -> ExecResult:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run(
            [sys.executable, "-I", path], capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limits(mem_mb, int(timeout) + 1) if os.name == "posix" else None,
        )
        return ExecResult(p.returncode == 0, p.stdout[-20000:], p.stderr[-5000:])
    except subprocess.TimeoutExpired as e:
        return ExecResult(False, (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout", True)
    finally:
        os.unlink(path)


_EXECUTOR: Callable[..., ExecResult] = _default_exec


def set_executor(fn: Callable[..., ExecResult]) -> None:
    """Replace the executor (e.g. with a Docker/gVisor/E2B-backed one)."""
    global _EXECUTOR
    _EXECUTOR = fn


def run_python(code: str, timeout: float = 10.0, mem_mb: int = 512) -> ExecResult:
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
