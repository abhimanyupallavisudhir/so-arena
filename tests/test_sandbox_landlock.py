"""The Landlock layer of the sandbox: alone (containers without user namespaces) and under bubblewrap/unshare.

Every probe runs under each configuration this machine supports - ``landlock`` alone, and a namespace backend
with Landlock as a second layer - and is skipped where the kernel has no Landlock (``landlock_abi() == 0``).
The refusal tests run everywhere: without an isolating backend agent code never runs silently unsandboxed.
"""

import os
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import so_arena
from so_arena.core import _landlock, sandbox
from so_arena.core.state import StateStore
from so_arena.core.verification import run_python

SECRET = "SECRET-ANSWER-4b2e"
MBPP = Path(so_arena.__file__).parent / "data" / "samples" / "mbpp_sample.jsonl"
HAVE_LANDLOCK = sandbox.landlock_abi() >= 1 and _landlock.seccomp_supported()


def _configure(monkeypatch, mode: str) -> None:
    sandbox._probe.cache_clear()
    monkeypatch.setenv(sandbox.BACKEND_ENV, "landlock" if mode == "landlock" else "auto")
    monkeypatch.delenv(sandbox.LANDLOCK_ENV, raising=False)
    monkeypatch.delenv(sandbox.ALLOW_ENV, raising=False)
    sandbox.allow_unsandboxed(None)
    want = ["landlock"] if mode == "landlock" else None
    got = sandbox.layers()
    if (want is not None and got != want) or (want is None and (len(got) != 2 or got[1] != "landlock")):
        pytest.skip(f"{mode}: not available here (layers {got})")


@pytest.fixture(params=["landlock", "layered"])
def mode(request, monkeypatch):
    if not HAVE_LANDLOCK:
        pytest.skip("the kernel has no Landlock (or no seccomp filter for this machine)")
    _configure(monkeypatch, request.param)
    yield request.param
    sandbox._probe.cache_clear()


@pytest.fixture
def store(tmp_path):
    st = StateStore(tmp_path / "states")
    sid = st.create({"task.txt": "Fix the bug.\n"}, hidden={"answer": SECRET})
    yield st, sid
    st.close()


def sh(cmd: str, workdir: Path, **kw) -> subprocess.CompletedProcess:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(workdir), "TMPDIR": str(workdir)}
    return subprocess.run(sandbox.wrap(["bash", "-c", cmd], workdir, **kw), cwd=workdir, capture_output=True,
                          text=True, env=env, stdin=subprocess.DEVNULL, timeout=60)


# ----------------------------------------------------------------------------- probes

def test_hidden_state_datasets_and_outside_files_are_unreadable(mode, store, tmp_path, monkeypatch):
    st, sid = store
    project = tmp_path / "project"  # the experimenter's working directory: runs, configs, data
    project.mkdir()
    (project / "answers.json").write_text(SECRET)
    monkeypatch.chdir(project)
    ws = st.fork(sid)
    hidden = " ".join(str(p) for p in st.root.rglob("hidden.json"))
    res = ws.run(f"cat task.txt; cat {hidden}; head -c 300 {MBPP}; cat {project}/answers.json; ls {project}; "
                 f"ls {Path(so_arena.__file__).parent}")
    assert "Fix the bug." in res.stdout and SECRET not in res.stdout + res.stderr and '"task_id"' not in res.stdout
    res = ws.run("python -c 'print(6 * 7)' > out.txt && cat out.txt")  # legitimate work still runs
    assert res.returncode == 0 and res.stdout.strip() == "42" and ws.read_text("out.txt").strip() == "42"
    target = next(st.root.rglob("hidden.json"))
    probe = (f"for p in [{str(MBPP)!r}, {str(target)!r}, {str(project / 'answers.json')!r}]:\n"
             "    try:\n        print(open(p).read()[:200])\n    except OSError as e:\n        print('blocked', type(e).__name__)\n")
    rc, out, _ = run_python(probe)  # an executable claim
    assert rc == 0 and out.count("blocked") == 3 and SECRET not in out
    st.discard(ws)


def test_hidden_directory_inside_a_visible_one(mode, tmp_path):
    vis = tmp_path / "toolchain"
    (vis / "secret").mkdir(parents=True)
    (vis / "lib.txt").write_text("public")
    (vis / "secret" / "key.txt").write_text(SECRET)
    (vis / "link").symlink_to(vis / "secret")  # a way round the hole
    sandbox.hide(vis / "secret")
    work = tmp_path / "work"
    work.mkdir()
    try:
        r = sh(f"cat {vis}/lib.txt; cat {vis}/secret/key.txt; cat {vis}/link/key.txt; ls {vis}/secret; "
               f"echo x > {vis}/planted; echo ok > mine.txt", work, visible=[vis])
    finally:
        sandbox.unhide(vis / "secret")
    assert "public" in r.stdout and SECRET not in r.stdout + r.stderr
    assert not (vis / "planted").exists() and (work / "mine.txt").read_text() == "ok\n"


def test_environment_and_other_processes_stay_private(mode, store, monkeypatch):
    st, sid = store
    monkeypatch.setenv("SO_ARENA_TEST_API_KEY", SECRET)
    ws = st.fork(sid)
    me = os.getpid()
    res = ws.run(f"echo \"[$SO_ARENA_TEST_API_KEY]\"; env; cat /proc/{me}/environ; cat /proc/$PPID/environ; "
                 f"cat /proc/{me}/cmdline; ls /proc/{me}/cwd/")
    assert "[]" in res.stdout and SECRET not in res.stdout + res.stderr
    rc, out, _ = run_python(f"import os\ntry:\n    print(open('/proc/{me}/environ').read())\nexcept OSError as e:\n"
                            "    print('blocked', type(e).__name__)\nprint(sorted(os.environ))")
    assert rc == 0 and "blocked" in out and SECRET not in out
    st.discard(ws)


def test_no_network(mode, tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port = srv.getsockname()[1]
    got: list[bytes] = []

    def serve():
        srv.settimeout(15)
        try:
            while True:
                conn, _ = srv.accept()
                got.append(conn.recv(100))
                conn.close()
        except OSError:
            pass

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    code = (f"import socket\ntry:\n    s = socket.create_connection(('127.0.0.1', {port}), timeout=3)\n"
            "    s.sendall(b'LEAK')\n    print('connected')\nexcept OSError as e:\n    print('blocked', e)\n"
            "try:\n    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
            f"    u.sendto(b'LEAK', ('127.0.0.1', {port}))\n    print('sent udp')\nexcept OSError as e:\n    print('blocked', e)\n")
    work = tmp_path / "w"
    work.mkdir()
    (work / "probe.py").write_text(code)
    r = subprocess.run(sandbox.wrap([sys.executable, "-I", "probe.py"], work), cwd=work, capture_output=True,
                       text=True, timeout=60)
    assert "connected" not in r.stdout and r.stdout.count("blocked") >= 1, r.stdout + r.stderr
    r = subprocess.run(sandbox.wrap([sys.executable, "-I", "probe.py"], work, network=True), cwd=work,
                       capture_output=True, text=True, timeout=60)
    assert "connected" in r.stdout, r.stdout + r.stderr  # the knob: network access when the experimenter allows it
    srv.close()
    t.join(5)
    assert got == [b"LEAK"]  # only the permitted connection arrived


def test_writes_only_in_the_working_directory(mode, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    outside = tmp_path / "outside.txt"
    r = sh(f"echo x > {outside}; echo y > {Path(sys.executable).parent}/planted; mkdir sub && echo z > sub/f && "
           "mv sub/f g && cat g", work)
    assert r.stdout.strip() == "z" and not outside.exists()
    assert not (Path(sys.executable).parent / "planted").exists()


def test_second_layer_narrows_the_namespace_view(monkeypatch, tmp_path):
    """Under namespaces the rest of the file system is visible read-only; the Landlock layer also keeps what
    no command needs (here /var) out of reach, so a gap in the namespace view still meets the allow-list."""
    if not HAVE_LANDLOCK:
        pytest.skip("the kernel has no Landlock")
    _configure(monkeypatch, "layered")
    work = tmp_path / "w"
    work.mkdir()
    try:
        assert sh("ls /var >/dev/null && echo listed", work).stdout.strip() == ""
        monkeypatch.setenv(sandbox.LANDLOCK_ENV, "off")
        assert sandbox.layers() == [sandbox.backend()]
        assert sh("ls /var >/dev/null && echo listed", work).stdout.strip() == "listed"
    finally:
        sandbox._probe.cache_clear()


# ----------------------------------------------------------------------------- fail closed

def test_launcher_refuses_rather_than_running_unconfined(tmp_path):
    """Whatever the kernel: a launcher that cannot apply what it was asked to (here a Landlock ABI no kernel
    has) exits 126 without running the command - never a silent fallback (attempt #1's bug N1)."""
    import json

    cfg = {"read": ["/usr"], "write": [str(tmp_path)], "min_abi": 99, "seccomp": True}
    r = subprocess.run([sys.executable, "-I", "-S", "-c", sandbox._launcher_source(), json.dumps(cfg), "--",
                        "touch", str(tmp_path / "ran")], capture_output=True, text=True, timeout=30)
    assert r.returncode == _landlock.REFUSED and "refusing" in r.stderr and not (tmp_path / "ran").exists()


def test_no_isolating_backend_means_refusal(store, monkeypatch):
    st, sid = store
    sandbox._probe.cache_clear()
    monkeypatch.delenv(sandbox.ALLOW_ENV, raising=False)
    sandbox.allow_unsandboxed(None)
    monkeypatch.setattr(_landlock, "landlock_abi", lambda: 0)  # a kernel without Landlock...
    monkeypatch.setattr(sandbox, "_installed", lambda kind: False)  # ... and no bubblewrap or unshare
    try:
        assert sandbox.backend() is None and sandbox.layers() == []
        for want in ("auto", "landlock"):
            monkeypatch.setenv(sandbox.BACKEND_ENV, want)
            ws = st.fork(sid)
            with pytest.raises(sandbox.SandboxUnavailable, match=sandbox.ALLOW_ENV):
                ws.run("cat task.txt")
            with pytest.raises(sandbox.SandboxUnavailable):
                run_python("print(1)")
            st.discard(ws)
    finally:
        sandbox._probe.cache_clear()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")
def test_landlock_abi_matches_the_kernel():
    abi = sandbox.landlock_abi()
    lsm = Path("/sys/kernel/security/lsm")
    if lsm.exists() and os.access(lsm, os.R_OK):
        assert (abi >= 1) == ("landlock" in lsm.read_text().split(","))
    if abi and sandbox.backend() in ("bwrap", "unshare") and os.environ.get(sandbox.LANDLOCK_ENV) is None:
        assert sandbox.layers() == [sandbox.backend(), "landlock"]  # the second layer wherever the kernel has it


def test_hidden_directory_behind_a_symlinked_read_root(mode, tmp_path, monkeypatch):
    """A system read root that is a symlink (``/lib`` -> ``/usr/lib`` on merged-/usr systems) must not expose
    what is hidden inside its target: Landlock rules follow symlinks, so the holes are matched after resolving."""
    real = tmp_path / "real"
    (real / "secret").mkdir(parents=True)
    (real / "secret" / "gt.txt").write_text(SECRET)
    (real / "public.txt").write_text("hello")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(sandbox, "_LANDLOCK_SYSTEM", (*sandbox._LANDLOCK_SYSTEM, str(alias)))
    sandbox.hide(real / "secret")
    try:
        r = sh(f"cat {alias}/public.txt {real}/public.txt; cat {alias}/secret/gt.txt; cat {real}/secret/gt.txt", wd)
    finally:
        sandbox.unhide(real / "secret")
    if mode == "landlock":  # under namespaces the scratch /tmp hides tmp_path altogether
        assert "hello" in r.stdout, r.stderr
    assert SECRET not in r.stdout
