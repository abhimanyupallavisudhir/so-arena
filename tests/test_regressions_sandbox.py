"""Regressions from the trust review: agent-controlled commands and untrusted code could read the hidden
ground truth they are graded against (snapshots' hidden state, bundled and cached datasets with hidden
tests), because they ran with the experimenter's full view of the file system."""

import logging
from pathlib import Path

import pytest

import so_arena
from so_arena.core import sandbox
from so_arena.core.state import StateStore
from so_arena.core.verification import run_python
from so_arena.domains.code import _spawn

SECRET = "SECRET-ANSWER-7c1f"
MBPP = Path(so_arena.__file__).parent / "data" / "samples" / "mbpp_sample.jsonl"

needs_sandbox = pytest.mark.skipif(not sandbox.available(), reason="no sandbox backend on this system")


@pytest.fixture
def store(tmp_path):
    st = StateStore(tmp_path / "states")
    sid = st.create({"task.txt": "Fix the bug.\n"}, hidden={"answer": SECRET})
    yield st, sid
    st.close()


@needs_sandbox
def test_agent_shells_cannot_read_hidden_state_or_datasets(store):
    st, sid = store
    ws = st.fork(sid)
    hidden_files = " ".join(str(p) for p in st.root.rglob("hidden.json"))
    assert hidden_files and MBPP.exists()
    res = ws.run(f"cat task.txt; cat {hidden_files}; head -c 300 {MBPP}; ls -a /tmp; umount /tmp; cat {hidden_files}")
    assert "Fix the bug." in res.stdout  # its own working copy is there
    assert SECRET not in res.stdout + res.stderr  # the snapshot's hidden state is not
    assert '"task_id"' not in res.stdout  # nor the bundled samples with their hidden tests
    # legitimate work still runs: python, writing files in the working copy
    res = ws.run("python -c 'print(6 * 7)' > out.txt && cat out.txt")
    assert res.returncode == 0 and res.stdout.strip() == "42" and ws.read_text("out.txt").strip() == "42"
    # host processes are invisible (no /proc/<pid>/root back into the full view)
    res = ws.run("ls /proc | grep -c '^[0-9]'")
    assert int(res.stdout.strip()) < 10
    st.discard(ws)


@needs_sandbox
def test_untrusted_code_cannot_open_hidden_tests(store):
    st, _ = store
    target = next(st.root.rglob("hidden.json"))
    probe = (f"for p in [{str(MBPP)!r}, {str(target)!r}]:\n"
             "    try:\n        print(open(p).read()[:200])\n    except OSError as e:\n        print('blocked', type(e).__name__)\n")
    rc, out, _ = run_python(probe)  # an executable claim
    assert rc == 0 and out.count("blocked") == 2 and SECRET not in out
    rc, out, _ = _spawn("import sys\nexec(sys.stdin.read())", probe, 10)  # a candidate program of the code domain
    assert out.count("blocked") == 2 and SECRET not in out


def test_agent_code_is_refused_without_a_sandbox(store, monkeypatch, caplog):
    st, sid = store
    monkeypatch.setenv(sandbox.BACKEND_ENV, "off")
    monkeypatch.delenv(sandbox.ALLOW_ENV, raising=False)
    sandbox.allow_unsandboxed(None)
    ws = st.fork(sid)
    with pytest.raises(sandbox.SandboxUnavailable, match=sandbox.ALLOW_ENV):
        ws.run("cat task.txt")
    with pytest.raises(sandbox.SandboxUnavailable):
        run_python("print(1)")
    monkeypatch.setenv(sandbox.ALLOW_ENV, "1")  # the experimenter's explicit opt-in
    monkeypatch.setattr(sandbox, "_WARNED", False)
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        assert ws.run("cat task.txt").stdout == "Fix the bug.\n"
    assert "WITHOUT a sandbox" in caplog.text
    st.discard(ws)


def test_state_stores_and_work_roots_are_registered_as_hidden(store):
    st, sid = store
    ws = st.fork(sid)
    assert str(st.root) in sandbox._HIDDEN and str(st.work_root) in sandbox._HIDDEN
    st.discard(ws)
