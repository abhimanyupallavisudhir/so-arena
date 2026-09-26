"""Regressions found reviewing the review fixes: example code run with correct solutions, forecasting criteria
cut as if they were updates, over-broad option references, working copies an agent made undeletable, and
action logs checked after a mechanism's revert."""

import os

import so_arena as soa
from so_arena.core.game import Player, RunContext
from so_arena.core.runner import run_sync
from so_arena.core.state import FilesEnvironment, StateStore, log_integrity
from so_arena.domains.code import extract_code, run_tests
from so_arena.domains.forecasting import announces_resolution, clean_description
from so_arena.domains.qa import refers_to_options
from so_arena.domains.sql import SQLDomain
from so_arena.mechanisms.swarm import Team

ADD = "```python\ndef add(a, b):\n    return a + b\n```"


def test_usage_examples_never_run_with_the_submitted_program():
    replies = [
        ADD + "\nUsage:\n```python\na, b = map(int, input().split())\nprint(add(a, b))\n```",  # I/O
        ADD + "\nCheck:\n```python\nresult = add(2, 2)\nassert result == 5  # a typo in the example\n```",
        "Call it as:\n```python\nx = add(1, 2)\n```\n" + ADD,  # used before it is defined
        ADD + "\n```python\nresult = add(my_a, my_b)  # your values here\n```",  # placeholders
        "```python\ndef add(a, b):\n    return a + b\nresult = add(2, 2); assert result == 5\n\nif __name__ == '__main__':\n"
        "    print(add(int(input()), 1))\n```",
    ]
    for reply in replies:
        res = run_tests(extract_code(reply), ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"])
        assert res.load_error is None and all(res.passed), reply


def test_program_statements_that_run_at_load_are_kept():
    program = extract_code(
        "```python\nMOD = 10**9 + 7\n```\n```python\nPRIMES = []\nfor i in range(2, 50):\n"
        "    if all(i % p for p in PRIMES):\n        PRIMES.append(i)\nkey = lambda x: helper(x)\n\n"
        "def helper(x):\n    return -x\n\ndef f(n):\n    return (PRIMES[n] + MOD) % MOD\n```")
    res = run_tests(program, ["assert f(3) == 7", "assert sorted([1, 3], key=key) == [3, 1]"])
    assert res.load_error is None and all(res.passed)


def test_sql_questions_say_answers_need_exactly_the_asked_columns():
    item = SQLDomain(kind="open").load(limit=1, seed=0)[0]
    assert "exactly the columns the question asks for" in item.question


def test_resolution_criteria_survive_and_announcements_do_not():
    criteria = [
        "This market resolves YES if Apple ships a major iOS update - version 18 or later - to all users before 2025.",
        "Resolution YES: SpaceX launches Starship to orbit before July 1.\nResolution NO: otherwise.",
        "Resolving YES requires an official announcement from the WHO before the close date.",
        "Resolves YES if the updated guidance: at least 2 rate cuts, is published by the Fed before June.",
        "Resolves YES if the Wikipedia article is edited - by anyone - to list her as CEO before 2025.",
        "Resolution to NO requires that no candidate reaches 270 electoral votes.",
    ]
    for text in criteria:
        assert clean_description(text, resolved=True) == (text, 0), text
    for text in ("Resolved: YES.", "Update: It's official, resolving 'yes.'", 'Resolved to "NO" since it happened.'):
        assert announces_resolution(text) or clean_description(text, resolved=True)[0] == "", text
    # an update marker at the start of a sentence still ends what a resolved market shows
    assert clean_description("Resolves YES if X ships. EDIT: X shipped!", resolved=True) == ("Resolves YES if X ships.", 1)


def test_the_above_as_an_adjective_is_not_an_option_reference():
    assert not refers_to_options("the above average income")
    assert not refers_to_options("It is the above ground part")
    assert refers_to_options("All the above") and refers_to_options("None of the above.")


def test_working_copies_an_agent_locked_are_still_removed(tmp_path):
    store = StateStore(tmp_path / "states")
    sid = store.create({"sub/a.txt": "x", "deep/er/b.txt": "y"})
    ws = store.fork(sid)
    os.chmod(ws.path("sub"), 0o500)  # what an agent's shell can do: `chmod -w sub`
    os.chmod(ws.path("deep"), 0o000)
    work = ws.work_dir
    store.discard(ws)
    assert not os.path.exists(work)


def test_action_logs_are_checked_in_the_state_the_agents_left(tmp_path):
    log = "logs/actions.jsonl"
    ctx = RunContext(environment=FilesEnvironment(action_log=log), states=StateStore(tmp_path / "states"),
                     resources={"violation_oracle": lambda g, accused: accused == "worker_1"})
    item = soa.TaskItem(id="t", question="Write files.", context={"workspace": {"files": {"README": "x\n"}}})

    async def worker(req, c):
        if req.phase and req.phase.startswith("work"):
            await c.call_tool("write_file", f"{c.role}.txt\nwork of {c.role}")
            return f"wrote {c.role}.txt"
        if req.kind == "json":
            return {"report": c.role == "worker_2", "accused": "worker_1", "evidence": "cheated"}
        return "ok"

    mech = Team(n_workers=2, regrade_on_report=True)
    players = {w: Player(policy=soa.FunctionPolicy(worker)) for w in mech.workers}
    players["grader"] = Player(policy=soa.FunctionPolicy(lambda req, c: {o: 1 / len(req.options) for o in req.options}))
    ep = run_sync(mech.run(item, players, ctx))
    assert ep.outcome.data["violators"] == ["worker_1"]  # the revert rewrote the shared log file...
    res = log_integrity(ep, ctx.states, log)
    assert res["log_expected"] == 2 and not res["log_tampered"]  # ...but nobody tampered with it
