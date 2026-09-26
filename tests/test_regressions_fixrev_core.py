"""Regressions from the fix review (core): information sets that split on hidden environment state or on
the number of hidden turns, and sandbox gaps (run directories, a writable root under the unshare backend)."""

import asyncio

import pytest

import so_arena as soa
from so_arena.analysis.optimization import BestOfN, evaluate_tree
from so_arena.core.game import Player, RunContext
from so_arena.core.mechanism import Mechanism, Outcome, RoleSpec
from so_arena.core.policy import FunctionPolicy
from so_arena.core.rewards import RewardRule
from so_arena.core.state import FilesEnvironment, StateStore, WorkspaceTool
from so_arena.core.tools import ToolResult
from so_arena.samplers.pools import expand_tree

# ------------------------------------------------------------------------------ hidden state and infosets


class Book(WorkspaceTool):
    name = "book"
    description = "record a booking (in the environment's hidden records)"

    async def run(self, args, ws, item, game=None):
        ws.hidden["booked"] = args.strip()
        return ToolResult(output="ok")


class Peek(WorkspaceTool):
    name = "peek"
    readonly = True
    description = "read the booking (a tool that reveals the hidden records)"

    async def run(self, args, ws, item, game=None):
        return ToolResult(output=str(ws.hidden.get("booked")))


class Stamp(WorkspaceTool):
    name = "stamp"
    description = "stamp the booking: writes a visible receipt and a hidden record that depends on the booking"

    async def run(self, args, ws, item, game=None):
        ws.hidden["stamped"] = f"{ws.hidden.get('booked')}-stamped"
        ws.write_text("receipt.txt", "stamped\n")
        return ToolResult(output="stamped")


class Hidden(Mechanism):
    name = "hidden_state_coin"

    def roles(self):
        return {"mover": RoleSpec(name="mover", trainable=False, state_access="write"),
                "observer": RoleSpec(name="observer", state_access="read")}

    async def protocol(self, g):
        await g.act("mover", kind="text", phase="move", prompt="move", visible_to=["mover"])
        await g.act("observer", kind="text", phase="guess", prompt="guess")
        return Outcome(decision="x")


class Match(RewardRule):
    def compute(self, ep):
        t = {t.role: t.text.strip() for t in ep.turns}
        return {"observer": float(t["mover"] == t["observer"])}


async def _mover(req, c):
    x = "HT"[c.sample_index % 2]
    await c.call_tool("book", x)
    return x


def _tree(tmp_path, observer, tools=None, **kw):
    item = soa.TaskItem(id="t", question="q", context={"workspace": {"files": {"a.txt": "same\n"}}})
    env = FilesEnvironment(extra_tools={"book": Book(), "peek": Peek(), "stamp": Stamp()})
    ctx = RunContext(environment=env, states=StateStore(tmp_path / "states"))
    players = {"mover": Player(policy=FunctionPolicy(_mover)), "observer": Player(policy=FunctionPolicy(observer))}
    mech = Hidden(reward=Match(), tools={"mover": ["book"], **(tools or {})}, **kw)
    return asyncio.run(expand_tree(mech, item, players, pool_sizes={"mover": 2, "observer": 2}, ctx=ctx,
                                   keep_episodes=True)) + (ctx,)


def test_hidden_state_does_not_split_an_observers_information_set(tmp_path):
    tree, _, _ = _tree(tmp_path, lambda req, c: "HT"[c.sample_index % 2])
    assert len({n.key for n in tree.nodes.values() if n.role == "observer"}) == 1  # was 2: the key hashed hidden
    assert evaluate_tree(tree, {"observer": BestOfN(2)}).rewards["observer"] == pytest.approx(0.5)  # was 1.0


def test_a_tool_that_reveals_hidden_state_lets_the_observer_tell_the_nodes_apart(tmp_path):
    async def peeker(req, c):  # reads the booking through a tool, then guesses it
        return await c.call_tool("peek", "")

    tree, _, _ = _tree(tmp_path, peeker, tools={"observer": ["peek"]})
    assert len({n.key for n in tree.nodes.values() if n.role == "observer"}) == 2  # its observations differ
    assert evaluate_tree(tree, {"observer": BestOfN(2)}).rewards["observer"] == pytest.approx(1.0)


def test_a_shared_candidate_has_each_nodes_own_effect_on_hidden_state(tmp_path):
    async def stamper(req, c):  # same action and observations in both branches
        await c.call_tool("stamp", "")
        return "H"

    # write access for the observer here: its receipt is kept, and the hidden stamp must follow each branch
    tree, eps, ctx = _tree(tmp_path, stamper, tools={"observer": ["stamp"]}, state_access={"observer": "write"})
    assert len({n.key for n in tree.nodes.values() if n.role == "observer"}) == 1
    assert len(eps) == 4
    for ep in eps:
        booked = [t.text for t in ep.turns if t.role == "mover"][0]
        view = ctx.states.view(ep.final_state)
        assert view.read_text("receipt.txt") == "stamped\n"
        assert view.hidden["stamped"] == f"{booked}-stamped"  # not the branch that happened to sample the pool


class Count(Mechanism):
    name = "count"

    def roles(self):
        return {"hider": RoleSpec(name="hider", trainable=False), "guesser": RoleSpec(name="guesser")}

    async def protocol(self, g):
        h = await g.act("hider", kind="text", phase="hide", prompt="hide", visible_to=["hider"])
        if h.text == "H":  # a private follow-up turn in one branch only
            await g.act("hider", kind="text", phase="extra", prompt="extra", visible_to=["hider"])
        await g.act("hider", kind="text", phase="announce", prompt="announce")
        await g.act("guesser", kind="text", phase="guess", prompt="guess")
        return Outcome(decision="x")


class Guess(RewardRule):
    def compute(self, ep):
        h = [t.text for t in ep.turns if t.phase == "hide"][0]
        return {"guesser": float([t.text for t in ep.turns if t.role == "guesser"][0] == h)}


def test_the_number_of_hidden_turns_does_not_split_an_information_set():
    hider = FunctionPolicy(lambda req, c: {"hide": "HT"[c.sample_index % 2], "extra": "ok", "announce": "done"}[req.phase])
    seen = []

    def guesser(req, c):
        seen.append([t.index for t in req.view.transcript])
        return "HT"[c.sample_index % 2]

    players = {"hider": Player(policy=hider), "guesser": Player(policy=FunctionPolicy(guesser))}
    item = soa.TaskItem(id="c0", question="guess")
    tree, _ = asyncio.run(expand_tree(Count(reward=Guess()), item, players, pool_sizes={"hider": 2, "guesser": 2},
                                      ctx=RunContext()))
    assert len({n.key for n in tree.nodes.values() if n.role == "guesser"}) == 1  # was 2: indices counted hidden turns
    assert evaluate_tree(tree, {"guesser": BestOfN(2)}).rewards["guesser"] == pytest.approx(0.5)
    assert all(ix == [0] for ix in seen)  # turns are numbered among those the viewer sees


# ------------------------------------------------------------------------------ sandbox


def _outside_dir():
    """A writable directory outside the home and temporary directories (as /code or /workspace in containers)."""
    import os
    import tempfile

    for base in ("/code", "/workspace", "/srv", "/data", "/mnt"):
        if os.path.isdir(base) and os.access(base, os.W_OK):
            return tempfile.mkdtemp(prefix="so_arena_outside_", dir=base)
    return None


def test_run_directories_and_the_working_directory_are_hidden(tmp_path, monkeypatch):
    from so_arena.core import sandbox
    from so_arena.core.store import RunStore

    store = RunStore(tmp_path / "runs" / "x")
    assert str((tmp_path / "runs" / "x").resolve()) in sandbox._HIDDEN  # stores register themselves
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    ops = dict(sandbox._plan(str(tmp_path / "work"), ()))
    assert ops[str(project.resolve())] == "cover"  # the experimenter's working directory (runs, configs, data)
    del store


@pytest.mark.skipif(not __import__("so_arena.core.sandbox", fromlist=["x"]).available(), reason="no sandbox backend")
def test_the_sandbox_makes_everything_but_its_working_directory_read_only(tmp_path):
    import subprocess

    from so_arena.core import sandbox
    from so_arena.core.store import RunStore

    work = tmp_path / "work"
    work.mkdir()
    r = subprocess.run(sandbox.wrap(["bash", "-c", "awk '$5 == \"/\" {print $6}' /proc/self/mountinfo; echo x > ok.txt"],
                                    work), cwd=work, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert r.stdout.split(",")[0].strip() == "ro" and (work / "ok.txt").read_text() == "x\n"
    outside = _outside_dir()
    if outside is None:
        pytest.skip("no writable directory outside home and /tmp to probe")
    import os
    import shutil

    try:
        runs = RunStore(f"{outside}/runs")
        (runs.path / "episodes.jsonl").write_text('{"ground_truth": {"correct": "B"}}\n')
        cmd = f"cat {runs.path}/episodes.jsonl; echo tamper >> {runs.path}/episodes.jsonl; touch {outside}/planted"
        r = subprocess.run(sandbox.wrap(["bash", "-c", cmd], work), cwd=work, capture_output=True, text=True,
                           stdin=subprocess.DEVNULL)
        assert '"correct"' not in r.stdout  # an earlier episode's ground truth is out of reach ...
        assert (runs.path / "episodes.jsonl").read_text() == '{"ground_truth": {"correct": "B"}}\n'  # ... unchanged
        assert not os.path.exists(f"{outside}/planted")  # nothing outside the working copy can be written
    finally:
        shutil.rmtree(outside, ignore_errors=True)


# ------------------------------------------------------------------------------ whistleblower theory


def test_the_analytic_model_charges_unverified_reporters_like_the_reward_rule():
    from so_arena.theory.whistleblower import payoff

    R, delta, s, q, c = 1.0, 0.5, 0.1, 0.6, 0.4
    for rho in (0.3, 0.7, 1.0):
        rb = 1 - rho
        for k in range(4):
            gain = payoff("report", k, R=R, delta=delta, s=s, rho=rho, q=q, c=c) - payoff(
                "silent", k, R=R, delta=delta, s=s, rho=rho, q=q, c=c)
            # an unverified accusation pays c as silence does: the gain is rho rb^k [(delta + c) q - delta] + s rho
            assert gain == pytest.approx(rho * rb ** k * ((delta + c) * q - delta) + s * rho)
            assert gain > 0  # s + (delta + c) q > delta: reporting is dominant for every rho > 0
