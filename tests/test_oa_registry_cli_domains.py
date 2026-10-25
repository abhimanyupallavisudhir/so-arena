import asyncio

import pytest
from click.testing import CliRunner

from oversight_arena.cli import main
from oversight_arena.config import experiment_from_config
from oversight_arena.registry import build


def test_registry_nested_build():
    m = build("mechanism", {"type": "debate", "rounds": 3, "reward": {"type": "judge_probability", "transform": "brier"},
                            "evidence": {"budget": 2}})
    assert m.rounds == 3 and m.reward.transform == "brier" and m.evidence.budget == 2


def test_config_experiment(tmp_path):
    cfg = {
        "name": "t", "out": str(tmp_path / "run"), "domain": {"type": "hidden_bits", "n_tasks": 4},
        "mechanisms": [{"type": "debate", "rounds": 1, "evidence": {"budget": 4}}, {"type": "propaganda"}],
        "agents": {"*": {"type": "bit_advocate"}, "kind:judge": {"type": "bayes_bit_judge", "trust": 0.6}},
        "profiles": {"type": "stances"},
    }
    exp = experiment_from_config(cfg)
    res = exp.run()
    assert len(res) == 4 * 2 * 2 and not res.errors
    # resume: nothing re-run
    res2 = experiment_from_config(cfg).run()
    assert {r.id for r in res2.records} == {r.id for r in res.records}


def test_cli_list_and_theory(tmp_path):
    r = CliRunner().invoke(main, ["list", "mechanism"])
    assert r.exit_code == 0 and "debate" in r.output
    r = CliRunner().invoke(main, ["theory", "swarm", "--bounty", "0.05"])
    assert r.exit_code == 0 and "all silent" in r.output


def test_code_utilities():
    from oversight_arena.domains.code import doc_examples, mutants, passes_hidden, passes_visible

    src = "def f(xs):\n    return sum(x for x in xs if x > 0)\n"
    ms = mutants(src)
    assert ms and all(m[1] != src for m in ms)
    assert passes_visible(src, [("f([1, -2, 3])", "4")])
    assert passes_hidden(src, "def check(c):\n    assert c([1, 2]) == 3\n", "f")
    assert not passes_hidden("def f(xs):\n    return 0\n", "def check(c):\n    assert c([1]) == 1\n", "f")
    assert doc_examples('    >>> f([1])\n    1\n') == [("f([1])", "1")]


def test_simops_environment():
    from oversight_arena.domains.swarm import SimOps

    dom = SimOps(n_tasks=1)
    task = dom.tasks()[0]
    env = dom.make_env(task)
    env.workers = ["worker_1", "worker_2", "worker_3"]
    tools = {t.name: t for t in env.tools_for("worker_1", set())}
    assert "list_files" in tools and "write_file" in tools
    before = env.apparent_score()
    asyncio.run(tools["write_file"].run({"path": "tests/test_visible.py", "content": "def test_ok():\n    assert True\n"}))
    assert env.apparent_score() == 1.0 > before
    assert env.clean_score() < 1.0
    assert "worker_1" in env.violators() and env.verify_accusation("worker_1", "")
    asyncio.run(env.teardown())


def test_lean_perturbation():
    from oversight_arena.domains.lean import perturb_statement

    stmt = "theorem t\n  (a : ℕ)\n  (h₀ : a = 2)\n  (h₁ : 0 < a) :\n  a + 1 = 3 := sorry"
    outs = {perturb_statement(stmt, s) for s in range(20)}
    assert all(o is None or o[0] != stmt for o in outs)
    assert any(o and o[1] == "constant changed" for o in outs)


@pytest.mark.stockfish
def test_chess_line_verifier():
    pytest.importorskip("chess")  # the optional python-chess extra
    from oversight_arena.channels.evidence import Claim, VerifyEnv
    from oversight_arena.core.task import TaskView
    from oversight_arena.domains.chess import LineVerifier, apply_line

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board, applied, err = apply_line(fen, "e4 e5 Nf3")
    assert err is None and applied == ["e4", "e5", "Nf3"]
    v = LineVerifier(depth=4)
    env = VerifyEnv(view=TaskView(id="x", domain="chess", question="q"), resources={"fen": fen})
    ev = asyncio.run(v.verify(Claim(kind="line", content="e4 e5 Qh5"), env))
    assert ev.verified and "eval_cp" in ev.data
    ev = asyncio.run(v.verify(Claim(kind="line", content="e4 e4"), env))
    assert ev.verified is False


@pytest.mark.network
def test_network_domains():
    from oversight_arena.domains.forecasting import ManifoldForecasting
    from oversight_arena.domains.sql import PrivateSQL

    assert len(PrivateSQL().tasks()) == 36
    assert ManifoldForecasting(n=3).tasks()


def test_every_registered_component_resolves():
    from oversight_arena.registry import REGISTRY, resolve

    optional = {("domain", "control_arena"), ("gt", "control_arena")}  # need `control-arena`
    for kind, table in REGISTRY.items():
        for name in list(table):
            if (kind, name) not in optional:
                assert resolve(kind, name) is not None, (kind, name)
