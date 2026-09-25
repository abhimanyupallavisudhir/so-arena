import json

import pytest

import oversight_arena as oa
from oversight_arena.domains.base import TaskListDomain
from oversight_arena.domains.forecasting import forecast_task
from oversight_arena.mechanisms import Forecast, JudgeRating
from oversight_arena.release import create_release, inclusion_proof, resolve_release, verify_release
from oversight_arena.release.commit import merkle_proof, merkle_root, verify_proof
from oversight_arena.theory import disclosure, swarm_game as sg


def test_merkle():
    leaves = [f"{i:064x}" for i in range(7)]
    root = merkle_root(leaves)
    for l in leaves:
        assert verify_proof(l, merkle_proof(leaves, l), root)
    assert not verify_proof(leaves[0], merkle_proof(leaves, leaves[1]), root)


def test_release_roundtrip(tmp_path):
    pending = [forecast_task(f"q{i}", f"q{i}?", None) for i in range(5)]
    agents = {"*": oa.ScriptedAgent(lambda o: {"probability": 0.8}, id="f"),
              "kind:judge": oa.ScriptedAgent(lambda o: {"probability": 0.5, "rating": 6}, id="j")}
    res = oa.Experiment(TaskListDomain(task_list=pending), Forecast(judge=True, reward=JudgeRating()), agents, progress=False).run()
    rel = create_release(res, tmp_path / "rel", title="t")
    assert verify_release(rel.path)["ok"]
    assert inclusion_proof(rel.path, res.records[0].id)["valid"]
    items = json.loads((rel.path / "items.json").read_text())
    items[0]["rewards"]["forecaster"] = 99
    (rel.path / "items.json").write_text(json.dumps(items))
    assert not verify_release(rel.path)["ok"]
    rel = create_release(res, tmp_path / "rel", title="t")
    report = resolve_release(rel.path, [forecast_task(f"q{i}", f"q{i}?", 1.0) for i in range(5)])
    assert report["verification"]["ok"] and report["n_resolved_rows"] == 5
    assert (rel.path / "index.html").read_text().count("<script>") == 1
    sealed = create_release(res, tmp_path / "sealed", sealed=True)
    assert not (sealed.path / "items.json").exists() and verify_release(sealed.path)["ok"]


def test_swarm_theory():
    g = 0.3
    lo, hi, zero = (sg.SwarmParams(n=3, g=g, b=b, P=0.3, o=0.8) for b in (0.1, 0.45, 0.0))
    assert sg.regime(lo) == "coordination" and sg.regime(hi) == "report dominant" and sg.regime(zero) == "silent dominant"
    assert {e["type"] for e in sg.equilibria(2, lo)} == {"all report", "all silent", "mixed"}
    assert sg.basin_of_reporting(hi) == 1.0 and 0 < sg.basin_of_reporting(lo) < 1 and sg.basin_of_reporting(zero) == 0.0
    # risk dominance with m = 2 observers: report selected iff 1.5 b + (1 + a) c > (1 - a) g
    assert sg.selected(2, sg.SwarmParams(g=g, b=0.21)) == "all report" and sg.selected(2, sg.SwarmParams(g=g, b=0.19)) == "all silent"
    assert sg.offender_gain(1.0, lo) < 0 < sg.offender_gain(0.0, lo)
    assert sg.SwarmParams.pooled(3, 0.9).g == pytest.approx(0.3)
    # full-game learning dynamics: coordination below the stake, deterrence from any start above it
    assert sg.mean_field(0.5, 0.1, lo)["x"].iloc[-1] > 0.99 and sg.mean_field(0.5, 0.9, lo)["x"].iloc[-1] < 0.01
    assert sg.mean_field(0.5, 0.1, hi)["x"].iloc[-1] < 0.01
    assert sg.basin_of_deterrence(hi) == 1.0 and 0 < sg.basin_of_deterrence(lo) < 1
    # the 2-observer matrix game agrees with the empirical-game solver
    from oversight_arena.analysis import EmpiricalGame

    for p, expect in ((lo, {("R", "R"), ("S", "S")}), (hi, {("R", "R")})):
        A, B = sg.payoff_matrix_2(p)
        game = EmpiricalGame.from_matrices(A, B, row_strats=["R", "S"], col_strats=["R", "S"])
        pure = {(game.strategies["row"][int(e.mix["row"].argmax())], game.strategies["col"][int(e.mix["col"].argmax())])
                for e in game.pure_nash()}
        assert pure == expect


def test_disclosure_theory():
    rows = {(r["k"], r["judge"]): r for r in disclosure.asd_table(15, [0, 8])}
    assert rows[(0, "naive")]["debate_asd"] == pytest.approx(0.0)
    assert rows[(8, "naive")]["debate_asd"] > 5  # both sides disclosing ⇒ unravelling
    assert rows[(8, "naive")]["propaganda_asd"] < 0.1  # naive judge fooled by selective disclosure
    assert rows[(8, "selection-aware")]["propaganda_asd"] > 5  # a sceptical judge is not
