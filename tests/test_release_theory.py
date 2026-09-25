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
    p = sg.SwarmParams(n=5, G=1.0, b=0.1, o=0.8)
    types = {e["type"] for e in sg.equilibria(3, p)}
    assert types == {"all report", "all silent", "mixed"}
    p2 = sg.SwarmParams(n=5, G=1.0, b=0.3, o=0.8)
    assert {e["type"] for e in sg.equilibria(3, p2)} == {"all report"}
    assert sg.basin_of_reporting(p2) == 1.0 and 0 < sg.basin_of_reporting(p) < 1
    assert sg.offender_gain(1.0, p) < 0 < sg.offender_gain(0.0, p)
    shared = sg.SwarmParams(n=5, G=1.0, b=0.0, o=0.8)
    assert sg.basin_of_reporting(shared) == 0.0 or {e["type"] for e in sg.equilibria(3, shared)} >= {"all silent"}


def test_disclosure_theory():
    rows = {(r["k"], r["judge"]): r for r in disclosure.asd_table(15, [0, 8])}
    assert rows[(0, "naive")]["debate_asd"] == pytest.approx(0.0)
    assert rows[(8, "naive")]["debate_asd"] > 5  # both sides disclosing ⇒ unravelling
    assert rows[(8, "naive")]["propaganda_asd"] < 0.1  # naive judge fooled by selective disclosure
    assert rows[(8, "selection-aware")]["propaganda_asd"] > 5  # a sceptical judge is not
