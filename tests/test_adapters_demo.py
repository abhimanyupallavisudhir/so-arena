import asyncio
import json

import pytest

from oversight_arena.demo import demo
from oversight_arena.storage import Store


def test_demo_and_delayed_resolution_cli(tmp_path, capsys):
    from unittest.mock import patch

    from oversight_arena.cli import main

    summary = asyncio.run(demo(tmp_path))
    assert summary["consultancy"]["estimate"]["value"] < 0
    assert summary["debate"]["estimate"]["value"] > 0
    assert "inflate" in summary["prompt_search"]["selected"]
    store = Store(tmp_path / "records")
    assert len(store.runs()) == 109
    assert all(r.status == "complete" for r in store.runs())
    resolution = tmp_path / "resolution.json"
    resolution.write_text(json.dumps({"forecast-1": True}))
    args = [
        "oversight-arena",
        "resolve",
        str(tmp_path / "records"),
        str(resolution),
        "--role",
        "forecaster",
        "--version",
        "resolved",
    ]
    with patch("sys.argv", args):
        main()
    labels = [e for e in store.evaluations() if e.scorer == "forecast"]
    assert len(labels) == 2
    assert (tmp_path / "report.html").exists() and (tmp_path / "forecast.html").exists()


def test_actual_inspect_models_and_log_import(monkeypatch):
    pytest.importorskip("inspect_ai")
    import inspect_ai.model
    from inspect_ai.log import EvalSample
    from inspect_ai.model import ModelOutput
    from inspect_ai.scorer import Score

    from oversight_arena import Observation, Task
    from oversight_arena.adapters import InspectPolicy, import_control_sample

    class Model:
        name = "fixture"

        async def generate(self, messages, config):
            assert "oracle" not in messages[-1].content
            return ModelOutput.from_content("fixture", '{"text":"hi","data":{"answer":3}}')

    monkeypatch.setattr(inspect_ai.model, "get_model", lambda model: Model())
    obs = Observation(Task("t", "q"), "a", "answer", (), 0)
    action = asyncio.run(InspectPolicy("fixture")(obs))
    assert action.data["answer"] == 3
    sample = EvalSample(
        id="t",
        epoch=1,
        input="PRIVATE INPUT",
        target="PRIVATE ANSWER",
        scores={"suspicion": Score(value=0.8), "task": Score(value=1.0)},
    )
    r, e = import_control_sample(
        sample,
        source="log@sha256",
        snapshot="repo@sha:db@revision",
        role="worker",
        reward_scorer="suspicion",
        reward_transform=lambda x: 1 - x,
        quality_scorers={"task": ("quality", "exact")},
        condition="attack",
    )
    assert r.reward("worker") == pytest.approx(0.2)
    assert e.scores["worker"]["quality"].value == 1
    assert "PRIVATE ANSWER" not in repr(r)
    assert "PRIVATE INPUT" not in repr(r)
    # Attack is a condition, never a zero-quality label.
    assert r.manifest["condition"]["id"] == "attack"


def test_real_inspect_eval_roundtrip_without_provider_calls(tmp_path):
    pytest.importorskip("inspect_ai")
    from inspect_ai import eval

    from oversight_arena import Task
    from oversight_arena.adapters import as_inspect_task, load_control_log
    from oversight_arena.demo import sql_experiment
    from oversight_arena.experiments import Condition

    task = as_inspect_task(
        [Task("t", "Compute net revenue")],
        sql_experiment(False),
        Condition("honest", {"worker": "Be faithful"}),
    )
    logs = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert len(logs) == 1 and logs[0].status == "success", logs[0].error
    samples = load_control_log(logs[0].location)
    assert samples[0].metadata["oversight_run"]["status"] == "complete"
    assert any("reward/worker" in s.value for s in samples[0].scores.values())
