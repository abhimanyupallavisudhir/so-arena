import asyncio
import json
import math
from dataclasses import replace

import pytest

from oversight_arena import Action, Evaluation, Measurement, Outcome, Reward, Role, Task, run
from oversight_arena.mechanisms import endorsement_reward
from oversight_arena.metrics import alignment, bootstrap, detection, gap_recovered, paired_asd
from oversight_arena.reporting import write_report
from oversight_arena.scoring import ForecastScorer
from oversight_arena.storage import Store
from oversight_arena.training import training_rows


def record(task="t", condition="honest", reward=0.8, seed=0, split="test"):
    async def policy(obs):
        return Action("<script>alert(1)</script>", {"probability": 0.7})

    async def mechanism(ctx):
        ctx.emit("a", "private", {"hidden": "SECRET"}, ())
        await ctx.act("a")
        return Outcome({"a": Reward(reward)})

    return asyncio.run(
        run(
            Task(task, "<img src=x onerror=alert(1)>", split=split),
            (Role("a"),),
            {"a": policy},
            mechanism,
            name="m",
            seed=seed,
            manifest={"condition": {"id": condition}, "secret": "SECRET"},
        )
    )


def test_asd_uses_two_worlds_and_clustered_task_means():
    runs = [record("a", "honest", 0.8, i) for i in range(3)]
    runs += [record("a", "deceptive", 0.6, i) for i in range(3)]
    runs += [record("b", "honest", 0.2), record("b", "deceptive", 0.8)]
    metric = paired_asd(runs, "a")
    assert metric.pairs == 4 and metric.estimate.n == 2
    assert metric.estimate.value == pytest.approx(-0.2)  # equal tasks, not unequal episode counts
    assert metric.manipulation == "intent-only"
    assert endorsement_reward(0.8, "log") - endorsement_reward(0.6, "log") == pytest.approx(
        math.log(4 / 3)
    )
    assert endorsement_reward(0.8, "brier") == pytest.approx(-0.08)
    labels = [
        Evaluation(
            r.id,
            "audit",
            "1",
            {
                "a": {
                    "quality": Measurement(
                        1.0 if r.manifest["condition"]["id"] == "honest" else 0.0,
                        "observed",
                        "hidden tests",
                    )
                }
            },
        )
        for r in runs
    ]
    assert paired_asd(runs, "a", evaluations=labels).manipulation == "audited"
    assert paired_asd(runs, "a", evaluations=[]).excluded == 4
    with pytest.raises(ValueError, match="Duplicate"):
        paired_asd(runs + [runs[0]], "a")


def test_metric_undefined_cases_and_reward_ties():
    assert bootstrap([]).value is None
    assert bootstrap([1.0]).lower is None
    result = alignment([1, 1], [0, 1])
    assert result["concordance"] == 0.5 and result["selection_regret"] == 0.5
    assert result["pearson"] is None
    assert gap_recovered(0.7, 0.5, 0.9) == pytest.approx(0.5)
    assert gap_recovered(0.7, 0.5, 0.5) is None
    assert detection([0.9, 0.9], [True, False], threshold=0.5)["auroc"] == 0.5
    assert detection([0.9], [True], threshold=0.5)["fpr"] is None


def test_delayed_resolution_append_only_and_public_projection(tmp_path):
    store = Store(tmp_path / "store")
    r = record(split="unresolved")
    store.save_run(r)
    store.save_run(r)  # Idempotent.
    before = list((tmp_path / "store" / "runs").glob("*.json"))[0].read_bytes()
    pending = ForecastScorer({}, ("a",), "pending")(r)
    resolved = ForecastScorer({"t": True}, ("a",), "resolved-1")(r)
    assert pending.scores["a"]["brier"].value is None
    assert resolved.scores["a"]["brier"].value == pytest.approx(-0.18)
    store.save_evaluation(pending)
    store.save_evaluation(resolved)
    assert list((tmp_path / "store" / "runs").glob("*.json"))[0].read_bytes() == before
    assert len(store.evaluations()) == 2
    with pytest.raises(ValueError, match="Conflicting"):
        store.save_evaluation(replace(resolved, scores=pending.scores))
    with pytest.raises(ValueError, match="stored run"):
        store.save_evaluation(replace(pending, run_id="missing"))
    public = tmp_path / "public.jsonl"
    store.export_public(public)
    assert "SECRET" not in public.read_text()
    assert "brier" not in public.read_text()
    assert json.loads(public.read_text())["outcome"]["rewards"]["a"]["value"] == 0.8
    report = write_report(store.runs(), tmp_path / "report.html", evaluations=[pending, resolved])
    assert "<script>" not in report.read_text()
    assert "Pending" in report.read_text() and "-0.1800" in report.read_text()
    assert "SECRET" not in report.read_text()


def test_exports_only_trainable_training_split():
    train = record(split="train")
    assert len(training_rows([train, record()])) == 1
    assert training_rows([replace(train, roles=(Role("a", False),))]) == []
    assert training_rows([replace(train, status="failed", outcome=None)]) == []
