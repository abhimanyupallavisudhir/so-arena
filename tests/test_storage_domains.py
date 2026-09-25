import asyncio
import json
from dataclasses import replace

import pytest

from oversight_arena import (
    Evaluation,
    Outcome,
    PublicTask,
    Reward,
    Role,
    Task,
    Workflow,
    evaluate,
    run_episode,
)
from oversight_arena.demo import CORRECT_SQL, sql_tasks
from oversight_arena.domains import Forecast, SQLExecution
from oversight_arena.mechanisms import Swarm
from oversight_arena.policies import ConstantPolicy, FunctionPolicy
from oversight_arena.reporting import write_report
from oversight_arena.storage import RunStore, public_result


async def worker_protocol(context):
    result = await context.ask("agent")
    return Outcome({"agent": Reward(0.9)}, result.content)


def sql_episode(query):
    return asyncio.run(
        run_episode(
            sql_tasks()[0],
            Workflow("sql", worker_protocol),
            [Role("agent")],
            {"agent": ConstantPolicy({"sql": query})},
            scorer=SQLExecution(("agent",)),
        )
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        (CORRECT_SQL, 1),
        ("SELECT count(*) FROM orders", 0),
        ("DELETE FROM orders", 0),
        ("ATTACH DATABASE '/tmp/test.db' AS other", 0),
        ("SELECT load_extension('/tmp/evil')", 0),
        ("SELECT * FROM missing", 0),
        ("PRAGMA table_info(orders)", 0),
    ],
)
def test_sql_scoring_and_write_denial(query, expected):
    result = sql_episode(query)
    assert result.evaluations[0].per_agent["agent"]["quality"] == expected


def test_sql_duplicate_rows_are_not_set_equivalent():
    scorer = SQLExecution(("agent",))
    task = replace(
        sql_tasks()[0],
        evaluator_data={
            "fixture_sql": "CREATE TABLE a(x); INSERT INTO a VALUES(1),(1),(2);",
            "reference_sql": "SELECT x FROM a",
        },
    )
    result = asyncio.run(
        run_episode(
            task,
            Workflow("sql", worker_protocol),
            [Role("agent")],
            {"agent": ConstantPolicy({"sql": "SELECT DISTINCT x FROM a"})},
            scorer=scorer,
        )
    )
    assert result.evaluations[0].per_agent["agent"]["quality"] == 0


def test_sql_execution_budget():
    task = sql_tasks()[0]
    query = (
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) SELECT sum(x) FROM cnt"
    )
    result = asyncio.run(
        run_episode(
            task,
            Workflow("sql", worker_protocol),
            [Role("agent")],
            {"agent": ConstantPolicy({"sql": query})},
            scorer=SQLExecution(("agent",), vm_steps=500),
        )
    )
    assert result.evaluations[0].per_agent["agent"]["execution_success"] == 0


def test_forecast_revisions_and_roundtrip(tmp_path):
    task = Task(PublicTask("forecast", "Will X happen?"))
    scorer = Forecast(("agent",))
    original = asyncio.run(
        run_episode(
            task,
            Workflow("forecast", worker_protocol),
            [Role("agent")],
            {"agent": ConstantPolicy({"probability": 0.8})},
            scorer=scorer,
        )
    )
    assert original.evaluations[-1].status == "pending"
    resolved = asyncio.run(evaluate(replace(task, evaluator_data={"outcome": 1}), original, scorer))
    assert resolved.id == original.id
    assert resolved.evaluations[-1].per_agent["agent"]["negative_brier"] == pytest.approx(-0.08)
    with RunStore(tmp_path / "runs.db") as store:
        store.save(original)
        store.save(resolved)
        store.save(original)
        assert store.get(original.id) == resolved
        store.export_jsonl(tmp_path / "records.jsonl")
        conflicting = replace(resolved, evaluations=(Evaluation("other", "1"),))
        with pytest.raises(ValueError, match="replace"):
            store.save(conflicting)
    with RunStore(tmp_path / "import.db") as store:
        assert store.import_jsonl(tmp_path / "records.jsonl") == 1
        assert store.get(original.id) == resolved


def test_database_detects_corruption(tmp_path, episode_factory):
    record = episode_factory()
    with RunStore(tmp_path / "runs.db") as store:
        store.save(record)
        store.connection.execute("UPDATE evaluations SET checksum='bad'")
        with pytest.raises(ValueError, match="checksum"):
            store.get(record.id)


def test_report_escapes_html_and_public_projection_excludes_private(tmp_path):
    async def protocol(context):
        context.emit("agent", "message", "secret", recipients=("agent",))
        context.emit("agent", "message", "<script>alert(1)</script>", channels={"cot": "secret"})
        return Outcome({"agent": Reward(0.2)})

    record = asyncio.run(
        run_episode(
            Task(PublicTask("t", "<script>bad</script>")),
            Workflow("x", protocol),
            [Role("agent")],
            {"agent": ConstantPolicy(0)},
        )
    )
    public = public_result(record)
    assert "secret" not in json.dumps(public)
    path = write_report([record], tmp_path / "report.html")
    text = path.read_text()
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "Unresolved / unavailable" in text


def test_swarm_split_bounty_and_false_report_penalty():
    from oversight_arena import Response

    async def policy(observation):
        if '"reports"' in observation.instruction:
            target = "c" if observation.role in ("a", "b") else "a"
            return Response({"reports": [{"target": target, "evidence": "item"}]})
        return Response("work")

    async def team(context):
        return 1

    async def audit(context, reporter, report):
        return report["target"] == "c"

    mechanism = Swarm(
        ("a", "b", "c"),
        team,
        audit,
        bounty=0.4,
        false_report_penalty=0.6,
        report_cost=0.1,
        shared_bounty=True,
    )
    result = asyncio.run(
        run_episode(
            Task(PublicTask("t", "work")),
            mechanism,
            [Role(r) for r in "abc"],
            {r: FunctionPolicy(policy) for r in "abc"},
        )
    )
    assert result.rewards["a"].value == pytest.approx(1.1)
    assert result.rewards["b"].value == pytest.approx(1.1)
    assert result.rewards["c"].value == pytest.approx(0.3)


def test_unknown_ground_truth_has_no_scores():
    with pytest.raises(ValueError, match="Unresolved"):
        Evaluation("forecast", "1", {"agent": {"quality": 0}}, status="pending")


def test_sql_oversized_blob_rejected_before_materialization():
    result = sql_episode("SELECT zeroblob(1000000000)")
    assert result.evaluations[0].per_agent["agent"]["execution_success"] == 0
