import asyncio
from dataclasses import asdict, replace

import pytest

from oversight_arena import (
    Budget,
    BudgetExceeded,
    Evaluation,
    Outcome,
    PublicTask,
    Response,
    Reward,
    Role,
    Task,
    Tool,
    Workflow,
    evaluate,
    run_episode,
)
from oversight_arena.core import episode_from_dict
from oversight_arena.experiments import run_campaign
from oversight_arena.mechanisms import Debate, Monitoring
from oversight_arena.policies import ConstantPolicy, FunctionPolicy
from oversight_arena.verification import Claim, receipt_matches, verify_claim


def test_oracle_visibility_and_copy_boundaries():
    seen = []

    async def policy(observation):
        seen.append(observation)
        observation.task.data["mutated"] = True
        return Response("answer", channels={"reasoning": "private reasoning"})

    async def protocol(context):
        assert not hasattr(context, "evaluator_data")
        context.emit("a", "message", "secret", recipients=("a",))
        await context.ask("a")
        await context.ask("b")
        assert "mutated" not in context.task.data
        return Outcome({"a": Reward(1)})

    task = Task(PublicTask("task", "public"), {"answer": "oracle"}, {"a": {"key": "own secret"}})
    result = asyncio.run(
        run_episode(
            task,
            Workflow("visibility", protocol),
            [Role("a"), Role("b", False)],
            {"a": FunctionPolicy(policy), "b": FunctionPolicy(policy)},
        )
    )
    assert seen[0].private == {"key": "own secret"}
    assert seen[1].private == {}
    assert all(e.content != "secret" for e in seen[1].history)
    assert all(not e.channels for e in seen[1].history)
    assert "oracle" not in str(asdict(result))
    assert "mutated" not in task.public.data


def test_channel_allowlist():
    seen = []

    async def observer(observation):
        seen.append(observation)
        return Response({"suspicion": 0.2})

    async def worker(observation):
        return Response("ok", {"reasoning": "r", "activations": [1]})

    asyncio.run(
        run_episode(
            Task(PublicTask("t", "p")),
            Monitoring(),
            [Role("worker"), Role("monitor", False)],
            {"worker": FunctionPolicy(worker), "monitor": FunctionPolicy(observer)},
            channel_access={"monitor": frozenset({"activations"})},
        )
    )
    assert seen[0].history[0].channels == {"activations": [1]}


def test_missing_trainable_judge_reward():
    with pytest.raises(ValueError, match="Missing rewards"):
        asyncio.run(
            run_episode(
                Task(PublicTask("t", "p")),
                Debate(),
                [Role("proposer"), Role("critic"), Role("judge")],
                {
                    "proposer": ConstantPolicy("yes"),
                    "critic": ConstantPolicy("no"),
                    "judge": ConstantPolicy({"scores": {"proposer": 1, "critic": 0}}),
                },
            )
        )


def test_trainable_judge_can_receive_explicit_reward():
    async def judge_reward(context, verdict):
        return Reward(0.4, {"audit": 0.4})

    result = asyncio.run(
        run_episode(
            Task(PublicTask("t", "p")),
            Debate(judge_reward=judge_reward),
            [Role("proposer"), Role("critic"), Role("judge")],
            {
                "proposer": ConstantPolicy("yes"),
                "critic": ConstantPolicy("no"),
                "judge": ConstantPolicy({"scores": {"proposer": 1, "critic": 0}}),
            },
        )
    )
    assert result.rewards["judge"].value == 0.4


def test_simultaneous_debate_uses_frozen_history():
    observations = []

    async def policy(obs):
        observations.append(obs)
        return Response("message")

    asyncio.run(
        run_episode(
            Task(PublicTask("t", "p")),
            Debate(rounds=2, simultaneous=True),
            [Role("proposer"), Role("critic"), Role("judge", False)],
            {
                "proposer": FunctionPolicy(policy),
                "critic": FunctionPolicy(policy),
                "judge": ConstantPolicy({"scores": {"proposer": 1, "critic": 0}}),
            },
        )
    )
    assert len(observations[0].history) == len(observations[1].history) == 0
    assert len(observations[2].history) == len(observations[3].history) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_nonfinite_rewards_rejected(value):
    with pytest.raises(ValueError):
        Reward(value)


def test_schema_hash_detects_tampering(episode_factory):
    original = episode_factory()
    assert episode_from_dict(asdict(original)) == original
    with pytest.raises(ValueError, match="content"):
        replace(original, output="tampered").validate()


def test_concurrent_calls_reserve_budget():
    async def protocol(context):
        await asyncio.gather(context.ask("a"), context.ask("a"))
        return Outcome({"a": Reward(1)})

    with pytest.raises(BudgetExceeded):
        asyncio.run(
            run_episode(
                Task(PublicTask("t", "p")),
                Workflow("budget", protocol),
                [Role("a")],
                {"a": ConstantPolicy("hi")},
                budget=Budget(calls=1),
            )
        )


def test_unknown_cost_does_not_pass_cost_budget():
    async def policy(_):
        return Response("hello")

    async def protocol(context):
        await context.ask("a")
        return Outcome({"a": Reward(1)})

    with pytest.raises(BudgetExceeded, match="unpriced"):
        asyncio.run(
            run_episode(
                Task(PublicTask("t", "p")),
                Workflow("cost", protocol),
                [Role("a")],
                {"a": FunctionPolicy(policy)},
                budget=Budget(cost=5),
            )
        )


def test_timeout_and_campaign_failures():
    async def protocol(context):
        await asyncio.sleep(0.1)
        return Outcome({"a": Reward(1)})

    result = asyncio.run(
        run_campaign(
            [Task(PublicTask("t", "p"))],
            [Workflow("slow", protocol)],
            [Role("a")],
            {"a": ConstantPolicy(0)},
            budget=Budget(timeout=0.001),
        )
    )
    assert not result.records
    assert result.failures[0].error_type == "TimeoutError"
    assert result.completion_rate == 0


def test_tool_permissions_and_verification_binding():
    async def verify(arguments):
        return {"valid": arguments["artifact"] == "proof", "details": "checked"}

    async def protocol(context):
        claim = Claim("a", "Statement", "proof", "spec")
        receipt = await verify_claim(context, claim, "kernel")
        assert receipt_matches(receipt, claim)
        assert not receipt_matches(receipt, replace(claim, specification="different"))
        assert not receipt_matches(receipt, replace(claim, artifact="different"))
        with pytest.raises(PermissionError):
            await context.use_tool("b", "kernel", {})
        return Outcome({"a": Reward(1)})

    episode = asyncio.run(
        run_episode(
            Task(PublicTask("t", "p")),
            Workflow("verify", protocol),
            [Role("a"), Role("b", False)],
            {"a": ConstantPolicy(0), "b": ConstantPolicy(0)},
            tools={"kernel": Tool("kernel", verify, frozenset({"a"}))},
        )
    )
    assert episode.usage["tool_calls"] == 1


def test_interactive_tool_loop():
    async def policy(observation):
        tools = [e for e in observation.history if e.kind == "tool"]
        return Response(
            {"final": tools[-1].content["result"]}
            if tools
            else {"tool": "add", "arguments": [2, 3]}
        )

    async def tool(arguments):
        return sum(arguments)

    async def protocol(context):
        event = await context.interact("a", "Add numbers")
        return Outcome({"a": Reward(1)}, event.content)

    result = asyncio.run(
        run_episode(
            Task(PublicTask("t", "p")),
            Workflow("loop", protocol),
            [Role("a")],
            {"a": FunctionPolicy(policy)},
            tools={"add": Tool("add", tool, frozenset({"a"}))},
        )
    )
    assert result.output == 5 and result.usage["calls"] == 2


def test_evaluator_cannot_mutate_mechanism_record(episode_factory):
    original = episode_factory()

    async def scorer(task, episode):
        episode.rewards.clear()
        return Evaluation("new", "1", {"agent": {"quality": 0}})

    updated = asyncio.run(evaluate(Task(original.task), original, scorer))
    assert updated.id == original.id and updated.rewards == original.rewards
    assert len(updated.evaluations) == 2
