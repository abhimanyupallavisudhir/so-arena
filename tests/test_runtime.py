import asyncio
from dataclasses import replace

import pytest

from oversight_arena import Action, Budget, Context, Outcome, Reward, Role, Task, run
from oversight_arena.experiments import Condition, Experiment
from oversight_arena.mechanisms import Debate, Swarm, respond


def execute(awaitable):
    return asyncio.run(awaitable)


async def empty(obs):
    return Action()


def test_trainable_judge_requires_reward_but_fixture_does_not():
    async def judge(obs):
        return Action(data={"scores": {"a": 0.8, "b": 0.2}})

    async def trial(trainable, additional=None):
        return await run(
            Task("t", "q"),
            (Role("a"), Role("b"), Role("judge", trainable)),
            {"a": empty, "b": empty, "judge": judge},
            Debate(("a", "b"), additional_rewards=additional),
            name="debate",
        )

    failed = execute(trial(True))
    assert failed.status == "failed" and "missing=['judge']" in failed.error
    with pytest.raises(ValueError):
        failed.reward("a")
    assert execute(trial(False)).reward("a") == 0.8

    async def reward_judge(ctx, output):
        return {"judge": Reward(0.4)}

    assert execute(trial(True, reward_judge)).reward("judge") == 0.4


def test_simultaneous_players_cannot_observe_same_round_peers():
    observed = {}

    async def policy(obs):
        observed[obs.role] = obs.events
        # Mutation of nested data cannot alter another player's observation.
        obs.task.data["secret"] = obs.role
        return Action(obs.role)

    ctx = Context(
        Task("t", "q", {"secret": "original"}),
        (Role("a"), Role("b")),
        {"a": policy, "b": policy},
        2,
        Budget(),
    )
    execute(ctx.simultaneous({"a": "", "b": ""}))
    assert observed == {"a": (), "b": ()}
    assert ctx.task.data == {"secret": "original"}
    assert [e.actor for e in ctx.events if e.kind == "action"] == ["a", "b"]


def test_visibility_permissions_and_budget():
    seen = []

    async def spy(obs):
        seen.append(obs)
        return Action()

    ctx = Context(
        Task("t", "public"),
        (Role("a", tools=("lookup",)), Role("b")),
        {"a": spy, "b": spy},
        0,
        Budget(calls=2, tool_calls=1),
        {"lookup": lambda args: asyncio.sleep(0, result={"secret": 42})},
    )
    execute(ctx.tool("a", "lookup", {}, ("a",)))
    execute(ctx.act("b"))
    assert not seen[0].events
    with pytest.raises(PermissionError):
        execute(ctx.tool("b", "lookup", {}))
    with pytest.raises(RuntimeError):
        execute(ctx.tool("a", "lookup", {}))
    execute(ctx.act("a"))
    assert any(e.kind == "tool" for e in seen[1].events)
    with pytest.raises(RuntimeError):
        execute(ctx.act("a"))


def test_random_streams_match_pairs_but_differ_across_tasks():
    seeds = []

    async def policy(obs):
        seeds.append(obs.seed)
        return Action()

    for task_id in ("first", "first", "second"):
        ctx = Context(Task(task_id, "q"), (Role("a"),), {"a": policy}, 7, Budget())
        execute(ctx.act("a"))
    assert seeds[0] == seeds[1] and seeds[1] != seeds[2]


def test_failed_simultaneous_peer_cancels_remaining_calls():
    cancelled = []

    async def broken(obs):
        await asyncio.sleep(0.01)
        raise ValueError("invalid reply")

    async def slow(obs):
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)

    async def mechanism(ctx):
        await ctx.simultaneous({"a": "", "b": ""})
        return Outcome({"a": Reward(1), "b": Reward(1)})

    result = execute(
        run(
            Task("t", "q"),
            (Role("a"), Role("b")),
            {"a": broken, "b": slow},
            mechanism,
            name="cancel",
        )
    )
    assert result.status == "failed" and cancelled == [True]


def test_tool_loop_and_timeout_are_real():
    async def worker(obs):
        if any(e.kind == "tool" for e in obs.events):
            return Action("done")
        return Action(data={"tool_calls": [{"name": "x", "arguments": {}}]})

    async def mech(ctx):
        result = await respond(ctx, "a", "use x")
        return Outcome({"a": Reward(float(result.text == "done"))})

    result = execute(
        run(
            Task("t", "q"),
            (Role("a", tools=("x",)),),
            {"a": worker},
            mech,
            name="tool",
            tools={"x": lambda args: asyncio.sleep(0, result=1)},
        )
    )
    assert result.reward("a") == 1 and result.usage == {"calls": 2, "tool_calls": 1}

    async def slow(ctx):
        await asyncio.sleep(1)

    timed = execute(
        run(
            Task("t", "q"),
            (Role("a"),),
            {"a": empty},
            slow,
            name="slow",
            budget=Budget(seconds=0.01),
        )
    )
    assert timed.status == "failed" and "TimeoutError" in timed.error


def test_paired_environments_and_private_interventions():
    from contextlib import asynccontextmanager

    entered, left, observations = [], [], []

    @asynccontextmanager
    async def env(task, seed):
        entered.append(seed)
        try:
            yield {}
        finally:
            left.append(seed)

    async def worker(obs):
        observations.append(obs)
        return Action("answer")

    async def mechanism(ctx):
        await ctx.act("worker")
        return Outcome({"worker": Reward(1)})

    exp = Experiment(
        "paired",
        (Role("worker"),),
        {"worker": lambda: worker},
        lambda: mechanism,
        {},
        environment=env,
    )
    conditions = [
        Condition("good", {"worker": "secret-good"}),
        Condition("bad", {"worker": "secret-bad"}),
    ]
    results = execute(exp.sweep([Task("t", "q", snapshot="sha")], conditions, (4,)))
    assert entered == left == [4, 4]
    assert results[0].seed == results[1].seed
    assert observations[0].seed == observations[1].seed
    assert "secret-good" in observations[0].instruction
    assert "secret-good" in next(
        e.data["instruction"] for e in results[0].events if e.kind == "policy_input"
    )
    assert "condition" not in observations[0].task.data
    assert all(e.audience == () for e in results[0].events if e.kind == "policy_input")
    with pytest.raises(ValueError, match="Stateful"):
        execute(
            replace(
                exp,
                environment=__import__(
                    "oversight_arena.experiments", fromlist=["stateless_environment"]
                ).stateless_environment,
            ).trial(Task("t", "q", snapshot="sha"), conditions[0], 4)
        )


def test_swarm_pays_only_adjudicated_reports_and_preserves_shared_credit():
    async def adjudicate(ctx, reports):
        assert set(reports) == {"a", "b"}
        return {"team_credit": 0.5, "verified_reporters": ["a"], "false_reporters": ["b"]}

    result = execute(
        run(
            Task("t", "q"),
            (Role("a"), Role("b")),
            {"a": empty, "b": empty},
            Swarm(("a", "b"), adjudicate, 0.4, 0.2),
            name="swarm",
        )
    )
    assert result.reward("a") == 0.9 and result.reward("b") == 0.3
    assert sum(e.kind == "action" and e.audience == () for e in result.events) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "1"])
def test_invalid_rewards_rejected(value):
    with pytest.raises(ValueError):
        Reward(value)
