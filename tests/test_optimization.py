import asyncio
import math
from dataclasses import replace
from itertools import product

import pytest

from oversight_arena import Action, Outcome, Reward, Role, Task, run
from oversight_arena.optimization import (
    Decision,
    Leaf,
    best_of_n,
    nested_best_of_n,
    optimization_curve,
    prompt_search,
)


@pytest.mark.parametrize(
    "scores,base",
    [([0, 1, 2], [0.2, 0.3, 0.5]), ([0, 1, 1], [0.2, 0.3, 0.5]), ([1, 1, 1], [0, 0.5, 0.5])],
)
@pytest.mark.parametrize("n", [1, 2, 4])
def test_bon_matches_independent_exhaustive_enumeration(scores, base, n):
    expected = [0.0] * len(scores)
    for draws in product(range(len(scores)), repeat=n):
        probability = math.prod(base[i] for i in draws)
        high = max(scores[i] for i in draws)
        winners = [i for i in draws if scores[i] == high]
        for i in winners:
            expected[i] += probability / len(winners)
    assert best_of_n(scores, n, base) == pytest.approx(expected)


def test_nested_bon_uses_each_players_reward_and_chance():
    tree = Decision(
        "a",
        (
            Decision("b", (Leaf({"a": 1, "b": 0}, {"a": 1}), Leaf({"a": 0, "b": 1}, {"a": 0}))),
            Leaf({"a": 0.4, "b": 0.2}, {"a": 0.5}),
        ),
    )
    unoptimized = nested_best_of_n(tree, {"a": 1, "b": 1})
    assert unoptimized.rewards["a"] == 0.45
    optimized = nested_best_of_n(tree, {"a": 10, "b": 10})
    assert 0.39 < optimized.rewards["a"] < 0.401
    chance = Decision(None, (Leaf({"a": 1}), Leaf({"a": 0})), (0.2, 0.8))
    assert nested_best_of_n(chance, {"a": 20}).rewards["a"] == 0.2


def test_increasing_reward_can_decrease_quality_and_missing_stays_missing():
    curve = optimization_curve([0.4, 0.9], [1, 0], [1, 2, 10])
    assert curve[-1]["reward"] > curve[0]["reward"]
    assert curve[-1]["quality"] < curve[0]["quality"]
    assert optimization_curve([0.4, 0.9], None, [1])[0]["quality"] is None


def test_extreme_bon_budget_keeps_probability_mass():
    assert best_of_n([0.0, 1.0, 2.0, 3.0], 10**16, [0.1, 0.2, 0.7, 0.0]) == [0.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize(
    "scores,n,base",
    [
        ([], 1, None),
        ([1], 0, None),
        ([1], True, None),
        ([float("nan")], 1, None),
        ([1, 2], 1, [1]),
        ([1, 2], 2, [0.6, 0.6]),
    ],
)
def test_invalid_bon_input(scores, n, base):
    with pytest.raises(ValueError):
        best_of_n(scores, n, base)


def test_search_uses_train_reward_and_freezes_selection_before_holdout():
    calls = []

    async def propose(history, stratum, rng):
        assert all(not hasattr(h, "quality") for h in history)
        return str(len(history))

    async def evaluate(prompt, tasks, seed):
        calls.append((prompt, [t.split for t in tasks]))

        async def policy(obs):
            return Action(prompt)

        async def mechanism(ctx):
            await ctx.act("a")
            return Outcome({"a": Reward(float(prompt))})

        return [
            await run(
                t,
                (Role("a"),),
                {"a": policy},
                mechanism,
                name="search",
                manifest={"prompt": prompt},
            )
            for t in tasks
        ]

    train, holdout = [Task("train", "q", split="train")], [Task("test", "q")]
    result = asyncio.run(
        prompt_search(
            propose=propose,
            evaluate=evaluate,
            train=train,
            holdout=holdout,
            role="a",
            iterations=3,
            opponent_snapshot="v1",
            constraint=lambda r: r.reward("a") < 2,
        )
    )
    assert result.selected == "1"
    assert calls == [("0", ["train"]), ("1", ["train"]), ("2", ["train"]), ("1", ["test"])]
    with pytest.raises(ValueError, match="disjoint"):
        asyncio.run(
            prompt_search(
                propose=propose,
                evaluate=evaluate,
                train=train,
                holdout=[replace(train[0], split="test")],
                role="a",
                iterations=1,
                opponent_snapshot="v1",
            )
        )


@pytest.mark.parametrize("schedule,opponent", [("simultaneous", "old"), ("alternating", "new")])
def test_joint_search_schedule_records_actual_opponents(schedule, opponent):
    from oversight_arena.optimization import joint_prompt_search

    async def propose(history, stratum, rng):
        return "new"

    async def evaluate(profile, tasks, seed):
        async def policy(obs):
            return Action()

        async def mechanism(ctx):
            return Outcome({r: Reward(float(p == "new")) for r, p in profile.items()})

        return [
            await run(
                t,
                (Role("a"), Role("b")),
                {"a": policy, "b": policy},
                mechanism,
                name="joint",
                manifest={"profile": profile},
            )
            for t in tasks
        ]

    result = asyncio.run(
        joint_prompt_search(
            initial={"a": "old", "b": "old"},
            proposers={"a": propose, "b": propose},
            evaluate=evaluate,
            train=[Task("train", "q", split="train")],
            holdout=[Task("test", "q")],
            steps=1,
            schedule=schedule,
        )
    )
    assert result.prompts == {"a": "new", "b": "new"}
    assert result.updates[1].opponents["a"] == opponent
    assert len(result.holdout_run_ids) == 1
