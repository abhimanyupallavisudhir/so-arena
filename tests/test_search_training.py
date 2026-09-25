import asyncio
from dataclasses import asdict, replace

import pytest

from oversight_arena import Outcome, PublicTask, Response, Reward, Role, Task, Workflow, run_episode
from oversight_arena.policies import ConstantPolicy, FunctionPolicy
from oversight_arena.search import prompt_search
from oversight_arena.training import CategoricalREINFORCE, training_batch


async def reward_protocol(context):
    a = await context.ask("a")
    b = await context.ask("b")
    return Outcome({"a": Reward(a.content), "b": Reward(b.content)})


def setup():
    async def policy(observation):
        return Response(1 if "optimized" in observation.instruction else 0, cost=0)

    return dict(
        train=[Task(PublicTask("train", "q"), {"secret": "oracle"}, split="train")],
        heldout=[Task(PublicTask("test", "q"), split="test")],
        mechanism=Workflow("rewards", reward_protocol),
        roles=[Role("a"), Role("b")],
        policies={"a": FunctionPolicy(policy), "b": FunctionPolicy(policy)},
    )


@pytest.mark.parametrize("schedule", ["simultaneous", "alternating"])
def test_search_peer_snapshots_and_oracle_exclusion(schedule):
    requests = []

    async def propose(request):
        requests.append(request)
        assert "oracle" not in str(asdict(request))
        return "optimized"

    result = asyncio.run(
        prompt_search(
            **setup(), proposers={"a": propose, "b": propose}, schedule=schedule, trials=2
        )
    )
    assert result.prompts == {"a": "optimized", "b": "optimized"}
    assert requests[1].peer_prompts["a"] == ("optimized" if schedule == "alternating" else "")
    assert all(not r.evaluations for trial in result.trials for r in trial.records)
    assert result.heldout[0].rewards["a"].value == 1


def test_search_keeps_incumbent_on_tie():
    async def propose(_):
        return "also optimized"

    result = asyncio.run(
        prompt_search(**setup(), proposers={"a": propose}, initial={"a": "optimized"}, trials=2)
    )
    assert result.prompts["a"] == "optimized"


def test_search_rejects_split_leakage_and_fixture_training():
    async def propose(_):
        return "optimized"

    kwargs = setup()
    kwargs["heldout"] = [replace(kwargs["train"][0], split="test")]
    with pytest.raises(ValueError, match="disjoint"):
        asyncio.run(prompt_search(**kwargs, proposers={"a": propose}))
    kwargs = setup()
    kwargs["roles"] = [Role("a", False), Role("b")]
    with pytest.raises(ValueError, match="trainable"):
        asyncio.run(prompt_search(**kwargs, proposers={"a": propose}))


def test_search_cohort_filter_is_enforced():
    async def propose(_):
        return "deceptive"

    with pytest.raises(ValueError, match="constraint"):
        asyncio.run(
            prompt_search(
                **setup(),
                proposers={"a": propose},
                trials=2,
                cohort="honest",
                prompt_filter=lambda cohort, p: p != "deceptive",
            )
        )


def test_reinforce_changes_policy_and_checkpoint_replays():
    kwargs = setup()
    trainer = CategoricalREINFORCE(
        {"a": [ConstantPolicy(0), ConstantPolicy(1)]}, learning_rate=0.8, seed=3
    )

    async def train():
        for _ in range(40):
            await trainer.step(
                kwargs["train"] * 4, kwargs["mechanism"], kwargs["roles"], {"b": ConstantPolicy(0)}
            )
        checkpoint = trainer.checkpoint()
        one = await trainer.step(
            kwargs["train"], kwargs["mechanism"], kwargs["roles"], {"b": ConstantPolicy(0)}
        )
        after = trainer.checkpoint()
        trainer.restore(checkpoint)
        two = await trainer.step(
            kwargs["train"], kwargs["mechanism"], kwargs["roles"], {"b": ConstantPolicy(0)}
        )
        assert one == two
        assert trainer.checkpoint() == after

    asyncio.run(train())
    assert trainer.distributions()["a"][1] > 0.95


def test_training_export_has_no_truth_or_private_peers():
    async def protocol(context):
        context.emit("b", "message", "secret", recipients=("b",))
        context.emit("a", "message", "public", channels={"reasoning": "secret"})
        return Outcome({"a": Reward(0.2)})

    episode = asyncio.run(
        run_episode(
            Task(PublicTask("t", "q")),
            Workflow("x", protocol),
            [Role("a"), Role("b", False)],
            {"a": ConstantPolicy(0), "b": ConstantPolicy(0)},
        )
    )
    batch = training_batch([episode])
    assert len(batch) == 1 and batch[0].reward == 0.2
    assert "secret" not in str(batch)


def test_reinforce_rejects_test_tasks():
    kwargs = setup()
    trainer = CategoricalREINFORCE({"a": [ConstantPolicy(1)]})
    with pytest.raises(ValueError, match="train"):
        asyncio.run(
            trainer.step(
                kwargs["heldout"], kwargs["mechanism"], kwargs["roles"], {"b": ConstantPolicy(0)}
            )
        )


def test_checkpoint_restore_is_atomic_and_restores_seed():
    import json

    trainer = CategoricalREINFORCE({"a": [ConstantPolicy(0), ConstantPolicy(1)]}, seed=5)
    checkpoint = json.loads(json.dumps(trainer.checkpoint()))
    restored = CategoricalREINFORCE(trainer.populations, seed=99)
    restored.restore(checkpoint)
    assert restored.checkpoint() == trainer.checkpoint()
    checkpoint["learning_rate"] = -1
    before = restored.checkpoint()
    with pytest.raises(ValueError):
        restored.restore(checkpoint)
    assert restored.checkpoint() == before
