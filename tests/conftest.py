import asyncio

import pytest

from oversight_arena import Outcome, PublicTask, Reward, Role, Task, Workflow, run_episode
from oversight_arena.domains import ExactMatch
from oversight_arena.policies import ConstantPolicy


@pytest.fixture
def episode_factory():
    def factory(reward=0.5, quality=1, task_id="task", seed=0, text=None):
        async def protocol(context):
            await context.ask("agent")
            return Outcome({"agent": Reward(reward)}, text)

        return asyncio.run(
            run_episode(
                Task(PublicTask(task_id, "question"), {"answer": quality}),
                Workflow("fixture", protocol),
                [Role("agent")],
                {"agent": ConstantPolicy({"answer": 1})},
                seed=seed,
                scorer=ExactMatch(("agent",)),
            )
        )

    return factory
