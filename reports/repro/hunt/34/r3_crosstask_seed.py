# Attempt 3: per-call seed = digest([episode_seed, role, index]) omits the task, so every task
# run with the same episode seed gets the identical random draw (MixturePolicy, seeded scripted agents).
import asyncio, sys
sys.path.insert(0, "/tmp/att3/src")
from oversight_arena.core import Outcome, PublicTask, Reward, Role, Task
from oversight_arena.policies import ConstantPolicy, MixturePolicy
from oversight_arena.runtime import Workflow
from oversight_arena.experiments import run_campaign
from oversight_arena.analysis import clustered_mean

async def mech(ctx):
    msg = await ctx.ask("agent", "go")
    return Outcome({"agent": Reward(float(msg.content["honest"]))})

async def main():
    tasks = [Task(PublicTask(f"t{i}", f"question {i}"), split="test") for i in range(200)]
    mix = MixturePolicy([ConstantPolicy({"honest": 1}), ConstantPolicy({"honest": 0})], [0.5, 0.5])
    for seeds in [(0,), (1,), (0, 1, 2)]:
        camp = await run_campaign(tasks, [Workflow("w", mech)], [Role("agent")], {"agent": mix}, seeds=seeds)
        est = clustered_mean([(r.task.id, r.rewards["agent"].value) for r in camp.records])
        print(f"seeds={seeds}: honest share over 200 tasks = {est.mean:.3f}, 95% CI [{est.low:.3f}, {est.high:.3f}]")
asyncio.run(main())
