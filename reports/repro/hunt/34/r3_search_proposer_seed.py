# Attempt 3: prompt_search gives every role's proposer the identical seed sequence
import asyncio, sys
sys.path.insert(0, "/tmp/att3/src")
from oversight_arena.core import Outcome, PublicTask, Reward, Role, Task
from oversight_arena.policies import ConstantPolicy
from oversight_arena.runtime import Workflow
from oversight_arena.search import prompt_search

async def mech(ctx):
    await ctx.ask("a"); await ctx.ask("b")
    return Outcome({"a": Reward(0.0), "b": Reward(0.0)})
seen = []
def proposer(name):
    async def p(req):
        seen.append((name, req.seed)); return f"{name}-{req.seed}"
    return p
async def main():
    await prompt_search(train=[Task(PublicTask("x", "q"), split="train")], heldout=[],
        mechanism=Workflow("w", mech), roles=[Role("a"), Role("b")],
        policies={"a": ConstantPolicy(1), "b": ConstantPolicy(1)},
        proposers={"a": proposer("a"), "b": proposer("b")}, trials=3, rounds=2)
    print(seen)
asyncio.run(main())
