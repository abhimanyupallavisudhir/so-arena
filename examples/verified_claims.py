"""A trainable judge paid for agreement with an explicitly available execution audit."""

import asyncio
from dataclasses import asdict

from oversight_arena import Action, Claim, Outcome, Reward, Role, Task, run
from oversight_arena.domains import SQLiteSnapshot
from oversight_arena.verification import VerifierRegistry


async def main():
    database = SQLiteSnapshot("CREATE TABLE items(x); INSERT INTO items VALUES(1),(2);")
    registry = VerifierRegistry()
    registry.register(
        "sql-result", "fixture-1", "Exact output on the public fixture", database.check_claim
    )

    async def worker(obs):
        return Action(
            "There are two items.",
            claims=(
                Claim(
                    "The query returns two",
                    {"query": "SELECT count(*) FROM items", "rows": [[2]]},
                    "sql-result",
                ),
            ),
        )

    async def judge(obs):
        return Action(data={"probability_valid": 0.8})

    async def mechanism(ctx):
        proposal = await ctx.act("worker", "Make a checkable claim.")
        judgment = await ctx.act("judge", "Predict whether the execution check passes.")
        # This audit is part of the mechanism, charged as a tool call. It is not the
        # independent evaluation oracle. A different hidden suite can still evaluate it.
        evidence = await ctx.tool("judge", "verify", asdict(proposal.claims[0]))
        if evidence["status"] not in {"verified", "refuted"}:
            raise ValueError("Cannot settle audit reward without a verification result")
        outcome = float(evidence["status"] == "verified")
        p = judgment.data["probability_valid"]
        return Outcome(
            {"worker": Reward(p), "judge": Reward(-2 * (p - outcome) ** 2)}, {"audit": evidence}
        )

    result = await run(
        Task("claim", "Count items in the fixture."),
        (Role("worker"), Role("judge", tools=("verify",))),
        {"worker": worker, "judge": judge},
        mechanism,
        name="execution-audited-judge",
        tools={"verify": registry.tool},
        manifest={"fixture": "items-v1", "judge_rule": "summed-brier-v1"},
    )
    assert result.status == "complete", result.error
    print({r: reward.value for r, reward in result.outcome.rewards.items()})
    print(result.outcome.output["audit"]["scope"])


if __name__ == "__main__":
    asyncio.run(main())
