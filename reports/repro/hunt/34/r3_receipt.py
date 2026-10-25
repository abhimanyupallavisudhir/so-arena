# Attempt 3: receipt_matches ignores verifier identity/version and the verdict itself
import asyncio, sys
sys.path.insert(0, "/tmp/att3/src")
from oversight_arena.core import PublicTask, Role
from oversight_arena.runtime import Context, Tool
from oversight_arena.verification import Claim, verify_claim, receipt_matches
from oversight_arena.policies import ConstantPolicy

async def always_true(args): return {"valid": True}
async def strict(args): return {"valid": False}
async def main():
    ctx = Context(PublicTask("t", "q"), (Role("debater"),), {"debater": ConstantPolicy(1)}, 0,
                  tools={"syntax_lint": Tool("syntax_lint", always_true, frozenset({"debater"})),
                         "hidden_tests": Tool("hidden_tests", strict, frozenset({"debater"}))})
    claim = Claim("debater", "passes all hidden tests", {"code": "..."}, "hidden test suite")
    weak = await verify_claim(ctx, claim, "syntax_lint")
    bad = await verify_claim(ctx, claim, "hidden_tests")
    print("lint receipt   :", weak.content["verifier"], weak.content["valid"], "matches claim ->", receipt_matches(weak, claim))
    print("refuted receipt:", bad.content["verifier"], bad.content["valid"], "matches claim ->", receipt_matches(bad, claim))
asyncio.run(main())
