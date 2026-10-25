# CalcVerifier: (1) commas are stripped before evaluation, so any multi-argument function
# (max/min/round) errors and a TRUE claim is REFUTED; (2) check the documented example.
import asyncio
from oversight_arena.domains.math import CalcVerifier
from oversight_arena.channels.evidence import Claim, VerifyEnv
from oversight_arena.core.task import Task

env = VerifyEnv(view=Task(id="t", domain="d", question="q").view())
v = CalcVerifier()
for c in ["12 * 7 = 84", "12 * 7 = 85", "max(3, 5) = 5", "min(12, 7) = 7", "round(2.567, 2) = 2.57", "1,000 + 1 = 1001"]:
    ev = asyncio.run(v.verify(Claim(kind="calc", content=c), env))
    print(f"{c!r:26} verified={ev.verified}  result={ev.result}")
