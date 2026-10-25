import asyncio
from oversight_arena.domains.math import CalcVerifier
from oversight_arena.channels.evidence import Claim, VerifyEnv
from oversight_arena.core.task import TaskView
env = VerifyEnv(view=TaskView(id="t", domain="gsm8k", question="q"))
for e in ["12 * 7 == 84", "1,000,000 * 2 = 2,000,000", "12 x 7 = 84", "12 × 7 = 84", "$12 * 7 = $84", "12 * 7 = 84.", "(1,200 + 300) / 2 = 750",
          "max(1,000, 5) = 1000", "84 = 12 * 7", "12*7=84=84", "50% * 200 = 100", "3.5 * 2 = 7", "2^10 = 1024", "((10**100)**100) = 1", "7 / 2 = 3.5"]:
    ev = asyncio.run(CalcVerifier().verify(Claim(kind="calc", content=e), env))
    print(f"{e:32} verified={ev.verified!s:5} result={ev.result[:60]}")
