import asyncio
from so_arena.core.verification import QuoteVerifier, Claim
from so_arena.core.items import TaskItem
P = "The suspect was unarmed when police arrived. \"I didn't do it,\" she said — twice. The U.S. contract (2019) was illegal under state law."
item = TaskItem(id="q", question="?", private={"passage": P})
v = QuoteVerifier()
for q in ["The suspect was unarmed", "police arrived. \"I didn't do it,\"", "I didn’t do it", "she said - twice",
          "The U.S. contract", "contract (2019) was illegal", "was unarmed ... under state law", "suspect was", "armed", "U.S.", "(2019)",
          "arrived.", "\"I didn't do it\""]:
    r = asyncio.run(v.verify(Claim(kind="quote", content=q), item))
    print(f"{q!r:40} -> {r.status}")
