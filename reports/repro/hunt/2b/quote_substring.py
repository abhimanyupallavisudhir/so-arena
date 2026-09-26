# QuoteVerifier checks normalized substring containment with no word boundaries: a "quote" that starts mid-word
# is verified, so cutting a negating prefix ("un", "il", "s") flips meaning while staying "verified".
import asyncio
from so_arena.core.verification import QuoteVerifier, Claim
from so_arena.core.items import TaskItem
item = TaskItem(id="q", question="?", private={"passage":
    "The suspect was unarmed when police arrived. The contract was illegal under state law. She denied it."})
v = QuoteVerifier()
for q in ("armed when police arrived", "legal under state law", "he denied it", "was armed when police arrived"):
    r = asyncio.run(v.verify(Claim(kind="quote", content=q), item))
    print(f"{q!r:34} -> {r.status}")
