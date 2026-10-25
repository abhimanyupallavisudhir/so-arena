# QuoteVerifier checks a raw substring of the normalised story with no word boundaries or minimum
# length, so a "verbatim" quote can start/end inside a word and invert the meaning.
import asyncio, json
from oversight_arena.domains.quality import QuoteVerifier, _norm
from oversight_arena.channels.evidence import Claim, VerifyEnv
from oversight_arena.core.task import Task
story = "The captain found it notable that the crew had not abandoned the ship. Hendricks was innocent."
env = VerifyEnv(view=Task(id="t", domain="d", question="q").view(), resources={"article": story})
for q in ["found it not", "the crew had not abandoned", "cks was innocent", "he crew had", "Hendricks was not innocent"]:
    ev = asyncio.run(QuoteVerifier().verify(Claim(kind="quote", content=q), env))
    print(f"{q!r:30} verified={ev.verified}")
# on the real QuALITY dev set: how many articles contain "not" only as a word-prefix fragment, e.g. "nothing"/"notable"
arts = {json.loads(l)["article"] for l in open("/home/user/.cache/oversight_arena/quality.dev.jsonl")}
import re
n = sum(1 for a in arts if re.search(r"\b(is|was|were|are|had|did) not(?=[a-z])", _norm(a)))
print(f"QuALITY dev articles containing '<aux> not' glued to a longer word (e.g. 'was nothing'): {n}/{len(arts)}")
