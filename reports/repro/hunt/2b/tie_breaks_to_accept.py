# HiddenTests / SQLAnswerScorer decide accept/reject with max(sub, key=sub.get): a 50/50 review (e.g. the
# parse-failure fallback) counts as ACCEPT, unlike JudgeCorrectness (ties = coin flip, 0.5).
import asyncio
from so_arena.core.mechanism import Episode, Outcome
from so_arena.core.ground_truth import JudgeCorrectness
from so_arena.domains.code import CodeDomain, HiddenTests
from so_arena.domains.sql import SQLAnswerScorer, SQLDomain
from so_arena.core.mechanism import PlayerRecord
from so_arena.core.game import Turn
item = CodeDomain(kind="implement").load(limit=1)[0]
ref = item.ground_truth.data["reference"]
for code, tag in ((ref, "correct work"), ("def nope(): pass", "wrong work")):
    ep = Episode(id="e", item_id=item.id, mechanism="m", players={"worker": PlayerRecord(policy_id="w")},
                 turns=[Turn(index=0, slot=0, role="worker", phase="work", text=f"```python\n{code}\n```")],
                 outcome=Outcome(decision="accept", probs={"accept": 0.5, "reject": 0.5}))
    ht = asyncio.run(HiddenTests().score(ep, item))
    ep.ground_truth = ht
    jc = asyncio.run(JudgeCorrectness().score(ep, item))
    print(f"{tag}: HiddenTests judge_correct={ht['judge_correct']} outcome_value={ht['outcome_value']} | "
          f"JudgeCorrectness judge_correct={jc.get('judge_correct')}")
