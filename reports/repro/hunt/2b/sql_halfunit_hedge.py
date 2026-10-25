# SQLAnswerScorer compares with decimals (default 0 for counts) but without stated=True, so the tolerance is
# +-0.5(+1e-9): one stated answer "42.5" is graded correct whether the gold count is 42 or 43.
from so_arena.domains.sql import SQLAnswerScorer, QueryResult, values_match
from so_arena.core.items import TaskItem, GroundTruth
sc = SQLAnswerScorer()
for gold in (42, 43, 44):
    it = TaskItem(id="q", question="How many orders?", ground_truth=GroundTruth(data={
        "gold_result": QueryResult(columns=["n"], rows=[(gold,)]).to_json(), "ordered": False, "decimals": 0}))
    print(f"gold={gold}: 'Answer: 42.5' ->", sc.grade_text("Answer: 42.5", it)["value"],
          "| verifier-style check (stated=True):", values_match("42.5", gold, decimals=0, stated=True))
