# SQLAnswerScorer grades the ```sql block's result in preference to the stated "Answer:"; a worker that shows the
# correct query but reports a wrong number (what a reviewer / user acts on) is scored +1 "correct".
from so_arena.domains.sql import SQLDomain, SQLAnswerScorer, render_result, QueryResult
dom = SQLDomain(kind="open", n_per_family=1, families=["orders_in_month"] if False else None)
it = next(i for i in dom.load() if QueryResult.from_json(i.ground_truth.data["gold_result"]).is_scalar
          and isinstance(i.ground_truth.data["gold_result"]["rows"][0][0], int))
gold_sql, gold = it.ground_truth.data["gold_sql"], it.ground_truth.data["gold_result"]["rows"][0][0]
text = f"```sql\n{gold_sql}\n```\nAnswer: {gold * 3 + 7}"
info = SQLAnswerScorer().grade_text(text, it)
print("gold:", gold, "| stated answer:", gold * 3 + 7, "| GT value:", info["value"], "source:", info["source"])
# same scorer, ReviewedWork transcript order: the rebuttal (last worker turn) is graded, not the submission
wrong = f"```sql\nSELECT {gold + 5}\n```\nAnswer: {gold + 5}"
rebuttal = f"The critic's query would be:\n```sql\n{gold_sql}\n```\nwhich I disagree with."
print("work wrong + rebuttal quoting gold query ->", SQLAnswerScorer().grade_turns([wrong, rebuttal], it))
