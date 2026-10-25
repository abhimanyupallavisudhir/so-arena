from so_arena.domains.sql import SQLDomain, SQLAnswerScorer, QueryResult
dom = SQLDomain(kind="open", n_per_family=1)
items = dom.load()
sc = SQLAnswerScorer()
shown = 0
for it in items:
    g = QueryResult.from_json(it.ground_truth.data["gold_result"])
    q = it.ground_truth.data["gold_sql"]
    rows = g.rows
    if g.is_scalar:
        v = rows[0][0]
        forms = [f"Answer: {v}", f"Answer: {v} orders", f"Answer: The total is {v}.", f"Answer: **{v}**",
                 f"The answer is {v}.", f"Answer: {v:,}" if isinstance(v, int) else f"Answer: {v}",
                 f"Answer: approximately {v}", f"Answer: {v} (as of 2024-03-31)"]
    else:
        flat = ", ".join(str(r[0]) for r in rows)
        forms = [f"Answer: {flat}", "Answer:\n" + "\n".join("- " + " | ".join(map(str, r)) for r in rows)]
    res = [(f[:40], sc.grade_text(f"```sql\n{q}\n```\n{f}", it)["value"], sc.grade_text(f, it)["value"]) for f in forms]
    bad = [r for r in res if r[1] < 1 or r[2] < 1]
    if bad:
        print(it.id, "gold:", str(rows)[:80], "cols", g.columns)
        for r in bad: print("   ", r)
        shown += 1
    if shown > 8: break
print("items", len(items))
