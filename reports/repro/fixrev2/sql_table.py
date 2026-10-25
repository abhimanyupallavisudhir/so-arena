from so_arena.domains.sql import SQLDomain, SQLAnswerScorer, QueryResult, stated_answer, render_result
dom = SQLDomain(kind="open", n_per_family=1)
sc = SQLAnswerScorer()
it = next(i for i in dom.load() if "dept_headcount" in i.id)
g = QueryResult.from_json(it.ground_truth.data["gold_result"]); q = it.ground_truth.data["gold_sql"]
rows = g.rows
forms = {
 "bullets a | b": "Answer:\n" + "\n".join(f"- {a} | {b}" for a, b in rows),
 "lines a, b": "Answer:\n" + "\n".join(f"{a}, {b}" for a, b in rows),
 "inline a: b": "Answer: " + "; ".join(f"{a}: {b}" for a, b in rows),
 "inline (a, b)": "Answer: " + ", ".join(f"({a}, {b})" for a, b in rows),
 "markdown table": "Answer:\n| department | active_employees |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows),
 "render_result": "Answer:\n" + render_result(QueryResult(columns=g.columns, rows=rows)),
}
for k, f in forms.items():
    wq = "```sql\n" + q + "\n```\n" + f
    print(k.ljust(16), "with query:", sc.grade_text(wq, it)['value'], " text only:", sc.grade_text(f, it)['value'], " stated=", str(stated_answer(f, g))[:70])
print("---")
for k, f in {"bullets a: b": "Answer:\n" + "\n".join(f"- {a}: {b}" for a, b in rows),
             "star a, b": "Answer:\n" + "\n".join(f"* {a}, {b}" for a, b in rows),
             "numbered": "Answer:\n" + "\n".join(f"{i}. {a} - {b}" for i, (a, b) in enumerate(rows, 1)),
             "bold md table": "Answer:\n| **department** | **active_employees** |\n|:--|--:|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)}.items():
    wq = "```sql\n" + q + "\n```\n" + f
    print(k.ljust(16), "with query:", sc.grade_text(wq, it)['value'], " stated=", str(stated_answer(f, g))[:60])
