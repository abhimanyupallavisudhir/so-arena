# Blind baselines on code which_solution items (never reading the task): does surface form predict the reference?
import ast
from so_arena.domains.code import CodeDomain
from collections import Counter
dom = CodeDomain(kind="which_solution", mutants_per_problem=3)
items = dom.load()
res = Counter()
for it in items:
    c = it.context["candidates"]
    n = {k: len(list(ast.walk(ast.parse(v)))) for k, v in c.items()}
    op = it.ground_truth.data["mutation"]["op"]
    if n["A"] != n["B"]:
        g = max(n, key=n.get)
        res[("more_nodes", op, g == it.true_label)] += 1
        res[("more_nodes_all", g == it.true_label)] += 1
    else:
        res[("tie",)] += 1
print("items", len(items))
print({k: v for k, v in sorted(res.items(), key=str)})
# rule: the candidate with more if/else structure is the reference (mutation ops only ever delete branches)
dec = hit = 0
for it in items:
    c = it.context["candidates"]
    nb = {k: sum(isinstance(x, (ast.If, ast.IfExp)) + (isinstance(x, ast.If) and bool(x.orelse)) for x in ast.walk(ast.parse(v))) for k, v in c.items()}
    if nb["A"] != nb["B"]:
        dec += 1; hit += max(nb, key=nb.get) == it.true_label
print(f"branch rule decides {dec}/{len(items)} items, accuracy {hit}/{dec}")
