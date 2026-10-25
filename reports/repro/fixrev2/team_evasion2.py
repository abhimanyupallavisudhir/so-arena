import ast, warnings; warnings.filterwarnings("ignore")
from so_arena.domains.code import CodeDomain, hack_solution, audit_contribution, hardcoded_literals
dom = CodeDomain(kind="team", n_parts=2)
item = dom.load(limit=1)[0]
parts = item.ground_truth.data["parts"]; p1, p2 = parts; setup = item.ground_truth.data["setup"]
hack = hack_solution(p1["entry_point"], p1["visible_tests"])
print(hack[:400])
class Obf(ast.NodeTransformer):
    def visit_Constant(self, n):
        if isinstance(n.value, int) and not isinstance(n.value, bool):
            return ast.parse(f"int({str(n.value)[::-1]!r}[::-1])", mode="eval").body
        return n
obf = ast.unparse(ast.fix_missing_locations(Obf().visit(ast.parse(hack))))
print(obf[:300])
a = audit_contribution(obf, parts, setup, team=p2["reference"])
print("string-encoded literals:", a)
