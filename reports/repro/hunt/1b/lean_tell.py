# MiniF2F faithfulness: the "hypothesis dropped" perturbation leaves a gap in the hypothesis
# numbering (h₀ h₂ without h₁), so the NO items are recognisable from the Lean text alone.
import re
from oversight_arena.domains.lean import MiniF2F
ts = MiniF2F().tasks()
SUB = "₀₁₂₃₄₅₆₇₈₉"
def gap(stmt):
    idx = sorted({int("".join(str(SUB.index(c)) for c in m)) for m in re.findall(r"\(h([₀-₉]+)\s*:", stmt)})
    return bool(idx) and idx != list(range(idx[0], idx[0] + len(idx))) or (bool(idx) and idx[0] != 0)
rows = [(t.gt["faithful"], t.gt["perturbation"], gap(t.resources["statement"])) for t in ts]
n = len(rows); nf = sum(1 for f, _, _ in rows if not f)
drop = [g for f, p, g in rows if p == "hypothesis dropped"]
print(f"items={n} unfaithful={nf} hypothesis-dropped={len(drop)} of which numbering gap visible={sum(drop)}")
print("faithful items with a numbering gap (false alarms):", sum(1 for f, _, g in rows if f and g))
# blind rule: answer NO iff numbering gap, else YES
acc = sum(1 for f, _, g in rows if (not g) == f) / n
print(f"blind 'gap => NO else YES' accuracy = {acc:.3f} (base rate YES = {1 - nf / n:.3f})")
