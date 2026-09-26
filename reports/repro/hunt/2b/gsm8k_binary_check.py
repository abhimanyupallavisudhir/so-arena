from so_arena.domains.qa import GSM8K, gsm8k_wrong_answers
items = GSM8K(binary=True, offline=True).load(split="test")
def slips(x): return {w for ws in gsm8k_wrong_answers(f"#### {x}", x).values() for w in ws}
h = {"slip-asym": 0, "larger": 0, "first": 0}
for it in items:
    a, b = it.answers
    sa, sb = b.text in slips(a.text), a.text in slips(b.text)
    g = a.label if sa and not sb else b.label if sb and not sa else a.label
    h["slip-asym"] += g == it.true_label
    h["larger"] += (a.label if float(a.text) > float(b.text) else b.label) == it.true_label
    h["first"] += "A" == it.true_label
print(len(items), {k: round(v/len(items), 2) for k, v in h.items()})
