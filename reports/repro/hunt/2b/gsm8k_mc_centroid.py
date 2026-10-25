# GSM8K multiple choice (binary=False): distractors are all slips *of the correct answer*, so the option that
# the other options are slips of is the correct one. A blind guesser that never reads the question:
from so_arena.domains.qa import GSM8K, gsm8k_wrong_answers, _num
dom = GSM8K(binary=False, offline=True)
items = dom.load(split="test")
def slips(x):  # slips of a candidate answer x (no solution text: only the answer-only kinds)
    return {w for ws in gsm8k_wrong_answers(f"#### {x}", x).values() for w in ws}
hits = 0
for it in items:
    opts = {a.label: a.text for a in it.answers}
    score = {l: sum(o in slips(t) for m, o in opts.items() if m != l) for l, t in opts.items()}
    guess = max(score, key=score.get)
    hits += guess == it.true_label
print(f"n={len(items)}  options/item={len(items[0].answers)}  blind accuracy={hits/len(items):.2f}  chance={1/len(items[0].answers):.2f}")
