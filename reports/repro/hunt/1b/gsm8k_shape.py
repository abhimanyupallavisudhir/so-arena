# GSM8K (SOlib) options: does the *form* of the option text reveal the correct one?
# GSM8K gold answers are integers; check how often the distractor is a non-integer / negative.
from oversight_arena.domains.math import GSM8K
ts = GSM8K().tasks()
def nonint(s):
    try: return float(s) != int(float(s))
    except Exception: return True
n = len(ts); cor_ni = inc_ni = 0; neg_inc = neg_cor = 0; wins = 0
for t in ts:
    c = next(o for o in t.options if o.value > 0).text; w = next(o for o in t.options if o.value < 0).text
    cor_ni += nonint(c); inc_ni += nonint(w)
    neg_cor += c.startswith("-"); neg_inc += w.startswith("-")
    # blind rule: prefer the integer / non-negative option, else option A
    score = lambda s: (not nonint(s)) + (not s.startswith("-"))
    pick = max(t.options, key=lambda o: (score(o.text), o.id == "A"))
    wins += pick.value > 0
print(f"tasks={n}  correct non-integer={cor_ni}  distractor non-integer={inc_ni}  "
      f"correct negative={neg_cor}  distractor negative={neg_inc}")
print(f"blind 'pick the integer, non-negative option' accuracy = {wins/n:.3f}")
