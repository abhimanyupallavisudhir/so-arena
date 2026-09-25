"""Do original and mutant renders differ in surface features a blind judge could use?"""
import re, collections
from so_arena.domains.lean import LeanFaithfulnessDomain, parse_statement, tokenize
d = LeanFaithfulnessDomain(offline=True)
items = d.load()
print("items", len(items), "order of truths:", "".join("Y" if i.ground_truth.correct == "yes" else "N" for i in items)[:20], "...")
feats = collections.defaultdict(collections.Counter)
def f_nospace_ops(t):  # operators without surrounding spaces
    return bool(re.search(r"\S[-+*/](?=\S)", t.split(":=")[0]))
for it in items:
    t = it.context["candidates"]["statement"]; y = it.ground_truth.correct
    feats["dbl_paren_or_neg_lit"][(bool(re.search(r"\(\(|= -\d", t)), y)] += 1
    feats["nospace_op"][(f_nospace_ops(t), y)] += 1
    feats["len>median"][(len(t) > 200, y)] += 1
for k, c in feats.items(): print(k, dict(c))
# per-problem: which arm has more hypotheses / is shorter
short = sum(1 for a, b in zip(items[::2], items[1::2]) if len(a.question) < len(b.question))
print("pairs where original shorter than mutant:", short, "/", len(items)//2)
# odd limit => imbalance
print("limit=5 truths:", [i.ground_truth.correct for i in d.load(limit=5)])
