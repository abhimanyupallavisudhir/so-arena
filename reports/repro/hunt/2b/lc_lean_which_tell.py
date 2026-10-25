"""which_formalization: can a blind rule (pick the longer / the one with more hypotheses) beat 50%?"""
import collections
from so_arena.domains.lean import LeanFaithfulnessDomain, parse_statement
for seed in (0, 1, 2):
    items = LeanFaithfulnessDomain("which_formalization", offline=True, seed=seed).load()
    longer = hyps = 0; ties = 0; ops = collections.Counter()
    for it in items:
        c = it.context["candidates"]; g = it.ground_truth.correct; o = "B" if g == "A" else "A"
        if len(c[g]) == len(c[o]): ties += 1
        pick = max("AB", key=lambda k: (len(c[k]), k == "A"))
        longer += pick == g
        nh = {k: len(parse_statement(c[k]).hypotheses) for k in "AB"}
        pick2 = "A" if nh["A"] > nh["B"] else "B" if nh["B"] > nh["A"] else pick
        hyps += pick2 == g
        ops[it.metadata["mutation"]] += 1
    print(f"seed {seed}: n={len(items)} 'pick longer' acc={longer/len(items):.2f} 'more hyps, else longer' acc={hyps/len(items):.2f} ties={ties} ops={dict(ops)}")

print("--- rule: if hypothesis counts differ, the one with MORE hypotheses is faithful")
from so_arena.domains.lean import mutation_candidates, MUTATION_OPERATORS
for seed in range(5):
    items = LeanFaithfulnessDomain("which_formalization", offline=True, seed=seed).load()
    hit = n = 0
    for it in items:
        c = it.context["candidates"]; g = it.ground_truth.correct
        nh = {k: len(parse_statement(c[k]).hypotheses) + len(parse_statement(c[k]).binders) for k in "AB"}
        if nh["A"] != nh["B"]:
            n += 1; hit += max("AB", key=nh.get) == g
    print(f"seed {seed}: decided {n}/{len(items)} items, accuracy {hit}/{n}")
