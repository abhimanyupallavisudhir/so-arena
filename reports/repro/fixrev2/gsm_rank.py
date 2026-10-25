from collections import Counter
from so_arena.domains.qa import GSM8K, _num
items = GSM8K(binary=False, offline=True).load(split="test")
rank, gaps, misc = Counter(), Counter(), Counter()
hits_int = hits_mid = 0; n=0
for it in items:
    vals = {a.label: _num(a.text) for a in it.answers}
    srt = sorted(vals, key=vals.get)
    rank[srt.index(it.true_label)] += 1
    v = sorted(vals.values()); d = [round(b - a, 6) for a, b in zip(v, v[1:])]
    gaps[len(set(d))] += 1
    # blind rule: prefer the option that is an integer / the "roundest"
    def roundness(x): 
        x = abs(x); k = 0
        while x and x % 10 == 0 and k < 6: x //= 10; k += 1
        return (float(x).is_integer(), k)
    g = max(vals, key=lambda l: roundness(vals[l])); hits_int += g == it.true_label
    n += 1
print("n", n, "rank of answer among sorted options:", dict(rank), "distinct gaps per item:", dict(gaps))
print("roundest-option rule:", hits_int / n)
print("example:", [(a.text) for a in items[0].answers], items[0].true_label)
