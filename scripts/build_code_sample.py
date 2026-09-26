"""Build the bundled MBPP sample of the code domain (``src/so_arena/data/samples/mbpp_sample.jsonl``).

Downloads MBPP "sanitized" (CC-BY-4.0), generates AST mutants of every reference solution in the
test split, keeps those that pass the visible (first) test but fail a hidden one, and writes the
first ``--n`` problems (by task id) that have at least one surviving mutant. Prints mutation
survival statistics per operator.

    python scripts/build_code_sample.py [--n 60] [--out PATH] [--keep 3] [--max-candidates 60]
"""

from __future__ import annotations

import argparse
import json

from so_arena.datasets import download, sample_path, write_jsonl
from so_arena.domains.code import MBPP_SPLITS, MBPP_URL, SAMPLE_NAME, prepare_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=60, help="number of problems to keep")
    ap.add_argument("--out", default=str(sample_path(SAMPLE_NAME)))
    ap.add_argument("--keep", type=int, default=3, help="surviving mutants stored per problem")
    ap.add_argument("--max-candidates", type=int, default=60, help="candidate mutants tried per problem")
    ap.add_argument("--split", default="test", choices=sorted(MBPP_SPLITS))
    args = ap.parse_args()

    rows = json.loads(download(MBPP_URL).read_text())
    lo, hi = MBPP_SPLITS[args.split]
    rows = sorted((r for r in rows if lo <= r["task_id"] <= hi), key=lambda r: r["task_id"])
    prepared, stats = prepare_rows(rows, max_candidates=args.max_candidates, keep=args.keep)
    chosen = [r for r in prepared if r["mutants"]][: args.n]
    write_jsonl(args.out, chosen)

    print(f"scanned {stats['problems']} {args.split}-split problems; {stats['usable']} usable (reference passes "
          f"its tests); {stats['with_survivor']} with >= 1 surviving mutant")
    c, v, s = stats["candidates"], stats["pass_visible"], stats["survivors"]
    print(f"candidate mutants {c}; pass the visible test {v} ({v / c:.0%}); survive (also fail a hidden test) "
          f"{s} ({s / c:.0%} of candidates, {s / max(v, 1):.0%} of visible-passing)")
    for op, d in stats["per_op"].items():
        print(f"  {op:7s} candidates {d['candidates']:5d}  pass-visible {d['pass_visible']:5d}  "
              f"survivors {d['survivors']:5d} ({d['survivors'] / max(d['candidates'], 1):.0%})")
    ids = [r["task_id"] for r in chosen]
    print(f"wrote {len(chosen)} problems (task ids {ids[0]}-{ids[-1]}), "
          f"{sum(len(r['mutants']) for r in chosen)} stored mutants -> {args.out}")


if __name__ == "__main__":
    main()
