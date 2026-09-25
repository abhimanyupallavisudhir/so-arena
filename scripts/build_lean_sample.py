"""Regenerate the bundled Lean sample (src/so_arena/data/samples/lean_minif2f_sample.jsonl).

    python scripts/build_lean_sample.py [--n 40] [--out PATH]

Downloads the test split of google-deepmind/miniF2F at the pinned commit (Apache-2.0), keeps the
theorems that carry an informal statement, parse, and admit at least one mutation, and takes the
first ``--n`` in a deterministic pseudo-random order (stable hash of the theorem name). Rows are
stored as parsed from the source (``name``, ``split``, ``line``, ``informal``, ``formal``); mutants
are generated at load time, so changing the mutation operators never requires a new sample.
"""

from __future__ import annotations

import argparse
import collections
import sys

from so_arena.core.policy import stable_hash
from so_arena.datasets import sample_path, write_jsonl
from so_arena.domains.lean import SAMPLE_FILE, fetch_minif2f, mutation_candidates, parse_statement, strip_answer_markers


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(sample_path(SAMPLE_FILE)))
    args = ap.parse_args(argv)

    rows = sorted(fetch_minif2f("test"), key=lambda r: stable_hash("lean-sample", r["name"]))
    keep, ops = [], collections.Counter()
    for row in rows:
        try:
            cands = mutation_candidates(parse_statement(strip_answer_markers(row["formal"])), informal=row["informal"])
        except ValueError:
            continue
        if not cands:
            continue
        keep.append(row)
        ops.update(cands.keys())
        if len(keep) >= args.n:
            break
    keep.sort(key=lambda r: r["line"])  # dataset order
    write_jsonl(args.out, keep)
    print(f"wrote {len(keep)} problems to {args.out}; problems admitting each operator: {dict(ops)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
