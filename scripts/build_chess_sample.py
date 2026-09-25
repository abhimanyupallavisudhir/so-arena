"""Regenerate the bundled chess sample (src/so_arena/data/samples/chess_puzzles_sample.jsonl).

    python scripts/build_chess_sample.py [--per-bin 30] [--max-bytes 1048576] [--out PATH]

Downloads a prefix of the Lichess puzzle database (CC0; ~20k puzzles per MB), keeps well-established
puzzles (rating deviation <= 90, >= 100 plays, popularity >= 50), stratifies them into 200-point
rating bins from 800 to 2800 and, in a deterministic pseudo-random order within each bin, analyses
them with Stockfish (Threads=1, Hash=16; see so_arena.domains.chess.analyse_puzzle) until every bin
has ``--per-bin`` puzzles with a confirmed best move and a clearly worse plausible alternative.
Takes roughly 3-4 s per puzzle on one core.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time

from so_arena.core.policy import stable_hash
from so_arena.datasets import sample_path, write_jsonl
from so_arena.domains.chess import SAMPLE_FILE, analyse_puzzle, fetch_lichess_puzzles, shared_engine


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--per-bin", type=int, default=30)
    ap.add_argument("--lo", type=int, default=800)
    ap.add_argument("--hi", type=int, default=2800)
    ap.add_argument("--bin-width", type=int, default=200)
    ap.add_argument("--max-bytes", type=int, default=1 << 20)
    ap.add_argument("--deep-nodes", type=int, default=1_000_000)
    ap.add_argument("--table-depth", type=int, default=12)
    ap.add_argument("--min-gap-cp", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(sample_path(SAMPLE_FILE)))
    args = ap.parse_args(argv)

    puzzles = fetch_lichess_puzzles(None, args.max_bytes)
    ok = [p for p in puzzles if p["rating_deviation"] <= 90 and p["nb_plays"] >= 100 and p["popularity"] >= 50
          and args.lo <= p["rating"] <= args.hi]
    bins: dict[int, list[dict]] = collections.defaultdict(list)
    for p in ok:
        bins[min((p["rating"] - args.lo) // args.bin_width, (args.hi - args.lo) // args.bin_width - 1)].append(p)
    print(f"{len(puzzles)} puzzles in prefix, {len(ok)} pass filters; per bin: "
          f"{ {args.lo + b * args.bin_width: len(v) for b, v in sorted(bins.items())} }", file=sys.stderr)

    engine = shared_engine()
    rows, skipped, t0 = [], 0, time.time()
    for b, cands in sorted(bins.items()):
        cands.sort(key=lambda p: stable_hash(args.seed, p["puzzle_id"]))
        kept = 0
        for p in cands:
            if kept >= args.per_bin:
                break
            try:
                a = analyse_puzzle(p, engine, deep_nodes=args.deep_nodes, table_depth=args.table_depth,
                                   min_gap_cp=args.min_gap_cp)
            except ValueError:
                a = None
            if a is None:
                skipped += 1
                continue
            rows.append({**p, "analysis": a})
            kept += 1
        print(f"bin {args.lo + b * args.bin_width}: kept {kept} ({time.time() - t0:.0f}s, {skipped} skipped so far)",
              file=sys.stderr)
    rows.sort(key=lambda r: r["puzzle_id"])
    write_jsonl(args.out, rows)
    sources = collections.Counter(r["analysis"]["alternative"]["source"] for r in rows)
    print(f"wrote {len(rows)} puzzles to {args.out} ({skipped} skipped; alternatives: {dict(sources)}; "
          f"engine: {engine.name}; {engine.searches} searches in {time.time() - t0:.0f}s)", file=sys.stderr)
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
