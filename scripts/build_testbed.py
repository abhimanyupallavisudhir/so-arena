"""Run the testbed's self-run experiments and write their results as committed datasets.

    python scripts/build_testbed.py rlhack --per-family 200      # tabular Q-learning, seconds per experiment
    python scripts/build_testbed.py agents --per-family 60       # LLM-agent experiments (API calls; see --estimate)

Results go to ``src/so_arena/data/samples/testbed_<domain>.jsonl``. Each record holds the full experiment
parameters and the measured outcome, so the dataset is reproducible from its records.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "src" / "so_arena" / "data" / "samples"


def _rlhack_family(args: tuple[str, int, int]) -> list[dict]:
    from so_arena.domains.testbed.rlhack import generate_records

    mode, n, seed = args
    return generate_records(mode, n, seed)


def build_rlhack(per_family: int, seed: int, workers: int) -> Path:
    from so_arena.domains.testbed.rlhack import BUG_MODES

    # each family is split into chunks run in parallel with different seeds
    chunks = [(mode, per_family // workers + 1, seed * 100 + k) for mode in BUG_MODES for k in range(workers)]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_rlhack_family, chunks))
    rows: dict[str, list[dict]] = {}
    for (mode, _, _), recs in zip(chunks, results):
        rows.setdefault(mode, []).extend(recs)
    out = SAMPLES / "testbed_rlhack.jsonl"
    with open(out, "w") as f:
        for mode, recs in rows.items():
            yes = [r for r in recs if r["exploits"] > 3][: (per_family + 1) // 2]
            no = [r for r in recs if r["exploits"] <= 3][: per_family // 2]
            for r in yes + no:
                f.write(json.dumps(r, sort_keys=True) + "\n")
            print(f"rlhack/{mode}: {len(yes)} exploit, {len(no)} not", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("domain", choices=["rlhack", "agents"])
    ap.add_argument("--per-family", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--estimate", action="store_true", help="agents: dry run with simulated models (prices calls)")
    a = ap.parse_args(argv)
    if a.domain == "rlhack":
        print(build_rlhack(a.per_family, a.seed, a.workers))
    else:
        from so_arena.domains.testbed.agents import build

        print(build(a.per_family, a.seed, estimate=a.estimate))


if __name__ == "__main__":
    main()
