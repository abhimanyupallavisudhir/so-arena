from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .demo import demo
from .reporting import write_report
from .scoring import ForecastScorer
from .storage import Store


def main():
    parser = argparse.ArgumentParser(description="OversightArena experiment tools")
    commands = parser.add_subparsers(dest="command", required=True)
    example = commands.add_parser("demo", help="Run deterministic examples without API keys")
    example.add_argument("--output", type=Path, default=Path("outputs/demo"))
    example.add_argument("--plots", action="store_true", help="Requires the plots extra")
    report = commands.add_parser("report", help="Render stored records")
    report.add_argument("store", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--mechanism-only", action="store_true")
    resolve = commands.add_parser("resolve", help="Append forecast resolutions to existing runs")
    resolve.add_argument("store", type=Path)
    resolve.add_argument("resolutions", type=Path, help="JSON object: task ID -> boolean outcome")
    resolve.add_argument("--role", required=True)
    resolve.add_argument("--version", required=True)
    args = parser.parse_args()
    if args.command == "demo":
        summary = asyncio.run(demo(args.output, plots=args.plots))
        print(summary["note"])
        print(f"Report: {args.output / 'report.html'}")
        for mechanism in ("consultancy", "debate"):
            print(f"{mechanism} paired ASD: {summary[mechanism]['estimate']['value']:.3f}")
    elif args.command == "report":
        store = Store(args.store)
        write_report(
            store.runs(),
            args.output,
            evaluations=None if args.mechanism_only else store.evaluations(),
        )
        print(args.output)
    elif args.command == "resolve":
        store = Store(args.store)
        resolutions = json.loads(args.resolutions.read_text())
        if not isinstance(resolutions, dict) or any(
            type(v) is not bool for v in resolutions.values()
        ):
            parser.error("Resolutions must map task IDs to boolean outcomes")
        selected = [r for r in store.runs() if r.task.id in resolutions]
        if set(resolutions) - {r.task.id for r in selected}:
            parser.error("Some resolution task IDs have no stored runs")
        if any(args.role not in {s.id for s in r.roles} for r in selected):
            parser.error("Role is absent from a selected run")
        scorer = ForecastScorer(resolutions, (args.role,), args.version)
        for run in selected:
            store.save_evaluation(scorer(run))
        print(f"Appended {len(selected)} evaluations; mechanism records unchanged.")


if __name__ == "__main__":
    main()
