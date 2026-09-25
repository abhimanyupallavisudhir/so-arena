from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .core import Task
from .demo import run_demo
from .domains import Forecast
from .reporting import write_report
from .runtime import evaluate
from .storage import RunStore, public_result


def main() -> None:
    parser = argparse.ArgumentParser(prog="oversight-arena")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser(
        "demo", help="Run offline SQL, search, RL, swarm and forecast examples"
    )
    demo.add_argument("--output", default="runs/demo")
    demo.add_argument("--plots", action="store_true")
    report = commands.add_parser("report", help="Render stored mechanism rewards and evaluations")
    report.add_argument("database")
    report.add_argument("--output", default="report.html")
    export = commands.add_parser("export", help="Export a public result projection")
    export.add_argument("database")
    export.add_argument("--output", required=True)
    resolve = commands.add_parser("resolve-forecast", help="Append a forecast resolution revision")
    resolve.add_argument("database")
    resolve.add_argument("episode_id")
    resolve.add_argument("--role", action="append", required=True)
    resolve.add_argument("--outcome", type=int, choices=(0, 1), required=True)
    resolve.add_argument("--source", required=True)
    resolve.add_argument("--resolved-at", required=True, help="Resolution timestamp (ISO 8601)")
    args = parser.parse_args()
    if args.command == "demo":
        summary = asyncio.run(run_demo(args.output, plots=args.plots))
        print(json.dumps(summary, indent=2))
    elif args.command == "report":
        with RunStore(args.database) as store:
            print(write_report(store.records(), args.output))
    elif args.command == "export":
        with RunStore(args.database) as store:
            Path(args.output).write_text(
                json.dumps([public_result(r) for r in store.records()], indent=2)
            )
        print(args.output)
    elif args.command == "resolve-forecast":
        from datetime import datetime

        datetime.fromisoformat(args.resolved_at.replace("Z", "+00:00"))
        with RunStore(args.database) as store:
            episode = store.get(args.episode_id)
            task = Task(
                episode.task,
                {"outcome": args.outcome, "source": args.source, "resolved_at": args.resolved_at},
            )
            updated = asyncio.run(evaluate(task, episode, Forecast(tuple(args.role))))
            store.save(updated)
        print(updated.id)


if __name__ == "__main__":
    main()
