"""Release mechanism outputs before ground truth exists; resolve them later.

Forecasters (LLMs) forecast open Manifold questions; a weak judge rates the forecasts (an immediate,
non-proper reward). The judge's rankings are published as a hash-committed release with a static
page; when the questions resolve, `resolve` attaches outcomes and scores both the judge's reward and
the proper log score, showing whether the immediate signal tracked the truth.

    OA_EXPERT_MODEL=... OA_JUDGE_MODEL=... python examples/forecast_release.py release --n 20
    python examples/forecast_release.py resolve        # weeks later
"""

import argparse
import json
from pathlib import Path

import oversight_arena as oa
from oversight_arena.core.task import Task
from oversight_arena.domains.forecasting import ManifoldForecasting
from oversight_arena.mechanisms import Forecast, JudgeRating
from oversight_arena.release import create_release, resolve_release

OUT = Path("runs/forecast_release")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["release", "resolve"])
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    if args.action == "release":
        dom = ManifoldForecasting(status="open", n=args.n)
        strategies = [oa.Strategy(name="calibrated", instructions="Give your honest, calibrated probability."),
                      oa.Strategy(name="bold", instructions="Commit to a confident, decisive forecast.")]
        profiles = [oa.Profile.of(label=s.name, forecaster=s) for s in strategies]
        res = oa.Experiment(dom, Forecast(judge=True, reward=JudgeRating()), oa.llm_agents(), profiles, out=OUT / "run").run()
        rel = create_release(res, OUT / "release", title="Judged forecasts on open Manifold questions",
                             description="A weak judge's ratings, published before resolution.")
        (OUT / "tasks.json").write_text(json.dumps([t.model_dump(mode="json") for t in dom.tasks()]))
        print(rel, "\npublish this root to timestamp the release:", rel.root)
    else:
        tasks = [Task.model_validate(t) for t in json.loads((OUT / "tasks.json").read_text())]
        report = resolve_release(OUT / "release", ManifoldForecasting.resolve(tasks))
        print(json.dumps({k: report[k] for k in report if k != "verification"}, indent=1, default=str)[:3000])
