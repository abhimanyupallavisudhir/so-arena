"""``oversight-arena`` command line."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
@click.version_option(package_name="oversight-arena")
def main() -> None:
    """OversightArena: experiments on scalable-oversight mechanisms."""


@main.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--limit", type=int, default=None, help="Limit the number of tasks.")
@click.option("--out", type=click.Path(), default=None, help="Output directory (overrides config).")
@click.option("--dry-run", is_flag=True, help="Only print the plan.")
def run(config: str, limit: int | None, out: str | None, dry_run: bool) -> None:
    """Run an experiment from a YAML/JSON config."""
    from .analysis.ic import ic_report
    from .analysis.report import html_report
    from .config import experiment_from_config, load_config

    cfg = load_config(config)
    exp = experiment_from_config(cfg, limit=limit, out=out or cfg.get("out") or f"runs/{Path(config).stem}")
    plan = exp.plan()
    click.echo(f"{exp.name}: {len(plan)} episodes over {len(exp.tasks())} tasks and {len(exp.mechanisms)} mechanism(s)")
    if dry_run:
        return
    res = exp.run()
    gt = (cfg.get("analysis") or {}).get("gt", "correct")
    df = res.df()
    if f"gt_{gt}" in df:
        t = ic_report(df[df["trainable"]], gt=gt)
        click.echo(t.round(3).to_string())
        t.to_csv(Path(exp.out) / "ic.csv", index=False)
    p = html_report(res, Path(exp.out) / "report.html", title=exp.name, gt=gt)
    click.echo(f"report: {p}")


@main.command()
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--gt", default="correct")
@click.option("--title", default=None)
def report(run_dir: str, gt: str, title: str | None) -> None:
    """(Re)build the HTML report for a run directory."""
    from .analysis.report import html_report
    from .experiment.results import Results

    res = Results.load(run_dir)
    p = html_report(res, Path(run_dir) / "report.html", title=title or Path(run_dir).name, gt=gt)
    click.echo(str(p))


@main.command("list")
@click.argument("kind", required=False)
def list_cmd(kind: str | None) -> None:
    """List registered components (domain, mechanism, reward, agent, gt, profiles, channel)."""
    from .registry import list_components

    for k, names in list_components(kind).items():
        click.echo(f"{k}: {', '.join(names)}")


@main.group()
def release() -> None:
    """Publish mechanism-only results now; resolve with ground truth later."""


@release.command("create")
@click.argument("run_dir", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path())
@click.option("--title", default="Mechanism results")
@click.option("--description", default="")
@click.option("--sealed", is_flag=True, help="Publish only salted commitments; reveal later.")
@click.option("--no-transcripts", is_flag=True)
def release_create(run_dir: str, out_dir: str, title: str, description: str, sealed: bool, no_transcripts: bool) -> None:
    from .experiment.results import Results
    from .release import create_release

    rel = create_release(Results.load(run_dir), out_dir, title=title, description=description, sealed=sealed,
                         transcripts=not no_transcripts)
    click.echo(f"{rel.path}  root={rel.root}")


@release.command("verify")
@click.argument("release_dir", type=click.Path(exists=True))
def release_verify(release_dir: str) -> None:
    from .release import verify_release

    click.echo(json.dumps(verify_release(release_dir), indent=1))


@release.command("resolve")
@click.argument("release_dir", type=click.Path(exists=True))
@click.option("--tasks", "tasks_path", type=click.Path(exists=True), default=None, help="JSON list of resolved tasks.")
@click.option("--manifold", is_flag=True, help="Re-fetch Manifold markets for resolution.")
def release_resolve(release_dir: str, tasks_path: str | None, manifold: bool) -> None:
    from .core.task import Task
    from .domains.forecasting import ManifoldForecasting, forecast_task
    from .release import resolve_release

    tasks = json.loads((Path(release_dir) / "tasks.json").read_text())
    if tasks_path:
        resolved = [Task.model_validate(t) for t in json.loads(Path(tasks_path).read_text())]
    elif manifold:
        stubs = [forecast_task(t["metadata"].get("qid", t["id"]), t["question"], None, source="manifold") for t in tasks]
        resolved = ManifoldForecasting.resolve(stubs)
    else:
        raise click.UsageError("give --tasks or --manifold")
    rep = resolve_release(release_dir, resolved)
    click.echo(json.dumps({k: rep[k] for k in rep if k != "verification"}, indent=1, default=str)[:4000])


@main.group()
def theory() -> None:
    """Analytic companions (see docs/theory.md)."""


@theory.command("swarm")
@click.option("--n", default=3, help="Workers.")
@click.option("--stake", "g", default=0.3, help="Each worker's payoff gain if the violation goes undetected.")
@click.option("--bounty", "b", default=0.1)
@click.option("--penalty", "P", default=0.3, help="Offender penalty on detection.")
@click.option("--audit", "a", default=0.0, help="Random audit probability.")
@click.option("--misprision", "c", default=0.0, help="Penalty for silent observers when detected.")
@click.option("--observe", "o", default=0.8, help="Probability each other worker observes the violation.")
@click.option("--plot", type=click.Path(), default=None, help="Save the phase diagram PNG here.")
def theory_swarm(n: int, g: float, b: float, P: float, a: float, c: float, o: float, plot: str | None) -> None:
    """Equilibria of the whistleblowing game."""
    from .theory import swarm_game as sg

    p = sg.SwarmParams(n=n, g=g, b=b, P=P, a=a, c=c, o=o)
    click.echo(json.dumps(sg.summary(p), indent=1))
    if plot:
        from .analysis import plots as P_

        fig, _ = P_.regime_map(sg.phase_diagram(n=n, g=g, P=P, c=c, o=o), "bounty_over_stake", "audit",
                               contour="basin_report", title="When does reporting pay?",
                               xlabel="bounty ÷ each worker's stake", ylabel="audit probability")
        click.echo(str(P_.save(fig, plot)))


@main.command()
@click.argument("name", type=click.Choice(["hiddenbits", "chess", "swarm", "bon", "optimize", "monitoring", "release", "all"]))
@click.option("--out", type=click.Path(), default="runs/demos")
def demo(name: str, out: str) -> None:
    """Run a built-in demo that needs no API keys (figures + HTML reports)."""
    from . import demos

    for p in demos.run(name, Path(out)):
        click.echo(str(p))


if __name__ == "__main__":
    main()
