"""Command-line interface: ``so-arena <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _cmd_run(a: argparse.Namespace) -> int:
    from so_arena.config import configure
    from so_arena.spec import load_spec, run_spec, usage_summary

    if a.simulate:
        configure(simulate=True)
    spec = load_spec(a.spec)
    run_dir = run_spec(spec, out=a.out, limit=a.limit)
    from so_arena.core.store import RunStore

    eps = RunStore(run_dir).episodes()
    trees = run_dir / "trees.jsonl"
    if trees.exists():
        n = sum(1 for line in trees.read_text().splitlines() if line.strip())
        print(f"{n} sampled game trees -> {run_dir} (grid_*.csv holds the optimization curves)")
    else:
        print(f"{len(eps)} episodes -> {run_dir}")
    if (run_dir / "report.html").exists():
        print(f"report: {run_dir / 'report.html'}")
    if a.simulate:
        print(usage_summary(eps).to_string(index=False))
    return 0


def _cmd_estimate(a: argparse.Namespace) -> int:
    """Dry run with simulated models: estimates tokens and cost without calling any API."""
    from so_arena.config import configure
    from so_arena.core.store import RunStore
    from so_arena.spec import load_spec, run_spec, usage_summary

    configure(simulate=True)
    spec = load_spec(a.spec)
    spec.report = False
    out = Path(a.out or f"runs/_estimate_{spec.name}")
    if out.exists():
        import shutil

        shutil.rmtree(out)
    run_dir = run_spec(spec, out=out, limit=a.limit)
    eps = RunStore(run_dir).episodes()
    summary = usage_summary(eps)
    print(f"Simulated {len(eps)} episodes (no API calls). Estimated usage:")
    print(summary.to_string(index=False))
    unpriced = [m for m in summary["model"].dropna().unique() if m and m.startswith("sim/")]
    if unpriced:
        from so_arena.models.registry import get_spec

        missing = [m for m in unpriced if get_spec(m.removeprefix("sim/")) is None]
        if missing:
            print(f"(no price known for: {', '.join(missing)} - add them to the model registry)")
    return 0


def _cmd_report(a: argparse.Namespace) -> int:
    from so_arena.analysis.report import build_report
    from so_arena.core.store import RunStore

    store = RunStore(a.run_dir)
    out = Path(a.out or store.path / "report.html")
    build_report(store.episodes(), out, title=a.title or store.path.name, hide_ground_truth=a.hide_ground_truth)
    print(out)
    return 0


def _cmd_release(a: argparse.Namespace) -> int:
    from so_arena.release import release_run

    man = release_run(a.run_dir, a.out, title=a.title or Path(a.run_dir).name, notes=a.notes or "")
    print(f"released {man.n_episodes} episodes on {man.n_items} items -> {a.out}")
    print(f"commitment digest: {man.digest}")
    return 0


def _cmd_verify(a: argparse.Namespace) -> int:
    from so_arena.release import verify

    ok = verify(a.release_dir)
    print("OK: release matches its manifest" if ok else "MISMATCH: release files were modified")
    return 0 if ok else 1


def _cmd_resolve(a: argparse.Namespace) -> int:
    from so_arena.release import resolve

    reward = None
    if a.market:
        from so_arena.mechanisms import MarketScoringReward

        reward = MarketScoringReward()
    if a.truth:
        p = Path(a.truth)
        if p.suffix == ".jsonl":
            truth = {}
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    truth[r["item_id"]] = r.get("answer", r.get("resolution"))
        else:
            truth = json.loads(p.read_text())
    elif a.domain:
        from so_arena.domains import get_domain

        dom = get_domain(a.domain)
        if not hasattr(dom, "resolve"):
            print(f"domain {a.domain} cannot fetch resolutions", file=sys.stderr)
            return 2
        truth = lambda items: {k: v for k, v in dom.resolve(items).items() if v is not None}  # noqa: E731
    else:
        print("pass --truth FILE or --domain NAME", file=sys.stderr)
        return 2
    res = resolve(a.release_dir, truth, out_dir=a.out, reward_rule=reward)
    print(f"resolved {res.n_resolved} episodes ({res.n_unresolved} still unresolved) for release {res.release_digest[:12]}")
    return 0


def _cmd_list(a: argparse.Namespace) -> int:
    what = a.what
    if what in ("mechanisms", "all"):
        from so_arena.mechanisms import MECHANISMS

        print("mechanisms:", ", ".join(sorted(MECHANISMS)))
    if what in ("domains", "all"):
        from so_arena.domains import list_domains

        print("domains:", ", ".join(list_domains()))
    if what in ("models", "all"):
        from so_arena.models import all_specs

        for s in all_specs():
            price = (f"${s.price_input_per_mtok}/${s.price_output_per_mtok} per Mtok"
                     if s.price_input_per_mtok is not None else "no price")
            print(f"  {s.name:45s} {price}")
    return 0


def _cmd_demo(a: argparse.Namespace) -> int:
    from so_arena import demos

    fns = {"asd": demos.demo_asd, "optimization": demos.demo_optimization, "swarm": demos.demo_swarm}
    names = list(fns) if a.name == "all" else [a.name]
    for n in names:
        out = fns[n](Path(a.out) / n)
        print(f"{n}: {out / 'report.html'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="so-arena", description="OversightArena: experiments on scalable-oversight mechanisms")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment spec (YAML/JSON)")
    r.add_argument("spec")
    r.add_argument("--out")
    r.add_argument("--limit", type=int)
    r.add_argument("--simulate", action="store_true", help="replace models by simulated ones (no API calls)")
    r.set_defaults(fn=_cmd_run)

    e = sub.add_parser("estimate", help="estimate tokens and cost of a spec with a simulated dry run")
    e.add_argument("spec")
    e.add_argument("--out")
    e.add_argument("--limit", type=int)
    e.set_defaults(fn=_cmd_estimate)

    rp = sub.add_parser("report", help="build the HTML report of a run directory")
    rp.add_argument("run_dir")
    rp.add_argument("--out")
    rp.add_argument("--title")
    rp.add_argument("--hide-ground-truth", action="store_true")
    rp.set_defaults(fn=_cmd_report)

    rl = sub.add_parser("release", help="publish mechanism results without ground truth (with a commitment digest)")
    rl.add_argument("run_dir")
    rl.add_argument("out")
    rl.add_argument("--title")
    rl.add_argument("--notes")
    rl.set_defaults(fn=_cmd_release)

    v = sub.add_parser("verify", help="check a release against its manifest")
    v.add_argument("release_dir")
    v.set_defaults(fn=_cmd_verify)

    rs = sub.add_parser("resolve", help="score a release once ground truth is known")
    rs.add_argument("release_dir")
    rs.add_argument("--truth", help="JSON {item_id: label} or JSONL rows {item_id, answer}")
    rs.add_argument("--domain", help="fetch resolutions from a domain (e.g. forecasting)")
    rs.add_argument("--market", action="store_true", help="recompute market-scoring-rule rewards")
    rs.add_argument("--out")
    rs.set_defaults(fn=_cmd_resolve)

    ls = sub.add_parser("list", help="list registered mechanisms, domains or models")
    ls.add_argument("what", nargs="?", default="all", choices=["all", "mechanisms", "domains", "models"])
    ls.set_defaults(fn=_cmd_list)

    d = sub.add_parser("demo", help="run offline demos (synthetic domains, no API keys)")
    d.add_argument("name", nargs="?", default="all", choices=["all", "asd", "optimization", "swarm"])
    d.add_argument("--out", default="runs/demos")
    d.set_defaults(fn=_cmd_demo)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
