"""Command-line interface: ``so-arena <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


ESTIMATE_MARKER = ".so-arena-estimate"  # marks directories `estimate` created (and may therefore replace)


def _cmd_run(a: argparse.Namespace) -> int:
    from so_arena.config import configure
    from so_arena.spec import SpecError, load_spec, run_counts, run_spec, run_usage

    if a.simulate:
        configure(simulate=True)
    try:
        spec = load_spec(a.spec)
        run_dir = run_spec(spec, out=a.out, limit=a.limit, force=a.force)
    except SpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    counts = run_counts(run_dir)
    note = " (grid_*.csv holds the optimization curves)" if "trees" in counts else ""
    print(f"{counts} -> {run_dir}{note}")
    if (run_dir / "report.html").exists():
        print(f"report: {run_dir / 'report.html'}")
    if a.simulate:
        print(run_usage(run_dir).to_string(index=False))
    return 0


def _cmd_estimate(a: argparse.Namespace) -> int:
    """Dry run with simulated models: estimates tokens and cost without calling any API."""
    import shutil

    from so_arena.config import configure
    from so_arena.spec import SpecError, load_spec, run_counts, run_spec, run_usage

    configure(simulate=True)
    try:
        spec = load_spec(a.spec)
    except SpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    spec.report = False
    out = Path(a.out or f"runs/_estimate_{spec.name}")
    if out.exists():
        if (out / ESTIMATE_MARKER).is_file():
            shutil.rmtree(out)  # an earlier estimate's scratch directory
        elif not out.is_dir() or any(out.iterdir()):
            print(f"error: {out} exists and was not created by `so-arena estimate` (no {ESTIMATE_MARKER} file); "
                  "refusing to replace it - pass a new or empty --out directory", file=sys.stderr)
            return 2
    out.mkdir(parents=True, exist_ok=True)
    (out / ESTIMATE_MARKER).write_text("created by `so-arena estimate`; the next estimate into this directory replaces it\n")
    try:
        run_dir = run_spec(spec, out=out, limit=a.limit)
    except SpecError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    summary = run_usage(run_dir)
    print(f"Simulated {run_counts(run_dir)} (no API calls). Estimated usage:")
    print(summary.to_string(index=False))
    models = [m for m in summary["model"].dropna().unique() if isinstance(m, str) and m]
    if models:
        from so_arena.models.registry import get_spec

        missing = [m for m in models if get_spec(m.removeprefix("sim/")) is None]
        if missing:
            print(f"(no price known for: {', '.join(missing)} - add them to the model registry; they count as $0)")
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

    try:
        man = release_run(a.run_dir, a.out, title=a.title or Path(a.run_dir).name, notes=a.notes or "",
                          private_keys=a.keep_private or (), public_labels=a.public_labels,
                          exclude_restricted=a.exclude_restricted)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"released {man.n_episodes} episodes on {man.n_items} items -> {a.out}")
    print(f"commitment digest: {man.digest}  (publish it: `so-arena verify {a.out} --digest <digest>` checks against it)")
    return 0


def _cmd_verify(a: argparse.Namespace) -> int:
    from so_arena.release import Manifest, release_digest, uncovered_files, verify

    d = Path(a.release_dir)
    man = Manifest.model_validate_json((d / "MANIFEST.json").read_text())
    now = release_digest(d)
    print(f"digest: {now}" + ("" if now == man.digest else f"  (MANIFEST.json says {man.digest})"))
    for name in uncovered_files(d):
        print(f"note: {name} is not covered by the manifest - nothing vouches for it")
    if not verify(d):
        print("MISMATCH: release files were modified (they do not match MANIFEST.json)")
        return 1
    if a.digest is None:
        print("OK: the files match MANIFEST.json - to rule out a rewritten manifest, compare the digest with the "
              "one published at release time (--digest)")
        return 0
    if not verify(d, a.digest):
        print("MISMATCH: the release is not the one the published digest commits to")
        return 1
    print("OK: the files match the published digest")
    return 0


def _cmd_resolve(a: argparse.Namespace) -> int:
    from so_arena.release import resolve

    reward = None
    if a.market:
        from so_arena.mechanisms import MarketScoringReward

        reward = MarketScoringReward()
    dom = None
    if a.domain:
        from so_arena.domains import get_domain

        dom = get_domain(a.domain)
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
    elif dom is not None:
        if not hasattr(dom, "resolve"):
            print(f"domain {a.domain} cannot fetch resolutions", file=sys.stderr)
            return 2
        truth = lambda items: {k: v for k, v in dom.resolve(items).items() if v is not None}  # noqa: E731
    else:
        print("pass --truth FILE or --domain NAME", file=sys.stderr)
        return 2
    # the domain's own scorers measure what matters there (e.g. forecast scores), not just the defaults
    scorers = dom.ground_truth_scorers() if dom is not None else None
    try:
        res = resolve(a.release_dir, truth, out_dir=a.out, reward_rule=reward, ground_truth=scorers, digest=a.digest)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
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
    for kind in ("reward", "verifier", "scorer"):
        if what in (f"{kind}s", "all"):
            from so_arena.registry import names

            print(f"{kind}s:", ", ".join(names(kind)))
    if what in ("models", "all"):
        from so_arena.models import all_specs

        for s in all_specs():
            price = (f"${s.price_input_per_mtok}/${s.price_output_per_mtok} per Mtok"
                     if s.price_input_per_mtok is not None else "no price")
            print(f"  {s.name:45s} {price}")
    return 0


def _cmd_demo(a: argparse.Namespace) -> int:
    from so_arena import demos

    fns = {"asd": demos.demo_asd, "optimization": demos.demo_optimization, "swarm": demos.demo_swarm,
           "work": demos.demo_work, "monitoring": demos.demo_monitoring, "hiddenbits": demos.demo_hiddenbits,
           "bon_budget": demos.demo_bon_budget, "release": demos.demo_release}
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
    r.add_argument("--force", action="store_true",
                   help="resume even if the output directory holds a run of a different spec")
    r.set_defaults(fn=_cmd_run)

    e = sub.add_parser("estimate", help="estimate tokens and cost of a spec with a simulated dry run")
    e.add_argument("spec")
    e.add_argument("--out", help=f"scratch directory (default runs/_estimate_<name>); must be new, empty or an "
                                 f"earlier estimate's (it holds a {ESTIMATE_MARKER} file) - it is replaced")
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
    rl.add_argument("--keep-private", action="append", metavar="KEY",
                    help="publish this key of the items' private information (repeatable; default: none)")
    rl.add_argument("--public-labels", action="store_true",
                    help="publish profile names, tags and behaviour labels (only if not defined relative to the truth)")
    rl.add_argument("--exclude-restricted", action="store_true",
                    help="leave out items whose licence forbids publication instead of refusing")
    rl.set_defaults(fn=_cmd_release)

    v = sub.add_parser("verify", help="check a release against its manifest and the published digest")
    v.add_argument("release_dir")
    v.add_argument("--digest", help="the commitment digest published at release time")
    v.set_defaults(fn=_cmd_verify)

    rs = sub.add_parser("resolve", help="score a release once ground truth is known")
    rs.add_argument("release_dir")
    rs.add_argument("--truth", help="JSON {item_id: label} or JSONL rows {item_id, answer}")
    rs.add_argument("--domain", help="score with a domain's ground-truth scorers; without --truth, also fetch "
                                     "its resolutions (e.g. forecasting)")
    rs.add_argument("--market", action="store_true", help="recompute market-scoring-rule rewards")
    rs.add_argument("--digest", help="refuse unless the release matches this published digest")
    rs.add_argument("--out")
    rs.set_defaults(fn=_cmd_resolve)

    ls = sub.add_parser("list", help="list registered mechanisms, domains, reward rules, verifiers, scorers or models")
    ls.add_argument("what", nargs="?", default="all",
                    choices=["all", "mechanisms", "domains", "rewards", "verifiers", "scorers", "models"])
    ls.set_defaults(fn=_cmd_list)

    d = sub.add_parser("demo", help="run offline demos (synthetic domains, no API keys)")
    d.add_argument("name", nargs="?", default="all", choices=["all", "asd", "optimization", "swarm", "work", "monitoring",
                                                             "hiddenbits", "bon_budget", "release"])
    d.add_argument("--out", default="runs/demos")
    d.set_defaults(fn=_cmd_demo)

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
