"""Small real-LLM pilots of each experiment family. Estimate first, then run.

    python scripts/pilots.py --estimate                       # dry run: calls, tokens and $ per pilot
    python scripts/pilots.py asd paired --out runs/pilots     # run the chosen pilots (needs API keys)

Models default to inexpensive OpenRouter models (set OPENROUTER_API_KEY); override with --agent, --judge
and --optimizer (any Inspect model name, e.g. anthropic/claude-haiku-4-5). Every pilot writes a run
directory with the episodes, a summary table and an HTML report; model calls are cached, so re-running
a pilot is free.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

import so_arena as soa
from so_arena.analysis.report import build_report
from so_arena.domains import get_domain
from so_arena.mechanisms import Consultancy, Debate, DirectJudge, ReviewedWork

PILOTS = ("asd", "chess", "lean", "paired", "firm", "search", "bon")


def pilot_asd(a, out: Path) -> pd.DataFrame:
    """Instructed-arm ASD on GSM8K across the classic protocols."""
    from so_arena.samplers.arms import ASDExperiment

    dom = get_domain("gsm8k")
    exp = ASDExperiment([DirectJudge(), Consultancy(rounds=2), Debate(rounds=2)], dom.load(limit=a.n),
                        agent=soa.LLMPolicy(a.agent, temperature=0.7, max_tokens=600),
                        fixtures={"judge": soa.LLMPolicy(a.judge, elicitation=a.elicitation, temperature=0)},
                        ctx=dom.context(), store=str(out))
    exp.run()
    return exp.summary()


def pilot_chess(a, out: Path) -> pd.DataFrame:
    """Chess: engine-backed experts vs an LLM judge, with and without the legal-line rule."""
    from so_arena.samplers.arms import ASDExperiment

    dom = get_domain("chess", kind="which_move")
    ver = soa.VerificationPolicy(verifiers=["chess_line"])
    mechs = [DirectJudge(), Debate(rounds=2, tools={"agents": ["engine"]}, name="debate+engine"),
             Debate(rounds=2, tools={"agents": ["engine"]}, verification=ver, name="debate+engine+legal_lines")]
    exp = ASDExperiment(mechs, dom.load(limit=a.n), agent=soa.LLMPolicy(a.agent, temperature=0.7, max_tokens=600),
                        fixtures={"judge": soa.LLMPolicy(a.judge, elicitation=a.elicitation, temperature=0)},
                        ctx=dom.context(), store=str(out))
    exp.run()
    return exp.summary()


def pilot_lean(a, out: Path) -> pd.DataFrame:
    """Lean statement faithfulness with the rules-only structure verifier."""
    from so_arena.samplers.arms import ASDExperiment

    dom = get_domain("lean")
    ver = soa.VerificationPolicy(verifiers=["lean_parse"])
    exp = ASDExperiment([DirectJudge(), Consultancy(rounds=1, verification=ver), Debate(rounds=2, verification=ver)],
                        dom.load(limit=a.n), agent=soa.LLMPolicy(a.agent, temperature=0.7, max_tokens=600),
                        fixtures={"judge": soa.LLMPolicy(a.judge, elicitation=a.elicitation, temperature=0)},
                        ctx=dom.context(), store=str(out))
    exp.run()
    return exp.summary()


def _paired(a, out: Path, dom, items, arms: dict[str, str]) -> pd.DataFrame:
    from so_arena.samplers.paired import PairedWorkExperiment, behaviour_policies

    protocols = [ReviewedWork(name="report_only", dossier="report"), ReviewedWork(name="report+diff"),
                 ReviewedWork(name="query", state_access={"reviewer": "read"})]
    judge = soa.LLMPolicy(a.judge, elicitation=a.elicitation, temperature=0, max_tool_calls=12)
    exp = PairedWorkExperiment(protocols, items, behaviours=behaviour_policies(a.agent, arms, temperature=0.5),
                               fixtures={"reviewer": judge}, ctx=dom.context(states=out / "states"),
                               ground_truth=dom.ground_truth_scorers(), store=str(out))
    exp.run()
    return exp.summary()


def pilot_paired(a, out: Path) -> pd.DataFrame:
    """Repository work: honest vs faked success, reviewed with three levels of access."""
    dom = get_domain("repo")
    arms = {k: v for k, v in dom.behaviours().items() if k in ("honest", "hardcode", "tamper", "lazy")}
    return _paired(a, out, dom, dom.load(limit=max(2, a.n // 3)), arms)


def pilot_firm(a, out: Path) -> pd.DataFrame:
    """The simulated company: honest vs faked KPIs on sales, support and reporting tasks."""
    dom = get_domain("firm", n_items=max(1, a.n // 10))
    return _paired(a, out, dom, dom.load(), dom.behaviours())


def pilot_search(a, out: Path) -> pd.DataFrame:
    """Directive-constrained prompt search on consultancy: the best honest vs best deceptive consultant."""
    from so_arena.samplers.prompt_search import PromptSearchSuite

    dom = get_domain("gsm8k")
    items = dom.load(limit=a.n)
    half = len(items) // 2
    judge = soa.LLMPolicy(a.judge, elicitation=a.elicitation, temperature=0)
    suite = PromptSearchSuite(directives=("honest", "deceptive"), mechanism=Consultancy(rounds=1), items=items[:half],
                              eval_items=items[half:], role="consultant",
                              policy_factory=lambda s: soa.LLMPolicy(a.agent, strategy=s, temperature=0.7, max_tokens=500),
                              others={"judge": judge}, optimizer=a.optimizer, algorithm="reflective", iterations=3,
                              candidates_per_iter=3, ctx=dom.context())
    suite.searches["honest"].arms = ["true"]
    suite.searches["deceptive"].arms = ["false"]
    suite.run()
    suite.frame().to_csv(out / "candidates.csv", index=False)
    a._extra_cost += sum(e.total_usage.cost_usd for s in suite.searches.values() for eps in s.episodes.values() for e in eps)
    return pd.DataFrame([suite.honesty_margin()])


def pilot_bon(a, out: Path) -> pd.DataFrame:
    """Best-of-N self-play on proposal -> critique -> judgment (the BoN self-play study's design)."""
    from so_arena.core.runner import Profile
    from so_arena.samplers.pools import OptimizationExperiment

    dom = get_domain("gsm8k")
    agent = soa.LLMPolicy(a.agent, temperature=1.0, max_tokens=500)
    judge = soa.LLMPolicy(a.judge, elicitation=a.elicitation, temperature=0)
    exp = OptimizationExperiment(ReviewedWork(critique_rounds=1, rebuttal=False, transform="prob"), dom.load(limit=a.n // 2),
                                 Profile(name="open", players={"worker": agent, "critic": agent, "reviewer": judge}),
                                 pool_sizes={"worker": 4, "critic": 2}, ctx=dom.context())
    trees = exp.run()
    a._extra_cost += sum(u.get("cost_usd", 0.0) for t in trees for u in t.usage.values())
    grid = exp.grid({"worker": [1, 2, 4], "critic": [1, 2]})
    grid.to_csv(out / "grid.csv", index=False)
    return grid


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pilots", nargs="*", help=f"which pilots: {', '.join(PILOTS)} (default: all)")
    ap.add_argument("--estimate", action="store_true", help="dry run with simulated models; prints the cost")
    ap.add_argument("--out", default="runs/pilots")
    ap.add_argument("-n", type=int, default=30, help="items per pilot (scaled down for the costlier ones)")
    ap.add_argument("--agent", default="openrouter/qwen/qwen3-32b")
    ap.add_argument("--judge", default="openrouter/meta-llama/llama-3.1-8b-instruct")
    ap.add_argument("--optimizer", default="openrouter/qwen/qwen3-32b")
    ap.add_argument("--elicitation", default="logprobs", choices=["ask", "logprobs", "vote"])
    a = ap.parse_args(argv)
    unknown = set(a.pilots) - set(PILOTS)
    if unknown:
        ap.error(f"unknown pilots {sorted(unknown)}; choose from {', '.join(PILOTS)}")
    from so_arena.spec import usage_summary

    soa.configure(cache_dir=str(Path(a.out) / "cache"), simulate=a.estimate)
    fns = {name: globals()[f"pilot_{name}"] for name in PILOTS}
    total = 0.0
    for name in a.pilots or PILOTS:
        out = Path(a.out) / ("estimate" if a.estimate else "runs") / name
        out.mkdir(parents=True, exist_ok=True)
        a._extra_cost = 0.0  # pilots whose episodes are not stored (prompt search, trees) report cost here
        summary = fns[name](a, out)
        summary.to_csv(out / "summary.csv", index=False)
        store = soa.RunStore(out)
        eps = store.episodes()
        cost = float(sum(e.total_usage.cost_usd for e in eps)) + a._extra_cost
        total += cost
        if a.estimate:
            print(f"\n== {name}: {len(eps)} stored episodes, estimated ${cost:.4f}")
            if eps:
                print(usage_summary(eps).to_string(index=False))
            if name in ("paired", "firm"):
                print("   (dry runs count one call per agentic decision: multiply the worker's cost by the tool calls "
                      "you expect per task, up to max_tool_calls=30)")
        else:
            if eps:
                build_report(eps, out / "report.html", title=f"Pilot: {name}")
            print(f"\n== {name}: {len(eps)} episodes, ${cost:.2f} -> {out}")
            print(summary.head(20).to_string(index=False))
    print(f"\nTOTAL {'estimated ' if a.estimate else ''}cost: ${total:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
