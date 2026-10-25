"""A real capability gap: engine-backed experts argue about chess moves before a weak judge.

Tasks: Lichess puzzle positions where a shallow search prefers the wrong one of two moves; ground
truth is a deep Stockfish search. Experts are engine-backed advocates (honest or cherry-picking
lines); claims are chess lines checked by trusted code (legality, optionally an evaluation). The
judge is an LLM (``OA_JUDGE_MODEL``) or, with ``--engine-judge D``, a depth-D engine search over the
revealed lines.

    python examples/chess_llm_judge.py --engine-judge 2        # no API keys needed
    OA_JUDGE_MODEL=openai/gpt-4.1-mini python examples/chess_llm_judge.py
"""

import argparse
from pathlib import Path

import oversight_arena as oa
from oversight_arena.agents.llm import LLMJudge, default_models
from oversight_arena.analysis import asd
from oversight_arena.analysis.diagnostics import outcome_metrics
from oversight_arena.channels import EvidencePolicy
from oversight_arena.domains.chess import ChessMoves
from oversight_arena.experiment.profiles import MapProfiles
from oversight_arena.mechanisms import Consultancy, Debate, NaiveJudge
from oversight_arena.sim.chess_agents import EngineAdvocate, EngineJudge

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--engine-judge", type=int, default=None, help="use a depth-D engine judge instead of an LLM")
    ap.add_argument("--out", default="runs/chess_llm_judge")
    args = ap.parse_args()
    dom = ChessMoves(n_puzzles=40, judge_trap_depth=2, gt_depth=14, verify_depth=8, limit=args.tasks)
    judge = EngineJudge(depth=args.engine_judge) if args.engine_judge is not None else LLMJudge(default_models()[1])
    agents = {"kind:judge": judge, "*": EngineAdvocate(depth=10)}
    pol = EvidencePolicy()
    mechanisms = [NaiveJudge(), Consultancy(rounds=2, evidence=pol), Debate(rounds=2, evidence=pol)]
    results = None
    for style in ("honest", "cherry_pick"):
        ms = [m.model_copy(update={"label": f"{m.name} · {style}"}) for m in mechanisms]
        res = oa.Experiment(dom, ms, agents, MapProfiles(source=oa.Stances(), params={"style": style}),
                            out=Path(args.out) / style, concurrency=2).run()
        results = res if results is None else results + res
    print(asd(results.df(trainable_only=True)).round(3).to_string(index=False))
    print(outcome_metrics(results)[["mechanism", "accuracy", "p_correct"]].round(3).to_string(index=False))
