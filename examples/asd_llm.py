"""Agent Score Difference with LLM agents (the ASD benchmark, on any domain).

For each task every protocol runs twice: once with its expert(s) arguing for the correct answer and
once for an incorrect one (counterbalanced across debaters). ASD = mean reward for truth minus
reward for falsehood; also reported: reward-GT alignment, judge accuracy and calibration, cost.

    OA_EXPERT_MODEL=openai/gpt-4.1 OA_JUDGE_MODEL=openai/gpt-4.1-nano python examples/asd_llm.py --domain gsm8k --tasks 50
"""

import argparse
from pathlib import Path

import oversight_arena as oa
from oversight_arena.analysis import ic_report
from oversight_arena.analysis.diagnostics import cost_summary, outcome_metrics
from oversight_arena.analysis.report import html_report
from oversight_arena.channels import EvidencePolicy
from oversight_arena.mechanisms import Consultancy, Debate, NaiveJudge, Propaganda
from oversight_arena.registry import build

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="gsm8k", help="registry name: gsm8k, quality, mmlu_pro, sql, code, minif2f, ...")
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--elicitation", default=None, help="judge probabilities: verbal | logprobs | sample")
    ap.add_argument("--out", default="runs/asd_llm")
    args = ap.parse_args()
    domain = build("domain", {"type": args.domain, "limit": args.tasks})
    pol = EvidencePolicy()  # verify the domain's checkable claims (quotes, arithmetic, SQL, code runs...)
    mechanisms = [NaiveJudge(), Propaganda(evidence=pol), Consultancy(rounds=2, evidence=pol), Debate(rounds=2, evidence=pol)]
    exp = oa.Experiment(domain, mechanisms, oa.llm_agents(judge_elicitation=args.elicitation), oa.Stances(),
                        out=args.out, name=f"asd-{args.domain}")
    res = exp.run()
    print(ic_report(res.df(trainable_only=True)).round(3).to_string(index=False))
    print(outcome_metrics(res).round(3).to_string(index=False))
    print(cost_summary(res).round(4).to_string(index=False))
    print(html_report(res, Path(args.out) / "report.html", title=f"ASD on {args.domain}"))
