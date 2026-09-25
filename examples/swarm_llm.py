"""LLM agent swarms: a shared objective vs individual bounties for reporting misconduct.

SimOps: several LLM workers maintain a small Python project through tools (read/write files, run
the visible tests, message board, activity log). One assignment cannot be solved honestly, but
its tests can be gamed; hidden tests and tamper-proof logs are the ground truth. The same workers
run under a fully shared reward and under whistleblower rules; compare misconduct, reporting and
the true project score. (For the equilibrium analysis and theory, see the `swarm` demo.)

    OA_EXPERT_MODEL=openai/gpt-4.1 python examples/swarm_llm.py --projects 5
"""

import argparse

import oversight_arena as oa
from oversight_arena.domains.swarm import SimOps
from oversight_arena.mechanisms import Swarm, TeamReward, Whistleblower

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    dom = SimOps(n_tasks=args.projects, n_workers=3)
    rules = {"shared reward": TeamReward(), "bounty > stake": Whistleblower(bounty=1.0, offender_penalty=1.0, misprision=0.3)}
    mechs = [Swarm(n_workers=3, rounds=args.rounds, reward=r, label=name) for name, r in rules.items()]
    res = oa.Experiment(dom, mechs, oa.llm_agents(max_tool_rounds=8), out="runs/swarm_llm", concurrency=2).run()
    for r in res.records:
        print(f"{r.mechanism:15s} violators={r.env_state.get('violators')} detected={r.outcome.get('detected')} "
              f"true={r.env_state.get('true_score'):.2f} apparent={r.env_state.get('apparent_score'):.2f} rewards={r.rewards}")
