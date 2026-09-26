"""Bottom-up cost estimates for the two grant designs (Thinking Machines safety grant, Tinker credits).

    python scripts/grant_budget.py               # uses the measured prompt sizes in docs/reports/grant_budget_prompts.json
    python scripts/grant_budget.py --remeasure   # re-measures them with a dry run (simulated models, no API calls)

Writes docs/reports/grant_budget.md. Cost = episodes x tokens per episode x published prices
(:mod:`so_arena.budget`). Prompt sizes are measured from the library's own prompts; behaviour the dry
run cannot see (reasoning length, tool turns, tool-output size, overseer verbosity) is an explicit
assumption with a low / central / high value. Pilot runs replace those with measurements.

Design T ("Tinker-first"): RL of an open worker against 13 mechanism configurations, screened first by
ASD, best-of-N and prompt search, all on Tinker. Design B ("broad", Task 4): protocol evaluation with
frontier API agents across five domains, prompt search, team experiments, 24 RL runs, activation
probes, paid experts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

from so_arena.budget import Budget, Fixed, Item, Tokens, agentic, conversation, tinker_snapshot_date

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "docs" / "reports" / "grant_budget_prompts.json"
REPORT = ROOT / "docs" / "reports" / "grant_budget.md"

# ------------------------------------------------------------------------------ measured prompt sizes

SIM_OUT = {"worker": 400, "reviewer": 200, "critic": 300, "monitor": 150, "grader": 150, "judge": 150,
           "debater_a": 400, "debater_b": 400}
DOMAINS = ("code", "sql", "firm", "chess", "repo", "lean", "forecasting")


def remeasure() -> dict:
    """Prompt sizes (tokens) per domain: the worker's task prompt and each overseer's fixed part
    (its prompt minus the worker text it was shown in the dry run)."""
    from so_arena.budget import measure
    from so_arena.domains import get_domain
    from so_arena.mechanisms import MonitoredWork, ReviewedWork

    out = {}
    for d in DOMAINS:
        dom = get_domain(d)
        items = dom.load(limit=4)
        rev = measure(ReviewedWork(), items, ctx=dom.context(), output_tokens=SIM_OUT)
        crit = measure(ReviewedWork(critique_rounds=1), items, ctx=dom.context(), output_tokens=SIM_OUT)
        mon = measure(MonitoredWork(), items, ctx=dom.context(), output_tokens=SIM_OUT)
        w = SIM_OUT["worker"]
        out[d] = {"worker": rev["worker"]["first_input"],
                  "reviewer": rev["reviewer"]["first_input"] - w,
                  "critic": crit["critic"]["first_input"] - w,
                  "monitor": mon["monitor"]["first_input"] - w,
                  "grader": mon["grader"]["first_input"] - w}
        print(d, {k: round(v) for k, v in out[d].items()}, flush=True)
    return out


# ------------------------------------------------------------------------------ assumptions

# per-domain behaviour at the central value: tool turns per episode, tokens per tool observation, and
# the size of the change a reviewer is shown (diff / actions summary)
BEHAVIOUR = {
    "code": {"turns": 6, "obs": 600, "diff": 300},
    "sql": {"turns": 5, "obs": 300, "diff": 150},
    "firm": {"turns": 8, "obs": 400, "diff": 200},
    "chess": {"turns": 3, "obs": 250, "diff": 0},
    "repo": {"turns": 20, "obs": 1200, "diff": 500},
    "lean": {"turns": 4, "obs": 400, "diff": 300},
    "forecasting": {"turns": 3, "obs": 800, "diff": 0},
}

RANGES = {  # (low, central, high)
    "reasoning_per_turn": (200, 400, 900),  # worker reasoning tokens before each tool call
    "final_reasoning": (400, 1000, 3000),  # worker reasoning before its final answer / report
    "final_output": (250, 400, 700),  # the answer / report the overseer reads
    "action": (80, 150, 300),  # tokens per tool call
    "turns_x": (0.6, 1.0, 1.6),  # x the domain's tool turns
    "obs_x": (0.5, 1.0, 2.0),  # x the domain's tool-output size
    "overseer_output": (150, 300, 1200),  # overseer tokens per call, its own reasoning included
    "critic_reasoning": (300, 600, 1500),  # critic / debater reasoning per call
    "prompt_x": (0.85, 1.0, 1.2),  # tokenizer error on the measured prompts (~4 chars / token)
    "overhead_x": (1.1, 1.25, 1.5),  # pilots, failed and repeated runs, debugging
}

# ------------------------------------------------------------------------------ token profiles

P: dict = {}


def worker(d, a, *, trained=False):
    b = BEHAVIOUR[d]
    return agentic(P[d]["worker"] * a["prompt_x"], b["turns"] * a["turns_x"], a["reasoning_per_turn"], a["action"],
                   b["obs"] * a["obs_x"], a["final_output"], a["final_reasoning"], trained=trained)


def seen(d, a):
    """What overseers can be shown of the worker's episode."""
    b = BEHAVIOUR[d]
    n = round(b["turns"] * a["turns_x"])
    return {"report": a["final_output"], "diff": b["diff"] * a["prompt_x"],
            "tool_io": n * (a["action"] + b["obs"] * a["obs_x"]),
            "cot": n * a["reasoning_per_turn"] + a["final_reasoning"], "length": worker(d, a, trained=True).train}


def followups(ctx, k, answer, a, *, question=100, trained=False):
    """The worker answers ``k`` follow-up messages in its own conversation (questions, critiques)."""
    step = question + a["reasoning_per_turn"] + answer
    return Tokens(prefill=k * question, cached=sum(ctx + i * step for i in range(k)),
                  sample=k * (a["reasoning_per_turn"] + answer), train=k * step if trained else 0.0)


def overseer(base, visible, a, calls=1, others=0.0):
    return conversation(base * a["prompt_x"] + visible, calls, a["overseer_output"], others=others)


def mechanism(m, d, a, W, O, *, trained):
    """Role -> (model, tokens) for one episode of configuration ``m`` on domain ``d`` (W: worker model,
    O: overseer model). The worker's own task episode is included."""
    s, p = seen(d, a), P[d]
    roles = {"worker": (W, worker(d, a, trained=trained))}
    add = lambda r, mdl, t: roles.__setitem__(r, (mdl, roles[r][1] + t) if r in roles else (mdl, t))
    if m in ("direct", "audit_p05", "audit_p20", "same_size", "activation_monitor", "evidence_audits",
             "reserve_adaptive_overseer"):
        add("reviewer", W if m == "same_size" else O, overseer(p["reviewer"], s["report"] + s["diff"], a))
    elif m == "consultancy":
        add("worker", W, followups(s["length"], 2, 300, a, trained=trained))
        add("reviewer", O, overseer(p["reviewer"], s["report"] + s["diff"], a, calls=3, others=300))
    elif m in ("debate", "debate_verified"):
        extra = 200 if m == "debate_verified" else 0  # cited claims
        add("worker", W, followups(s["length"], 1, 300 + extra, a, trained=trained))
        add("critic", W, conversation(p["critic"] * a["prompt_x"] + s["report"] + s["diff"], 2, 300 + extra,
                                      a["critic_reasoning"], others=300 + extra, trained=trained))
        add("reviewer", O, overseer(p["reviewer"], s["report"] + s["diff"] + 2 * (300 + extra) + 300 + extra, a))
    elif m in ("action_monitor", "cot_monitor", "cot_audit"):
        vis = s["report"] + s["tool_io"] + (s["cot"] if m != "action_monitor" else 0)
        add("monitor", O, overseer(p["monitor"], vis, a))
        add("grader", O, overseer(p["grader"], s["report"] + s["diff"], a))
    elif m == "confession":
        add("worker", W, followups(s["length"], 1, 300, a, trained=trained))
        add("reviewer", O, overseer(p["reviewer"], s["report"] + s["diff"] + 300, a))
    elif m == "full_verification":
        pass
    else:
        raise KeyError(m)
    return roles


def mix(weights, fn):
    """Weighted mixture of per-domain profiles, role by role (tokens per episode)."""
    out: dict = {}
    for d, w in weights.items():
        for r, (mdl, t) in fn(d).items():
            key = (r, mdl)
            out[key] = out.get(key, Tokens()) + t.scale(w)
    return {f"{r}@{mdl}": (mdl, t) for (r, mdl), t in out.items()}


def minus_worker_task(roles, d, a, W):
    """A mechanism's cost on top of an already generated (shared) worker episode."""
    base = worker(d, a)
    out = dict(roles)
    mdl, t = out["worker"]
    rest = Tokens(t.prefill - base.prefill, t.cached - base.cached, t.sample - base.sample, 0.0)
    if rest.prefill + rest.cached + rest.sample > 0:
        out["worker"] = (mdl, rest)
    else:
        del out["worker"]
    return out


def gpu_hours_cost(tokens, tok_per_s, usd_per_gpu_h=4.29):
    """Own-GPU forward passes (activation collection): Lambda on-demand 1x H100 SXM price."""
    return tokens / tok_per_s / 3600 * usd_per_gpu_h


def cpu_cost(tool_calls, seconds, usd_per_vcpu_h=0.05):
    return tool_calls * seconds / 3600 * usd_per_vcpu_h


# ------------------------------------------------------------------------------ design T (Tinker-first)

T_WORKER, T_OVERSEER, T_BIG = "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-397B-A17B"
T_TRAIN = {"code": 1 / 3, "sql": 1 / 6, "firm": 1 / 6, "chess": 1 / 3}
T_CONDITIONS = ["direct", "same_size", "consultancy", "debate", "debate_verified", "confession", "audit_p05", "audit_p20",
                "action_monitor", "cot_monitor", "cot_audit", "full_verification"]
T_EXTRA_SEED = ["direct", "debate", "cot_monitor"]
STEPS, BATCH = 150, 32 * 8  # RL steps x (prompts x group size) per step


def design_t(W=T_WORKER, O=T_OVERSEER, *, steps=STEPS, K=32, ckpts=6, screen_per_domain=200) -> Budget:
    items = []
    ov = lambda a: a["overhead_x"]
    eps_run = steps * BATCH
    runs = {m: 2 + (m in T_EXTRA_SEED) for m in T_CONDITIONS}
    for m in T_CONDITIONS:
        items.append(Item(f"RL: {m} ({runs[m]} seeds)", lambda a, n=runs[m]: n * eps_run * ov(a),
                          lambda a, m=m: mix(T_TRAIN, lambda d: mechanism(m, d, a, W, O, trained=True)),
                          group="3 RL matrix", note=f"{steps} steps x {BATCH} episodes per run"))
    n_runs = sum(runs.values())
    screen_items = {d: screen_per_domain for d in T_TRAIN}
    n_items = sum(screen_items.values())
    wts = {d: n / n_items for d, n in screen_items.items()}
    mechs = [m for m in T_CONDITIONS if m != "full_verification"]
    # ASD: 6 instructed arms; each (item, arm) work episode is shared by every configuration
    items.append(Item("ASD: worker episodes (6 arms, shared)", lambda a: n_items * 6 * ov(a),
                      lambda a: mix(wts, lambda d: {"worker": (W, worker(d, a))}), group="1 screens"))
    items.append(Item(f"ASD: {len(mechs)} mechanisms on the shared episodes", lambda a: n_items * 6 * ov(a),
                      lambda a: _sum_mechs(mechs, wts, a, W, O), group="1 screens"))
    items.append(Item(f"best-of-N: worker pools (K={K})", lambda a: n_items * K * ov(a),
                      lambda a: mix(wts, lambda d: {"worker": (W, worker(d, a))}), group="1 screens"))
    items.append(Item("best-of-N: mechanisms per candidate (4 continuations for interactive ones)",
                      lambda a: n_items * K * ov(a), lambda a: _sum_mechs(mechs, wts, a, W, O, interactive=4),
                      group="1 screens"))
    searches, rollouts, reflections = len(mechs) * 3, 1500, 150
    items.append(Item(f"prompt search: {searches} searches x {rollouts} rollouts", lambda a: searches * rollouts * ov(a),
                      lambda a: _avg_mechs(mechs, wts, a, W, O), group="2 prompt search"))
    items.append(Item(f"prompt search: reflection LM ({reflections} calls per search)",
                      lambda a: searches * reflections * ov(a),
                      lambda a: {"reflector": (T_BIG, Tokens(prefill=8000 * a["prompt_x"], sample=2000))},
                      group="2 prompt search"))
    items.append(Item("prompt search: validation (5 prompts x 200 items)", lambda a: searches * 5 * 200 * ov(a),
                      lambda a: _avg_mechs(mechs, wts, a, W, O), group="2 prompt search"))
    held, transfer = 400, 50
    items.append(Item(f"eval: {n_runs + 1} runs x {ckpts} checkpoints x {held} held-out items",
                      lambda a: (n_runs + 1) * ckpts * held * ov(a),
                      lambda a: _avg_mechs(T_CONDITIONS, T_TRAIN, a, W, O, diagnostic=True), group="4 evaluation"))
    items.append(Item(f"eval: repository transfer ({transfer} tasks per checkpoint)",
                      lambda a: (n_runs + 1) * ckpts * transfer * ov(a),
                      lambda a: _avg_mechs(T_CONDITIONS, {"repo": 1.0}, a, W, O, diagnostic=True), group="4 evaluation"))
    items.append(Fixed("Tinker checkpoint storage", (20, 100, 300), provider="tinker", group="5 other",
                       note="LoRA checkpoints, $0.10/GB-month"))
    return Budget(items, RANGES, extra={"runs": n_runs, "rl_episodes": n_runs * eps_run, "models": (W, O, T_BIG),
                                        "steps": steps, "K": K, "ckpts": ckpts})


def _sum_mechs(mechs, wts, a, W, O, interactive=1):
    tot: dict = {}
    for m in mechs:
        k = interactive if m in ("consultancy", "debate", "debate_verified", "confession") else 1
        prof = mix(wts, lambda d, m=m: minus_worker_task(mechanism(m, d, a, W, O, trained=False), d, a, W))
        for key, (mdl, t) in prof.items():
            tot[key] = (mdl, tot.get(key, (mdl, Tokens()))[1] + t.scale(k))
    return tot


def _avg_mechs(mechs, wts, a, W, O, diagnostic=False):
    tot: dict = {}
    for m in mechs:
        prof = mix(wts, lambda d, m=m: mechanism(m, d, a, W, O, trained=False))
        for key, (mdl, t) in prof.items():
            tot[key] = (mdl, tot.get(key, (mdl, Tokens()))[1] + t.scale(1 / len(mechs)))
    if diagnostic:  # frozen diagnostic monitor (strong model, no training reward) reads tool I/O and reasoning
        d_tok = Tokens()
        for d, w in wts.items():
            s = seen(d, a)
            d_tok = d_tok + overseer(P[d]["monitor"], s["report"] + s["tool_io"] + s["cot"], a).scale(w)
        tot["diagnostic"] = (T_BIG, d_tok)
    return tot


# ------------------------------------------------------------------------------ design B (broad, Task 4)

B_AGENT, B_JUDGE = "anthropic/claude-sonnet-5", "anthropic/claude-haiku-4-5"
B_RL_WORKER, B_RL_OVERSEER = "Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-4B"
B_ITEMS = {"repo": 300, "chess": 1000, "sql": 100, "firm": 100, "lean": 120, "forecasting": 350}
# Experiment A: 4 protocols x 3 evidence conditions; Experiment B: 4 monitoring channels
B_CONFIGS = ["direct", "consultancy", "debate", "debate_verified"] * 3 + ["action_monitor", "cot_monitor", "cot_audit",
                                                                          "activation_monitor"]
B_RL = {"direct": 3, "debate": 3, "action_monitor": 3, "cot_monitor": 3, "activation_monitor": 3, "evidence_audits": 3,
        "full_verification": 3, "reserve_adaptive_overseer": 3}  # 21 core runs + 3 reserve


def design_b(bon=(B_AGENT, B_JUDGE), search=(B_AGENT, B_JUDGE)) -> Budget:
    """Task 4's design. ``bon`` / ``search``: (agent, judge) models of the best-of-N pools and the prompt
    searches (the API models as written; open models on Tinker for the feasible variant)."""
    items = []
    ov = lambda a: a["overhead_x"]
    n_items = sum(B_ITEMS.values())
    wts = {d: n / n_items for d, n in B_ITEMS.items()}
    items.append(Item("instructed arms: agent episodes (6 arms, shared)", lambda a: n_items * 6 * ov(a),
                      lambda a: mix(wts, lambda d: {"worker": (B_AGENT, worker(d, a))}), group="1 protocol evaluation"))
    items.append(Item(f"instructed arms: {len(B_CONFIGS)} configurations on the shared episodes",
                      lambda a: n_items * 6 * ov(a), lambda a: _sum_mechs(B_CONFIGS, wts, a, B_AGENT, B_JUDGE),
                      group="1 protocol evaluation"))
    pools = {"repo": (300, 16), "chess": (1000, 64), "sql": (100, 64), "firm": (100, 64)}
    n_bon = sum(n * k for n, k in pools.values())
    wb = {d: n * k / n_bon for d, (n, k) in pools.items()}
    items.append(Item("best-of-N: agent pools (repo K=16, chess/SQL K=64)", lambda a: n_bon * ov(a),
                      lambda a: mix(wb, lambda d: {"worker": (bon[0], worker(d, a))}), group="1 protocol evaluation"))
    items.append(Item("best-of-N: configurations per candidate", lambda a: n_bon * ov(a),
                      lambda a: _sum_mechs(B_CONFIGS, wb, a, *bon, interactive=4), group="1 protocol evaluation"))
    main = {"repo": 1 / 3, "chess": 1 / 3, "sql": 1 / 6, "firm": 1 / 6}
    searches = 3 * 8  # 3 populations x 8 distinct protocol/monitor families
    items.append(Item(f"prompt search: {searches} searches x 1000 rollouts (GEPA + reflective baseline x2)",
                      lambda a: 2 * searches * 1000 * ov(a), lambda a: _avg_mechs(B_CONFIGS, main, a, *search),
                      group="2 prompt search"))
    items.append(Item("prompt search: reflection LM (150 calls per search)", lambda a: 2 * searches * 150 * ov(a),
                      lambda a: {"reflector": (B_AGENT if search[0] == B_AGENT else T_BIG,
                                               Tokens(prefill=8000 * a["prompt_x"], sample=2000))},
                      group="2 prompt search"))
    # team incentives (Experiment D) and population adaptation, on open models (Tinker sampling): 3 agents
    team = lambda a: {"team": (B_RL_WORKER, worker("firm", a).scale(3)),
                      "auditor": (B_RL_OVERSEER, overseer(P["firm"]["reviewer"], 3 * seen("firm", a)["report"], a))}
    items.append(Item("teams: 24 dev configurations x 1000 episodes", lambda a: 24_000 * ov(a), team, group="3 teams"))
    items.append(Item("teams: 4 locked configurations x 2000 held-out episodes", lambda a: 8_000 * ov(a), team, group="3 teams"))
    items.append(Item("teams + monitors: population (PSRO) best responses, 60k episodes", lambda a: 60_000 * ov(a), team,
                      group="3 teams"))
    items.append(Item("forecasting: 350 questions x 4 mechanisms x 5 samples", lambda a: 7_000 * ov(a),
                      lambda a: _avg_mechs(["direct", "debate", "consultancy", "direct"], {"forecasting": 1.0}, a, B_AGENT,
                                           B_JUDGE), group="4 delayed ground truth"))
    eps_run = STEPS * BATCH
    rl_train = {"code": 1 / 3, "sql": 1 / 6, "firm": 1 / 6, "chess": 1 / 3}
    for m, n in B_RL.items():
        # the probe replaces the LLM monitor (activations: own GPU); the reserve runs an audit-updated overseer
        prof_m = "direct" if m in ("activation_monitor", "reserve_adaptive_overseer") else m
        items.append(Item(f"RL: {m} ({n} runs)", lambda a, n=n: n * eps_run * ov(a),
                          lambda a, pm=prof_m: mix(rl_train, lambda d: mechanism(pm, d, a, B_RL_WORKER, B_RL_OVERSEER,
                                                                                  trained=True)),
                          group="5 RL"))
    n_runs = sum(B_RL.values())
    items.append(Item(f"eval: {n_runs} runs x 6 checkpoints x 450 items", lambda a: n_runs * 6 * 450 * ov(a),
                      lambda a: _avg_mechs(list(B_RL), {**{d: w * 400 / 450 for d, w in rl_train.items()}, "repo": 50 / 450},
                                           a, B_RL_WORKER, B_RL_OVERSEER, diagnostic=True), group="5 RL"))
    # own-GPU forward passes for activations: the activation-monitor RL runs (every rollout) and the probe study
    def act_tokens(a):
        L = sum(w * seen(d, a)["length"] for d, w in rl_train.items())
        return (B_RL["activation_monitor"] * eps_run + 50_000) * L * a["overhead_x"]

    items.append(Fixed("GPU: activations (RL probe runs + probe study), 1x H100 at $4.29/h",
                       tuple(gpu_hours_cost(act_tokens(RANGES_S[w]), tps) for w, tps in
                             (("low", 20_000), ("central", 8_000), ("high", 3_000))),
                       provider="gpu", group="6 other", note="3k-20k tokens/s forward with hooks on a 9B model"))
    items.append(Fixed("expert validation and paid judging (Task 4: 100 h x $80 + $2k)", (6_000, 10_000, 14_000),
                       provider="people", group="6 other", note="a design choice, not a token estimate"))
    items.append(Fixed("contracted reliability work (Task 4)", (2_000, 4_000, 6_000), provider="people", group="6 other"))
    return Budget(items, RANGES, extra={"runs": n_runs})



RANGES_S = {w: {k: v[i] for k, v in RANGES.items()} for i, w in enumerate(("low", "central", "high"))}


def with_cpu(b: Budget, name: str) -> Budget:
    """Add CPU for sandboxed tool calls and ground-truth checks: tool calls x 1-5 s x $0.05 per vCPU-hour."""
    calls = {}
    for w, a in RANGES_S.items():
        n = 0.0
        for it in b.items:
            if isinstance(it, Item):
                per_ep = sum(BEHAVIOUR[d]["turns"] for d in ("code", "sql", "firm", "chess")) / 4 * a["turns_x"] + 1
                n += it.n(a) * per_ep
        calls[w] = n
    usd = (cpu_cost(calls["low"], 1), cpu_cost(calls["central"], 2), cpu_cost(calls["high"], 5))
    b.items = [*b.items, Fixed(f"CPU: sandboxed tools and checks ({calls['central'] / 1e6:.1f}M calls)", usd,
                               provider="cpu", group="6 other" if name == "B" else "5 other")]
    return b


# ------------------------------------------------------------------------------ report


def money(x):
    return f"${x:,.0f}"


def md_table(df, cols, fmt):
    lines = ["| " + " | ".join(c for c, _ in cols) + " |", "|" + "|".join("---:" if k in fmt else "---" for _, k in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(k, str)(r[k]) if k in r and r[k] == r[k] else "" for _, k in cols) + " |")
    return "\n".join(lines)


def section(title, b: Budget):
    t = b.table()
    cols = [("Group", "group"), ("Item", "item"), ("Episodes", "episodes"), ("Sample Mtok", "sample_M"),
            ("Train Mtok", "train_M"), ("Paid by", "provider"), ("Low", "usd_low"), ("Central", "usd_central"),
            ("High", "usd_high")]
    fmt = {"episodes": lambda x: f"{x:,.0f}", "sample_M": lambda x: f"{x:,.0f}", "train_M": lambda x: f"{x:,.0f}",
           "usd_low": money, "usd_central": money, "usd_high": money}
    tot = b.totals()
    sens = b.sensitivity().head(6)
    iv = b.interval()
    out = [f"## {title}", "", md_table(t, cols, fmt), "",
           "**Totals by who pays**", "", "| Paid by | Low | Central | High |", "|---|---:|---:|---:|"]
    out += [f"| {i} | {money(r['low'])} | {money(r['central'])} | {money(r['high'])} |" for i, r in tot.iterrows()]
    out += ["", f"**Total, 80% interval** (assumptions drawn independently from triangular distributions on their "
            f"ranges): {money(iv[0.1])} - {money(iv[0.9])}, median {money(iv[0.5])}.",
            "", "**What the total depends on** (one assumption at a time at its low / high value)", "",
            "| Assumption | Range | Total at low | Total at high |", "|---|---|---:|---:|"]
    out += [f"| {r['assumption']} | {r['low']:g} / {r['central']:g} / {r['high']:g} | {money(r['total_at_low'])} | "
            f"{money(r['total_at_high'])} |" for _, r in sens.iterrows()]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remeasure", action="store_true")
    ap.add_argument("--out", default=str(REPORT))
    args = ap.parse_args()
    if args.remeasure or not PROMPTS.exists():
        logging.disable(logging.WARNING)
        warnings.filterwarnings("ignore")
        os.environ.setdefault("SO_ARENA_ALLOW_UNSANDBOXED", "1")  # dry run: simulated models run no code
        PROMPTS.write_text(json.dumps(remeasure(), indent=1))
    P.update(json.loads(PROMPTS.read_text()))
    bt, bb = with_cpu(design_t(), "T"), with_cpu(design_b(), "B")
    fit = {"steps": 110, "K": 16, "ckpts": 4}
    bt_fit = with_cpu(design_t(**fit), "T")
    bf = with_cpu(design_b(bon=(T_WORKER, T_OVERSEER), search=(T_WORKER, T_OVERSEER)), "B")
    alt = {w: design_t(W=w).total() for w in ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.8-27B")}
    per_run = {m: next(it for it in bt.items if isinstance(it, Item) and it.name.startswith(f"RL: {m} "))
               for m in T_CONDITIONS}
    a = bt.scenario("central")
    run_cost = {m: sum(it.by_provider(a).values()) / it.n(a) * STEPS * BATCH for m, it in per_run.items()}
    lines = [
        "# Grant budget: bottom-up cost estimates",
        "",
        f"Generated by `scripts/grant_budget.py`. Tinker prices: list prices from the snapshot of "
        f"{tinker_snapshot_date()} (`src/so_arena/data/tinker_prices.json`); API prices from the model registry "
        "(Claude Sonnet 5 $2 / $10, Haiku 4.5 $1 / $5 per Mtok, cached input at 10%); GPU: Lambda 1x H100 SXM $4.29/h.",
        "",
        "Cost = episodes x tokens per episode x price, per role. Prompt sizes are measured from the library's own "
        "prompts (dry run, `grant_budget_prompts.json`). Behaviour a dry run cannot see is an assumption with a "
        "low / central / high value (below); *Low* and *High* set all of them at once, so they bound rather than "
        "estimate. A role keeps its reasoning in its context (one trained sequence per episode); tokens added by "
        "others or by tools are uncached prefill, the rest are prompt-cache hits. Tinker trains every token of the "
        "sequence at its train price. Every token-priced line includes the overhead factor (pilots, failed runs).",
        "",
        "| Assumption | Low | Central | High |", "|---|---:|---:|---:|",
        *[f"| {k} | {v[0]:g} | {v[1]:g} | {v[2]:g} |" for k, v in RANGES.items()],
        "",
        "Per-domain behaviour at the central value (tool turns, tokens per tool output, diff shown to reviewers): "
        + "; ".join(f"{d} {b['turns']} / {b['obs']} / {b['diff']}" for d, b in BEHAVIOUR.items()) + ".",
        "",
        section(f"Design T: Tinker-first ({bt.extra['runs']} RL runs; worker {T_WORKER}, overseer {T_OVERSEER})", bt),
        "",
        "**Cost of one RL run** (central, before overhead; "
        f"{STEPS} steps x {BATCH} episodes): "
        + ", ".join(f"{m} {money(c / a['overhead_x'])}" for m, c in run_cost.items()) + ".",
        "",
        section(f"Design T, fitted to $50k of credits ({fit['steps']} RL steps, best-of-N K={fit['K']}, "
                f"{fit['ckpts']} evaluated checkpoints per run)", bt_fit),
        "",
        "**Other workers** (design T total, central): "
        + ", ".join(f"{w} {money(v)}" for w, v in alt.items()) + ".",
        "",
        section(f"Design B: broad (Task 4; API agents {B_AGENT}, judge {B_JUDGE}; RL on {B_RL_WORKER})", bb),
        "",
        section(f"Design B': Task 4 with best-of-N and prompt search on open models ({T_WORKER} / {T_OVERSEER}, "
                "Tinker); only the instructed-arm evaluation and forecasting on API models", bf),
        "",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
