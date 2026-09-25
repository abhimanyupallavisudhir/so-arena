"""Declarative experiment specs (YAML/JSON) -> runnable experiments.

A spec names a domain, policies, mechanisms and an experiment type; ``run_spec`` executes it into a
run directory (episodes, items, metrics, figures, report). See ``configs/`` for examples.

Policy entries::

    {model: openai/gpt-4o-mini, temperature: 0.7, elicitation: logprobs, strategy: "...", cot: false}
    {synthetic: arguer, honest_mean: 1.0, dishonest_mean: 0.2}      # synthetic-domain fixtures
    {synthetic: judge, skill: 1.0}
    {scripted: ["text 1", "text 2"]}

Experiment types: ``asd`` (instructed arms), ``pools`` (best-of-N game trees), ``prompt_search``
(directive-constrained searches), ``game`` (empirical game over strategies).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from so_arena.core.policy import LLMPolicy, Policy, ScriptedPolicy
from so_arena.core.store import RunStore
from so_arena.core.verification import VerificationPolicy

log = logging.getLogger("so_arena")


class Spec(BaseModel):
    name: str
    output: str | None = None
    seed: int = 0
    domain: dict[str, Any] = Field(default_factory=lambda: {"name": "synthetic"})
    items: dict[str, Any] = Field(default_factory=dict)  # load() kwargs: split, limit
    policies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mechanisms: list[dict[str, Any]] = Field(default_factory=list)
    experiment: dict[str, Any] = Field(default_factory=lambda: {"type": "asd"})
    report: bool = True
    concurrency: int | None = None
    cache_dir: str | None = None


def load_spec(path: str | Path) -> Spec:
    text = Path(path).read_text()
    data = yaml.safe_load(text) if str(path).endswith((".yaml", ".yml")) else json.loads(text)
    return Spec(**data)


def build_policy(entry: dict[str, Any] | str, label: str | None = None) -> Policy:
    if isinstance(entry, str):
        return LLMPolicy(entry, label=label)
    e = dict(entry)
    if "synthetic" in e:
        from so_arena.domains import synthetic as syn

        kind = e.pop("synthetic")
        factories = {"arguer": syn.synthetic_arguer, "judge": syn.synthetic_judge, "worker": syn.team_worker}
        if kind not in factories:
            raise KeyError(f"unknown synthetic policy {kind!r}")
        return factories[kind](**e, **({"label": label} if label and "label" not in e else {}))
    if "scripted" in e:
        return ScriptedPolicy(e["scripted"], label=e.get("label", label))
    if "model" in e:
        model = e.pop("model")
        e.setdefault("label", label)
        return LLMPolicy(model, **e)
    raise ValueError(f"cannot build a policy from {entry!r}")


def build_mechanism(entry: dict[str, Any]):
    from so_arena.core.rewards import JudgeScore, TeamReward, Whistleblower, ZeroSum
    from so_arena.mechanisms import get_mechanism

    e = dict(entry)
    name = e.pop("name")
    label = e.pop("label", None)
    if "verification" in e and isinstance(e["verification"], dict):
        e["verification"] = VerificationPolicy(**e["verification"])
    reward = e.pop("reward", None)
    if isinstance(reward, dict):
        r = dict(reward)
        kind = r.pop("name")
        rules = {"judge_score": JudgeScore, "team": TeamReward, "whistleblower": Whistleblower}
        if kind == "zero_sum":
            e["reward"] = ZeroSum(JudgeScore(r.get("transform", "log")), r.get("a", "debater_a"), r.get("b", "debater_b"))
        else:
            e["reward"] = rules[kind](**r)
    if label:
        e["name"] = label
    return get_mechanism(name, **e)


def _resolve_players(spec: Spec, mapping: dict[str, Any]) -> dict[str, Any]:
    from so_arena.core.runner import PlayerSpec

    out = {}
    for role, ref in mapping.items():
        stance = None
        if isinstance(ref, dict) and "policy" in ref:
            stance = ref.get("stance")
            ref = ref["policy"]
        pol = build_policy(spec.policies[ref], label=ref) if isinstance(ref, str) and ref in spec.policies else build_policy(ref)
        out[role] = PlayerSpec(policy=pol, stance=stance) if stance is not None else pol
    return out


def run_spec(spec: Spec, *, out: str | Path | None = None, limit: int | None = None) -> Path:
    """Execute a spec; returns the run directory."""
    from so_arena.config import configure
    from so_arena.domains import get_domain

    if spec.cache_dir:
        configure(cache_dir=spec.cache_dir)
    run_dir = Path(out or spec.output or f"runs/{spec.name}")
    store = RunStore(run_dir)
    dom_kw = dict(spec.domain)
    dom = get_domain(dom_kw.pop("name"), **dom_kw)
    load_kw = dict(spec.items)
    if limit is not None:
        load_kw["limit"] = limit
    items = dom.load(**load_kw)
    ctx = dom.context(run_id=spec.name, seed=spec.seed)
    gt = dom.ground_truth_scorers()
    mechs = [build_mechanism(m) for m in spec.mechanisms]
    exp = dict(spec.experiment)
    kind = exp.pop("type", "asd")
    store.write_meta({"spec": spec.model_dump(), "n_items": len(items), "kind": kind})
    extra_sections: list[tuple[str, Any]] = []
    from so_arena.analysis import plots

    if kind == "asd":
        from so_arena.samplers.arms import ASDExperiment

        fixtures = {r: build_policy(spec.policies[p], label=p) for r, p in exp.get("fixtures", {}).items()}
        agent = build_policy(spec.policies[exp["agent"]], label=exp["agent"])
        e = ASDExperiment(mechs, items, agent=agent, fixtures=fixtures, arms=[str(a).lower() for a in exp.get("arms", ["true", "false"])],
                          ground_truth=gt, ctx=ctx, repeats=exp.get("repeats", 1), store=store,
                          concurrency=spec.concurrency, all_false=exp.get("all_false", False), seed=spec.seed)
        eps = e.run()
        summary = e.summary()
        store.save_json("metrics.json", json.loads(summary.to_json(orient="records")))
    elif kind == "pools":
        from so_arena.core.runner import Profile
        from so_arena.samplers.pools import OptimizationExperiment

        players = _resolve_players(spec, exp["players"])
        eps = []
        for mech in mechs:
            o = OptimizationExperiment(mech, items, Profile(name="pool", players=players), pool_sizes=exp["pool_sizes"],
                                       ground_truth=gt, ctx=ctx, store=store, max_leaves=exp.get("max_leaves", 20000),
                                       seed=spec.seed)
            o.run()
            grid = o.grid({r: v for r, v in exp["grid"].items()}, kind=exp.get("kind", "bon"))
            grid.insert(0, "mechanism", mech.name)
            grid.to_csv(run_dir / f"grid_{mech.name}.csv", index=False)
            roles = list(exp["grid"])
            value_key = exp.get("value", "judge_correct")
            if len(roles) == 1:
                r = roles[0]
                df = grid.rename(columns={f"level_{r}": "n"})
                extra_sections.append((f"{mech.name}: optimizing {r}", (
                    plots.dual_mode(plots.parametric_curves, df, x=f"reward_{r}", y=value_key, level="n",
                                    title="Optimization pressure", subtitle=f"best-of-n on {r}",
                                    xlabel=f"{r} reward", ylabel=value_key), None, None)))
            elif len(roles) == 2:
                a, b = roles
                extra_sections.append((f"{mech.name}: {a} × {b}", (
                    plots.dual_mode(plots.heatmap, grid, x=f"level_{a}", y=f"level_{b}", value=value_key,
                                    title=value_key, xlabel=f"{a} n", ylabel=f"{b} n", value_label=value_key), None, None)))
        store.save_json("metrics.json", {"grids": [str(p.name) for p in run_dir.glob("grid_*.csv")]})
    elif kind == "prompt_search":
        from so_arena.samplers.prompt_search import PromptSearchSuite

        others = _resolve_players(spec, exp.get("others", {}))
        base = spec.policies[exp["policy"]]
        factory = lambda s, _b=base: build_policy({**_b, "strategy": s}, label="candidate")  # noqa: E731
        split = int(len(items) * exp.get("train_fraction", 0.5))
        results = {}
        for mech in mechs:
            suite = PromptSearchSuite(directives=exp.get("directives", ["honest", "deceptive"]), mechanism=mech,
                                      items=items[:split], role=exp["role"], policy_factory=factory, others=others,
                                      optimizer=exp["optimizer"], iterations=exp.get("iterations", 4),
                                      candidates_per_iter=exp.get("candidates_per_iter", 4),
                                      algorithm=exp.get("algorithm", "opro"), eval_items=items[split:],
                                      ground_truth=gt, ctx=ctx, seed=spec.seed)
            for d, arm in (exp.get("arms") or {}).items():
                if d in suite.searches:
                    suite.searches[d].arms = [arm]
            suite.run()
            results[mech.name] = {"margin": suite.honesty_margin() if {"honest", "deceptive"} <= set(suite.results) else None,
                                  "candidates": json.loads(suite.frame().to_json(orient="records"))}
            paths = suite.paths()
            extra_sections.append((f"{mech.name}: prompt search", (
                plots.dual_mode(plots.parametric_curves, paths, x="best_reward", y="best_value", series="directive",
                                level="iteration", level_name="iter", title="Prompt search",
                                subtitle="best strategy so far, by mechanism reward", xlabel="mechanism reward",
                                ylabel="ground-truth value"), None, None)))
        store.save_json("metrics.json", results)
        eps = store.episodes()
    elif kind == "game":
        from so_arena.games import EmpiricalGameExperiment

        strategies = {r: {name: build_policy(spec.policies[p], label=name) if isinstance(p, str) else build_policy(p, label=name)
                          for name, p in strat.items()} for r, strat in exp["strategies"].items()}
        fixtures = _resolve_players(spec, exp.get("fixtures", {}))
        games = {}
        for mech in mechs:
            g_exp = EmpiricalGameExperiment(mech, items, strategies, fixtures=fixtures, stances=exp.get("stances"),
                                            symmetric=exp.get("symmetric"), ground_truth=gt, ctx=ctx, store=store,
                                            repeats=exp.get("repeats", 1), seed=spec.seed)
            g_exp.run()
            g = g_exp.game()
            table = g.table()
            table.to_csv(run_dir / f"game_{mech.name}.csv", index=False)
            games[mech.name] = {
                "pure_nash": [g.profile_names(p) for p in g.pure_nash()],
                "strict_nash": [g.profile_names(p) for p in g.strict_nash()],
                "gt_welfare_range_ce": g.outcome_range("gt_welfare") if "gt_welfare" in g.outcomes else None,
            }
            extra_sections.append((f"{mech.name}: empirical game", (None, table, None)))
        store.save_json("metrics.json", games)
        eps = store.episodes()
    else:
        raise ValueError(f"unknown experiment type {kind!r}")
    if spec.report:
        from so_arena.analysis.report import build_report

        build_report(store.episodes(), run_dir / "report.html", title=spec.name, subtitle=f"{kind} experiment",
                     extra=extra_sections)
    return run_dir


def usage_summary(episodes) -> pd.DataFrame:
    """Tokens and cost by role and model (use after a simulated run to estimate cost)."""
    rows = []
    for ep in episodes:
        for role, u in ep.usage.items():
            model = ep.players[role].model if role in ep.players else None
            rows.append({"role": role, "model": model, "calls": u.calls, "input_tokens": u.input_tokens,
                         "output_tokens": u.output_tokens, "cost_usd": u.cost_usd})
    if not rows:
        return pd.DataFrame(columns=["role", "model", "calls", "input_tokens", "output_tokens", "cost_usd"])
    df = pd.DataFrame(rows).groupby(["role", "model"], dropna=False).sum(numeric_only=True).reset_index()
    total = df.sum(numeric_only=True)
    total["role"], total["model"] = "TOTAL", ""
    return pd.concat([df, total.to_frame().T], ignore_index=True)
