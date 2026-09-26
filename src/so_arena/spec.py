"""Declarative experiment specs (YAML/JSON) -> runnable experiments.

A spec names a domain, policies, mechanisms and an experiment type; ``run_spec`` executes it into a
run directory (episodes, items, metrics, figures, report, model usage). See ``configs/`` for examples.

Policy entries::

    {model: openai/gpt-4o-mini, temperature: 0.7, elicitation: logprobs, strategy: "...", cot: false}
    {synthetic: arguer, honest_mean: 1.0, dishonest_mean: 0.2}      # synthetic-domain fixtures
    {synthetic: judge, skill: 1.0}
    {scripted: ["text 1", "text 2"]}

Mechanism entries name a registered mechanism (``name:`` or ``type:``), an optional display ``label``
and constructor arguments; reward rules, verifiers and ground-truth scorers inside them are built from
``{type: ...}`` mappings by :mod:`so_arena.registry` (``so-arena list rewards|verifiers|scorers``)::

    {name: debate, rounds: 2, label: debate+zs, reward: {type: zero_sum, transform: prob},
     verification: {verifiers: [fact, {type: python, timeout: 2}], budget_per_role: 2}}

``ground_truth:`` (optional) lists the scorers to use instead of the domain's: ``domain`` for the domain's
own, types or ``{type: ...}`` mappings for others.

Experiment types (their keys: :data:`EXPERIMENT_KEYS`): ``asd`` (instructed arms), ``pools``
(best-of-N game trees), ``prompt_search`` (directive-constrained searches), ``game`` (empirical game
over strategies), ``paired`` (behaviour arms on stateful tasks, reviewed by each mechanism; behaviours
are ``{scripted: [labels]}``, ``{model: ..., arms: [labels]}`` for the domain's behaviour prompts, or
``{label: policy}``). Specs are checked before anything runs: a misspelled key is an error naming the
closest valid key, never a silently applied default.

A run directory holds one spec: ``run.json`` records it with its hash, and :func:`run_spec` refuses
to resume a directory that holds a different spec unless forced.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from so_arena.core.policy import LLMPolicy, Policy, ScriptedPolicy
from so_arena.core.store import RunStore
from so_arena.core.types import Usage
from so_arena.models.base import Model
from so_arena.registry import RegistryError, build, params as _params, spec_errors, unknown as _unknown

log = logging.getLogger("so_arena")


class SpecError(ValueError):
    """An invalid spec, or a run directory that holds a different spec."""


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a misspelled key is an error, not a silent default

    name: str
    output: str | None = None
    seed: int = 0
    domain: dict[str, Any] = Field(default_factory=lambda: {"name": "synthetic"})
    items: dict[str, Any] = Field(default_factory=dict)  # load() kwargs: split, limit, seed
    policies: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mechanisms: list[dict[str, Any]] = Field(default_factory=list)
    experiment: dict[str, Any] = Field(default_factory=lambda: {"type": "asd"})
    ground_truth: list[Any] | None = None  # scorers replacing the domain's ("domain" = the domain's own)
    report: bool = True
    concurrency: int | None = None
    cache_dir: str | None = None


# experiment type -> {key: required}; "type" is always allowed
EXPERIMENT_KEYS: dict[str, dict[str, bool]] = {
    "asd": {"agent": True, "fixtures": False, "arms": False, "repeats": False, "all_false": False},
    "pools": {"players": True, "pool_sizes": True, "grid": True, "kind": False, "value": False, "max_leaves": False,
              "min_coverage": False},
    "prompt_search": {"policy": True, "role": True, "optimizer": True, "others": False, "directives": False,
                      "arms": False, "train_fraction": False, "iterations": False, "candidates_per_iter": False,
                      "algorithm": False},
    "game": {"strategies": True, "fixtures": False, "stances": False, "symmetric": False, "repeats": False},
    "paired": {"behaviours": True, "fixtures": False, "honest": False, "repeats": False},
}
ITEMS_KEYS = ("split", "limit", "seed")
POLICY_KINDS = ("model", "synthetic", "scripted")
# what decides a run's results; output, report, concurrency and cache settings do not
RESULT_KEYS = ("name", "seed", "domain", "items", "policies", "mechanisms", "experiment")
# also decide results, but are hashed only when set, so specs written before they existed keep their hash
OPTIONAL_RESULT_KEYS = ("ground_truth",)


def _synthetic_factories() -> dict[str, Callable[..., Policy]]:
    from so_arena.domains import synthetic as syn

    return {"arguer": syn.synthetic_arguer, "judge": syn.synthetic_judge, "worker": syn.team_worker}


def _is_model_name(ref: str) -> bool:
    return "/" in ref or ref in ("mock", "human")


def _policy_errors(where: str, entry: Any) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"{where}: expected a mapping such as {{model: provider/name}}, got {entry!r}"]
    kinds = [k for k in POLICY_KINDS if k in entry]
    if len(kinds) != 1:
        return [f"{where} needs exactly one of {', '.join(POLICY_KINDS)} (it has {', '.join(map(str, entry)) or 'no keys'})"]
    kind = kinds[0]
    if kind == "scripted":
        valid: set[str] | None = {"scripted", "label"}
    elif kind == "model":
        valid = (_params(LLMPolicy.__init__) or set()) | {"model"}
    else:
        factories = _synthetic_factories()
        if entry["synthetic"] not in factories:
            return [_unknown(f"synthetic policy in {where}", entry["synthetic"], factories)]
        valid = _params(factories[entry["synthetic"]])
        valid = None if valid is None else valid | {"synthetic"}
    return [] if valid is None else [_unknown(f"key in {where}", k, valid) for k in entry if k not in valid]


def _reference_errors(spec: Spec, kind: str) -> list[str]:
    exp, errs = spec.experiment, []

    def mapping(key: str) -> Mapping[str, Any]:
        v = exp.get(key, {})
        if not isinstance(v, Mapping):
            errs.append(f"experiment.{key} must be a mapping, got {v!r}")
            return {}
        return v

    def named(ref: Any, where: str) -> None:  # the name of a policy entry
        if not isinstance(ref, str) or ref not in spec.policies:
            errs.append(_unknown(f"policy in {where}", ref, spec.policies))

    def player(ref: Any, where: str) -> None:  # a policy name, a model name, an entry, or {policy, stance}
        if isinstance(ref, Mapping) and "policy" in ref:
            errs.extend(_unknown(f"key in {where}", k, ("policy", "stance")) for k in ref if k not in ("policy", "stance"))
            ref = ref["policy"]
        if isinstance(ref, Mapping):
            errs.extend(_policy_errors(where, ref))
        elif not isinstance(ref, str) or (ref not in spec.policies and not _is_model_name(ref)):
            errs.append(_unknown(f"policy in {where}", ref, spec.policies) + "; model names look like provider/model")

    if kind == "asd":
        if "agent" in exp:
            named(exp["agent"], "experiment.agent")
        for r, p in mapping("fixtures").items():
            named(p, f"experiment.fixtures.{r}")
    elif kind == "pools":
        for r, p in mapping("players").items():
            player(p, f"experiment.players.{r}")
    elif kind == "prompt_search":
        # the searched policy's strategies are prompts, so it needs a model; the optimizer is a model
        # policy or a model name
        for key in ("policy", "optimizer"):
            ref = exp.get(key)
            is_model = key == "optimizer" and isinstance(ref, str) and ref not in spec.policies and _is_model_name(ref)
            if ref is None or is_model:
                continue
            named(ref, f"experiment.{key}")
            if isinstance(ref, str) and "model" not in spec.policies.get(ref, {"model": None}):
                errs.append(f"experiment.{key} must name a {{model: ...}} policy, not {ref!r}")
        for r, p in mapping("others").items():
            player(p, f"experiment.others.{r}")
    elif kind == "game":
        for r, strat in mapping("strategies").items():
            if not isinstance(strat, Mapping):
                errs.append(f"experiment.strategies.{r} must map strategy names to policies, got {strat!r}")
                continue
            for name, p in strat.items():
                if isinstance(p, Mapping):
                    errs.extend(_policy_errors(f"experiment.strategies.{r}.{name}", p))
                else:
                    named(p, f"experiment.strategies.{r}.{name}")
        for r, p in mapping("fixtures").items():
            player(p, f"experiment.fixtures.{r}")
    elif kind == "paired":
        beh = mapping("behaviours")
        if "scripted" in beh:
            if not isinstance(beh["scripted"], list) or len(beh) != 1:
                errs.append("experiment.behaviours: {scripted: [labels]} takes a list of the domain's scripted arms")
        elif "model" in beh:
            valid = (_params(LLMPolicy.__init__) or set()) | {"model", "arms"}
            errs.extend(_unknown("key in experiment.behaviours", k, valid) for k in beh if k not in valid)
            arms = beh.get("arms")
            if arms is not None and (not isinstance(arms, list) or not all(isinstance(a, str) for a in arms)):
                errs.append(f"experiment.behaviours.arms takes a list of the domain's behaviour labels, got {arms!r}")
        else:
            for label, ref in beh.items():
                player(ref, f"experiment.behaviours.{label}")
        for r, p in mapping("fixtures").items():
            player(p, f"experiment.fixtures.{r}")
    return errs


def mechanism_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """A spec's mechanism entry as a registry spec: ``name`` (the registered mechanism, as ``type``) and
    ``label`` (its display name, as ``name``); a reward written ``{name: judge_score, ...}`` (the older
    form) as ``{type: judge_score, ...}``."""
    e = dict(entry)
    if "type" not in e and "name" in e:
        e["type"] = e.pop("name")
    if "label" in e:
        e["name"] = e.pop("label")
    r = e.get("reward")
    if isinstance(r, Mapping) and "type" not in r and "name" in r:
        e["reward"] = {"type": r["name"], **{k: v for k, v in r.items() if k != "name"}}
    return e


def _mechanism_errors(entries: list[dict[str, Any]]) -> list[str]:
    from so_arena.mechanisms import MECHANISMS

    if not entries:
        return ["the spec lists no mechanisms (key 'mechanisms')"]
    errs = []
    for i, m in enumerate(entries):
        where = f"mechanisms[{i}]"
        if not isinstance(m, Mapping) or ("name" in m) == ("type" in m):
            errs.append(f"{where} needs a 'name' (one of {', '.join(sorted(MECHANISMS))}); a display name goes in 'label'")
            continue
        errs += spec_errors("mechanism", mechanism_entry(m), where)
    return errs


def _ground_truth_errors(entries: Any) -> list[str]:
    if entries is None:
        return []
    return [e for i, g in enumerate(entries) if g != "domain" for e in spec_errors("scorer", g, f"ground_truth[{i}]")]


def _invalid(source: str, errs: list[str]) -> SpecError:
    return SpecError(f"{source}: invalid spec\n  - " + "\n  - ".join(errs))


def _paired_arm_errors(dom: Any, beh: Mapping[str, Any]) -> list[str]:
    """Labels a paired experiment's behaviours name that the domain lacks - its scripted arms, or the
    behaviour prompts ``arms:`` selects - checked once the domain is built (a label is never dropped)."""
    if "scripted" in beh:
        what, where, labels, have = "scripted arm", "experiment.behaviours.scripted", beh["scripted"], dom.scripted_arms()
    elif "model" in beh and beh.get("arms") is not None:
        what, where, labels, have = "behaviour", "experiment.behaviours.arms", beh["arms"], dom.behaviours()
    else:
        return []
    if not labels:
        return [f"{where} lists no arms (the {dom.name} domain has: {', '.join(sorted(have)) or 'none'})"]
    return [_unknown(f"{dom.name} {what} in {where}", a, have) for a in labels if not isinstance(a, str) or a not in have]


def check_spec(spec: Spec, *, source: str = "spec") -> None:
    """Check a spec before anything runs: the experiment type and its keys (a misspelled key names the
    closest valid one), required keys, policy entries and the policies the experiment refers to,
    mechanism, reward and domain names. Raises :class:`SpecError` listing every problem."""
    if errs := _spec_errors(spec):
        raise _invalid(source, errs)


def _spec_errors(spec: Spec) -> list[str]:
    from so_arena.domains import list_domains

    errs: list[str] = []
    kind = spec.experiment.get("type", "asd")
    if kind not in EXPERIMENT_KEYS:
        errs.append(_unknown("experiment type", kind, EXPERIMENT_KEYS))
    else:
        keys = EXPERIMENT_KEYS[kind]
        errs += [_unknown(f"key in the {kind} experiment", k, [*keys, "type"]) for k in spec.experiment
                 if k != "type" and k not in keys]
        errs += [f"a {kind} experiment needs {k!r}" for k, required in keys.items() if required and k not in spec.experiment]
        errs += _reference_errors(spec, kind)
    for name, entry in spec.policies.items():
        errs += _policy_errors(f"policy {name!r}", entry)
    errs += _mechanism_errors(spec.mechanisms)
    errs += _ground_truth_errors(spec.ground_truth)
    domains = list_domains()
    if spec.domain.get("name") not in domains:
        errs.append(_unknown("domain", spec.domain.get("name"), domains))
    errs += [_unknown("items key", k, ITEMS_KEYS) for k in spec.items if k not in ITEMS_KEYS]
    return errs


def parse_spec(data: Any, *, source: str = "spec") -> Spec:
    """Build a :class:`Spec` from a mapping (e.g. parsed YAML), checked with :func:`check_spec`; every
    problem found is reported at once."""
    if not isinstance(data, Mapping):
        raise SpecError(f"{source}: a spec is a mapping of keys to values, got {type(data).__name__}")
    errs = []
    for k in data:
        if k in Spec.model_fields:
            continue
        owners = [t for t, keys in EXPERIMENT_KEYS.items() if k in keys]
        errs.append(f"key {k!r} belongs under 'experiment' (a key of {' and '.join(owners)} experiments)" if owners
                    else _unknown("key", k, Spec.model_fields))
    try:
        spec = Spec(**{k: v for k, v in data.items() if k in Spec.model_fields})
    except ValidationError as e:
        raise _invalid(source, errs + [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()]) from None
    if errs := errs + _spec_errors(spec):
        raise _invalid(source, errs)
    return spec


def load_spec(path: str | Path) -> Spec:
    text = Path(path).read_text()
    data = yaml.safe_load(text) if str(path).endswith((".yaml", ".yml")) else json.loads(text)
    return parse_spec(data, source=str(path))


def _result_part(d: Mapping[str, Any]) -> dict[str, Any]:
    # items.limit only changes how many items run, like --limit (reports cover this run's episodes)
    part = {k: d.get(k) for k in RESULT_KEYS}
    part.update({k: d[k] for k in OPTIONAL_RESULT_KEYS if d.get(k) is not None})
    part["items"] = {k: v for k, v in (part["items"] or {}).items() if k != "limit"}
    return part


def spec_hash(spec: Spec | Mapping[str, Any]) -> str:
    """Hash of what decides a spec's results (:data:`RESULT_KEYS`, without ``items.limit``)."""
    d = spec.model_dump(mode="json") if isinstance(spec, Spec) else spec
    return hashlib.sha256(json.dumps(_result_part(d), sort_keys=True, default=str).encode()).hexdigest()[:16]


def _earlier_specs(run_dir: Path, spec: Spec, force: bool) -> list[dict[str, Any]]:
    """Specs run into ``run_dir`` before, refusing (unless ``force``) if the last one differs from ``spec``."""
    meta = RunStore(run_dir).meta() if (run_dir / "run.json").exists() else {}
    old = meta.get("spec")
    if not isinstance(old, dict):
        return []
    history = list(meta.get("previous_specs") or [])
    old_hash, new_hash = meta.get("spec_hash") or spec_hash(old), spec_hash(spec)
    if old_hash == new_hash:
        return history
    if not force:
        a, b = _result_part(old), _result_part(spec.model_dump(mode="json"))
        dump = lambda x: json.dumps(x, sort_keys=True, default=str)  # noqa: E731
        changed = [k for k in RESULT_KEYS if dump(a[k]) != dump(b[k])]
        raise SpecError(
            f"{run_dir} holds a run of a different spec (changed: {', '.join(changed) or 'unknown'}; spec hash "
            f"{old_hash} -> {new_hash}). Resuming would mix its episodes with this spec's and overwrite its "
            "record. Run into another directory (--out, or `output:` in the spec), or pass --force (force=True) "
            "to resume anyway: episodes whose ids still match are reused - ids cover the mechanism "
            "configuration, item content, players and seed, not e.g. domain options that change tools or "
            "scorers - and run.json keeps the earlier spec under previous_specs.")
    return history + [{"spec_hash": old_hash, "spec": old}]


def build_policy(entry: dict[str, Any] | str, label: str | None = None) -> Policy:
    if isinstance(entry, str):
        return LLMPolicy(entry, label=label)
    e = dict(entry)
    if "synthetic" in e:
        factories = _synthetic_factories()
        kind = e.pop("synthetic")
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
    """A mechanism from a spec entry (:func:`mechanism_entry`), its reward rule, verification policy and
    any nested components built by :func:`so_arena.registry.build`."""
    try:
        return build("mechanism", mechanism_entry(entry))
    except RegistryError as e:
        raise SpecError(str(e)) from None


def build_ground_truth(entries: Sequence[Any] | None, dom: Any) -> list[Any]:
    """The scorers a spec's ``ground_truth`` lists (``"domain"``: the domain's own), else the domain's."""
    if entries is None:
        return dom.ground_truth_scorers()
    out: list[Any] = []
    for g in entries:
        out += dom.ground_truth_scorers() if g == "domain" else [build("scorer", g)]
    return out


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


def _model_of(player: Any) -> str | None:
    """Model name behind a player spec, policy or model name (None for scripted/synthetic policies)."""
    from so_arena.core.runner import PlayerSpec

    if isinstance(player, PlayerSpec):
        player = player.policy
    if isinstance(player, Policy):
        return player.model_name
    return player if isinstance(player, str) else None


def _usage_row(role: str, model: str | None, u: Usage) -> dict[str, Any]:
    return {"role": role, "model": model, "calls": u.calls, "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens, "cost_usd": u.cost_usd}


class _Metered(Model):
    """A model that adds up the usage of its calls: for calls outside episodes (the prompt optimizer),
    which no episode records.

    Simulated replies (dry runs) are filler text in which a search would find no strategy, so it
    would evaluate its seed strategies only and the estimate would miss most of the cost; they are
    reshaped into ``proposals`` distinct proposals, the most a search asks for at a time.
    """

    def __init__(self, inner: Model, *, proposals: int = 0):
        self.inner, self.name, self.supports_logprobs = inner, inner.name, inner.supports_logprobs
        self.proposals = proposals
        self.usage = Usage()

    async def generate(self, messages, options=None, *, sample_index=0):
        out = await self.inner.generate(messages, options, sample_index=sample_index)
        self.usage = self.usage + out.usage
        if self.proposals and out.metadata.get("simulated"):
            words, k = out.text.split(), self.proposals
            parts = [" ".join(words[i::k]) for i in range(k)]
            out = out.model_copy(update={"text": "\n".join(
                f"<hypothesis>simulated</hypothesis><rationale>simulated</rationale>"
                f"<strategy>[{sample_index}.{i}] {part}</strategy>" for i, part in enumerate(parts))})
        return out


def run_spec(spec: Spec, *, out: str | Path | None = None, limit: int | None = None, force: bool = False) -> Path:
    """Execute a spec; returns the run directory.

    A directory whose ``run.json`` holds a different spec (see :func:`spec_hash`) is refused unless
    ``force`` (then the run resumes there and run.json keeps the earlier spec in ``previous_specs``).
    Metrics, the report, ``usage.csv`` (every model call, including game-tree pools and the prompt
    optimizer) and the counts in run.json cover this run's episodes only.
    """
    from so_arena.config import configure
    from so_arena.domains import get_domain

    check_spec(spec)
    if spec.cache_dir:
        configure(cache_dir=spec.cache_dir)
    run_dir = Path(out or spec.output or f"runs/{spec.name}")
    earlier = _earlier_specs(run_dir, spec, force)
    store = RunStore(run_dir)
    dom_kw = dict(spec.domain)
    dom = get_domain(dom_kw.pop("name"), **dom_kw)
    if spec.experiment.get("type") == "paired" and (errs := _paired_arm_errors(dom, spec.experiment["behaviours"])):
        raise _invalid("spec", errs)  # before anything is written to the run directory
    load_kw = dict(spec.items)
    if limit is not None:
        load_kw["limit"] = limit
    items = dom.load(**load_kw)
    stateful = dom.environment() is not None
    ctx = dom.context(run_id=spec.name, seed=spec.seed, **({"states": run_dir / "states"} if stateful else {}))
    gt = build_ground_truth(spec.ground_truth, dom)
    mechs = [build_mechanism(m) for m in spec.mechanisms]
    exp = dict(spec.experiment)
    kind = exp.pop("type", "asd")
    meta: dict[str, Any] = {"spec": spec.model_dump(mode="json"), "spec_hash": spec_hash(spec), "n_items": len(items),
                            "kind": kind, **({"previous_specs": earlier} if earlier else {})}
    store.write_meta(meta)
    extra_sections: list[tuple[str, Any]] = []
    eps: list[Any] = []  # this run's episodes (a forced resume may leave other specs' episodes in the store)
    trees: list[Any] = []
    extra_usage: list[dict[str, Any]] = []  # model calls no episode records
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
        for mech in mechs:
            o = OptimizationExperiment(mech, items, Profile(name="pool", players=players), pool_sizes=exp["pool_sizes"],
                                       ground_truth=gt, ctx=ctx, max_leaves=exp.get("max_leaves", 20000),
                                       seed=spec.seed)
            trees += o.run()
            grid = o.grid({r: v for r, v in exp["grid"].items()}, kind=exp.get("kind", "bon"),
                          min_coverage=float(exp.get("min_coverage", 0.5)))
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
        with open(run_dir / "trees.jsonl", "w") as f:  # every mechanism's trees (one experiment per mechanism)
            for t in trees:
                f.write(t.model_dump_json() + "\n")
        # a tree's usage counts every sampled candidate once, like its leaf episodes' usage summed
        extra_usage += [_usage_row(r, _model_of(players.get(r)), Usage.model_validate(u))
                        for t in trees for r, u in t.usage.items()]
        store.save_json("metrics.json", {"grids": [f"grid_{m.name}.csv" for m in mechs]})
    elif kind == "prompt_search":
        from so_arena.models.base import get_model
        from so_arena.samplers.prompt_search import PromptSearchSuite

        others = _resolve_players(spec, exp.get("others", {}))
        base = spec.policies[exp["policy"]]
        factory = lambda s, _b=base: build_policy({**_b, "strategy": s}, label="candidate")  # noqa: E731
        opt = exp["optimizer"]
        optimizer = _Metered(get_model(spec.policies[opt]["model"] if opt in spec.policies else opt),
                             proposals=exp.get("candidates_per_iter", 4))
        split = int(len(items) * exp.get("train_fraction", 0.5))
        results = {}
        for mech in mechs:
            suite = PromptSearchSuite(directives=exp.get("directives", ["honest", "deceptive"]), mechanism=mech,
                                      items=items[:split], role=exp["role"], policy_factory=factory, others=others,
                                      optimizer=optimizer, iterations=exp.get("iterations", 4),
                                      candidates_per_iter=exp.get("candidates_per_iter", 4),
                                      algorithm=exp.get("algorithm", "opro"), eval_items=items[split:],
                                      ground_truth=gt, ctx=ctx, seed=spec.seed)
            for d, arm in (exp.get("arms") or {}).items():
                if d in suite.searches:
                    suite.searches[d].arms = [arm]
            suite.run()
            for search in suite.searches.values():  # searches run without a store: record their episodes
                for batch in search.episodes.values():
                    eps += batch
                    for ep in batch:
                        store.append(ep)
            results[mech.name] = {"margin": suite.honesty_margin() if {"honest", "deceptive"} <= set(suite.results) else None,
                                  "candidates": json.loads(suite.frame().to_json(orient="records"))}
            paths = suite.paths()
            extra_sections.append((f"{mech.name}: prompt search", (
                plots.dual_mode(plots.parametric_curves, paths, x="best_reward", y="best_value", series="directive",
                                level="iteration", level_name="iter", title="Prompt search",
                                subtitle="best strategy so far, by mechanism reward", xlabel="mechanism reward",
                                ylabel="ground-truth value"), None, None)))
        extra_usage.append(_usage_row("optimizer", optimizer.name, optimizer.usage))
        store.save_json("metrics.json", results)
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
            eps += g_exp.run()
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
    elif kind == "paired":
        from so_arena.samplers.paired import PairedWorkExperiment, behaviour_policies

        fixtures = _resolve_players(spec, exp.get("fixtures", {}))
        beh = exp["behaviours"]
        if "scripted" in beh:  # labels checked by _paired_arm_errors
            arms = dom.scripted_arms()
            behaviours = {m: arms[m]() for m in beh["scripted"]}
        elif "model" in beh:
            b = dict(beh)
            model, only = b.pop("model"), b.pop("arms", None)
            prompts = {k: v for k, v in dom.behaviours().items() if only is None or k in only}
            behaviours = behaviour_policies(model, prompts, **b)
        else:
            behaviours = {label: build_policy(spec.policies[ref], label=label) if isinstance(ref, str) and ref in spec.policies
                          else build_policy(ref, label=label) for label, ref in beh.items()}
        e = PairedWorkExperiment(mechs, items, behaviours=behaviours, fixtures=fixtures, ctx=ctx, ground_truth=gt,
                                 honest=exp.get("honest", "honest"), repeats=exp.get("repeats", 1), store=store,
                                 concurrency=spec.concurrency, seed=spec.seed)
        eps = e.run()
        summary = e.summary()
        store.save_json("metrics.json", json.loads(summary.to_json(orient="records")))
        rows = summary[summary["arm"] != "worst"] if not summary.empty else summary
        if not rows.empty:
            extra_sections.append(("ASD against each deceptive arm", (
                plots.dual_mode(plots.heatmap, rows, x="mechanism", y="arm", value="asd", center=0.0, decimals=2,
                                title="Does the protocol pay honest work more?", xlabel="protocol", ylabel="",
                                value_label="ASD"), summary, None)))
    else:  # unreachable after check_spec
        raise SpecError(f"unknown experiment type {kind!r}")
    usage_summary(eps, extra_usage).to_csv(run_dir / "usage.csv", index=False)
    store.write_meta({**meta, "n_episodes": len(eps), "n_trees": len(trees)})
    if spec.report:
        from so_arena.analysis.report import build_report

        build_report(eps, run_dir / "report.html", title=spec.name, subtitle=f"{kind} experiment",
                     extra=extra_sections)
    return run_dir


def usage_summary(episodes, extra: Iterable[dict[str, Any]] = ()) -> pd.DataFrame:
    """Tokens and cost by role and model (use after a simulated run to estimate cost); ``extra`` rows
    (``role, model, calls, input_tokens, output_tokens, cost_usd``) add calls no episode records."""
    rows = []
    for ep in episodes:
        for role, u in ep.usage.items():
            model = ep.players[role].model if role in ep.players else None
            rows.append(_usage_row(role, model, u))
    rows += list(extra)
    if not rows:
        return pd.DataFrame(columns=["role", "model", "calls", "input_tokens", "output_tokens", "cost_usd"])
    df = pd.DataFrame(rows).groupby(["role", "model"], dropna=False).sum(numeric_only=True).reset_index()
    total = df.sum(numeric_only=True)
    total["role"], total["model"] = "TOTAL", ""
    return pd.concat([df, total.to_frame().T], ignore_index=True)


def run_usage(run_dir: str | Path) -> pd.DataFrame:
    """The usage table :func:`run_spec` wrote (``usage.csv``), else one from the stored episodes."""
    p = Path(run_dir) / "usage.csv"
    return pd.read_csv(p, keep_default_na=False) if p.exists() else usage_summary(RunStore(run_dir).episodes())


def run_counts(run_dir: str | Path) -> str:
    """What the last run in ``run_dir`` produced, e.g. ``"12 episodes"`` or ``"3 sampled game trees"``."""
    meta = RunStore(run_dir).meta()
    if meta.get("n_trees"):
        return f"{meta['n_trees']} sampled game trees"
    n = meta["n_episodes"] if "n_episodes" in meta else len(RunStore(run_dir).episodes())
    return f"{n} episodes"
