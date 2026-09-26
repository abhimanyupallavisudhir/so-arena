"""Registry of buildable components: reward rules, verifiers, ground-truth scorers and mechanisms.

Specs name a component by its registered type and give its constructor arguments::

    {type: judge_score, transform: brier}
    {type: sum, rules: [{type: judge_score}, {type: scaled, rule: {type: monitor_penalty}, c: 0.5}]}
    {type: debate, rounds: 2, reward: {type: zero_sum, transform: prob},
     verification: {verifiers: [fact, {type: python, timeout: 2}], budget_per_role: 2}}

Nested mappings with a ``type`` are built recursively. The kind of a nested component follows from the
argument it is passed as (``reward``, ``inner``, ``rule`` and ``rules`` take reward rules; ``verifiers``
takes verifiers; ``ground_truth`` and ``scorers`` take scorers), else from its type name when exactly one
kind registers it. A ``verification`` mapping becomes a
:class:`~so_arena.core.verification.VerificationPolicy`. Strings are never built: ``verifiers: [fact]``
names a verifier of the run's domain, as before. Components whose arguments are Python objects (a
function, an audit oracle) are built from Python, where any value can be passed as it is.

Register your own with :func:`register` (mechanisms: :func:`so_arena.mechanisms.register_mechanism`,
which this registry reads). Entries are ``"module:attribute"`` strings, imported on first use, so a
domain's optional dependencies are only needed when one of its components is built.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
from collections.abc import Callable, Iterable, Mapping
from typing import Any

KINDS = ("mechanism", "reward", "verifier", "scorer")


def _zero_sum(inner: Any = None, a: str = "debater_a", b: str = "debater_b", transform: str = "log"):
    """:class:`~so_arena.core.rewards.ZeroSum` of ``inner`` (default: the judge score with ``transform``)."""
    from so_arena.core.rewards import JudgeScore, ZeroSum

    return ZeroSum(inner if inner is not None else JudgeScore(transform), a, b)


_P = "so_arena"
_REGISTRY: dict[str, dict[str, str | Callable[..., Any]]] = {
    "reward": {
        "none": f"{_P}.core.rewards:NoReward",
        "sum": f"{_P}.core.rewards:Sum",
        "scaled": f"{_P}.core.rewards:Scaled",
        "constant": f"{_P}.core.rewards:Constant",
        "judge_score": f"{_P}.core.rewards:JudgeScore",
        "zero_sum": _zero_sum,
        "from_outcome": f"{_P}.core.rewards:FromOutcome",
        "team": f"{_P}.core.rewards:TeamReward",
        "whistleblower": f"{_P}.core.rewards:Whistleblower",
        "monitor_penalty": f"{_P}.core.rewards:MonitorPenalty",
        "random_audit": f"{_P}.core.rewards:RandomAudit",
        "judge_audit": f"{_P}.core.rewards:JudgeAuditScore",
        "resolution": f"{_P}.core.rewards:ResolutionScore",
        "market_scoring": f"{_P}.mechanisms.markets:MarketScoringReward",
        "monitored": f"{_P}.mechanisms.work:MonitoredReward",
        "preference": f"{_P}.mechanisms.preference:PreferenceScore",
        "prover": f"{_P}.mechanisms.preference:ProverReward",
        "rating": f"{_P}.mechanisms.markets:rating_reward",
    },
    "verifier": {
        "quote": f"{_P}.core.verification:QuoteVerifier",
        "python": f"{_P}.core.verification:PythonExecVerifier",
        "fact_checker": f"{_P}.core.verification:ModelFactChecker",
        "command": f"{_P}.core.state:CommandClaimVerifier",
        "protected_command": f"{_P}.core.state:ProtectedCommandVerifier",
        "query": f"{_P}.core.state:QueryClaimVerifier",
        "bit": f"{_P}.domains.hidden_bits:BitVerifier",
        "chess_line": f"{_P}.domains.chess:ChessLineVerifier",
        "chess_eval": f"{_P}.domains.chess:ChessEvalVerifier",
        "code_exec": f"{_P}.domains.code:CodeExecVerifier",
        "visible_tests": f"{_P}.domains.code:VisibleTestsVerifier",
        "repo_tests": f"{_P}.domains.repo:RepoTestsVerifier",
        "lean_parse": f"{_P}.domains.lean:LeanParseVerifier",
        "lean": f"{_P}.domains.lean:LeanVerifier",
        "sql": f"{_P}.domains.sql:SQLVerifier",
    },
    "scorer": {
        "stance_value": f"{_P}.core.ground_truth:StanceValue",
        "label_value": f"{_P}.core.ground_truth:LabelValue",
        "judge_correctness": f"{_P}.core.ground_truth:JudgeCorrectness",
        "model_audit": f"{_P}.core.ground_truth:ModelAudit",
        "manipulation_check": f"{_P}.core.ground_truth:PositionFollowed",
        "arm_value": f"{_P}.samplers.paired:ArmValue",
        "work_review_info": f"{_P}.samplers.paired:WorkReviewInfo",
        "bit_claims": f"{_P}.domains.hidden_bits:BitClaims",
        "best_move": f"{_P}.domains.chess:BestMoveScorer",
        "hidden_tests": f"{_P}.domains.code:HiddenTests",
        "team_hidden_tests": f"{_P}.domains.code:TeamHiddenTests",
        "repo_audit": f"{_P}.domains.repo:RepoAudit",
        "repo_team_audit": f"{_P}.domains.repo:RepoTeamAudit",
        "firm_audit": f"{_P}.domains.firm:FirmAudit",
        "forecast_score": f"{_P}.domains.forecasting:ForecastScore",
        "sql_answer": f"{_P}.domains.sql:SQLAnswerScorer",
    },
}

# argument name -> the kind of component it takes (a mapping with a ``type``, or a list of them)
FIELD_KINDS = {"reward": "reward", "inner": "reward", "rule": "reward", "rules": "reward",
               "verifiers": "verifier", "ground_truth": "scorer", "scorers": "scorer"}


class RegistryError(ValueError):
    """An unknown component type, or a spec its constructor cannot take."""


def _table(kind: str) -> dict[str, Any]:
    if kind == "mechanism":
        from so_arena.mechanisms import MECHANISMS

        return MECHANISMS  # type: ignore[return-value]
    if kind not in _REGISTRY:
        raise RegistryError(unknown("component kind", kind, KINDS))
    return _REGISTRY[kind]


def register(kind: str, name: str | None = None, obj: Any = None) -> Any:
    """Register a component: ``register("reward", "my_rule", MyRule)``, or as a class decorator
    (``@register("verifier", "my_check")``); the name defaults to the class's ``name`` attribute."""

    def deco(o: Any) -> Any:
        key = name or getattr(o, "name", None) or o.__name__.lower()
        if kind == "mechanism":
            from so_arena.mechanisms import register_mechanism

            register_mechanism(o, key)
        else:
            _table(kind)[key] = o
        return o

    return deco if obj is None else deco(obj)


def names(kind: str) -> list[str]:
    """The registered types of a kind (``"reward"``, ``"verifier"``, ``"scorer"``, ``"mechanism"``)."""
    return sorted(_table(kind))


def unknown(what: str, key: Any, valid: Iterable[Any]) -> str:
    """An error message naming the closest valid key."""
    names_ = sorted(map(str, valid))
    close = difflib.get_close_matches(str(key), names_, n=1, cutoff=0.6)
    return (f"unknown {what} {key!r}" + (f" - did you mean {close[0]!r}?" if close else "")
            + f" (valid: {', '.join(names_) or 'none'})")


def params(fn: Callable[..., Any]) -> set[str] | None:
    """Keyword arguments ``fn`` (a function or a class) accepts; None: any (it takes ``**kwargs``)."""
    ps = inspect.signature(fn).parameters.values()
    if any(p.kind == p.VAR_KEYWORD for p in ps):
        return None
    return {p.name for p in ps if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)} - {"self"}


def _required(fn: Callable[..., Any]) -> set[str]:
    return {p.name for p in inspect.signature(fn).parameters.values()
            if p.default is p.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self"}


def resolve(kind: str, type_: str) -> Callable[..., Any]:
    """The class or factory registered as ``type_``; a ``"package.module:Name"`` reference also works."""
    table = _table(kind)
    if type_ not in table:
        if isinstance(type_, str) and ":" in type_:
            mod, attr = type_.split(":", 1)
            return getattr(importlib.import_module(mod), attr)
        raise RegistryError(unknown(kind, type_, table))
    ref = table[type_]
    if isinstance(ref, str):
        mod, attr = ref.split(":", 1)
        try:
            ref = getattr(importlib.import_module(mod), attr)
        except ModuleNotFoundError as e:
            raise RegistryError(f"{kind} {type_!r} is unavailable: {e} (a missing optional dependency)") from None
        table[type_] = ref
    return ref


def kind_of(type_: str) -> str | None:
    """The kind registering ``type_``, if exactly one does."""
    kinds = [k for k in KINDS if type_ in _table(k)]
    return kinds[0] if len(kinds) == 1 else None


def _nested_kind(key: str, value: Mapping[str, Any]) -> str | None:
    return FIELD_KINDS.get(key) or kind_of(str(value.get("type")))


# kinds whose bare strings are types (verifier strings name the domain's verifiers)
_STRING_KINDS = ("reward", "scorer")


def _build_value(key: str, v: Any) -> Any:
    if isinstance(v, str) and FIELD_KINDS.get(key) in _STRING_KINDS:
        return build(FIELD_KINDS[key], v)
    if isinstance(v, Mapping) and "type" in v:
        kind = _nested_kind(key, v)
        if kind is None:
            raise RegistryError(f"{key}: cannot tell which kind of component {v.get('type')!r} is "
                                f"(registered as: {[k for k in KINDS if v.get('type') in _table(k)] or 'nothing'})")
        return build(kind, v)
    if key == "verification" and isinstance(v, Mapping):
        from so_arena.core.verification import VerificationPolicy

        return VerificationPolicy(**{k: _build_value(k, x) for k, x in v.items()})
    if isinstance(v, list) and key in FIELD_KINDS:
        return [_build_value(key, x) for x in v]
    return v


def build(kind: str, spec: Any) -> Any:
    """Build a component from ``{"type": ..., **kwargs}`` (or a bare type name), nested components included.
    Anything that is not a spec (an already-built object) is returned as it is."""
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, Mapping):
        return spec
    if "type" not in spec:
        raise RegistryError(f"a {kind} spec needs a 'type' (one of {', '.join(names(kind))}), got {dict(spec)!r}")
    kw = dict(spec)
    factory = resolve(kind, kw.pop("type"))
    return factory(**{k: _build_value(k, v) for k, v in kw.items()})


def spec_errors(kind: str, spec: Any, where: str) -> list[str]:
    """Every problem with a component spec, checked without building it: unknown types (naming the closest),
    arguments the constructor does not take, missing required ones - recursively for nested components."""
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, Mapping):
        return []  # an already-built object
    if "type" not in spec:
        return [f"{where} needs a 'type' (one of {', '.join(names(kind))})"]
    try:
        factory = resolve(kind, spec["type"])
    except RegistryError as e:
        return [f"{where}: {e}"]
    errs: list[str] = []
    valid = params(factory)
    given = set(spec) - {"type"}
    if valid is not None:
        errs += [unknown(f"key in {where} ({spec['type']})", k, valid) for k in sorted(given - valid)]
        errs += [f"{where} ({spec['type']}) needs {k!r}" for k in sorted(_required(factory) - given)]
    for k, v in spec.items():
        errs += _value_errors(k, v, f"{where}.{k}")
    return errs


def _value_errors(key: str, v: Any, where: str) -> list[str]:
    if isinstance(v, str) and FIELD_KINDS.get(key) in _STRING_KINDS:
        return spec_errors(FIELD_KINDS[key], v, where)
    if isinstance(v, Mapping) and "type" in v:
        kind = _nested_kind(key, v)
        if kind is None:
            return [f"{where}: {unknown('component type', v.get('type'), [n for k in KINDS for n in _table(k)])}"]
        return spec_errors(kind, v, where)
    if key == "verification" and isinstance(v, Mapping):
        from so_arena.core.verification import VerificationPolicy

        fields = set(VerificationPolicy.model_fields)
        errs = [unknown(f"key in {where}", k, fields) for k in v if k not in fields]
        for k, x in v.items():
            errs += _value_errors(k, x, f"{where}.{k}")
        return errs
    if isinstance(v, list) and key in FIELD_KINDS:
        return [e for i, x in enumerate(v) for e in _value_errors(key, x, f"{where}[{i}]")]
    return []
