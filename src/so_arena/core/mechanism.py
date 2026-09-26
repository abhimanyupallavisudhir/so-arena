"""Mechanisms: oversight protocols that also assign rewards to agents.

A scalable-oversight mechanism, in the sense used here, is any protocol (control workflow) together
with a *reward rule* that scores each trainable agent - so that it can be used for training. Roles
without rewards (``trainable=False``) are fixtures with fixed behaviour, e.g. a trusted weak judge.

A :class:`Mechanism` is the *game form* (:meth:`protocol`: who acts when, what they see, which tools
and verifiers they have, what the outcome is) plus a :class:`~so_arena.core.rewards.RewardRule`
(the payments). Keeping the two separate matters: the same protocol with a different reward rule is
a different mechanism (debate with log-score vs. win/lose rewards; a team with a common reward vs.
one with whistleblower bounties), and reward rules can be re-applied to recorded episodes post hoc.
"""

from __future__ import annotations

import abc
import datetime as _dt
import enum
import functools
import hashlib
import json
import os
import traceback
import types
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field

from so_arena.core.game import BranchController, Game, Player, RunContext, Turn
from so_arena.core.items import TaskItem
from so_arena.core.policy import Policy
from so_arena.core.state import STATE_ACCESS, StateAccess
from so_arena.core.types import Usage
from so_arena.core.verification import VerificationPolicy, Verifier

if TYPE_CHECKING:
    from so_arena.core.rewards import RewardRule

RoleKind = Literal["agent", "judge", "monitor", "auditor", "grader", "market", "reporter"]


def _qualname(x: Any) -> str:
    return f"{getattr(x, '__module__', None) or ''}.{getattr(x, '__qualname__', None) or getattr(x, '__name__', '?')}"


def describe_config(obj: Any, _depth: int = 8, _seen: frozenset[int] = frozenset()) -> Any:
    """A plain, process-independent description of a configuration value (for config hashes and logs).

    Scalars stay as they are and containers are described element-wise; pydantic models by their
    fields; functions by qualified name plus the plain values they close over or default to (see
    :func:`_describe_function`: a function is identified by its name and parameters, not its code); reward
    rules and verifiers - the parts of a mechanism - by class and public attributes; policies by their
    ``describe()``; any other object only by class and ``name`` (its attributes may hold clients,
    caches or other run state). Never contains memory addresses, so hashes agree across processes.
    """
    from so_arena.core.rewards import RewardRule

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, enum.Enum):
        return f"{_qualname(type(obj))}.{obj.name}"
    if isinstance(obj, (bytes, bytearray)):
        return "sha256:" + hashlib.sha256(bytes(obj)).hexdigest()[:16]
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if id(obj) in _seen or _depth <= 0:
        return {"class": _qualname(type(obj))}
    seen = _seen | {id(obj)}

    def sub(x: Any) -> Any:
        return describe_config(x, _depth, seen)

    def deeper(x: Any) -> Any:
        return describe_config(x, _depth - 1, seen)

    if isinstance(obj, Mapping):
        return {k if isinstance(k, str) else json.dumps(sub(k), sort_keys=True): sub(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sub(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((sub(x) for x in obj), key=lambda d: json.dumps(d, sort_keys=True))
    if isinstance(obj, functools.partial):
        return {"partial": sub(obj.func), "args": sub(obj.args), "keywords": sub(obj.keywords)}
    if isinstance(obj, types.MethodType):
        return {"method": _qualname(obj.__func__), "of": deeper(obj.__self__)}
    if isinstance(obj, types.FunctionType):
        return _describe_function(obj, deeper)
    if isinstance(obj, (types.BuiltinFunctionType, type)):
        return _qualname(obj)
    # options added after a class's first release are left out at their defaults (``HASH_OMIT_DEFAULTS``), so the
    # configurations that existed before keep their hashes - and resumable run stores their episodes
    omit = getattr(type(obj), "HASH_OMIT_DEFAULTS", {})
    if isinstance(obj, BaseModel):
        fields = {f: getattr(obj, f) for f in type(obj).model_fields}
        return {"class": _qualname(type(obj)),
                **{f: deeper(v) for f, v in fields.items() if not (f in omit and v == omit[f])}}
    if isinstance(obj, Policy):
        return deeper(obj.describe())
    if isinstance(obj, (RewardRule, Verifier)):
        attrs = {k: v for k, v in getattr(obj, "__dict__", {}).items()
                 if not k.startswith("_") and not (k in omit and v == omit[k])}
        return {"class": _qualname(type(obj)), **{k: deeper(attrs[k]) for k in sorted(attrs)}}
    out: dict[str, Any] = {"class": _qualname(type(obj))}
    if isinstance(getattr(obj, "name", None), str):
        out["name"] = obj.name
    return out


def _describe_function(fn: types.FunctionType, describe: Callable[[Any], Any]) -> Any:
    """A function's qualified name, plus the plain values it closes over or defaults to: its behaviour
    depends on them, so ``length_penalty(1.0)`` and ``length_penalty(100.0)`` - or an audit oracle built
    with another noise level - must give different config hashes and episode ids (a resumed store would
    otherwise return episodes rewarded under the old rule).

    Only scalars, strings, functions, classes and tuples of them count: mutable closure values (a list of
    calls, a cache) are usually run state, which must not change episode ids between a run and its
    resumption.
    """
    from so_arena.core.policy import _plain

    code = fn.__code__
    values: dict[str, Any] = {}
    for name, cell in zip(code.co_freevars, fn.__closure__ or ()):
        try:
            values[name] = cell.cell_contents
        except ValueError:  # an empty cell (a local assigned after the function was defined)
            continue
    positional = code.co_varnames[: code.co_argcount]
    values.update(zip(positional[len(positional) - len(fn.__defaults__ or ()):], fn.__defaults__ or ()))
    values.update(fn.__kwdefaults__ or {})
    params = {k: describe(v) for k, v in sorted(values.items()) if _plain(v) and v is not fn}
    return {"function": _qualname(fn), "params": params} if params else _qualname(fn)


class RoleSpec(BaseModel):
    name: str
    kind: RoleKind = "agent"
    trainable: bool = True
    description: str = ""
    affordances: list[str] = Field(default_factory=list)  # keys of TaskItem.private it can read
    tools: list[str] = Field(default_factory=list)  # tool names from the run context
    sees_reasoning_of: list[str] = Field(default_factory=list)  # roles whose CoT it sees ("*" = all)
    required: bool = True
    title: str | None = None
    # access to the episode's state (repository, database, ...), if it has one: "write" acts on it,
    # "read" acts on a throwaway copy (e.g. a reviewer running the tests), "none" sees only messages
    state_access: StateAccess = "none"


class Outcome(BaseModel):
    """The result of a protocol run."""

    decision: str | None = None  # e.g. the judge's chosen answer, "accept"/"reject"
    probs: dict[str, float] | None = None  # the deciding role's final distribution
    output: Any = None  # free-form final output (e.g. the accepted artifact)
    data: dict[str, Any] = Field(default_factory=dict)  # mechanism-specific (reports, prices, ...)


class PlayerRecord(BaseModel):
    policy_id: str
    label: str | None = None
    stance: str | None = None
    model: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class Episode(BaseModel):
    """Everything recorded about one run of a mechanism on one item under one profile."""

    id: str
    run_id: str = "run"
    item_id: str
    domain: str = "generic"
    mechanism: str
    mechanism_config: dict[str, Any] = Field(default_factory=dict)
    profile: str = "default"
    players: dict[str, PlayerRecord] = Field(default_factory=dict)
    positions: dict[str, str | None] = Field(default_factory=dict)
    turns: list[Turn] = Field(default_factory=list)
    outcome: Outcome = Field(default_factory=Outcome)
    rewards: dict[str, float | None] = Field(default_factory=dict)
    reward_status: Literal["final", "pending", "error"] = "final"
    reward_details: dict[str, Any] = Field(default_factory=dict)
    # Filled by ground-truth scorers (experimenter side). Keys: role_values, outcome_value, ...
    ground_truth: dict[str, Any] = Field(default_factory=dict)
    # "error": a ground-truth scorer failed (e.g. a sandbox or network error) - the episode is scored again
    # when its store is resumed, rather than keeping a missing value for good
    gt_status: Literal["known", "pending", "unknown", "unscored", "error"] = "unscored"
    # model usage of the play; in a sampled game tree each (shared) candidate pool is charged to one
    # canonical leaf, so the tree's leaf episodes sum to its total without double counting
    # (see samplers.pools.expand_tree)
    usage: dict[str, Usage] = Field(default_factory=dict)
    # the mechanism's uses of ground truth (audits, simulated probes): channel, role, cost - never findings
    gt_access: list[dict[str, Any]] = Field(default_factory=list)
    tags: dict[str, Any] = Field(default_factory=dict)
    trainable_roles: list[str] = Field(default_factory=list)
    role_kinds: dict[str, str] = Field(default_factory=dict)
    repeat: int = 0
    seed: int = 0
    created_at: str = Field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc).isoformat())
    error: str | None = None
    # stateful tasks: the snapshot the episode started from, the one it ended in, and where they are stored
    initial_state: str | None = None
    final_state: str | None = None
    state_store: str | None = None

    # -------------------------------------------------------------------- convenience accessors
    def reward(self, role: str) -> float | None:
        return self.rewards.get(role)

    def value(self, role: str) -> float | None:
        return (self.ground_truth.get("role_values") or {}).get(role)

    def label(self, role: str) -> str | None:
        p = self.players.get(role)
        return p.label if p else None

    def turns_of(self, role: str) -> list[Turn]:
        return [t for t in self.turns if t.role == role]

    def last_turn(self, role: str, phase: str | None = None) -> Turn | None:
        ts = [t for t in self.turns if t.role == role and (phase is None or t.phase == phase)]
        return ts[-1] if ts else None

    @property
    def total_usage(self) -> Usage:
        total = Usage()
        for u in self.usage.values():
            total = total + u
        return total

    def verifications(self, role: str | None = None) -> list:
        out = []
        for t in self.turns:
            if role is None or t.role == role:
                out += t.verifications
        return out


class Mechanism(abc.ABC):
    """Base class for mechanisms. Subclasses define :meth:`roles` and :meth:`protocol`.

    Constructor keyword arguments are stored as the mechanism's config (logged with every episode
    and used in episode ids, so changing the config never reuses stale results).
    """

    name: ClassVar[str] = "mechanism"
    description: ClassVar[str] = ""

    def __init__(self, *, reward: "RewardRule | None" = None, verification: VerificationPolicy | None = None,
                 name: str | None = None, affordances: dict[str, list[str]] | None = None,
                 tools: dict[str, list[str]] | None = None, sees_reasoning: dict[str, list[str]] | None = None,
                 trainable: dict[str, bool] | None = None, state_access: dict[str, str] | None = None,
                 **config: Any):
        """Args (common to all mechanisms):
            reward: reward rule (defaults to :meth:`default_reward`).
            verification: which claims are verified and how results are shown.
            affordances / tools / sees_reasoning: per-role overrides of the role specs. Keys are role
                names, ``"agents"`` (every agent-kind role) or ``"all"`` (any other key is an error,
                raised when the role specs are first built); values are lists of private item keys /
                tool names / roles whose chain of thought the role sees. These overrides only *add* to
                the role's defaults (set union): removing a default needs a different mechanism
                argument or a subclass.
            trainable: per-role override of whether a role receives reward (e.g. train the judge).
            state_access: per-role override of access to a stateful task's state (``"none"``,
                ``"read"``, ``"write"``), e.g. ``{"reviewer": "read"}`` to let a reviewer run the tests.
        """
        bad = {v for v in (state_access or {}).values() if v not in STATE_ACCESS}
        if bad:
            raise ValueError(f"state_access values must be one of {STATE_ACCESS}, got {sorted(bad)}")
        self.config = config
        self.role_overrides = {
            "affordances": dict(affordances or {}),
            "tools": dict(tools or {}),
            "sees_reasoning_of": dict(sees_reasoning or {}),
            "trainable": dict(trainable or {}),
            "state_access": dict(state_access or {}),
        }
        for k, v in self.role_overrides.items():
            if v:
                self.config[f"role_{k}"] = v
        self.reward_rule = reward if reward is not None else self.default_reward()
        self.verification = verification
        if name:
            self.name = name  # type: ignore[misc]

    def role_specs(self) -> dict[str, RoleSpec]:
        """:meth:`roles` with the per-role overrides applied (this is what the runtime uses).

        Raises ValueError for an override key that is neither a role nor ``"agents"``/``"all"``: a
        mistyped role would otherwise silently leave the experiment unchanged.
        """
        specs = {r: s.model_copy(deep=True) for r, s in self.roles().items()}
        for field, overrides in self.role_overrides.items():
            unknown = sorted(k for k in overrides if k not in specs and k not in ("agents", "all"))
            if unknown:
                arg = "sees_reasoning" if field == "sees_reasoning_of" else field
                raise ValueError(f"{self.name}: {arg} override for unknown role(s) {unknown}; roles are {sorted(specs)} "
                                 "(or use 'agents' / 'all')")
        for field in ("affordances", "tools", "sees_reasoning_of"):
            for key, vals in self.role_overrides[field].items():
                for r, spec in specs.items():
                    if key == r or key == "all" or (key == "agents" and spec.kind == "agent"):
                        setattr(spec, field, sorted(set(getattr(spec, field)) | set(vals)))
        for field, cast in (("trainable", bool), ("state_access", str)):
            for key, val in self.role_overrides[field].items():
                for r, spec in specs.items():
                    if key == r or key == "all" or (key == "agents" and spec.kind == "agent"):
                        setattr(spec, field, cast(val))
        return specs

    # ------------------------------------------------------------------ to implement
    @abc.abstractmethod
    def roles(self) -> dict[str, RoleSpec]: ...

    @abc.abstractmethod
    async def protocol(self, g: Game) -> Outcome: ...

    def default_reward(self) -> "RewardRule":
        from so_arena.core.rewards import NoReward

        return NoReward()

    # ------------------------------------------------------------------ descriptions
    def role_title(self, role: str, g: Game | None = None) -> str:
        spec = self.role_specs().get(role)
        title = (spec.title if spec and spec.title else role.replace("_", " ").title())
        if g is not None and g.positions.get(role) and (spec is None or spec.kind == "agent"):
            title += f" (arguing for {g.positions[role]})"
        return title

    def trainable_roles(self) -> list[str]:
        return [r for r, s in self.role_specs().items() if s.trainable]

    def describe(self, role: str | None = None) -> str:
        """Natural-language description of the rules and rewards (given to prompt optimizers that
        are meant to 'know the mechanism')."""
        lines = [f"Mechanism: {self.name}. {self.description}".strip()]
        for r, s in self.role_specs().items():
            t = "trainable (receives reward)" if s.trainable else "fixed behaviour (no reward)"
            lines.append(f"- role {r} [{s.kind}, {t}]: {s.description}")
        lines.append(f"Rewards: {self.reward_rule.describe()}")
        if self.verification is not None and self.verification.verifiers:
            names = [v if isinstance(v, str) else v.name for v in self.verification.verifiers]
            lines.append(f"Verifiable claim kinds: {', '.join(names)}")
        if self.config:
            lines.append(f"Config: {json.dumps(describe_config(self.config))}")
        if role:
            lines.append(f"You are optimizing the behaviour of role '{role}'.")
        return "\n".join(lines)

    def config_dict(self) -> dict[str, Any]:
        """The full configuration (logged with every episode, hashed into episode ids): constructor
        arguments, the reward rule's structure (every parameter, also those its description omits)
        and the whole verification policy (verifiers, display, budget, roles, ...)."""
        return {
            "name": self.name,
            "class": type(self).__name__,
            "config": describe_config(self.config),
            "reward": self.reward_rule.describe(),
            "verification": (
                [v if isinstance(v, str) else v.name for v in self.verification.verifiers]
                if self.verification else None
            ),
            "reward_rule": describe_config(self.reward_rule),
            "verification_policy": describe_config(self.verification) if self.verification is not None else None,
        }

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.config_dict(), sort_keys=True).encode()).hexdigest()[:10]

    # ------------------------------------------------------------------ running
    async def run(
        self,
        item: TaskItem,
        players: dict[str, Player],
        ctx: RunContext | None = None,
        *,
        episode_id: str | None = None,
        profile: str = "default",
        branch: BranchController | None = None,
        tags: dict[str, Any] | None = None,
        repeat: int = 0,
        seed: int = 0,
    ) -> Episode:
        """Run the protocol on (a censored copy of) ``item`` and compute rewards.

        The item is always censored - also without ground truth (e.g. read back from a release), whose
        experimenter-side ``metadata`` must not reach the policies either.
        """
        ctx = ctx or RunContext()
        censored = item.censored()
        eid = episode_id or f"{item.id}:{self.name}:{profile}:{repeat}"
        base, head = initial_states(item, ctx)
        g = Game(self, censored, players, ctx=ctx, episode_id=eid, branch=branch, seed=seed, repeat=repeat,
                 state=head, base_state=base)
        error = None
        try:
            outcome = await self.protocol(g)
        except Exception:
            outcome = Outcome()
            error = traceback.format_exc()
        ep = Episode(
            id=eid,
            run_id=ctx.run_id,
            item_id=item.id,
            domain=item.domain,
            mechanism=self.name,
            mechanism_config=self.config_dict(),
            profile=profile,
            players={
                r: PlayerRecord(policy_id=p.policy.id, label=p.behaviour, stance=p.stance,
                                model=p.policy.model_name, config=p.policy.describe())
                for r, p in players.items()
            },
            positions=dict(g.positions),
            turns=list(g.turns),
            outcome=outcome,
            usage=g.episode_usage(),  # in branch mode: none (expand_tree charges each shared pool to one leaf)
            gt_access=list(g.gt_access),
            tags=dict(tags or {}),
            trainable_roles=self.trainable_roles(),
            role_kinds={r: spec.kind for r, spec in g.roles.items()},
            repeat=repeat,
            seed=seed,
            error=error,
            initial_state=head,
            final_state=g.state,
            state_store=str(ctx.states.root) if head is not None else None,
        )
        if error is None:
            try:
                rewards = await self.reward_rule.acompute(ep, g)
                ep.rewards = rewards
                ep.reward_status = "pending" if any(v is None for r, v in rewards.items()
                                                    if r in ep.trainable_roles) else "final"
            except Exception:
                ep.error = traceback.format_exc()
                ep.reward_status = "error"
        else:
            ep.reward_status = "error"
        return ep


def initial_states(item: TaskItem, ctx: RunContext) -> tuple[str | None, str | None]:
    """(base, head) snapshots an episode on ``item`` starts from.

    An item may reference existing states (``context["state"] = {"base": ..., "head": ...}``: e.g. work
    under review, shown as changes from ``base``); otherwise the run's environment builds the item's
    starting state from the *uncensored* item (hidden environment state may depend on ground truth).
    """
    ref = item.context.get("state")
    if isinstance(ref, dict) and (ref.get("head") or ref.get("base")):
        head = ref.get("head") or ref.get("base")
        return ref.get("base") or head, head
    if ctx.environment is not None:
        s0 = ctx.environment.initial_state(item, ctx.states)
        return s0, s0
    return None, None
