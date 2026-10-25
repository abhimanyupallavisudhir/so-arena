"""The runtime of one episode: transcript, visibility, affordances, verification and branching.

A mechanism's :meth:`~so_arena.core.mechanism.Mechanism.protocol` is ordinary async Python that
calls :meth:`Game.act` for each decision. The game builds the acting role's view (filtering private
information by affordance and turns by visibility), asks the role's policy for an action, verifies
any claims, and records the turn.

Branching. With a :class:`BranchController`, every decision point becomes a node of a *sampled game
tree*: the controller draws a pool of K candidate actions for the acting role (K set per role),
memoizes them, and follows the candidate chosen by its plan. Re-running the protocol under different
plans replays shared prefixes from the memo, so an imperative protocol can be expanded into a full
tree without being written as one (see :mod:`so_arena.samplers.pools`). Node keys depend only on
the role, phase, a deterministic slot counter and the choices made at earlier decisions, so
simultaneous moves (launched together via :meth:`Game.simultaneous`) do not observe each other.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from so_arena.core.actions import Action, ActionKind, ActionRequest, GameView, TurnView
from so_arena.core.items import TaskItem
from so_arena.core.policy import ActContext, Policy, record_tool_call, recording_tool_calls, stable_hash
from so_arena.core.state import Environment, StateStore, WorkspaceSlot, diff_trees, using_workspace
from so_arena.core.tools import Tool
from so_arena.core.types import Message, Usage
from so_arena.core.verification import (
    JUDGE_VERIFICATION_NOTE,
    Verification,
    VerificationNoiseError,
    Verifier,
    annotate,
    neutralize_markers,
    parse_claims,
)

if TYPE_CHECKING:
    from so_arena.core.mechanism import Mechanism


_ACCESS_ORDER = {"none": 0, "read": 1, "write": 2}


class _RecordingTool(Tool):
    """Delegates to a tool and records each call (the trusted record of what a role did, whatever its policy)
    in the current context's sink (see :func:`~so_arena.core.policy.recording_tool_calls`): one per sampled
    candidate, and one per candidate of a best-of-N policy, so a discarded attempt never enters the record."""

    def __init__(self, inner: Tool):
        self.inner = inner
        self.name, self.description, self.example = inner.name, inner.description, inner.example

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.inner, attr)

    def instructions(self) -> str:
        return self.inner.instructions()

    async def call(self, args, item, game=None):
        res = await self.inner.call(args, item, game)
        record_tool_call({"name": self.inner.name, "args": args, "result": res.output, "error": res.error})
        return res


def _call_keys(calls: list[dict[str, Any]]) -> list[tuple[Any, str, str]]:
    return [(c.get("name"), str(c.get("args")), str(c.get("result"))) for c in calls]


class Player(BaseModel):
    """A policy filling a role, with an optional assigned stance (answer label to argue for)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy: Policy
    stance: str | None = None
    label: str | None = None  # behaviour label for analysis; defaults to the policy's label

    @property
    def behaviour(self) -> str:
        return self.label or self.policy.label or self.policy.id


class Turn(BaseModel):
    index: int
    slot: int
    role: str
    phase: str = ""
    kind: ActionKind = "text"
    text: str = ""  # what the policy produced (public part)
    shown: str = ""  # what other roles see (claims replaced by verification markers, forged markers escaped)
    # VerificationPolicy.show_to: the roles (names or kinds) that see ``shown``; the others see ``unmarked``
    # (the claims as written, escaped, without verdicts). None = everyone sees ``shown``
    verdicts_to: list[str] | None = None
    unmarked: str | None = None
    reasoning: str | None = None
    visible_to: list[str] | None = None  # None = everyone
    choice: str | None = None
    probs: dict[str, float] | None = None
    score: float | None = None
    data: dict[str, Any] | None = None
    verifications: list[Verification] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    parse_ok: bool = True
    node: str | None = None
    candidate: int | None = None
    group: str | None = None
    state: str | None = None  # the snapshot this turn left behind (roles with write access to the state)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunContext:
    """Resources shared by all episodes of a run."""

    def __init__(
        self,
        *,
        run_id: str = "run",
        seed: int = 0,
        verifiers: dict[str, Verifier] | None = None,
        tools: dict[str, Tool] | None = None,
        resources: dict[str, Any] | None = None,
        environment: Environment | None = None,
        states: StateStore | str | None = None,
    ):
        """``environment`` builds each item's starting state and adds its tools; ``states`` is where
        snapshots are kept (a :class:`StateStore` or a directory; default: see :class:`StateStore`)."""
        self.run_id = run_id
        self.seed = seed
        self.verifiers = dict(verifiers or {})
        self.environment = environment
        self.environment_tools = environment.tools() if environment is not None else {}
        self.tools = {**self.environment_tools, **dict(tools or {})}
        self.resources = dict(resources or {})
        self._states = StateStore(states) if isinstance(states, (str, Path)) else states

    @property
    def states(self) -> StateStore:
        """The snapshot store (created on first use)."""
        if self._states is None:
            self._states = StateStore()
        return self._states


class NodeRecord(BaseModel):
    key: str
    role: str
    phase: str
    k: int
    choice: int
    group: str | None = None
    slot: int = 0


class _Produced(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: Action
    verifications: list[Verification]
    shown: str
    usage: Usage
    state: str | None = None  # snapshot left behind by a role with write access
    checked: int = 0  # claims sent to a verifier (charged to the role's verification budget)
    spent: float = 0.0  # their cost (charged to a cost budget)
    unmarked: str | None = None  # the text without verdicts, for roles outside VerificationPolicy.show_to


class _Checked(BaseModel):
    """What checking one action's claims produced (:meth:`Game._verify`)."""

    verifications: list[Verification] = Field(default_factory=list)
    shown: str = ""
    unmarked: str | None = None
    usage: Usage = Field(default_factory=Usage)
    checked: int = 0
    spent: float = 0.0


class BranchController:
    """Pools of candidate actions at decision nodes, and a plan selecting which candidate to follow.

    ``pool_sizes`` maps ``"role"`` or ``"role:phase"`` -> K (the phase-specific key wins; unlisted
    decisions get K=1). Decisions are keyed by information set (:meth:`Game._infoset_key`) and the memo
    is shared across replays, so each information set's pool is sampled exactly once, whichever play
    reaches it first; ``usage`` (shared by all forks) counts each sample once.
    :func:`~so_arena.samplers.pools.expand_tree` charges each pool to one canonical leaf episode, so the
    leaves of a fully expanded tree sum to ``usage``.
    """

    def __init__(self, pool_sizes: dict[str, int] | None = None, plan: dict[str, int] | None = None,
                 memo: dict[str, list[_Produced]] | None = None, locks: dict[str, asyncio.Lock] | None = None,
                 usage: dict[str, Usage] | None = None, roles: dict[str, str] | None = None,
                 origins: dict[str, str | None] | None = None, adapted: dict[str, list[_Produced] | None] | None = None):
        self.pool_sizes = dict(pool_sizes or {})
        self.plan = dict(plan or {})
        self.memo = memo if memo is not None else {}
        self.locks = locks if locks is not None else {}
        self.usage = usage if usage is not None else defaultdict(Usage)
        self.roles = roles if roles is not None else {}  # pool key -> the role it belongs to
        self.origins = origins if origins is not None else {}  # pool key -> the state it was sampled on
        self.adapted = adapted if adapted is not None else {}  # (pool key, state) -> the pool as it plays out there
        self.trace: list[NodeRecord] = []

    def fork(self, plan: dict[str, int]) -> "BranchController":
        return BranchController(self.pool_sizes, plan, self.memo, self.locks, self.usage, self.roles, self.origins,
                                self.adapted)

    def pool_size(self, role: str, phase: str = "") -> int:
        """Pool size for a decision: a ``"role:phase"`` key overrides a ``"role"`` key (default 1)."""
        k = self.pool_sizes.get(f"{role}:{phase}", self.pool_sizes.get(role, 1))
        return max(1, int(k))

    async def _pool(self, key: str, role: str, phase: str, sampler: Callable[[int], Awaitable[_Produced]],
                    origin: str | None) -> list[_Produced]:
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self.memo:
                k = self.pool_size(role, phase)
                pool = list(await asyncio.gather(*[sampler(i) for i in range(k)]))
                for p in pool:
                    self.usage[role] = self.usage[role] + p.usage
                self.memo[key] = pool
                self.roles[key] = role
                self.origins[key] = origin
        return self.memo[key]

    async def decide(self, key: str, role: str, phase: str, group: str | None, slot: int,
                     sampler: Callable[[int], Awaitable[_Produced]], *, origin: str | None = None,
                     adapt: Callable[[_Produced, str | None], Awaitable[_Produced | None]] | None = None
                     ) -> tuple[_Produced, int, str]:
        """The candidate the plan follows at this decision, its index, and the key of its pool.

        An information set's pool is sampled once, on the state ``origin`` of the node that reaches it
        first. Another node of the set may be in a different state - one differing only in what the role
        cannot see (hidden environment state; or any state, for a role without access). There each
        candidate is ``adapt``-ed: replayed on that node's state (:meth:`Game._adapt`). If a candidate would
        observe something different there, the role can tell the nodes apart, and the node gets a pool of
        its own (keyed by the set and its state). Which node samples first can then decide which nodes
        share a pool; with tools that reveal nothing hidden - the usual case - every node shares it.
        """
        pool = await self._pool(key, role, phase, sampler, origin)
        if adapt is not None and self.origins.get(key) != origin:
            ak = f"{key}|{origin}"
            lock = self.locks.setdefault(ak, asyncio.Lock())
            async with lock:
                if ak not in self.adapted:
                    here: list[_Produced] | None = []
                    for p in pool:
                        a = await adapt(p, self.origins.get(key))
                        if a is None:
                            here = None
                            break
                        here.append(a)  # type: ignore[union-attr]
                    self.adapted[ak] = here
            if self.adapted[ak] is None:
                key = hashlib.sha256(ak.encode()).hexdigest()[:20]
                pool = await self._pool(key, role, phase, sampler, origin)
            else:
                pool = self.adapted[ak]  # type: ignore[assignment]
        idx = self.plan.get(key, 0)
        self.trace.append(NodeRecord(key=key, role=role, phase=phase, k=len(pool), choice=idx, group=group, slot=slot))
        return pool[idx], idx, key


class Game:
    """State and helpers for one episode of a mechanism."""

    def __init__(
        self,
        mechanism: "Mechanism",
        item: TaskItem,
        players: dict[str, Player],
        *,
        ctx: RunContext | None = None,
        episode_id: str = "episode",
        branch: BranchController | None = None,
        seed: int = 0,
        repeat: int = 0,
        state: str | None = None,
        base_state: str | None = None,
    ):
        """``state`` is the snapshot the episode starts from (the item's $S_0$, or the work under review);
        ``base_state`` the task's original starting state, against which changes are shown."""
        self.mechanism = mechanism
        self.repeat = repeat
        self.item = item
        self.players = players
        self.ctx = ctx or RunContext()
        self.episode_id = episode_id
        self.branch = branch
        self.seed = seed
        self.roles = mechanism.role_specs()
        missing = [r for r, spec in self.roles.items() if spec.required and r not in players]
        if missing:
            raise ValueError(f"{mechanism.name}: no player for roles {missing}")
        self.turns: list[Turn] = []
        self.positions: dict[str, str | None] = {r: p.stance for r, p in players.items()}
        self._public_positions: set[str] = set()
        self.usage: dict[str, Usage] = defaultdict(Usage)  # outside branch mode (see episode_usage)
        self.data: dict[str, Any] = {}
        self.round = 0
        self._slot = 0
        self._decisions: dict[int, tuple[str, int]] = {}
        self._slot_group: dict[int, str | None] = {}
        self._slot_role: dict[int, str] = {}
        # slot -> (role, claims charged to its budget, their cost)
        self._verif_used: dict[int, tuple[str, int, float]] = {}
        self._group_counter = 0
        self.state = state
        self.base_state = base_state if base_state is not None else state
        self._group_writers: dict[str, str] = {}
        self._group_state: dict[str, str | None] = {}  # simultaneous group -> the state its movers act on
        # reverts of stateful work (state_without): reverted roles ("a,b") -> paths whose changes did not merge
        self.revert_conflicts: dict[str, list[str]] = {}
        # uses of experimenter-side ground truth by the mechanism (audits, simulated probes): see log_gt_access
        self.gt_access: list[dict[str, Any]] = []
        import random

        # per-episode randomness: in a game tree it differs by path, so nature's moves use chance() instead
        self.rng = random.Random(stable_hash(seed, episode_id))
        vp = mechanism.verification
        self.verifiers: dict[str, Verifier] = vp.resolve(self.ctx.verifiers) if vp is not None else {}

    def chance(self, tag: str) -> "random.Random":
        """A random stream for one of nature's moves (an audit, a tie-break) that must not depend on what any
        role did: seeded from the item, the repeat, the run seed and ``tag`` only.

        :attr:`rng` is seeded from the episode id, which in a sampled game tree encodes the path - including
        the sampled actions the move may be about - so best-of-N over a pool would select luck (e.g. the
        unaudited candidates). These draws are shared by every path of a tree and every profile of a run on
        the item (common random numbers: arms are compared on the same audits); vary the seed or the repeat
        to redraw them. Use one tag per move, so draws never shift when another move is skipped.
        """
        import random

        return random.Random(stable_hash("chance", tag, self.item.id, self.repeat, self.seed))

    def log_gt_access(self, channel: str, *, role: str | None = None, cost: float = 1.0) -> None:
        """Record that the mechanism consulted ground truth through ``channel`` (an audit, a simulated probe)
        about ``role``, at ``cost``: recorded as ``Episode.gt_access``, so analyses can count how much ground
        truth a mechanism consumed. What was found is never recorded here (the ledger is published)."""
        self.gt_access.append({"channel": channel, "role": role, "cost": float(cost), "slot": self._slot})

    # ------------------------------------------------------------------------------ views
    def stance(self, role: str) -> str | None:
        return self.positions.get(role)

    def set_position(self, role: str, label: str | None) -> None:
        self.positions[role] = label

    def publish_positions(self, *roles: str) -> None:
        """Show these roles' positions to every role from now on (e.g. when prompts announce them).

        By default another role's position appears in a view only once its holder has a turn the
        viewer can see - the transcript names it then ("arguing for X") - so a role that never
        speaks (the phantom agent of a naive-judge baseline) does not reveal the arm's stance.
        """
        self._public_positions.update(roles)

    def can_see(self, viewer: str, turn: Turn) -> bool:
        return turn.visible_to is None or viewer in turn.visible_to or viewer == turn.role

    def shown_to(self, viewer: str, turn: Turn) -> str:
        """``turn`` as ``viewer`` reads it: its author's own text; else the text with verification markers -
        or, for a role outside the verification policy's ``show_to``, the claims as written, without verdicts."""
        if viewer == turn.role:
            return turn.text
        if turn.verdicts_to is None or turn.unmarked is None:
            return turn.shown
        spec = self.roles.get(viewer)
        return turn.shown if viewer in turn.verdicts_to or (spec is not None and spec.kind in turn.verdicts_to) \
            else turn.unmarked

    def sees_reasoning(self, viewer: str, author: str) -> bool:
        if viewer == author:
            return True
        spec = self.roles.get(viewer)
        return bool(spec and ("*" in spec.sees_reasoning_of or author in spec.sees_reasoning_of))

    def visible_turns(self, viewer: str, *, exclude_group: str | None = None) -> list[Turn]:
        return [
            t for t in sorted(self.turns, key=lambda t: t.slot)
            if self.can_see(viewer, t) and (exclude_group is None or t.group != exclude_group)
        ]

    def item_view(self, role: str) -> TaskItem:
        spec = self.roles.get(role)
        allowed = set(spec.affordances) if spec else set()
        private = {k: v for k, v in self.item.private.items() if k in allowed or "*" in allowed}
        return self.item.model_copy(update={"private": private})

    def view(self, role: str, *, exclude_group: str | None = None) -> GameView:
        turns = []
        visible = self.visible_turns(role, exclude_group=exclude_group)
        for i, t in enumerate(visible):  # numbered among the turns the viewer sees: hidden ones are not counted
            own = t.role == role
            reasoning = t.reasoning if self.sees_reasoning(role, t.role) else None
            turns.append(TurnView(index=i, role=t.role, phase=t.phase, text=self.shown_to(role, t),
                                  reasoning=reasoning if own or reasoning is None else neutralize_markers(reasoning)))
        # public positions: the viewer's own, those of roles it has heard from, and published ones
        speakers = {t.role for t in visible}
        positions = {r: p for r, p in self.positions.items()
                     if r == role or r in speakers or r in self._public_positions}
        return GameView(role=role, item=self.item_view(role), transcript=turns, stance=self.stance(role),
                        positions=positions, round=self.round)

    def role_title(self, role: str) -> str:
        return self.mechanism.role_title(role, self)

    def transcript_text(
        self,
        viewer: str,
        *,
        phases: Sequence[str] | None = None,
        roles: Sequence[str] | None = None,
        include_reasoning: bool = True,
        empty: str = "(no messages yet)",
    ) -> str:
        """Render the transcript as ``viewer`` sees it."""
        blocks = []
        for t in self.visible_turns(viewer):
            if phases is not None and t.phase not in phases:
                continue
            if roles is not None and t.role not in roles:
                continue
            if t.kind != "text" and not t.text:
                continue
            body = self.shown_to(viewer, t)
            head = f"### {self.role_title(t.role)}" + (f" ({t.phase})" if t.phase else "")
            block = f"{head}\n{body.strip()}"
            if include_reasoning and t.reasoning and self.sees_reasoning(viewer, t.role) and t.role != viewer:
                block += f"\n<private_reasoning>\n{neutralize_markers(t.reasoning.strip())}\n</private_reasoning>"
            blocks.append(block)
        return "\n\n".join(blocks) if blocks else empty

    def private_context(self, role: str) -> str:
        """Render the private information this role can access."""
        view = self.item_view(role)
        if not view.private:
            return ""
        parts = []
        for k, v in view.private.items():
            if isinstance(v, (dict, list)):
                v = json.dumps(v, indent=1, default=str)[:20000]
            parts.append(f"<{k}>\n{v}\n</{k}>")
        return "Private information available to you:\n" + "\n".join(parts)

    def tools_for(self, role: str, access: str | None = None) -> dict[str, Tool]:
        """The role's granted tools, plus - on stateful tasks - the environment's tools its state access
        allows: all of them with write access, the read-only ones (``readonly = True``) with read access."""
        spec = self.roles.get(role)
        names = list(spec.tools if spec else [])
        access = self.state_access(role) if access is None else access
        if access != "none":
            names += [n for n, t in self.ctx.environment_tools.items()
                      if access == "write" or getattr(t, "readonly", False)]
        return {n: self.ctx.tools[n] for n in dict.fromkeys(names) if n in self.ctx.tools}

    def verifiers_for(self, role: str) -> dict[str, Verifier]:
        vp = self.mechanism.verification
        if vp is None or not self.verifiers:
            return {}
        spec = self.roles.get(role)
        if vp.roles is not None:
            return self.verifiers if role in vp.roles else {}
        return self.verifiers if (spec is None or spec.kind == "agent") else {}

    def claim_instructions(self, role: str) -> str:
        vp = self.mechanism.verification
        vs = self.verifiers_for(role)
        if vp is None or not vs or not vp.announce:
            return ""
        return vp.agent_instructions(vs)

    def judge_note(self) -> str:
        if not self.verifiers:
            return ""
        vp = self.mechanism.verification
        noise = vp.max_noise() if vp is not None else 0.0
        return JUDGE_VERIFICATION_NOTE + (
            f" The verification tool is imperfect: each verdict or output it shows is wrong with probability up "
            f"to {noise:g}, and a wrong one looks exactly like a correct one." if noise > 0 else "")

    # ------------------------------------------------------------------------------ state
    @property
    def states(self) -> StateStore:
        return self.ctx.states

    def state_access(self, role: str, limit: str | None = None) -> str:
        """``"none"``, ``"read"`` or ``"write"``: how ``role`` may act on the episode's state (none without
        state), at most ``limit`` (a mechanism may restrict a single decision, e.g. read-only reporting)."""
        if self.state is None:
            return "none"
        spec = self.roles.get(role)
        access = spec.state_access if spec is not None else "none"
        if limit is not None:
            if limit not in _ACCESS_ORDER:
                raise ValueError(f"access must be one of {tuple(_ACCESS_ORDER)}, got {limit!r}")
            access = min(access, limit, key=_ACCESS_ORDER.__getitem__)
        return access

    def state_without(self, roles: Sequence[str]) -> str | None:
        """The state as if ``roles`` had never changed it: the task's starting state plus every other
        role's state changes, merged in order (reverting a contribution; see
        :meth:`~so_arena.core.state.StateStore.replay_with_conflicts`).

        Where a kept change overlaps a reverted one, the entry keeps neither and the revert is not clean:
        its paths are recorded in :attr:`revert_conflicts` (under the reverted roles, e.g. ``"worker_1"``)
        for mechanisms and audits to report."""
        if self.state is None or self.base_state is None:
            return self.state
        excluded, transitions, current = set(roles), [], self.base_state
        for t in sorted(self.turns, key=lambda t: t.slot):
            if t.state is not None:
                if t.role not in excluded:
                    transitions.append((current, t.state))
                current = t.state
        sid, conflicts = self.states.replay_with_conflicts(self.base_state, transitions)
        self.revert_conflicts[",".join(sorted(excluded))] = conflicts
        return sid

    def state_diff(self, since: str | None = None, **kw: Any) -> str:
        """Reviewer-readable changes from ``since`` (default: the task's starting state) to the current state.

        File contents are the agents' work, so marker tags in them are escaped like any role's text.
        """
        base = since or self.base_state
        if self.state is None or base is None:
            return ""
        return neutralize_markers(diff_trees(self.states.files_dir(base), self.states.files_dir(self.state), **kw))

    # ------------------------------------------------------------------------------ acting
    def _node_key(self, role: str, phase: str, slot: int, group: str | None) -> str:
        """A plain run's decision key (it seeds the policy's randomness): the decision's position and every
        earlier decision. Decisions of the same simultaneous group are excluded."""
        path = [
            self._decisions[s] for s in sorted(self._decisions)
            if s < slot and (group is None or self._slot_group.get(s) != group)
        ]
        raw = json.dumps([role, phase, slot, path])
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    def _infoset_key(self, role: str, request: ActionRequest, *, slot: int, access: str, state: str | None,
                     used: Any) -> str:
        """A game-tree decision's key: its *information set* - everything the role can condition on here.

        That is the request it is sent (prompt and view: every turn it may see, as shown to it, numbered
        among those it sees), its own earlier decisions and the candidates it took (perfect recall), the
        *visible* files of the state it acts on if it can see them (or if its claims are checked against
        it) - never hidden environment state, which no role can read (``state`` is a
        :meth:`~so_arena.core.state.StateStore.visible_id`) - and the verifications it has used. Moves it
        cannot see - another role's private turn, a simultaneous partner's current move, a dealer's hidden
        card, a write to hidden records - do not enter the key, so the nodes that differ only in them share
        one pool of candidates (adapted to each node's state, :meth:`BranchController.decide`), and selection
        (:func:`~so_arena.analysis.optimization.evaluate_tree`) picks one distribution for all of them. Keying
        by the whole path instead would let best-of-N choose separately behind every hidden move, i.e. act on
        information the role never had.

        Tools are assumed to reveal only what the item and the visible state determine (as the built-in ones
        do): a tool reading hidden, path-dependent game data would make a role's observations differ within
        what its key treats as one information set.
        """
        own = [self._decisions[s] for s in sorted(self._decisions) if s < slot and self._slot_role.get(s) == role]
        sees_state = access != "none" or any(getattr(v, "uses_state", False) for v in self.verifiers_for(role).values())
        raw = json.dumps([role, request.phase, access, state if sees_state else None, used, own,
                          request.model_dump(mode="json")], sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    async def _produce(self, role: str, request: ActionRequest, sample_index: int, key: str = "",
                       budget_used: tuple[int, float] = (0, 0.0), access: str | None = None,
                       parent: str | None = None) -> _Produced:
        """Sample one candidate action on the state ``parent`` (see :meth:`act`); ``budget_used`` is how
        many verifications ``role`` has already used on this play, and their cost (each candidate is charged
        separately: siblings never share a budget)."""
        player = self.players[role]
        # Seed policy randomness from the decision key (+ the episode id outside branch mode, so that
        # repeats differ). In branch mode the episode id depends on the path, so it is excluded:
        # a node's pool must be the same whichever path first reaches it.
        eid = "" if self.branch is not None else self.episode_id
        # A role with state access acts in its own working copy of the current state (one per sampled
        # candidate); with write access the frozen copy becomes the state the rest of the play sees.
        access = self.state_access(role, access)
        slot = WorkspaceSlot(self.states, parent, access) if access != "none" and parent else None
        # repeats draw fresh samples (distinct cache keys) rather than replaying cached completions
        calls: list[dict[str, Any]] = []  # this candidate's trusted record of tool calls
        tools = {n: _RecordingTool(t) for n, t in self.tools_for(role, access).items()}
        actx = ActContext(role=role, sample_index=self.repeat * 10_000 + sample_index, game=self,
                          tools=tools, seed=stable_hash(self.seed, self.item.id, key, eid), workspace=slot,
                          draw=f"{role}|{self.item.id}|{key}")  # separate samples per role, item and decision
        new_state: str | None = None
        try:
            with using_workspace(slot), recording_tool_calls(calls):
                action = await player.policy.act(request, actx)
        except BaseException:
            if slot is not None:
                slot.close(keep=False)
            raise
        # The turn's tool calls are what ran through the game's tools, whatever the policy says it did: a
        # policy's own account (an agent may misreport) is kept apart when it differs. Tools that run outside
        # the game are only in the trusted record if they are given to the game as tools.
        record = [{k: v for k, v in c.items() if k != "error"} for c in calls]
        if action.tool_calls and _call_keys(action.tool_calls) != _call_keys(record):
            action.metadata["reported_tool_calls"] = action.tool_calls
        action.tool_calls = record
        self._mirror(slot, access, role, request.phase, calls)
        if slot is not None:
            kept = slot.close(keep=access == "write")
            new_state = kept if kept != parent else None
        usage = action.usage + actx.usage if action.usage.calls or action.usage.effort_seconds else actx.usage
        action.usage = usage
        v = await self._verify(role, action, budget_used, new_state or parent)
        return _Produced(action=action, verifications=v.verifications, shown=v.shown, usage=usage + v.usage,
                         state=new_state, checked=v.checked, spent=v.spent, unmarked=v.unmarked)

    def _mirror(self, slot: WorkspaceSlot | None, access: str, role: str, phase: str,
                calls: Sequence[dict[str, Any]]) -> None:
        """An agent-visible, agent-writable mirror of the trusted record: what an overseer reading logs sees."""
        log_path = getattr(self.ctx.environment, "action_log", None)
        if log_path and calls and slot is not None and access == "write":
            ws = slot.get()
            for c in calls:
                ws.append_jsonl(log_path, {"role": role, "phase": phase, "tool": c["name"],
                                           "args": str(c["args"])[:2000], "result": str(c["result"])[:2000]})

    async def _verify(self, role: str, action: Action, budget_used: tuple[int, float], target: str | None
                      ) -> _Checked:
        """Check the claims in ``action``: the verifications, the text others see (claims replaced by
        markers), the verifiers' usage and how many claims - at what cost - were charged to the role's budgets
        (``budget_used``: its claims and cost so far on this play). A claim about the state is checked on a
        scratch copy of ``target``, the state the claimant left.

        With verification noise, whether the role's n-th checked claim on this play errs is a chance move
        (tag ``verification_noise:<role>:<n>``), as is the forged result (``verification_forgery:...``): the
        same for every candidate of the decision, so best-of-N cannot select the candidates whose checks
        erred in their favour."""
        out = _Checked(shown=neutralize_markers(action.text))  # marker tags the role wrote itself are escaped
        vs = self.verifiers_for(role)
        if not (vs and action.text):
            return out
        vp = self.mechanism.verification
        assert vp is not None
        n_used, cost_used = budget_used
        for claim in parse_claims(action.text, role):
            v = vs.get(claim.kind)
            if v is None:
                out.verifications.append(Verification(claim=claim, status="unknown_kind"))
                continue
            cost = vp.cost_of(v)
            if ((vp.budget_per_role is not None and n_used + out.checked >= vp.budget_per_role)
                    or (vp.budget is not None and cost_used + out.spent + cost > vp.budget + 1e-9)):
                out.verifications.append(Verification(claim=claim, status="over_budget"))
                continue
            index = n_used + out.checked  # the role's n-th checked claim on this play
            out.checked += 1
            out.spent += cost
            # a fresh scratch copy per claim: an earlier claim's command must not rig the state a later
            # claim is checked on
            scratch = self.states.fork(target, access="read") if getattr(v, "uses_state", False) and target else None
            try:
                with using_workspace(scratch):
                    res = await v.verify(claim, self.item, self)
            except Exception as e:  # verifier failures are logged, not fatal
                res = Verification(claim=claim, status="error", detail=repr(e))
            finally:
                if scratch is not None:
                    self.states.discard(scratch)
            rate = vp.noise_for(v)
            if rate > 0 and self.chance(f"verification_noise:{role}:{index}").random() < rate:
                res = self._err(v, res, self.chance(f"verification_forgery:{role}:{index}"))
            out.usage = out.usage + res.usage
            out.verifications.append(res)
        out.shown = annotate(action.text, out.verifications, display=vp.display, show_output=vp.show_output)
        if vp.show_to is not None:
            out.unmarked = neutralize_markers(action.text)
        return out

    @staticmethod
    def _err(v: Verifier, res: Verification, rng: random.Random) -> Verification:
        """``res`` as the erring verifier shows it, with the correct result kept as its truth."""
        forged = v.forge(res, rng)
        if forged is None:
            raise VerificationNoiseError(
                f"verifier {v.name!r} cannot err realistically on a {res.status!r} result (its forge() returned "
                "None): implement forge() or set its noise to 0 (VerificationPolicy(noise={name: p}))")
        if (forged.status, forged.output) == (res.status, res.output):
            return res
        return forged.model_copy(update={"claim": res.claim, "true_status": res.status, "true_output": res.output})

    async def _adapt(self, p: _Produced, role: str, request: ActionRequest, access: str | None,
                     parent: str | None, origin: str | None, budget_used: tuple[int, float]) -> _Produced | None:
        """Candidate ``p`` - sampled at another node of this information set, in state ``origin`` - as it plays
        out at this node, in state ``parent``, which differs from ``origin`` only in what ``role`` cannot see.

        A policy's action is a function of what it observed. The candidate's recorded tool calls are
        replayed on this node's state: if every call returns what it returned there, the role would have
        acted the same, and the replay gives the action's effect here (hidden records included) and the
        verdicts on its claims here. If a call returns something else, the role could tell the nodes apart:
        None, and the node gets a pool of its own - as for a writer whose edits did not all go through
        tools (a script editing its workspace directly), whose visible result the replay must reproduce.
        """
        access = self.state_access(role, access)
        slot = WorkspaceSlot(self.states, parent, access) if access != "none" and parent else None
        tools = self.tools_for(role, access)
        calls: list[dict[str, Any]] = []
        same = True
        try:
            with using_workspace(slot), recording_tool_calls(calls):
                for c in p.action.tool_calls:
                    name, args = c.get("name"), c.get("args")
                    tool = tools.get(name) if isinstance(name, str) else None
                    if tool is None:
                        out: Any = f"error: unknown tool {name!r}; available: {sorted(tools)}"
                        record_tool_call({"name": name, "args": args, "result": out, "error": True})
                    else:
                        out = (await _RecordingTool(tool).call(args, self.item, self)).output
                    if str(out) != str(c.get("result")):
                        same = False
                        break
        except BaseException:
            if slot is not None:
                slot.close(keep=False)
            raise
        if not same:
            if slot is not None:
                slot.close(keep=False)
            return None
        self._mirror(slot, access, role, request.phase, calls)
        new_state = None
        if slot is not None:
            kept = slot.close(keep=access == "write")
            new_state = kept if kept != parent else None
        if access == "write":
            here, there = new_state or parent, p.state or origin
            if (here is None) != (there is None) or (here and there and self.states.visible_id(here) != self.states.visible_id(there)):
                return None
        v = await self._verify(role, p.action, budget_used, new_state or parent)
        return p.model_copy(deep=True, update={"state": new_state, "verifications": v.verifications, "shown": v.shown,
                                               "checked": v.checked, "spent": v.spent, "unmarked": v.unmarked})

    async def act(
        self,
        role: str,
        request: ActionRequest | None = None,
        *,
        kind: ActionKind = "text",
        prompt: Sequence[Message] | str | None = None,
        phase: str = "",
        visible_to: Sequence[str] | None = None,
        record: bool = True,
        group: str | None = None,
        access: str | None = None,
        **fields: Any,
    ) -> Action:
        """Ask ``role`` for an action and record the turn.

        Either pass a full ``request`` or its fields (``kind``, ``prompt``, ``options``, ...). A string
        prompt becomes a single user message. ``visible_to`` restricts who sees the turn (the author
        always does); None means everyone. ``access`` caps the role's state access for this decision
        (e.g. ``"read"`` for a worker's private report, so simultaneous reporters never write).
        """
        if role not in self.players:
            raise KeyError(f"no player for role {role!r}")
        if request is None:
            if isinstance(prompt, str):
                prompt = [Message.user(prompt)]
            request = ActionRequest(kind=kind, prompt=list(prompt or []), phase=phase, **fields)
        # everything up to the first await runs synchronously: slot, key and view are deterministic
        if group is not None and self.state_access(role, access) == "write":
            other = self._group_writers.setdefault(group, role)
            if other != role:
                raise ValueError(f"{self.mechanism.name}: roles {other!r} and {role!r} move simultaneously and both write "
                                 "the shared state; let them act in turn")
        slot = self._slot
        self._slot += 1
        self._slot_group[slot] = group
        self._slot_role[slot] = role
        request = request.model_copy(update={"view": self.view(role, exclude_group=group), "phase": request.phase or phase})
        # verifications the role used on this play so far, and their cost (earlier decisions only, like the node key)
        mine = [(n, c) for s, (r, n, c) in self._verif_used.items() if r == role and s < slot]
        used = (sum(n for n, _ in mine), sum(c for _, c in mine))
        # the state the decision acts on; simultaneous movers all act on the state the stage began with, as
        # their views exclude each other's moves (a partner may finish first, in plain runs and in replays)
        parent = self._group_state.setdefault(group, self.state) if group is not None else self.state
        if self.branch is not None:  # game trees: one pool per information set, keyed on what the role can see
            visible = self.states.visible_id(parent) if parent else None
            vp = self.mechanism.verification
            # what the role's budgets have left is part of what it acts on (its cost only under a cost budget)
            spent = used if vp is not None and vp.budget is not None else used[0]
            key = self._infoset_key(role, request, slot=slot, access=self.state_access(role, access), state=visible,
                                    used=spent)
        else:
            key = self._node_key(role, request.phase, slot, group)

        async def sample(i: int) -> _Produced:
            return await self._produce(role, request, i, key, used, access, parent)

        if self.branch is not None:
            async def adapt(p: _Produced, origin: str | None) -> _Produced | None:
                return await self._adapt(p, role, request, access, parent, origin, used)

            produced, idx, key = await self.branch.decide(key, role, request.phase, group, slot, sample, origin=parent,
                                                          adapt=adapt if parent is not None else None)
            produced = produced.model_copy(deep=True)
            node, cand = key, idx
        else:
            produced = await sample(0)
            node, cand = None, None
            self.usage[role] = self.usage[role] + produced.usage
        self._decisions[slot] = (key, cand or 0)
        self._verif_used[slot] = (role, produced.checked, produced.spent)
        if produced.state is not None:
            self.state = produced.state

        a = produced.action
        if record:
            self.turns.append(
                Turn(
                    index=len(self.turns), slot=slot, role=role, phase=request.phase, kind=request.kind,
                    text=a.text, shown=produced.shown, reasoning=a.reasoning,
                    verdicts_to=(list(self.mechanism.verification.show_to)
                                 if produced.unmarked is not None and self.mechanism.verification is not None else None),
                    unmarked=produced.unmarked,
                    visible_to=list(visible_to) if visible_to is not None else None,
                    choice=a.choice, probs=a.probs, score=a.score, data=a.data,
                    verifications=produced.verifications, tool_calls=a.tool_calls,
                    usage=produced.usage if self.branch is None else Usage(), parse_ok=a.parse_ok,
                    node=node, candidate=cand, group=group, state=produced.state, metadata=a.metadata,
                )
            )
            self.turns.sort(key=lambda t: t.slot)
            for i, t in enumerate(self.turns):
                t.index = i
        return a

    async def simultaneous(self, calls: Sequence[tuple[str, dict[str, Any]]]) -> list[Action]:
        """Run several ``act`` calls simultaneously: no participant sees the others' current actions.

        On stateful tasks every mover acts on the state as it was when the stage began (a reader never
        sees the writer's simultaneous move); the result of the (at most one) writer is the state after it.
        """
        self._group_counter += 1
        gid = f"g{self._group_counter}"
        self._group_state[gid] = self.state
        return list(await asyncio.gather(*[self.act(role, group=gid, **kw) for role, kw in calls]))

    # ------------------------------------------------------------------------------ export
    def episode_usage(self) -> dict[str, Usage]:
        """Model usage to record with this play's episode.

        Outside branch mode: every call the play made. In branch mode a pool (all K candidates of an
        information set) is shared by every play through the information set - including plays that
        differ only in moves the role cannot see - so no single play can tell whether it should pay for
        it: the play records nothing here, and :func:`~so_arena.samplers.pools.expand_tree` charges each
        pool to one canonical leaf once the tree is known.
        """
        if self.branch is None:
            return dict(self.usage)
        return {}

    def total_usage(self) -> Usage:
        total = Usage()
        for u in self.episode_usage().values():
            total = total + u
        return total
