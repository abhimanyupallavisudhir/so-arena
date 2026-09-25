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
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from so_arena.core.actions import Action, ActionKind, ActionRequest, GameView, TurnView
from so_arena.core.items import TaskItem
from so_arena.core.policy import ActContext, Policy, stable_hash
from so_arena.core.state import Environment, StateStore, Workspace, WorkspaceSlot, diff_trees, using_workspace
from so_arena.core.tools import Tool
from so_arena.core.types import Message, Usage
from so_arena.core.verification import (
    JUDGE_VERIFICATION_NOTE,
    Verification,
    Verifier,
    annotate,
    neutralize_markers,
    parse_claims,
)

if TYPE_CHECKING:
    from so_arena.core.mechanism import Mechanism


_ACCESS_ORDER = {"none": 0, "read": 1, "write": 2}


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


class BranchController:
    """Pools of candidate actions at decision nodes, and a plan selecting which candidate to follow.

    ``pool_sizes`` maps ``"role"`` or ``"role:phase"`` -> K (the phase-specific key wins; unlisted
    decisions get K=1). The memo is shared across replays, so each node's pool is sampled exactly
    once; ``usage`` (shared by all forks) counts each sample once. Episodes of the replays record
    the pools they are the canonical play of (see :meth:`Game.episode_usage`), so the distinct plays
    of a fully expanded tree sum to ``usage``.
    """

    def __init__(self, pool_sizes: dict[str, int] | None = None, plan: dict[str, int] | None = None,
                 memo: dict[str, list[_Produced]] | None = None, locks: dict[str, asyncio.Lock] | None = None,
                 usage: dict[str, Usage] | None = None):
        self.pool_sizes = dict(pool_sizes or {})
        self.plan = dict(plan or {})
        self.memo = memo if memo is not None else {}
        self.locks = locks if locks is not None else {}
        self.usage = usage if usage is not None else defaultdict(Usage)
        self.trace: list[NodeRecord] = []

    def fork(self, plan: dict[str, int]) -> "BranchController":
        return BranchController(self.pool_sizes, plan, self.memo, self.locks, self.usage)

    def pool_size(self, role: str, phase: str = "") -> int:
        """Pool size for a decision: a ``"role:phase"`` key overrides a ``"role"`` key (default 1)."""
        k = self.pool_sizes.get(f"{role}:{phase}", self.pool_sizes.get(role, 1))
        return max(1, int(k))

    async def decide(self, key: str, role: str, phase: str, group: str | None, slot: int,
                     sampler: Callable[[int], Awaitable[_Produced]]) -> tuple[_Produced, int]:
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key not in self.memo:
                k = self.pool_size(role, phase)
                pool = list(await asyncio.gather(*[sampler(i) for i in range(k)]))
                for p in pool:
                    self.usage[role] = self.usage[role] + p.usage
                self.memo[key] = pool
        pool = self.memo[key]
        idx = self.plan.get(key, 0)
        self.trace.append(NodeRecord(key=key, role=role, phase=phase, k=len(pool), choice=idx, group=group, slot=slot))
        return pool[idx], idx


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
        self._verif_used: dict[int, tuple[str, int]] = {}  # slot -> (role, claims charged to its budget)
        self._group_counter = 0
        self.state = state
        self.base_state = base_state if base_state is not None else state
        self._group_writers: dict[str, str] = {}
        import random

        self.rng = random.Random(stable_hash(seed, episode_id))
        vp = mechanism.verification
        self.verifiers: dict[str, Verifier] = vp.resolve(self.ctx.verifiers) if vp is not None else {}

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
        for t in visible:
            own = t.role == role
            reasoning = t.reasoning if self.sees_reasoning(role, t.role) else None
            turns.append(TurnView(index=t.index, role=t.role, phase=t.phase, text=t.text if own else t.shown,
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
            body = t.text if t.role == viewer else t.shown
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
        return JUDGE_VERIFICATION_NOTE if self.verifiers else ""

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
        role's state changes, replayed file by file in order (reverting a contribution)."""
        if self.state is None or self.base_state is None:
            return self.state
        excluded, transitions, current = set(roles), [], self.base_state
        for t in sorted(self.turns, key=lambda t: t.slot):
            if t.state is not None:
                if t.role not in excluded:
                    transitions.append((current, t.state))
                current = t.state
        return self.states.replay(self.base_state, transitions)

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
        # Decisions of the same simultaneous group are excluded: a simultaneous mover's pool must not
        # depend on its partners' current choices (in replays they may complete synchronously first).
        path = [
            self._decisions[s] for s in sorted(self._decisions)
            if s < slot and (group is None or self._slot_group.get(s) != group)
        ]
        raw = json.dumps([role, phase, slot, path])
        return hashlib.sha256(raw.encode()).hexdigest()[:20]

    async def _produce(self, role: str, request: ActionRequest, sample_index: int, key: str = "",
                       budget_used: int = 0, access: str | None = None) -> _Produced:
        """Sample one candidate action; ``budget_used`` is how many verifications ``role`` has already
        used on this play (each candidate is charged separately: siblings never share a budget)."""
        player = self.players[role]
        # Seed policy randomness from the decision key (+ the episode id outside branch mode, so that
        # repeats differ). In branch mode the episode id depends on the path, so it is excluded:
        # a node's pool must be the same whichever path first reaches it.
        eid = "" if self.branch is not None else self.episode_id
        # A role with state access acts in its own working copy of the current state (one per sampled
        # candidate); with write access the frozen copy becomes the state the rest of the play sees.
        access = self.state_access(role, access)
        parent = self.state
        slot = WorkspaceSlot(self.states, parent, access) if access != "none" and parent else None
        # repeats draw fresh samples (distinct cache keys) rather than replaying cached completions
        actx = ActContext(role=role, sample_index=self.repeat * 10_000 + sample_index, game=self,
                          tools=self.tools_for(role, access), seed=stable_hash(self.seed, self.item.id, key, eid),
                          workspace=slot)
        new_state: str | None = None
        try:
            with using_workspace(slot):
                action = await player.policy.act(request, actx)
        except BaseException:
            if slot is not None:
                slot.close(keep=False)
            raise
        if slot is not None:
            kept = slot.close(keep=access == "write")
            new_state = kept if kept != parent else None
        usage = action.usage + actx.usage if action.usage.calls or action.usage.effort_seconds else actx.usage
        action.usage = usage
        verifs: list[Verification] = []
        checked = 0
        # what others see: marker tags the role wrote itself are escaped, so only verifiers make markers
        shown = neutralize_markers(action.text)
        vs = self.verifiers_for(role)
        if vs and action.text:
            vp = self.mechanism.verification
            assert vp is not None
            scratch: Workspace | None = None
            try:
                for claim in parse_claims(action.text, role):
                    v = vs.get(claim.kind)
                    if v is None:
                        verifs.append(Verification(claim=claim, status="unknown_kind"))
                        continue
                    if vp.budget_per_role is not None and budget_used + checked >= vp.budget_per_role:
                        verifs.append(Verification(claim=claim, status="over_budget"))
                        continue
                    checked += 1
                    # claims about the state are checked on a scratch copy of the state the claimant left
                    target = new_state or parent
                    if getattr(v, "uses_state", False) and scratch is None and target:
                        scratch = self.states.fork(target, access="read")
                    try:
                        with using_workspace(scratch if getattr(v, "uses_state", False) else None):
                            res = await v.verify(claim, self.item, self)
                    except Exception as e:  # verifier failures are logged, not fatal
                        res = Verification(claim=claim, status="error", detail=repr(e))
                    usage = usage + res.usage
                    verifs.append(res)
            finally:
                if scratch is not None:
                    self.states.discard(scratch)
            shown = annotate(action.text, verifs, display=vp.display, show_output=vp.show_output)
        return _Produced(action=action, verifications=verifs, shown=shown, usage=usage, state=new_state, checked=checked)

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
        key = self._node_key(role, request.phase, slot, group)
        # verifications the role used on this play so far (earlier decisions only, like the node key)
        used = sum(n for s, (r, n) in self._verif_used.items() if r == role and s < slot)

        async def sample(i: int) -> _Produced:
            return await self._produce(role, request, i, key, used, access)

        if self.branch is not None:
            produced, idx = await self.branch.decide(key, role, request.phase, group, slot, sample)
            produced = produced.model_copy(deep=True)
            node, cand = key, idx
        else:
            produced = await sample(0)
            node, cand = None, None
            self.usage[role] = self.usage[role] + produced.usage
        self._decisions[slot] = (key, cand or 0)
        self._verif_used[slot] = (role, produced.checked)
        if produced.state is not None:
            self.state = produced.state

        a = produced.action
        if record:
            self.turns.append(
                Turn(
                    index=len(self.turns), slot=slot, role=role, phase=request.phase, kind=request.kind,
                    text=a.text, shown=produced.shown, reasoning=a.reasoning,
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
        """Run several ``act`` calls simultaneously: no participant sees the others' current actions."""
        self._group_counter += 1
        gid = f"g{self._group_counter}"
        return list(await asyncio.gather(*[self.act(role, group=gid, **kw) for role, kw in calls]))

    # ------------------------------------------------------------------------------ export
    def episode_usage(self) -> dict[str, Usage]:
        """Model usage to record with this play's episode.

        Outside branch mode: every call the play made. In branch mode a node's pool (all K candidates)
        is shared by every play through the node, so it is charged to one *canonical* play: the one
        taking the first candidate there and at every decision the node's key does not depend on
        (later decisions and simultaneous partners). Each pool thus counts exactly once among the
        distinct plays of a tree - its leaf episodes sum to the controller's ``usage`` when the tree
        is fully expanded - and the attribution does not depend on which replay sampled a pool first.
        """
        if self.branch is None:
            return dict(self.usage)
        out: dict[str, Usage] = {}
        slots = sorted(self._decisions)
        for s in slots:
            key, _ = self._decisions[s]
            group = self._slot_group.get(s)
            free = [t for t in slots if t >= s or (group is not None and self._slot_group.get(t) == group)]
            if any(self._decisions[t][1] != 0 for t in free):
                continue
            role = self._slot_role[s]
            for p in self.branch.memo.get(key, []):
                out[role] = out.get(role, Usage()) + p.usage
        return out

    def total_usage(self) -> Usage:
        total = Usage()
        for u in self.episode_usage().values():
            total = total + u
        return total
