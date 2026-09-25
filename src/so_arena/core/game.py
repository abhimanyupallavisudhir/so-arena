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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from so_arena.core.actions import Action, ActionKind, ActionRequest, GameView, TurnView
from so_arena.core.items import TaskItem
from so_arena.core.policy import ActContext, Policy, stable_hash
from so_arena.core.tools import Tool
from so_arena.core.types import Message, Usage
from so_arena.core.verification import (
    JUDGE_VERIFICATION_NOTE,
    Verification,
    Verifier,
    annotate,
    parse_claims,
)

if TYPE_CHECKING:
    from so_arena.core.mechanism import Mechanism


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
    shown: str = ""  # what other roles see (after verification markers)
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
    ):
        self.run_id = run_id
        self.seed = seed
        self.verifiers = dict(verifiers or {})
        self.tools = dict(tools or {})
        self.resources = dict(resources or {})


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


class BranchController:
    """Pools of candidate actions at decision nodes, and a plan selecting which candidate to follow.

    ``pool_sizes`` maps ``"role"`` or ``"role:phase"`` -> K (the phase-specific key wins; unlisted
    decisions get K=1). The memo is shared across replays, so each node's pool is sampled exactly
    once; ``usage`` counts each sample once.
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
    ):
        self.mechanism = mechanism
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
        self.usage: dict[str, Usage] = defaultdict(Usage)
        self.data: dict[str, Any] = {}
        self.round = 0
        self._slot = 0
        self._decisions: dict[int, tuple[str, int]] = {}
        self._slot_group: dict[int, str | None] = {}
        self._verif_counts: dict[str, int] = defaultdict(int)
        self._group_counter = 0
        import random

        self.rng = random.Random(stable_hash(seed, episode_id))
        vp = mechanism.verification
        self.verifiers: dict[str, Verifier] = vp.resolve(self.ctx.verifiers) if vp is not None else {}

    # ------------------------------------------------------------------------------ views
    def stance(self, role: str) -> str | None:
        return self.positions.get(role)

    def set_position(self, role: str, label: str | None) -> None:
        self.positions[role] = label

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
        for t in self.visible_turns(role, exclude_group=exclude_group):
            text = t.text if t.role == role else t.shown
            turns.append(TurnView(index=t.index, role=t.role, phase=t.phase, text=text,
                                  reasoning=t.reasoning if self.sees_reasoning(role, t.role) else None))
        return GameView(role=role, item=self.item_view(role), transcript=turns, stance=self.stance(role),
                        positions=dict(self.positions), round=self.round)

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
                block += f"\n<private_reasoning>\n{t.reasoning.strip()}\n</private_reasoning>"
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

    def tools_for(self, role: str) -> dict[str, Tool]:
        spec = self.roles.get(role)
        names = spec.tools if spec else []
        return {n: self.ctx.tools[n] for n in names if n in self.ctx.tools}

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

    async def _produce(self, role: str, request: ActionRequest, sample_index: int, key: str = "") -> _Produced:
        player = self.players[role]
        # Seed policy randomness from the decision key (+ the episode id outside branch mode, so that
        # repeats differ). In branch mode the episode id depends on the path, so it is excluded:
        # a node's pool must be the same whichever path first reaches it.
        eid = "" if self.branch is not None else self.episode_id
        actx = ActContext(role=role, sample_index=sample_index, game=self, tools=self.tools_for(role),
                          seed=stable_hash(self.seed, self.item.id, key, eid))
        action = await player.policy.act(request, actx)
        usage = action.usage + actx.usage if action.usage.calls or action.usage.effort_seconds else actx.usage
        action.usage = usage
        verifs: list[Verification] = []
        shown = action.text
        vs = self.verifiers_for(role)
        if vs and action.text:
            vp = self.mechanism.verification
            assert vp is not None
            for claim in parse_claims(action.text, role):
                v = vs.get(claim.kind)
                if v is None:
                    verifs.append(Verification(claim=claim, status="unknown_kind"))
                    continue
                if vp.budget_per_role is not None and self._verif_counts[role] >= vp.budget_per_role:
                    verifs.append(Verification(claim=claim, status="over_budget"))
                    continue
                self._verif_counts[role] += 1
                try:
                    res = await v.verify(claim, self.item, self)
                except Exception as e:  # verifier failures are logged, not fatal
                    res = Verification(claim=claim, status="error", detail=repr(e))
                usage = usage + res.usage
                verifs.append(res)
            shown = annotate(action.text, verifs, display=vp.display, show_output=vp.show_output)
        return _Produced(action=action, verifications=verifs, shown=shown, usage=usage)

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
        **fields: Any,
    ) -> Action:
        """Ask ``role`` for an action and record the turn.

        Either pass a full ``request`` or its fields (``kind``, ``prompt``, ``options``, ...). A string
        prompt becomes a single user message. ``visible_to`` restricts who sees the turn (the author
        always does); None means everyone.
        """
        if role not in self.players:
            raise KeyError(f"no player for role {role!r}")
        if request is None:
            if isinstance(prompt, str):
                prompt = [Message.user(prompt)]
            request = ActionRequest(kind=kind, prompt=list(prompt or []), phase=phase, **fields)
        # everything up to the first await runs synchronously: slot, key and view are deterministic
        slot = self._slot
        self._slot += 1
        self._slot_group[slot] = group
        request = request.model_copy(update={"view": self.view(role, exclude_group=group), "phase": request.phase or phase})
        key = self._node_key(role, request.phase, slot, group)

        async def sample(i: int) -> _Produced:
            return await self._produce(role, request, i, key)

        if self.branch is not None:
            produced, idx = await self.branch.decide(key, role, request.phase, group, slot, sample)
            produced = produced.model_copy(deep=True)
            node, cand = key, idx
        else:
            produced = await sample(0)
            node, cand = None, None
            self.usage[role] = self.usage[role] + produced.usage
        self._decisions[slot] = (key, cand or 0)

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
                    node=node, candidate=cand, group=group, metadata=a.metadata,
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
    def total_usage(self) -> Usage:
        total = Usage()
        for u in self.usage.values():
            total = total + u
        return total
