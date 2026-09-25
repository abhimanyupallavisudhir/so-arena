"""Mechanisms and the episode context they are written against.

A **mechanism** = a *protocol* (game form: who acts when, seeing what) + a *reward rule*
(payoffs to trainable roles). It is domain-general: it talks to the task only through a
ground-truth-free :class:`~oversight_arena.core.task.TaskView`, and to agents only through
:meth:`EpisodeContext.ask`.

Writing a mechanism::

    class MyProtocol(Mechanism):
        rounds: int = 2
        reward: RewardRule = JudgeProbability(transform="log")

        def roles(self):
            return [RoleSpec(name="expert", kind="expert"),
                    RoleSpec(name="judge", kind="judge", trainable=False)]

        async def run(self, ctx):
            pos = ctx.position("expert") or ctx.task.option_ids[0]
            for t in range(self.rounds):
                await ctx.ask("expert", f"Argue for ({pos}).", turn=t)
            v = await ctx.ask("judge", "Which option is correct?",
                              response=ResponseSpec.distribution(ctx.task.option_ids))
            ctx.set_outcome(probs=v.data["probs"], positions={"expert": pos},
                            decision=v.data["choice"])
"""

from __future__ import annotations

import asyncio
import inspect
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

from ..agents.base import Agent, Observation, ResponseSpec
from ..channels.evidence import EvidencePolicy, Verifier, VerifyEnv, annotate, claim_help_text, extract_claims
from ..channels.gt_channels import GTChannel
from ..core.episode import ChannelUse
from ..core.rewards import NoReward, RewardRule
from ..core.roles import RoleSpec
from ..core.strategy import BoundStrategy, fill_placeholders
from ..core.task import Task, TaskView
from ..core.tools import Tool
from ..core.transcript import Entry, Evidence, Transcript
from ..core.types import ToolTrace, Usage
from ..core.util import rng_for, stable_hash

if TYPE_CHECKING:  # pragma: no cover
    from ..domains.base import Domain, Environment


class Mechanism(BaseModel, ABC):
    """Base class for scalable-oversight mechanisms (protocol + reward rule)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: ClassVar[str] = "mechanism"
    reward: RewardRule = Field(default_factory=NoReward)
    evidence: EvidencePolicy | None = None
    reveal_incentives: bool = True  # tell agents how they are rewarded
    max_words: int | None = 200
    label: str | None = None  # display name override

    @abstractmethod
    def roles(self) -> list[RoleSpec]: ...

    @abstractmethod
    async def run(self, ctx: "EpisodeContext") -> None: ...

    # --------------------------------------------------------------- prompts
    def brief(self, role: str, ctx: "EpisodeContext") -> str:
        """Rules-of-the-game text shown to ``role`` (override for mechanism-specific briefs)."""
        spec = ctx.role_spec(role)
        return spec.description or f"You are {spec.display} in a {self.display_name} protocol."

    def incentive_text(self, role: str, ctx: "EpisodeContext") -> str:
        """How ``role`` is rewarded (shown when ``reveal_incentives``). Override per mechanism."""
        return ""

    # --------------------------------------------------------------- identity
    @property
    def display_name(self) -> str:
        return self.label or self.name

    def config(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["reward"] = self.reward.name
        return d

    def config_hash(self) -> str:
        return stable_hash(type(self).__name__, self.config(), length=12)

    def describe(self) -> dict[str, Any]:
        return {"mechanism": self.name, **self.config()}

    def with_reward(self, reward: RewardRule) -> "Mechanism":
        return self.model_copy(update={"reward": reward})


class EpisodeContext:
    """The API mechanisms use to run an episode. Holds the GT-firewalled view of the task.

    Information flow is enforced here: each role observes a task view filtered by its
    clearance, and transcript entries with other roles' private reasoning, unshared tool traces,
    evidence it may not see and harness-private data (``_``-prefixed keys) removed. Strategy
    stances and tags (which encode ground truth, e.g. "argue the *incorrect* answer") are private
    to the harness: mechanisms only get :meth:`position`.
    """

    def __init__(
        self,
        *,
        mechanism: Mechanism,
        task: Task,
        agents: dict[str, Agent],
        bound: dict[str, BoundStrategy],
        clearances: dict[str, set[str]],
        domain: "Domain | None" = None,
        env: "Environment | None" = None,
        verifiers: list[Verifier] | None = None,
        episode_key: str = "",
        seed: int = 0,
    ):
        self.mechanism = mechanism
        self._task_full = task  # private: only GT channels may touch this
        self.task: TaskView = task.view()
        self.agents = agents
        self._bound = bound  # private: stances/tags encode ground truth
        self.clearances = clearances
        self.domain = domain
        self.env = env
        self.episode_key = episode_key
        self.seed = seed
        self.rng = rng_for("episode", episode_key, seed)
        self.transcript = Transcript()
        self.outcome: dict[str, Any] = {}
        self.usage: dict[str, Usage] = {}
        self.channel_uses: list[ChannelUse] = []
        self.meta: dict[str, Any] = {}
        self._roles = {r.name: r for r in mechanism.roles()}
        self._verifiers = {v.tag: v for v in (verifiers or [])}
        self._evidence_spent: dict[str, float] = {}
        self._assigned: dict[str, str] = {}
        self._chosen: dict[str, str] = {}

    # --------------------------------------------------------------- roles / positions
    @property
    def roles(self) -> list[RoleSpec]:
        return list(self._roles.values())

    def role_spec(self, role: str) -> RoleSpec:
        return self._roles[role]

    def title(self, role: str) -> str:
        return self._roles[role].display if role in self._roles else role

    def position(self, role: str) -> str | None:
        """The option ``role`` argues for: chosen in-protocol (:meth:`assign`), else assigned by
        the profile, else by :meth:`positions`."""
        if role in self._chosen:
            return self._chosen[role]
        b = self._bound.get(role)
        return (b.target if b else None) or self._assigned.get(role)

    def assign(self, role: str, option: str) -> None:
        """Fix ``role``'s position in-protocol (e.g. the side a debater *chose* in open debate);
        takes precedence over profile targets from then on."""
        self._chosen[role] = option

    def positions(self, roles: list[str], distinct: bool = True) -> dict[str, str]:
        """Positions for ``roles``: profile-assigned where given, remaining options otherwise.
        Assignments are remembered, so agents are told the position the mechanism gave them."""
        out: dict[str, str] = {}
        used: set[str] = set()
        for r in roles:
            b = self._bound.get(r)
            p = self._chosen.get(r) or (b.target if b else None)
            if p is not None:
                out[r] = p
                used.add(p)
        free = [o for o in self.task.option_ids if not (distinct and o in used)]
        for r in roles:
            if r not in out:
                if r in self._assigned:
                    out[r] = self._assigned[r]
                    continue
                if not free:
                    free = list(self.task.option_ids)
                out[r] = free.pop(0)
                self._assigned[r] = out[r]
        return out

    # --------------------------------------------------------------- observation
    def clearance(self, role: str) -> set[str]:
        return set(self.clearances.get(role, set()))

    def task_view(self, role: str) -> TaskView:
        """The task as ``role`` may see it (privileged info blocks filtered by clearance)."""
        return self.task.restricted(self.clearance(role))

    def task_text(self, role: str) -> str:
        from ..domains.base import default_render

        if self.domain is not None:
            text = self.domain.render_task(self.task, self.clearance(role), self._roles.get(role))
        else:
            text = default_render(self.task, self.clearance(role))
        if self.env is not None:
            extra = self.env.observation_text(role)
            if extra:
                text += "\n\n" + extra
        return text

    def tools_for(self, role: str) -> list[Tool]:
        tools: list[Tool] = []
        if self.env is not None:
            tools.extend(self.env.tools_for(role, self.clearance(role)))
        pol = self.mechanism.evidence
        if pol is not None and role in pol.requests:
            tools.extend(self._verification_request_tools(role))
        return tools

    def visible_entries(self, role: str) -> list[Entry]:
        """Transcript entries visible to ``role``, sanitised for it (see class docstring)."""
        return [_sanitize(e, role) for e in self.transcript.visible(role)]

    def observe(
        self,
        role: str,
        prompt: str = "",
        response: ResponseSpec | None = None,
        *,
        step: str | None = None,
        turn: int | None = None,
        tools: bool | list[Tool] = True,
        extra_brief: str = "",
    ) -> Observation:
        spec = self._roles[role]
        b = self._bound.get(role)
        brief = self.mechanism.brief(role, self)
        if self.mechanism.reveal_incentives:
            inc = self.mechanism.incentive_text(role, self)
            if inc:
                brief += "\n\n## How you are scored\n" + inc
        if extra_brief:
            brief += "\n\n" + extra_brief
        response = response or ResponseSpec.text(self.mechanism.max_words)
        if isinstance(tools, list):
            tool_list = tools
        else:
            tool_list = self.tools_for(role) if tools else []
        titles = {r: self.title(r) for r in self._roles}
        target = self.position(role)
        target_text = ""
        if target is not None:
            try:
                target_text = self.task.option(target).text
            except KeyError:
                target_text = str(target)
        instructions = b.instructions if b else ""
        if b is not None and b.target is None:  # position assigned by the mechanism (or none at all)
            instructions = fill_placeholders(instructions, {"target": target or "", "target_text": target_text})
        return Observation(
            role=role,
            role_title=spec.display,
            role_kind=spec.kind,
            mechanism=self.mechanism.display_name,
            brief=brief,
            strategy=instructions,
            target=target,
            target_text=target_text,
            task=self.task_view(role),
            task_text=self.task_text(role),
            private=self._private_data(role),
            entries=self.visible_entries(role),
            transcript_text=self.transcript.render(for_role=role, titles=titles),
            prompt=prompt,
            response=response,
            tools=tool_list,
            claim_help=self._claim_help(role),
            seed=sample_index(role, b.seed if b else 0, self.seed),
            turn=turn,
            step=step,
            params=dict(b.params) if b else {},
            tags=dict(b.tags) if b else {},
        )

    def _private_data(self, role: str) -> dict[str, Any]:
        """Structured privileged data a role is cleared for (for programmatic agents)."""
        cl = self.clearance(role)
        t = self._task_full
        out = {
            k: v for k, v in t.resources.items()
            if (k in t.resource_access and (t.resource_access[k] == "public" or t.resource_access[k] in cl))
        }
        if self.env is not None:
            out.update(self.env.observation(role))
        return out

    # --------------------------------------------------------------- acting
    async def ask(
        self,
        role: str,
        prompt: str = "",
        *,
        response: ResponseSpec | None = None,
        step: str | None = None,
        turn: int | None = None,
        visible_to: list[str] | None = None,
        reasoning_visible_to: list[str] | None = None,
        tools: bool | list[Tool] = True,
        kind: str = "message",
        record: bool = True,
        data: dict[str, Any] | None = None,
        extra_brief: str = "",
        observation: Observation | None = None,
        verify: bool = True,
    ) -> Entry:
        """Ask ``role`` to act; verify its claims (unless ``verify=False``); append (and return)
        the transcript entry."""
        agent = self.agents[role]
        obs = observation or self.observe(role, prompt, response, step=step, turn=turn, tools=tools, extra_brief=extra_brief)
        action = await agent.act(obs)
        _normalize_parsed(action, obs.response)
        entry = Entry(
            kind=kind,  # type: ignore[arg-type]
            role=role,
            content=strip_status_marks(action.text),  # only trusted code may mark claims as checked
            visible_to=visible_to,
            reasoning=action.reasoning,
            reasoning_visible_to=reasoning_visible_to,
            tool_trace=action.tool_trace,
            data={**action.parsed, **(data or {})},
            step=step,
            turn=turn,
            usage=action.usage,
        )
        if action.error:
            entry.data["_parse_error"] = action.error
        self.usage[role] = self.usage.get(role, Usage()) + action.usage
        pol = self.mechanism.evidence
        if pol is not None:
            if verify and pol.auto_verify and pol.applies_to(role):
                await self._verify_inline(role, entry, pol)
            if pol.share_tool_results and entry.tool_trace:
                entry.data["_share_tools"] = True
        if record:
            self.transcript.add(entry)
        return entry

    async def gather(self, *coros: Awaitable[Any]) -> list[Any]:
        return list(await asyncio.gather(*coros))

    async def simultaneous(self, asks: dict[str, dict[str, Any]]) -> dict[str, Entry]:
        """All roles in ``asks`` act on the same transcript state; entries revealed together.
        Every observation is built before anyone acts, so nothing produced during the round
        (e.g. requested verifications) can leak into another role's move."""
        obs = {
            r: self.observe(r, kw.get("prompt", ""), kw.get("response"), step=kw.get("step"), turn=kw.get("turn"),
                            tools=kw.get("tools", True), extra_brief=kw.get("extra_brief", ""))
            for r, kw in asks.items()
        }
        pending = {r: self.ask(r, record=False, observation=obs[r], **kw) for r, kw in asks.items()}
        results = await asyncio.gather(*pending.values())
        out = {}
        for r, e in zip(pending.keys(), results):
            out[r] = self.transcript.add(e)
        return out

    def post(
        self, content: str, *, kind: str = "system", role: str | None = None,
        visible_to: list[str] | None = None, data: dict[str, Any] | None = None, step: str | None = None,
    ) -> Entry:
        return self.transcript.add(
            Entry(kind=kind, role=role, content=content, visible_to=visible_to, data=data or {}, step=step)  # type: ignore[arg-type]
        )

    def reveal(self, *entries: Entry, to: list[str] | None = None) -> None:
        """Make entries visible to more roles (``to=None``: everyone). Public entries stay public."""
        for e in entries:
            if e.visible_to is None:
                continue
            e.visible_to = None if to is None else sorted(set(e.visible_to + to))

    def set_outcome(self, **kw: Any) -> None:
        self.outcome.update(kw)

    def transcript_len(self) -> int:
        return len(self.transcript.entries)

    # --------------------------------------------------------------- evidence
    def enabled_verifiers(self) -> list[Verifier]:
        pol = self.mechanism.evidence
        if pol is None:
            return []
        vs = list(self._verifiers.values())
        if pol.verifiers is not None:
            vs = [v for v in vs if v.name in pol.verifiers or v.tag in pol.verifiers]
        return vs

    def _claim_help(self, role: str) -> str:
        pol = self.mechanism.evidence
        if pol is None or not pol.applies_to(role):
            return ""
        return claim_help_text(self.enabled_verifiers(), pol.budget)

    def _budget_left(self, role: str, pol: EvidencePolicy) -> float:
        if pol.budget is None:
            return float("inf")
        return pol.budget - self._evidence_spent.get(role, 0.0)

    async def verify(self, role: str, tag: str, content: str, args: dict[str, str] | None = None) -> Evidence:
        """Explicitly verify a claim on behalf of ``role`` (budget-checked)."""
        return (await self._verify(role, tag, content, args))[0]

    async def _verify(self, role: str, tag: str, content: str, args: dict[str, str] | None = None) -> tuple[Evidence, dict[str, Any] | None]:
        from ..channels.evidence import Claim

        pol = self.mechanism.evidence or EvidencePolicy()
        v = self._verifiers.get(tag)
        if v is None:
            return Evidence(verifier="none", kind=tag, claim=content, result=f"no verifier for <{tag}>", requested_by=role), None
        if self._budget_left(role, pol) < v.cost:
            return Evidence(verifier=v.name, kind=tag, claim=content, result="not checked (budget exhausted)", requested_by=role), None
        return await self._run_verifier(v, Claim(kind=tag, content=content, args=args or {}), role, pol)

    async def _run_verifier(self, v: Verifier, claim: Any, role: str, pol: EvidencePolicy) -> tuple[Evidence, dict[str, Any]]:
        """Run a verifier. Returns (evidence as shown, harness-private truth record). With
        verification noise the shown verdict may be flipped; the truth never reaches agents."""
        venv = VerifyEnv(view=self.task, resources=self._task_full.resources, role=role, env=self.env)
        try:
            ev = await v.verify(claim, venv)
        except Exception as e:  # verifier failure = unverified
            ev = Evidence(verifier=v.name, kind=claim.kind, claim=claim.content, result=f"verifier error: {e}")
        self._evidence_spent[role] = self._evidence_spent.get(role, 0.0) + v.cost
        ev.cost = v.cost
        ev.requested_by = role
        truth = {"verified": ev.verified, "result": ev.result, "flipped": False}
        if pol.noise > 0 and ev.verified is not None:
            if rng_for("vnoise", self.episode_key, role, claim.content).random() < pol.noise:
                shown = ev.model_copy(update={"verified": not ev.verified})
                ev = v.forge(claim, shown)
                truth["flipped"] = True
        if pol.show_to is not None:
            ev.visible_to = sorted(set(pol.show_to))
        return ev, truth

    async def _verify_inline(self, role: str, entry: Entry, pol: EvidencePolicy) -> None:
        vs = {v.tag: v for v in self.enabled_verifiers()}
        claims = extract_claims(entry.content, list(vs))
        if not claims:
            return
        statuses = []
        truths = []
        for c in claims:
            v = vs[c.kind]
            if self._budget_left(role, pol) < v.cost:
                statuses.append("UNCHECKED")
                continue
            ev, truth = await self._run_verifier(v, c, role, pol)
            entry.evidence.append(ev)
            truths.append(truth)
            statuses.append({True: "VERIFIED", False: "REFUTED", None: "CHECKED"}[ev.verified])
        if pol.show_to is None:  # verdicts are public: mark them in the message itself
            if pol.annotate_unchecked or any(s != "UNCHECKED" for s in statuses):
                entry.content = annotate(entry.content, claims, statuses)
            entry.data["claims"] = [{"kind": c.kind, "content": c.content, "status": s} for c, s in zip(claims, statuses)]
        entry.data["_claims"] = [{"kind": c.kind, "content": c.content, "status": s} for c, s in zip(claims, statuses)]
        entry.data["_evidence_truth"] = truths

    def _verification_request_tools(self, role: str) -> list[Tool]:
        tools = []
        pol = self.mechanism.evidence or EvidencePolicy()
        for v in self.enabled_verifiers():
            async def _fn(content: str, _v: Verifier = v) -> str:
                ev, truth = await self._verify(role, _v.tag, content)
                vis = None if pol.show_to is None else sorted(set(pol.show_to) | {role})
                self.transcript.add(Entry(kind="evidence", role=role, content=ev.render(), evidence=[ev], visible_to=vis,
                                          data={"_evidence_truth": [truth] if truth else []}))
                return ev.render()

            tools.append(
                Tool(
                    name=f"verify_{v.tag}",
                    description=f"Ask trusted code to check a claim. {v.help} (cost {v.cost:g})",
                    parameters={"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
                    fn=_fn,
                    group="public",
                )
            )
        return tools

    # --------------------------------------------------------------- ground-truth channels
    async def query(self, channel: GTChannel, role: str | None = None, **kw: Any) -> Any:
        """Query a declared ground-truth channel (audit, probe, resolution)."""
        return await channel.query(self, role, **kw)

    def log_channel(self, channel: str, role: str | None, cost: float, result: Any) -> None:
        self.channel_uses.append(ChannelUse(channel=channel, role=role, cost=cost, result=result))

    def log(self, **kw: Any) -> None:
        self.meta.update(kw)


_STATUS_ATTR = re.compile(r'(<[A-Za-z_][\w\-]*\b[^<>]*?)\s+status\s*=\s*"[^"]*"')


def strip_status_marks(text: str) -> str:
    """Remove ``status="..."`` attributes from claim markup: only trusted code may mark claims."""
    prev = None
    while prev != text:
        prev, text = text, _STATUS_ATTR.sub(r"\1", text)
    return text


def _sanitize(e: Entry, role: str) -> Entry:
    """A copy of ``e`` as ``role`` may see it."""
    own = e.role == role
    return e.model_copy(update={
        "reasoning": e.reasoning if (own or e.reasoning_visible(role)) else None,
        "tool_trace": list(e.tool_trace) if (own or e.data.get("_share_tools")) else [],
        "evidence": [ev for ev in e.evidence if ev.visible_to is None or role in ev.visible_to],
        "data": {k: v for k, v in e.data.items() if not str(k).startswith("_")},
    })


def sample_index(role: str, sample: int, episode_seed: int) -> int:
    """Model sample index for a role's turn: distinct per (role, sample, episode seed), so roles
    with identical prompts (e.g. independent reporters) draw independent samples."""
    return int(stable_hash("sample", role, sample, episode_seed, length=12), 16) % (2**31)


def _normalize_parsed(action: Any, spec: ResponseSpec) -> None:
    """Make structured answers uniform whatever the agent type (LLM, scripted, human)."""
    p = action.parsed
    opts = spec.options or []
    if spec.kind == "distribution" and opts:
        probs = p.get("probs")
        if isinstance(probs, dict):
            clean = {o: max(float(probs.get(o, 0.0) or 0.0), 0.0) for o in opts}
            tot = sum(clean.values())
            p["probs"] = {o: v / tot for o, v in clean.items()} if tot > 0 else {o: 1 / len(opts) for o in opts}
        elif p.get("choice") in opts:
            p["probs"] = {o: 1.0 if o == p["choice"] else 0.0 for o in opts}
        if "probs" in p and p.get("choice") not in opts:
            p["choice"] = max(p["probs"], key=p["probs"].get)
    elif spec.kind == "choice" and opts and p.get("choice") not in opts and isinstance(p.get("probs"), dict):
        p["choice"] = max(p["probs"], key=p["probs"].get)


# ---------------------------------------------------------------------- decorator API
def mechanism(
    fn: Callable[..., Awaitable[None]] | None = None,
    *,
    roles: list[RoleSpec] | Callable[[Any], list[RoleSpec]],
    reward: RewardRule | None = None,
    name: str | None = None,
    brief: Callable[[Any, str, EpisodeContext], str] | None = None,
) -> Any:
    """Define a mechanism from an async function ``fn(ctx, **params)``.

    Keyword parameters of ``fn`` (with defaults) become validated, serialisable config fields.
    """

    def build(f: Callable[..., Awaitable[None]]) -> type[Mechanism]:
        sig = inspect.signature(f)
        fields: dict[str, Any] = {}
        for p in list(sig.parameters.values())[1:]:
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else Any
            default = p.default if p.default is not inspect.Parameter.empty else ...
            fields[p.name] = (ann, default)
        fields["reward"] = (RewardRule, Field(default_factory=lambda: reward or NoReward()))
        mech_name = name or f.__name__

        async def _run(self: Mechanism, ctx: EpisodeContext) -> None:
            kwargs = {k: getattr(self, k) for k in list(sig.parameters)[1:]}
            await f(ctx, **kwargs)

        def _roles(self: Mechanism) -> list[RoleSpec]:
            return roles(self) if callable(roles) else list(roles)

        ns: dict[str, Any] = {"run": _run, "roles": _roles, "__doc__": f.__doc__}
        if brief is not None:
            ns["brief"] = lambda self, role, ctx: brief(self, role, ctx)
        cls = create_model(  # type: ignore[call-overload]
            "".join(w.title() for w in mech_name.split("_")),
            __base__=Mechanism,
            __module__=f.__module__,
            **fields,
        )
        for k, v in ns.items():
            setattr(cls, k, v)
        cls.name = mech_name  # type: ignore[misc]
        cls.__abstractmethods__ = frozenset()  # type: ignore[misc]
        return cls

    if fn is not None:
        return build(fn)
    return build
