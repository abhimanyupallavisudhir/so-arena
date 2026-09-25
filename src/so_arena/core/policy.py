"""Policies: behaviours for a role (strategies, in game-theoretic terms).

A policy maps an :class:`ActionRequest` to an :class:`Action`. The same mechanism can be played by
prompted LLMs (:class:`LLMPolicy`), scripted fixtures (:class:`ScriptedPolicy`,
:class:`FunctionPolicy`), fixed outputs (:class:`FixedPolicy`, e.g. an RL trainee's completion),
externally driven agents (:class:`ExternalPolicy`, for step-wise RL environments), best-of-N
optimizers (:class:`BestOfNPolicy`) and mixtures (:class:`MixturePolicy`, meta-strategies).

A behaviour's *label* (e.g. ``"honest"``, ``"argue_false"``, ``"prompt#17"``) is what analyses group by.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import contextvars
import enum
import hashlib
import math
import random
import types
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

from so_arena.core.actions import Action, ActionRequest
from so_arena.core.parsing import (
    parse_choice,
    parse_json_object,
    parse_probabilities,
    parse_score,
    probs_from_logprobs,
    probs_from_mapping,
    split_reasoning,
    truncate_words,
)
from so_arena.core.tools import TOOL_CALL_RE, Tool, tool_instructions
from so_arena.core.types import Completion, GenerateOptions, Message, Usage

if TYPE_CHECKING:
    from so_arena.core.game import Game
    from so_arena.core.state import Workspace, WorkspaceSlot
    from so_arena.models.base import Model


def stable_hash(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


_TOOL_CALLS: contextvars.ContextVar["list[dict[str, Any]] | None"] = contextvars.ContextVar("so_arena_tool_calls",
                                                                                             default=None)


@contextlib.contextmanager
def recording_tool_calls(sink: list[dict[str, Any]] | None) -> Iterator[list[dict[str, Any]] | None]:
    """Collect the tool calls made in this context into ``sink``: the trusted record of a decision.

    The game opens one sink per sampled candidate. It is context-local, so wrappers that sample several
    candidates per decision (e.g. :class:`BestOfNPolicy`) give each candidate a sink of its own and pass
    on only the kept candidate's records - a discarded attempt is not part of what the role did.
    """
    token = _TOOL_CALLS.set(sink)
    try:
        yield sink
    finally:
        _TOOL_CALLS.reset(token)


def record_tool_call(record: dict[str, Any]) -> None:
    """Add a tool call to the current context's record (no-op outside a recorded decision)."""
    sink = _TOOL_CALLS.get()
    if sink is not None:
        sink.append(record)


def _plain(v: Any) -> bool:
    if v is None or isinstance(v, (bool, int, float, str, enum.Enum, types.FunctionType, type)):
        return True
    return isinstance(v, (tuple, frozenset)) and all(_plain(x) for x in v)


def describe_callable(fn: Callable[..., Any]) -> Any:
    """A process-independent description of a function: its qualified name, plus the plain values it
    closes over or defaults to (so ``synthetic_arguer(sd=1)`` and ``synthetic_arguer(sd=2)`` differ).

    Only scalars, functions, classes and tuples of them count: mutable closure values (a list of calls, a
    cache) are usually run state, which must not change episode ids between a run and its resumption.
    """
    from so_arena.core.mechanism import describe_config

    desc = describe_config(fn)
    if not isinstance(fn, types.FunctionType):
        return desc
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
    params = {k: describe_config(v) for k, v in sorted(values.items()) if _plain(v)}
    return {"function": desc, "params": params} if params else desc


class ActContext:
    """Per-call context the game hands to a policy.

    ``seed`` identifies the decision (the game derives it from the run seed, the item and the decision's
    node); ``rng`` is seeded from it, the role and ``sample_index``. Wrappers that act several times per
    decision (e.g. :class:`BestOfNPolicy`) derive their sub-contexts' seeds from it.
    """

    def __init__(
        self,
        *,
        role: str,
        sample_index: int = 0,
        game: "Game | None" = None,
        tools: dict[str, Tool] | None = None,
        seed: int = 0,
        workspace: "Workspace | WorkspaceSlot | None" = None,
    ):
        self.role = role
        self.sample_index = sample_index
        self.game = game
        self.tools = tools or {}
        self.seed = seed
        self.rng = random.Random(stable_hash(seed, role, sample_index))
        self.usage = Usage()
        from so_arena.core.state import Workspace as _Workspace, WorkspaceSlot as _Slot

        self._slot = _Slot.of(workspace) if isinstance(workspace, _Workspace) else workspace

    @property
    def workspace(self) -> "Workspace | None":
        """The working copy of the task's state this decision acts on (roles with state access only),
        forked on first use; scripted policies may edit it directly, LLM policies use workspace tools."""
        return self._slot.get() if self._slot is not None else None

    async def generate(self, model: "Model", messages: Sequence[Message], options: GenerateOptions | None = None,
                       *, sample_offset: int = 0) -> Completion:
        out = await model.generate(list(messages), options, sample_index=self.sample_index * 1000 + sample_offset)
        self.usage = self.usage + out.usage
        return out

    async def call_tool(self, name: str, args: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            # an attempt is part of the trusted record too, so it matches what an LLM policy reports
            msg = f"error: unknown tool {name!r}; available: {sorted(self.tools)}"
            record_tool_call({"name": name, "args": args, "result": msg, "error": True})
            return msg
        from so_arena.core.state import _CURRENT, using_workspace

        def _current_slot():
            return _CURRENT.get()

        item = self.game.item if self.game is not None else None
        with using_workspace(self._slot if self._slot is not None else _current_slot()):
            res = await tool.call(args, item, self.game)  # type: ignore[arg-type]
        self.usage = self.usage + res.usage
        return res.output


class Policy(abc.ABC):
    """Base class. Subclasses implement :meth:`act`."""

    def __init__(self, *, label: str | None = None, id: str | None = None):
        self.label = label
        self._id = id

    @property
    def id(self) -> str:
        return self._id or self.label or type(self).__name__

    @abc.abstractmethod
    async def act(self, request: ActionRequest, ctx: ActContext) -> Action: ...

    def describe(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "id": self.id, "label": self.label}

    @property
    def model_name(self) -> str | None:
        return None


# ------------------------------------------------------------------------------------ helpers


def _with_system(messages: list[Message], extra: str) -> list[Message]:
    if not extra:
        return messages
    if messages and messages[0].role == "system":
        return [Message.system(messages[0].content + "\n\n" + extra)] + messages[1:]
    return [Message.system(extra)] + messages


def _append_user(messages: list[Message], text: str) -> list[Message]:
    if not text:
        return messages
    if messages and messages[-1].role == "user":
        return messages[:-1] + [Message.user(messages[-1].content + "\n\n" + text)]
    return messages + [Message.user(text)]


def format_instructions(request: ActionRequest, elicitation: str = "ask") -> str:
    k = request.kind
    if k == "choice":
        opts = ", ".join(request.options or [])
        return f"Think it through, then end your reply with a line 'Answer: X' where X is one of: {opts}."
    if k == "probabilities":
        opts = request.options or []
        if elicitation == "logprobs":
            return f"Reply with only the label of the correct answer ({', '.join(opts)}) and nothing else."
        example = "{" + ", ".join(f'"{o}": <probability>' for o in opts) + "}"
        return (
            "Think it through, then end your reply with your probability for each answer as a JSON "
            f"object on the last line: {example}. The probabilities must sum to 1."
        )
    if k == "score":
        lo, hi = request.score_range or (0, 10)
        meaning = f" ({request.score_meaning})" if request.score_meaning else ""
        return f"Think it through, then end with a line 'Score: N' where N is a number from {lo:g} to {hi:g}{meaning}."
    if k == "json":
        keys = ", ".join(request.json_keys or [])
        return f"End your reply with a JSON object with keys: {keys}."
    if request.word_limit:
        return f"(Limit: {request.word_limit} words.)"
    return ""


def neutral_action(request: ActionRequest, text: str = "") -> Action:
    """Fallback action when a response cannot be parsed."""
    opts = request.options or []
    if request.kind == "probabilities" and opts:
        return Action(text=text, probs={o: 1 / len(opts) for o in opts}, parse_ok=False)
    if request.kind == "choice" and opts:
        return Action(text=text, choice=opts[0], parse_ok=False)
    if request.kind == "score":
        lo, hi = request.score_range or (0, 10)
        return Action(text=text, score=(lo + hi) / 2, parse_ok=False)
    if request.kind == "json":
        return Action(text=text, data={}, parse_ok=False)
    return Action(text=text, parse_ok=bool(text))


def option_texts(request: ActionRequest) -> dict[str, str]:
    """Answer texts of the request's options (from the item the role sees, overridden by ``option_texts``)."""
    item = request.view.item if request.view is not None else None
    texts = {a.label: a.text for a in (item.answers or [])} if item is not None else {}
    return {**texts, **(request.option_texts or {})}


def finalize_from_text(request: ActionRequest, text: str, *, reasoning: str | None = None) -> Action:
    """Turn raw text into a typed action according to ``request.kind``."""
    opts = request.options or []
    if request.kind == "text":
        return Action(text=truncate_words(text, request.word_limit), reasoning=reasoning)
    if request.kind == "choice":
        c = parse_choice(text, opts)
        if c is None:
            a = neutral_action(request, text)
        else:
            a = Action(text=text, choice=c)
    elif request.kind == "probabilities":
        p = parse_probabilities(text, opts, option_texts(request))
        a = neutral_action(request, text) if p is None else Action(text=text, probs=p)
    elif request.kind == "score":
        lo, hi = request.score_range or (0, 10)
        s = parse_score(text, lo, hi)
        a = neutral_action(request, text) if s is None else Action(text=text, score=s)
    else:
        d = parse_json_object(text)
        a = neutral_action(request, text) if d is None else Action(text=text, data=d)
    a.reasoning = reasoning
    return a


# ------------------------------------------------------------------------------------ LLM policy

# the investigation before a decision that is not free text (see LLMPolicy's ``investigate``)
INVESTIGATE_PROMPT = ("Before you answer, investigate: use your tools to check whatever you need for this decision. "
                      "Then write brief notes on what you found - they are private, only you will see them. Do not "
                      "give your answer yet: you will be asked for it next.")
DECIDE_PROMPT = "Your investigation is over (no more tool calls). Now give your answer."


class LLMPolicy(Policy):
    """A prompted language model.

    Args:
        model: model name or :class:`Model`.
        strategy: behavioural instructions (e.g. from prompt optimization) added to the system prompt.
        system_prompt: extra system text (persona, etc.).
        elicitation: how probabilities are obtained - ``"ask"`` (verbalized JSON), ``"logprobs"``
            (next-token probabilities over labels, "tip of the tongue"), ``"vote"`` (frequency over
            ``n_votes`` sampled choices).
        cot: ask the model to reason privately in ``<thinking>`` tags; the reasoning is recorded but
            only shown to roles allowed to see it (e.g. a CoT monitor).
        use_tools: allow calling the tools the role is granted.
        investigate: before a decision that is not free text (a choice, probabilities, a score, JSON), let
            a role with tools investigate first: the tool loop, ending in brief private notes; the decision
            is then elicited (verbalized, by logprobs or by vote) after the investigation. Without it such
            decisions use no tools - e.g. a reviewer with read access to the work could not inspect it.
    """

    def __init__(
        self,
        model: "str | Model",
        *,
        strategy: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = 1024,
        elicitation: str = "ask",
        n_votes: int = 5,
        cot: bool = False,
        use_tools: bool = True,
        max_tool_calls: int = 6,
        reasoning_effort: str | None = None,
        seed: int | None = None,
        label: str | None = None,
        id: str | None = None,
        investigate: bool = True,
    ):
        from so_arena.models.base import get_model

        super().__init__(label=label, id=id)
        self.model = get_model(model)
        self.strategy = strategy
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.elicitation = elicitation
        self.n_votes = n_votes
        self.cot = cot
        self.use_tools = use_tools
        self.investigate = investigate
        self.max_tool_calls = max_tool_calls
        self.reasoning_effort = reasoning_effort
        self.seed = seed

    @property
    def id(self) -> str:
        if self._id:
            return self._id
        base = self.label or "llm"
        return f"{base}@{self.model.name}"

    @property
    def model_name(self) -> str | None:
        return self.model.name

    def describe(self) -> dict[str, Any]:
        return {
            "type": "LLMPolicy",
            "id": self.id,
            "label": self.label,
            "model": self.model.name,
            "strategy": self.strategy,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "elicitation": self.elicitation,
            "n_votes": self.n_votes,
            "cot": self.cot,
            "use_tools": self.use_tools,
            "investigate": self.investigate,
            "max_tool_calls": self.max_tool_calls,
            "reasoning_effort": self.reasoning_effort,
            "seed": self.seed,
        }

    def _options(self, **kw: Any) -> GenerateOptions:
        base = GenerateOptions(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            reasoning_effort=self.reasoning_effort,
        )
        return base.merged(**kw) if kw else base

    def _compose(self, request: ActionRequest, ctx: ActContext) -> list[Message]:
        msgs = list(request.prompt) or [Message.user("(no prompt)")]
        extra = []
        if self.system_prompt:
            extra.append(self.system_prompt)
        if self.strategy:
            extra.append("Your strategy:\n" + self.strategy)
        if self._tools_on(request, ctx) and (request.kind == "text" or self.investigate):
            extra.append(tool_instructions(ctx.tools, self.max_tool_calls))
        if self.cot and not (request.kind == "probabilities" and self.elicitation == "logprobs"):
            extra.append(
                "First reason privately inside <thinking>...</thinking> tags. Only the text outside "
                "the tags is shown to others."
            )
        return _with_system(msgs, "\n\n".join(extra))

    def _tools_on(self, request: ActionRequest, ctx: ActContext) -> bool:
        return bool(self.use_tools and request.allow_tools and ctx.tools)

    async def act(self, request: ActionRequest, ctx: ActContext) -> Action:
        msgs = self._compose(request, ctx)
        # Tools serve free text directly; any other decision (a reviewer's verdict, a monitor's score) is made
        # after an investigation, in whatever way it is elicited - logprobs and votes cannot run tools.
        calls: list[dict] = []
        notes: str | None = None
        if request.kind != "text" and self.investigate and self._tools_on(request, ctx):
            msgs, calls, notes = await self._investigate(msgs, ctx)
        if request.kind == "probabilities" and self.elicitation in ("logprobs", "vote"):
            action = await self._elicit_probs(request, msgs, ctx)
        else:
            msgs = _append_user(msgs, format_instructions(request, self.elicitation))
            text, reasoning, tool_calls, meta = await self._generate_with_tools(request, msgs, ctx,
                                                                                tools=request.kind == "text")
            action = finalize_from_text(request, text, reasoning=reasoning)
            action.tool_calls = tool_calls
            if meta:
                # white-box backends can attach e.g. probe readings to completions; monitors read them here
                action.metadata["completion"] = meta
                if "probe_scores" in meta:
                    action.metadata["probe_scores"] = meta["probe_scores"]
        if calls or notes:  # the investigation is part of the decision's record, and private
            action.tool_calls = calls + action.tool_calls
            action.reasoning = "\n\n".join(r for r in (notes, action.reasoning) if r) or None
        return action

    async def _tool_loop(self, msgs: list[Message], ctx: ActContext, *, tools: bool
                         ) -> tuple[str, list[Message], list[str], list[dict], dict]:
        """Generate; while the response calls a tool (and ``tools``), run it and continue with the result.

        Returns the final response (tool tags removed), the conversation it answers, native reasoning, the
        tool calls and the final completion's metadata. Simulated completions call no tools: one call.
        """
        tool_calls: list[dict] = []
        convo = list(msgs)
        reasoning_parts: list[str] = []
        for step in range(self.max_tool_calls + 1):
            out = await ctx.generate(self.model, convo, self._options(), sample_offset=step)
            raw = out.text
            meta = {k: v for k, v in out.metadata.items() if k != "simulated"}
            if out.reasoning:
                reasoning_parts.append(out.reasoning)
            m = TOOL_CALL_RE.search(raw) if tools else None
            if m is None or step == self.max_tool_calls:
                return TOOL_CALL_RE.sub("", raw), convo, reasoning_parts, tool_calls, meta
            name, args = m.group("name"), m.group("args").strip()
            result = await ctx.call_tool(name, args)
            tool_calls.append({"name": name, "args": args, "result": result})
            prefix = raw[: m.end()]
            convo = convo + [Message.assistant(prefix), Message.user(f'<tool_result name="{name}">\n{result}\n</tool_result>')]
        return "", convo, reasoning_parts, tool_calls, {}  # pragma: no cover

    async def _generate_with_tools(self, request, msgs, ctx, *, tools: bool | None = None
                                   ) -> tuple[str, str | None, list[dict], dict]:
        raw, _, reasoning_parts, tool_calls, meta = await self._tool_loop(
            msgs, ctx, tools=self._tools_on(request, ctx) if tools is None else tools)
        public, cot = split_reasoning(raw) if self.cot else (raw.strip(), None)
        if cot:
            reasoning_parts.append(cot)
        return public, ("\n\n".join(reasoning_parts) or None), tool_calls, meta

    async def _investigate(self, msgs: list[Message], ctx: ActContext) -> tuple[list[Message], list[dict], str | None]:
        """Use the tools, then write private notes. Returns the conversation in which the decision is then
        elicited (the investigation, the notes, and a request to decide), the tool calls, and the private
        reasoning (native reasoning and the notes)."""
        convo = _append_user(msgs, INVESTIGATE_PROMPT)
        raw, convo, reasoning, calls, _ = await self._tool_loop(convo, ctx, tools=True)
        notes = raw.strip()
        public, cot = split_reasoning(notes) if self.cot else (notes, None)
        decide = convo + [Message.assistant(notes or "(no notes)"), Message.user(DECIDE_PROMPT)]
        return decide, calls, "\n\n".join(r for r in (*reasoning, cot, public) if r) or None

    async def _elicit_probs(self, request, msgs, ctx) -> Action:
        opts = request.options or []
        if self.elicitation == "logprobs" and self.model.supports_logprobs:
            m = _append_user(msgs, format_instructions(request, "logprobs"))
            out = await ctx.generate(self.model, m, self._options(temperature=0.0, max_tokens=3, logprobs=True, top_logprobs=20))
            if out.metadata.get("simulated"):  # dry run: keep the real call pattern (one call, no vote fallback)
                return Action(text=out.text, probs={o: 1 / len(opts) for o in opts}, parse_ok=False,
                              metadata={"elicitation": "logprobs", "simulated": True})
            p = probs_from_logprobs(out.logprobs, opts)
            if p is not None:
                return Action(text=out.text, probs=p, metadata={"elicitation": "logprobs"})
            c = parse_choice(out.text, opts)
            if c is not None:
                eps = 1e-3
                return Action(text=out.text, probs={o: (1 - eps * (len(opts) - 1)) if o == c else eps for o in opts},
                              parse_ok=False, metadata={"elicitation": "logprobs-fallback-choice"})
        # vote (also the fallback when logprobs are unavailable)
        choice_req = request.model_copy(update={"kind": "choice"})
        m = _append_user(msgs, format_instructions(choice_req))
        n = self.n_votes if self.elicitation in ("vote", "logprobs") else 1
        outs = await asyncio.gather(*[
            ctx.generate(self.model, m, self._options(temperature=max(self.temperature or 0.0, 0.7)), sample_offset=i)
            for i in range(n)
        ])
        # each vote's private reasoning (<thinking>, native reasoning) stays private, as on the "ask" path:
        # only the public part is parsed and becomes the turn's text
        publics, reasoning = [], []
        for o in outs:
            public, cot = split_reasoning(o.text) if self.cot else (o.text, None)
            publics.append(public)
            reasoning += [r for r in (o.reasoning, cot) if r]
        votes = Counter(parse_choice(p, opts) for p in publics)
        alpha = 0.5
        total = sum(votes[o] for o in opts) + alpha * len(opts)
        probs = {o: (votes[o] + alpha) / total for o in opts}
        return Action(text=publics[0], reasoning="\n\n".join(dict.fromkeys(reasoning)) or None, probs=probs,
                      parse_ok=votes.get(None, 0) < n,
                      metadata={"elicitation": "vote", "votes": {str(k): v for k, v in votes.items()}})


# ------------------------------------------------------------------------------------ fixtures


class ScriptedPolicy(Policy):
    """Returns scripted outputs.

    ``script`` may be a list (used in order, cycling), a dict mapping phase -> text (or list), or a
    callable ``(request, ctx) -> str | Action | dict``. Text is parsed according to the request kind,
    so e.g. ``'{"A": 0.8, "B": 0.2}'`` works for a probabilities request.
    """

    def __init__(self, script: Any, *, label: str | None = None, id: str | None = None):
        super().__init__(label=label, id=id)
        self.script = script
        self._counters: dict[Any, int] = {}

    def describe(self) -> dict[str, Any]:
        from so_arena.core.mechanism import describe_config

        s = self.script
        return {**super().describe(), "script": describe_callable(s) if callable(s) else describe_config(s)}

    def _next(self, key: Any, seq: Sequence[Any]) -> Any:
        i = self._counters.get(key, 0)
        self._counters[key] = i + 1
        return seq[i % len(seq)]

    async def act(self, request: ActionRequest, ctx: ActContext) -> Action:
        s = self.script
        # sequences advance per (episode, role): each episode replays the script from the start
        episode = ctx.game.episode_id if ctx.game is not None else None
        if callable(s):
            out = s(request, ctx)
            if asyncio.iscoroutine(out):
                out = await out
        elif isinstance(s, dict):
            val = s.get(request.phase, s.get("*", ""))
            out = self._next((episode, ctx.role, request.phase), val) if isinstance(val, list) else val
        elif isinstance(s, list):
            out = self._next((episode, ctx.role), s)
        else:
            out = s
        return coerce_action(request, out)


def coerce_action(request: ActionRequest, out: Any) -> Action:
    """Turn a policy's output (an :class:`Action`, a dict, a number or text) into an action for ``request``.

    A dict means what the same JSON would mean as text: ``{"A": 0.8}`` for a probabilities request over
    A, B is A: 0.8, B: 0.2; ``{"choice": "B"}`` / ``{"score": 7}`` answer choice / score requests (and
    a dict without the field is a parse failure, not an action without a choice).
    """
    if isinstance(out, Action):
        return out
    if isinstance(out, dict):
        if request.kind == "probabilities":
            p = probs_from_mapping(out, request.options or [str(k) for k in out])
            return Action(text=str(out), probs=p) if p else neutral_action(request, str(out))
        if request.kind in ("choice", "score"):
            keys = ("choice", "answer") if request.kind == "choice" else ("score",)
            val = next((out[k] for k in keys if k in out), None)
            a = finalize_from_text(request, "" if val is None else f"{keys[0]}: {val}")
            a.text, a.data = str(out.get("text", "")), out
            return a
        return Action(text=str(out.get("text", "")), data=out)
    if isinstance(out, (int, float)) and request.kind == "score":
        return Action(text=str(out), score=float(out))
    text = str(out)
    public, reasoning = split_reasoning(text)
    return finalize_from_text(request, public, reasoning=reasoning)


class FunctionPolicy(ScriptedPolicy):
    """A policy given by a Python function ``fn(request, ctx) -> str | Action | dict`` (sync or async)."""

    def __init__(self, fn: Callable[[ActionRequest, ActContext], Any], *, label: str | None = None, id: str | None = None):
        super().__init__(fn, label=label, id=id)


class FixedPolicy(Policy):
    """Always returns the same output (e.g. a completion produced by an RL trainee)."""

    def __init__(self, output: Any, *, label: str | None = None, id: str | None = None):
        super().__init__(label=label, id=id)
        self.output = output

    def describe(self) -> dict[str, Any]:
        from so_arena.core.mechanism import describe_config

        return {**super().describe(), "output": describe_config(self.output)}

    async def act(self, request, ctx):
        return coerce_action(request, self.output)


class ExternalPolicy(Policy):
    """A policy whose actions are supplied from outside (used to expose a mechanism as a step-wise env).

    When the game asks this policy to act, the request is put on ``requests`` and the policy waits
    for :meth:`respond` to be called with the action.
    """

    def __init__(self, *, label: str | None = None, id: str | None = None):
        super().__init__(label=label, id=id)
        self.requests: asyncio.Queue = asyncio.Queue()

    async def act(self, request, ctx):
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        await self.requests.put((request, ctx, fut))
        out = await fut
        return coerce_action(request, out)


class BestOfNPolicy(Policy):
    """Samples ``n`` candidate actions from ``base`` and returns the one ``scorer`` rates highest.

    ``scorer(request, action, ctx) -> float`` may be async (e.g. a preview judge or reward model).
    This *runs* an optimized agent; for the analysis of optimization pressure from sampled pools use
    :mod:`so_arena.analysis.optimization`, which computes Bo(n) exactly from a single pool.
    """

    def __init__(self, base: Policy, n: int, scorer: Callable[..., Any], *, label: str | None = None, id: str | None = None):
        super().__init__(label=label or f"bo{n}({base.label or base.id})", id=id)
        self.base = base
        self.n = n
        self.scorer = scorer

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "base": self.base.describe(), "n": self.n, "scorer": describe_callable(self.scorer)}

    async def act(self, request, ctx):
        from so_arena.core.state import WorkspaceSlot, using_workspace

        # Candidates are seeded from the decision (independent draws across items and episodes). With state
        # access each works on its own copy of the decision's starting state, and the chosen candidate's
        # copy becomes the decision's - as for the sampled candidates of a game tree. Each candidate's tool
        # calls are recorded separately, and only the chosen one's join the decision's trusted record.
        slot = ctx._slot
        fork = slot is not None and slot.ws is None and bool(slot.parent)
        outer = _TOOL_CALLS.get()
        cands, slots, records = [], [], []
        try:
            for i in range(self.n):
                sub_slot = WorkspaceSlot(slot.store, slot.parent, slot.access) if fork else slot
                slots.append(sub_slot)
                sub = ActContext(role=ctx.role, sample_index=ctx.sample_index * self.n + i, game=ctx.game,
                                 tools=ctx.tools, seed=stable_hash(ctx.seed, i), workspace=sub_slot)
                records.append([])
                with using_workspace(sub_slot) if fork else contextlib.nullcontext(), recording_tool_calls(records[i]):
                    a = await self.base.act(request, sub)
                    s = self.scorer(request, a, sub)
                    if asyncio.iscoroutine(s):
                        s = await s
                ctx.usage = ctx.usage + sub.usage
                cands.append((float(s), i, a))
            best = max(cands, key=lambda t: (t[0], -t[1]))
        except BaseException:
            for s_ in slots if fork else ():
                s_.close(keep=False)
            raise
        if fork:
            for j, s_ in enumerate(slots):
                if j != best[1]:
                    s_.close(keep=False)
            slot.close(keep=False)
            slot.ws = slots[best[1]].ws
        if outer is not None:
            outer.extend(records[best[1]])
        best[2].metadata["bon_scores"] = [c[0] for c in cands]
        return best[2]


class MixturePolicy(Policy):
    """A mixed strategy: draws one component policy per episode (a meta-strategy in EGTA/PSRO).

    The draw depends only on the run seed, the item, the repeat and the role - not on the episode id - so
    every path of a game tree plays one component, and every candidate strategy evaluated against the
    mixture on an item (e.g. in a PSRO best-response search) faces the same draw: comparisons are paired.
    """

    def __init__(self, policies: Sequence[Policy], weights: Sequence[float] | None = None, *, label: str | None = None, id: str | None = None):
        super().__init__(label=label or "mixture", id=id)
        self.policies = list(policies)
        w = list(weights) if weights is not None else [1.0] * len(self.policies)
        s = sum(w)
        self.weights = [x / s for x in w]

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "components": [p.describe() for p in self.policies], "weights": self.weights}

    def pick(self, key: Any) -> Policy:
        r = (stable_hash(key) % 10**9) / 10**9
        acc = 0.0
        for p, w in zip(self.policies, self.weights):
            acc += w
            if r < acc:
                return p
        return self.policies[-1]

    async def act(self, request, ctx):
        g = ctx.game
        key = (g.seed, g.item.id, g.repeat, ctx.role) if g is not None else (ctx.seed, ctx.role)
        chosen = self.pick(key)
        a = await chosen.act(request, ctx)
        a.metadata["mixture_component"] = chosen.id
        return a


def entropy(p: dict[str, float]) -> float:
    return -sum(v * math.log(v) for v in p.values() if v > 0)


PolicyFactory = Callable[[str], Policy] | Callable[[str], Awaitable[Policy]]
