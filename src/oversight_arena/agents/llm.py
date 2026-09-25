"""LLM agents: prompt rendering, tool-use loop, structured-response elicitation."""

from __future__ import annotations

import math
from typing import Any

from ..core.types import ChatMessage, ToolTrace, Usage
from ..core.util import normalize, word_count
from ..models import GenConfig, Model, get_model
from .base import Action, Agent, Observation, ResponseSpec
from .parsing import parse_choice, parse_distribution, parse_json, parse_scalar, split_thinking


def format_instructions(spec: ResponseSpec, scratchpad: bool = False) -> str:
    parts: list[str] = []
    if scratchpad:
        parts.append(
            "You may think privately inside <thinking>...</thinking> tags first; only text outside "
            "those tags is shown to others."
        )
    opts = ", ".join(spec.options or [])
    if spec.kind == "text":
        if spec.max_words:
            parts.append(f"Keep your message under {spec.max_words} words.")
    elif spec.kind == "choice":
        parts.append(f"End your response with a final line of the form `ANSWER: X`, where X is one of {opts}.")
    elif spec.kind == "distribution":
        ex = ", ".join(f'"{o}": ...' for o in (spec.options or []))
        parts.append(
            "End your response with a JSON object giving your probability that each option is "
            f"correct, e.g. {{{ex}}}. The probabilities must sum to 1."
        )
    elif spec.kind == "scalar":
        parts.append(
            f"End your response with a final line of the form `{spec.scalar_name.upper()}: <number>`, "
            f"with a number between {spec.lo:g} and {spec.hi:g}."
        )
    elif spec.kind == "json":
        fields = "; ".join(f'"{k}": {v}' for k, v in (spec.fields or {}).items())
        parts.append(f"End your response with a JSON object with these fields: {fields}.")
    if spec.kind != "text" and not spec.reasoning:
        parts.append("Do not include any reasoning; give only the requested final answer.")
    return "\n".join(parts)


def render_observation(obs: Observation, persona: str | None = None, scratchpad: bool = False) -> list[ChatMessage]:
    """Default rendering of an observation into chat messages."""
    sys_parts = [p for p in [persona, obs.brief] if p]
    if obs.strategy:
        sys_parts.append("## Your private instructions\n" + obs.strategy)
    user_parts = []
    if obs.task_text:
        user_parts.append(obs.task_text)
    if obs.transcript_text:
        user_parts.append("## Transcript so far\n" + obs.transcript_text)
    if obs.claim_help:
        user_parts.append(obs.claim_help)
    if obs.prompt:
        user_parts.append("## Your turn\n" + obs.prompt)
    fi = format_instructions(obs.response, scratchpad=scratchpad)
    if fi:
        user_parts.append(fi)
    msgs = []
    if sys_parts:
        msgs.append(ChatMessage.system("\n\n".join(sys_parts)))
    msgs.append(ChatMessage.user("\n\n".join(user_parts)))
    return msgs


def _letter_probs_from_logprobs(logprobs: Any, options: list[str]) -> dict[str, float] | None:
    """Probability of each option letter from the first token position that mentions one.

    Uses the top-k alternatives (deduplicated by token string, so the sampled token — which
    also appears in the top-k list — is not double counted) and renormalises over options.
    """
    if not logprobs:
        return None
    for tok in logprobs:
        cands: dict[str, float] = {}
        for t, lp in list(tok.top) + [(tok.token, tok.logprob)]:
            cands.setdefault(t, lp)
        mass: dict[str, float] = {}
        for t, lp in cands.items():
            key = t.strip().strip("()*[]:.").upper()
            for o in options:
                if key == o.upper():
                    mass[o] = mass.get(o, 0.0) + math.exp(lp)
        if mass:
            return normalize({o: mass.get(o, 0.0) for o in options}, eps=1e-6)
    return None


class LLMAgent(Agent):
    """An agent backed by a :class:`~oversight_arena.models.Model`.

    Args:
        model: model spec (e.g. ``"openai/gpt-4o-mini"``) or a Model instance.
        persona: optional base system prompt prepended to the mechanism brief.
        temperature: default sampling temperature (strategies may override via ``params``).
        scratchpad: let the agent reason privately in <thinking> tags (stored as reasoning,
            visible only to roles the mechanism grants, e.g. a CoT monitor).
        max_tool_rounds: maximum tool-call rounds per turn.
        retries: re-asks when a structured answer cannot be parsed.
        hard_word_limit: if set, truncate public text to ``max_words * hard_word_limit``.
    """

    def __init__(
        self,
        model: str | Model,
        persona: str | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
        scratchpad: bool = False,
        max_tool_rounds: int = 6,
        retries: int = 2,
        hard_word_limit: float | None = None,
        reasoning_effort: str | None = None,
        elicitation: str | None = None,
        n_samples: int | None = None,
        id: str | None = None,
    ):
        self.model = get_model(model) if isinstance(model, str) else model
        self.elicitation = elicitation  # overrides the mechanism's elicitation for distributions
        self.n_samples = n_samples
        self.persona = persona
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.scratchpad = scratchpad
        self.max_tool_rounds = max_tool_rounds
        self.retries = retries
        self.hard_word_limit = hard_word_limit
        self.reasoning_effort = reasoning_effort
        self.id = id or self.model.name

    def describe(self) -> dict[str, Any]:
        return {
            "type": "LLMAgent",
            "model": self.model.describe(),
            "persona": self.persona,
            "temperature": self.temperature,
            "scratchpad": self.scratchpad,
            "max_tokens": self.max_tokens,
            "elicitation": self.elicitation,
        }

    def _config(self, obs: Observation, **over: Any) -> GenConfig:
        p = obs.params
        return GenConfig(
            temperature=p.get("temperature", self.temperature),
            max_tokens=p.get("max_tokens", self.max_tokens),
            reasoning_effort=p.get("reasoning_effort", self.reasoning_effort),
        ).merged(**over)

    async def _generate_with_tools(
        self, obs: Observation, messages: list[ChatMessage], config: GenConfig, sample: int
    ) -> tuple[str, str | None, list[ToolTrace], Usage, list[ChatMessage]]:
        tools = {t.name: t for t in obs.tools}
        specs = [t.spec() for t in obs.tools] or None
        usage = Usage()
        trace: list[ToolTrace] = []
        reasoning_parts: list[str] = []
        out = None
        for rnd in range(self.max_tool_rounds + 1):
            allow = specs if (specs and rnd < self.max_tool_rounds) else None
            out = await self.model.generate(messages, config, allow, sample)
            usage = usage + out.usage
            if out.reasoning:
                reasoning_parts.append(out.reasoning)
            if out.tool_calls and allow:
                messages = messages + [ChatMessage.assistant(out.text, out.tool_calls)]
                for tc in out.tool_calls:
                    t = tools.get(tc.name)
                    if t is None:
                        res, err = f"Error: unknown tool {tc.name}", "unknown tool"
                    else:
                        try:
                            res, err = await t.run(tc.arguments), None
                        except Exception as e:  # tool errors are shown to the model
                            res, err = f"Error: {type(e).__name__}: {e}", str(e)
                    trace.append(ToolTrace(name=tc.name, arguments=tc.arguments, result=res, error=err))
                    messages = messages + [ChatMessage.tool(res, tc.id, tc.name)]
                continue
            break
        assert out is not None
        messages = messages + [ChatMessage.assistant(out.text)]
        return out.text, ("\n\n".join(reasoning_parts) or None), trace, usage, messages

    async def act(self, obs: Observation) -> Action:
        spec = obs.response
        if spec.kind == "distribution" and (self.elicitation or self.n_samples):
            spec = spec.model_copy(
                update={
                    "elicitation": self.elicitation or spec.elicitation,
                    "n_samples": self.n_samples or spec.n_samples,
                }
            )
            obs = obs.model_copy(update={"response": spec})
        scratch = bool(obs.params.get("scratchpad", self.scratchpad))
        if spec.kind == "distribution" and spec.elicitation == "logprobs":
            return await self._act_logprobs(obs, scratch)
        if spec.kind == "distribution" and spec.elicitation == "sample":
            return await self._act_sample(obs, scratch)
        messages = render_observation(obs, self.persona, scratch)
        text, reasoning, trace, usage, messages = await self._generate_with_tools(
            obs, messages, self._config(obs), obs.seed
        )
        public, private = split_thinking(text) if scratch or "<think" in text else (text, None)
        reasoning = "\n\n".join(x for x in [reasoning, private] if x) or None
        parsed, err = self._parse(public, spec)
        tries = 0
        while err and tries < self.retries:
            tries += 1
            messages = messages + [
                ChatMessage.user(
                    "I could not parse the final answer from your response. "
                    + format_instructions(spec)
                    + " Reply again with the corrected final answer."
                )
            ]
            out = await self.model.generate(messages, self._config(obs, temperature=0.0), None, obs.seed)
            usage = usage + out.usage
            messages = messages + [ChatMessage.assistant(out.text)]
            p2, e2 = self._parse(split_thinking(out.text)[0], spec)
            if not e2:
                parsed, err = p2, None
        if self.hard_word_limit and spec.max_words:
            lim = int(spec.max_words * self.hard_word_limit)
            if word_count(public) > lim:
                public = " ".join(public.split()[:lim]) + " [truncated]"
        return Action(
            text=public, parsed=parsed, reasoning=reasoning, tool_trace=trace, usage=usage,
            messages=messages, error=err,
        )

    def _parse(self, text: str, spec: ResponseSpec) -> tuple[dict[str, Any], str | None]:
        if spec.kind == "text":
            return {}, None
        if spec.kind == "choice":
            c = parse_choice(text, spec.options or [], spec.option_texts)
            return ({"choice": c}, None) if c else ({}, "unparseable choice")
        if spec.kind == "distribution":
            d = parse_distribution(text, spec.options or [])
            if d is None:
                c = parse_choice(text, spec.options or [], spec.option_texts)
                if c is None:
                    return {"probs": {o: 1 / len(spec.options or [1]) for o in spec.options or []}}, "unparseable distribution"
                d = {o: (0.9 if o == c else 0.1 / max(1, len(spec.options or []) - 1)) for o in spec.options or []}
                return {"probs": d, "choice": c, "probs_inferred": True}, None
            return {"probs": d, "choice": max(d, key=d.get)}, None  # type: ignore[arg-type]
        if spec.kind == "scalar":
            v = parse_scalar(text, spec.lo, spec.hi, spec.scalar_name)
            return ({spec.scalar_name: v}, None) if v is not None else ({}, "unparseable scalar")
        if spec.kind == "json":
            obj = parse_json(text)
            return (obj, None) if obj is not None else ({}, "unparseable json")
        return {}, None

    async def _act_logprobs(self, obs: Observation, scratch: bool) -> Action:
        spec = obs.response
        opts = spec.options or []
        usage = Usage()
        reasoning = None
        messages = render_observation(
            obs.model_copy(update={"response": ResponseSpec(kind="text")}), self.persona, scratch
        )
        if spec.reasoning:
            messages[-1] = ChatMessage.user(
                messages[-1].content + "\nThink it through briefly, but do not state your final answer yet."
            )
            out = await self.model.generate(messages, self._config(obs), None, obs.seed)
            usage = usage + out.usage
            reasoning = out.text
            messages = messages + [ChatMessage.assistant(out.text)]
            messages.append(ChatMessage.user(f"Now answer with only the letter of the correct option ({', '.join(opts)})."))
        else:
            messages[-1] = ChatMessage.user(
                messages[-1].content + f"\nAnswer with only the letter of the correct option ({', '.join(opts)})."
            )
        out = await self.model.generate(
            messages, self._config(obs, temperature=0.0, max_tokens=5).merged(logprobs=True, top_logprobs=20), None, obs.seed
        )
        usage = usage + out.usage
        probs = _letter_probs_from_logprobs(out.logprobs, opts)
        err = None
        if probs is None:  # provider lacks logprobs: fall back to the stated letter
            c = parse_choice(out.text, opts)
            if c is None:
                probs, err = {o: 1 / len(opts) for o in opts}, "no logprobs and unparseable"
            else:
                probs = {o: (0.99 if o == c else 0.01 / max(1, len(opts) - 1)) for o in opts}
        return Action(
            text=out.text, parsed={"probs": probs, "choice": max(probs, key=probs.get)},  # type: ignore[arg-type]
            reasoning=reasoning, usage=usage, messages=messages, error=err,
        )

    async def _act_sample(self, obs: Observation, scratch: bool) -> Action:
        spec = obs.response
        opts = spec.options or []
        counts = {o: 0.0 for o in opts}
        usage = Usage()
        texts = []
        sub = obs.model_copy(update={"response": ResponseSpec(kind="choice", options=opts, option_texts=spec.option_texts, reasoning=spec.reasoning)})
        for k in range(spec.n_samples):
            messages = render_observation(sub, self.persona, scratch)
            out = await self.model.generate(
                messages, self._config(obs, temperature=obs.params.get("temperature", 1.0) or 1.0), None, obs.seed * 1000 + k
            )
            usage = usage + out.usage
            c = parse_choice(split_thinking(out.text)[0], opts, spec.option_texts)
            if c:
                counts[c] += 1
            texts.append(out.text)
        alpha = 0.5
        probs = normalize({o: counts[o] + alpha for o in opts})
        return Action(
            text=texts[0] if texts else "", parsed={"probs": probs, "choice": max(probs, key=probs.get), "counts": counts},  # type: ignore[arg-type]
            usage=usage,
        )


def LLMJudge(model: str | Model, elicitation: str | None = None, temperature: float | None = 0.0, **kw: Any) -> LLMAgent:
    """Convenience constructor for judge agents (deterministic by default).

    ``elicitation`` ("verbal" | "logprobs" | "sample") overrides how probability
    distributions are elicited from this judge.
    """
    return LLMAgent(model, temperature=temperature, elicitation=elicitation, **kw)
