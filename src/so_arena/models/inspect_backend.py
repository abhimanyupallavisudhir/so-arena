"""Inspect AI backend: gives access to every provider Inspect supports.

Model names are Inspect model strings, e.g. ``openai/gpt-4o-mini``, ``anthropic/claude-haiku-4-5``,
``google/gemini-2.5-flash``, ``openrouter/qwen/qwen3-32b``, ``vllm/<hf-model>``, ``ollama/<model>``,
``hf/<model>``, ``mockllm/model``. Credentials come from the usual environment variables.
"""

from __future__ import annotations

from typing import Any

from so_arena.core.types import Completion, GenerateOptions, TokenLogprob, TopLogprob, Usage
from so_arena.models.base import Model
from so_arena.models.registry import get_spec, supports_logprobs


class InspectModel(Model):
    def __init__(self, name: str, *, max_connections: int | None = None, **model_args: Any):
        self.name = name
        self._model_args = model_args
        self._max_connections = max_connections
        self._model = None
        self.spec = get_spec(name)
        self.supports_logprobs = supports_logprobs(name)

    def _get(self):
        if self._model is None:
            from inspect_ai.model import get_model

            self._model = get_model(self.name, **self._model_args)
        return self._model

    async def generate(self, messages, options=None, *, sample_index=0):
        from inspect_ai.model import (
            ChatMessageAssistant,
            ChatMessageSystem,
            ChatMessageUser,
            ContentReasoning,
            GenerateConfig,
        )

        options = options or GenerateOptions()
        converted = []
        for m in messages:
            if m.role == "system":
                converted.append(ChatMessageSystem(content=m.content))
            elif m.role == "assistant":
                converted.append(ChatMessageAssistant(content=m.content))
            else:
                converted.append(ChatMessageUser(content=m.content))
        from so_arena.models.base import draw_seed

        seed = None
        if options.seed is not None:
            seed = options.seed + sample_index + draw_seed(options)
        config = GenerateConfig(
            temperature=options.temperature,
            max_tokens=options.max_tokens,
            top_p=options.top_p,
            seed=seed,
            logprobs=True if options.logprobs else None,
            top_logprobs=options.top_logprobs if options.logprobs else None,
            stop_seqs=list(options.stop) if options.stop else None,
            reasoning_effort=options.reasoning_effort,  # type: ignore[arg-type]
            max_connections=self._max_connections,
        )
        out = await self._get().generate(converted, config=config, cache=False)

        reasoning = None
        logprobs = None
        if out.choices:
            msg = out.choices[0].message
            if isinstance(msg.content, list):
                parts = [c.reasoning for c in msg.content if isinstance(c, ContentReasoning)]
                reasoning = "\n".join(p for p in parts if p) or None
            lp = out.choices[0].logprobs
            if lp is not None and lp.content:
                logprobs = [
                    TokenLogprob(
                        token=t.token,
                        logprob=t.logprob,
                        top=[TopLogprob(token=x.token, logprob=x.logprob) for x in (t.top_logprobs or [])],
                    )
                    for t in lp.content
                ]
        u = out.usage
        usage = Usage(calls=1)
        if u is not None:
            # Inspect's input_tokens exclude cache reads and writes, and its output_tokens include reasoning
            # (see ModelSpec.cost); cache writes are fresh input, billed at least at the input price
            usage = Usage(
                input_tokens=(u.input_tokens or 0) + (u.input_tokens_cache_write or 0),
                output_tokens=u.output_tokens or 0,
                cached_input_tokens=u.input_tokens_cache_read or 0,
                reasoning_tokens=u.reasoning_tokens or 0,
                calls=1,
            )
            cost = u.total_cost
            if cost is None and self.spec is not None:
                cost = self.spec.cost(usage)
            usage = usage.model_copy(update={"cost_usd": cost or 0.0})
        return Completion(
            text=out.completion,
            model=self.name,
            reasoning=reasoning,
            logprobs=logprobs,
            usage=usage,
            metadata={"error": out.error} if out.error else {},
        )
