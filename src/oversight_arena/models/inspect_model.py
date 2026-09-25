"""Inspect AI backend: any provider Inspect supports (OpenAI, Anthropic, Google, OpenRouter,
Together, vLLM, Ollama, HF, mockllm, ...). Works outside of an Inspect eval."""

from __future__ import annotations

from typing import Any

from ..core.types import ChatMessage, TokenLogprob, ToolCall, Usage
from .base import GenConfig, Model, ModelOutput, ToolSpec


def _to_inspect_messages(messages: list[ChatMessage]) -> list[Any]:
    from inspect_ai.model import (
        ChatMessageAssistant,
        ChatMessageSystem,
        ChatMessageTool,
        ChatMessageUser,
    )
    from inspect_ai.tool import ToolCall as ITC

    out: list[Any] = []
    for m in messages:
        if m.role == "system":
            out.append(ChatMessageSystem(content=m.content))
        elif m.role == "user":
            out.append(ChatMessageUser(content=m.content))
        elif m.role == "assistant":
            tcs = (
                [ITC(id=tc.id, function=tc.name, arguments=tc.arguments) for tc in m.tool_calls]
                if m.tool_calls
                else None
            )
            out.append(ChatMessageAssistant(content=m.content, tool_calls=tcs))
        elif m.role == "tool":
            out.append(
                ChatMessageTool(content=m.content, tool_call_id=m.tool_call_id, function=m.name)
            )
    return out


def _to_tool_info(t: ToolSpec) -> Any:
    from inspect_ai.tool import ToolInfo, ToolParams

    params = dict(t.parameters or {})
    return ToolInfo(
        name=t.name,
        description=t.description,
        parameters=ToolParams(
            type="object",
            properties=params.get("properties", {}),
            required=params.get("required", []),
        ),
    )


def _extract_text_and_reasoning(message: Any) -> tuple[str, str | None]:
    from inspect_ai.model import ContentReasoning, ContentText

    content = message.content
    if isinstance(content, str):
        return content, None
    texts, reasons = [], []
    for c in content:
        if isinstance(c, ContentText):
            texts.append(c.text)
        elif isinstance(c, ContentReasoning):
            if not getattr(c, "redacted", False):
                reasons.append(c.reasoning or (c.summary or ""))
    return "\n".join(texts), ("\n".join(reasons) if reasons else None)


class InspectModel(Model):
    """Model backed by ``inspect_ai.model.get_model``."""

    def __init__(self, spec: str, **model_args: Any):
        self.spec = spec
        self.name = spec
        self.model_args = model_args
        self._model = None

    def _get(self) -> Any:
        if self._model is None:
            from inspect_ai.model import get_model

            self._model = get_model(self.spec, **self.model_args)
        return self._model

    async def generate(self, messages, config=None, tools=None, sample=0) -> ModelOutput:
        from inspect_ai.model import GenerateConfig

        cfg = config or GenConfig()
        icfg = GenerateConfig(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            stop_seqs=cfg.stop,
            logprobs=cfg.logprobs or None,
            top_logprobs=cfg.top_logprobs,
            reasoning_effort=cfg.reasoning_effort,  # type: ignore[arg-type]
            reasoning_tokens=cfg.reasoning_tokens,
            seed=cfg.seed,
            **cfg.extra,
        )
        model = self._get()
        out = await model.generate(
            _to_inspect_messages(messages),
            tools=[_to_tool_info(t) for t in (tools or [])],
            config=icfg,
            cache=False,
        )
        choice = out.choices[0] if out.choices else None
        text, reasoning = ("", None)
        tool_calls: list[ToolCall] = []
        logprobs = None
        if choice is not None:
            text, reasoning = _extract_text_and_reasoning(choice.message)
            for tc in choice.message.tool_calls or []:
                tool_calls.append(ToolCall(id=tc.id, name=tc.function, arguments=dict(tc.arguments or {})))
            if choice.logprobs is not None:
                logprobs = [
                    TokenLogprob(
                        token=lp.token,
                        logprob=lp.logprob,
                        top=[(t.token, t.logprob) for t in (lp.top_logprobs or [])],
                    )
                    for lp in choice.logprobs.content
                ]
        u = out.usage
        usage = Usage(
            calls=1,
            input_tokens=(u.input_tokens if u else 0) or 0,
            output_tokens=(u.output_tokens if u else 0) or 0,
            reasoning_tokens=(u.reasoning_tokens if u else 0) or 0,
            cost_usd=float((u.total_cost if u else 0) or 0.0),
        )
        return ModelOutput(
            text=text,
            tool_calls=tool_calls,
            reasoning=reasoning,
            logprobs=logprobs,
            usage=usage,
            model=out.model or self.spec,
            stop_reason=str(choice.stop_reason) if choice else None,
        )
