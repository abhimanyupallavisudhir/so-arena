"""In-process local models via llama.cpp (``pip install llama-cpp-python``).

``get_model("llamacpp/path/to/model.gguf")`` loads a GGUF model once per process (and per load settings)
and serves chat completions with logprobs, without a server. Handy for cheap judges, CI runs and
experiments on small open-weight models; for throughput use vLLM through Inspect (``vllm/<hf-model>``).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from so_arena.core.types import Completion, GenerateOptions, TokenLogprob, TopLogprob, Usage
from so_arena.models.base import Model

# (file, load settings) -> (loaded llama.cpp model, the lock serializing calls to it)
_LOADED: dict[tuple[Any, ...], tuple[Any, threading.Lock]] = {}
_LOCK = threading.Lock()


class LlamaCppModel(Model):
    def __init__(self, path: str, *, n_ctx: int = 4096, n_threads: int | None = None, name: str | None = None,
                 logits_all: bool = False, **kwargs: Any):
        """``logits_all=True`` enables logprobs (needed for "tip of the tongue" judges) but allocates
        n_ctx x vocabulary floats; without it, LLMPolicy falls back to verbalized or voted probabilities."""
        self.path = path
        self.name = name or f"llamacpp/{path}"
        self.n_ctx, self.n_threads, self.kwargs = n_ctx, n_threads, kwargs
        self.logits_all = logits_all
        self.supports_logprobs = logits_all

    def _llm(self) -> tuple[Any, threading.Lock]:
        """The model loaded from this file *with these settings* (``logits_all`` and ``n_ctx`` are fixed at
        load: a model loaded without ``logits_all`` cannot return logprobs), and its lock. A llama.cpp context
        is not thread-safe, so all model objects sharing a loaded model share its one lock."""
        key = (self.path, self.n_ctx, self.n_threads, self.logits_all,
               tuple(sorted((k, repr(v)) for k, v in self.kwargs.items())))
        with _LOCK:
            if key not in _LOADED:
                from llama_cpp import Llama

                llm = Llama(model_path=self.path, n_ctx=self.n_ctx, n_threads=self.n_threads,
                            logits_all=self.logits_all, verbose=False, **self.kwargs)
                _LOADED[key] = (llm, threading.Lock())
            return _LOADED[key]

    def _generate(self, messages, options: GenerateOptions, sample_index: int) -> Completion:
        llm, lock = self._llm()
        from so_arena.models.base import draw_seed

        seed = (options.seed if options.seed is not None else 0) + sample_index + draw_seed(options)
        kw: dict[str, Any] = dict(
            messages=[{"role": m.role if m.role != "tool" else "user", "content": m.content} for m in messages],
            temperature=0.0 if options.temperature is None else options.temperature,
            max_tokens=options.max_tokens or 512,
            seed=seed,
        )
        if options.top_p is not None:
            kw["top_p"] = options.top_p
        if options.stop:
            kw["stop"] = list(options.stop)
        if options.logprobs and self.logits_all:
            kw["logprobs"] = True
            kw["top_logprobs"] = options.top_logprobs or 5
        with lock:  # a llama.cpp context is not thread-safe
            out = llm.create_chat_completion(**kw)
        choice = out["choices"][0]
        text = choice["message"].get("content") or ""
        logprobs = None
        lp = choice.get("logprobs")
        if lp and lp.get("content"):
            logprobs = [TokenLogprob(token=t["token"], logprob=t["logprob"],
                                     top=[TopLogprob(token=x["token"], logprob=x["logprob"]) for x in t.get("top_logprobs") or []])
                        for t in lp["content"]]
        u = out.get("usage") or {}
        return Completion(text=text, model=self.name, logprobs=logprobs,
                          usage=Usage(input_tokens=u.get("prompt_tokens", 0), output_tokens=u.get("completion_tokens", 0), calls=1))

    async def generate(self, messages, options=None, *, sample_index=0):
        return await asyncio.to_thread(self._generate, list(messages), options or GenerateOptions(), sample_index)
