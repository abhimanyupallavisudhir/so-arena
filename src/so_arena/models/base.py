"""Model protocol and simple in-process models.

Everything that produces text in the library goes through :class:`Model`. Backends:

* :class:`~so_arena.models.inspect_backend.InspectModel` - any provider Inspect supports (default; model
  names resolve to a :class:`ProviderModel`, which applies the simulate / cache settings at call time).
* :class:`MockModel` / :class:`FunctionModel` - deterministic scripted models for tests and fixtures.
* :class:`~so_arena.models.simulated.SimulatedModel` - dry runs for cost estimation.
* :class:`~so_arena.models.human.HumanModel` - a human at the terminal (human judges).

``sample_index`` identifies independent samples of the same prompt. It is part of the cache key, so
best-of-N pools are reproducible yet contain N distinct draws.
"""

from __future__ import annotations

import abc
import asyncio
import itertools
from collections.abc import Callable, Sequence
from typing import Any

from so_arena.core.types import Completion, GenerateOptions, Message, TokenLogprob, TopLogprob, Usage


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token for English)."""
    return max(1, (len(text) + 3) // 4)


def messages_tokens(messages: Sequence[Message]) -> int:
    return sum(estimate_tokens(m.content) + 4 for m in messages)


class Model(abc.ABC):
    """Abstract text model."""

    name: str = "model"
    supports_logprobs: bool = False

    @abc.abstractmethod
    async def generate(
        self,
        messages: Sequence[Message],
        options: GenerateOptions | None = None,
        *,
        sample_index: int = 0,
    ) -> Completion: ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


Responder = Callable[[Sequence[Message], GenerateOptions, int], "str | Completion"]


class FunctionModel(Model):
    """A model defined by a Python function ``fn(messages, options, sample_index) -> str | Completion``.

    Useful for fixtures with fixed behaviour (e.g. a rule-based judge) and for tests. The function
    may be sync or async.
    """

    def __init__(self, fn: Callable[..., Any], name: str = "function", supports_logprobs: bool = False):
        self.fn = fn
        self.name = name
        self.supports_logprobs = supports_logprobs

    async def generate(self, messages, options=None, *, sample_index=0):
        options = options or GenerateOptions()
        out = self.fn(list(messages), options, sample_index)
        if asyncio.iscoroutine(out):
            out = await out
        if isinstance(out, Completion):
            return out
        text = str(out)
        return Completion(
            text=text,
            model=self.name,
            usage=Usage(
                input_tokens=messages_tokens(messages),
                output_tokens=estimate_tokens(text),
                calls=1,
            ),
        )


class MockModel(FunctionModel):
    """Returns scripted responses in order (cycling), or a constant default.

    ``MockModel(["first", "second"])`` returns "first" then "second" then "first" ...
    ``MockModel(logprobs={"A": -0.1, "B": -2.3})`` additionally returns a one-token completion with
    the given top-logprobs (for testing logprob-based judges).
    """

    def __init__(
        self,
        responses: Sequence[str] | str | None = None,
        *,
        name: str = "mock",
        logprobs: dict[str, float] | None = None,
    ):
        if responses is None:
            responses = ["(mock response)"]
        if isinstance(responses, str):
            responses = [responses]
        self._cycle = itertools.cycle(list(responses))
        self._logprobs = logprobs

        def fn(messages, options, sample_index):
            text = next(self._cycle)
            lp = None
            if options.logprobs and self._logprobs:
                best = max(self._logprobs, key=self._logprobs.get)
                text = best
                lp = [
                    TokenLogprob(
                        token=best,
                        logprob=self._logprobs[best],
                        top=[TopLogprob(token=t, logprob=v) for t, v in self._logprobs.items()],
                    )
                ]
            return Completion(
                text=text,
                model=name,
                logprobs=lp,
                usage=Usage(
                    input_tokens=messages_tokens(messages), output_tokens=estimate_tokens(text), calls=1
                ),
            )

        super().__init__(fn, name=name, supports_logprobs=logprobs is not None)


class ProviderModel(Model):
    """A provider model resolved by name (what :func:`get_model` returns for e.g. ``"openai/gpt-4o-mini"``).

    It follows the global settings *at call time*: while ``settings.simulate`` is on, calls go to a
    :class:`~so_arena.models.simulated.SimulatedModel` (no API call), otherwise to the Inspect backend,
    through the response cache when ``settings.cache_dir`` is set. So :func:`so_arena.configure` also
    applies to policies and fixtures built before it: a dry run never reaches an API.
    """

    def __init__(self, name: str, **model_args: Any):
        from so_arena.models.registry import get_spec, supports_logprobs

        self.name = name
        self.spec = get_spec(name)
        self.supports_logprobs = supports_logprobs(name)  # the same for the real and the simulated backend
        self._args = model_args
        self._real: Model | None = None
        self._simulated: Model | None = None
        self._cached: dict[str, Model] = {}

    def backend(self) -> Model:
        """The model serving calls under the current settings."""
        from so_arena.config import settings

        if settings.simulate:
            if self._simulated is None:
                import inspect

                from so_arena.models.simulated import SimulatedModel

                accepted = inspect.signature(SimulatedModel).parameters
                self._simulated = SimulatedModel(self.name, **{k: v for k, v in self._args.items() if k in accepted})
            return self._simulated
        if self._real is None:
            from so_arena.models.inspect_backend import InspectModel

            self._real = InspectModel(self.name, **self._args)
        if settings.cache_dir is None:
            return self._real
        key = str(settings.cache_dir)
        if key not in self._cached:
            from so_arena.models.cache import CachedModel, ResponseCache

            self._cached[key] = CachedModel(self._real, ResponseCache.at(settings.cache_dir))
        return self._cached[key]

    async def generate(self, messages, options=None, *, sample_index=0):
        return await self.backend().generate(messages, options, sample_index=sample_index)


# ---------------------------------------------------------------------------------------------
# Resolution of model names
# ---------------------------------------------------------------------------------------------

_MODEL_CACHE: dict[tuple, Model] = {}
_NAMED: dict[str, Model] = {}


def register_model_instance(name: str, model: Model) -> Model:
    """Make an in-process model resolvable by name (e.g. a scripted fixture referenced from a config)."""
    _NAMED[name] = model
    return model


def get_model(model: "str | Model", **model_args: Any) -> Model:
    """Resolve a model name to a :class:`Model`.

    * ``Model`` instances are returned unchanged.
    * ``"mock"`` / ``"mock/<name>"`` -> :class:`MockModel`.
    * ``"sim/<provider/model>"`` -> :class:`SimulatedModel` (cost dry-run).
    * ``"human"`` -> :class:`HumanModel`.
    * ``"llamacpp/<path.gguf>"`` -> in-process llama.cpp model (:mod:`so_arena.models.local`).
    * anything else -> :class:`ProviderModel` over the Inspect backend (e.g. ``"openai/gpt-4o-mini"``,
      ``"anthropic/claude-haiku-4-5"``, ``"openrouter/qwen/qwen3-8b"``, ``"vllm/..."``).

    Provider models follow :func:`so_arena.configure` at call time, also when resolved before it:
    ``simulate`` replaces their calls by simulated ones, ``cache_dir`` routes them through a response cache.
    """
    if isinstance(model, Model):
        return model
    if model in _NAMED:
        return _NAMED[model]
    key = (model, tuple(sorted((k, repr(v)) for k, v in model_args.items())))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    resolved: Model
    if model == "mock" or model.startswith("mock/"):
        resolved = MockModel(name=model, **model_args)
    elif model == "human" or model.startswith("human/"):
        from so_arena.models.human import HumanModel

        resolved = HumanModel(name=model, **model_args)
    elif model.startswith("llamacpp/"):
        from so_arena.models.local import LlamaCppModel

        resolved = LlamaCppModel(model.removeprefix("llamacpp/"), **model_args)
    elif model.startswith("sim/"):
        from so_arena.models.simulated import SimulatedModel

        resolved = SimulatedModel(model.removeprefix("sim/"), **model_args)
    else:
        resolved = ProviderModel(model, **model_args)
    _MODEL_CACHE[key] = resolved
    return resolved
