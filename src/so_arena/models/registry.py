"""Model registry: capability and cost metadata used for scaling curves and cost estimates.

Scaling experiments need to know *how capable* each model is (parameters, training compute, an
Elo-like rating, a domain rating such as chess Elo) so that metrics can be plotted against the
agent-judge capability gap. Cost estimation needs prices. Both live in :class:`ModelSpec`.

The bundled ``data/models.yaml`` holds a small default set; prices there are indicative and should
be checked. Register your own with :func:`register_model` or :func:`load_registry`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from so_arena.core.types import Usage


class ModelSpec(BaseModel):
    name: str
    family: str | None = None
    organization: str | None = None
    params_b: float | None = None  # total parameters, billions
    active_params_b: float | None = None  # for mixture-of-experts
    training_flops: float | None = None
    release_date: str | None = None
    ratings: dict[str, float] = Field(default_factory=dict)  # e.g. {"arena_elo": 1250, "chess_elo": 1500}
    price_input_per_mtok: float | None = None
    price_output_per_mtok: float | None = None
    price_cached_input_per_mtok: float | None = None
    supports_logprobs: bool | None = None
    reasoning: bool | None = None
    notes: str | None = None

    def cost(self, usage: Usage) -> float | None:
        """Price of ``usage`` in the library's (and Inspect's) convention: ``input_tokens`` are the input
        tokens not read from the provider's prompt cache (those are ``cached_input_tokens``), and
        ``output_tokens`` include reasoning tokens (``reasoning_tokens`` is the part of them spent reasoning)."""
        if self.price_input_per_mtok is None or self.price_output_per_mtok is None:
            return None
        cached_price = (
            self.price_cached_input_per_mtok
            if self.price_cached_input_per_mtok is not None
            else self.price_input_per_mtok
        )
        return (
            usage.input_tokens * self.price_input_per_mtok
            + usage.cached_input_tokens * cached_price
            + usage.output_tokens * self.price_output_per_mtok
        ) / 1e6

    def rating(self, key: str) -> float | None:
        return self.ratings.get(key)


_REGISTRY: dict[str, ModelSpec] = {}
_DEFAULTS_LOADED = False


def _norm(name: str) -> str:
    return name.strip().lower()


def register_model(spec: ModelSpec | None = None, **kwargs: Any) -> ModelSpec:
    """Add (or override) a model's spec; the bundled registry is loaded first, so registering a model never
    hides the built-in ones."""
    _ensure_defaults()
    spec = spec or ModelSpec(**kwargs)
    _REGISTRY[_norm(spec.name)] = spec
    return spec


def load_registry(path: str | Path) -> list[ModelSpec]:
    data = yaml.safe_load(Path(path).read_text())
    specs = [register_model(ModelSpec(**entry)) for entry in data.get("models", [])]
    return specs


def get_spec(name: str) -> ModelSpec | None:
    """Look up a spec by exact name, then by suffix (so 'openai/gpt-4o-mini' matches 'gpt-4o-mini')."""
    _ensure_defaults()
    key = _norm(name)
    if key in _REGISTRY:
        return _REGISTRY[key]
    for prefix in ("sim/", "openrouter/", "openai/", "anthropic/", "google/", "together/"):
        key = key.removeprefix(prefix)
    if key in _REGISTRY:
        return _REGISTRY[key]
    tail = key.split("/")[-1]
    for k, spec in _REGISTRY.items():
        if k.split("/")[-1] == tail:
            return spec
    return None


# providers whose APIs return token logprobs (OpenAI-compatible APIs and local inference servers)
LOGPROB_PROVIDERS = ("openai/", "together/", "vllm/", "hf/", "mockllm/", "openrouter/", "fireworks/")


def supports_logprobs(name: str) -> bool:
    """Whether calls to model ``name`` return token logprobs: the registry's ``supports_logprobs`` where it
    says, else by provider. Real and simulated models both use this, so a dry run makes the calls the real
    run would (a logprob judgment is one call; without logprobs it falls back to votes)."""
    spec = get_spec(name)
    if spec is not None and spec.supports_logprobs is not None:
        return spec.supports_logprobs
    return name.removeprefix("sim/").startswith(LOGPROB_PROVIDERS)


def all_specs() -> list[ModelSpec]:
    _ensure_defaults()
    return list(_REGISTRY.values())


def _ensure_defaults() -> None:
    """Load the bundled registry once (``data/models.yaml``); specs registered by the user take precedence."""
    global _DEFAULTS_LOADED
    if _DEFAULTS_LOADED:
        return
    _DEFAULTS_LOADED = True
    path = Path(__file__).resolve().parent.parent / "data" / "models.yaml"
    if path.exists():
        for entry in yaml.safe_load(path.read_text()).get("models", []):
            spec = ModelSpec(**entry)
            _REGISTRY.setdefault(_norm(spec.name), spec)


def cost_of(model_name: str, usage: Usage) -> float | None:
    spec = get_spec(model_name)
    return spec.cost(usage) if spec else None
