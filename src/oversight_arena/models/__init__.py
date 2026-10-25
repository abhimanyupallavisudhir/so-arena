"""Model backends.

``get_model("openai/gpt-4o-mini")`` returns a cached Inspect-backed model (any provider Inspect
supports, including ``mockllm/model`` for smoke tests). Special specs: ``"mock"`` (constant) and
``"random"`` (null baseline); wrap any Python function with :class:`FunctionModel`. For
LLM-free experiments use the programmatic agents in :mod:`oversight_arena.sim`. Set
``OA_CACHE=0`` to disable the persistent cache (``OA_CACHE_DIR`` moves it).
"""

from __future__ import annotations

import os
from typing import Any

from .base import GenConfig, Model, ModelOutput, ToolSpec
from .cache import CachedModel
from .scripted import FunctionModel, MockModel, RandomChoiceModel

_MODELS: dict[str, Model] = {}


def get_model(spec: "str | Model", cache: bool | None = None, **model_args: Any) -> Model:
    if isinstance(spec, Model):
        return spec
    use_cache = (os.environ.get("OA_CACHE", "1") != "0") if cache is None else cache
    key = f"{spec}|{use_cache}|{sorted(model_args.items())}"
    if key in _MODELS:
        return _MODELS[key]
    m: Model
    if spec == "mock":
        m = MockModel()
    elif spec == "random":
        m = RandomChoiceModel()
    else:
        from .inspect_model import InspectModel

        m = InspectModel(spec, **model_args)
    if use_cache and not spec.startswith("mockllm") and spec not in ("mock", "random"):
        m = CachedModel(m)
    _MODELS[key] = m
    return m


__all__ = [
    "GenConfig",
    "Model",
    "ModelOutput",
    "ToolSpec",
    "CachedModel",
    "FunctionModel",
    "MockModel",
    "RandomChoiceModel",
    "get_model",
]
