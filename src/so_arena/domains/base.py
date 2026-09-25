"""Domains: sources of task items plus the verifiers, tools and ground-truth scorers that go with them.

Mechanisms are domain-general; a domain adapts them to a setting by supplying items (with public
question text, private information keyed by affordance, and experimenter-side ground truth),
verifiers for checkable claims, tools that create capability gaps, and ground-truth scorers.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, ClassVar

from so_arena.core.game import RunContext
from so_arena.core.ground_truth import GroundTruthScorer, default_scorers
from so_arena.core.items import TaskItem
from so_arena.core.state import Environment, StateStore
from so_arena.core.tools import Tool
from so_arena.core.verification import Verifier


class Domain(abc.ABC):
    name: ClassVar[str] = "domain"
    description: ClassVar[str] = ""
    # which private keys experts are typically granted (used by experiment builders as defaults)
    expert_affordances: ClassVar[list[str]] = []
    expert_tools: ClassVar[list[str]] = []

    @abc.abstractmethod
    def load(self, *, split: str = "test", limit: int | None = None, seed: int = 0) -> list[TaskItem]: ...

    def verifiers(self) -> dict[str, Verifier]:
        return {}

    def tools(self) -> dict[str, Tool]:
        return {}

    def ground_truth_scorers(self) -> list[GroundTruthScorer]:
        return default_scorers()

    def environment(self) -> Environment | None:
        """Stateful domains: the environment that builds each item's starting state and acts on it."""
        return None

    def context(self, *, run_id: str = "run", seed: int = 0, resources: dict[str, Any] | None = None,
                states: StateStore | str | None = None) -> RunContext:
        """The run context: verifier and tool registries, resources and - for stateful domains - the
        environment and the snapshot store (``states``: a store or directory; default: see StateStore)."""
        return RunContext(run_id=run_id, seed=seed, verifiers=self.verifiers(), tools=self.tools(),
                          resources=resources, environment=self.environment(), states=states)

    def behaviours(self) -> dict[str, str]:
        """Strategy prompts for commonly used behaviour labels in this domain (optional)."""
        return {}


_DOMAINS: dict[str, Callable[..., Domain]] = {}


def register_domain(name: str) -> Callable[[type[Domain]], type[Domain]]:
    def deco(cls: type[Domain]) -> type[Domain]:
        _DOMAINS[name] = cls
        return cls

    return deco


def get_domain(name: str, **kwargs: Any) -> Domain:
    if name not in _DOMAINS:
        import so_arena.domains  # noqa: F401  (populate registry)
    if name not in _DOMAINS:
        raise KeyError(f"unknown domain {name!r}; available: {sorted(_DOMAINS)}")
    return _DOMAINS[name](**kwargs)


def list_domains() -> list[str]:
    import so_arena.domains  # noqa: F401

    return sorted(_DOMAINS)
