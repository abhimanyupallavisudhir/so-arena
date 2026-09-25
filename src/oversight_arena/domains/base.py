"""Domains (≈ ControlArena *settings*): task sources plus domain-specific tools, verifiers,
information access, environments and ground truth.

Mechanisms are written against the generic interface below, so any mechanism runs on any
domain; domains adapt themselves to mechanisms (not vice versa).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, PrivateAttr

from ..core.roles import RoleSpec
from ..core.task import Task, TaskView
from ..core.tools import Tool
from ..core.util import rng_for

if TYPE_CHECKING:  # pragma: no cover
    from ..channels.evidence import Verifier
    from ..ground_truth.base import GTScorer

JUDGE_KINDS = {"judge", "client", "monitor", "overseer"}


class Environment:
    """Per-episode environment. The default is static: a fixed tool list gated by clearance.

    Agentic domains subclass this to hold mutable state (files, sandboxes, message boards) and
    to expose ground-truth logs (``bad_roles()``) and a final ``state()`` for GT scoring.
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools = tools or []

    async def setup(self) -> None:  # noqa: B027
        pass

    async def teardown(self) -> None:  # noqa: B027
        pass

    def tools_for(self, role: str, clearance: set[str]) -> list[Tool]:
        return [t for t in self._tools if t.group == "public" or t.group in clearance]

    def state(self) -> dict[str, Any]:
        return {}

    def bad_roles(self) -> set[str] | None:
        return None

    # optional hooks for agentic / multi-agent environments
    def observation(self, role: str) -> dict[str, Any]:
        """Structured, role-private observations of the environment (for programmatic agents)."""
        return {}

    def observation_text(self, role: str) -> str:
        """Role-private observations rendered for LLM agents (appended to the task text)."""
        return ""

    def apply(self, role: str, data: dict[str, Any]) -> None:
        """Apply structured actions returned by programmatic agents (LLMs act via tools)."""


class Domain(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    name: ClassVar[str] = "domain"
    description: ClassVar[str] = ""

    limit: int | None = None
    seed: int = 0
    shuffle: bool = False
    expert_clearance: list[str] = ["private", "tools"]
    judge_clearance: list[str] = []

    _cache: list[Task] | None = PrivateAttr(default=None)

    # ------------------------------------------------------------------ tasks
    @abstractmethod
    def load(self) -> list[Task]:
        """Load all tasks (called once; cached)."""

    def tasks(self) -> list[Task]:
        if self._cache is None:
            self._cache = self.load()
        ts = list(self._cache)
        if self.shuffle:
            rng_for("domain-shuffle", self.name, self.seed).shuffle(ts)
        return ts[: self.limit] if self.limit is not None else ts

    def get_task(self, task_id: str) -> Task:
        for t in self._cache or self.load():
            if t.id == task_id:
                return t
        raise KeyError(task_id)

    # ------------------------------------------------------------------ access / rendering
    def default_clearance(self, role: RoleSpec) -> set[str]:
        return set(self.judge_clearance if role.kind in JUDGE_KINDS else self.expert_clearance)

    def render_task(self, view: TaskView, clearance: set[str], role: RoleSpec | None = None) -> str:
        return default_render(view, clearance)

    # ------------------------------------------------------------------ tools / verification / env
    def tools(self, task: Task) -> list[Tool]:
        return []

    def verifiers(self, task: Task) -> list["Verifier"]:
        return []

    def make_env(self, task: Task) -> Environment:
        return Environment(self.tools(task))

    # ------------------------------------------------------------------ ground truth
    def gt_scorers(self) -> list["GTScorer"]:
        from ..ground_truth.common import DecisionCorrect, JudgeProbCorrect, TargetCorrect

        return [TargetCorrect(), DecisionCorrect(), JudgeProbCorrect()]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, **self.model_dump(serialize_as_any=True)}


def default_render(view: TaskView, clearance: set[str] | None = None) -> str:
    clearance = clearance or set()
    parts = []
    for b in view.visible_info(clearance):
        title = b.title or b.key.replace("_", " ").title()
        tag = "" if b.access == "public" else " (privileged: only some participants can see this)"
        parts.append(f"## {title}{tag}\n{b.content}")
    parts.append(f"## Question\n{view.question}")
    if view.options:
        parts.append("## Options\n" + "\n".join(f"({o.id}) {o.text}" for o in view.options))
    return "\n\n".join(parts)


class TaskListDomain(Domain):
    """A domain built from an explicit list of tasks (handy for tests and custom datasets)."""

    name: ClassVar[str] = "tasklist"
    task_list: list[Task] = []
    tool_list: list[Tool] = []
    verifier_list: list[Any] = []

    def load(self) -> list[Task]:
        return list(self.task_list)

    def tools(self, task: Task) -> list[Tool]:
        return list(self.tool_list)

    def verifiers(self, task: Task) -> list["Verifier"]:
        return list(self.verifier_list)
