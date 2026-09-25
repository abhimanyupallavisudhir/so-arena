"""Episode records: everything that happened in one run of a mechanism on one task."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .roles import RoleSpec
from .strategy import Profile, Stance
from .transcript import Transcript
from .types import Usage


class BoundInfo(BaseModel):
    """Public-to-analysis summary of how a role was played in an episode."""

    agent: str = ""
    strategy_id: str = ""
    strategy_name: str = ""
    stance: Stance = Stance.FREE
    target: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    step_seeds: dict[str, int] = Field(default_factory=dict)

    def seed_for(self, step: str | None) -> int:
        return self.step_seeds.get(step, self.seed) if step is not None else self.seed


class ChannelUse(BaseModel):
    """Use of a declared ground-truth channel (audit, simulated probe, resolution...)."""

    channel: str
    role: str | None = None
    cost: float = 0.0
    result: Any = None


class EpisodeRecord(BaseModel):
    id: str
    key: str  # deterministic cache key (mechanism, task, profile, agents, seed)
    experiment: str | None = None
    mechanism: str
    mechanism_config: dict[str, Any] = Field(default_factory=dict)
    mechanism_hash: str = ""
    reward_rule: str = ""
    task_id: str
    domain: str
    profile: Profile = Field(default_factory=Profile)
    bound: dict[str, BoundInfo] = Field(default_factory=dict)
    roles: list[RoleSpec] = Field(default_factory=list)
    transcript: Transcript = Field(default_factory=Transcript)
    outcome: dict[str, Any] = Field(default_factory=dict)
    rewards: dict[str, float] = Field(default_factory=dict)
    gt: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    gt_status: Literal["none", "complete", "partial", "pending"] = "none"
    channels: list[ChannelUse] = Field(default_factory=list)
    usage: dict[str, Usage] = Field(default_factory=dict)
    env_state: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    seed: int = 0
    created_at: str = ""
    elapsed_s: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def trainable_roles(self) -> list[str]:
        return [r.name for r in self.roles if r.trainable]

    def role_spec(self, name: str) -> RoleSpec | None:
        for r in self.roles:
            if r.name == name:
                return r
        return None

    def total_usage(self) -> Usage:
        u = Usage()
        for v in self.usage.values():
            u = u + v
        return u

    def gt_value(self, scorer: str, role: str) -> float | None:
        return self.gt.get(scorer, {}).get(role)
