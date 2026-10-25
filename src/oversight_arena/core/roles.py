"""Role specifications."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

RoleKind = Literal[
    "expert", "judge", "worker", "monitor", "critic", "client", "overseer", "forecaster", "agent"
]


class RoleSpec(BaseModel):
    """A slot in a mechanism.

    ``trainable`` roles receive rewards from the mechanism (and are the subject of incentive
    analysis); non-trainable roles are *fixtures* with fixed behaviour (e.g. a weak trusted
    judge). ``kind`` lets domains assign default clearances/tools (experts get privileged
    information and tools; judges typically do not).
    """

    name: str
    kind: RoleKind = "agent"
    trainable: bool = True
    title: str | None = None
    description: str = ""

    @property
    def display(self) -> str:
        return self.title or self.name.replace("_", " ").title()
