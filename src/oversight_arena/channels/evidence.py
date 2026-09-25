"""Verified claims.

Agents can embed *checkable claims* in their messages using XML-ish markup, e.g.
``<quote>exact words from the article</quote>`` or ``<sql>SELECT ...</sql>``. When a
mechanism has an :class:`EvidencePolicy`, claims are extracted and checked by trusted
:class:`Verifier` code (subject to per-role budgets and optional verification noise), and
the verdicts are shown to the other roles. Verification capability is thereby a first-class,
systematically variable experimental factor (none / quotes / execution / engine / kernel...).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..core.task import TaskView
from ..core.transcript import Evidence


class Claim(BaseModel):
    kind: str  # the verifier tag, e.g. "quote"
    content: str
    args: dict[str, str] = Field(default_factory=dict)
    raw: str = ""
    span: tuple[int, int] = (0, 0)


class VerifyEnv(BaseModel):
    """What a verifier may use: public view + privileged *resources* (never GT labels)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    view: TaskView
    resources: dict[str, Any] = Field(default_factory=dict)
    role: str | None = None
    env: Any = None  # the episode environment (e.g. sandbox), if any


class Verifier(ABC):
    """Trusted code that checks one kind of claim."""

    name: str = "verifier"
    tag: str = "claim"  # markup tag agents use
    cost: float = 1.0
    help: str = ""  # one-line usage description shown to agents

    @abstractmethod
    async def verify(self, claim: Claim, env: VerifyEnv) -> Evidence: ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "tag": self.tag, "cost": self.cost}


def extract_claims(text: str, tags: list[str]) -> list[Claim]:
    if not tags:
        return []
    alt = "|".join(re.escape(t) for t in tags)
    pat = re.compile(rf"<({alt})((?:\s+[a-zA-Z_]+\s*=\s*\"[^\"]*\")*)\s*>(.*?)</\1\s*>", re.S)
    out = []
    for m in pat.finditer(text):
        args = dict(re.findall(r'([a-zA-Z_]+)\s*=\s*"([^"]*)"', m.group(2) or ""))
        out.append(
            Claim(kind=m.group(1), content=m.group(3).strip(), args=args, raw=m.group(0), span=(m.start(), m.end()))
        )
    return out


def annotate(text: str, claims: list[Claim], statuses: list[str]) -> str:
    """Rewrite claim markup to carry verification status, e.g. <quote status="VERIFIED">."""
    out, last = [], 0
    for c, st in zip(claims, statuses):
        out.append(text[last : c.span[0]])
        attrs = "".join(f' {k}="{v}"' for k, v in c.args.items() if k != "status")
        out.append(f'<{c.kind}{attrs} status="{st}">{c.content}</{c.kind}>')
        last = c.span[1]
    out.append(text[last:])
    return "".join(out)


class EvidencePolicy(BaseModel):
    """How a mechanism handles verifiable claims.

    Attributes:
        verifiers: enabled verifier names (None = all verifiers the domain provides).
        roles: roles whose inline claims are verified (None = every speaker).
        budget: max verification cost per role per episode (None = unlimited).
        auto_verify: verify inline claims when a message is posted.
        show_to: roles that see verification verdicts (None = everyone who sees the message).
        noise: probability that a verdict is flipped (imperfect verification).
        annotate_unchecked: mark claims beyond budget as UNCHECKED (else leave raw markup).
        share_tool_results: show speakers' trusted tool calls/outputs to the other roles.
        requests: roles that may actively request verification via tools (e.g. ["judge"]).
    """

    verifiers: list[str] | None = None
    roles: list[str] | None = None
    budget: float | None = None
    auto_verify: bool = True
    show_to: list[str] | None = None
    noise: float = 0.0
    annotate_unchecked: bool = True
    share_tool_results: bool = False
    requests: list[str] = Field(default_factory=list)

    def applies_to(self, role: str) -> bool:
        return self.roles is None or role in self.roles


def claim_help_text(verifiers: list[Verifier], budget: float | None = None) -> str:
    if not verifiers:
        return ""
    lines = [
        "## Verifiable claims",
        "You can make claims that are checked by trusted code; the verdict (VERIFIED / REFUTED) is "
        "shown to the other participants. Unverifiable assertions carry no such guarantee.",
    ]
    for v in verifiers:
        lines.append(f"- `<{v.tag}>...</{v.tag}>`: {v.help} (cost {v.cost:g})")
    if budget is not None:
        lines.append(f"Your verification budget this episode: {budget:g} cost units; claims beyond it are marked UNCHECKED.")
    return "\n".join(lines)
