"""Transcripts: ordered entries with per-role visibility, evidence and (private) reasoning.

Provenance is marked by construction, not by pattern matching: everything the *system* writes
(speaker labels, moderator lines, trusted tool output, verification verdicts) is labelled in
reserved brackets ⟦…⟧, and participant-written text (messages, reasoning, tool arguments and
results, claim text) is passed through :func:`untrusted`, which cannot produce those brackets.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, Field

from .types import ToolTrace, Usage

EntryKind = Literal[
    "message", "evidence", "tool", "system", "verdict", "probe", "audit", "report", "action", "note"
]


OPEN, CLOSE = "⟦", "⟧"  # reserved for text written by the system
_BRACKET_LOOKALIKES = str.maketrans({c: "[" for c in "⟦〚【〖［⟬"} | {c: "]" for c in "⟧〛】〗］⟭"})
# invisible and bidirectional control characters: they hide or reorder text
_INVISIBLE = re.compile("[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180f\u200b-\u200f\u202a-\u202e"
                        "\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\U000e0000-\U000e007f]")


def label(text: str) -> str:
    """A system-written label, e.g. ``⟦Moderator⟧``."""
    return f"{OPEN}{text}{CLOSE}"


def untrusted(text: Any) -> str:
    """Participant-written text as it may appear in a transcript: no reserved brackets (or their
    look-alikes), no invisible or direction-changing characters."""
    s = unicodedata.normalize("NFC", str(text))
    return _INVISIBLE.sub("", s).translate(_BRACKET_LOOKALIKES)


def _indented(text: str, pad: str = "      ") -> str:
    """Continuation lines of a block, indented so they cannot pose as new transcript lines."""
    return text.replace("\n", "\n" + pad)


class Evidence(BaseModel):
    """The result of a verification: a claim checked by trusted code (or a trusted oracle)."""

    verifier: str
    kind: str = ""
    claim: str = ""
    result: str = ""
    verified: bool | None = None  # True confirmed / False refuted / None = informational
    cost: float = 0.0
    requested_by: str | None = None
    visible_to: list[str] | None = None  # None = whoever sees the entry
    data: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        mark = {True: "VERIFIED", False: "REFUTED", None: "CHECKED"}[self.verified]
        claim, result = untrusted(self.claim), untrusted(self.result)  # claims (and echoed data) come from agents
        body = f"{claim} -> {result}" if claim else result
        return f"{label(f'{mark} by {self.verifier}')} {_indented(body)}"


class Entry(BaseModel):
    idx: int = -1
    kind: EntryKind = "message"
    role: str | None = None  # speaker (None = mechanism/moderator)
    content: str = ""  # public (rendered) content
    visible_to: list[str] | None = None  # None = visible to all roles
    reasoning: str | None = None  # private chain-of-thought / scratchpad
    reasoning_visible_to: list[str] | None = None  # default: speaker only
    evidence: list[Evidence] = Field(default_factory=list)
    tool_trace: list[ToolTrace] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)  # parsed / structured payload
    step: str | None = None
    turn: int | None = None
    usage: Usage | None = None

    def visible(self, role: str) -> bool:
        return self.visible_to is None or role in self.visible_to or role == self.role

    def reasoning_visible(self, role: str) -> bool:
        if self.reasoning is None:
            return False
        if role == self.role:
            return True
        return self.reasoning_visible_to is not None and role in self.reasoning_visible_to


class Transcript(BaseModel):
    entries: list[Entry] = Field(default_factory=list)

    def add(self, entry: Entry) -> Entry:
        entry.idx = len(self.entries)
        self.entries.append(entry)
        return entry

    def visible(self, role: str) -> list[Entry]:
        return [e for e in self.entries if e.visible(role)]

    def by_role(self, role: str, kind: str | None = "message") -> list[Entry]:
        return [e for e in self.entries if e.role == role and (kind is None or e.kind == kind)]

    def last(self, role: str | None = None, kind: str | None = None) -> Entry | None:
        for e in reversed(self.entries):
            if (role is None or e.role == role) and (kind is None or e.kind == kind):
                return e
        return None

    def render(
        self,
        for_role: str | None = None,
        titles: dict[str, str] | None = None,
        include_reasoning: bool = True,
        include_tools: bool = True,
    ) -> str:
        """Plain-text rendering of the entries visible to ``for_role`` (all if None). System-written
        parts are labelled ⟦…⟧; participant-written parts cannot contain those brackets."""
        titles = titles or {}
        lines: list[str] = []
        for e in self.entries:
            if for_role is not None and not e.visible(for_role):
                continue
            who = titles.get(e.role or "", e.role or "Moderator") if e.role else "Moderator"
            if e.kind == "system" or e.role is None:
                lines.append(f"{label('Moderator')}: {e.content}")
            elif e.kind == "evidence":
                lines.append(e.content)
            else:
                name = who if e.kind == "message" else f"{who} ({e.kind})"
                lines.append(f"{label(name)}: {untrusted(e.content)}")
            shared = e.data.get("_share_tools") or (for_role is not None and for_role in (e.data.get("_tools_visible_to") or []))
            if include_tools and e.tool_trace and shared:
                for t in e.tool_trace:
                    lines.append(f"    {label('trusted tool output')} {t.name}({_fmt_args(t.arguments)}) -> "
                                 f"{_indented(untrusted(t.result))}")
            if include_reasoning and for_role is not None and e.reasoning_visible(for_role) and e.role != for_role:
                lines.append(f"    {label(f'private reasoning of {who}')}: {_indented(untrusted(e.reasoning))}")
            for ev in e.evidence:
                if e.kind != "evidence" and (
                    for_role is None or ev.visible_to is None or for_role in ev.visible_to
                ):
                    lines.append("    " + ev.render())
        return "\n\n".join(lines)


def _fmt_args(args: dict[str, Any], limit: int = 2000) -> str:
    s = untrusted(", ".join(f"{k}={v!r}" for k, v in args.items()))
    return s if len(s) <= limit else s[:limit] + f"... [{len(s) - limit} more characters]"
