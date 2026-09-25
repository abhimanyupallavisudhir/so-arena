"""Claims are assertions. Receipts bind a verifier result to the exact claim and artifact."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .core import JSON, Event, digest
from .runtime import Context


@dataclass(frozen=True)
class Claim:
    author: str
    statement: str
    artifact: JSON
    specification: str
    scope: str = "artifact satisfies specification"

    @property
    def id(self) -> str:
        return digest(asdict(self))


async def verify_claim(
    context: Context, claim: Claim, verifier: str, *, recipients: tuple[str, ...] | None = None
) -> Event:
    """Verifier tool returns {valid: bool, details?: ...}; errors do not become invalid claims."""
    result = await context.use_tool(claim.author, verifier, asdict(claim), recipients=recipients)
    data = result.content["result"]
    if type(data.get("valid")) is not bool:
        raise ValueError("Verifier must return an explicit boolean validity")
    return context.emit(
        "mechanism",
        "verification",
        {
            "claim_id": claim.id,
            "artifact_hash": digest(claim.artifact),
            "specification_hash": digest(claim.specification),
            "scope": claim.scope,
            "verifier": verifier,
            "version": result.content["version"],
            "valid": data["valid"],
            "details": data.get("details"),
            "tool_event": result.id,
        },
        recipients=recipients,
    )


def receipt_matches(receipt: Event, claim: Claim) -> bool:
    return (
        receipt.actor == "mechanism"
        and receipt.kind == "verification"
        and receipt.content.get("claim_id") == claim.id
        and receipt.content.get("artifact_hash") == digest(claim.artifact)
        and receipt.content.get("specification_hash") == digest(claim.specification)
    )
