"""Claims carry scoped evidence. Successful checking is not a claim of semantic truth."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from .types import Claim


@dataclass(frozen=True)
class Verification:
    claim_id: str
    verifier: str
    version: str
    status: str
    scope: str
    detail: str = ""

    def __post_init__(self):
        if self.status not in {"verified", "refuted", "unknown", "error"}:
            raise ValueError("Invalid verification status")


Checker = Callable[[Claim], Awaitable[tuple[bool | None, str]]]


class VerifierRegistry:
    def __init__(self):
        self._checkers: dict[str, tuple[str, str, Checker]] = {}

    def register(self, name: str, version: str, scope: str, checker: Checker):
        if name in self._checkers:
            raise ValueError(f"Verifier already registered: {name}")
        self._checkers[name] = version, scope, checker

    async def verify(self, claim: Claim) -> Verification:
        if claim.verifier not in self._checkers:
            return Verification(
                claim.id, claim.verifier, "unknown", "unknown", "none", "No checker registered"
            )
        version, scope, checker = self._checkers[claim.verifier]
        # Scope is supplied by the trusted checker, never by the claimant.
        try:
            result, detail = await checker(claim)
            if result is not True and result is not False and result is not None:
                raise TypeError("Checker must return True, False or None")
            status = "unknown" if result is None else "verified" if result else "refuted"
        except Exception as exc:
            status, detail = "error", f"{type(exc).__name__}: {exc}"
        return Verification(claim.id, claim.verifier, version, status, scope, detail)

    async def tool(self, arguments: dict) -> dict:
        return asdict(await self.verify(Claim(**arguments)))
