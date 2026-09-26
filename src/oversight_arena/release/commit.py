"""Hash commitments: leaf hashes, Merkle roots, inclusion proofs (tamper-evident releases)."""

from __future__ import annotations

import hashlib
import secrets
from typing import Any

from ..core.util import canonical_json


def h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leaf_hash(item: Any, salt: str = "") -> str:
    """Leaf commitment to a JSON-able item; a secret ``salt`` makes it *hiding* (sealed)."""
    return h(b"\x00" + salt.encode() + canonical_json(item).encode())


def merkle_root(leaves: list[str]) -> str:
    if not leaves:
        return h(b"")
    level = sorted(leaves)
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(h(b"\x01" + bytes.fromhex(a) + bytes.fromhex(b)))
        level = nxt
    return level[0]


def merkle_proof(leaves: list[str], leaf: str) -> list[tuple[str, str]]:
    """Inclusion proof: list of (sibling hash, side) from the leaf to the root."""
    level = sorted(leaves)
    idx = level.index(leaf)
    proof = []
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            if i == idx or i + 1 == idx:
                proof.append((b, "R") if i == idx else (a, "L"))
            nxt.append(h(b"\x01" + bytes.fromhex(a) + bytes.fromhex(b)))
        idx //= 2
        level = nxt
    return proof


def verify_proof(leaf: str, proof: list[tuple[str, str]], root: str) -> bool:
    cur = leaf
    for sib, side in proof:
        cur = h(b"\x01" + bytes.fromhex(cur) + bytes.fromhex(sib)) if side == "R" else h(b"\x01" + bytes.fromhex(sib) + bytes.fromhex(cur))
    return cur == root


def new_salt() -> str:
    return secrets.token_hex(16)
