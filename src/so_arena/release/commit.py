"""Hash commitments for sealed releases: salted per-item commitments, a Merkle root and inclusion proofs.

A commitment to ``content`` is ``SHA-256(0x00 || salt || canonical JSON)`` with a secret random 32-byte salt per
leaf: *binding* (no other content has the same commitment) and *hiding* (it says nothing about the content,
even content with few possible values such as a yes/no decision). The Merkle tree hashes inner nodes as
``SHA-256(0x01 || left || right)`` - the prefixes keep leaves and inner nodes apart - and promotes an unpaired
last node to the next level unchanged (no duplication, so no two leaf lists share a root). An inclusion proof
lists the siblings from a leaf up to the root: with the root alone (the published digest), one revealed item
and its proof, anyone can check that the item was committed to at release time.
"""

from __future__ import annotations

import hashlib
import json
import secrets
from typing import Any

LEAF, NODE = b"\x00", b"\x01"


def canonical(content: Any) -> bytes:
    """The bytes a commitment covers: JSON with sorted keys and no insignificant whitespace (no NaN)."""
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def new_salt() -> str:
    return secrets.token_hex(32)


def commitment(content: Any, salt: str) -> str:
    return hashlib.sha256(LEAF + bytes.fromhex(salt) + canonical(content)).hexdigest()


def _node(left: str, right: str) -> str:
    return hashlib.sha256(NODE + bytes.fromhex(left) + bytes.fromhex(right)).hexdigest()


def merkle_root(leaves: list[str]) -> str:
    """The root over ``leaves`` (hex digests) in the given order; SHA-256 of nothing for no leaves."""
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    level = list(leaves)
    while len(level) > 1:
        level = [_node(level[i], level[i + 1]) if i + 1 < len(level) else level[i] for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: list[str], index: int) -> list[list[str]]:
    """The inclusion proof of ``leaves[index]``: ``[sibling, side]`` pairs from the leaf up, ``side`` being where
    the sibling sits (``"L"`` or ``"R"``)."""
    if not 0 <= index < len(leaves):
        raise IndexError(f"leaf {index} of {len(leaves)}")
    proof, level, i = [], list(leaves), index
    while len(level) > 1:
        if i % 2 == 0 and i + 1 < len(level):
            proof.append([level[i + 1], "R"])
        elif i % 2 == 1:
            proof.append([level[i - 1], "L"])
        level = [_node(level[j], level[j + 1]) if j + 1 < len(level) else level[j] for j in range(0, len(level), 2)]
        i //= 2
    return proof


def verify_proof(leaf: str, proof: list[list[str]], root: str) -> bool:
    cur = leaf
    for sibling, side in proof:
        if side == "R":
            cur = _node(cur, sibling)
        elif side == "L":
            cur = _node(sibling, cur)
        else:
            return False
    return cur == root


def verify_opening(opening: dict[str, Any], proof: list[list[str]], root: str) -> bool:
    """Whether a revealed leaf (``{"salt", "content", "commitment"}``) matches its commitment and the commitment
    is included under ``root`` - all a third party needs besides the published digest."""
    try:
        c = commitment(opening["content"], opening["salt"])
    except (KeyError, TypeError, ValueError):
        return False
    return c == opening.get("commitment") and verify_proof(c, proof, root)
