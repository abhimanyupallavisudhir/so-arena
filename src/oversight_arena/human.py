"""Blinded review packets and auditable human ratings, independent of mechanism rewards."""

from __future__ import annotations

from dataclasses import asdict

from .types import Evaluation, Measurement, Run, digest, finite


def review_packet(run: Run, role: str, rubric: str) -> dict:
    if role not in {r.id for r in run.roles}:
        raise ValueError("Unknown role")
    return {
        "packet_id": digest([run.id, role, rubric]),
        "run_id": run.id,
        "role": role,
        "prompt": run.task.prompt,
        "rubric": rubric,
        "evidence": [asdict(e) for e in run.events if e.audience is None],
    }


def import_rating(
    packet: dict,
    *,
    value: float,
    rater: str,
    seconds: float,
    dimension: str = "quality",
    version: str = "1",
) -> Evaluation:
    finite(value)
    if not rater or finite(seconds) < 0:
        raise ValueError("Rater and nonnegative review time required")
    return Evaluation(
        packet["run_id"],
        f"human:{rater}",
        version,
        {
            packet["role"]: {
                dimension: Measurement(
                    value,
                    "observed",
                    rater,
                    "human",
                    f"packet={packet['packet_id']}; review_seconds={seconds}",
                )
            }
        },
    )
