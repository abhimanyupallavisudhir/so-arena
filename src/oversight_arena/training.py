"""Reward exports plus an actual tabular multi-agent policy-gradient reference trainer."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from .games import EmpiricalGame
from .types import Run, finite


def training_rows(runs: Sequence[Run]) -> list[dict]:
    """Terminal episodic credit, one row per trainable role. No evaluation labels.

    A neural trainer must tokenize trajectories and choose its own credit assignment,
    off-policy correction, KL regularization and optimizer; this is not PPO by itself.
    """
    rows = []
    for run in runs:
        if run.status != "complete" or run.task.split != "train":
            continue
        for role in run.roles:
            if role.trainable:
                rows.append(
                    {
                        "run_id": run.id,
                        "task_id": run.task.id,
                        "role": role.id,
                        "reward": run.reward(role.id),
                        "seed": run.seed,
                        "trajectory": [
                            e.data
                            for e in run.events
                            if e.actor == role.id and e.kind == "policy_input"
                        ],
                        "actions": [
                            e.data for e in run.events if e.actor == role.id and e.kind == "action"
                        ],
                    }
                )
    return rows


def softmax(logits: Sequence[float]) -> list[float]:
    if not logits:
        raise ValueError("Empty policy")
    for x in logits:
        finite(x)
    offset = max(logits)
    values = [math.exp(x - offset) for x in logits]
    return [v / sum(values) for v in values]


@dataclass(frozen=True)
class Checkpoint:
    step: int
    policies: tuple[tuple[float, ...], ...]
    expected_rewards: tuple[float, ...]
    regrets: dict[str, float]


def reinforce(
    game: EmpiricalGame,
    *,
    steps: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    seed: int = 0,
    trainable: Sequence[str] | None = None,
    schedule: str = "simultaneous",
    initial_logits: Sequence[Sequence[float]] | None = None,
    checkpoint_every: int = 100,
) -> list[Checkpoint]:
    """On-policy REINFORCE on categorical strategies; updates policy parameters.

    Opponents are frozen within each batch. Simultaneous updates use one old joint policy;
    alternating updates optimize one player per batch. Baselines use previous batches only.
    This tests finite policy dynamics, not cross-task neural generalization.
    """
    finite(learning_rate)
    if (
        steps < 0
        or batch_size < 1
        or learning_rate <= 0
        or checkpoint_every < 1
        or schedule not in {"simultaneous", "alternating"}
    ):
        raise ValueError("Invalid training configuration")
    selected = set(game.players if trainable is None else trainable)
    if not selected or not selected <= set(game.players):
        raise ValueError("Unknown or empty trainable set")
    indices = [i for i, player in enumerate(game.players) if player in selected]
    logits = (
        [list(x) for x in initial_logits]
        if initial_logits is not None
        else [[0.0] * len(a) for a in game.actions]
    )
    if len(logits) != len(game.players) or any(
        len(x) != len(a) for x, a in zip(logits, game.actions, strict=True)
    ):
        raise ValueError("Initial policy dimensions mismatch")
    rng = random.Random(seed)
    baseline = [0.0] * len(game.players)
    history = []
    for step in range(steps + 1):
        policies = [softmax(x) for x in logits]
        if step % checkpoint_every == 0 or step == steps:
            history.append(
                Checkpoint(
                    step,
                    tuple(map(tuple, policies)),
                    game.expected(policies),
                    game.regrets(policies),
                )
            )
        if step == steps:
            break
        gradients = [[0.0] * len(a) for a in game.actions]
        totals = [0.0] * len(game.players)
        active = indices if schedule == "simultaneous" else [indices[step % len(indices)]]
        for _ in range(batch_size):
            choices = [rng.choices(range(len(p)), weights=p)[0] for p in policies]
            profile = tuple(game.actions[i][a] for i, a in enumerate(choices))
            rewards = game.payoffs[profile]
            for i in range(len(game.players)):
                totals[i] += rewards[i]
            for i in active:
                advantage = rewards[i] - baseline[i]
                for a, p in enumerate(policies[i]):
                    gradients[i][a] += advantage * (float(a == choices[i]) - p) / batch_size
        for i in active:
            logits[i] = [
                x + learning_rate * g for x, g in zip(logits[i], gradients[i], strict=True)
            ]
        baseline = [t / batch_size for t in totals]
    return history
