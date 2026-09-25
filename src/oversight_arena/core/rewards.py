"""Reward rules and scoring transforms.

A mechanism = protocol (game form) + reward rule. Reward rules are *pure functions of the
recorded episode*, so recorded episodes can be re-scored under alternative rules without
re-running any model (e.g. log vs Brier, zero-sum vs not, different bounty sizes).

Caveat: re-scoring is exact for "one step of optimisation" analyses (best-of-N selection,
preference data), where behaviour is sampled from a fixed base policy. When agents are told
the incentive structure, a different rule may induce different behaviour; then re-run.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from .episode import EpisodeRecord
from .util import clamp

Transform = Literal["log", "brier", "linear", "logit", "spherical", "win", "centered"]


def score_prob(p: float, transform: Transform = "log", eps: float = 1e-4, n_options: int = 2) -> float:
    """Map a judge probability for the answer an agent argued for into an agent score.

    - ``log``: $\\log p$ (ASD's canonical choice)
    - ``brier``: $-(1-p)^2$ (bounded; the ASD paper's recommended numerics)
    - ``linear``: $p$
    - ``logit``: $\\log\\frac{p}{1-p}$
    - ``spherical``: $p/\\sqrt{p^2+(1-p)^2}$ (binary spherical score)
    - ``win``: $1[p>1/n]$ (+0.5 at ties)
    - ``centered``: $p - 1/n$
    """
    p = clamp(float(p), eps, 1 - eps)
    if transform == "log":
        return math.log(p)
    if transform == "brier":
        return -((1 - p) ** 2)
    if transform == "linear":
        return p
    if transform == "logit":
        return math.log(p / (1 - p))
    if transform == "spherical":
        return p / math.sqrt(p * p + (1 - p) ** 2)
    if transform == "win":
        thr = 1.0 / n_options
        return 1.0 if p > thr + 1e-9 else (0.5 if abs(p - thr) <= 1e-9 else 0.0)
    if transform == "centered":
        return p - 1.0 / n_options
    raise ValueError(f"unknown transform {transform}")


class RewardRule(BaseModel, ABC):
    """Maps an episode record to rewards for the trainable roles."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    batch: ClassVar[bool] = False  # rewards depend on other episodes (peer prediction etc.)
    delayed: ClassVar[bool] = False  # rewards require later resolution

    @property
    def name(self) -> str:
        """Type and full configuration (nested rules by name), e.g. ``Combined(rules=[...])``."""

        def fmt(v: Any) -> str:
            if isinstance(v, RewardRule):
                return v.name
            if isinstance(v, (list, tuple)):
                return "[" + ",".join(fmt(x) for x in v) + "]"
            if isinstance(v, BaseModel):
                return f"{type(v).__name__}({v.model_dump(mode='json')})"
            return repr(v) if isinstance(v, str) else str(v)

        args = ",".join(f"{k}={fmt(getattr(self, k))}" for k in type(self).model_fields)
        return f"{type(self).__name__}({args})"

    @abstractmethod
    def __call__(self, record: EpisodeRecord) -> dict[str, float]: ...


class BatchRewardRule(RewardRule):
    """Rewards computed jointly over a batch of episodes (e.g. multi-task peer prediction)."""

    batch: ClassVar[bool] = True

    def __call__(self, record: EpisodeRecord) -> dict[str, float]:
        return self.compute_batch([record])[0]

    @abstractmethod
    def compute_batch(self, records: list[EpisodeRecord]) -> list[dict[str, float]]: ...


class NoReward(RewardRule):
    def __call__(self, record: EpisodeRecord) -> dict[str, float]:
        return {}


class OutcomeField(RewardRule):
    """Reward each role with a numeric field the mechanism wrote to ``outcome['rewards']``.

    Useful for mechanisms whose payoff logic is intrinsically procedural (e.g. markets).
    """

    field: str = "raw_rewards"

    def __call__(self, record: EpisodeRecord) -> dict[str, float]:
        raw = record.outcome.get(self.field, {}) or {}
        return {k: float(v) for k, v in raw.items() if v is not None}


class JudgeProbability(RewardRule):
    """Reward = transform(judge probability of the option the role argued for).

    Reads ``outcome['probs']`` (option -> probability) and ``outcome['positions']``
    (role -> option). With ``zero_sum=True`` and exactly two positioned roles, rewards are
    antisymmetrised: $r_A = s(p_A) - s(p_B)$, $r_B = -r_A$.
    """

    transform: Transform = "log"
    zero_sum: bool = False
    roles: list[str] | None = None  # restrict to these roles

    def __call__(self, record: EpisodeRecord) -> dict[str, float]:
        probs: dict[str, float] = record.outcome.get("probs") or {}
        positions: dict[str, Any] = record.outcome.get("positions") or {}
        n = max(len(probs), 2)
        out: dict[str, float] = {}
        trainable = set(record.trainable_roles)
        for role, pos in positions.items():
            if pos is None or role not in trainable:
                continue
            if self.roles is not None and role not in self.roles:
                continue
            out[role] = score_prob(probs.get(pos, 0.0), self.transform, n_options=n)
        if self.zero_sum and len(out) == 2:
            (ra, va), (rb, vb) = list(out.items())
            out = {ra: va - vb, rb: vb - va}
        return out


def reward_rule_from_spec(spec: str | dict | RewardRule | None) -> RewardRule | None:
    """Build a reward rule from a registry spec (see :mod:`oversight_arena.registry`)."""
    if spec is None or isinstance(spec, RewardRule):
        return spec
    from ..registry import build

    return build("reward", spec)


class DelayedRewardRule(RewardRule):
    """Rewards that need information only available at resolution time (e.g. a forecasting
    question's outcome). ``score(record, task)`` is called when the task is resolved; until then
    the episode's rewards are pending. Use :meth:`Results.resolve` to fill them in later.
    """

    delayed: ClassVar[bool] = True

    def __call__(self, record: EpisodeRecord) -> dict[str, float]:
        raise RuntimeError("DelayedRewardRule needs the task: call score(record, task)")

    @abstractmethod
    def score(self, record: EpisodeRecord, task: Any) -> dict[str, float]: ...


class Combined(RewardRule):
    """Sum several reward rules (e.g. debater rewards + a label-trained judge's reward)."""

    rules: list[RewardRule]

    def __call__(self, record: EpisodeRecord) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.rules:
            for k, v in r(record).items():
                out[k] = out.get(k, 0.0) + v
        return out


class JudgeLabelScore(RewardRule):
    """Reward a *trainable judge* with the score of its probability on the labelled answer, on
    episodes where the mechanism obtained a label (``outcome['label']``) through a declared
    :class:`~oversight_arena.channels.Label` channel; 0 otherwise."""

    transform: Transform = "log"
    role: str = "judge"

    def __call__(self, record: EpisodeRecord) -> dict[str, float]:
        if self.role not in record.trainable_roles:
            return {}
        label = record.outcome.get("label")
        probs = record.outcome.get("probs") or {}
        if not label:
            return {self.role: 0.0}
        p = sum(float(probs.get(o, 0.0)) for o in label)
        return {self.role: score_prob(p, self.transform, n_options=max(2, len(probs)))}
