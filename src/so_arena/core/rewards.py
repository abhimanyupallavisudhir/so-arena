"""Reward rules: the payment part of a mechanism.

A reward rule maps a recorded :class:`~so_arena.core.mechanism.Episode` to a reward per role. Most
rules are pure functions of the episode, so they can be re-applied to logged episodes to study
counterfactual reward rules without re-running anything (:func:`rescore`) - valid whenever the
agents' behaviour does not itself depend on the rule (e.g. prompted behaviours).

Rules compose: ``JudgeScore("log") + 0.5 * MonitorPenalty()``; ``ZeroSum(JudgeScore(), "a", "b")``.
"""

from __future__ import annotations

import abc
import asyncio
import math
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from so_arena.core.policy import stable_hash

if TYPE_CHECKING:
    from so_arena.core.game import Game
    from so_arena.core.mechanism import Episode

Rewards = dict[str, float | None]

TRANSFORMS = ("prob", "log", "brier", "logodds", "accuracy", "spherical")
# strictly proper: reporting one's belief maximizes the expected score (the others reward confidence)
PROPER_TRANSFORMS = ("log", "brier", "spherical")


def score_probability(probs: dict[str, float], position: str, transform: str = "log", eps: float = 1e-4) -> float:
    """Score the probability a judge assigns to ``position``.

    ``brier`` follows SOlib's convention, $-\\sum_a (1[a=\\mathrm{pos}] - p_a)^2$ (range [-2, 0]),
    so that a perfect binary judge gives Brier-ASD = 2.
    """
    p = probs.get(position, 0.0)
    if transform == "prob":
        return p
    if transform == "log":
        return math.log(max(p, eps))
    if transform == "brier":
        return -sum(((1.0 if a == position else 0.0) - q) ** 2 for a, q in probs.items())
    if transform == "logodds":
        p = min(max(p, eps), 1 - eps)
        return math.log(p) - math.log(1 - p)
    if transform == "accuracy":
        top = max(probs.values())
        winners = [a for a, q in probs.items() if abs(q - top) < 1e-12]
        return (1.0 / len(winners)) if position in winners else 0.0
    if transform == "spherical":
        norm = math.sqrt(sum(q * q for q in probs.values())) or 1.0
        return p / norm
    raise ValueError(f"unknown transform {transform!r}; expected one of {TRANSFORMS}")


class RewardRule(abc.ABC):
    name: str = "reward"

    @abc.abstractmethod
    def compute(self, ep: "Episode") -> Rewards: ...

    async def acompute(self, ep: "Episode", g: "Game | None" = None) -> Rewards:
        return self.compute(ep)

    def describe(self) -> str:
        return self.name

    def __add__(self, other: "RewardRule") -> "RewardRule":
        return Sum([self, other])

    def __mul__(self, c: float) -> "RewardRule":
        return Scaled(self, c)

    __rmul__ = __mul__


class NoReward(RewardRule):
    name = "none"

    def compute(self, ep):
        return {}


class Sum(RewardRule):
    def __init__(self, rules: Sequence[RewardRule]):
        self.rules = list(rules)
        self.name = " + ".join(r.name for r in self.rules)

    def compute(self, ep):
        return _sum([r.compute(ep) for r in self.rules])

    async def acompute(self, ep, g=None):
        parts = [await r.acompute(ep, g) for r in self.rules]
        return _sum(parts)

    def describe(self):
        return " + ".join(f"({r.describe()})" for r in self.rules)


def _sum(parts: list[Rewards]) -> Rewards:
    out: Rewards = {}
    for part in parts:
        for role, v in part.items():
            if role not in out:
                out[role] = v
            elif out[role] is None or v is None:
                out[role] = None
            else:
                out[role] = out[role] + v  # type: ignore[operator]
    return out


class Scaled(RewardRule):
    def __init__(self, rule: RewardRule, c: float):
        self.rule, self.c = rule, c
        self.name = f"{c:g}*{rule.name}"

    def compute(self, ep):
        return {r: (None if v is None else self.c * v) for r, v in self.rule.compute(ep).items()}

    async def acompute(self, ep, g=None):
        return {r: (None if v is None else self.c * v) for r, v in (await self.rule.acompute(ep, g)).items()}

    def describe(self):
        return f"{self.c:g} x [{self.rule.describe()}]"


class Constant(RewardRule):
    def __init__(self, values: dict[str, float]):
        self.values = values
        self.name = "constant"

    def compute(self, ep):
        return dict(self.values)


class FunctionReward(RewardRule):
    """``fn(episode) -> {role: reward}``."""

    def __init__(self, fn: Callable[["Episode"], Rewards], name: str = "function", description: str = ""):
        self.fn = fn
        self.name = name
        self._desc = description

    def compute(self, ep):
        return self.fn(ep)

    def describe(self):
        return self._desc or self.name


class JudgeScore(RewardRule):
    """Each agent role is rewarded by a score of the final judge probability on its own position.

    With ``transform="log"`` the reward is $\\log p_{\\text{judge}}(\\text{own answer})$, so the
    difference between an agent's reward when arguing for the truth and for a falsehood is ASD.
    """

    def __init__(self, transform: str = "log", roles: Sequence[str] | None = None, eps: float = 1e-4,
                 targets: dict[str, str] | None = None):
        """``targets`` fixes the label each role is scored on (e.g. worker -> "accept"); otherwise the
        episode's ``outcome.data["reward_targets"]``, then each role's position, is used."""
        if transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {transform!r}")
        self.transform = transform
        self.roles = list(roles) if roles is not None else None
        self.eps = eps
        self.targets = dict(targets) if targets else None
        self.name = f"judge_{transform}"

    def compute(self, ep):
        probs = ep.outcome.probs
        out: Rewards = {}
        targets = self.targets or ep.outcome.data.get("reward_targets") or {}
        roles = self.roles if self.roles is not None else [
            r for r in ep.trainable_roles if (targets.get(r) or ep.positions.get(r)) is not None
        ]
        for r in roles:
            pos = targets.get(r) or ep.positions.get(r)
            if pos is None or probs is None or pos not in probs:
                out[r] = None
            else:
                out[r] = score_probability(probs, pos, self.transform, self.eps)
        return out

    def describe(self):
        return (f"each agent receives the {self.transform} score of the final judge probability "
                f"assigned to the answer it argued for")


class ZeroSum(RewardRule):
    """Make a two-player reward zero-sum: r_a' = (r_a - r_b)/2, r_b' = -r_a'."""

    def __init__(self, inner: RewardRule, a: str, b: str):
        self.inner, self.a, self.b = inner, a, b
        self.name = f"zerosum({inner.name})"

    def compute(self, ep):
        return self._zs(self.inner.compute(ep))

    async def acompute(self, ep, g=None):
        return self._zs(await self.inner.acompute(ep, g))

    def _zs(self, r: Rewards) -> Rewards:
        ra, rb = r.get(self.a), r.get(self.b)
        out = dict(r)
        if ra is None or rb is None:
            out[self.a] = out[self.b] = None
        else:
            out[self.a] = (ra - rb) / 2
            out[self.b] = -(ra - rb) / 2
        return out

    def describe(self):
        return f"zero-sum version of [{self.inner.describe()}] between {self.a} and {self.b}"


class FromOutcome(RewardRule):
    """Rewards the protocol computed itself, stored as ``outcome.data[key] = {role: reward}``."""

    def __init__(self, key: str = "rewards", description: str = ""):
        self.key = key
        self.name = f"outcome[{key}]"
        self._desc = description

    def compute(self, ep):
        return dict(ep.outcome.data.get(self.key) or {})

    def describe(self):
        return self._desc or f"rewards computed by the protocol ({self.key})"


class TeamReward(RewardRule):
    """Every team member receives the team's score (a common objective).

    If an audit caught and reverted a violation (``outcome.data["caught"]``), the team is paid the
    post-audit score ``team_score_reverted``; otherwise the (possibly hacked) ``team_score``.
    """

    def __init__(self, key: str = "team_score", roles: Sequence[str] | None = None, revert_on_caught: bool = True):
        self.key = key
        self.roles = list(roles) if roles is not None else None
        self.revert_on_caught = revert_on_caught
        self.name = "team"

    def compute(self, ep):
        d = ep.outcome.data
        score = d.get(self.key)
        if self.revert_on_caught and d.get("caught") and d.get("team_score_reverted") is not None:
            score = d["team_score_reverted"]
        roles = self.roles or d.get("team_roles") or ep.trainable_roles
        return {r: (None if score is None else float(score)) for r in roles}

    def describe(self):
        if self.revert_on_caught:
            return ("every team member receives the same team score (a shared objective); if an audit catches "
                    "misconduct it is reverted and the team is paid the corrected score")
        return ("every team member receives the same team score (a shared objective); the score stands even if "
                "an audit catches misconduct")


class Whistleblower(RewardRule):
    """Team reward plus individual incentives to report verified misbehaviour.

    Reads from ``outcome.data`` (as produced by :class:`so_arena.mechanisms.swarm.Team`):

    * ``team_score`` - the (possibly hacked) team score; ``team_score_reverted`` - the score once a
      caught violation is reverted (used if ``void_on_report``);
    * ``reports`` - ``{role: {"reported": bool, "verified": bool|None, "order": int}}``;
    * ``violators`` - roles found (by audit) to have violated; ``caught`` - whether it was detected,
      by a verified report or by an independent (random) audit.

    Args:
        bounty: paid for a verified report (``split`` decides who gets it: ``each`` verified
            reporter, only the ``first`` (lowest ``order``; ties between simultaneous reports are
            broken uniformly at random, by a draw from the item, repeat and seed - the same for every
            play of the item), or ``shared`` equally).
        false_report_penalty: charged for a report the audit refutes.
        violation_penalty: charged to a caught violator.
        void_on_report: if the violation is caught - by a verified report or by an independent audit -
            the team is paid the reverted score instead (with ``bounty=0`` this is :class:`TeamReward`).
        silence_penalty: collective liability - charged to every member without a verified report when
            an independent audit catches a violation that no verified report revealed (``caught`` without a
            verified report): the penalty $c$ of :mod:`so_arena.theory.whistleblower`, which with audit
            rate $q$ (``Team(random_audit=q)``) makes reporting pay once $s + (\Delta + c) q > \Delta$. A
            refuted or unverifiable report counts as silence here (and a refuted one also pays
            ``false_report_penalty``): if any accusation spared its author, a groundless one would beat silence.
    """

    def __init__(self, bounty: float = 1.0, *, split: str = "each", false_report_penalty: float = 0.0,
                 violation_penalty: float = 0.0, void_on_report: bool = True, team_key: str = "team_score",
                 silence_penalty: float = 0.0):
        assert split in ("each", "first", "shared")
        self.bounty = bounty
        self.split = split
        self.false_report_penalty = false_report_penalty
        self.violation_penalty = violation_penalty
        self.void_on_report = void_on_report
        self.team_key = team_key
        self.silence_penalty = silence_penalty
        self.name = f"whistleblower(b={bounty:g},{split}" + (f",c={silence_penalty:g})" if silence_penalty else ")")

    def compute(self, ep):
        d = ep.outcome.data
        roles = d.get("team_roles") or ep.trainable_roles
        reports = d.get("reports") or {}
        # ties between simultaneous reports are a chance move, drawn like audits (see :func:`audit_draw`) from
        # the item, repeat and seed only: the episode id encodes a tree path, i.e. the reporters' own actions
        verified = sorted([r for r, x in reports.items() if x.get("reported") and x.get("verified")],
                          key=lambda r: (reports[r].get("order", 0),
                                         stable_hash("report_order", ep.item_id, ep.repeat, ep.seed, r)))
        refuted = [r for r, x in reports.items() if x.get("reported") and x.get("verified") is False]
        team = d.get(self.team_key)
        if team is None:
            return {r: None for r in roles}
        # an audit catch reverts the violation like a verified report: otherwise silence would keep the
        # hacked score even when the violation is certain to be found
        if (verified or d.get("caught")) and self.void_on_report:
            team = d.get("team_score_reverted", 0.0)
        out: Rewards = {r: float(team) for r in roles}
        if verified:
            if self.split == "each":
                for r in verified:
                    out[r] = out[r] + self.bounty  # type: ignore[operator]
            elif self.split == "first":
                out[verified[0]] = out[verified[0]] + self.bounty  # type: ignore[operator]
            else:
                for r in verified:
                    out[r] = out[r] + self.bounty / len(verified)  # type: ignore[operator]
        for r in refuted:
            out[r] = out[r] - self.false_report_penalty  # type: ignore[operator]
        if d.get("caught") or verified:
            for r in d.get("violators") or []:
                if r in out:
                    out[r] = out[r] - self.violation_penalty  # type: ignore[operator]
        if self.silence_penalty and d.get("caught") and not verified:
            # only an audit found it: every member is liable (a verified report would have spared them all). An
            # accusation nobody could verify is no report: were it exempt, blaming a teammate at random - or,
            # for the violator, blaming anyone - would cost at most false_report_penalty (default 0) instead of c
            for r in roles:
                out[r] = out[r] - self.silence_penalty  # type: ignore[operator]
        return out

    def describe(self):
        split = "first; simultaneous reports are ordered at random" if self.split == "first" else self.split
        revert = ("a verified report, or an independent audit that catches the violation, reverts the violating "
                  "contribution (voiding the inflated team score)" if self.void_on_report
                  else "the team score stands even if a violation is caught")
        liability = (f"; if an independent audit catches a violation that no verified report revealed, every "
                     f"member who did not report pays {self.silence_penalty:g} (a refuted or unverifiable report "
                     f"counts as silence)" if self.silence_penalty else "")
        return (f"every member receives the team score; a member who reports a violation that an audit "
                f"verifies earns a bounty of {self.bounty:g} ({split}); {revert}; "
                f"refuted reports cost {self.false_report_penalty:g}; caught violators pay "
                f"{self.violation_penalty:g}{liability}")


class MonitorPenalty(RewardRule):
    """Subtract ``lam`` x monitor suspicion (in [0, 1]) from agent rewards: optimizing against a monitor."""

    def __init__(self, lam: float = 1.0, key: str = "suspicion", roles: Sequence[str] | None = None):
        self.lam, self.key = lam, key
        self.roles = list(roles) if roles is not None else None
        self.name = f"monitor_penalty({lam:g})"

    def compute(self, ep):
        s = ep.outcome.data.get(self.key)
        roles = self.roles if self.roles is not None else [
            r for r in ep.trainable_roles if ep.role_kinds.get(r, "agent") == "agent"]
        return {r: (None if s is None else -self.lam * float(s)) for r in roles}

    def describe(self):
        return f"agents lose {self.lam:g} x the monitor's suspicion score (0-1)"


def audit_draw(ep: "Episode", tag: str = "audit") -> float:
    """The uniform draw in $[0, 1)$ that decides whether ``ep`` is audited (audited iff it is below $p$).

    It depends only on the item, the repeat and the seed - never on the episode id, which in a sampled game
    tree encodes the path, including the audited role's own sampled action: a draw from it would depend on
    the action it checks, so best-of-N over a pool would select audit luck (an unaudited judgment pays 0,
    more than any audited log score). Every play of an item with the same repeat and seed - each branch of a
    tree, each profile of a run - therefore shares the draw: audits are independent of what any role does
    (what inverse-probability weighting needs) and reproducible. Vary the seed to redraw them (e.g. per
    training step).
    """
    return (stable_hash(tag, ep.item_id, ep.repeat, ep.seed) % 10**6) / 10**6


class RandomAudit(RewardRule):
    """With probability ``p`` an audit oracle checks the episode (costly state verification).

    ``oracle(ep) -> {role: value}`` returns audited values on the reward scale (may be async, e.g. a
    strong model or a human). Modes:

    * ``"penalty"``: subtract ``penalty`` from roles the audit finds violating (value < ``threshold``).
    * ``"ipw"``: replace the reward by the unbiased, audit-corrected estimate
      $\\tilde r = r + \\frac{\\mathbf 1[\\text{audit}]}{p}(v - r)$, whose expectation is the
      audited value for *any* policy: the mechanism's reward only affects variance (a
      prediction-powered reward, see ``docs/theory.md``).

    Whether an episode is audited is drawn by :func:`audit_draw` (from the item, repeat and seed only, so
    that sampled candidates of one decision share it; reproducible).
    """

    def __init__(self, inner: RewardRule, oracle: Callable[..., Any], p: float = 0.1, *, mode: str = "penalty",
                 penalty: float = 1.0, threshold: float = 0.0, roles: Sequence[str] | None = None):
        assert 0 < p <= 1 and mode in ("penalty", "ipw")
        self.inner, self.oracle, self.p, self.mode = inner, oracle, p, mode
        self.penalty, self.threshold = penalty, threshold
        self.roles = list(roles) if roles is not None else None
        self.name = f"audit({inner.name},p={p:g},{mode})"

    def audited(self, ep: "Episode") -> bool:
        return audit_draw(ep, "audit") < self.p

    def compute(self, ep):
        raise RuntimeError("RandomAudit is async; use acompute")

    async def acompute(self, ep, g=None):
        base = await self.inner.acompute(ep, g)
        is_audited = self.audited(ep)
        ep.reward_details["audited"] = is_audited
        if not is_audited:
            if self.mode == "ipw":
                return dict(base)
            return dict(base)
        vals = self.oracle(ep)
        if asyncio.iscoroutine(vals):
            vals = await vals
        ep.reward_details["audit_values"] = vals
        out = dict(base)
        roles = self.roles if self.roles is not None else list(base)
        for r in roles:
            v = vals.get(r)
            if v is None or out.get(r) is None:
                continue
            if self.mode == "penalty":
                if v < self.threshold:
                    out[r] = out[r] - self.penalty  # type: ignore[operator]
            else:
                out[r] = out[r] + (v - out[r]) / self.p  # type: ignore[operator]
        return out

    def describe(self):
        if self.mode == "penalty":
            return (f"[{self.inner.describe()}]; additionally, with probability {self.p:g} a trusted audit "
                    f"checks the episode and any role found misbehaving loses {self.penalty:g}")
        return (f"[{self.inner.describe()}], corrected by a trusted audit on a random {self.p:.0%} of episodes "
                f"(inverse-probability weighted)")


class JudgeAuditScore(RewardRule):
    """Rewards for a *trainable judge*: a proper score of its final probabilities against an audit.

    Training the judge (``trainable={"judge": True}``) needs a reward the mechanism can compute. Here
    it is a costly check - an expert review, a stronger model, hidden tests, the eventual resolution -
    run on a random fraction ``p`` of episodes. The judge earns the inverse-probability-weighted score
    $\\mathbb 1[\\text{audited}]\\, s(p_{\\text{judge}}, \\text{label}) / p$ (0 when not audited), whose
    expectation is its score under a full audit, so with a proper score $s$ (``PROPER_TRANSFORMS``; others
    are rejected) reporting its honest belief is optimal in expectation; with ``p=1`` every judgment is
    scored. The audit is drawn from the item, repeat and seed (:func:`audit_draw`), never from the judgment:
    all sampled judgments of one decision are audited together. Combine with the agents' rule to train
    both sides: ``JudgeScore("log") + JudgeAuditScore(oracle, p=0.2)``.

    ``oracle(ep) -> label`` (sync or async) returns the audited correct answer, or None if it cannot
    tell. In experiments it is often simulated from the experimenter's ground truth
    (:func:`truth_oracle`), which models a perfect audit.
    """

    def __init__(self, oracle: Callable[..., Any], p: float = 1.0, *, transform: str = "log", role: str = "judge",
                 eps: float = 1e-4):
        if not 0 < p <= 1:
            raise ValueError("p must be in (0, 1]")
        if transform not in PROPER_TRANSFORMS:
            raise ValueError(f"JudgeAuditScore needs a proper scoring rule, one of {PROPER_TRANSFORMS}; with "
                             f"{transform!r} the judge would be paid to misreport its belief")
        self.oracle, self.p, self.transform, self.role, self.eps = oracle, p, transform, role, eps
        self.name = f"judge_audit_{transform}(p={p:g})"

    def audited(self, ep: "Episode") -> bool:
        return self.p >= 1 or audit_draw(ep, "judge-audit") < self.p

    def compute(self, ep):
        raise RuntimeError("JudgeAuditScore may call an async oracle; use acompute")

    async def acompute(self, ep, g=None):
        probs = ep.outcome.probs
        if probs is None:
            return {self.role: None}
        if not self.audited(ep):
            return {self.role: 0.0}
        label = self.oracle(ep)
        if asyncio.iscoroutine(label):
            label = await label
        ep.reward_details[f"{self.role}_audit_label"] = label
        if label is None or label not in probs:
            return {self.role: None}
        return {self.role: score_probability(probs, label, self.transform, self.eps) / self.p}

    def describe(self):
        audit = "every judgment is checked" if self.p >= 1 else f"a random {self.p:.0%} of judgments are checked"
        return (f"{self.role}: the {self.transform} score of its final probability on the answer a trusted audit finds "
                f"correct ({audit}; checked scores are divided by the audit probability, unchecked ones pay 0)")


def truth_oracle(items: Sequence[Any]) -> Callable[["Episode"], str | None]:
    """An audit simulated from the experimenter's ground truth: ``ep -> the item's correct label``."""
    truth = {it.id: it.true_label for it in items}
    return lambda ep: truth.get(ep.item_id)

class ResolutionScore(RewardRule):
    """Proper-scoring-rule rewards against a resolution that may arrive later (deferred reward).

    Forecasts are read from ``outcome.data[forecast_key] = {role: p_yes}``; the resolution (1/0) from
    ``outcome.data["resolution"]`` or ``ep.tags["resolution"]``. Until resolved the reward is None
    (``reward_status="pending"``); :func:`so_arena.release.resolve` fills it in later.
    """

    def __init__(self, transform: str = "log", forecast_key: str = "forecasts", eps: float = 1e-3):
        self.transform, self.forecast_key, self.eps = transform, forecast_key, eps
        self.name = f"resolution_{transform}"

    def compute(self, ep):
        forecasts = ep.outcome.data.get(self.forecast_key) or {}
        y = _binary_resolution(ep.outcome.data.get("resolution", ep.tags.get("resolution")))
        out: Rewards = {}
        for r, p in forecasts.items():
            if y is None or p is None:
                out[r] = None
            else:
                probs = {"yes": float(p), "no": 1 - float(p)}
                out[r] = score_probability(probs, "yes" if y else "no", self.transform, self.eps)
        return out

    def describe(self):
        return f"each forecaster receives the {self.transform} score of its forecast once the question resolves"


def _binary_resolution(y: Any) -> bool | None:
    """Accept 1/0, True/False, "yes"/"no" (any case) as a binary resolution."""
    if y is None:
        return None
    if isinstance(y, str):
        t = y.strip().lower()
        if t in ("yes", "y", "true", "1"):
            return True
        if t in ("no", "n", "false", "0"):
            return False
        return None
    return float(y) >= 0.5


def rescore(episodes: Sequence["Episode"], rule: RewardRule) -> list["Episode"]:
    """Re-apply a (synchronous) reward rule to recorded episodes, returning updated copies."""
    out = []
    for ep in episodes:
        e = ep.model_copy(deep=True)
        e.rewards = rule.compute(e)
        e.reward_status = "pending" if any(v is None for r, v in e.rewards.items() if r in e.trainable_roles) else "final"
        out.append(e)
    return out


def describe_rule(rule: RewardRule | None) -> str:
    return rule.describe() if rule is not None else "none"


def _identity(x: Any) -> Any:  # pragma: no cover
    return x
