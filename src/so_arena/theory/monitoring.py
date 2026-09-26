r"""Monitor-based rewards and obfuscation (docs/theory.md, section 6.1, Proposition 4).

An actor chooses a behaviour $a$ with *measured* success $m_a$ (what the grader sees) and alarm
probability $d_a$ (how often the monitor flags it). Its reward is the measured success minus a penalty
$\lambda$ per alarm; if an alarm also forfeits the credit (``forfeit``: a flag that pays $-\lambda$
outright, or deferral to a trusted model) the success is lost too:

$$\mathbb E[r_a] = m_a\,(1 - f d_a) - \lambda\, d_a, \qquad f = \mathbb 1[\text{forfeit}].$$

Each behaviour's expected reward is a line in $\lambda$, and the best response is their upper envelope
(:func:`envelope`). Proposition 4 is the case of three behaviours: honest ($m = s$), overt misbehaviour
($s + \Delta$) and covert misbehaviour ($s + \Delta - \kappa$), with $d_h \le d_c < d_o$. A penalty on a
manipulable monitor's alarms first turns overt misbehaviour covert and deters it only much later (or
never); a detector with a fixed accuracy against ground truth (:func:`probe_rates`, e.g. a simulated
probe with a target AUROC) flags covert and overt misbehaviour alike, so hiding never pays.

``MonitoredWork(penalty="flag", forfeit=False)`` implements the reward with $f = 0$; ``forfeit=True`` (the
default) and ``defer=True`` the reward with $f = 1$; ``domains.monitoring`` is the matching testbed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
INF = math.inf


@dataclass(frozen=True)
class Behaviour:
    """A behaviour's measured success $m_a$ and alarm probability $d_a$."""

    name: str
    success: float
    alarm: float


def expected_reward(b: Behaviour, lam: float, *, forfeit: bool = False) -> float:
    """$m_a (1 - f d_a) - \\lambda d_a$."""
    return b.success * (1 - (b.alarm if forfeit else 0.0)) - lam * b.alarm


def envelope(behaviours: Sequence[Behaviour], *, forfeit: bool = False) -> list[tuple[float, float, str]]:
    """The best response on $\\lambda \\in [0, \\infty)$: segments ``(start, end, name)`` of the upper envelope.

    At a crossing the actor is indifferent; the segment boundary is the crossing, and the behaviour with the
    lower alarm probability takes over from it. Behaviours with identical lines go to the one listed first
    (list the honest behaviour first so that ties favour it).
    """
    lines = [(expected_reward(b, 0.0, forfeit=forfeit), b.alarm, i) for i, b in enumerate(behaviours)]
    if not lines:
        return []
    # at lambda = 0: the highest intercept; among ties, the lowest alarm rate (it stays ahead), then list order
    cur = max(lines, key=lambda x: (x[0], -x[1], -x[2]))
    start, out = 0.0, []
    while True:
        c0, d0, _ = cur
        # lines with a lower alarm rate overtake the current one where they cross it
        cands = [((c0 - c) / (d0 - d), d, c, i) for c, d, i in lines if d < d0]
        cands = [x for x in cands if x[0] >= start - 1e-12]
        if not cands:
            out.append((start, INF, behaviours[cur[2]].name))
            return out
        lam = min(x[0] for x in cands)
        # of the lines crossing there, the flattest wins from then on (then the highest, then list order)
        _, d, c, i = min((x for x in cands if x[0] <= lam + 1e-12), key=lambda x: (x[1], -x[2], x[3]))
        lam = max(lam, start)
        if lam > start:
            out.append((start, lam, behaviours[cur[2]].name))
        cur, start = (c, d, i), lam


def best_response_of(behaviours: Sequence[Behaviour], lam: float, *, forfeit: bool = False) -> str:
    """The behaviour maximizing expected reward at ``lam`` (the envelope's segment containing it)."""
    for lo, hi, name in envelope(behaviours, forfeit=forfeit):
        if lo <= lam < hi:
            return name
    return envelope(behaviours, forfeit=forfeit)[-1][2]


# ------------------------------------------------------------------------------------ Proposition 4


@dataclass(frozen=True)
class MonitorParams:
    """Proposition 4's actor: honest ($s$), overt ($s + \\Delta$) or covert ($s + \\Delta - \\kappa$) work."""

    s: float = 0.5
    gain: float = 0.3  # Delta
    hide_cost: float = 0.05  # kappa
    d_overt: float = 1.0
    d_covert: float = 0.1
    d_honest: float = 0.05
    forfeit: bool = False

    def behaviours(self) -> list[Behaviour]:
        return [Behaviour("honest", self.s, self.d_honest),
                Behaviour("covert", self.s + self.gain - self.hide_cost, self.d_covert),
                Behaviour("overt", self.s + self.gain, self.d_overt)]


def expected_rewards(lam: float, p: MonitorParams) -> dict[str, float]:
    return {b.name: expected_reward(b, lam, forfeit=p.forfeit) for b in p.behaviours()}


def best_response(lam: float, p: MonitorParams) -> str:
    return best_response_of(p.behaviours(), lam, forfeit=p.forfeit)


def switch_points(behaviours: Sequence[Behaviour], *, forfeit: bool = False) -> dict[str, float]:
    """Penalties at which an honest / covert / overt actor's best response changes, read off the envelope.

    ``overt_to_covert``: where covert misbehaviour becomes the best response (0 if it already is without a
    penalty, inf if never);
    ``deterred``: from where honest work is the best response (inf if never, 0 if always). Works for any
    lines - e.g. behaviours estimated from episodes, to compare with :func:`thresholds`.
    """
    segs = envelope(behaviours, forfeit=forfeit)
    first = {name: lo for lo, _, name in reversed(segs)}
    return {"overt_to_covert": first.get("covert", INF),
            "deterred": first.get("honest", INF) if segs[-1][2] == "honest" else INF}


def thresholds(p: MonitorParams) -> dict[str, float]:
    """:func:`switch_points` of Proposition 4's actor (with forfeiture if ``p.forfeit``)."""
    return switch_points(p.behaviours(), forfeit=p.forfeit)


def proposition4(p: MonitorParams) -> dict[str, float]:
    """Proposition 4's closed forms (no forfeiture, $d_h \\le d_c < d_o$, $\\Delta > 0$).

    If hiding is cheap, $\\kappa/(d_o - d_c) < (\\Delta - \\kappa)/(d_c - d_h)$: overt misbehaviour turns covert
    at $\\kappa/(d_o - d_c)$ and is deterred only at $(\\Delta - \\kappa)/(d_c - d_h)$ (never if $d_c = d_h$).
    Otherwise covert misbehaviour is never optimal and overt misbehaviour stops at $\\Delta/(d_o - d_h)$.
    """
    if p.forfeit or not (p.d_honest <= p.d_covert < p.d_overt) or p.gain <= 0:
        raise ValueError("Proposition 4 assumes forfeit=False, d_honest <= d_covert < d_overt and gain > 0")
    to_covert = p.hide_cost / (p.d_overt - p.d_covert)
    deter_covert = (p.gain - p.hide_cost) / (p.d_covert - p.d_honest) if p.d_covert > p.d_honest else INF
    if p.gain > p.hide_cost and to_covert < deter_covert:
        return {"overt_to_covert": to_covert, "deterred": deter_covert}
    return {"overt_to_covert": INF, "deterred": p.gain / (p.d_overt - p.d_honest)}


def sweep(lams: Sequence[float], p: MonitorParams) -> pd.DataFrame:
    """Best response, misbehaviour and alarm rates over a grid of penalties."""
    alarm = {b.name: b.alarm for b in p.behaviours()}
    rows = []
    for lam in lams:
        a = best_response(float(lam), p)
        rows.append({"penalty": float(lam), "action": a, "misbehaviour": float(a != "honest"),
                     "alarm_rate": alarm[a], "alarm_rate_when_misbehaving": alarm[a] if a != "honest" else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ detectors


def dprime(auroc: float) -> float:
    """Separation $d' = \\sqrt 2\\,\\Phi^{-1}(\\mathrm{AUROC})$ of two unit-variance Gaussians with this AUROC."""
    if not 0.0 < auroc < 1.0:
        raise ValueError("auroc must be in (0, 1)")
    return math.sqrt(2) * _N.inv_cdf(auroc)


def probe_rates(auroc: float, threshold: float = 0.5) -> tuple[float, float]:
    """(false-alarm rate, detection rate) of a simulated probe flagging readings above ``threshold``.

    Readings are $\\Phi(z - d'/2)$ with $z \\sim N(0, 1)$ on honest behaviour and $N(d', 1)$ on misbehaviour
    (:class:`~so_arena.integrations.probes.SimulatedProbe`), so a reading exceeds $t$ iff
    $z > d'/2 + \\Phi^{-1}(t)$.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    d, q = dprime(auroc), _N.inv_cdf(threshold)
    return 1 - _N.cdf(d / 2 + q), 1 - _N.cdf(q - d / 2)


def with_detector(p: MonitorParams, false_alarm: float, detection: float) -> MonitorParams:
    """The same actor facing a detector whose alarms depend only on *whether* it misbehaves (not how)."""
    return replace(p, d_honest=false_alarm, d_covert=detection, d_overt=detection)


# ------------------------------------------------------------------------------------ audits


def min_audit_rate(gain: float, penalty: float, *, forfeit: bool = True) -> float:
    """Smallest audit probability $a$ deterring a violation worth ``gain`` with a bounded ``penalty``.

    A caught violator pays $P$ and, if ``forfeit``, loses the gain: deterrence iff $aP \\ge (1-a)G$, i.e.
    $a \\ge G/(G+P)$; if the gain is kept, iff $aP \\ge G$, i.e. $a \\ge G/P$ (inf when that exceeds 1: no
    audit rate suffices). Becker (1968) with limited liability: bounded penalties need frequent audits.
    """
    if gain <= 0:
        return 0.0
    if penalty <= 0:
        return 1.0 if forfeit else INF
    a = gain / (gain + penalty) if forfeit else gain / penalty
    return a if a <= 1 else INF
