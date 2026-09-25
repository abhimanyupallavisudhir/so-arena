r"""Analytic model of whistleblowing in agent swarms (see docs/theory.md, §8).

Setting (one violation has occurred). $n$ workers are paid from a shared score that the
violation inflated. Each worker's *stake* $g$ is how much its own payoff rises if the violation
goes undetected ($g = G$ when every worker is paid the team score, as with
:class:`~oversight_arena.mechanisms.swarm.TeamReward`; $g = G/n$ when a pool $G$ is split).
Each of the other workers observes the violation independently with probability $o$, as in the
simulated environments, so an observer does not know how many others saw it. (``observers=m``
instead fixes the number of observers at $m$, common knowledge.) Each observer chooses to
**report** or stay **silent**. A verified report detects the violation; otherwise a random audit
detects it with probability $a$. On detection the gain is forfeited, the offender pays $P$,
reporters share a bounty $b$ (equal split, or winner-take-all in random order: same expectation)
and observers who stayed silent pay $c$ ("misprision").

Let $K$ be the number of *other* observers who report. Reporting's advantage over silence is
$$\Delta(K) = \begin{cases} b + a\,c - (1-a)\,g & K = 0 \\ \dfrac{b}{1+K} + c & K \ge 1 \end{cases}$$
If every observer reports with probability $q$, then $K \sim \mathrm{Bin}(n-2, oq)$ (or
$\mathrm{Bin}(m-1, q)$ with $m$ known observers). With $T = (1-a)g - ac$:

- $b > T$: reporting is a best response to anything; all reporting is the unique equilibrium.
- $b < T$: all silence is a strict equilibrium. All reporting is one too iff $b \ge b_R$
  (:func:`report_equilibrium_bounty`); then there is also a unique mixed equilibrium and the game
  is a coordination game: whether snitching pays depends on what the others are expected to do.
  Below $b_R$ silence is the only equilibrium, because an observer may be the only one.
- With a fully shared objective ($b = c = 0$), silence weakly dominates (strictly, if detection
  forfeits the gain): nobody reports and, anticipating that, misconduct pays $(1-a)g$ per
  worker. Every worker is *complicit* in the offender's gain.

Equilibrium selection: for two known observers the game is a 2x2 coordination game and
:func:`risk_dominant` applies (Harsanyi-Selten). For more observers $\Delta(K)$ falls in $K \ge 1$
(bounties are shared), so the game is not supermodular and global-games selection does not
apply; learning dynamics select by basin (:func:`basin_of_reporting`, :func:`mean_field`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom


@dataclass(frozen=True)
class SwarmParams:
    n: int = 3  # workers
    g: float = 0.3  # each worker's payoff gain if the violation goes undetected (its stake)
    b: float = 0.1  # bounty (total, shared among verified reporters)
    P: float = 0.3  # offender penalty on detection (beyond forfeiting the gain)
    a: float = 0.0  # random audit probability
    c: float = 0.0  # penalty for silent observers when the violation is detected
    o: float = 0.8  # probability each other worker observes the violation
    beta: float = 0.0  # offender's private benefit from misconduct (beyond its stake)
    observers: int | None = None  # None: random observers (each w.p. o); m: exactly m, common knowledge

    @classmethod
    def pooled(cls, n: int, G: float, **kw: float) -> "SwarmParams":
        """A pool inflated by $G$ is split equally, so each worker's stake is $G/n$."""
        return cls(n=n, g=G / n, **kw)  # type: ignore[arg-type]

    @property
    def threshold(self) -> float:
        """Bounty above which reporting is a best response even if nobody else reports."""
        return (1 - self.a) * self.g - self.a * self.c


_EPS = 1e-12


def advantage(K: int, p: SwarmParams) -> float:
    r"""$\Delta(K)$: gain from reporting vs staying silent when $K$ other observers report."""
    if K == 0:
        return p.b + p.a * p.c - (1 - p.a) * p.g
    return p.b / (1 + K) + p.c


def _others(p: SwarmParams, m: int | None) -> tuple[int, float]:
    r"""(N, r): an observer faces $K \sim \mathrm{Bin}(N, r\,q)$ other reporters."""
    m = p.observers if m is None else m
    return (max(p.n - 2, 0), p.o) if m is None else (max(m - 1, 0), 1.0)


def other_reporters(q: float, p: SwarmParams, m: int | None = None) -> np.ndarray:
    """Distribution of $K$ (index = number of other reporters) when each observer reports w.p. $q$."""
    N, r = _others(p, m)
    return binom.pmf(np.arange(N + 1), N, r * q) if N > 0 else np.array([1.0])


def expected_advantage(q: float, p: SwarmParams, m: int | None = None) -> float:
    r"""$E[\Delta(K)]$ for an observer when every other observer reports w.p. $q$."""
    w = other_reporters(q, p, m)
    return float(sum(wi * advantage(k, p) for k, wi in enumerate(w)))


def report_equilibrium_bounty(p: SwarmParams, m: int | None = None) -> float:
    r"""Smallest bounty at which *all report* is an equilibrium: $E[\Delta(K)] \ge 0$ at $q = 1$.
    $b_R = (P_0 T - (1-P_0)c) / (P_0 + \sum_{k\ge1} P_k/(1+k))$ with $P_k$ the distribution of $K$ at $q=1$."""
    w = other_reporters(1.0, p, m)
    num = w[0] * p.threshold - (1 - w[0]) * p.c
    den = w[0] + sum(wk / (1 + k) for k, wk in enumerate(w) if k >= 1)
    return max(0.0, num / den) if den > 0 else 0.0


def equilibria(p: SwarmParams, m: int | None = None) -> list[dict]:
    """Symmetric equilibria of the reporting subgame ($q$ = report probability of an observer)."""
    f = lambda q: expected_advantage(q, p, m)  # noqa: E731
    f0, f1 = f(0.0), f(1.0)
    f0, f1 = (0.0 if abs(v) < _EPS else v for v in (f0, f1))  # knife edges (e.g. b = b_R) are weak
    N, _ = _others(p, m)
    out = []
    if f1 >= 0:
        out.append({"q": 1.0, "type": "all report", "stable": f1 > 0 or N == 0})
    if f0 <= 0:
        out.append({"q": 0.0, "type": "all silent", "stable": f0 < 0})
    if f0 < 0 < f1 and N > 0:
        out.append({"q": float(brentq(f, 0.0, 1.0)), "type": "mixed", "stable": False})
    return out


def risk_dominant(p: SwarmParams) -> str:
    r"""Risk-dominant equilibrium of the game between two known observers (Harsanyi & Selten, 1988):
    reporting iff $\tfrac32 b + (1+a)c > (1-a)g$. Only defined for two observers: with more,
    $\Delta(K)$ is not increasing in $K$, so global-games selection results do not apply."""
    u, _ = payoff_matrix_2(p)
    report_loss = u[0, 0] - u[1, 0]  # what a deviation from (report, report) costs
    silent_loss = u[1, 1] - u[0, 1]  # what a deviation from (silent, silent) costs
    if silent_loss <= 0:
        return "all report"
    return "all report" if report_loss > silent_loss else ("all silent" if report_loss < silent_loss else "tie")


def detection_prob(q: float, p: SwarmParams) -> float:
    """P(violation detected) if each other worker observes w.p. $o$ and then reports w.p. $q$
    (with ``observers=m``: exactly $m$ observers)."""
    none_report = (1 - p.o * q) ** (p.n - 1) if p.observers is None else (1 - q) ** p.observers
    return 1 - (1 - p.a) * none_report


def offender_gain(q: float, p: SwarmParams) -> float:
    """Expected payoff of committing the violation vs not, given reporting propensity $q$."""
    d = detection_prob(q, p)
    return (1 - d) * p.g + p.beta - d * p.P


def basin_of_reporting(p: SwarmParams, m: int | None = None) -> float:
    r"""Fraction of initial report propensities from which (replicator) learning reaches 'report'.

    $1 - q^*$ where $E[\Delta](q^*) = 0$; 1 if reporting is a best response even when nobody
    else reports; 0 if silence is a best response even when everyone reports."""
    f = lambda q: expected_advantage(q, p, m)  # noqa: E731
    f0, f1 = f(0.0), f(1.0)
    if f0 >= -_EPS:
        return 1.0
    if f1 <= _EPS:
        return 0.0
    return 1 - brentq(f, 0.0, 1.0)


def regime(p: SwarmParams, m: int | None = None) -> str:
    """'report dominant', 'coordination' (both pure equilibria strict) or 'silent dominant'."""
    types = {e["type"] for e in equilibria(p, m) if e["stable"]}
    if types == {"all report"}:
        return "report dominant"
    if {"all silent", "all report"} <= types:
        return "coordination"
    return "silent dominant"


def phase_diagram(
    b_over_stake: np.ndarray | None = None, audit: np.ndarray | None = None, **kw: float
) -> pd.DataFrame:
    """Grid over bounty (in units of each worker's stake $g$) and audit probability.

    Columns: ``regime`` ('report dominant' / 'coordination' / 'silent dominant'),
    ``basin_report`` (share of initial report propensities that learning takes to reporting),
    whether misconduct is deterred in the *best* and *worst* equilibria, and (for two known
    observers only) the ``risk_dominant`` equilibrium.
    """
    b_over_stake = np.linspace(0, 1.5, 61) if b_over_stake is None else b_over_stake
    audit = np.linspace(0, 0.9, 46) if audit is None else audit
    base = SwarmParams(**kw)  # type: ignore[arg-type]
    rows = []
    for a in audit:
        for x in b_over_stake:
            p = replace(base, a=float(a), b=float(x) * base.g)
            eqs = equilibria(p)
            q_best = max(e["q"] for e in eqs)
            q_worst = min(e["q"] for e in eqs)
            row = {
                "audit": float(a), "bounty_over_stake": float(x), "regime": regime(p),
                "basin_report": basin_of_reporting(p),
                "deterred_best": offender_gain(q_best, p) < 0, "deterred_worst": offender_gain(q_worst, p) < 0,
            }
            if base.observers == 2:
                row["risk_dominant"] = risk_dominant(p)
            rows.append(row)
    return pd.DataFrame(rows)


def mean_field(x0: float, q0: float, p: SwarmParams, iters: int = 600, lr: float = 1.0,
               opportunity: float | None = None) -> pd.DataFrame:
    r"""Learning dynamics of the *full* game — whether to commit the violation and whether to report
    others' — for symmetric workers trained by multiplicative weights (the replicator dynamics
    :meth:`EmpiricalGame.replicator` runs, and the mean-field limit of natural-policy-gradient training).

    Each worker has the chance to commit the violation w.p. ``opportunity`` (default $1/n$: one
    random worker per project) and does so w.p. $x$; each other worker observes it w.p. $o$ and
    then reports w.p. $q$. Because a worker's payoff is additively separable in its two
    decisions, the multiplicative-weights update preserves independence, so the marginals obey
    $$\mathrm{logit}\,x \mathrel{+}= \eta\,\pi\,\mathrm{gain}(q),\qquad
      \mathrm{logit}\,q \mathrel{+}= \eta\,(1-\pi)\,x\,o\,E_{K\sim\mathrm{Bin}(n-2,\,oq)}[\Delta(K)],$$
    with $\pi$ the opportunity probability and $\mathrm{gain}(q)$ = :func:`offender_gain`.
    Returns the trajectory (columns ``step, x, q``). (Random observers; ``p.observers`` is ignored.)
    """
    pi = 1.0 / p.n if opportunity is None else opportunity
    lx = np.log(x0 / (1 - x0))
    lq = np.log(q0 / (1 - q0))
    ks = np.arange(max(p.n - 1, 1))
    rows = []
    for t in range(iters + 1):
        x, q = 1 / (1 + np.exp(-lx)), 1 / (1 + np.exp(-lq))
        rows.append({"step": t, "x": float(x), "q": float(q)})
        if t == iters:
            break
        w = binom.pmf(ks, p.n - 2, p.o * q) if p.n > 2 else np.array([1.0])
        rep_adv = float(sum(wi * advantage(int(k), p) for wi, k in zip(w, ks)))
        lx = float(np.clip(lx + lr * pi * offender_gain(q, replace(p, observers=None)), -50, 50))
        lq = float(np.clip(lq + lr * (1 - pi) * x * p.o * rep_adv, -50, 50))
    return pd.DataFrame(rows)


def basin_of_deterrence(p: SwarmParams, x0: float = 0.5, q0s: np.ndarray | None = None, iters: int = 600,
                        lr: float = 1.0) -> float:
    """Share of initial report propensities $q_0$ (with initial violation propensity $x_0$) from
    which the full-game dynamics (:func:`mean_field`) end with the violation deterred ($x<1/2$)."""
    q0s = np.linspace(0.02, 0.98, 25) if q0s is None else q0s
    return float(np.mean([mean_field(x0, float(q), p, iters, lr)["x"].iloc[-1] < 0.5 for q in q0s]))


def payoff_matrix_2(p: SwarmParams) -> tuple[np.ndarray, np.ndarray]:
    """2-observer reporting game (rows/cols: report, silent), payoffs relative to the clean
    outcome — handy for :class:`~oversight_arena.analysis.games.EmpiricalGame` checks."""
    u = np.zeros((2, 2))
    u[0, 0] = p.b / 2  # both report: detected, bounty split
    u[0, 1] = p.b  # I report alone
    u[1, 0] = -p.c  # the other reports; I stayed silent
    u[1, 1] = (1 - p.a) * p.g - p.a * p.c  # nobody reports: undetected unless audited
    return u, u.T


def summary(p: SwarmParams) -> dict:
    return {
        "stake": p.g,
        "observers": p.observers if p.observers is not None else f"random (each w.p. {p.o:g})",
        "report_dominant_above": p.threshold,
        "all_report_equilibrium_above": report_equilibrium_bounty(p),
        "bounty": p.b,
        "regime": regime(p),
        "equilibria": equilibria(p),
        "risk_dominant": risk_dominant(p) if p.observers == 2 else None,
        "basin_report": basin_of_reporting(p),
        "offender_gain_if_all_report": offender_gain(1.0, p),
        "offender_gain_if_all_silent": offender_gain(0.0, p),
    }
