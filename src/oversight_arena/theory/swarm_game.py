"""Analytic model of whistleblowing in agent swarms (see docs/theory.md, §6).

Setting (one violation has occurred). $n$ workers are paid from a shared score that the
violation inflated. Each worker's *stake* $g$ is how much its own payoff rises if the violation
goes undetected ($g = G$ when every worker is paid the team score, as with
:class:`~oversight_arena.mechanisms.swarm.TeamReward`; $g = G/n$ when a pool $G$ is split).
Each of the other workers observed the violation independently with probability $o$; say $m$
did. Each observer chooses to **report** or stay **silent**. A verified report detects the
violation; otherwise a random audit detects it with probability $a$. On detection the gain is
forfeited, the offender pays $P$, reporters share a bounty $b$ (equal split, or
winner-take-all in random order — same expectation) and observers who stayed silent pay $c$
("misprision").

Let $K$ be the number of *other* observers who report. Reporting's advantage over silence is
$$\\Delta(K) = \\begin{cases} b + a\\,c - (1-a)\\,g & K = 0 \\\\ \\dfrac{b}{1+K} + c & K \\ge 1 \\end{cases}$$
Reporting when others report is always (weakly) better — the gain is lost anyway and the bounty
is pure upside. Reporting when nobody else does costs your stake. Hence:

- $b > (1-a)g - ac$ ⇒ reporting is a best response to anything ⇒ unique equilibrium: report
  (*full implementation* of honest oversight).
- $b < (1-a)g - ac$ ⇒ a coordination game: *all silent* and *all report* are both equilibria
  (plus an unstable mixed one at $q^*$). Whether snitching pays depends on what you expect
  others to do. Replicator/policy-gradient learning converges to *report* iff the initial report
  propensity exceeds $q^*$; global-games/risk-dominance selection picks *report* iff
  $\\frac1m\\sum_{K=0}^{m-1}\\Delta(K) > 0$ (Laplacian beliefs; for $m=2$:
  $\\tfrac32 b + (1+a)c > (1-a)g$).
- With a fully shared objective ($b = c = 0$), silence weakly dominates (strictly, if detection
  forfeits the gain): nobody reports and, anticipating that, misconduct pays $(1-a)g$ per
  worker — every worker is *complicit* in the offender's gain.
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

    @classmethod
    def pooled(cls, n: int, G: float, **kw: float) -> "SwarmParams":
        """A pool inflated by $G$ is split equally, so each worker's stake is $G/n$."""
        return cls(n=n, g=G / n, **kw)  # type: ignore[arg-type]

    @property
    def threshold(self) -> float:
        """Bounty above which reporting is a best response even if nobody else reports."""
        return (1 - self.a) * self.g - self.a * self.c

    @property
    def m(self) -> int:
        """Typical number of observers (≥ 2 so that the reporting subgame is a game)."""
        return max(2, round(self.o * (self.n - 1)))


def advantage(K: int, p: SwarmParams) -> float:
    """$\\Delta(K)$: gain from reporting vs staying silent when $K$ other observers report."""
    if K == 0:
        return p.b + p.a * p.c - (1 - p.a) * p.g
    return p.b / (1 + K) + p.c


def expected_advantage(q: float, m: int, p: SwarmParams) -> float:
    """$E[\\Delta]$ for an observer when each of the other $m-1$ observers reports w.p. $q$."""
    ks = np.arange(m)
    w = binom.pmf(ks, m - 1, q) if m > 1 else np.array([1.0])
    return float(sum(wi * advantage(int(k), p) for wi, k in zip(w, ks)))


def equilibria(m: int, p: SwarmParams) -> list[dict]:
    """Symmetric equilibria of the reporting subgame among $m$ observers ($q$ = report prob.)."""
    f = lambda q: expected_advantage(q, m, p)  # noqa: E731
    out = []
    if f(1.0) >= 0:
        out.append({"q": 1.0, "type": "all report", "stable": f(1.0) > 0 or m == 1})
    if f(0.0) <= 0:
        out.append({"q": 0.0, "type": "all silent", "stable": f(0.0) < 0})
    if f(0.0) < 0 < f(1.0) and m > 1:
        q = brentq(f, 0.0, 1.0)
        out.append({"q": float(q), "type": "mixed", "stable": False})
    return out


def selected(m: int, p: SwarmParams) -> str:
    """Equilibrium picked by global-games / risk-dominance reasoning (Laplacian beliefs: the
    number of *other* reporters is uniform on $\\{0,…,m-1\\}$)."""
    avg = float(np.mean([advantage(k, p) for k in range(m)]))
    return "all report" if avg > 0 else "all silent"


def detection_prob(q: float, p: SwarmParams) -> float:
    """P(violation detected) if each other worker observes w.p. $o$ and then reports w.p. $q$."""
    none_report = (1 - p.o * q) ** (p.n - 1)
    return 1 - (1 - p.a) * none_report


def offender_gain(q: float, p: SwarmParams) -> float:
    """Expected payoff of committing the violation vs not, given reporting propensity $q$."""
    d = detection_prob(q, p)
    return (1 - d) * p.g + p.beta - d * p.P


def basin_of_reporting(p: SwarmParams, m: int | None = None) -> float:
    """Fraction of initial report propensities from which (replicator) learning reaches 'report'.

    $1 - q^*$ where $E[\\Delta(q^*)] = 0$; 1 if reporting is a best response even when nobody
    else reports; 0 if silence is a best response even when everyone reports."""
    m = m or p.m
    f = lambda q: expected_advantage(q, m, p)  # noqa: E731
    f0, f1 = f(0.0), f(1.0)
    if f0 >= -1e-12:
        return 1.0
    if f1 <= 0:
        return 0.0
    return 1 - brentq(f, 0.0, 1.0)


def regime(p: SwarmParams, m: int | None = None) -> str:
    """'report dominant', 'coordination' (both pure equilibria strict) or 'silent dominant'."""
    types = {e["type"] for e in equilibria(m or p.m, p) if e["stable"]}
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
    ``selected`` (risk-dominant equilibrium), and whether misconduct is deterred in the *best*
    and *worst* equilibria.
    """
    b_over_stake = np.linspace(0, 1.5, 61) if b_over_stake is None else b_over_stake
    audit = np.linspace(0, 0.9, 46) if audit is None else audit
    base = SwarmParams(**kw)  # type: ignore[arg-type]
    m = base.m
    rows = []
    for a in audit:
        for x in b_over_stake:
            p = replace(base, a=float(a), b=float(x) * base.g)
            eqs = [e for e in equilibria(m, p) if e["stable"] or e["type"] != "mixed"]
            q_best = max(e["q"] for e in eqs)
            q_worst = min(e["q"] for e in eqs)
            rows.append({
                "audit": float(a), "bounty_over_stake": float(x), "regime": regime(p, m),
                "basin_report": basin_of_reporting(p, m), "selected": selected(m, p),
                "deterred_best": offender_gain(q_best, p) < 0, "deterred_worst": offender_gain(q_worst, p) < 0,
            })
    return pd.DataFrame(rows)


def mean_field(x0: float, q0: float, p: SwarmParams, iters: int = 600, lr: float = 1.0,
               opportunity: float | None = None) -> pd.DataFrame:
    """Learning dynamics of the *full* game — whether to commit the violation and whether to report
    others' — for symmetric workers trained by multiplicative weights (the replicator dynamics
    :meth:`EmpiricalGame.replicator` runs, and the mean-field limit of natural-policy-gradient training).

    Each worker has the chance to commit the violation w.p. ``opportunity`` (default $1/n$: one
    random worker per project) and does so w.p. $x$; each other worker observes it w.p. $o$ and
    then reports w.p. $q$. Because a worker's payoff is additively separable in its two
    decisions, the multiplicative-weights update preserves independence, so the marginals obey
    $$\mathrm{logit}\,x \mathrel{+}= \eta\,\pi\,\mathrm{gain}(q),\qquad
      \mathrm{logit}\,q \mathrel{+}= \eta\,(1-\pi)\,x\,o\,E_{K\sim\mathrm{Bin}(n-2,\,oq)}[\Delta(K)],$$
    with $\pi$ the opportunity probability and $\mathrm{gain}(q)$ = :func:`offender_gain`.
    Returns the trajectory (columns ``step, x, q``).
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
        lx = float(np.clip(lx + lr * pi * offender_gain(q, p), -50, 50))
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
    m = p.m
    return {
        "stake": p.g,
        "threshold_bounty": p.threshold,
        "bounty": p.b,
        "regime": regime(p, m),
        "equilibria": equilibria(m, p),
        "risk_dominant": selected(m, p),
        "basin_report": basin_of_reporting(p, m),
        "offender_gain_if_all_report": offender_gain(1.0, p),
        "offender_gain_if_all_silent": offender_gain(0.0, p),
    }
