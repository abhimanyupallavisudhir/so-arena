"""Analytic model of whistleblowing in agent swarms.

Setting (one violation has occurred). $n$ workers share a score; the violation inflated it by
$G$ (so each worker's share of the illicit gain is $G/n$ if it goes undetected). Each of the
other workers observed it independently with probability $o$; say $m$ did. Each observer
chooses to **report** or stay **silent**. A verified report detects the violation; otherwise a
random audit detects it with probability $a$. On detection the illicit gain is forfeited, the
offender pays $P$, reporters share a bounty $b$ (equal split, or winner-take-all by random
order — same expectation), and — optionally — observers who stayed silent pay $c$
("failure to report").

Let $K$ be the number of *other* observers who report. The reporting advantage is
$$\\Delta(K) = \\frac{b}{1+K} - \\begin{cases} (1-a)\\,G/n - a\\,c & K = 0 \\\\ -c & K \\ge 1 \\end{cases}$$
Reporting when others report is always (weakly) better — the gain is lost anyway, and the
bounty is pure upside. Reporting when nobody else does costs your share of the gain. Hence:

- $b > (1-a)G/n - ac$ ⇒ reporting is a best response to anything ⇒ unique equilibrium: report.
- $b < (1-a)G/n - ac$ ⇒ a coordination game: *all silent* and *all report* are both equilibria
  (plus an unstable mixed one at $q^*$), and whether snitching pays depends on what you expect
  others to do. Learning dynamics converge to *report* iff the initial report propensity
  exceeds $q^*$ — the basin of the honest equilibrium shrinks as $b\\,n/G \\to 0$.
- With a fully shared objective ($b=c=0$), silence weakly dominates: nobody reports and,
  anticipating that, misconduct pays $G/n$ per worker.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom


@dataclass
class SwarmParams:
    n: int = 5  # workers
    G: float = 1.0  # illicit gain to the team score
    b: float = 0.1  # bounty
    P: float = 1.0  # offender penalty on detection
    a: float = 0.0  # random audit probability
    c: float = 0.0  # penalty for silent observers when detected
    o: float = 0.8  # probability each other worker observes the violation
    beta: float = 0.0  # offender's private benefit from misconduct (beyond the shared gain)

    @property
    def share(self) -> float:
        return self.G / self.n

    @property
    def threshold(self) -> float:
        """Bounty above which reporting is a best response even if nobody else reports."""
        return (1 - self.a) * self.share - self.a * self.c


def advantage(K: int, p: SwarmParams) -> float:
    if K == 0:
        return p.b - ((1 - p.a) * p.share - p.a * p.c)
    return p.b / (1 + K) + p.c


def expected_advantage(q: float, m: int, p: SwarmParams) -> float:
    """E[Δ] for an observer when each of the other m-1 observers reports with prob. q."""
    ks = np.arange(m)
    w = binom.pmf(ks, m - 1, q) if m > 1 else np.array([1.0])
    return float(sum(wi * advantage(int(k), p) for wi, k in zip(w, ks)))


def equilibria(m: int, p: SwarmParams) -> list[dict]:
    """Symmetric equilibria of the reporting subgame among m observers (q = report prob.)."""
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


def detection_prob(q: float, p: SwarmParams) -> float:
    """P(violation detected) if each other worker observes w.p. o and then reports w.p. q."""
    none_report = (1 - p.o * q) ** (p.n - 1)
    return 1 - (1 - p.a) * none_report


def offender_gain(q: float, p: SwarmParams) -> float:
    """Expected payoff of committing the violation vs not, given reporting propensity q."""
    d = detection_prob(q, p)
    return (1 - d) * p.share + p.beta - d * p.P


def basin_of_reporting(p: SwarmParams, m: int | None = None) -> float:
    """Fraction of initial report propensities from which (replicator) learning reaches 'report'.

    $1 - q^*$ where $q^*$ solves $E[\\Delta(q^*)] = 0$; 1 if reporting is a best response even
    when nobody else reports; 0 if silence is a best response even when everyone reports."""
    m = m or max(2, round(p.o * (p.n - 1)))
    f = lambda q: expected_advantage(q, m, p)  # noqa: E731
    f0, f1 = f(0.0), f(1.0)
    if f0 >= -1e-12:
        return 1.0
    if f1 <= 0:
        return 0.0
    return 1 - brentq(f, 0.0, 1.0)


def phase_diagram(
    b_over_share: np.ndarray | None = None, audit: np.ndarray | None = None, **kw: float
) -> pd.DataFrame:
    """Grid over bounty (in units of the per-worker illicit share $G/n$) and audit probability.

    Columns: regime ('report dominant' / 'coordination' / 'silent dominant'), basin of the
    reporting equilibrium, and whether misconduct is deterred in the *best* and *worst*
    equilibria.
    """
    b_over_share = np.linspace(0, 1.5, 61) if b_over_share is None else b_over_share
    audit = np.linspace(0, 0.9, 46) if audit is None else audit
    base = SwarmParams(**kw)  # type: ignore[arg-type]
    m = max(2, round(base.o * (base.n - 1)))
    rows = []
    for a in audit:
        for x in b_over_share:
            p = SwarmParams(**{**base.__dict__, "a": float(a), "b": float(x) * base.share})
            eqs = equilibria(m, p)
            types = {e["type"] for e in eqs}
            regime = (
                "report dominant" if types == {"all report"} else
                "coordination" if "all silent" in types and "all report" in types else
                "silent dominant"
            )
            q_best = max(e["q"] for e in eqs if e["stable"] or e["type"] != "mixed")
            q_worst = min(e["q"] for e in eqs if e["stable"] or e["type"] != "mixed")
            rows.append({
                "audit": float(a), "bounty_over_share": float(x), "regime": regime,
                "basin_report": basin_of_reporting(p, m),
                "deterred_best": offender_gain(q_best, p) < 0, "deterred_worst": offender_gain(q_worst, p) < 0,
            })
    return pd.DataFrame(rows)


def payoff_matrix_2(p: SwarmParams) -> tuple[np.ndarray, np.ndarray]:
    """2-observer reporting game (rows/cols: report, silent) — handy for EmpiricalGame checks."""
    u = np.zeros((2, 2))
    # (report, report)
    u[0, 0] = p.b / 2
    u[0, 1] = p.b  # I report, other silent
    u[1, 0] = -p.c  # I silent, other reports -> detected
    u[1, 1] = (1 - p.a) * p.share - p.a * p.c
    return u, u.T


def summary(p: SwarmParams) -> dict:
    m = max(2, round(p.o * (p.n - 1)))
    return {
        "threshold_bounty": p.threshold,
        "bounty": p.b,
        "equilibria": equilibria(m, p),
        "basin_report": basin_of_reporting(p, m),
        "offender_gain_if_all_report": offender_gain(1.0, p),
        "offender_gain_if_all_silent": offender_gain(0.0, p),
    }
