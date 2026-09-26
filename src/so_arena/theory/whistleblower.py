"""An analytic model of reporting in teams (the "snitching" game).

A violation (e.g. a reward hack) has occurred in a team of ``n`` agents; if it stands, every agent
receives the inflated team reward ``R``; if it is reported and verified it is reverted and everyone
loses ``delta`` (the hack's contribution to the team score). Each agent simultaneously chooses to
stay *silent* or *report*.

* Each report is verified independently with probability ``rho`` (audit reliability).
* A verified reporter earns a bounty ``s``: paid to ``each`` verified reporter, or (``first`` /
  ``shared``) split so that in expectation the bounty is divided among verified reporters.
* If no report is verified, an independent audit still detects the violation with probability ``q``;
  then the hack is reverted and every agent without a verified report pays a penalty ``c`` (collective
  liability): an unverified accusation does not buy exemption.

Payoff to one agent when ``k`` of the others report (with $\\bar\\rho = 1-\\rho$):

.. math::
    u(\\text{report}\\mid k) = R - \\Delta\\,[1-\\bar\\rho^{k+1}] - (\\Delta + c) q\\,\\bar\\rho^{k+1} + B(k), \\qquad
    u(\\text{silent}\\mid k) = R - \\Delta\\,[1-\\bar\\rho^{k}] - (\\Delta + c) q\\,\\bar\\rho^{k},

with $B(k) = s\\rho$ for ``each`` and $B(k) = s\\,(1-\\bar\\rho^{k+1})/(k+1)$ for ``first``/``shared``.

Key facts (``rho=1, q=c=0``): with a purely common reward (``s=0``) silence is weakly dominant, so
training on the shared objective pushes every agent to stay silent. With a bounty, reporting is
dominant iff ``s > delta``; for ``0 < s < delta`` there are two strict equilibria - all silent and all
report - and a symmetric mixed equilibrium with report probability
$p^* = 1 - (s/\\Delta)^{1/(n-1)}$ (``each``) that separates their basins under learning dynamics:
whether snitching pays depends on whether an agent expects others to snitch. The silent basin
shrinks as the team grows.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import binom

from so_arena.games.normal_form import NormalFormGame, game_from_function


def bounty_share(k: int, *, s: float, split: str, rho: float) -> float:
    if split == "each":
        return s * rho
    if split in ("first", "shared"):
        if rho <= 0:
            return 0.0
        return s * (1 - (1 - rho) ** (k + 1)) / (k + 1)
    raise ValueError(split)


def payoff(action: str, k: int, *, R: float = 1.0, delta: float = 0.5, s: float = 0.0, split: str = "each",
           rho: float = 1.0, q: float = 0.0, c: float = 0.0) -> float:
    """Expected payoff of ``action`` when ``k`` of the other agents report."""
    rb = 1 - rho
    if action == "report":
        # its own report unverified too (prob rho_bar): the audit's penalty c falls on it as on the silent
        return R - delta * (1 - rb ** (k + 1)) - (delta + c) * q * rb ** (k + 1) + bounty_share(k, s=s, split=split, rho=rho)
    if action == "silent":
        return R - delta * (1 - rb ** k) - (delta + c) * q * rb ** k
    raise ValueError(action)


def report_gain(p: float, n: int, **params) -> float:
    """Expected gain from reporting over silence when each of the n-1 others reports with probability p."""
    ks = np.arange(n)
    w = binom.pmf(ks, n - 1, p)
    return float(sum(wk * (payoff("report", int(k), **params) - payoff("silent", int(k), **params)) for k, wk in zip(ks, w)))


def whistleblower_game(n: int = 3, **params) -> NormalFormGame:
    """The n-player normal-form game; outcomes: ``reverted`` (P violation reverted) = ground-truth welfare."""
    players = [f"agent_{i + 1}" for i in range(n)]
    strategies = {p: ["silent", "report"] for p in players}
    rho, q = params.get("rho", 1.0), params.get("q", 0.0)

    def pay(names: dict[str, str]) -> dict[str, float]:
        out = {}
        for p in players:
            k = sum(names[o] == "report" for o in players if o != p)
            out[p] = payoff(names[p], k, **params)
        return out

    def reverted(names: dict[str, str]) -> float:
        m = sum(v == "report" for v in names.values())
        p_rep = 1 - (1 - rho) ** m
        return p_rep + (1 - p_rep) * q

    return game_from_function(players, strategies, pay, outcomes={"reverted": reverted, "gt_welfare": reverted},
                              name=f"whistleblower(n={n})")


def symmetric_equilibria(n: int = 3, *, grid: int = 400, **params) -> list[dict[str, float | str]]:
    """Symmetric equilibria as report probabilities p, with stability under learning dynamics.

    An equilibrium is *stable* if the report gain crosses zero from above there (for interior
    points) or points into the boundary (for pure points).
    """
    eqs = []
    g0, g1 = report_gain(0.0, n, **params), report_gain(1.0, n, **params)
    if g0 <= 1e-12:
        eqs.append({"p_report": 0.0, "kind": "pure", "stable": g0 < -1e-12})
    if g1 >= -1e-12:
        eqs.append({"p_report": 1.0, "kind": "pure", "stable": g1 > 1e-12})
    ps = np.linspace(0, 1, grid + 1)
    gs = np.array([report_gain(p, n, **params) for p in ps])
    tol = 1e-12
    roots: list[tuple[float, bool]] = []
    for i in range(1, grid):  # roots exactly on interior grid points
        if abs(gs[i]) < tol:
            roots.append((float(ps[i]), bool(gs[i - 1] > 0 > gs[i + 1])))
    for i in range(grid):
        if gs[i] * gs[i + 1] < 0 and abs(gs[i]) >= tol and abs(gs[i + 1]) >= tol:
            lo, hi = ps[i], ps[i + 1]
            for _ in range(80):
                mid = (lo + hi) / 2
                if report_gain(lo, n, **params) * report_gain(mid, n, **params) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(((lo + hi) / 2, bool(gs[i] > 0 > gs[i + 1])))
    for p_root, stable in sorted(roots):
        if 1e-9 < p_root < 1 - 1e-9:
            eqs.append({"p_report": float(p_root), "kind": "mixed", "stable": stable})
    return eqs


def mixed_threshold_each(n: int, s: float, delta: float) -> float | None:
    """Closed form p* = 1 - (s/delta)^(1/(n-1)) for split='each', rho=1, q=c=0 (None if no interior eq)."""
    if n < 2 or not (0 < s < delta):
        return None
    return 1 - (s / delta) ** (1 / (n - 1))


def sweep(ratios: Sequence[float], ns: Sequence[int] = (2, 3, 5, 10), *, delta: float = 1.0, split: str = "each",
          rho: float = 1.0, q: float = 0.0, c: float = 0.0) -> pd.DataFrame:
    """Equilibrium structure as the bounty-to-stake ratio s/delta varies.

    Columns: n, ratio, silent_eq (is all-silent an equilibrium), report_eq, report_dominant,
    p_threshold (interior equilibrium = boundary of the silent basin), silent_basin (fraction of
    initial report propensities from which dynamics converge to silence).
    """
    rows = []
    for n in ns:
        for r in ratios:
            params = dict(R=1.0, delta=delta, s=r * delta, split=split, rho=rho, q=q, c=c)
            eqs = symmetric_equilibria(n, **params)
            silent = any(e["p_report"] == 0.0 for e in eqs)
            report = any(e["p_report"] == 1.0 for e in eqs)
            interior = [e["p_report"] for e in eqs if e["kind"] == "mixed"]
            gain0 = report_gain(0.0, n, **params)
            if interior:
                basin = float(min(interior))
            elif silent and not report:
                basin = 1.0
            elif report and not silent:
                basin = 0.0
            else:
                basin = 1.0 if gain0 < 0 else 0.0
            rows.append({
                "n": n, "ratio": r, "silent_eq": silent, "report_eq": report,
                "report_dominant": all(payoff("report", k, **params) > payoff("silent", k, **params) for k in range(n)),
                "p_threshold": float(min(interior)) if interior else math.nan,
                "silent_basin": basin,
            })
    return pd.DataFrame(rows)
