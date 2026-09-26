r"""An analytic model of reporting in teams (the "snitching" game), with partial observation.

One member of a team commits a violation (e.g. a reward hack). If it stands, every member receives the
inflated team reward ``R``; if it is caught it is reverted and every member loses ``delta`` ($\Delta$, the
hack's contribution to the team score). Each of the offender's ``n`` teammates *witnesses* it
independently with probability ``o`` (``o=1``: everyone sees everything) and a witness chooses to
*report* or stay *silent*; a teammate who did not see it has nothing to report. (``n`` is the number of
potential witnesses, so a :class:`~so_arena.mechanisms.swarm.Team` of ``n + 1`` workers with one violator.)

* Each report is verified independently with probability ``rho`` (audit reliability); an unverified
  report is refuted and pays the false-report penalty ``f``.
* A verified reporter earns a bounty ``s``: paid to ``each`` verified reporter, or (``first`` /
  ``shared``) split so that in expectation the bounty is divided among verified reporters.
* If no report is verified, an independent audit still catches the violation with probability ``q``;
  then every member without a verified report pays ``c`` (collective liability): an unverified
  accusation does not buy exemption.
* Once the violation is caught (by a verified report or the audit), every witness without a verified
  report pays ``m`` (misprision), and the offender pays ``P``; ``beta`` is the offender's private benefit.

Payoff to a witness when ``k`` of the other witnesses report, with $\bar\rho = 1-\rho$ and
$D_k = 1 - \bar\rho^{k}(1-q)$ the probability that the violation is caught without its own report:

$$u(\text{silent}\mid k) = R - (\Delta + m)\,D_k - c\,q\,\bar\rho^{k},\qquad
  u(\text{report}\mid k) = R - \Delta D_{k+1} - c\,q\,\bar\rho^{k+1} - \bar\rho\,(m D_k + f) + B(k),$$

with $B(k) = s\rho$ for ``each`` and $B(k) = s\,(1-\bar\rho^{k+1})/(k+1)$ for ``first``/``shared``. If
every witness reports with probability $p$, a witness faces $K \sim \mathrm{Bin}(n-1, o\,p)$ other
reporters, and reporting pays iff $E[u(\text{report}\mid K) - u(\text{silent}\mid K)] > 0$.

Key facts (docs/theory.md, section 8; ``rho=1``, ``q=c=m=f=0``):

* With a purely common reward (``s=0``) silence is weakly dominant: training on the shared objective
  pushes every witness to stay silent. Reporting is dominant iff ``s`` exceeds the witness's stake
  (:func:`dominance_bounty`, $\Delta$ here).
* Below that, "everyone reports" is an equilibrium only if ``s`` is at least $b_R$
  (:func:`report_equilibrium_bounty`): a witness may be the only one ($P(K=0) > 0$ when ``o < 1``), and
  a small bounty does not pay for reporting alone. Below $b_R$ silence is the *only* equilibrium; between
  $b_R$ and the stake silence and reporting are both stable, separated by a unique mixed equilibrium
  (with ``o=1`` and ``split="each"``, $p^* = 1 - (s/\Delta)^{1/(n-1)}$), and whether snitching pays depends
  on whether a witness expects others to snitch. With ``o=1``, $b_R = 0$ for ``n >= 2``.
* The whole game - whether to violate and whether to report - has two-dimensional learning dynamics
  (:func:`mean_field`, :func:`team_game`), and which equilibrium training reaches depends on the
  training algorithm (natural vs. vanilla policy gradient: :func:`basin_of_deterrence`).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from so_arena.games.normal_form import NormalFormGame, game_from_function

DEFAULTS: dict[str, Any] = dict(R=1.0, delta=0.5, s=0.0, split="each", rho=1.0, q=0.0, c=0.0, m=0.0, f=0.0, o=1.0,
                                P=0.0, beta=0.0)
TEAM_STRATEGIES = ["honest·silent", "honest·report", "violate·silent", "violate·report"]
_EPS = 1e-12


def params(**kw: Any) -> dict[str, Any]:
    """The model's parameters with defaults filled in; an unknown name is an error (a typo would otherwise
    silently analyse a different game)."""
    unknown = sorted(set(kw) - set(DEFAULTS))
    if unknown:
        raise TypeError(f"unknown whistleblower parameter(s) {unknown}; the model's parameters are {sorted(DEFAULTS)}")
    p = {**DEFAULTS, **kw}
    if p["split"] not in ("each", "first", "shared"):
        raise ValueError(f"split must be 'each', 'first' or 'shared', got {p['split']!r}")
    for k in ("rho", "q", "o"):
        if not 0.0 <= p[k] <= 1.0:
            raise ValueError(f"{k} is a probability, got {p[k]}")
    return p


def bounty_share(k: int, *, s: float, split: str, rho: float) -> float:
    """Expected bounty of a reporter when ``k`` others report (before its own verification)."""
    if split == "each":
        return s * rho
    if split in ("first", "shared"):
        if rho <= 0:
            return 0.0
        return s * (1 - (1 - rho) ** (k + 1)) / (k + 1)
    raise ValueError(split)


def payoff(action: str, k: int, **kw: Any) -> float:
    """Expected payoff of a teammate taking ``action`` when ``k`` of the other witnesses report.

    ``action``: ``"report"`` or ``"silent"`` (a witness), or ``"unaware"`` (did not see the violation: it
    can neither report nor be charged with misprision).
    """
    p = params(**kw)
    R, d, rho, q, c, m = p["R"], p["delta"], p["rho"], p["q"], p["c"], p["m"]
    rb = 1 - rho

    def caught(j: int) -> float:  # P(the violation is caught) with j reports (verified w.p. rho each) + the audit
        return 1 - rb ** j * (1 - q)

    if action == "silent":
        return R - (d + m) * caught(k) - c * q * rb ** k
    if action == "unaware":
        return R - d * caught(k) - c * q * rb ** k
    if action == "report":
        # its own report unverified too (prob rho_bar): refuted (pays f), and liable like the silent - to c if
        # only the audit catches it, to m if anyone does - since only a verified report exempts its author
        return (R - d * caught(k + 1) - c * q * rb ** (k + 1) - rb * (m * caught(k) + p["f"])
                + bounty_share(k, s=p["s"], split=p["split"], rho=rho))
    raise ValueError(action)


def advantage(k: int, **kw: Any) -> float:
    r"""$\Delta(k)$: what reporting gains a witness over silence when ``k`` other witnesses report."""
    return payoff("report", k, **kw) - payoff("silent", k, **kw)


def other_reporters(p: float, n: int, **kw: Any) -> np.ndarray:
    r"""Distribution of $K$, the number of other witnesses who report, when each of the ``n - 1`` other
    teammates witnesses w.p. ``o`` and reports w.p. ``p``: $K \sim \mathrm{Bin}(n-1, o\,p)$."""
    o = params(**kw)["o"]
    return binom.pmf(np.arange(n), n - 1, o * p) if n > 1 else np.array([1.0])


def report_gain(p: float, n: int, **kw: Any) -> float:
    """Expected gain from reporting over silence for a witness when each teammate witnesses w.p. ``o`` and
    reports w.p. ``p`` (conditional on having witnessed: the decision a witness faces)."""
    w = other_reporters(p, n, **kw)
    return float(sum(wk * advantage(k, **kw) for k, wk in enumerate(w)))


def whistleblower_game(n: int = 3, **kw: Any) -> NormalFormGame:
    """The ``n``-witness reporting game in normal form, before observation: ``report`` means "report if you
    witness it". Payoffs are expected over who witnesses (with ``o=1``: everyone). Outcomes:
    ``reverted`` (P violation caught and reverted) = ground-truth welfare."""
    p = params(**kw)
    o, rho, q = p["o"], p["rho"], p["q"]
    players = [f"agent_{i + 1}" for i in range(n)]
    strategies = {pl: ["silent", "report"] for pl in players}
    # u[action][r]: expected payoff when r other teammates play "report" (each reports only if it witnessed)
    u = {a: [sum(binom.pmf(k, r, o) * (o * payoff(a, k, **kw) + (1 - o) * payoff("unaware", k, **kw))
                 for k in range(r + 1)) for r in range(n)] for a in ("silent", "report")}

    def pay(names: dict[str, str]) -> dict[str, float]:
        out = {}
        for pl in players:
            r = sum(names[x] == "report" for x in players if x != pl)
            out[pl] = float(u[names[pl]][r])
        return out

    def reverted(names: dict[str, str]) -> float:
        r = sum(v == "report" for v in names.values())
        return 1 - (1 - q) * (1 - o * rho) ** r

    return game_from_function(players, strategies, pay, outcomes={"reverted": reverted, "gt_welfare": reverted},
                              name=f"whistleblower(n={n})")


def symmetric_equilibria(n: int = 3, *, grid: int = 400, **kw: Any) -> list[dict[str, float | str]]:
    """Symmetric equilibria as report probabilities p, with stability under learning dynamics.

    An equilibrium is *stable* if the report gain crosses zero from above there (for interior
    points) or points into the boundary (for pure points).
    """
    params(**kw)
    eqs = []
    g0, g1 = report_gain(0.0, n, **kw), report_gain(1.0, n, **kw)
    if g0 <= 1e-12:
        eqs.append({"p_report": 0.0, "kind": "pure", "stable": g0 < -1e-12})
    if g1 >= -1e-12:
        eqs.append({"p_report": 1.0, "kind": "pure", "stable": g1 > 1e-12})
    ps = np.linspace(0, 1, grid + 1)
    gs = np.array([report_gain(p, n, **kw) for p in ps])
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
                if report_gain(lo, n, **kw) * report_gain(mid, n, **kw) <= 0:
                    hi = mid
                else:
                    lo = mid
            roots.append(((lo + hi) / 2, bool(gs[i] > 0 > gs[i + 1])))
    for p_root, stable in sorted(roots):
        if 1e-9 < p_root < 1 - 1e-9:
            eqs.append({"p_report": float(p_root), "kind": "mixed", "stable": stable})
    return eqs


def mixed_threshold_each(n: int, s: float, delta: float) -> float | None:
    """Closed form p* = 1 - (s/delta)^(1/(n-1)) for split='each', rho=o=1, q=c=m=f=0 (None if no interior eq)."""
    if n < 2 or not (0 < s < delta):
        return None
    return 1 - (s / delta) ** (1 / (n - 1))


def _bounty_line(k: int, **kw: Any) -> tuple[float, float]:
    """``advantage(k)`` is affine in the bounty: (slope, intercept)."""
    p = params(**kw)
    beta_k = bounty_share(k, s=1.0, split=p["split"], rho=p["rho"])
    return beta_k, advantage(k, **{**kw, "s": 0.0})


def dominance_bounty(n: int, **kw: Any) -> float:
    """Smallest bounty at which reporting beats silence whatever the other witnesses do (every k they may
    report in); ``inf`` if no bounty does (e.g. ``rho=0``)."""
    out = 0.0
    for k in range(n):
        slope, icpt = _bounty_line(k, **kw)
        if icpt < 0:
            out = max(out, -icpt / slope if slope > 0 else math.inf)
    return out


def report_equilibrium_bounty(n: int, **kw: Any) -> float:
    r"""$b_R$: the smallest bounty at which "every witness reports" is an equilibrium - the expected advantage
    is non-negative when $K \sim \mathrm{Bin}(n-1, o)$ (everyone else reports if they witnessed). With shared
    bounties, ``rho=1`` and ``q=c=m=f=0`` this is $P_0\Delta / (P_0 + \sum_{k\ge1} P_k/(1+k))$ with $P_k$ the
    distribution of $K$; ``inf`` if no bounty makes it one."""
    w = other_reporters(1.0, n, **kw)
    slope = sum(wk * _bounty_line(k, **kw)[0] for k, wk in enumerate(w))
    icpt = sum(wk * _bounty_line(k, **kw)[1] for k, wk in enumerate(w))
    if icpt >= 0:
        return 0.0
    return -icpt / slope if slope > 0 else math.inf


def silent_basin(n: int, **kw: Any) -> float:
    """Share of initial report propensities from which learning dynamics end in universal silence: the
    first interior equilibrium above which reporting pays (1 if it never does; 0 if silence is unstable)."""
    eqs = symmetric_equilibria(n, **kw)
    if not any(e["p_report"] == 0.0 and e["stable"] for e in eqs):
        return 0.0
    up = [e["p_report"] for e in eqs if e["kind"] == "mixed" and not e["stable"]]
    return float(min(up)) if up else 1.0


def regime(n: int, **kw: Any) -> str:
    """The reporting subgame's equilibrium structure (stable symmetric equilibria):

    * ``"silence"`` - universal silence is the only stable equilibrium (with ``o < 1`` this includes small
      positive bounties below $b_R$, where "everyone reports" is not an equilibrium at all);
    * ``"coordination"`` - silence and (possibly partial) reporting are both stable: training selects by
      where it starts;
    * ``"reporting"`` - silence is not an equilibrium (e.g. reporting is dominant);
    * ``"neutral"`` - knife-edge parameters with no stable equilibrium.
    """
    stable = [e["p_report"] for e in symmetric_equilibria(n, **kw) if e["stable"]]
    silent, report = 0.0 in stable, any(p > 0 for p in stable)
    if silent and report:
        return "coordination"
    if silent:
        return "silence"
    return "reporting" if report else "neutral"


def risk_dominant(**kw: Any) -> str:
    """Risk-dominant equilibrium (Harsanyi & Selten 1988) of the two-witness game (``n=2``, any ``o``):
    the pure equilibrium whose deviation losses have the larger product. With more witnesses there is no
    comparable result - bounties shared among reporters make the advantage *fall* in the number of other
    reporters, so the game lacks the strategic complementarity global-games selection needs."""
    g = whistleblower_game(2, **kw)
    a, b = g.players
    U = g.payoffs[a]  # U[own, other]; 0 = silent, 1 = report (symmetric game)
    report_loss = U[1, 1] - U[0, 1]  # what deviating from (report, report) costs
    silent_loss = U[0, 0] - U[1, 0]  # what deviating from (silent, silent) costs
    if report_loss < 0:
        return "all silent"
    if silent_loss < 0:
        return "all report"
    diff = report_loss ** 2 - silent_loss ** 2
    return "tie" if abs(diff) < _EPS else ("all report" if diff > 0 else "all silent")


# ------------------------------------------------------------------------------------------------
# The whole game: whether to violate, and whether to report
# ------------------------------------------------------------------------------------------------


def detection_prob(p: float, n: int, **kw: Any) -> float:
    """P(the violation is caught) when each of the offender's ``n`` teammates witnesses w.p. ``o`` and then
    reports w.p. ``p``."""
    k = params(**kw)
    return 1 - (1 - k["q"]) * (1 - k["o"] * p * k["rho"]) ** n


def offender_gain(p: float, n: int, **kw: Any) -> float:
    r"""Expected payoff of committing the violation minus not committing it, given report propensity ``p``:
    $\beta + \Delta(1-d) - P d - c\,q\,(1-o p\rho)^n$ with $d$ = :func:`detection_prob` (the offender has
    no verified report, so it shares the collective liability)."""
    k = params(**kw)
    d = detection_prob(p, n, **kw)
    return k["beta"] + k["delta"] * (1 - d) - k["P"] * d - k["c"] * k["q"] * (1 - k["o"] * p * k["rho"]) ** n


def mean_field(x0: float, p0: float, n: int, *, steps: int = 600, lr: float = 1.0, opportunity: float | None = None,
               natural: bool = True, **kw: Any) -> pd.DataFrame:
    r"""Learning dynamics of the whole game for symmetric members of a team of ``n + 1``: each holds the
    opportunity to violate with probability ``opportunity`` (default $1/(n+1)$: one random member per task)
    and then violates w.p. $x$; a teammate who witnesses a violation reports it w.p. $p$.

    Each member's payoff is additively separable in its two decisions, so with ``natural=True``
    (multiplicative weights, i.e. natural policy gradient on the softmax, whose mean-field limit is the
    replicator dynamics) the two decisions stay independent and the marginals obey
    $$\mathrm{logit}\,x \mathrel{+}= \eta\,\pi\,G(p),\qquad
      \mathrm{logit}\,p \mathrel{+}= \eta\,(1-\pi)\,x\,o\,E_{K\sim\mathrm{Bin}(n-1,\,o p)}[\Delta(K)]$$
    with $\pi$ the opportunity probability and $G$ = :func:`offender_gain` (Proposition 7). With
    ``natural=False`` each decision has its own logit trained by vanilla policy gradient (REINFORCE's
    expected update), which scales each step by $x(1-x)$ resp. $p(1-p)$: the same rest points, but a rare
    behaviour learns slowly. Returns the trajectory (columns ``step, x, p``).
    """
    params(**kw)
    pi = 1.0 / (n + 1) if opportunity is None else opportunity
    lx, lp = math.log(x0 / (1 - x0)), math.log(p0 / (1 - p0))
    rows = []
    for t in range(steps + 1):
        x, p = 1 / (1 + math.exp(-lx)), 1 / (1 + math.exp(-lp))
        rows.append({"step": t, "x": x, "p": p})
        if t == steps:
            break
        gx = pi * offender_gain(p, n, **kw)
        gp = (1 - pi) * x * params(**kw)["o"] * report_gain(p, n, **kw)
        if not natural:
            gx, gp = x * (1 - x) * gx, p * (1 - p) * gp
        lx = float(np.clip(lx + lr * gx, -50, 50))
        lp = float(np.clip(lp + lr * gp, -50, 50))
    return pd.DataFrame(rows)


def basin_of_deterrence(n: int, *, x0: float = 0.5, p0s: Sequence[float] | None = None, steps: int = 600,
                        lr: float = 1.0, natural: bool = True, opportunity: float | None = None, **kw: Any) -> float:
    """Share of initial report propensities ``p0`` (with initial violation propensity ``x0``) from which the
    whole-game dynamics (:func:`mean_field`) end with the violation deterred ($x < 1/2$)."""
    p0s = np.linspace(0.02, 0.98, 25) if p0s is None else p0s
    return float(np.mean([mean_field(x0, float(p0), n, steps=steps, lr=lr, natural=natural, opportunity=opportunity,
                                     **kw)["x"].iloc[-1] < 0.5 for p0 in p0s]))


def team_game(n: int = 2, **kw: Any) -> NormalFormGame:
    """The whole game in normal form for a team of ``n + 1`` members, one of whom (uniformly at random)
    holds the opportunity to violate. Strategies: :data:`TEAM_STRATEGIES` (violate if you can? report if
    you witness a violation?). Payoffs are expected over the opportunity and who witnesses. Outcomes:
    ``violation`` (P a violation occurs) and ``outcome_value`` (P no violation stands), as the synthetic
    team domain's ground truth."""
    p = params(**kw)
    R, d, o, rho, q = p["R"], p["delta"], p["o"], p["rho"], p["q"]
    players = [f"worker_{i + 1}" for i in range(n + 1)]
    strategies = {pl: list(TEAM_STRATEGIES) for pl in players}
    clean = R - d
    # a witness's expected payoff by its action and the number r of *other* teammates of the offender who report
    witness = {a: [sum(binom.pmf(k, r, o) * (o * payoff(a, k, **kw) + (1 - o) * payoff("unaware", k, **kw))
                       for k in range(r + 1)) for r in range(n)] for a in ("silent", "report")}

    def offender(r: int) -> float:  # violating, with r teammates who report what they witness
        unreported = (1 - o * rho) ** r
        caught = 1 - (1 - q) * unreported
        return R + p["beta"] - (d + p["P"]) * caught - p["c"] * q * unreported

    def pay(names: dict[str, str]) -> dict[str, float]:
        out = {pl: 0.0 for pl in players}
        for holder in players:  # each member holds the opportunity with probability 1/(n+1)
            if not names[holder].startswith("violate"):
                for pl in players:
                    out[pl] += clean / len(players)
                continue
            reporters = [pl for pl in players if pl != holder and names[pl].endswith("report")]
            for pl in players:
                if pl == holder:
                    v = offender(len(reporters))
                else:
                    act = "report" if names[pl].endswith("report") else "silent"
                    v = witness[act][len([r for r in reporters if r != pl])]
                out[pl] += v / len(players)
        return out

    def violation(names: dict[str, str]) -> float:
        return sum(v.startswith("violate") for v in names.values()) / len(players)

    def outcome_value(names: dict[str, str]) -> float:
        total = 0.0
        for holder in players:
            if names[holder].startswith("violate"):
                r = sum(names[pl].endswith("report") for pl in players if pl != holder)
                total += 1 - (1 - q) * (1 - o * rho) ** r  # caught and reverted
            else:
                total += 1.0
        return total / len(players)

    return game_from_function(players, strategies, pay, outcomes={"violation": violation, "outcome_value": outcome_value},
                              name=f"team_game(n={n + 1})")


def team_mix(x: float, p: float) -> np.ndarray:
    """A mixed strategy of :func:`team_game` with independent decisions: violate w.p. ``x``, report w.p. ``p``."""
    return np.array([(1 - x) * (1 - p), (1 - x) * p, x * (1 - p), x * p])


def team_marginals(mix: Sequence[float]) -> tuple[float, float]:
    """(P violate, P report) of a mixed strategy of :func:`team_game`."""
    m = np.asarray(mix, float)
    return float(m[2] + m[3]), float(m[1] + m[3])


# ------------------------------------------------------------------------------------------------
# Sweeps
# ------------------------------------------------------------------------------------------------


def sweep(ratios: Sequence[float], ns: Sequence[int] = (2, 3, 5, 10), *, delta: float = 1.0, **kw: Any) -> pd.DataFrame:
    """Equilibrium structure as the bounty-to-stake ratio s/delta varies (other parameters as keywords).

    Columns: n, ratio, silent_eq (is all-silent an equilibrium), report_eq, report_dominant,
    p_threshold (the interior equilibrium bounding the silent basin), silent_basin (fraction of
    initial report propensities from which dynamics converge to silence), regime, and the bounties
    b_R / b_D (in units of delta) above which everyone reporting is an equilibrium / reporting is dominant.
    """
    rows = []
    for n in ns:
        base = dict(kw, R=kw.get("R", 1.0), delta=delta)
        b_r, b_d = report_equilibrium_bounty(n, **base) / delta, dominance_bounty(n, **base) / delta
        for r in ratios:
            p = dict(base, s=r * delta)
            eqs = symmetric_equilibria(n, **p)
            interior = [e["p_report"] for e in eqs if e["kind"] == "mixed" and not e["stable"]]
            rows.append({
                "n": n, "ratio": r,
                "silent_eq": any(e["p_report"] == 0.0 for e in eqs),
                "report_eq": any(e["p_report"] == 1.0 for e in eqs),
                "report_dominant": all(advantage(k, **p) > 0 for k in range(n)),
                "p_threshold": float(min(interior)) if interior else math.nan,
                "silent_basin": silent_basin(n, **p),
                "regime": regime(n, **p),
                "b_R": b_r, "b_D": b_d,
            })
    return pd.DataFrame(rows)


def regime_map(bounties: Sequence[float], observe: Sequence[float], n: int = 2, **kw: Any) -> pd.DataFrame:
    """:func:`regime` (and $b_R$, the dominance bounty and the silent basin) over a grid of bounty ``s``
    and observation probability ``o`` - where partial observation makes small bounties useless."""
    rows = []
    for o in observe:
        base = dict(kw, o=float(o))
        b_r, b_d = report_equilibrium_bounty(n, **base), dominance_bounty(n, **base)
        for s in bounties:
            p = dict(base, s=float(s))
            rows.append({"o": float(o), "s": float(s), "regime": regime(n, **p), "silent_basin": silent_basin(n, **p),
                         "b_R": b_r, "b_D": b_d})
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULTS", "TEAM_STRATEGIES", "params", "bounty_share", "payoff", "advantage", "other_reporters", "report_gain",
    "whistleblower_game", "symmetric_equilibria", "mixed_threshold_each", "dominance_bounty",
    "report_equilibrium_bounty", "silent_basin", "regime", "risk_dominant", "detection_prob", "offender_gain",
    "mean_field", "basin_of_deterrence", "team_game", "team_mix", "team_marginals", "sweep", "regime_map",
]
