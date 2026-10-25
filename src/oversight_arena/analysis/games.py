"""Empirical games (meta-games) over sampled strategies, and equilibrium analysis.

For multi-agent mechanisms incentive compatibility is an *equilibrium* property: honest
play should be an equilibrium (ideally the only reachable one), and no dishonest profile
should be. Given a finite set of strategies per role (sampled responses, prompts, optimised
policies), the mechanism induces a normal-form game with payoff tensors $u_i(s_1,\\dots,s_n)$
(mean rewards) and ground-truth tensors $g_i(s)$. This module estimates equilibria and the
ground truth *at* equilibrium, deviation incentives (regret/exploitability) of any profile,
and learning dynamics (replicator / multiplicative weights ≈ natural-policy-gradient training).
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linprog

Mix = dict[str, np.ndarray]


@dataclass
class Equilibrium:
    mix: Mix
    payoffs: dict[str, float]
    gt: dict[str, float]
    regret: float
    kind: str = "nash"

    def support(self, game: "EmpiricalGame", tol: float = 1e-6) -> dict[str, dict[str, float]]:
        return {
            r: {s: float(p) for s, p in zip(game.strategies[r], self.mix[r]) if p > tol}
            for r in game.roles
        }


class EmpiricalGame:
    """Normal-form game estimated from episodes.

    Args:
        roles: players (roles whose strategies vary).
        strategies: per role, the list of strategy labels (axis order).
        payoffs: per *trainable* role, payoff tensor of shape (|S_1|, ..., |S_n|).
        gt: per metric name → tensor (e.g. ``"correct:debater_a"`` or ``"_outcome"``).
        counts: number of episodes per cell.
    """

    def __init__(
        self,
        roles: list[str],
        strategies: dict[str, list[str]],
        payoffs: dict[str, np.ndarray],
        gt: dict[str, np.ndarray] | None = None,
        counts: np.ndarray | None = None,
    ):
        self.roles = roles
        self.strategies = strategies
        self.payoffs = payoffs
        self.gt = gt or {}
        self.shape = tuple(len(strategies[r]) for r in roles)
        self.counts = counts if counts is not None else np.ones(self.shape)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_results(
        cls,
        results,
        roles: Sequence[str],
        strategy_col: str = "strategy_name",
        payoff_roles: Sequence[str] | None = None,
        gt_metrics: Sequence[str] = ("correct",),
        task: str | None = None,
        mechanism: str | None = None,
        reward: Any = None,
    ) -> "EmpiricalGame":
        """Estimate the game from episodes: cell = joint strategy labels of ``roles``; payoffs =
        mean rewards; GT tensors = mean GT per ``gt_metrics`` scorer (keys ``"<scorer>:<role>"``).

        ``reward``: evaluate payoffs under an alternative (non-batch) reward rule instead of the
        recorded rewards — exact for programmatic/fixed-policy agents whose behaviour does not
        depend on the stated incentives, and cheap (no episodes are re-run).
        """
        recs = [r for r in (results.records if hasattr(results, "records") else results) if r.error is None]
        if task is not None:
            recs = [r for r in recs if r.task_id == task]
        from ..experiment.results import Results

        names = Results(recs).mechanism_names()
        if mechanism is not None:  # a unique name, a display name shared by one configuration, or a hash
            recs = [r for r in recs if mechanism in (names[(r.mechanism, r.mechanism_hash)], r.mechanism_hash)
                    or (r.mechanism == mechanism and names[(r.mechanism, r.mechanism_hash)] == r.mechanism)]
        if len({r.mechanism_hash for r in recs}) > 1:
            raise ValueError("these episodes come from several mechanism configurations "
                             f"({sorted({names[(r.mechanism, r.mechanism_hash)] for r in recs})}); pass mechanism=... "
                             "so that different games are not pooled into one")
        if reward is not None:
            recs = [r.model_copy(update={"rewards": {k: float(v) for k, v in reward(r).items()}}) for r in recs]
        roles = list(roles)
        key_attr = {"strategy_name": "strategy_name", "strategy": "strategy_id"}.get(strategy_col, strategy_col)
        strat: dict[str, list[str]] = {r: [] for r in roles}
        cells: dict[tuple, list] = {}
        for rec in recs:
            key = []
            for r in roles:
                b = rec.bound.get(r)
                lab = getattr(b, key_attr) if b is not None else "default"
                if lab not in strat[r]:
                    strat[r].append(lab)
                key.append(lab)
            cells.setdefault(tuple(key), []).append(rec)
        for r in roles:
            strat[r] = sorted(strat[r])
        pr = list(payoff_roles) if payoff_roles is not None else sorted({x for rec in recs for x in rec.rewards})
        shape = tuple(len(strat[r]) for r in roles)
        pay = {r: np.full(shape, np.nan) for r in pr}
        gts: dict[str, np.ndarray] = {}
        counts = np.zeros(shape)
        for key, rs in cells.items():
            idx = tuple(strat[r].index(k) for r, k in zip(roles, key))
            counts[idx] = len(rs)
            for r in pr:
                vals = [x.rewards[r] for x in rs if r in x.rewards]
                if vals:
                    pay[r][idx] = float(np.mean(vals))
            for m in gt_metrics:
                names = sorted({k for x in rs for k in x.gt.get(m, {})})
                for who in names:
                    vals = [x.gt[m][who] for x in rs if x.gt.get(m, {}).get(who) is not None]
                    if vals:
                        gts.setdefault(f"{m}:{who}", np.full(shape, np.nan))[idx] = float(np.mean(vals))
        return cls(roles, strat, pay, gts, counts)

    @classmethod
    def from_matrices(cls, A: np.ndarray, B: np.ndarray | None = None, row: str = "row", col: str = "col",
                      row_strats: list[str] | None = None, col_strats: list[str] | None = None,
                      gt: dict[str, np.ndarray] | None = None) -> "EmpiricalGame":
        A = np.asarray(A, float)
        B = -A if B is None else np.asarray(B, float)
        rs = row_strats or [f"r{i}" for i in range(A.shape[0])]
        cs = col_strats or [f"c{j}" for j in range(A.shape[1])]
        return cls([row, col], {row: rs, col: cs}, {row: A, col: B}, gt)

    # ------------------------------------------------------------------ evaluation
    @property
    def complete(self) -> bool:
        return all(not np.isnan(p).any() for p in self.payoffs.values())

    def uniform(self) -> Mix:
        return {r: np.full(len(self.strategies[r]), 1 / len(self.strategies[r])) for r in self.roles}

    def mix(self, probs: "dict[str, float] | dict[str, dict[str, float]]") -> Mix:
        """Build a mixed profile from strategy *names* (axis order is sorted, so never build mixes
        by position). ``probs`` maps strategy → probability (applied to every role that has
        those strategies) or role → {strategy: probability}; unnormalised weights are rescaled."""
        out = {}
        per_role = all(isinstance(v, dict) for v in probs.values())
        for r in self.roles:
            d = probs.get(r, {}) if per_role else probs
            v = np.array([float(d.get(s, 0.0)) for s in self.strategies[r]])  # type: ignore[union-attr]
            if v.sum() <= 0:
                if r in self.payoffs:
                    raise ValueError(f"no probability mass on {r}'s strategies {self.strategies[r]}")
                v = np.ones(len(v))  # fixtures not mentioned: uniform over their (fixed) strategies
            out[r] = v / v.sum()
        return out

    def marginal(self, mix: Mix, role: str, strategies: Sequence[str]) -> float:
        """Total probability ``role``'s mix puts on the named strategies."""
        return float(sum(mix[role][self.strategies[role].index(s)] for s in strategies))

    def pure(self, profile: dict[str, str | int]) -> Mix:
        mix = {}
        for r in self.roles:
            v = np.zeros(len(self.strategies[r]))
            s = profile[r]
            v[self.strategies[r].index(s) if isinstance(s, str) else s] = 1.0
            mix[r] = v
        return mix

    def fill_missing(self, value: float = 0.0) -> "EmpiricalGame":
        """A copy with unobserved payoff cells set to ``value`` (an explicit modelling choice)."""
        return EmpiricalGame(self.roles, self.strategies, {r: np.nan_to_num(P, nan=value) for r, P in self.payoffs.items()},
                             self.gt, self.counts)

    def _contract(self, T: np.ndarray, mix: Mix, skip: str | None = None, allow_nan: bool = False) -> np.ndarray | float:
        if np.isnan(T).any():
            if not allow_nan:
                raise ValueError(
                    "payoff tensor has unobserved cells (no episodes for some strategy profiles); run the missing "
                    "profiles, restrict the strategy sets, or call .fill_missing(value) explicitly")
            T = np.nan_to_num(T, nan=0.0)
        out = T
        # contract from the last axis to keep indices valid
        for ax in reversed(range(len(self.roles))):
            r = self.roles[ax]
            if r == skip:
                continue
            out = np.tensordot(out, mix[r], axes=([ax], [0]))
        return out

    def value(self, T: np.ndarray, mix: Mix) -> float:
        return float(self._contract(T, mix))

    def expected_payoffs(self, mix: Mix) -> dict[str, float]:
        return {r: self.value(P, mix) for r, P in self.payoffs.items()}

    def expected_gt(self, mix: Mix) -> dict[str, float]:
        """Expected GT under ``mix``, averaging only over cells where the GT is defined."""
        out = {}
        for k, T in self.gt.items():
            if np.isnan(T).all():
                continue
            mask = ~np.isnan(T)
            w = self._contract(mask.astype(float), mix)
            num = self._contract(T, mix, allow_nan=True)
            out[k] = float(num) / float(w) if float(w) > 0 else float("nan")
        return out

    def deviation_payoffs(self, role: str, mix: Mix) -> np.ndarray:
        if role not in self.payoffs:
            raise KeyError(f"{role} has no payoffs (fixture?)")
        return np.asarray(self._contract(self.payoffs[role], mix, skip=role))

    def regret(self, mix: Mix) -> dict[str, float]:
        """Per role: gain from the best unilateral deviation (0 at a Nash equilibrium)."""
        out = {}
        for r in self.roles:
            if r not in self.payoffs:
                continue
            dev = self.deviation_payoffs(r, mix)
            out[r] = float(np.max(dev) - np.dot(dev, mix[r]))
        return out

    def coalition_gain(self, mix: Mix, coalition: Sequence[str] | None = None) -> tuple[float, dict[str, str]]:
        """Largest gain in the coalition's *total* payoff from a joint pure deviation (others
        fixed at ``mix``), and the deviation achieving it. Nash equilibria only rule out
        unilateral deviations; with non-zero-sum rewards (e.g. log-score debate) copies of one
        model can profit *jointly* — e.g. both debaters keeping the judge at 50/50 — which is the
        collusion risk of self-play with shared weights. Default coalition: all payoff roles."""
        coal = [r for r in (coalition or [r for r in self.roles if r in self.payoffs])]
        base = sum(self.expected_payoffs(mix)[r] for r in coal)
        best, arg = 0.0, {}
        for combo in itertools.product(*[range(len(self.strategies[r])) for r in coal]):
            m = dict(mix)
            m.update({r: np.eye(len(self.strategies[r]))[i] for r, i in zip(coal, combo)})
            tot = sum(self.value(self.payoffs[r], m) for r in coal)
            if tot - base > best + 1e-12:
                best, arg = tot - base, {r: self.strategies[r][i] for r, i in zip(coal, combo)}
        return float(best), arg

    def exploitability(self, mix: Mix) -> float:
        reg = self.regret(mix)
        return float(sum(reg.values())) if reg else 0.0

    def best_response(self, role: str, mix: Mix) -> tuple[str, float]:
        dev = self.deviation_payoffs(role, mix)
        i = int(np.argmax(dev))
        return self.strategies[role][i], float(dev[i])

    # ------------------------------------------------------------------ equilibria
    def pure_nash(self, tol: float = 1e-9) -> list[Equilibrium]:
        eqs = []
        for idx in itertools.product(*[range(n) for n in self.shape]):
            mix = self.pure({r: i for r, i in zip(self.roles, idx)})
            reg = self.regret(mix)
            if all(v <= tol for v in reg.values()):
                eqs.append(self._eq(mix, "pure_nash"))
        return eqs

    def _eq(self, mix: Mix, kind: str) -> Equilibrium:
        return Equilibrium(mix, self.expected_payoffs(mix), self.expected_gt(mix), self.exploitability(mix), kind)

    def nash(self, max_support: int | None = None, starts: int = 16, seed: int = 0) -> list[Equilibrium]:
        """Nash equilibria: exact support enumeration for 2 trainable players, else pure NE
        plus approximate mixed equilibria from replicator dynamics with several starts."""
        trainable = [r for r in self.roles if r in self.payoffs]
        if len(self.roles) == 2 and len(trainable) == 2 and self.complete:
            return self._support_enumeration(max_support)
        eqs = self.pure_nash()
        rng = np.random.default_rng(seed)
        for k in range(starts):
            x0 = self.uniform() if k == 0 else {r: rng.dirichlet(np.ones(len(self.strategies[r]))) for r in self.roles}
            traj = self.replicator(x0, iters=3000, lr=0.5)
            mix = traj[-1]
            if self.exploitability(mix) < 1e-3:
                eq = self._eq(mix, "replicator")
                if not any(_close(eq.mix, e.mix) for e in eqs):
                    eqs.append(eq)
        return eqs

    def _support_enumeration(self, max_support: int | None = None, max_pairs: int = 20000) -> list[Equilibrium]:
        """Support enumeration. Equal-size supports are solved exactly (enough for nondegenerate
        games); in degenerate games equilibria can have supports of different sizes, so every
        pair of supports is also tried with a linear program (up to ``max_pairs`` pairs)."""
        r0, r1 = self.roles
        A, B = self.payoffs[r0], self.payoffs[r1]
        m, n = A.shape
        eqs: list[Equilibrium] = []

        def add(xf: np.ndarray, yf: np.ndarray) -> None:
            mix = {r0: xf, r1: yf}
            if self.exploitability(mix) < 1e-7 and not any(_close(mix, e.mix) for e in eqs):
                eqs.append(self._eq(mix, "nash"))

        km, kn = (m, n) if max_support is None else (min(max_support, m), min(max_support, n))
        for k in range(1, min(km, kn) + 1):
            for I in itertools.combinations(range(m), k):
                for J in itertools.combinations(range(n), k):
                    y = _solve_indiff(A[np.ix_(I, J)])
                    x = _solve_indiff(B[np.ix_(I, J)].T)
                    if x is None or y is None:
                        continue
                    xf, yf = np.zeros(m), np.zeros(n)
                    xf[list(I)], yf[list(J)] = x, y
                    add(xf, yf)
        supports = lambda size, kmax: [c for k in range(1, kmax + 1) for c in itertools.combinations(range(size), k)]  # noqa: E731
        rows, cols = supports(m, km), supports(n, kn)
        if len(rows) * len(cols) <= max_pairs:
            for I in rows:
                for J in cols:
                    if len(I) == len(J):
                        continue
                    y = _support_lp(A, I, J)  # column mix on J making every row in I a best response
                    x = _support_lp(B.T, J, I) if y is not None else None
                    if x is not None:
                        xf, yf = np.zeros(m), np.zeros(n)
                        xf[list(I)], yf[list(J)] = x, y
                        add(xf, yf)
        return eqs

    def zero_sum_value(self, role: str | None = None) -> tuple[float, Mix]:
        """Maximin value (for ``role``, default the first) and optimal strategies of a 2-player
        game treated as zero-sum in ``role``'s payoff, via LP."""
        r0, r1 = self.roles
        me = role or r0
        other = r1 if me == r0 else r0
        A = self.payoffs[me] if me == r0 else self.payoffs[me].T  # rows = my strategies
        if np.isnan(A).any():
            raise ValueError("payoff matrix has unobserved cells")
        m, n = A.shape
        # row: maximise v s.t. x^T A >= v, sum x = 1
        c = np.zeros(m + 1)
        c[-1] = -1
        A_ub = np.hstack([-A.T, np.ones((n, 1))])
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n), A_eq=[np.r_[np.ones(m), 0]], b_eq=[1],
                      bounds=[(0, None)] * m + [(None, None)])
        x = res.x[:m]
        c2 = np.zeros(n + 1)
        c2[-1] = 1
        A_ub2 = np.hstack([A, -np.ones((m, 1))])
        res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(m), A_eq=[np.r_[np.ones(n), 0]], b_eq=[1],
                       bounds=[(0, None)] * n + [(None, None)])
        y = res2.x[:n]
        return float(-res.fun), {me: x, other: y}

    # ------------------------------------------------------------------ learning dynamics
    def replicator(self, x0: Mix | None = None, iters: int = 1000, lr: float = 0.1, record_every: int = 0) -> list[Mix]:
        """Discrete-time multiplicative-weights (exponential replicator) dynamics.

        This is the mean-field limit of independent softmax learners trained by *natural* policy
        gradient / exponential weights (vanilla REINFORCE has the same rest points but slows
        down rare strategies by a factor of their probability) — a cheap model of *training*
        under the mechanism. Fixture roles (no payoffs) stay fixed at ``x0``.
        """
        x = {r: np.array(v, dtype=float) for r, v in (x0 or self.uniform()).items()}
        traj = [{r: v.copy() for r, v in x.items()}]
        for t in range(iters):
            new = {}
            for r in self.roles:
                if r not in self.payoffs:
                    new[r] = x[r]
                    continue
                dev = self.deviation_payoffs(r, x)
                z = np.log(np.maximum(x[r], 1e-300)) + lr * (dev - dev.max())
                z = np.exp(z - z.max())
                new[r] = z / z.sum()
            x = new
            if record_every and (t + 1) % record_every == 0:
                traj.append({r: v.copy() for r, v in x.items()})
        if not record_every or len(traj) == 1 or not _close(traj[-1], x):
            traj.append({r: v.copy() for r, v in x.items()})
        return traj

    def fictitious_play(self, iters: int = 2000) -> Mix:
        counts = {r: np.ones(len(self.strategies[r])) for r in self.roles}
        for _ in range(iters):
            mix = {r: c / c.sum() for r, c in counts.items()}
            for r in self.roles:
                if r in self.payoffs:
                    counts[r][int(np.argmax(self.deviation_payoffs(r, mix)))] += 1
        return {r: c / c.sum() for r, c in counts.items()}

    def basins(self, grid: int = 21, iters: int = 400, lr: float = 0.5) -> pd.DataFrame:
        """For 2x2-style games: where replicator dynamics end up from a grid of initial mixes.

        Returns the final expected payoffs/GT for each starting point (probability of each
        role's *first* strategy on the grid)."""
        if len(self.roles) != 2:
            raise ValueError("basins() supports two roles")
        r0, r1 = self.roles
        rows = []
        for p in np.linspace(0.01, 0.99, grid):
            for q in np.linspace(0.01, 0.99, grid):
                x0 = {r0: _two_or_more(p, len(self.strategies[r0])), r1: _two_or_more(q, len(self.strategies[r1]))}
                fin = self.replicator(x0, iters=iters, lr=lr)[-1]
                row = {"p0": p, "q0": q, **{f"final_{r}_{s}": float(fin[r][i]) for r in self.roles for i, s in enumerate(self.strategies[r])}}
                row.update({f"gt[{k}]": v for k, v in self.expected_gt(fin).items()})
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ reporting
    def table(self, role: str | None = None) -> pd.DataFrame:
        """Payoff table as a DataFrame (2-player: matrix of 'u0, u1' strings)."""
        if len(self.roles) == 2:
            r0, r1 = self.roles
            data = {}
            for j, s1 in enumerate(self.strategies[r1]):
                col = []
                for i, _ in enumerate(self.strategies[r0]):
                    vals = [self.payoffs[r][i, j] for r in (role and [role] or [r0, r1]) if r in self.payoffs]
                    col.append(", ".join(f"{v:.3f}" for v in vals))
                data[s1] = col
            return pd.DataFrame(data, index=self.strategies[r0])
        rows = []
        for idx in itertools.product(*[range(n) for n in self.shape]):
            row = {r: self.strategies[r][i] for r, i in zip(self.roles, idx)}
            row.update({f"u[{r}]": P[idx] for r, P in self.payoffs.items()})
            row.update({f"gt[{k}]": T[idx] for k, T in self.gt.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def outcomes(self, eqs: Sequence[Equilibrium] | None = None, digits: int = 3,
                 keys: Sequence[str] | None = None) -> pd.DataFrame:
        """Distinct equilibrium *outcomes*: many games have families of payoff-equivalent equilibria
        (e.g. whether an honest worker would report is irrelevant when nobody misbehaves); what
        matters for oversight is the range of outcomes they induce. Equilibria are identified by
        their payoffs and GT rounded to ``digits`` (or only by the GT metrics in ``keys``)."""
        eqs = self.nash() if eqs is None else list(eqs)
        rows, seen = [], set()
        for e in eqs:
            vals = [e.gt.get(k, float("nan")) for k in keys] if keys is not None else list(e.payoffs.values()) + list(e.gt.values())
            key = tuple("nan" if v != v else round(v, digits) for v in vals)  # NaN != NaN: use a sentinel
            if key in seen:
                continue
            seen.add(key)
            sup = e.support(self)
            rows.append({
                "kind": e.kind,
                "support": "; ".join(f"{r}: " + ", ".join(f"{s} {p:.2f}" for s, p in d.items()) for r, d in sup.items()),
                **{f"u[{r}]": v for r, v in e.payoffs.items()},
                **{f"gt[{k}]": v for k, v in e.gt.items()},
            })
        return pd.DataFrame(rows)

    def summary(self, reference: dict[str, str] | None = None) -> dict[str, Any]:
        """Equilibria with their GT; optionally the regret of a reference (e.g. honest) profile."""
        eqs = self.nash()
        out: dict[str, Any] = {
            "equilibria": [
                {"support": e.support(self), "payoffs": e.payoffs, "gt": e.gt, "kind": e.kind} for e in eqs
            ]
        }
        if reference is not None:
            mix = self.pure(reference)
            out["reference"] = reference
            out["reference_regret"] = self.regret(mix)
            out["reference_gt"] = self.expected_gt(mix)
            out["reference_is_nash"] = all(v <= 1e-9 for v in out["reference_regret"].values())
        return out


def _two_or_more(p: float, k: int) -> np.ndarray:
    if k == 1:
        return np.array([1.0])
    rest = (1 - p) / (k - 1)
    return np.array([p] + [rest] * (k - 1))


def _support_lp(M: np.ndarray, I: Sequence[int], J: Sequence[int]) -> np.ndarray | None:
    """Opponent mix ``y`` with support exactly ``J`` under which every row in ``I`` is a best
    response for the player with payoffs ``M`` (rows = own strategies): maximise the smallest
    probability on ``J`` subject to indifference on ``I`` and no better row outside it."""
    m = M.shape[0]
    k = len(J)
    # variables: y_J (k), u (free), t
    c = np.zeros(k + 2)
    c[-1] = -1.0
    A_eq = [np.r_[np.ones(k), 0.0, 0.0]] + [np.r_[M[i, list(J)], -1.0, 0.0] for i in I]
    b_eq = [1.0] + [0.0] * len(I)
    A_ub = [np.r_[M[i, list(J)], -1.0, 0.0] for i in range(m) if i not in I]
    A_ub += [np.r_[-np.eye(k)[j], 0.0, 1.0] for j in range(k)]  # t <= y_j
    res = linprog(c, A_ub=np.array(A_ub), b_ub=np.zeros(len(A_ub)), A_eq=np.array(A_eq), b_eq=b_eq,
                  bounds=[(0, None)] * k + [(None, None), (0, 1)], method="highs")
    if res.status != 0 or res.x[-1] <= 1e-9:
        return None
    y = np.clip(res.x[:k], 0, None)
    return y / y.sum()


def _solve_indiff(M: np.ndarray) -> np.ndarray | None:
    """Solve for a mix y (sum 1, y>=0) making all rows of M y equal."""
    k = M.shape[0]
    Asys = np.zeros((k + 1, M.shape[1] + 1))
    Asys[:k, :-1] = M
    Asys[:k, -1] = -1
    Asys[k, :-1] = 1
    b = np.zeros(k + 1)
    b[k] = 1
    try:
        sol, *_ = np.linalg.lstsq(Asys, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.allclose(Asys @ sol, b, atol=1e-8):
        return None
    y = sol[:-1]
    if np.any(y < -1e-9):
        return None
    y = np.clip(y, 0, None)
    return y / y.sum()


def _close(a: Mix, b: Mix, tol: float = 1e-3) -> bool:
    return all(np.allclose(a[r], b[r], atol=tol) for r in a)
