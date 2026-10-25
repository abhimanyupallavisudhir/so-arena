"""Normal-form games (empirical or analytic) and their equilibria.

Multi-agent mechanisms are games: whether a behaviour is incentivized depends on what the other
agents do. Empirical game-theoretic analysis (EGTA) estimates a normal-form game over a finite set
of strategies (prompts, policies, checkpoints) by simulation - see
:class:`so_arena.games.egta.EmpiricalGameExperiment` - and then asks which profiles are equilibria,
how robust they are, and how good they are by *ground truth* (not by the mechanism's own rewards).

Payoffs are stored as one tensor per player, indexed by the players' strategy indices. Additional
per-profile statistics (``outcomes``; e.g. ground-truth welfare, violation rate, judge accuracy) let
equilibria be evaluated by what actually matters; ``coverage`` records, per outcome and profile, the
fraction of the profile's episodes that had a value (ground truth may be missing or pending), so an
expectation over a mixed profile can say how much of its mass was measured (:meth:`expected_outcome`).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import linprog

Profile = tuple[int, ...]
Mixed = list[np.ndarray]


class NormalFormGame:
    def __init__(self, players: Sequence[str], strategies: dict[str, Sequence[str]], payoffs: dict[str, np.ndarray],
                 *, stderr: dict[str, np.ndarray] | None = None, counts: np.ndarray | None = None,
                 outcomes: dict[str, np.ndarray] | None = None, name: str = "game",
                 coverage: dict[str, np.ndarray] | None = None):
        self.players = list(players)
        self.strategies = {p: list(strategies[p]) for p in self.players}
        self.shape = tuple(len(self.strategies[p]) for p in self.players)
        self.payoffs = {p: np.asarray(payoffs[p], dtype=float).reshape(self.shape) for p in self.players}
        self.stderr = {p: np.asarray(v, dtype=float).reshape(self.shape) for p, v in (stderr or {}).items()}
        self.counts = counts
        self.outcomes = {k: np.asarray(v, dtype=float).reshape(self.shape) for k, v in (outcomes or {}).items()}
        # per outcome: fraction of each profile's episodes with a value (default: 1 where the outcome is finite)
        self.coverage = {k: np.asarray(v, dtype=float).reshape(self.shape) for k, v in (coverage or {}).items()}
        self.name = name

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return len(self.players)

    def profiles(self):
        return itertools.product(*[range(s) for s in self.shape])

    def profile_names(self, prof: Profile) -> dict[str, str]:
        return {p: self.strategies[p][i] for p, i in zip(self.players, prof)}

    def index(self, names: dict[str, str]) -> Profile:
        return tuple(self.strategies[p].index(names[p]) for p in self.players)

    def uniform(self) -> Mixed:
        return [np.full(s, 1.0 / s) for s in self.shape]

    def n_missing(self) -> int:
        """Number of missing (NaN) payoff entries - e.g. profiles whose episodes all errored."""
        return int(sum(np.isnan(v).sum() for v in self.payoffs.values()))

    def imputed(self, *, warn: bool = True) -> "NormalFormGame":
        """A copy whose missing payoffs are filled conservatively: each with that player's lowest observed payoff.

        Solvers need every entry, but a fill must not favour the unobserved: 0, for instance, is the
        best possible log-score reward and would make a never-observed profile the equilibrium. With
        the minimum, no player prefers a profile for its missing entries. Logs a warning when anything
        was filled; raises ``ValueError`` if a player has no observed payoff at all.
        """
        missing = self.n_missing()
        if not missing:
            return self
        payoffs = {}
        for p, v in self.payoffs.items():
            if np.isnan(v).all():
                raise ValueError(f"{self.name}: no observed payoff for player {p!r} (every profile errored?)")
            payoffs[p] = np.where(np.isnan(v), np.nanmin(v), v)
        if warn:
            import logging

            logging.getLogger("so_arena").warning(
                "%s: %d missing payoff entries filled with each player's lowest observed payoff", self.name, missing)
        return NormalFormGame(self.players, self.strategies, payoffs, stderr=self.stderr, counts=self.counts,
                              outcomes=self.outcomes, name=self.name, coverage=self.coverage)

    def pure(self, prof: Profile) -> Mixed:
        out = []
        for s, i in zip(self.shape, prof):
            x = np.zeros(s)
            x[i] = 1.0
            out.append(x)
        return out

    def _contract(self, T: np.ndarray, mixed: Mixed, skip: int | None = None) -> np.ndarray | float:
        out = T
        for b in reversed(range(self.n)):
            if b != skip:
                out = np.tensordot(out, mixed[b], axes=([b], [0]))
        return out

    def expected(self, mixed: Mixed, key: str | None = None, player: str | None = None) -> float:
        T = self.payoffs[player] if player is not None else self.outcomes[key]  # type: ignore[index]
        return float(self._contract(T, mixed))

    def expected_outcome(self, mixed: Mixed, key: str, *, min_coverage: float = 0.5,
                         warn: bool = True) -> tuple[float, float]:
        r"""(expectation, coverage) of outcome ``key`` under independent mixed strategies.

        The expectation averages the profiles where the outcome was observed, renormalized; the coverage
        is the probability mass of measured episodes, $\sum_s x(s)\,c(s)$ with $c(s)$ the fraction of
        profile $s$'s episodes that had a value (:attr:`coverage`; 1 for every finite cell if not
        recorded). A value averaged over little of the mass describes other behaviour than the profile's,
        so below ``min_coverage`` the expectation is NaN (with a warning).
        """
        T = self.outcomes[key]
        joint = mixed[0]
        for x in mixed[1:]:
            joint = np.multiply.outer(joint, x)
        joint = np.asarray(joint, dtype=float).reshape(self.shape)
        finite = np.isfinite(T)
        cov_t = self.coverage.get(key)
        cov_t = np.where(finite, 1.0, 0.0) if cov_t is None else np.where(finite, np.nan_to_num(cov_t), 0.0)
        coverage = float((joint * cov_t).sum())
        mass = float(joint[finite].sum())
        value = float((joint[finite] * T[finite]).sum() / mass) if mass > 0 else math.nan
        if coverage < min_coverage:
            if warn and coverage > 0:
                import logging

                logging.getLogger("so_arena").warning(
                    "%s: %r measured on %.0f%% of the profile's mass (< %.0f%%): reported as NaN", self.name, key,
                    100 * coverage, 100 * min_coverage)
            value = math.nan
        return value, coverage

    def deviation_payoffs(self, player: str, mixed: Mixed) -> np.ndarray:
        a = self.players.index(player)
        return np.asarray(self._contract(self.payoffs[player], mixed, skip=a))

    def regrets(self, mixed: Mixed) -> dict[str, float]:
        out = {}
        for p in self.players:
            dev = self.deviation_payoffs(p, mixed)
            out[p] = float(dev.max() - dev @ mixed[self.players.index(p)])
        return out

    def nash_conv(self, mixed: Mixed) -> float:
        return float(sum(self.regrets(mixed).values()))

    def payoff_scale(self) -> float:
        """The largest payoff range of any player: the unit in which regrets are compared to tolerances."""
        return max(1e-12, max(float(np.nanmax(v) - np.nanmin(v)) if np.isfinite(v).any() else 0.0
                              for v in self.payoffs.values()))

    def _contract_pair(self, T: np.ndarray, mixed: Mixed, a: int, b: int) -> np.ndarray:
        """Contract all players but ``a`` and ``b``: the matrix indexed by (strategy of a, strategy of b)."""
        out = T
        for c in reversed(range(self.n)):
            if c not in (a, b):
                out = np.tensordot(out, mixed[c], axes=([c], [0]))
        return out if a < b else out.T

    def nash_lyapunov(self, mixed: Mixed, *, grad: bool = False) -> float | tuple[float, Mixed]:
        """$V(x) = \\sum_p \\sum_s \\max(0, u_p(s, x_{-p}) - u_p(x))^2$ on payoffs divided by
        :meth:`payoff_scale`: nonnegative and zero exactly at the Nash equilibria, and continuously
        differentiable - so equilibria of $n$-player games are the global minima of a smooth function
        (with ``grad=True`` also returns $\\partial V / \\partial x_p$ per player)."""
        scale = self.payoff_scale()
        value = 0.0
        g = [np.zeros(s) for s in self.shape]
        for p_i, p in enumerate(self.players):
            U = self.payoffs[p] / scale
            d = np.asarray(self._contract(U, mixed, skip=p_i))
            r = np.clip(d - d @ mixed[p_i], 0, None)
            value += float(r @ r)
            if not grad or not r.any():
                continue
            g[p_i] += -2 * r.sum() * d  # u_p(x) moves with x_p; each u_p(s, x_{-p}) does not
            for q in range(self.n):
                if q != p_i:
                    M = self._contract_pair(U, mixed, p_i, q)
                    g[q] += 2 * (r @ M - r.sum() * (mixed[p_i] @ M))
        return (value, g) if grad else value

    def _polish(self, x0: Mixed, groups: Sequence[Sequence[int]], maxiter: int = 500) -> Mixed:
        """Minimize :meth:`nash_lyapunov` from ``x0`` on the product of simplices (SLSQP); players in one
        of ``groups`` share one mixed strategy."""
        from scipy.optimize import minimize

        pops = [list(g) for g in groups] + [[a] for a in range(self.n) if not any(a in g for g in groups)]
        sizes = [self.shape[pop[0]] for pop in pops]
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(int)
        slices = [slice(int(s), int(s) + k) for s, k in zip(starts, sizes)]

        def unpack(z: np.ndarray) -> Mixed:
            out: list[np.ndarray] = [np.zeros(0)] * self.n
            for pop, sl in zip(pops, slices):
                v = np.clip(z[sl], 0, None)
                v = v / v.sum() if v.sum() > 0 else np.full(len(v), 1 / len(v))
                for a in pop:
                    out[a] = v
            return out

        def f(z: np.ndarray) -> tuple[float, np.ndarray]:
            val, g = self.nash_lyapunov(unpack(z), grad=True)  # type: ignore[misc]
            return val, np.concatenate([sum(g[a] for a in pop) for pop in pops])

        z0 = np.concatenate([np.mean([x0[a] for a in pop], axis=0) for pop in pops])
        cons = []
        for sl in slices:
            ind = np.zeros(len(z0))
            ind[sl] = 1.0
            cons.append({"type": "eq", "fun": lambda z, sl=sl: float(z[sl].sum() - 1.0), "jac": lambda z, ind=ind: ind})
        try:
            res = minimize(f, z0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * len(z0), constraints=cons,
                           options={"maxiter": maxiter, "ftol": 1e-20})
            z = res.x
        except (ValueError, np.linalg.LinAlgError):  # pragma: no cover - numerical failure: keep the start
            z = z0
        return unpack(z)

    def approximate_nash(self, x0: Mixed | None = None, *, restarts: int = 8, seed: int = 0, tol: float = 1e-4,
                         shared: Sequence[Sequence[str]] | None = None, steps: int = 3000) -> Mixed:
        """A Nash equilibrium of an $n$-player game, as nearly as can be found: the profile with the lowest
        NashConv among several searches, stopping at the first within ``tol`` x :meth:`payoff_scale`.

        Replicator dynamics alone need not converge for $n > 2$ (in three-player matching pennies they
        spiral out to the boundary, NashConv 2 on payoffs of $\\pm 1$), and neither need fictitious play
        (Jordan's counterexample). So the candidates are, in order: replicator dynamics from ``x0``
        (uniform by default: the previous behaviour, kept when it reaches an equilibrium), then minima of
        :meth:`nash_lyapunov` - whose zeros are exactly the equilibria - started from the replicator's end
        point, its time average, uniform, fictitious play and ``restarts`` random profiles. Check the
        result with :meth:`nash_conv`: it is an approximate equilibrium when that exceeds the tolerance.
        ``shared`` groups of exchangeable players get one mixed strategy (a symmetric equilibrium).
        """
        groups = [[self.players.index(p) for p in g] for g in (shared or []) if len(g) > 1]
        limit = tol * self.payoff_scale()
        best: Mixed | None = None
        best_nc = math.inf

        def consider(x: Mixed) -> bool:
            nonlocal best, best_nc
            nc = self.nash_conv(x)
            if nc < best_nc:
                best, best_nc = [np.asarray(v, dtype=float).copy() for v in x], nc
            return best_nc <= limit

        xr, traj = self.replicator(x0, steps=steps, shared=shared)
        if consider(xr):
            return best  # type: ignore[return-value]
        starts: list[Mixed] = [xr, [np.mean([t[a] for t in traj], axis=0) for a in range(self.n)], self.uniform(),
                               self.fictitious_play(min(steps, 2000))]
        rng = np.random.default_rng(seed)
        starts += [[rng.dirichlet(np.ones(s)) for s in self.shape] for _ in range(restarts)]
        for x in starts:
            if consider(self._polish(x, groups)):
                break
        return best  # type: ignore[return-value]

    def is_nash(self, mixed_or_profile, tol: float = 1e-9) -> bool:
        mixed = self.pure(mixed_or_profile) if isinstance(mixed_or_profile, tuple) else mixed_or_profile
        return all(r <= tol for r in self.regrets(mixed).values())

    def best_responses(self, player: str, mixed: Mixed, tol: float = 1e-9) -> list[str]:
        dev = self.deviation_payoffs(player, mixed)
        return [self.strategies[player][i] for i in np.flatnonzero(dev >= dev.max() - tol)]

    # ------------------------------------------------------------------ equilibria
    def pure_nash(self, tol: float = 1e-9) -> list[Profile]:
        return [prof for prof in self.profiles() if self.is_nash(prof, tol)]

    def strict_nash(self, tol: float = 1e-9) -> list[Profile]:
        """Pure profiles where every player's strategy is its *unique* best response (strict equilibria;
        these are the asymptotically stable rest points of replicator dynamics)."""
        out = []
        for prof in self.profiles():
            mixed = self.pure(prof)
            ok = True
            for a, p in enumerate(self.players):
                dev = self.deviation_payoffs(p, mixed)
                others = np.delete(dev, prof[a])
                if others.size and not (dev[prof[a]] > others.max() + tol):
                    ok = False
                    break
            if ok:
                out.append(prof)
        return out

    def coalition_deviations(self, prof: Profile, *, min_size: int = 2, tol: float = 1e-9) -> list[dict[str, Any]]:
        """Joint deviations from a pure profile that make every member of a coalition strictly better off.

        An equilibrium with no such deviation is (pure) *coalition-proof in the strong-Nash sense*; a
        mechanism whose honest profile admits one is vulnerable to collusion (e.g. two debaters who
        both gain by not exposing each other).
        """
        out = []
        base = {p: float(self.payoffs[p][prof]) for p in self.players}
        for size in range(min_size, self.n + 1):
            for coalition in itertools.combinations(range(self.n), size):
                choices = [range(self.shape[a]) if a in coalition else [prof[a]] for a in range(self.n)]
                for dev in itertools.product(*choices):
                    if dev == prof:
                        continue
                    gains = {self.players[a]: float(self.payoffs[self.players[a]][dev]) - base[self.players[a]]
                             for a in coalition}
                    if all(g > tol for g in gains.values()):
                        out.append({"coalition": [self.players[a] for a in coalition],
                                    "deviation": self.profile_names(dev), "gains": gains,
                                    "outcomes": {k: float(v[dev]) for k, v in self.outcomes.items()}})
        return out

    def dominant_strategies(self, strict: bool = False) -> dict[str, list[str]]:
        """Strategies that are (weakly or strictly) best against every opponent profile."""
        out = {}
        for a, p in enumerate(self.players):
            T = np.moveaxis(self.payoffs[p], a, 0).reshape(self.shape[a], -1)
            dom = []
            for i in range(self.shape[a]):
                others = np.delete(T, i, axis=0)
                if others.size == 0:
                    dom.append(self.strategies[p][i])
                    continue
                cmp = T[i][None, :] > others if strict else T[i][None, :] >= others - 1e-12
                if cmp.all():
                    dom.append(self.strategies[p][i])
            out[p] = dom
        return out

    def support_enumeration(self, tol: float = 1e-9) -> list[Mixed]:
        """All Nash equilibria of a 2-player game with supports of equal size (nondegenerate games)."""
        if self.n != 2:
            raise ValueError("support enumeration is implemented for 2-player games")
        A, B = self.payoffs[self.players[0]], self.payoffs[self.players[1]]
        m, k = A.shape
        found: list[Mixed] = []
        for size in range(1, min(m, k) + 1):
            for S1 in itertools.combinations(range(m), size):
                for S2 in itertools.combinations(range(k), size):
                    y = _indifference(A[np.ix_(S1, S2)])  # column player's mix makes row player indifferent
                    x = _indifference(B[np.ix_(S1, S2)].T)
                    if x is None or y is None:
                        continue
                    xf, yf = np.zeros(m), np.zeros(k)
                    xf[list(S1)], yf[list(S2)] = x, y
                    mixed = [xf, yf]
                    if self.nash_conv(mixed) <= 1e-7 and not any(_close(mixed, f) for f in found):
                        found.append(mixed)
        return found

    def correlated_equilibrium(self, objective: str | np.ndarray | None = None, maximize: bool = True,
                               coarse: bool = False) -> tuple[np.ndarray, float] | None:
        """A (coarse) correlated equilibrium optimizing a linear objective over joint distributions.

        ``objective`` is an outcome key (e.g. ``"welfare"``) or a tensor. Returns (joint distribution
        with the game's shape, objective value). Useful to bound ground-truth welfare over *all*
        (coarse) correlated equilibria, which is where no-regret learners end up.
        """
        size = int(np.prod(self.shape))
        if objective is None:
            c = np.zeros(size)
        else:
            T = self.outcomes[objective] if isinstance(objective, str) else np.asarray(objective)
            c = (-T if maximize else T).reshape(-1)
        rows = []
        for a, p in enumerate(self.players):
            U = self.payoffs[p]
            if coarse:
                for s_dev in range(self.shape[a]):
                    dev = np.take(U, [s_dev], axis=a)
                    dev = np.broadcast_to(dev, U.shape)
                    rows.append((dev - U).reshape(-1))
            else:
                for s in range(self.shape[a]):
                    for s_dev in range(self.shape[a]):
                        if s == s_dev:
                            continue
                        mask = np.zeros(self.shape)
                        idx = [slice(None)] * self.n
                        idx[a] = s
                        mask[tuple(idx)] = 1.0
                        dev = np.broadcast_to(np.take(U, [s_dev], axis=a), U.shape)
                        rows.append((mask * (dev - U)).reshape(-1))
        A_ub = np.array(rows) if rows else None
        b_ub = np.zeros(len(rows)) if rows else None
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=np.ones((1, size)), b_eq=[1.0], bounds=[(0, 1)] * size,
                      method="highs")
        if not res.success:
            return None
        mu = res.x.reshape(self.shape)
        val = float(-res.fun if (maximize and objective is not None) else res.fun)
        return mu, val

    def outcome_range(self, key: str, *, coarse: bool = False) -> tuple[float, float]:
        """Min and max of an outcome statistic over all (coarse) correlated equilibria."""
        lo = self.correlated_equilibrium(key, maximize=False, coarse=coarse)
        hi = self.correlated_equilibrium(key, maximize=True, coarse=coarse)
        return (lo[1] if lo else math.nan, hi[1] if hi else math.nan)

    # ------------------------------------------------------------------ dynamics
    def replicator(self, x0: Mixed | None = None, *, steps: int = 5000, dt: float = 0.05,
                   tol: float = 1e-10, shared: Sequence[Sequence[str]] | None = None) -> tuple[Mixed, list[Mixed]]:
        """Multi-population discrete replicator dynamics; returns (final, trajectory).

        ``shared``: groups of exchangeable players that form one population (one mixed strategy).
        Without it, rounding differences between symmetric players can grow along unstable directions
        and carry the dynamics away from a symmetric rest point.
        """
        x = [np.array(v, dtype=float) for v in (x0 or self.uniform())]
        groups = [[self.players.index(p) for p in g] for g in (shared or [])]
        traj = [[v.copy() for v in x]]
        scale = max(1e-12, max(float(np.ptp(self.payoffs[p])) for p in self.players))
        for _ in range(steps):
            new = []
            for a, p in enumerate(self.players):
                f = self.deviation_payoffs(p, x) / scale
                avg = f @ x[a]
                v = x[a] * (1 + dt * (f - avg))
                v = np.clip(v, 0, None)
                new.append(v / v.sum())
            for g in groups:
                mean = np.mean([new[a] for a in g], axis=0)
                for a in g:
                    new[a] = mean.copy()
            delta = max(float(np.abs(u - v).max()) for u, v in zip(new, x))
            x = new
            traj.append([v.copy() for v in x])
            if delta < tol:
                break
        return x, traj

    def fictitious_play(self, iters: int = 2000) -> Mixed:
        counts = [np.ones(s) for s in self.shape]
        for _ in range(iters):
            mixed = [c / c.sum() for c in counts]
            for a, p in enumerate(self.players):
                counts[a][int(np.argmax(self.deviation_payoffs(p, mixed)))] += 1
        return [c / c.sum() for c in counts]

    def regret_matching(self, iters: int = 5000, seed: int = 0) -> np.ndarray:
        """Average joint play of regret-matching learners: converges to the set of coarse correlated equilibria."""
        rng = np.random.default_rng(seed)
        regrets = [np.zeros(s) for s in self.shape]
        joint = np.zeros(self.shape)
        for _ in range(iters):
            strat = []
            for r in regrets:
                pos = np.clip(r, 0, None)
                strat.append(pos / pos.sum() if pos.sum() > 0 else np.full(len(r), 1 / len(r)))
            prof = tuple(int(rng.choice(len(s), p=s)) for s in strat)
            joint[prof] += 1
            for a, p in enumerate(self.players):
                idx = list(prof)
                u = self.payoffs[p][prof]
                for s in range(self.shape[a]):
                    idx[a] = s
                    regrets[a][s] += self.payoffs[p][tuple(idx)] - u
        return joint / joint.sum()

    def basin(self, target: Profile, *, n_starts: int = 200, seed: int = 0, steps: int = 3000, radius: float = 0.05) -> float:
        """Fraction of random initial mixed profiles from which replicator dynamics reaches ``target``."""
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(n_starts):
            x0 = [rng.dirichlet(np.ones(s)) for s in self.shape]
            xf, _ = self.replicator(x0, steps=steps)
            if all(v[i] > 1 - radius for v, i in zip(xf, target)):
                hits += 1
        return hits / n_starts

    # ------------------------------------------------------------------ symmetric games
    def is_symmetric(self, tol: float = 1e-9) -> bool:
        if len(set(self.shape)) != 1 or len({tuple(s) for s in self.strategies.values()}) != 1:
            return False
        U0 = self.payoffs[self.players[0]]
        for a, p in enumerate(self.players):
            perm = list(range(self.n))
            perm[0], perm[a] = perm[a], perm[0]
            if not np.allclose(np.transpose(U0, perm), self.payoffs[p], atol=tol):
                return False
        return True

    def symmetric_payoff(self, x: np.ndarray) -> np.ndarray:
        """Payoff of each pure strategy for one player when all others play ``x`` (symmetric games)."""
        return self.deviation_payoffs(self.players[0], [x] * self.n)

    def symmetric_replicator(self, x0: np.ndarray | None = None, steps: int = 5000, dt: float = 0.05) -> np.ndarray:
        s = self.shape[0]
        x = np.full(s, 1 / s) if x0 is None else np.asarray(x0, dtype=float)
        scale = max(1e-12, float(np.ptp(self.payoffs[self.players[0]])))
        for _ in range(steps):
            f = self.symmetric_payoff(x) / scale
            v = np.clip(x * (1 + dt * (f - f @ x)), 0, None)
            v = v / v.sum()
            if np.abs(v - x).max() < 1e-12:
                x = v
                break
            x = v
        return x

    def symmetric_equilibria(self, grid: int = 200, tol: float = 1e-6) -> list[np.ndarray]:
        """Symmetric Nash equilibria of a 2-strategy symmetric game (pure and interior), by scanning."""
        if self.shape[0] != 2:
            raise ValueError("symmetric_equilibria scans 2-strategy games; use replicator dynamics otherwise")
        eqs = []

        def gain(p: float) -> float:
            f = self.symmetric_payoff(np.array([1 - p, p]))
            return float(f[1] - f[0])

        if gain(0.0) <= tol:
            eqs.append(np.array([1.0, 0.0]))
        if gain(1.0) >= -tol:
            eqs.append(np.array([0.0, 1.0]))
        ps = np.linspace(0, 1, grid + 1)
        gs = [gain(p) for p in ps]
        for i in range(grid):
            if gs[i] == 0 or gs[i] * gs[i + 1] < 0:
                lo, hi = ps[i], ps[i + 1]
                for _ in range(60):
                    mid = (lo + hi) / 2
                    if gain(lo) * gain(mid) <= 0:
                        hi = mid
                    else:
                        lo = mid
                p = (lo + hi) / 2
                if 1e-6 < p < 1 - 1e-6:
                    eqs.append(np.array([1 - p, p]))
        return eqs

    # ------------------------------------------------------------------ export
    def table(self) -> Any:
        import pandas as pd

        rows = []
        for prof in self.profiles():
            row = {f"s_{p}": self.strategies[p][i] for p, i in zip(self.players, prof)}
            row.update({f"u_{p}": float(self.payoffs[p][prof]) for p in self.players})
            row.update({k: float(v[prof]) for k, v in self.outcomes.items()})
            row.update({f"{k}_coverage": float(v[prof]) for k, v in self.coverage.items()})
            if self.counts is not None:
                row["n"] = int(self.counts[prof])
            row["nash"] = self.is_nash(prof)
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return f"NormalFormGame({self.name!r}, players={self.players}, shape={self.shape})"


def _indifference(M: np.ndarray) -> np.ndarray | None:
    """Mix y over columns making all rows of M equal: M y = v 1, sum y = 1, y >= 0."""
    k = M.shape[1]
    A = np.zeros((M.shape[0] + 1, k + 1))
    A[: M.shape[0], :k] = M
    A[: M.shape[0], k] = -1
    A[M.shape[0], :k] = 1
    b = np.zeros(M.shape[0] + 1)
    b[-1] = 1
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if not np.allclose(A @ sol, b, atol=1e-8):
        return None
    y = sol[:k]
    if (y < -1e-9).any():
        return None
    y = np.clip(y, 0, None)
    return y / y.sum()


def _close(a: Mixed, b: Mixed, tol: float = 1e-6) -> bool:
    return all(np.allclose(x, y, atol=tol) for x, y in zip(a, b))


def zero_sum_value(A: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Value and maximin strategies of a two-player zero-sum game with row-player payoff matrix A."""
    A = np.asarray(A, dtype=float)
    m, k = A.shape
    # row player: max v s.t. x^T A >= v, sum x = 1
    c = np.zeros(m + 1)
    c[-1] = -1
    A_ub = np.hstack([-A.T, np.ones((k, 1))])
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(k), A_eq=np.hstack([np.ones((1, m)), [[0]]]), b_eq=[1],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    x = res.x[:m]
    v = -res.fun
    c2 = np.zeros(k + 1)
    c2[-1] = 1
    A_ub2 = np.hstack([A, -np.ones((m, 1))])
    res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(m), A_eq=np.hstack([np.ones((1, k)), [[0]]]), b_eq=[1],
                   bounds=[(0, None)] * k + [(None, None)], method="highs")
    y = res2.x[:k]
    return float(v), x, y


def game_from_function(players: Sequence[str], strategies: dict[str, Sequence[str]],
                       payoff: Callable[[dict[str, str]], dict[str, float]],
                       outcomes: dict[str, Callable[[dict[str, str]], float]] | None = None, name: str = "game") -> NormalFormGame:
    """Build a game by evaluating ``payoff(profile_names) -> {player: u}`` on every profile."""
    shape = tuple(len(strategies[p]) for p in players)
    U = {p: np.zeros(shape) for p in players}
    O = {k: np.zeros(shape) for k in (outcomes or {})}
    for prof in itertools.product(*[range(s) for s in shape]):
        names = {p: strategies[p][i] for p, i in zip(players, prof)}
        u = payoff(names)
        for p in players:
            U[p][prof] = u[p]
        for k, fn in (outcomes or {}).items():
            O[k][prof] = fn(names)
    return NormalFormGame(players, strategies, U, outcomes=O, name=name)
