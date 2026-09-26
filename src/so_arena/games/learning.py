"""Learning dynamics on (empirical) games: what does training on a mechanism's rewards converge to?

Treat each strategy in an empirical game (prompts, checkpoints, behaviours) as an action of a
softmax policy and run independent, exact policy-gradient ascent on each player's expected reward,
optionally KL-regularized toward an initial policy (as in RLHF: maximize
$E_\\pi[u] - \\tau\\,KL(\\pi\\|\\pi_0)$). The trajectory of (expected mechanism reward, expected
ground-truth outcome) over training steps is the RL analogue of the best-of-N parametric curves:
it shows whether optimizing the mechanism's rewards moves agents toward or away from the truth,
and which equilibrium (e.g. "everyone reports" vs "nobody reports") training selects from a given
starting propensity.

For a single player facing fixed opponents, the stationary point of KL-regularized ascent is the
tilted policy $\\pi \\propto \\pi_0 e^{u/\\tau}$ (see :mod:`so_arena.analysis.optimization`).

Vanilla and *natural* policy gradient (``natural=True``: the Fisher-preconditioned step, which for a
softmax moves each logit by its strategy's advantage $u_a - \\bar u$ - exponential weights, whose
continuous-time limit is the replicator dynamics) have the same rest points, but vanilla gradient moves a
logit in proportion to its strategy's probability, so a rarely played strategy learns slowly. From the
same start the two can reach different equilibria: equilibrium selection depends on the training
algorithm (docs/theory.md, section 8). :class:`StrategyGradient` runs either on *sampled episodes* of a
mechanism instead of an estimated game.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from so_arena.games.normal_form import Mixed, NormalFormGame


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _warn_low_coverage(game: NormalFormGame, low: dict[str, float], min_coverage: float) -> None:
    if low:
        import logging

        logging.getLogger("so_arena").warning(
            "%s: outcome(s) %s measured on less than %.0f%% of the policy's mass at some steps (lowest %s): "
            "reported as NaN there", game.name, sorted(low), 100 * min_coverage,
            ", ".join(f"{k} {v:.0%}" for k, v in sorted(low.items())))


def policy_gradient(game: NormalFormGame, *, init: Mixed | None = None, lr: float = 0.5, steps: int = 500,
                    kl: float = 0.0, reference: Mixed | None = None, shared: list[list[str]] | None = None,
                    record_every: int = 1, outcome_keys: list[str] | None = None,
                    min_coverage: float = 0.5, natural: bool = False) -> pd.DataFrame:
    """Independent softmax policy gradient with exact expected gradients.

    Args:
        init: initial mixed strategies (default uniform); also the KL reference unless ``reference``.
        kl: KL-regularization strength $\\tau$ toward the reference policy.
        natural: natural policy gradient - each logit moves by its strategy's advantage
            $u_a - \\bar u$ (minus $\\tau$ times its centred log-ratio to the reference) instead of
            $\\pi_a$ times it: exponential weights, the discrete-time replicator dynamics.
        shared: groups of players sharing one set of logits (self-play of identical agents).
        outcome_keys: outcome tensors (e.g. ``"gt_welfare"``) to track; default: all.

    Returns a frame with one row per recorded step: ``p_<player>_<strategy>``, ``reward_<player>``
    and each outcome's expectation with its coverage ``<key>_coverage`` - the policy's probability mass
    of episodes that had a value (:meth:`NormalFormGame.expected_outcome`); below ``min_coverage`` the
    expectation is NaN (one warning per call).

    Missing payoffs are filled with the player's lowest observed payoff (with a warning; see
    :meth:`NormalFormGame.imputed`) - filling them with 0, the best log-score reward, would make
    training converge to a profile nobody observed. Outcome expectations average over the observed
    profiles only (renormalized), and says how much of the mass that is.
    """
    players = game.players
    x0 = [np.asarray(v, float) for v in (init or game.uniform())]
    ref = [np.asarray(v, float) for v in (reference or x0)]
    logits = [np.log(np.clip(v, 1e-12, None)) for v in x0]
    groups = shared or []
    keys = outcome_keys if outcome_keys is not None else list(game.outcomes)
    clean = game.imputed()
    rows = []
    low: dict[str, float] = {}
    for t in range(steps + 1):
        pis = [softmax(z) for z in logits]
        if t % record_every == 0 or t == steps:
            row = {"step": t}
            for a, p in enumerate(players):
                for s, prob in zip(game.strategies[p], pis[a]):
                    row[f"p_{p}_{s}"] = float(prob)
                row[f"reward_{p}"] = clean.expected(pis, player=p)
            for k in keys:
                row[k], row[f"{k}_coverage"] = game.expected_outcome(pis, k, min_coverage=min_coverage, warn=False)
                if 0 < row[f"{k}_coverage"] < min_coverage:
                    low[k] = min(low.get(k, 1.0), row[f"{k}_coverage"])
            rows.append(row)
        if t == steps:
            break
        grads = []
        for a, p in enumerate(players):
            f = clean.deviation_payoffs(p, pis)
            g = f - f @ pis[a]  # advantages; times pi: d E[u] / d logits for a softmax policy
            if kl > 0:
                logr = np.log(np.clip(pis[a], 1e-12, None)) - np.log(np.clip(ref[a], 1e-12, None))
                g = g - kl * (logr - logr @ pis[a])
            # the Fisher information of a softmax is diag(pi) - pi pi^T: preconditioning the gradient
            # pi * g by its inverse leaves the advantages g (up to a constant shift, which softmax ignores)
            grads.append(g if natural else pis[a] * g)
        for gi, g in enumerate(groups):  # shared logits: average the group's gradients
            idx = [players.index(p) for p in g]
            avg = np.mean([grads[i] for i in idx], axis=0)
            for i in idx:
                grads[i] = avg
        for a in range(len(players)):
            logits[a] = logits[a] + lr * grads[a]
        for gi, g in enumerate(groups):
            idx = [players.index(p) for p in g]
            avg = np.mean([logits[i] for i in idx], axis=0)
            for i in idx:
                logits[i] = avg.copy()
    _warn_low_coverage(game, low, min_coverage)
    return pd.DataFrame(rows)


def training_outcome(game: NormalFormGame, init_report: float, player_strategy: str = "report", **kw) -> dict[str, float]:
    """Convenience for symmetric 2-strategy games: run shared-logit policy gradient from an initial
    propensity for ``player_strategy`` and return the final propensity and outcomes."""
    s = game.strategies[game.players[0]]
    j = s.index(player_strategy)
    x = np.full(len(s), (1 - init_report) / (len(s) - 1))
    x[j] = init_report
    df = policy_gradient(game, init=[x.copy() for _ in game.players], shared=[list(game.players)], **kw)
    last = df.iloc[-1]
    return {"final_p": float(last[f"p_{game.players[0]}_{player_strategy}"]),
            **{k: float(last[k]) for k in game.outcomes}}


def natural_estimate(picks: np.ndarray, rewards: np.ndarray, baselines: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Unbiased estimate of the natural gradient (the advantages $u_a - \\bar u$) from sampled plays.

    ``picks[j]`` is the strategy index sampled for play ``j``, ``rewards[j]`` its reward and ``baselines[j]``
    a baseline independent of that play (e.g. the mean reward of *other* episodes); ``probs`` is the policy
    the picks were drawn from. The importance-weighted sum $\\hat g_a = \\frac1B\\sum_j \\mathbb 1[a_j = a]
    (r_j - b_j)/\\pi_a$ has expectation $u_a - E[b]$ for every strategy, however rarely it is played (the
    softmax ignores the common shift $E[b]$) - as in EXP3. A baseline computed from the play itself (a batch
    mean including it) would bias it by $O(1/B)$, and far more for rare strategies.
    """
    g = np.zeros(len(probs))
    np.add.at(g, picks, rewards - baselines)
    return g / (len(picks) * np.clip(probs, 1e-12, None))


def vanilla_estimate(picks: np.ndarray, rewards: np.ndarray, baselines: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """REINFORCE: $\\frac1B\\sum_j (r_j - b_j)\\,\\nabla_\\theta \\log\\pi(a_j)$ with $\\nabla \\log \\pi(a) = e_a - \\pi$;
    its expectation is the exact gradient $\\pi_a(u_a - \\bar u)$ when ``baselines`` are independent of the plays."""
    g = -np.outer(rewards - baselines, probs)
    np.add.at(g, (np.arange(len(picks)), picks), rewards - baselines)
    return g.mean(axis=0)


class StrategyGradient:
    """Strategy-level policy-gradient training on *sampled episodes* of a mechanism.

    Each trainable role holds a softmax policy over a finite strategy set (prompts, scripted behaviours,
    checkpoints). Every iteration draws a batch of episodes - an item uniformly at random and each role's
    strategy from its current policy - runs them, and updates the logits from the mechanism's rewards by
    REINFORCE or, with ``natural=True``, by the natural policy gradient (:func:`natural_estimate`). Both are
    unbiased: the baseline for a play is the mean reward of the batch's *other* episodes, which cannot
    depend on its actions. In expectation this is :func:`policy_gradient` on the empirical game, without
    simulating every profile first; the sampling noise is real training's.

    Args:
        strategies: role -> {strategy name -> policy}; the roles being trained.
        fixtures: policies for the other roles.
        shared: groups of roles sharing one policy (self-play of identical agents): each member samples its
            own strategy, and every member's plays update the shared logits.
        init: role -> initial probabilities (default uniform).
        natural: natural policy gradient instead of REINFORCE.
        lr / batch / iterations: step size on the logits, episodes per iteration, number of updates.
        outcomes: per-episode statistics to average per iteration (defaults as in
            :class:`~so_arena.games.EmpiricalGameExperiment`).

    Chance moves (audits, who witnesses what) use a fresh seed per sampled episode, derived from ``seed``,
    the iteration and the batch index; the strategy draws come from a separate stream. A missing reward
    (an errored episode, or a rule that could not score it) counts as the lowest reward observed for the
    role so far (with a warning), as :meth:`NormalFormGame.imputed` does: dropping it would reward
    strategies that break the episode.
    """

    def __init__(self, mechanism: Any, items: Sequence[Any], strategies: dict[str, dict[str, Any]], *,
                 fixtures: dict[str, Any] | None = None, shared: list[list[str]] | None = None,
                 init: dict[str, Sequence[float]] | None = None, natural: bool = False, lr: float = 0.5,
                 batch: int = 16, iterations: int = 20, ground_truth: Sequence[Any] | None = None, ctx: Any = None,
                 outcomes: dict[str, Any] | None = None, seed: int = 0, concurrency: int | None = None):
        from so_arena.games.egta import DEFAULT_OUTCOMES

        self.mechanism, self.items = mechanism, list(items)
        self.strategies = {r: dict(s) for r, s in strategies.items()}
        self.roles = list(self.strategies)
        self.fixtures = dict(fixtures or {})
        self.groups = [list(g) for g in (shared or [])]
        grouped = [r for g in self.groups for r in g]
        unknown = sorted(set(grouped) - set(self.roles))
        if unknown or len(grouped) != len(set(grouped)):
            raise ValueError(f"shared groups must be disjoint sets of trained roles (unknown: {unknown})")
        for g in self.groups:
            if len({tuple(self.strategies[r]) for r in g}) != 1:
                raise ValueError(f"roles sharing a policy must share the strategy set: {g}")
        self.groups += [[r] for r in self.roles if r not in grouped]
        self.natural, self.lr, self.batch, self.iterations = natural, lr, batch, iterations
        self.ground_truth, self.ctx, self.seed, self.concurrency = ground_truth, ctx, seed, concurrency
        self.outcome_fns = {**DEFAULT_OUTCOMES, **(outcomes or {})}
        self.logits: dict[str, np.ndarray] = {}
        for g in self.groups:
            names = list(self.strategies[g[0]])
            x = np.asarray(init[g[0]], float) if init and g[0] in init else np.full(len(names), 1 / len(names))
            if x.shape != (len(names),):
                raise ValueError(f"init for {g[0]!r} needs {len(names)} probabilities")
            z = np.log(np.clip(x, 1e-12, None))
            for r in g:
                self.logits[r] = z
        self.episodes: list[Any] = []
        self._worst: dict[str, float] = {}
        self._warned = False

    def probs(self, role: str) -> np.ndarray:
        return softmax(self.logits[role])

    def _profiles(self, it: int, rng: np.random.Generator) -> list[tuple[Any, Any, dict[str, int]]]:
        from so_arena.core.runner import PlayerSpec, Profile

        out = []
        for b in range(self.batch):
            item = self.items[int(rng.integers(len(self.items)))]
            picks = {r: int(rng.choice(len(self.strategies[r]), p=self.probs(r))) for r in self.roles}
            assign = {r: list(self.strategies[r])[i] for r, i in picks.items()}
            players = {r: PlayerSpec(policy=pol) for r, pol in self.fixtures.items()}
            for r, name in assign.items():
                players[r] = PlayerSpec(policy=self.strategies[r][name], label=name)
            name = f"sg{it}.{b}|" + "|".join(f"{r}={assign[r]}" for r in self.roles)
            out.append((item, Profile(name=name, players=players, tags={"profile_assign": assign}), picks))
        return out

    async def arun(self) -> pd.DataFrame:
        import asyncio

        from so_arena.core.policy import stable_hash
        from so_arena.core.runner import run_episodes

        rows = []
        for it in range(self.iterations + 1):
            rng = np.random.default_rng([self.seed, it])
            jobs = self._profiles(it, rng)
            runs = await asyncio.gather(*[
                run_episodes(self.mechanism, [item], [prof], ctx=self.ctx, ground_truth=self.ground_truth,
                             concurrency=self.concurrency, seed=stable_hash("strategy-gradient", self.seed, it, b) % 10**9)
                for b, (item, prof, _) in enumerate(jobs)])
            eps = [r[0] if r else None for r in runs]
            self.episodes += [e for e in eps if e is not None]
            rewards = {r: self._rewards(r, eps) for r in self.roles}
            row: dict[str, Any] = {"iteration": it}
            for r in self.roles:
                for s, p in zip(self.strategies[r], self.probs(r)):
                    row[f"p_{r}_{s}"] = float(p)
                row[f"reward_{r}"] = float(np.nanmean(rewards[r])) if np.isfinite(rewards[r]).any() else math.nan
            for k, fn in self.outcome_fns.items():
                vals = [v for e in eps if e is not None and e.error is None and (v := fn(e)) is not None and np.isfinite(v)]
                row[k] = float(np.mean(vals)) if vals else math.nan
            row["errors"] = sum(e is None or e.error is not None for e in eps)
            rows.append(row)
            if it == self.iterations:
                break
            for g in self.groups:
                picks = np.array([p[r] for r in g for (_, _, p) in jobs])
                rs = np.concatenate([rewards[r] for r in g])
                if not np.isfinite(rs).all():
                    continue  # no reward observed for this role yet: nothing to learn from
                # baseline: the mean reward of the *other* episodes (independent of this play's actions)
                ep_mean = np.mean([rewards[r] for r in g], axis=0)
                others = (ep_mean.sum() - ep_mean) / max(1, len(ep_mean) - 1)
                base = np.tile(others, len(g))
                est = (natural_estimate if self.natural else vanilla_estimate)(picks, rs, base, self.probs(g[0]))
                z = self.logits[g[0]] + self.lr * est
                for r in g:
                    self.logits[r] = z
        return pd.DataFrame(rows)

    def run(self) -> pd.DataFrame:
        from so_arena.core.runner import run_sync

        return run_sync(self.arun())

    def _rewards(self, role: str, eps: list[Any]) -> np.ndarray:
        vals = np.array([math.nan if e is None or e.error is not None or e.rewards.get(role) is None
                         else float(e.rewards[role]) for e in eps])
        ok = np.isfinite(vals)
        if ok.any():
            self._worst[role] = min(self._worst.get(role, math.inf), float(vals[ok].min()))
        if not ok.all() and role in self._worst:
            if not self._warned:
                import logging

                logging.getLogger("so_arena").warning(
                    "%s: %d episode(s) without a reward for %s; counted as its lowest reward so far (%.3g)",
                    self.mechanism.name, int((~ok).sum()), role, self._worst[role])
                self._warned = True
            vals[~ok] = self._worst[role]
        return vals
