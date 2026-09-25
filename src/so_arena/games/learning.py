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
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from so_arena.games.normal_form import Mixed, NormalFormGame


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _observed_expectation(T: np.ndarray, pis: Mixed) -> float:
    """Expectation of an outcome tensor under independent mixed strategies, over its finite cells
    (renormalized); NaN if no observed profile has positive probability."""
    joint = pis[0]
    for x in pis[1:]:
        joint = np.multiply.outer(joint, x)
    mask = np.isfinite(T)
    den = float(joint[mask].sum())
    return float((joint[mask] * T[mask]).sum() / den) if den > 0 else float("nan")


def policy_gradient(game: NormalFormGame, *, init: Mixed | None = None, lr: float = 0.5, steps: int = 500,
                    kl: float = 0.0, reference: Mixed | None = None, shared: list[list[str]] | None = None,
                    record_every: int = 1, outcome_keys: list[str] | None = None) -> pd.DataFrame:
    """Independent softmax policy gradient with exact expected gradients.

    Args:
        init: initial mixed strategies (default uniform); also the KL reference unless ``reference``.
        kl: KL-regularization strength $\\tau$ toward the reference policy.
        shared: groups of players sharing one set of logits (self-play of identical agents).
        outcome_keys: outcome tensors (e.g. ``"gt_welfare"``) to track; default: all.

    Returns a frame with one row per recorded step: ``p_<player>_<strategy>``, ``reward_<player>``
    and each outcome's expectation.

    Missing payoffs are filled with the player's lowest observed payoff (with a warning; see
    :meth:`NormalFormGame.imputed`) - filling them with 0, the best log-score reward, would make
    training converge to a profile nobody observed. Outcome expectations average over the observed
    profiles only.
    """
    players = game.players
    x0 = [np.asarray(v, float) for v in (init or game.uniform())]
    ref = [np.asarray(v, float) for v in (reference or x0)]
    logits = [np.log(np.clip(v, 1e-12, None)) for v in x0]
    groups = shared or []
    keys = outcome_keys if outcome_keys is not None else list(game.outcomes)
    clean = game.imputed()
    rows = []
    for t in range(steps + 1):
        pis = [softmax(z) for z in logits]
        if t % record_every == 0 or t == steps:
            row = {"step": t}
            for a, p in enumerate(players):
                for s, prob in zip(game.strategies[p], pis[a]):
                    row[f"p_{p}_{s}"] = float(prob)
                row[f"reward_{p}"] = clean.expected(pis, player=p)
            for k in keys:
                row[k] = _observed_expectation(game.outcomes[k], pis)
            rows.append(row)
        if t == steps:
            break
        grads = []
        for a, p in enumerate(players):
            f = clean.deviation_payoffs(p, pis)
            g = pis[a] * (f - f @ pis[a])  # d E[u] / d logits for a softmax policy
            if kl > 0:
                logr = np.log(np.clip(pis[a], 1e-12, None)) - np.log(np.clip(ref[a], 1e-12, None))
                g = g - kl * pis[a] * (logr - logr @ pis[a])
            grads.append(g)
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
