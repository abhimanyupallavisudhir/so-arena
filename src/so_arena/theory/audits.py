"""Audits as control variates: how much ground truth is an oversight mechanism worth?

If a trusted audit (ground truth, a human expert, the eventual outcome) is available on a random
fraction ``p`` of episodes, the audit-corrected reward

    r~ = lam * u + (1[audit] / p) * (v - lam * u)

is unbiased for the audited value v under *any* policy (so it is incentive-compatible in
expectation), and the mechanism reward u only changes its variance. With the optimal lam =
Cov(u, v) / Var(u), the variance of the correction term is Var(v) (1 - rho^2) where rho = corr(u, v);
so, for the same variance, the mechanism multiplies the effective number of audits by
1 / (1 - rho^2). This reframes "how good is an oversight mechanism" as "how much audit budget does
it save" - with the caveat that rho must be measured under the (optimized) policy distribution.
"""

from __future__ import annotations

import numpy as np


def optimal_lambda(u, v) -> float:
    u, v = np.asarray(u, float), np.asarray(v, float)
    var = np.var(u)
    return float(np.cov(u, v, ddof=0)[0, 1] / var) if var > 0 else 0.0


def audit_corrected(u, v, audited, p: float, lam: float | None = None) -> np.ndarray:
    """Per-episode unbiased rewards; ``audited`` is a boolean mask, ``v`` is used only where audited."""
    u, v, a = np.asarray(u, float), np.asarray(v, float), np.asarray(audited, bool)
    lam = optimal_lambda(u[a], v[a]) if lam is None else lam
    out = lam * u
    out[a] = out[a] + (v[a] - lam * u[a]) / p
    return out


def label_efficiency(u, v) -> float:
    """Effective audit multiplier 1/(1 - rho^2) of mechanism reward u for ground truth v."""
    u, v = np.asarray(u, float), np.asarray(v, float)
    if np.std(u) == 0 or np.std(v) == 0:
        return 1.0
    rho = float(np.corrcoef(u, v)[0, 1])
    return float("inf") if abs(rho) >= 1 else 1.0 / (1.0 - rho ** 2)
