"""Propositions 1 and 2 of docs/theory.md, checked numerically against the library."""

import math

import numpy as np
import pandas as pd
import pytest

from so_arena.analysis.optimization import BestOfN, pool_curve


def test_proposition_1_asd_is_the_policy_gradient():
    rng = np.random.default_rng(0)
    c = rng.normal(0, 1.5, size=40)  # per-item offsets of the propensity
    u_plus, u_minus = rng.normal(0.3, 1, 40), rng.normal(-0.2, 1, 40)
    asd_t = u_plus - u_minus
    sig = lambda x: 1 / (1 + np.exp(-x))  # noqa: E731

    def U(theta):
        p = sig(theta + c)
        return float(np.mean(p * u_plus + (1 - p) * u_minus))

    theta, h = 0.4, 1e-6
    grad = (U(theta + h) - U(theta - h)) / (2 * h)
    p = sig(theta + c)
    assert grad == pytest.approx(float(np.mean(p * (1 - p) * asd_t)), rel=1e-6)
    # natural gradient: the p(1-p)-weighted mean of ASD_t, which is ASD itself when p_t is constant
    fisher = float(np.mean(p * (1 - p)))
    assert grad / fisher == pytest.approx(float(np.average(asd_t, weights=p * (1 - p))), rel=1e-6)
    c[:] = 0.0
    p = sig(theta + c)
    assert (U(theta + h) - U(theta - h)) / (2 * h) / float(np.mean(p * (1 - p))) == pytest.approx(asd_t.mean(), rel=1e-6)


def test_proposition_2_extremal_goodhart_example():
    # the base policy argues honestly half of the time: honest 0.5 always; dishonest 0.9 w.p. 0.1, else 0
    rewards = [0.5] * 10 + [0.9] + [0.0] * 9
    values = [1.0] * 10 + [0.0] * 10
    pool = pd.DataFrame({"item_id": "x", "reward": rewards, "value": values})
    asd = 0.5 - (0.9 * 0.1)
    assert asd > 0 and np.cov(rewards, values)[0, 1] > 0
    ns = [1, 2, 4, 8, 32, 256]
    curve = pool_curve(pool, selections=[BestOfN(n, "plugin") for n in ns])
    expected = [0.95 ** n - 0.45 ** n for n in ns]
    assert curve["value"].tolist() == pytest.approx(expected, abs=1e-12)
    assert max(expected) == pytest.approx(0.95 ** 4 - 0.45 ** 4) and curve["value"].iloc[-1] < 1e-5
    # the limit is the value at the reward's maximum, whatever the sign of ASD
    assert math.isclose(pool.loc[pool["reward"].idxmax(), "value"], 0.0)
