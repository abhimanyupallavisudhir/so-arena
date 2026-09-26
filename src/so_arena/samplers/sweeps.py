"""Sweeps: how incentive compatibility changes with capability gaps, budgets or any other parameter.

Scaling questions ("does debate's advantage persist as the agent-judge gap grows?", "how does ASD
depend on the judge's evaluation time?") are sweeps: run the same experiment over a grid of
configurations and track a metric. :func:`sweep` runs an ASD experiment per configuration and returns
one tidy table; :func:`breakdown_point` estimates where a fitted trend crosses zero - e.g. the
capability gap at which a protocol stops rewarding honesty. Capability numbers come from the model
registry (``ModelSpec.params_b``, ``training_flops``, ``ratings``) via :func:`capability`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from so_arena.core.mechanism import Episode


def sweep(configs: Sequence[dict[str, Any]], make: Callable[[dict[str, Any]], Any],
          metric: Callable[[Sequence[Episode]], pd.DataFrame] | None = None) -> pd.DataFrame:
    """Run ``make(config).run()`` for each config and collect ``metric(episodes)`` rows tagged with the config.

    ``make`` returns anything with ``run() -> episodes`` (e.g. an :class:`ASDExperiment`); ``metric``
    defaults to ASD per mechanism.
    """
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd

    metric = metric or (lambda eps: asd(role_frame(eps)))
    frames = []
    for cfg in configs:
        eps = make(cfg).run()
        m = metric(eps)
        for k, v in cfg.items():
            m[k] = v if not isinstance(v, (list, dict)) else str(v)
        frames.append(m)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def breakdown_point(df: pd.DataFrame, *, x: str, y: str = "asd", by: str | None = "mechanism") -> pd.DataFrame:
    """Least-squares line y ~ a + b x per group and the x where it crosses zero (NaN if it never does).

    For y = ASD and x = capability gap (agent minus judge), the crossing is the gap beyond which the
    protocol, on this trend, stops rewarding honesty.
    """
    rows = []
    groups = df.groupby(by) if by else [(None, df)]
    for key, g in groups:
        g = g[[x, y]].dropna()
        if len(g) < 2 or g[x].nunique() < 2:
            continue
        b, a = np.polyfit(g[x].to_numpy(float), g[y].to_numpy(float), 1)
        cross = -a / b if b != 0 else math.nan
        rows.append({**({by: key} if by else {}), "intercept": float(a), "slope": float(b),
                     "breakdown": float(cross) if b < 0 else math.nan, "n": len(g)})
    return pd.DataFrame(rows)


def capability(model_name: str, key: str = "arena_elo") -> float | None:
    """A capability number for a model from the registry: a rating (``key``), else log10 params."""
    from so_arena.models.registry import get_spec

    spec = get_spec(model_name)
    if spec is None:
        return None
    if key in spec.ratings:
        return spec.ratings[key]
    if key == "log_params" and spec.params_b:
        return math.log10(spec.params_b * 1e9)
    if key == "log_flops" and spec.training_flops:
        return math.log10(spec.training_flops)
    return None
