"""Parameter sweeps: run an experiment factory over a grid and collect one tidy table.

Systematic studies vary one experimental factor at a time — verification budget or noise,
judge strength, protocol rounds, bounty size — and read off how incentive compatibility and
outcomes respond::

    def make(budget, trust):
        return Experiment(HiddenBits(n_tasks=60), [Propaganda(evidence=EvidencePolicy(budget=budget)),
                                                    Debate(evidence=EvidencePolicy(budget=budget))],
                          {"kind:judge": BayesianBitJudge(trust=trust), "*": BitAdvocate()}, Stances())

    table = sweep(make, {"budget": [0, 2, 4, 8], "trust": [0.5, 0.8]})   # one row per mechanism × grid point

For reward-rule factors no episodes need re-running: re-score instead (:meth:`Results.rescore`).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
from pydantic import BaseModel

from .results import Results
from .runner import Experiment


def label(v: Any) -> Any:
    """Short, readable cell value for a grid setting (non-default fields of pydantic models)."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, BaseModel):
        fields = type(v).model_fields
        diff = {k: getattr(v, k) for k, f in fields.items() if getattr(v, k) != f.get_default(call_default_factory=True)}
        return f"{type(v).__name__}({', '.join(f'{k}={label(x)}' for k, x in diff.items())})"
    if hasattr(v, "id"):
        return str(v.id)
    return repr(v)


def default_analysis(res: Results, gt: str = "correct") -> pd.DataFrame:
    """IC (ASD / alignment / frontier on trainable roles) joined with outcome metrics, per mechanism."""
    from ..analysis.diagnostics import outcome_metrics
    from ..analysis.ic import ic_report

    d = res.df(trainable_only=True)
    parts = []
    if f"gt_{gt}" in d and d[f"gt_{gt}"].notna().any():
        parts.append(ic_report(d, gt=gt))
    om = outcome_metrics(res)
    if len(om):
        parts.append(om)
    if not parts:
        return pd.DataFrame({"mechanism": sorted({r.mechanism for r in res.records})})
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="mechanism", how="outer")
    return out


def sweep(
    factory: Callable[..., "Experiment | Results"],
    grid: dict[str, Sequence[Any]],
    analyze: Callable[[Results], pd.DataFrame] | None = None,
    labels: dict[str, Callable[[Any], Any]] | None = None,
    keep: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[tuple, Results]]:
    """Run ``factory(**point)`` for every point of the ``grid`` (Cartesian product); analyse each
    run (default: :func:`default_analysis`) and add the grid values as columns. ``keep`` also
    returns the Results per grid point."""
    analyze = analyze or default_analysis
    tables, kept = [], {}
    for combo in itertools.product(*grid.values()):
        point = dict(zip(grid, combo))
        exp = factory(**point)
        res = exp.run() if isinstance(exp, Experiment) else exp
        t = analyze(res).copy()
        for k, v in point.items():
            t[k] = (labels or {}).get(k, label)(v)
        tables.append(t)
        if keep:
            kept[tuple(label(v) for v in combo)] = res
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    front = [k for k in grid if k in table.columns]
    table = table[front + [c for c in table.columns if c not in front]]
    return (table, kept) if keep else table


__all__ = ["sweep", "default_analysis", "label"]
