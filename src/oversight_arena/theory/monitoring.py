"""Monitor-based rewards and obfuscation (docs/theory.md, Proposition 4).

An actor chooses honest work (measured success $s$), overt misbehaviour ($s+\\Delta$, alarm
probability $d_o$) or covert misbehaviour ($s+\\Delta-\\kappa$, alarm probability $d_c$); honest work
raises false alarms with probability $d_h$. Reward = measured success $- \\lambda\\cdot$alarm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonitorParams:
    s: float = 0.5
    gain: float = 0.3  # Delta
    hide_cost: float = 0.05  # kappa
    d_overt: float = 1.0
    d_covert: float = 0.1
    d_honest: float = 0.05


def expected_rewards(lam: float, p: MonitorParams) -> dict[str, float]:
    return {"honest": p.s - lam * p.d_honest, "overt": p.s + p.gain - lam * p.d_overt,
            "covert": p.s + p.gain - p.hide_cost - lam * p.d_covert}


def best_response(lam: float, p: MonitorParams) -> str:
    r = expected_rewards(lam, p)
    return max(r, key=r.get)


def thresholds(p: MonitorParams) -> dict[str, float]:
    """Penalty at which overt misbehaviour turns covert, and at which misbehaviour stops
    (inf if covert misbehaviour is no easier to detect than honest work)."""
    to_covert = p.hide_cost / (p.d_overt - p.d_covert) if p.d_overt > p.d_covert else float("inf")
    deter = (p.gain - p.hide_cost) / (p.d_covert - p.d_honest) if p.d_covert > p.d_honest else float("inf")
    return {"overt_to_covert": to_covert, "deterred": deter}


def sweep(lams: np.ndarray, p: MonitorParams) -> pd.DataFrame:
    rows = []
    for lam in lams:
        a = best_response(float(lam), p)
        rows.append({"penalty": float(lam), "action": a, "misbehaviour": float(a != "honest"),
                     "alarm_rate_when_misbehaving": {"overt": p.d_overt, "covert": p.d_covert}.get(a, np.nan)})
    return pd.DataFrame(rows)
