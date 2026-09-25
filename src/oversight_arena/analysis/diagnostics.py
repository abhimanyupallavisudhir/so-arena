"""Beyond incentive compatibility: outcome quality, reward-signal quality, judge biases,
compliance of simulated behaviours, and cost."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .stats import bootstrap_ci


def _df(results) -> pd.DataFrame:
    return results if isinstance(results, pd.DataFrame) else results.df()


def outcome_metrics(results, by: Sequence[str] = ("mechanism",)) -> pd.DataFrame:
    """Principal-level quality: decision accuracy, judge P(correct), log score, calibration (ECE)."""
    ep = results.episodes_df() if hasattr(results, "episodes_df") else results
    rows = []
    for keys, g in ep.groupby(list(by)):
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        for col, name in [("gt_decision_correct[_outcome]", "accuracy"), ("gt_judge_p_correct[_outcome]", "p_correct"),
                          ("gt_judge_p_correct[_outcome_log]", "log_score")]:
            if col in g and g[col].notna().any():
                est, lo, hi = bootstrap_ci(g[col].dropna().to_numpy())
                row.update({name: est, f"{name}_lo": lo, f"{name}_hi": hi})
        if "gt_judge_p_correct[_outcome]" in g and g["gt_judge_p_correct[_outcome]"].notna().any():
            row["ece"] = ece(g)
        row["episodes"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def ece(ep: pd.DataFrame, bins: int = 10) -> float:
    """Expected calibration error of the judge's confidence in its decision (any number of
    options): confidence = the judge's largest option probability, accuracy = whether the
    decision was correct."""
    pcols = [c for c in ep.columns if c.startswith("p[")]
    if not pcols or "gt_decision_correct[_outcome]" not in ep:
        return float("nan")
    conf = ep[pcols].max(axis=1).to_numpy(float)
    correct = ep["gt_decision_correct[_outcome]"].to_numpy(float)
    ok = ~(np.isnan(conf) | np.isnan(correct))
    conf, correct = conf[ok], correct[ok]
    if not len(conf):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    err = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            err += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(err)


def reward_snr(results, roles: str | Sequence[str] | None = None, strategy_col: str = "strategy_name",
               by: Sequence[str] = ("mechanism",)) -> pd.DataFrame:
    """Signal-to-noise of the reward as a *training signal*.

    Within each task: variance of per-strategy mean rewards (signal an optimiser can climb)
    vs. mean within-strategy variance across seeds (noise). Also the fraction of reward
    variance explained by tasks (irrelevant to the agent's choice)."""
    d = _df(results)
    if roles is not None:
        roles = [roles] if isinstance(roles, str) else list(roles)
        d = d[d["role"].isin(roles)]
    d = d.dropna(subset=["reward"])
    rows = []
    for keys, g in d.groupby(list(by)):
        sig, noi = [], []
        for _, t in g.groupby("task"):
            means = t.groupby(strategy_col)["reward"].mean()
            vars_ = t.groupby(strategy_col)["reward"].var(ddof=1).dropna()
            if len(means) > 1:
                sig.append(means.var(ddof=1))
            if len(vars_):
                noi.append(vars_.mean())
        total = g["reward"].var(ddof=1)
        task_share = g.groupby("task")["reward"].mean().var(ddof=1) / total if total and total > 0 else np.nan
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        s = float(np.mean(sig)) if sig else np.nan
        n = float(np.mean(noi)) if noi else np.nan
        row.update(signal_var=s, noise_var=n, snr=s / n if n and n > 0 else np.nan, task_variance_share=task_share)
        rows.append(row)
    return pd.DataFrame(rows)


def position_bias(results, roles: Sequence[str] = ("debater_a", "debater_b"), gt: str = "correct") -> pd.DataFrame:
    """Reward advantage of the first vs second slot at equal ground truth (debate order/slot bias)."""
    d = _df(results)
    col = f"gt_{gt}"
    d = d[d["role"].isin(roles)].dropna(subset=["reward", col])
    rows = []
    for mech, g in d.groupby("mechanism"):
        diffs = []
        for val in (0.0, 1.0):
            a = g[(g["role"] == roles[0]) & (g[col] == val)]["reward"]
            b = g[(g["role"] == roles[1]) & (g[col] == val)]["reward"]
            if len(a) and len(b):
                diffs.append(a.mean() - b.mean())
        rows.append({"mechanism": mech, "slot_advantage": float(np.mean(diffs)) if diffs else np.nan})
    return pd.DataFrame(rows)


def _names(results) -> dict[tuple[str, str], str]:
    """Unique mechanism names (configurations sharing a display name are told apart)."""
    if hasattr(results, "mechanism_names"):
        return results.mechanism_names()
    from ..experiment.results import Results

    return Results(list(results)).mechanism_names()


def option_label_bias(results, label: str = "A") -> pd.DataFrame:
    """Bias toward an option *label*: the judge's mean probability on ``label`` minus how often
    ``label`` is actually correct (needs ``results.tasks``). Also split by whether it is correct."""
    rows = []
    names = _names(results)
    for mech in sorted(set(names.values())):
        ps, cs = [], []
        for r in results.records:
            t = results.tasks.get(r.task_id)
            probs = r.outcome.get("probs") or {}
            if names[(r.mechanism, r.mechanism_hash)] != mech or t is None or label not in probs or not t.has_values():
                continue
            ps.append(float(probs[label]))
            cs.append(float(label in t.correct_ids()))
        if not ps:
            continue
        p, c = np.array(ps), np.array(cs)
        rows.append({"mechanism": mech, "label": label, "mean_p": p.mean(), "share_correct": c.mean(), "bias": p.mean() - c.mean(),
                     "mean_p_when_correct": p[c == 1].mean() if (c == 1).any() else np.nan,
                     "mean_p_when_incorrect": p[c == 0].mean() if (c == 0).any() else np.nan, "episodes": len(p)})
    return pd.DataFrame(rows)


def length_bias(results, roles: str | Sequence[str] | None = None, gt: str = "correct",
                by: Sequence[str] = ("mechanism",)) -> pd.DataFrame:
    """Within-task Spearman correlation between reward and words written, *holding GT fixed* —
    a classic judge exploit (verbosity)."""
    from scipy import stats as sps

    d = _df(results)
    if roles is not None:
        roles = [roles] if isinstance(roles, str) else list(roles)
        d = d[d["role"].isin(roles)]
    col = f"gt_{gt}"
    d = d.dropna(subset=["reward"])
    rows = []
    for keys, g in d.groupby(list(by)):
        rhos = []
        groups = g.groupby(["task", col]) if col in g else g.groupby("task")
        for _, t in groups:
            if len(t) > 2 and t["n_words"].nunique() > 1 and t["reward"].nunique() > 1:
                rhos.append(sps.spearmanr(t["n_words"], t["reward"]).statistic)
        est, lo, hi = bootstrap_ci(rhos)
        row = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        row.update(length_rho=est, lo=lo, hi=hi, groups=len(rhos))
        rows.append(row)
    return pd.DataFrame(rows)


def compliance(results, roles: str | Sequence[str] | None = None) -> pd.DataFrame:
    """Did simulated behaviours do what they were told? (answer = assigned target) and parse errors."""
    recs = results.records
    roles_set = None if roles is None else ({roles} if isinstance(roles, str) else set(roles))
    rows = []
    names = _names(results)
    for mech in sorted(set(names.values())):
        ok = n = perr = 0
        for r in recs:
            if names[(r.mechanism, r.mechanism_hash)] != mech:
                continue
            ans = r.outcome.get("answers") or {}
            for role, b in r.bound.items():
                if roles_set and role not in roles_set:
                    continue
                if b.target is not None and role in ans:
                    n += 1
                    ok += int(ans[role] == b.target)
            perr += sum(1 for e in r.transcript.entries if e.data.get("_parse_error"))
        rows.append({"mechanism": mech, "answer_matches_target": ok / n if n else np.nan, "checked": n, "parse_errors": perr})
    return pd.DataFrame(rows)


def cost_summary(results) -> pd.DataFrame:
    ep = results.episodes_df()
    return ep.groupby("mechanism").agg(
        episodes=("episode", "count"), tokens=("tokens", "sum"), cost_usd=("cost_usd", "sum"),
        tokens_per_episode=("tokens", "mean"), gt_channel_cost=("gt_channel_cost", "sum"),
    ).reset_index()
