"""Incentive-compatibility metrics.

Notation: for a role, :math:`u` is the mechanism's reward and :math:`v` the ground-truth value of the
role's behaviour. A mechanism is incentive-compatible (in the local, training-relevant sense) when
behaviours with higher :math:`v` receive higher :math:`u`.

* **ASD** (agent score difference; Pallavi Sudhir et al. 2025): :math:`E[u \\mid v=+1] - E[u \\mid v=-1]`,
  paired by item. With log-score rewards it is :math:`\\log p_\\top - \\log p_\\bot`.
* **Graded ASD**: :math:`\\sum_c (v_c - \\bar v)\\, u_c` per item - the covariance of reward with value
  across behaviour cases; reduces to ASD for two cases valued +/-1.
* **Pairwise IC** (concordance): :math:`P(u_i > u_j \\mid v_i > v_j)` over pairs of behaviours on the
  same item; for binary values this is the AUROC of the reward as a detector of good behaviour.
* **Covariance / first-order policy improvement**: :math:`\\mathrm{Cov}(u, v)` under the sampling
  distribution is exactly :math:`\\frac{d}{d\\beta} E_{\\pi_\\beta}[v]` at :math:`\\beta = 0` for the
  exponentially tilted policy :math:`\\pi_\\beta \\propto \\pi_0 e^{\\beta u}` (see docs/theory.md).
* **Incentive gap recovered**: :math:`(\\mathrm{ASD}_P - \\mathrm{ASD}_{\\text{naive}})/(\\mathrm{ASD}_{\\max} - \\mathrm{ASD}_{\\text{naive}})`.
* **Label efficiency**: :math:`1/(1-\\rho^2)`, how many ground-truth labels one audited episode is worth
  when the mechanism reward is used as a control variate for audits.

All functions take the tidy frame from :func:`so_arena.analysis.frames.role_frame` and return tidy
frames with cluster-bootstrap confidence intervals (clusters = items).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sps

from so_arena.core.mechanism import Episode
from so_arena.core.rewards import JudgeScore, rescore


# ----------------------------------------------------------------------------------- bootstrap

def bootstrap_mean_ci(values: np.ndarray, *, n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (math.nan, math.nan)
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def cluster_bootstrap(clusters: Sequence[pd.DataFrame], stat: Callable[[pd.DataFrame], float], *,
                      n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of ``stat`` over resamples of whole clusters (e.g. items)."""
    if not clusters:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    vals = []
    n = len(clusters)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = pd.concat([clusters[i] for i in idx], ignore_index=True)
        v = stat(sample)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return (math.nan, math.nan)
    lo, hi = np.quantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def _select(df: pd.DataFrame, roles: Sequence[str] | None) -> pd.DataFrame:
    if df.empty:
        return df
    d = df[df["reward"].notna()]
    if roles is not None:
        d = d[d["role"].isin(list(roles))]
    else:
        d = d[(d["kind"] == "agent") & (d["trainable"])]
    return d


def _groups(df: pd.DataFrame, by: Sequence[str]):
    by = [b for b in by if b in df.columns]
    if not by:
        yield (), df
        return
    for key, g in df.groupby(by, dropna=False, sort=True):
        yield (key if isinstance(key, tuple) else (key,)), g


# ----------------------------------------------------------------------------------- ASD

def asd(df: pd.DataFrame, *, roles: Sequence[str] | None = None, by: Sequence[str] = ("mechanism",),
        value_col: str = "value", reward_col: str = "reward", pos: float = 1.0, neg: float = -1.0,
        n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Agent score difference, paired by item: mean over items of E[u | v=pos] - E[u | v=neg].

    Rows of the selected roles are pooled (e.g. both debaters). Items lacking either arm are dropped.
    """
    d = _select(df, roles)
    out = []
    for key, g in _groups(d, by):
        g = g[g[value_col].isin([pos, neg])]
        per = g.groupby(["item_id", value_col])[reward_col].mean().unstack(value_col)
        if pos not in per.columns or neg not in per.columns:
            continue
        per = per.dropna(subset=[pos, neg])
        diffs = (per[pos] - per[neg]).to_numpy()
        if len(diffs) == 0:
            continue
        lo, hi = bootstrap_mean_ci(diffs, n_boot=n_boot, ci=ci, seed=seed)
        se = float(np.std(diffs, ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else math.nan
        row = dict(zip([b for b in by if b in d.columns], key))
        row.update({
            "asd": float(diffs.mean()), "ci_low": lo, "ci_high": hi, "se": se, "n_items": len(diffs),
            "reward_true": float(per[pos].mean()), "reward_false": float(per[neg].mean()),
            "p_asd_le_0": float(sps.ttest_1samp(diffs, 0.0, alternative="greater").pvalue) if len(diffs) > 2 and np.std(diffs) > 0 else math.nan,
        })
        out.append(row)
    return pd.DataFrame(out)


def asd_by_transform(episodes: Sequence[Episode], transforms: Sequence[str] = ("log", "brier", "prob", "accuracy"),
                     **kwargs) -> pd.DataFrame:
    """ASD under several proper-score transforms, re-scoring judge-based rewards post hoc."""
    from so_arena.analysis.frames import role_frame

    frames = []
    for t in transforms:
        eps = rescore(episodes, JudgeScore(t))
        # keep the original ground truth
        df = asd(role_frame(eps), **kwargs)
        df.insert(0, "transform", t)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def graded_asd(df: pd.DataFrame, *, roles: Sequence[str] | None = None, by: Sequence[str] = ("mechanism",),
               value_col: str = "value", reward_col: str = "reward", normalize: bool = False,
               n_boot: int = 2000, ci: float = 0.95, seed: int = 0) -> pd.DataFrame:
    """Graded ASD: per item, :math:`\\sum_c (v_c - \\bar v) u_c` over behaviour cases c (mean reward per case).

    With ``normalize=True`` the per-item sum is divided by :math:`\\sum_c (v_c-\\bar v)^2`, giving the
    regression slope of reward on value (scale-free in v).
    """
    d = _select(df, roles)
    d = d[d[value_col].notna()]
    out = []
    for key, g in _groups(d, by):
        per_item = []
        for _, gi in g.groupby("item_id"):
            cases = gi.groupby(value_col)[reward_col].mean()
            if len(cases) < 2:
                continue
            v = cases.index.to_numpy(dtype=float)
            u = cases.to_numpy(dtype=float)
            dv = v - v.mean()
            s = float((dv * u).sum())
            if normalize:
                s /= float((dv ** 2).sum())
            per_item.append(s)
        if not per_item:
            continue
        arr = np.array(per_item)
        lo, hi = bootstrap_mean_ci(arr, n_boot=n_boot, ci=ci, seed=seed)
        row = dict(zip([b for b in by if b in d.columns], key))
        row.update({"graded_asd": float(arr.mean()), "ci_low": lo, "ci_high": hi, "n_items": len(arr)})
        out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------- alignment

def pairwise_concordance(g: pd.DataFrame, value_col: str = "value", reward_col: str = "reward") -> tuple[float, int]:
    """P(u_i > u_j | v_i > v_j) over within-item pairs (ties in u count 1/2)."""
    num, n = 0.0, 0
    for _, gi in g.groupby("item_id"):
        u = gi[reward_col].to_numpy(dtype=float)
        v = gi[value_col].to_numpy(dtype=float)
        if len(u) < 2:
            continue
        du = u[:, None] - u[None, :]
        dv = v[:, None] - v[None, :]
        mask = dv > 0
        if not mask.any():
            continue
        num += float((du[mask] > 0).sum() + 0.5 * (du[mask] == 0).sum())
        n += int(mask.sum())
    return (num / n if n else math.nan), n


def incentive_alignment(df: pd.DataFrame, *, roles: Sequence[str] | None = None, by: Sequence[str] = ("mechanism",),
                        value_col: str = "value", reward_col: str = "reward", n_boot: int = 500,
                        seed: int = 0) -> pd.DataFrame:
    """Correlation-type IC measures between reward and ground-truth value.

    Returns Pearson/Spearman correlations (overall and within-item, i.e. after removing item means),
    covariance, the regression slope of u on v, pairwise concordance with a cluster-bootstrap CI,
    and the label-efficiency multiplier :math:`1/(1-\\rho^2)` of the within-item correlation.
    """
    d = _select(df, roles)
    d = d[d[value_col].notna()]
    out = []
    for key, g in _groups(d, by):
        if len(g) < 3 or g[value_col].nunique() < 2:
            continue
        u = g[reward_col].to_numpy(dtype=float)
        v = g[value_col].to_numpy(dtype=float)
        uw = u - g.groupby("item_id")[reward_col].transform("mean").to_numpy(dtype=float)
        vw = v - g.groupby("item_id")[value_col].transform("mean").to_numpy(dtype=float)
        pear = float(np.corrcoef(u, v)[0, 1]) if np.std(u) > 0 else math.nan
        spear = float(sps.spearmanr(u, v).statistic) if np.std(u) > 0 else math.nan
        within = float(np.corrcoef(uw, vw)[0, 1]) if np.std(uw) > 0 and np.std(vw) > 0 else math.nan
        conc, npairs = pairwise_concordance(g, value_col, reward_col)
        clusters = [gi for _, gi in g.groupby("item_id")]
        lo, hi = cluster_bootstrap(clusters, lambda s: pairwise_concordance(s, value_col, reward_col)[0],
                                   n_boot=n_boot, seed=seed) if len(clusters) > 1 else (math.nan, math.nan)
        cov = float(np.cov(u, v)[0, 1])
        slope = cov / float(np.var(v, ddof=1)) if np.var(v) > 0 else math.nan
        row = dict(zip([b for b in by if b in d.columns], key))
        row.update({
            "pearson": pear, "spearman": spear, "within_item_corr": within, "cov": cov, "slope": slope,
            "concordance": conc, "concordance_ci_low": lo, "concordance_ci_high": hi, "n_pairs": npairs,
            "label_efficiency": (1 / (1 - within ** 2)) if np.isfinite(within) and abs(within) < 1 else math.inf,
            "n": len(g), "n_items": g["item_id"].nunique(),
        })
        out.append(row)
    return pd.DataFrame(out)


def incentive_gap_recovered(asd_protocol: float, asd_naive: float, asd_max: float = 2.0) -> float:
    """Share of the gap between a no-agent baseline and a perfect judge closed by the protocol.

    ``asd_max=2`` is the ceiling on the Brier scale (a perfect binary judge).
    """
    denom = asd_max - asd_naive
    return (asd_protocol - asd_naive) / denom if denom else math.nan


# ----------------------------------------------------------------------------------- expected scores

def open_probs(scores: dict[str, float], beta: float) -> dict[str, float]:
    """Propensity to argue for each position when free to choose: softmax of scores at temperature beta.

    ``beta=0`` is argmax (ties split), ``beta=inf`` is uniform (SOlib's convention).
    """
    keys = list(scores)
    if not keys:
        return {}
    vals = np.array([scores[k] for k in keys], dtype=float)
    if beta == 0:
        m = vals.max()
        w = (np.abs(vals - m) < 1e-12).astype(float)
    elif math.isinf(beta):
        w = np.ones_like(vals)
    else:
        z = (vals - vals.max()) / beta
        w = np.exp(z)
    w = w / w.sum()
    return dict(zip(keys, w.tolist()))


def expected_scores(df: pd.DataFrame, *, betas: Sequence[float] = (0.0, 1.0, math.inf),
                    roles: Sequence[str] | None = None, by: Sequence[str] = ("mechanism",),
                    judge_col: str = "judge_correct") -> pd.DataFrame:
    """Expected agent score and expected judge score/accuracy if agents chose positions by softmax(ASD/beta).

    Per item, the agent's mean reward for arguing each position determines its propensity to argue
    that position; the expected judge score weights the judge outcome of each position's episodes.
    """
    d = _select(df, roles)
    d = d[d["position"].notna()]
    out = []
    for key, g in _groups(d, by):
        for beta in betas:
            agent_scores, judge_scores, values = [], [], []
            for _, gi in g.groupby("item_id"):
                per = gi.groupby("position").agg(u=("reward", "mean"), j=(judge_col, "mean"), v=("value", "mean"))
                if len(per) < 2:
                    continue
                pi = open_probs(per["u"].to_dict(), beta)
                agent_scores.append(sum(pi[p] * per.loc[p, "u"] for p in pi))
                if per["j"].notna().all():
                    judge_scores.append(sum(pi[p] * per.loc[p, "j"] for p in pi))
                if per["v"].notna().all():
                    values.append(sum(pi[p] * per.loc[p, "v"] for p in pi))
            if not agent_scores:
                continue
            row = dict(zip([b for b in by if b in d.columns], key))
            row.update({
                "beta": beta,
                "expected_agent_score": float(np.mean(agent_scores)),
                "expected_judge_score": float(np.mean(judge_scores)) if judge_scores else math.nan,
                "expected_value": float(np.mean(values)) if values else math.nan,
                "n_items": len(agent_scores),
            })
            out.append(row)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------------- control-style

def judge_accuracy(df: pd.DataFrame, *, by: Sequence[str] = ("mechanism",), n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Accuracy and probability-on-truth of the final decision (the "control" measure), per group.

    Uses one row per episode (the first role row), so pooled roles are not double counted.
    """
    if df.empty or "judge_correct" not in df:
        return pd.DataFrame()
    d = df.drop_duplicates("episode_id")
    d = d[d["judge_correct"].notna()]
    out = []
    for key, g in _groups(d, by):
        acc = g.groupby("item_id")["judge_correct"].mean().to_numpy(dtype=float)
        ptrue = g.groupby("item_id")["judge_p_true"].mean().to_numpy(dtype=float)
        lo, hi = bootstrap_mean_ci(acc, n_boot=n_boot, seed=seed)
        row = dict(zip([b for b in by if b in d.columns], key))
        row.update({"accuracy": float(acc.mean()), "acc_ci_low": lo, "acc_ci_high": hi,
                    "p_true": float(np.nanmean(ptrue)), "n_items": len(acc), "n_episodes": len(g)})
        out.append(row)
    return pd.DataFrame(out)


def auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    a = np.asarray(scores_pos, dtype=float)
    b = np.asarray(scores_neg, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return math.nan
    diff = a[:, None] - b[None, :]
    return float((diff > 0).mean() + 0.5 * (diff == 0).mean())


def tpr_at_fpr(scores_pos: Sequence[float], scores_neg: Sequence[float], fpr: float = 0.01) -> float:
    """Detection rate of ``pos`` at a threshold giving false-positive rate ``fpr`` on ``neg``."""
    b = np.sort(np.asarray(scores_neg, dtype=float))
    if len(b) == 0:
        return math.nan
    thr = np.quantile(b, 1 - fpr)
    return float((np.asarray(scores_pos, dtype=float) > thr).mean())


def summary(df: pd.DataFrame, *, roles: Sequence[str] | None = None, by: Sequence[str] = ("mechanism",)) -> pd.DataFrame:
    """One table joining ASD, graded ASD, alignment, judge accuracy and cost per group."""
    parts = []
    a = asd(df, roles=roles, by=by)
    if not a.empty:
        parts.append(a[[*[b for b in by if b in a.columns], "asd", "ci_low", "ci_high", "n_items"]]
                     .rename(columns={"ci_low": "asd_ci_low", "ci_high": "asd_ci_high", "n_items": "asd_n_items"}))
    ia = incentive_alignment(df, roles=roles, by=by, n_boot=200)
    if not ia.empty:
        parts.append(ia[[*[b for b in by if b in ia.columns], "concordance", "within_item_corr", "label_efficiency"]])
    ja = judge_accuracy(df, by=by)
    if not ja.empty:
        parts.append(ja[[*[b for b in by if b in ja.columns], "accuracy", "p_true"]])
    if "cost_usd" in df and not df.empty:
        c = df.drop_duplicates("episode_id").groupby([b for b in by if b in df.columns])["cost_usd"].mean().reset_index()
        parts.append(c.rename(columns={"cost_usd": "cost_per_episode"}))
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, how="outer", on=[b for b in by if b in out.columns and b in p.columns])
    return out
