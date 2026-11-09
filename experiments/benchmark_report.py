"""Report of the general model benchmark on the conceptual-validation testbed (``benchmark_solo.py``).

Reads the benchmark's rows (``<run>/rows``) and the panel preview's solo rows, and writes to
``docs/results/benchmark/``:

* ``benchmark.md`` tables (the narrative report is ``docs/benchmark.md``);
* figures ``fig*.png`` / ``.pdf``;
* ``models.csv`` (accuracy and Brier per model, item-bootstrap 95% CIs, residual from the fit on the AA index),
  ``families.csv`` (per testbed family: how its accuracies track the AA index), ``family_accuracy.csv``,
  ``effort.csv`` (GPT-5.6 Sol ladder), ``consensus_misses.csv`` (items nearly every strong model misses),
  ``item_set.csv`` (the stratified item list, with every model's solo correctness);
* ``data/rows.csv.gz`` and ``data/retest.csv.gz``: the compact data every table is computed from (``--from-bundle``
  reruns the report without the run directory).

Uncertainty: per-model numbers resample items (2,000 item-bootstrap draws). Relations across models (the fit on the
AA index, its correlation and the shaded band) resample models and items together, since with 20 models the choice
of models matters more than item noise.

    python experiments/benchmark_report.py --run runs/benchmark
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "benchmark"
BUNDLE = OUT / "data"
B = 2000
SEED = 0

# The strong tier: the six strongest distinct models (GPT-5.6 Sol at its highest effort), AA index >= 33.8. The weak
# tier: every configuration under AA 10, the range of the oversight study's judges.
STRONG = ["gemini-3.8-flash@low", "kimi-k3@low", "grok-4.6@low", "claude-opus-5@low", "gpt-5.6-sol@high",
          "gpt-6-astra@low"]
WEAK = ["llama-3.1-8b", "gpt-4o-mini", "qwen3-14b", "llama-3.1-70b", "mistral-small-4"]
LADDER = ["gpt-5.6-sol@none", "gpt-5.6-sol@low", "gpt-5.6-sol@medium", "gpt-5.6-sol@high"]
STRATA = ["gap", "easy", "contested", "hard"]
EXCLUDE = ("unstable", "noisy")
# DTBench's authors ask that it not be republished in plain text: its questions are referred to by id only.
NO_TEXT = ("dtbench/",)


# ----------------------------------------------------------------------------------------------- data


def model_table() -> pd.DataFrame:
    from decisive_panel import panel_config

    rows = []
    for m in panel_config()["agents"]:
        path = m["model"].removeprefix("openrouter/")
        rows.append({"model": m["name"], "aa": float(m["aa_index"]), "effort": m.get("reasoning_effort") or "-",
                     "reasoning": m.get("reasoning_effort") not in (None, "none"),
                     "developer": path.split("/")[0]})
    return pd.DataFrame(rows)


def item_table(run: Path) -> pd.DataFrame:
    """id, family, domain, label, position in the panel's fixed order, question text (not for DTBench), and the
    label's evidence (the experiment's measurement or the computed quantity)."""
    from benchmark_solo import bench_items

    rows = []
    for i, it in enumerate(bench_items(run)):
        fam = it.metadata.get("family", it.domain)
        gt = it.ground_truth
        hide = fam.startswith(NO_TEXT)
        rows.append({"item": it.id, "order": i, "family": fam, "domain": fam.split("/")[0],
                     "split": it.metadata.get("split"), "label": it.true_label,
                     "label_text": None if hide else {a.label: a.text for a in it.answers}.get(it.true_label),
                     "question": None if hide else it.question,
                     "options": None if hide else " | ".join(f"{a.label}: {a.text}" for a in it.answers),
                     "evidence": json.dumps((gt.data if gt else None) or {}, default=str)[:600]})
    return pd.DataFrame(rows)


def load_rows(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One sample per (model, item): this benchmark's rows, plus the preview's first sample (``arm == True``) on
    its 300 items for every model whose policy is unchanged. Also the preview's second sample of those models
    (test-retest)."""
    from benchmark_solo import CHANGED, preview_rows

    cols = ["model", "item", "correct", "p_true", "parse_ok", "cost"]
    files = sorted((run / "rows").glob("*.csv"))
    bench = pd.concat([pd.read_csv(f) for f in files]) if files else pd.DataFrame(columns=cols)
    bench = bench.drop_duplicates(["model", "item"], keep="first").assign(source="benchmark")[cols + ["source"]]
    pv = preview_rows()
    pv = pv[~pv.model.isin(CHANGED)]
    first = pv[pv.arm == True].assign(source="preview")[cols + ["source"]]  # noqa: E712
    second = pv[pv.arm == False][cols]  # noqa: E712
    rows = pd.concat([first, bench]).drop_duplicates(["model", "item"], keep="first")
    return rows.reset_index(drop=True), second.reset_index(drop=True)


def save_bundle(rows: pd.DataFrame, retest: pd.DataFrame, items: pd.DataFrame, bundle: Path = BUNDLE) -> None:
    bundle.mkdir(parents=True, exist_ok=True)
    rows.to_csv(bundle / "rows.csv.gz", index=False, float_format="%.6g")
    retest.to_csv(bundle / "retest.csv.gz", index=False, float_format="%.6g")
    items.drop(columns=["question"]).to_csv(bundle / "items.csv.gz", index=False)


def load_bundle(bundle: Path = BUNDLE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(bundle / "rows.csv.gz"), pd.read_csv(bundle / "retest.csv.gz"),
            pd.read_csv(bundle / "items.csv.gz"))


class Matrix:
    """Models x items arrays of correctness, probability on the truth and Brier score, on the items every model
    answered (a complete design: each bootstrap draw compares models on the same items)."""

    def __init__(self, rows: pd.DataFrame, models: list[str], items: pd.DataFrame):
        have = rows[rows.model.isin(models)]
        counts = have.groupby("item").model.nunique()
        keep = [i for i in items.item if counts.get(i, 0) == len(models)]
        self.models, self.items = list(models), keep
        self.meta = items.set_index("item").loc[keep].reset_index()
        piv = lambda c: have.pivot(index="model", columns="item", values=c).loc[models, keep].to_numpy(float)  # noqa
        self.correct = piv("correct")
        self.p = piv("p_true")
        self.parse = piv("parse_ok").astype(float)
        self.brier = (1.0 - self.p) ** 2

    @property
    def n(self) -> int:
        return len(self.items)

    def weights(self, b: int = B, seed: int = SEED, mask: np.ndarray | None = None) -> np.ndarray:
        """Item-bootstrap weights (b x items): multinomial counts over the items in ``mask`` (default all)."""
        rng = np.random.default_rng(seed)
        idx = np.flatnonzero(mask) if mask is not None else np.arange(self.n)
        w = np.zeros((b, self.n))
        for k in range(b):
            w[k] += np.bincount(idx[rng.integers(0, len(idx), len(idx))], minlength=self.n)
        return w


def ci(draws: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    return np.nanpercentile(draws, 2.5, axis=axis), np.nanpercentile(draws, 97.5, axis=axis)


def boot_means(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted means of each row of ``x`` (models x items) under each bootstrap weight vector: draws x models."""
    return (w @ x.T) / w.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------------------- analyses


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, icept = np.polyfit(x, y, 1)
    return float(slope), float(icept)


def model_metrics(mx: Matrix, mt: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    acc_b, bri_b = boot_means(mx.correct, w), boot_means(mx.brier, w)
    conf = np.maximum(mx.p, 1 - mx.p)  # confidence in the answer the model leans to
    out = mt.set_index("model").loc[mx.models].reset_index()
    out["accuracy"] = mx.correct.mean(1)
    out["acc_lo"], out["acc_hi"] = ci(acc_b)
    out["brier"] = mx.brier.mean(1)
    out["brier_lo"], out["brier_hi"] = ci(bri_b)
    out["overconfidence"] = conf.mean(1) - mx.correct.mean(1)
    out["parse_rate"] = mx.parse.mean(1)
    for col, link in (("accuracy", "logit"), ("brier", "log")):
        f, finv = LINKS[link]
        s, c = fit_line(out.aa.to_numpy(), f(out[col].to_numpy()))
        out[f"{col}_fit"] = finv(s * out.aa + c)
        out[f"{col}_resid"] = out[col] - out[f"{col}_fit"]
    # residual in units of the model's own item-bootstrap SE: is it more than measurement noise?
    out["accuracy_resid_z"] = out.accuracy_resid / acc_b.std(0)
    return out


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


LINKS = {"identity": (lambda v: v, lambda v: v), "logit": (logit, expit), "log": (np.log, np.exp)}


def relation(mx: Matrix, x: np.ndarray, y_of: np.ndarray, w: np.ndarray, grid: np.ndarray, seed: int = 1, *,
             link: str = "identity") -> dict:
    """Fit of a per-model score on ``x`` with a bootstrap over models and items together: the point fit, Pearson r
    and Spearman rho with their 95% CIs, the slope per 10 index points (on the link scale) and a 95% band over
    ``grid``. ``link="logit"`` fits accuracy on the log-odds scale, so the curve saturates below 100% instead of a
    straight line overshooting it."""
    from scipy.stats import spearmanr

    f, finv = LINKS[link]
    rng = np.random.default_rng(seed)
    y = y_of.mean(1)
    s, c = fit_line(x, f(y))
    yb = boot_means(y_of, w)
    rs, rhos, slopes, lines = [], [], [], []
    for k in range(len(w)):
        ms = rng.integers(0, len(x), len(x))
        if len(np.unique(x[ms])) < 3:
            continue
        sk, ck = fit_line(x[ms], f(yb[k, ms]))
        slopes.append(sk)
        rs.append(np.corrcoef(x[ms], yb[k, ms])[0, 1])
        rhos.append(spearmanr(x[ms], yb[k, ms])[0])
        lines.append(finv(sk * grid + ck))
    lo, hi = ci(np.array(lines))
    pct = lambda v, q: float(np.nanpercentile(v, q))  # noqa: E731
    return {"link": link, "slope10": 10 * s, "intercept": c, "r": float(np.corrcoef(x, y)[0, 1]),
            "r_lo": pct(rs, 2.5), "r_hi": pct(rs, 97.5), "rho": float(spearmanr(x, y)[0]),
            "rho_lo": pct(rhos, 2.5), "rho_hi": pct(rhos, 97.5),
            "slope10_lo": 10 * pct(slopes, 2.5), "slope10_hi": 10 * pct(slopes, 97.5),
            "grid": grid, "fit": finv(s * grid + c), "band_lo": lo, "band_hi": hi,
            "predict": lambda xs: finv(s * np.asarray(xs) + c)}


def family_metrics(mx: Matrix, aa: np.ndarray, overall: np.ndarray, *, b: int = B) -> tuple[pd.DataFrame,
                                                                                               pd.DataFrame]:
    """Per testbed family: each model's accuracy, its correlation and slope with the AA index (item-bootstrap
    CIs within the family), and how far the models' family accuracies scatter around the family's own linear fit
    on the index beyond what item sampling explains (``excess_sd``: the 'something else' a family measures). Also
    the correlation with the rest of the benchmark (accuracy on every other family)."""
    rows, accs = [], {}
    fams = mx.meta.family.to_numpy()
    for k, fam in enumerate(sorted(set(fams))):
        mask = fams == fam
        n = int(mask.sum())
        acc = mx.correct[:, mask].mean(1)
        accs[fam] = acc
        w = mx.weights(b, seed=100 + k, mask=mask)
        ab = boot_means(mx.correct, w)
        rb = np.array([np.corrcoef(aa, a)[0, 1] if a.std() > 0 else np.nan for a in ab])
        sb = np.array([fit_line(aa, a)[0] for a in ab])
        s, c = fit_line(aa, acc)
        resid = acc - (s * aa + c)
        noise = np.mean(ab.var(0))  # item-sampling variance of a model's family accuracy
        excess = max(0.0, resid.var(ddof=2) - noise)
        rest = (overall * mx.n - acc * n) / (mx.n - n)
        top = np.argsort(-np.abs(resid))[:3]
        rows.append({"family": fam, "domain": fam.split("/")[0], "items": n, "mean_acc": acc.mean(),
                     "weak_acc": np.nan, "strong_acc": np.nan,
                     "r_aa": np.corrcoef(aa, acc)[0, 1] if acc.std() > 0 else np.nan,
                     "r_lo": np.nanpercentile(rb, 2.5), "r_hi": np.nanpercentile(rb, 97.5),
                     "slope10": 10 * s, "slope10_lo": 10 * np.percentile(sb, 2.5),
                     "slope10_hi": 10 * np.percentile(sb, 97.5),
                     "r_rest": np.corrcoef(rest, acc)[0, 1] if acc.std() > 0 else np.nan,
                     "resid_sd": resid.std(ddof=2), "noise_sd": np.sqrt(noise), "excess_sd": np.sqrt(excess),
                     "largest_residuals": "; ".join(f"{mx.models[i]} {100 * resid[i]:+.0f}" for i in top)})
    fam_df = pd.DataFrame(rows)
    acc_df = pd.DataFrame(accs, index=mx.models).T
    for tier, names in (("weak_acc", WEAK), ("strong_acc", STRONG)):
        fam_df[tier] = acc_df.loc[fam_df.family, [m for m in names if m in mx.models]].mean(1).to_numpy()
    return fam_df.sort_values("r_aa", ascending=False), acc_df


def effort_metrics(mx: Matrix, rows: pd.DataFrame, w: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GPT-5.6 Sol at none / low / medium / high: accuracy, Brier and cost per item; paired differences between
    steps (item bootstrap; exact McNemar test on the items where the two disagree)."""
    from scipy.stats import binomtest

    idx = [mx.models.index(m) for m in LADDER if m in mx.models]
    acc_b, bri_b = boot_means(mx.correct, w), boot_means(mx.brier, w)
    cost = rows[rows.source == "benchmark"].groupby("model").cost.mean()
    lev = []
    for i in idx:
        m = mx.models[i]
        lo, hi = ci(acc_b[:, i])
        blo, bhi = ci(bri_b[:, i])
        lev.append({"model": m, "effort": m.split("@")[1], "accuracy": mx.correct[i].mean(), "acc_lo": lo,
                    "acc_hi": hi, "brier": mx.brier[i].mean(), "brier_lo": blo, "brier_hi": bhi,
                    "cost_per_item": cost.get(m, np.nan)})
    pairs = []
    for a, b_ in [(idx[k], idx[k + 1]) for k in range(len(idx) - 1)] + [(idx[0], idx[-1]), (idx[1], idx[-1])]:
        d = acc_b[:, b_] - acc_b[:, a]
        db = bri_b[:, b_] - bri_b[:, a]
        ca, cb = mx.correct[a] >= 0.5, mx.correct[b_] >= 0.5
        up, down = int((cb & ~ca).sum()), int((ca & ~cb).sum())
        p = binomtest(up, up + down).pvalue if up + down else 1.0
        pairs.append({"from": mx.models[a].split("@")[1], "to": mx.models[b_].split("@")[1],
                      "d_accuracy": mx.correct[b_].mean() - mx.correct[a].mean(), "d_acc_lo": ci(d)[0],
                      "d_acc_hi": ci(d)[1], "d_brier": mx.brier[b_].mean() - mx.brier[a].mean(),
                      "d_brier_lo": ci(db)[0], "d_brier_hi": ci(db)[1], "fixed": up, "broken": down,
                      "mcnemar_p": p})
    return pd.DataFrame(lev), pd.DataFrame(pairs)


def tiers(mx: Matrix) -> pd.DataFrame:
    """Per item: how many strong / weak / all models get it right alone, and which."""
    c = mx.correct >= 0.5
    s = [mx.models.index(m) for m in STRONG if m in mx.models]
    wk = [mx.models.index(m) for m in WEAK if m in mx.models]
    df = mx.meta[["item", "family", "domain", "order"]].copy()
    df["strong_correct"] = c[s].sum(0)
    df["weak_correct"] = c[wk].sum(0)
    df["n_correct"] = c.sum(0)
    df["strong_right"] = [", ".join(mx.models[i] for i in s if c[i, j]) for j in range(mx.n)]
    return df


def poisson_binomial(p: np.ndarray) -> np.ndarray:
    """P(k successes) for independent trials with success probabilities ``p``."""
    dist = np.array([1.0])
    for q in p:
        dist = np.convolve(dist, [1 - q, q])
    return dist


def ceiling(mx: Matrix, tier: pd.DataFrame) -> dict:
    """How far the strong tier's misses concentrate on the same items beyond what independent errors at each
    model's accuracy would give, the panel's oracle accuracy, and the strong tier's majority vote."""
    s = [mx.models.index(m) for m in STRONG if m in mx.models]
    c = mx.correct[s] >= 0.5
    k = c.sum(0)
    # independence baseline, family by family (models' accuracies differ by family)
    expected = np.zeros(len(s) + 1)
    fams = mx.meta.family.to_numpy()
    for fam in set(fams):
        m = fams == fam
        expected += m.sum() * poisson_binomial(c[:, m].mean(1))
    return {"strong": [mx.models[i] for i in s], "k_observed": np.bincount(k, minlength=len(s) + 1),
            "k_expected": expected, "oracle_strong": float((k > 0).mean()),
            "oracle_all": float(((mx.correct >= 0.5).sum(0) > 0).mean()),
            "majority_strong": float((k > len(s) / 2).mean()), "consensus_miss": int((k <= 1).sum())}


def retest(second: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Two independent samples of the same model on the same item (the preview's two draws): agreement."""
    a = rows[rows.source == "preview"].merge(second, on=["model", "item"], suffixes=("", "_2"))
    g = a.assign(agree=(a.correct >= 0.5) == (a.correct_2 >= 0.5),
                 both_wrong=(a.correct < 0.5) & (a.correct_2 < 0.5)).groupby("model")
    return g.agg(items=("item", "size"), acc_1=("correct", "mean"), acc_2=("correct_2", "mean"),
                 agreement=("agree", "mean"), both_wrong=("both_wrong", "mean")).reset_index()


def stratify(tier: pd.DataFrame, review: pd.DataFrame | None, seed: int = 7) -> pd.DataFrame:
    """The stratified item list. Strata (6 strong models, 5 weak ones, each alone):

    * ``gap``: at least 5 of 6 strong models right, at most 2 of 5 weak ones (what oversight must transfer);
    * ``easy``: at least 5 strong and at least 3 weak right;
    * ``contested``: 2 to 4 strong right;
    * ``hard``: at most one strong model right, label reviewed or plausible.

    Items whose label is unstable on a rerun or noisy (``label_audit``) are left out. Order: the strata in turn (gap, easy,
    contested, hard), and within a stratum the domains in turn (random within a domain, fixed seed), so the first
    ``4k`` items hold ``k`` of each stratum while every stratum lasts, spread over the domains."""
    df = tier.copy()
    s, w = df.strong_correct, df.weak_correct
    df["stratum"] = np.select([s >= 5, s >= 2], [np.where(w <= 2, "gap", "easy"), "contested"], "hard")
    df["label_review"] = ""
    if review is not None and len(review):
        df = df.merge(review[["item", "verdict"]], on="item", how="left")
        df["label_review"] = df.pop("verdict").fillna("")
        df = df[~df.label_review.isin(EXCLUDE)]
    rng = np.random.default_rng(seed)
    order = {}
    for st in STRATA:
        part = df[df.stratum == st]
        queues = [list(rng.permutation(g.item.to_numpy())) for _, g in part.groupby("domain")]
        rng.shuffle(queues)
        seq = []
        while any(queues):
            for q in queues:
                if q:
                    seq.append(q.pop(0))
        order[st] = seq
    ranked = []
    while any(order.values()):
        for st in STRATA:
            if order[st]:
                ranked.append(order[st].pop(0))
    df = df.set_index("item").loc[ranked].reset_index()
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df



# Hand checks of the strong tier's consensus misses outside the experiment families (docs/benchmark.md).
SIEVE_CHECKED = ["numbers-prime_race-s0-0000", "numbers-prime_race-s0-0038", "numbers-prime_race-s0-0052",
                 "numbers-prime_race-s0-0062", "numbers-prime_race-s0-0077", "numbers-prime_race-s0-0096",
                 "numbers-prime_race-s0-0134"]
READ_CHECKED = ["dtbench-25.1", "dtbench-66.2EDT", "dtbench-12EDT", "dtbench-88.2INF", "dtbench-39.1CDT"]
COMPUTED = ("decisions/", "socialchoice/", "numbers/")
NOISY = 0.8  # items whose label is less likely than this to have the sign of the true effect leave the item set


def label_audit(items: pd.DataFrame, rel: pd.DataFrame, reps: pd.DataFrame) -> pd.DataFrame:
    """Per item: how its label was made, the probability it has the sign of the true effect (1 for computed and
    expert labels), the outcome of a rerun where one was made, and a verdict:

    * ``computed`` / ``expert``: decision theory, social choice, number theory / DTBench;
    * ``sieve-checked``, ``read``: consensus misses checked by hand (all labels found correct);
    * ``unstable``: an independent rerun of the experiment gave the opposite label;
    * ``noisy``: P(label = true effect) < 0.8 from the measurement's own spread;
    * ``replicated`` / ``weak`` (the rerun was void: an effect below the void gap) / ``measured`` (not rerun)."""
    df = items[["item", "family"]].merge(rel[["item", "p_label_true", "p_replicate"]], on="item", how="left")
    st = reps.set_index("item").status if len(reps) else pd.Series(dtype=str)
    df["rerun"] = df.item.map(st).fillna("")
    expert = df.family.str.startswith("dtbench/")
    computed = df.family.str.startswith(COMPUTED)
    df.loc[expert | computed, "p_label_true"] = 1.0
    v = np.select([df.item.isin(SIEVE_CHECKED), df.item.isin(READ_CHECKED), computed, expert,
                   df.rerun == "flipped", df.p_label_true < NOISY, df.rerun == "same", df.rerun == "void"],
                  ["sieve-checked", "read", "computed", "expert", "unstable", "noisy", "replicated", "weak"],
                  "measured")
    df["verdict"] = v
    return df


# ---------------------------------------------------------------------------------------- figures


def place_labels(ax, xs, ys, texts, *, pad: float = 3.0, avoid=()):
    """Direct labels beside points, each at the first of eight positions around its point that overlaps no
    placed label, no point and stays inside the axes; a hairline leader joins a label placed farther out."""
    import matplotlib.transforms as mt

    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([np.r_[xs, [a[0] for a in avoid]],
                                                  np.r_[ys, [a[1] for a in avoid]]]))
    box_ax = ax.get_window_extent(r)
    placed = []
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]
    order = np.argsort(-np.asarray(ys))
    for i in order:
        best = None
        for dist in (5, 11, 18, 26):
            for dx, dy in dirs:
                ha = "left" if dx > 0 else "right" if dx < 0 else "center"
                va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
                off = (dx * dist, dy * dist * 0.8)
                t = ax.annotate(texts[i], (xs[i], ys[i]), xytext=off, textcoords="offset points", ha=ha, va=va,
                                fontsize=6.6, color="#52514e")
                bb = t.get_window_extent(r).expanded(1.0, 1.0)
                bb_p = mt.Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
                ok = (box_ax.contains(bb.x0, bb.y0) and box_ax.contains(bb.x1, bb.y1)
                      and not any(bb_p.overlaps(o) for o in placed)
                      and not any(bb_p.contains(px, py) for px, py in pts))
                t.remove()
                if ok:
                    best = (off, ha, va, dist)
                    break
            if best:
                break
        off, ha, va, dist = best or ((5, 0), "left", "center", 5)
        t = ax.annotate(texts[i], (xs[i], ys[i]), xytext=off, textcoords="offset points", ha=ha, va=va,
                        fontsize=6.6, color="#52514e",
                        arrowprops=dict(arrowstyle="-", color="#c9c7c0", lw=0.6, shrinkA=0, shrinkB=3)
                        if dist > 5 else None)
        placed.append(t.get_window_extent(r))


def short(m: str) -> str:
    return (m.replace("claude-", "").replace("gpt-5.6-sol", "Sol").replace("@", " ")
            .replace("gemini-3.8-flash", "gemini-3.8-flash").replace("llama-3.1-", "llama-"))


def fig_aa(models: pd.DataFrame, rels: dict, out: Path, n_items: int):
    import eic_style as st

    fig, axs = st.plt.subplots(1, 2, figsize=(10.2, 5.4))
    for ax, col, lab in ((axs[0], "accuracy", "Accuracy alone"), (axs[1], "brier", "Brier score (lower is better, log scale)")):
        st.axes(ax)
        rel = rels[col]
        ax.fill_between(rel["grid"], rel["band_lo"], rel["band_hi"], color=st.INK2, alpha=0.10, lw=0, zorder=1)
        ax.plot(rel["grid"], rel["fit"], color=st.INK2, lw=1.4, zorder=2)
        lad = models.set_index("model").loc[[m for m in LADDER if m in set(models.model)]]
        ax.plot(lad.aa, lad[col], color=st.CONTEXT, lw=1.2, zorder=2)
        for reasoning, color in ((False, st.BLUE), (True, st.ORANGE)):
            d = models[models.reasoning == reasoning]
            ax.vlines(d.aa, d[f"{'acc' if col == 'accuracy' else col}_lo"],
                      d[f"{'acc' if col == 'accuracy' else col}_hi"], color=color, lw=1.2, zorder=3)
            ax.plot(d.aa, d[col], "o", color=color, ms=7, mec=st.SURFACE, mew=1.5, zorder=4)
        ax.set_xlabel("Artificial Analysis Intelligence Index v4.3")
        ax.set_title(lab)
        if col == "accuracy":
            ax.set_ylim(0.5, 0.97)
            ax.yaxis.set_major_formatter(st.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        else:
            ax.set_yscale("log")
            ticks = [0.04, 0.06, 0.1, 0.15, 0.2, 0.3]
            ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
            ax.minorticks_off()
            ax.set_ylim(0.03, 0.42)
        ax.set_xlim(3, 49)
        place_labels(ax, models.aa.to_numpy(), models[col].to_numpy(), [short(m) for m in models.model])
        ax.text(0.98, 0.04 if col == "accuracy" else 0.96,
                f"Pearson r = {rel['r']:.2f} [{rel['r_lo']:.2f}, {rel['r_hi']:.2f}]\n"
                f"Spearman \u03c1 = {rel['rho']:.2f} [{rel['rho_lo']:.2f}, {rel['rho_hi']:.2f}]",
                transform=ax.transAxes, ha="right", va="bottom" if col == "accuracy" else "top", fontsize=7.2,
                color=st.INK2)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [Line2D([], [], marker="o", ls="", color=st.BLUE, mec=st.SURFACE, ms=7, label="Non-reasoning"),
               Line2D([], [], marker="o", ls="", color=st.ORANGE, mec=st.SURFACE, ms=7, label="Reasoning"),
               Line2D([], [], color=st.CONTEXT, lw=1.2, label="GPT-5.6 Sol effort ladder"),
               (Line2D([], [], color=st.INK2, lw=1.4), Patch(color=st.INK2, alpha=0.10)),
               ]
    fig.legend(handles, ["Non-reasoning", "Reasoning", "GPT-5.6 Sol effort ladder (none, low, medium, high)",
                         "Fit (log-odds of accuracy, log Brier), 95% band"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Each model alone on {n_items:,} testbed items, against the Artificial Analysis index", x=0.01,
                 ha="left", fontsize=10.5, fontweight="bold", color=st.INK)
    fig.text(0.01, 0.925, "One sample per model and item. Vertical bars: 95% item-bootstrap CIs (mostly hidden by the "
             "dots). Band: 95% bootstrap over models and items.", ha="left", fontsize=7.5, color=st.INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    st.save(fig, out, "fig1_aa")


def fig_families(fam: pd.DataFrame, out: Path):
    import eic_style as st

    d = fam.sort_values("r_aa")
    fig, axs = st.plt.subplots(1, 2, figsize=(10.2, 0.21 * len(d) + 1.6), sharey=True,
                               gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(d))
    labels = [f"{f}  ({n})" for f, n in zip(d.family, d["items"])]
    ax = st.axes(axs[0], grid="x")
    st.zero(ax)
    big = d["items"].to_numpy() >= 30
    for sel, color in ((big, st.BLUE), (~big, st.CONTEXT)):
        ax.hlines(y[sel], d.r_lo[sel], d.r_hi[sel], color=color, lw=1.4)
        ax.plot(d.r_aa[sel], y[sel], "o", color=color, ms=6, mec=st.SURFACE, mew=1.5)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("r between family accuracy and AA index (20 models)")
    ax.set_title("Does the family track general capability?")
    ax = st.axes(axs[1], grid="x")
    ax.barh(y, d.excess_sd * 100, height=0.55, color=[st.BLUE if b else st.CONTEXT for b in big])
    ax.set_xlabel("SD of residuals beyond item noise (accuracy points)")
    ax.set_title("What else it measures")
    from matplotlib.lines import Line2D

    fig.legend([Line2D([], [], marker="o", ls="", color=st.BLUE, ms=6), Line2D([], [], marker="o", ls="",
                                                                                   color=st.CONTEXT, ms=6)],
               ["30 or more items (95% item-bootstrap CI)", "Fewer than 30 items"], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    st.save(fig, out, "fig2_families")


def fig_heatmap(acc: pd.DataFrame, models: pd.DataFrame, fam: pd.DataFrame, out: Path):
    import eic_style as st
    from matplotlib.colors import LinearSegmentedColormap

    order_m = models.sort_values("aa").model.tolist()
    order_f = fam.sort_values("r_aa", ascending=False).family.tolist()
    a = acc.loc[order_f, order_m].to_numpy()
    cmap = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95",
                                                      "#0d366b"])
    fig, ax = st.plt.subplots(figsize=(10.2, 0.22 * len(order_f) + 2.4))
    im = ax.imshow(a, cmap=cmap, vmin=0.3, vmax=1.0, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(range(len(order_m)), [f"{short(m)} ({models.set_index('model').aa[m]:g})" for m in order_m],
                  rotation=55, ha="right", fontsize=6.8)
    n = fam.set_index("family")["items"]
    ax.set_yticks(range(len(order_f)), [f"{f} ({n[f]})" for f in order_f], fontsize=6.8)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, f"{100 * a[i, j]:.0f}", ha="center", va="center", fontsize=5.4,
                    color="#ffffff" if a[i, j] > 0.66 else st.INK2)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, format=st.matplotlib.ticker.PercentFormatter(1.0, 0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=st.INK2)
    ax.set_title("Accuracy alone by testbed family (rows: by correlation with the AA index; columns: by index)")
    fig.tight_layout()
    st.save(fig, out, "fig3_family_heatmap")


def fig_effort(lev: pd.DataFrame, fam_gain: pd.DataFrame, out: Path):
    import eic_style as st

    fig, axs = st.plt.subplots(1, 3, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    x = lev.cost_per_item.to_numpy() * 100  # cents
    for ax, col, lo, hi, lab in ((axs[0], "accuracy", "acc_lo", "acc_hi", "Accuracy alone"),
                                 (axs[1], "brier", "brier_lo", "brier_hi", "Brier score")):
        st.axes(ax)
        ax.fill_between(x, lev[lo], lev[hi], color=st.BLUE, alpha=0.10, lw=0)
        ax.plot(x, lev[col], "-o", color=st.BLUE, ms=7, mec=st.SURFACE, mew=1.5)
        ax.vlines(x, lev[lo], lev[hi], color=st.BLUE, lw=1.2)
        ax.set_xscale("log")
        ax.set_xticks([0.3, 1, 3, 10], ["0.3", "1", "3", "10"])
        ax.minorticks_off()
        ax.set_xlabel("Cost per item (US cents, log scale)")
        ax.set_title(lab)
        for xi, yi, e in zip(x, lev[col], lev.effort):
            ax.annotate(e, (xi, yi), xytext=(0, 9 if col == "accuracy" else -9), textcoords="offset points",
                        ha="center", va="bottom" if col == "accuracy" else "top", fontsize=7, color=st.INK2)
        if col == "accuracy":
            ax.yaxis.set_major_formatter(st.matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.margins(x=0.15, y=0.2)
    ax = st.axes(axs[2], grid="x")
    d = fam_gain.sort_values("gain")
    y = np.arange(len(d))
    st.zero(ax)
    ax.hlines(y, d.lo * 100, d.hi * 100, color=st.BLUE, lw=1.2)
    ax.plot(d.gain * 100, y, "o", color=st.BLUE, ms=6, mec=st.SURFACE, mew=1.5)
    ax.set_yticks(y, [f"{f} ({n})" for f, n in zip(d.domain, d["items"])], fontsize=7)
    ax.set_xlabel("Accuracy gain, none to high effort (points, 95% CI)")
    ax.set_title("Where effort helps (by domain)")
    fig.suptitle("GPT-5.6 Sol: reasoning effort", x=0.01, ha="left", fontsize=10.5, fontweight="bold",
                 color=st.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    st.save(fig, out, "fig4_effort")


def fig_ceiling(ceil: dict, fam_ceiling: pd.DataFrame, out: Path):
    import eic_style as st

    fig, axs = st.plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = st.axes(axs[0], grid="y")
    k = np.arange(len(ceil["k_observed"]))
    ax.bar(k - 0.19, ceil["k_observed"], width=0.36, color=st.BLUE, label="Observed")
    ax.bar(k + 0.19, ceil["k_expected"], width=0.36, color=st.CONTEXT, label="If errors were independent")
    for xi, v in zip(k - 0.19, ceil["k_observed"]):
        ax.annotate(f"{v:,}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.6, color=st.INK2)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([0, 10, 100, 1000], ["0", "10", "100", "1,000"])
    ax.set_xticks(k)
    ax.set_xlabel(f"Strong models right alone (of {len(k) - 1})")
    ax.set_ylabel("Items")
    ax.set_title("The strong models miss the same items")
    ax = st.axes(axs[1], grid="x")
    d = fam_ceiling.sort_values("strong_acc", ascending=False)
    y = np.arange(len(d))
    ax.hlines(y, d.strong_acc * 100, d.ceiling.fillna(d.strong_acc) * 100, color=st.CONTEXT, lw=1.2, zorder=1)
    ax.plot(d.ceiling * 100, y, "|", color=st.INK2, ms=10, mew=1.6, zorder=2)
    ax.plot(d.strong_acc * 100, y, "o", color=st.BLUE, ms=6.5, mec=st.SURFACE, mew=1.5, zorder=3)
    ax.set_yticks(y, [f"{f} ({n})" + ("  no estimate" if pd.isna(c) else "")
                      for f, n, c in zip(d.family, d["items"], d.ceiling)], fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Strong tier vs. label-noise ceiling (experiment families)")
    from matplotlib.lines import Line2D

    fig.legend([Line2D([], [], marker="s", ls="", color=st.BLUE, ms=7),
                Line2D([], [], marker="s", ls="", color=st.CONTEXT, ms=7),
                Line2D([], [], marker="o", ls="", color=st.BLUE, ms=6.5),
                Line2D([], [], marker="|", ls="", color=st.INK2, ms=10, mew=1.6)],
               ["Observed", "If errors were independent", "Strong tier, mean of 6 models",
                "Label-noise ceiling: expected accuracy knowing every true effect"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    st.save(fig, out, "fig5_ceiling")


# ---------------------------------------------------------------------------------------- report


def pct(v, d=1):
    return "-" if pd.isna(v) else f"{100 * v:.{d}f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.astype(object).itertuples(index=False):  # iterrows would upcast integer columns to float
        lines.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(lines)


def main(argv=None):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/benchmark")
    ap.add_argument("--from-bundle", action="store_true", help="use the committed compact data")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.from_bundle:
        rows, second, items = load_bundle(out / "data")
    else:
        run = Path(a.run)
        rows, second = load_rows(run)
        items = item_table(run)
        save_bundle(rows, second, items, out / "data")
        from benchmark_labels import reliability

        reliability().to_csv(out / "data" / "label_reliability.csv", index=False, float_format="%.4g")
        rp = run / "replications.jsonl"
        reps = pd.read_json(rp, lines=True) if rp.exists() else pd.DataFrame(columns=["item", "status"])
        reps.to_csv(out / "data" / "replications.csv", index=False, float_format="%.4g")
    rel = pd.read_csv(out / "data" / "label_reliability.csv")
    reps = pd.read_csv(out / "data" / "replications.csv")
    audit = label_audit(items, rel, reps)
    mt = model_table()
    names = mt.model.tolist()
    mx = Matrix(rows, names, items)
    w = mx.weights()
    models = model_metrics(mx, mt, w)
    grid = np.linspace(4, 48, 89)
    aa = models.aa.to_numpy()
    rels = {"accuracy": relation(mx, aa, mx.correct, w, grid, link="logit"), "brier": relation(mx, aa, mx.brier, w, grid, link="log")}
    fam, fam_acc = family_metrics(mx, aa, models.accuracy.to_numpy())
    lev, steps = effort_metrics(mx, rows, w)
    tier = tiers(mx)
    ceil = ceiling(mx, tier)
    rt = retest(second, rows)
    # effort gain by domain (none -> high), item bootstrap within domain
    i0, i1 = mx.models.index(LADDER[0]), mx.models.index(LADDER[-1])
    gains = []
    for k, (dom, g) in enumerate(mx.meta.groupby("domain")):
        m = mx.meta.domain.isin([dom]).to_numpy()
        diff = mx.correct[i1, m] - mx.correct[i0, m]
        rng = np.random.default_rng(300 + k)
        bs = [diff[rng.integers(0, m.sum(), m.sum())].mean() for _ in range(B)]
        gains.append({"domain": dom, "items": int(m.sum()), "gain": diff.mean(), "lo": np.percentile(bs, 2.5),
                      "hi": np.percentile(bs, 97.5)})
    fam_gain = pd.DataFrame(gains)
    # consensus misses
    meta = items.set_index("item")
    aud = audit.set_index("item")
    miss = tier[tier.strong_correct <= 1].merge(
        items[["item", "label", "label_text", "options", "evidence"] + (["question"] if "question" in items else [])],
        on="item").merge(audit[["item", "p_label_true", "rerun", "verdict"]], on="item", how="left")
    # label-noise ceiling: the accuracy a forecaster who knew every true effect would expect, family by family
    s_idx = [mx.models.index(m) for m in STRONG]
    t2 = tier.assign(strong_acc=mx.correct[s_idx].mean(0), p=tier.item.map(aud.p_label_true),
                     miss=tier.strong_correct <= 1)
    fam_ceiling = (t2.groupby("family").agg(items=("item", "size"), ceiling=("p", "mean"),
                                             strong_acc=("strong_acc", "mean"), misses=("miss", "sum"))
                   .reset_index())
    fam_ceiling["experiment"] = ~fam_ceiling.family.str.startswith(COMPUTED + ("dtbench/",))
    overall_ceiling = float(t2.p.fillna(1.0).mean())
    verdicts = pd.crosstab(audit.family, audit.verdict)
    miss_verdicts = miss.verdict.value_counts()
    groups = Path(a.run) / "replication_items.csv"
    if groups.exists():
        pd.read_csv(groups).to_csv(out / "data" / "replication_items.csv", index=False)
    gp = out / "data" / "replication_items.csv"
    rg = reps.merge(pd.read_csv(gp), on="item") if gp.exists() and len(reps) else reps.assign(group="-")
    rep_table = (rg.groupby(["group", "status"]).size().unstack(fill_value=0).reset_index()
                 if len(rg) else pd.DataFrame())
    iset = stratify(tier, audit)
    iset = iset.merge(audit[["item", "p_label_true", "rerun"]], on="item", how="left")
    corr = pd.DataFrame((mx.correct >= 0.5).astype(int).T, columns=mx.models)
    corr.insert(0, "item", mx.items)
    iset = iset.merge(corr, on="item", how="left")

    # tables
    models.to_csv(out / "models.csv", index=False, float_format="%.4f")
    fam.to_csv(out / "families.csv", index=False, float_format="%.4f")
    fam_acc.to_csv(out / "family_accuracy.csv", float_format="%.4f")
    lev.to_csv(out / "effort.csv", index=False, float_format="%.4f")
    steps.to_csv(out / "effort_steps.csv", index=False, float_format="%.4g")
    fam_gain.to_csv(out / "effort_by_domain.csv", index=False, float_format="%.4f")
    rt.to_csv(out / "retest.csv", index=False, float_format="%.4f")
    miss.sort_values(["family", "order"]).to_csv(out / "consensus_misses.csv", index=False)
    iset.to_csv(out / "item_set.csv", index=False)
    audit.to_csv(out / "label_audit.csv", index=False, float_format="%.4g")
    fam_ceiling.to_csv(out / "ceiling_by_family.csv", index=False, float_format="%.4f")
    summary = {"items": mx.n, "models": len(names), "fit": {k: {kk: vv for kk, vv in v.items()
                                                               if not isinstance(vv, np.ndarray) and not callable(vv)}
                                                           for k, v in rels.items()},
               "ceiling": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ceil.items()},
               "strata": iset.stratum.value_counts().to_dict(), "label_noise_ceiling": overall_ceiling,
               "excluded_from_item_set": audit.verdict.isin(EXCLUDE).sum(),
               "consensus_miss_verdicts": miss_verdicts.to_dict(),
               "spend": json.loads((Path(a.run) / "spend.json").read_text())
               if (Path(a.run) / "spend.json").exists() else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))

    # figures
    fig_aa(models, rels, out, mx.n)
    fig_families(fam, out)
    fig_heatmap(fam_acc, models, fam, out)
    fig_effort(lev, fam_gain, out)
    fig_ceiling(ceil, fam_ceiling[fam_ceiling.experiment], out)

    # tables in markdown
    t = models.sort_values("aa").assign(
        AA=lambda d: d.aa.map("{:g}".format),
        accuracy_=lambda d: [f"{pct(v)} [{pct(lo)}, {pct(hi)}]" for v, lo, hi in zip(d.accuracy, d.acc_lo, d.acc_hi)],
        brier_=lambda d: [f"{v:.3f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in zip(d.brier, d.brier_lo, d.brier_hi)],
        resid=lambda d: [f"{100 * v:+.1f} ({z:+.1f} SE)" for v, z in zip(d.accuracy_resid, d.accuracy_resid_z)],
        overconf=lambda d: d.overconfidence.map(lambda v: f"{100 * v:+.1f}"),
        parse=lambda d: d.parse_rate.map(pct))
    lines = [f"# Benchmark tables ({mx.n:,} items, {len(names)} model configurations)", "",
             "Generated by `experiments/benchmark_report.py`; see `docs/benchmark.md` for the report.", "",
             "## Models", "",
             md_table(t[["model", "AA", "accuracy_", "brier_", "resid", "overconf", "parse"]].rename(columns={
                 "accuracy_": "accuracy [95% CI]", "brier_": "Brier [95% CI]",
                 "resid": "accuracy residual from AA fit (pts)", "overconf": "overconfidence (pts)",
                 "parse": "parsed"})), "",
             f"Fit on the AA index: accuracy Pearson r = {rels['accuracy']['r']:.2f} [{rels['accuracy']['r_lo']:.2f}, "
             f"{rels['accuracy']['r_hi']:.2f}], Spearman \u03c1 = {rels['accuracy']['rho']:.2f} "
             f"[{rels['accuracy']['rho_lo']:.2f}, {rels['accuracy']['rho_hi']:.2f}]; logistic fit "
             f"{rels['accuracy']['slope10']:+.2f} log-odds per 10 index points [{rels['accuracy']['slope10_lo']:+.2f}, "
             f"{rels['accuracy']['slope10_hi']:+.2f}] (residuals above are from this fit). Brier Pearson r = "
             f"{rels['brier']['r']:.2f} [{rels['brier']['r_lo']:.2f}, {rels['brier']['r_hi']:.2f}], Spearman \u03c1 = "
             f"{rels['brier']['rho']:.2f}; log-linear fit, Brier x{np.exp(rels['brier']['slope10']):.2f} per 10 index points.", "",
             "## Families", "",
             md_table(fam.assign(
                 r=lambda d: [f"{v:.2f} [{lo:.2f}, {hi:.2f}]" for v, lo, hi in zip(d.r_aa, d.r_lo, d.r_hi)],
                 slope=lambda d: [f"{100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
                                  for v, lo, hi in zip(d.slope10, d.slope10_lo, d.slope10_hi)],
                 weak=lambda d: d.weak_acc.map(pct), strong=lambda d: d.strong_acc.map(pct),
                 rest=lambda d: d.r_rest.map("{:.2f}".format),
                 excess=lambda d: d.excess_sd.map(lambda v: f"{100 * v:.1f}"))[
                 ["family", "items", "weak", "strong", "r", "slope", "rest", "excess", "largest_residuals"]].rename(
                 columns={"r": "r with AA [95% CI]", "slope": "pts per 10 AA", "rest": "r with rest of benchmark",
                          "excess": "excess SD (pts)", "weak": "weak tier", "strong": "strong tier",
                          "largest_residuals": "largest residuals (pts)"})), "",
             "## Reasoning effort (GPT-5.6 Sol)", "",
             md_table(lev.assign(acc=lambda d: [f"{pct(v)} [{pct(lo)}, {pct(hi)}]" for v, lo, hi in
                                                 zip(d.accuracy, d.acc_lo, d.acc_hi)],
                                 bri=lambda d: [f"{v:.3f} [{lo:.3f}, {hi:.3f}]" for v, lo, hi in
                                                zip(d.brier, d.brier_lo, d.brier_hi)],
                                 cents=lambda d: d.cost_per_item.map(lambda v: f"{100 * v:.2f}"))[
                 ["effort", "acc", "bri", "cents"]].rename(columns={"acc": "accuracy [95% CI]",
                                                                    "bri": "Brier [95% CI]",
                                                                    "cents": "cents per item"})), "",
             md_table(steps.assign(
                 d_acc=lambda d: [f"{100 * v:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]" for v, lo, hi in
                                  zip(d.d_accuracy, d.d_acc_lo, d.d_acc_hi)],
                 d_bri=lambda d: [f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]" for v, lo, hi in
                                  zip(d.d_brier, d.d_brier_lo, d.d_brier_hi)],
                 p=lambda d: d.mcnemar_p.map("{:.2g}".format))[
                 ["from", "to", "d_acc", "d_bri", "fixed", "broken", "p"]].rename(
                 columns={"d_acc": "accuracy change (pts)", "d_bri": "Brier change", "fixed": "items fixed",
                          "broken": "items broken", "p": "McNemar p"})), "",
             "## Ceiling", "",
             f"Strong tier: {', '.join(ceil['strong'])}. Items by number of strong models right alone "
             f"(observed / expected if their errors were independent within each family):", "",
             md_table(pd.DataFrame({"strong right": np.arange(len(ceil['k_observed'])),
                                    "observed": np.asarray(ceil["k_observed"]).astype(int),
                                    "independent": np.round(ceil["k_expected"], 1)})), "",
             f"Oracle (some strong model right): {pct(ceil['oracle_strong'])}; some of the 20 right: "
             f"{pct(ceil['oracle_all'])}; strong-tier majority vote: {pct(ceil['majority_strong'])}.", "",
             f"Label-noise ceiling over all {mx.n:,} items (computed and expert labels count as certain): "
             f"{pct(overall_ceiling)}.", "",
             md_table(fam_ceiling[fam_ceiling.experiment].sort_values("ceiling").assign(
                 ceiling=lambda d: d.ceiling.map(pct), strong_acc=lambda d: d.strong_acc.map(pct))[
                 ["family", "items", "strong_acc", "ceiling", "misses"]].rename(
                 columns={"strong_acc": "strong tier", "ceiling": "label-noise ceiling",
                          "misses": "consensus misses"})), "",
             "Consensus misses by label verdict (`label_audit.csv`):", "",
             md_table(miss_verdicts.rename_axis("verdict").reset_index(name="items")), "",
             "Reruns (`benchmark_labels.py replicate`), consensus misses and a control sample the strong tier gets "
             "right:", "",
             md_table(rep_table), "",
             "Test-retest on the preview's 300 items (two independent samples):", "",
             md_table(rt.assign(**{c: rt[c].map(pct) for c in ("acc_1", "acc_2", "agreement", "both_wrong")})), "",
             "## Stratified item set", "",
             md_table(iset.groupby(["stratum", "domain"]).size().unstack(fill_value=0).loc[
                 [s for s in STRATA if s in set(iset.stratum)]].T.reset_index()), ""]
    (out / "benchmark.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
