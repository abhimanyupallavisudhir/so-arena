"""Decisive study, tournament report: persuasiveness ratings and judge accuracy against them.

Ratings are fitted from every cross-play and self-play debate of ``decisive_tournament.py``: the judge's
probability on the honest debater's answer is a soft outcome of a Bradley-Terry model

    P(honest H beats liar L) = sigmoid(a + r_H - r_L),

with one persuasiveness rating r per model (mean zero) and a common honesty advantage a, fitted by maximum
likelihood over (judge, item, pairing) observations with a small ridge penalty; reported on the Elo scale
(400 / ln 10 per logit). CIs: bootstrap over items, refitting. Against the ratings:

* self-play debate accuracy (both debaters the same model; Khan et al.'s main plot),
* consultancy accuracy (the model as consultant, both arms),
* the model alone (with no judge) - its knowledge of the answers;

each pooled over the three judges, with item-bootstrap CIs, and the slope of accuracy on rating.

    python experiments/decisive_tournament_report.py runs/decisive [docs/results]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decisive_tournament import TOURNAMENT  # noqa: E402

ELO = 400 / np.log(10)


def fit_ratings(df: pd.DataFrame, models: list[str], ridge: float = 1e-3) -> tuple[dict[str, float], float]:
    """Bradley-Terry ratings (logits) with soft outcomes ``p_true`` and a common honesty advantage."""
    idx = {m: i for i, m in enumerate(models)}
    h = df.honest.map(idx).to_numpy()
    li = df.liar.map(idx).to_numpy()
    y = df.p_true.clip(1e-4, 1 - 1e-4).to_numpy(dtype=float)
    k = len(models)

    def nll(theta):
        r, a = theta[:k] - theta[:k].mean(), theta[k]
        z = a + r[h] - r[li]
        p = 1 / (1 + np.exp(-z))
        ll = y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)
        return -ll.sum() + ridge * (r ** 2).sum()

    res = minimize(nll, np.zeros(k + 1), method="L-BFGS-B")
    r = res.x[:k] - res.x[:k].mean()
    return {m: float(r[idx[m]]) for m in models}, float(res.x[k])


def boot_items(df: pd.DataFrame, n_boot: int, seed: int = 0):
    items = df.item.unique()
    rng = np.random.default_rng(seed)
    groups = {i: g for i, g in df.groupby("item")}
    for _ in range(n_boot):
        pick = rng.choice(items, len(items), replace=True)
        yield pd.concat([groups[i] for i in pick], ignore_index=True)


def item_mean_ci(frame: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    per = frame.groupby("item")[col].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    reps = per[rng.integers(0, len(per), (n_boot, len(per)))].mean(axis=1)
    return float(per.mean()), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def main(run: str, out: str = "docs/results", n_boot: int = 200) -> None:
    run, out = Path(run), Path(out)
    rows = pd.read_csv(run / "tournament" / "rows.csv")
    deb = rows[(rows.kind == "debate") & rows.p_true.notna()]
    models = [m for m in TOURNAMENT if m in set(deb.honest) | set(deb.liar)]
    r, a = fit_ratings(deb, models)
    boots = [fit_ratings(b, models) for b in boot_items(deb, n_boot)]
    ratings = pd.DataFrame({"model": models, "elo": [ELO * r[m] for m in models],
                            "elo_lo": [ELO * np.quantile([b[0][m] for b in boots], 0.025) for m in models],
                            "elo_hi": [ELO * np.quantile([b[0][m] for b in boots], 0.975) for m in models]})
    honesty_adv = (ELO * a, ELO * np.quantile([b[1] for b in boots], 0.025),
                   ELO * np.quantile([b[1] for b in boots], 0.975))
    acc = []
    for m in models:
        selfp = deb[(deb.honest == m) & (deb.liar == m)]
        con = rows[(rows.kind == "consultancy") & (rows.consultant == m)]
        solo = rows[(rows.kind == "solo") & (rows.model == m)]
        hon = deb[(deb.honest == m)].assign(win=lambda d: (d.p_true > 0.5).astype(float))
        lie = deb[(deb.liar == m)].assign(win=lambda d: (d.p_true < 0.5).astype(float))
        row = {"model": m}
        for name, f, col in (("self_play", selfp, "correct"), ("consultancy", con, "correct"),
                             ("alone", solo, "correct"), ("win_as_honest", hon, "win"), ("win_as_liar", lie, "win")):
            if len(f):
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = item_mean_ci(f, col)
                row[f"{name}_items"] = int(f.item.nunique())
        acc.append(row)
    acc = ratings.merge(pd.DataFrame(acc), on="model")
    # slope of accuracy on rating (per 100 Elo), with the bootstrap of ratings and accuracies over items
    slopes = {}
    for col in ("self_play", "consultancy"):
        x, y = acc.elo.to_numpy(), acc[col].to_numpy()
        slopes[col] = float(np.polyfit(x, y, 1)[0] * 100) if np.isfinite(y).all() else float("nan")
    matrix = deb.assign(win=(deb.p_true > 0.5).astype(float)).pivot_table(index="honest", columns="liar", values="win")
    matrix = matrix.reindex(index=models, columns=models)
    by_judge = (deb[deb.honest == deb.liar].groupby(["judge", "honest"]).correct.mean().unstack().reindex(columns=models))
    out.mkdir(parents=True, exist_ok=True)
    acc.to_csv(out / "decisive_tournament.csv", index=False)
    matrix.to_csv(out / "decisive_tournament_honest_wins.csv")
    by_judge.to_csv(out / "decisive_tournament_selfplay_by_judge.csv")
    figure(acc, out)
    print(acc.round(3).to_string(index=False))
    print(f"honesty advantage {honesty_adv[0]:.0f} Elo [{honesty_adv[1]:.0f}, {honesty_adv[2]:.0f}]; slopes per 100 Elo:",
          {k: round(v, 3) for k, v in slopes.items()})
    print(matrix.round(2))
    print(by_judge.round(2))
    (out / "decisive_tournament_summary.txt").write_text(
        f"honesty advantage (Elo): {honesty_adv[0]:.1f} [{honesty_adv[1]:.1f}, {honesty_adv[2]:.1f}]\n"
        f"accuracy slope per 100 Elo: self-play debate {slopes['self_play']:.4f}, consultancy {slopes['consultancy']:.4f}\n"
        f"debates: {len(deb)} judge verdicts on {deb.item.nunique()} items\n")


def figure(acc: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from decisive_report import INK, INK2, SERIES, _axes

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _axes(ax)
    for k, (col, name) in enumerate((("self_play", "debate (self-play)"), ("consultancy", "consultancy"),
                                     ("alone", "debater alone"))):
        if col not in acc:
            continue
        # percentile intervals need not contain the point estimate: clip the bars at it
        yerr = np.clip([acc[col] - acc[f"{col}_lo"], acc[f"{col}_hi"] - acc[col]], 0, None)
        xerr = np.clip([acc.elo - acc.elo_lo, acc.elo_hi - acc.elo], 0, None)
        ax.errorbar(acc.elo, acc[col], yerr=yerr, xerr=xerr, fmt="o", color=SERIES[k], ms=6, mec="white",
                    lw=1.2, label=name, alpha=0.9 if col != "alone" else 0.5)
        if col != "alone":
            b = np.polyfit(acc.elo, acc[col], 1)
            xs = np.linspace(acc.elo.min(), acc.elo.max(), 10)
            ax.plot(xs, np.polyval(b, xs), color=SERIES[k], lw=1, ls="--")
    for m, x, y in zip(acc.model, acc.elo, acc.self_play):
        ax.annotate(m, (x, y), fontsize=6.5, color=INK2, xytext=(4, -9), textcoords="offset points")
    ax.set_xlabel("debater persuasiveness (Elo, from the judges' verdicts)", color=INK2, fontsize=8.5)
    ax.set_ylabel("judge accuracy (pooled over 3 judges; 95% CI)", color=INK2, fontsize=8.5)
    ax.set_title("more persuasive debaters, more truthful verdicts?", color=INK, fontsize=9, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "decisive_tournament.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
