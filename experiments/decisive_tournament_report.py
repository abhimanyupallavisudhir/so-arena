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
from decisive_report import _found  # noqa: E402

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


def fit_role_ratings(df: pd.DataFrame, models: list[str], ridge: float = 1e-3) -> tuple[dict, dict, float]:
    """Role-specific Bradley-Terry: P(honest H beats liar L) = sigmoid(a + h_H - l_L); h and l mean zero."""
    idx = {m: i for i, m in enumerate(models)}
    h = df.honest.map(idx).to_numpy()
    li = df.liar.map(idx).to_numpy()
    y = df.p_true.clip(1e-4, 1 - 1e-4).to_numpy(dtype=float)
    k = len(models)

    def nll(theta):
        hh, ll, a = theta[:k] - theta[:k].mean(), theta[k:2 * k] - theta[k:2 * k].mean(), theta[2 * k]
        p = 1 / (1 + np.exp(-(a + hh[h] - ll[li])))
        return -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum() + ridge * ((hh ** 2).sum() + (ll ** 2).sum())

    x = minimize(nll, np.zeros(2 * k + 1), method="L-BFGS-B").x
    hh, ll = x[:k] - x[:k].mean(), x[k:2 * k] - x[k:2 * k].mean()
    return {m: float(hh[idx[m]]) for m in models}, {m: float(ll[idx[m]]) for m in models}, float(x[2 * k])


def committed_debates(run: Path, rows: pd.DataFrame) -> pd.DataFrame | None:
    """Tournament rows (debates and consultancies) whose liar the gpt-5 audit found committed, with the verdicts."""
    vf, rf = _found(run / "audit" / "verdicts_gpt-5.jsonl"), run / "audit" / "rows.csv.gz"
    if not (vf.exists() and rf.exists()):
        return None
    ar = pd.read_csv(rf, low_memory=False)
    ar = ar[ar.kind == "tournament"][["episode", "key"]]
    if ar.empty:
        return None
    v = pd.read_json(vf, lines=True).drop_duplicates("key", keep="last")[["key", "verdict"]]
    lab = ar.merge(v, on="key", how="inner").drop_duplicates("episode")[["episode", "verdict"]]
    return rows.merge(lab, on="episode", how="inner")


def table_for(deb: pd.DataFrame, rows: pd.DataFrame, models: list[str], n_boot: int) -> tuple[pd.DataFrame, tuple]:
    r, a = fit_ratings(deb, models)
    hr, lr, _ = fit_role_ratings(deb, models)
    boots = [fit_ratings(b, models) for b in boot_items(deb, n_boot)]
    out = pd.DataFrame({"model": models, "elo": [ELO * r[m] for m in models],
                        "elo_lo": [ELO * np.quantile([b[0][m] for b in boots], 0.025) for m in models],
                        "elo_hi": [ELO * np.quantile([b[0][m] for b in boots], 0.975) for m in models],
                        "elo_honest": [ELO * hr[m] for m in models], "elo_liar": [ELO * lr[m] for m in models]})
    adv = (ELO * a, ELO * np.quantile([b[1] for b in boots], 0.025), ELO * np.quantile([b[1] for b in boots], 0.975))
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
        if "verdict" in deb:
            lv = deb[deb.liar == m].drop_duplicates("episode")
            row["liar_committed"] = float((lv.verdict == "committed").mean()) if len(lv) else np.nan
        acc.append(row)
    return out.merge(pd.DataFrame(acc), on="model"), adv


def slope(acc: pd.DataFrame, x: str, y: str) -> float:
    xs, ys = acc[x].to_numpy(dtype=float), acc[y].to_numpy(dtype=float)
    m = np.isfinite(xs) & np.isfinite(ys)
    return float(np.polyfit(xs[m], ys[m], 1)[0] * 100) if m.sum() > 2 else float("nan")


def main(run: str, out: str = "docs/results", n_boot: int = 200) -> None:
    run, out = Path(run), Path(out)
    rows = pd.read_csv(_found(run / "tournament" / "rows.csv"), low_memory=False)
    rows["arm"] = rows.arm.astype(str).str.lower()  # pandas reads a column of "true"/"false" as booleans
    deb = rows[(rows.kind == "debate") & rows.p_true.notna()]
    models = [m for m in TOURNAMENT if m in set(deb.honest) | set(deb.liar)]
    labelled = committed_debates(run, rows)
    if labelled is not None:  # every debate with its liar's verdict (for the commitment rates)
        deb = deb.merge(labelled[["episode", "judge", "verdict"]].drop_duplicates(), on=["episode", "judge"], how="left")
    acc, adv = table_for(deb, rows, models, n_boot)
    tables = {"all": acc}
    summary = [f"honesty advantage (Elo), all debates: {adv[0]:.1f} [{adv[1]:.1f}, {adv[2]:.1f}]"]
    if labelled is not None and "verdict" in deb:
        dc = deb[deb.verdict == "committed"]
        rc = pd.concat([rows[rows.kind != "consultancy"],
                        labelled[(labelled.kind == "consultancy") & (labelled.verdict == "committed")]
                        .drop(columns="verdict")])
        # consultancy: keep the true arm and the committed false-arm episodes
        rc = pd.concat([rc, rows[(rows.kind == "consultancy") & (rows.arm == "true")]])
        accc, advc = table_for(dc, rc, models, n_boot)
        tables["committed"] = accc
        summary.append(f"honesty advantage (Elo), committed liars: {advc[0]:.1f} [{advc[1]:.1f}, {advc[2]:.1f}]")
    out.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(out / ("decisive_tournament.csv" if k == "all" else f"decisive_tournament_{k}.csv"), index=False)
        for x in ("elo", "elo_honest", "elo_liar"):
            for y in ("self_play", "consultancy"):
                summary.append(f"[{k}] slope of {y} accuracy per 100 {x}: {slope(t, x, y):+.4f}")
        if "liar_committed" in t:
            summary.append(f"[{k}] slope of self-play accuracy per +0.10 liar commitment: "
                           f"{slope(t.assign(c=t.liar_committed * 1000), 'c', 'self_play'):+.4f}")
    matrix = deb.assign(win=(deb.p_true > 0.5).astype(float)).pivot_table(index="honest", columns="liar", values="win")
    matrix.reindex(index=models, columns=models).to_csv(out / "decisive_tournament_honest_wins.csv")
    by_judge = deb[deb.honest == deb.liar].groupby(["judge", "honest"]).correct.mean().unstack().reindex(columns=models)
    by_judge.to_csv(out / "decisive_tournament_selfplay_by_judge.csv")
    summary.append(f"debates: {len(deb)} judge verdicts on {deb.item.nunique()} items")
    figure(tables, out)
    for k, t in tables.items():
        print(k)
        print(t.round(3).to_string(index=False))
    print("\n".join(summary))
    (out / "decisive_tournament_summary.txt").write_text("\n".join(summary) + "\n")


def figure(tables: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from decisive_report import INK, INK2, SERIES, _axes

    panels = [("all", "elo_liar", "liar's persuasiveness (Elo as liar)", "all debates"),
              ("committed" if "committed" in tables else "all", "elo", "debater persuasiveness (Elo)",
               "committed liars only" if "committed" in tables else "all debates")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, (key, x, xlabel, title) in zip(axes, panels):
        _axes(ax)
        acc = tables[key]
        # committed-only consultancy pairs a committed false arm with every true arm: not comparable across
        # models (Opus 5.5 keeps 30 false-arm episodes), so that panel shows debate only
        cols = (("self_play", "debate (self-play)"),) if key == "committed" else (
            ("self_play", "debate (self-play)"), ("consultancy", "consultancy"))
        for k, (col, name) in enumerate(cols):
            yerr = np.clip([acc[col] - acc[f"{col}_lo"], acc[f"{col}_hi"] - acc[col]], 0, None)
            ax.errorbar(acc[x], acc[col], yerr=yerr, fmt="o", color=SERIES[k], ms=6, mec="white", lw=1.2, label=name)
            b = np.polyfit(acc[x], acc[col], 1)
            xs = np.linspace(acc[x].min(), acc[x].max(), 10)
            ax.plot(xs, np.polyval(b, xs), color=SERIES[k], lw=1, ls="--")
        for m, xx, yy in zip(acc.model, acc[x], acc.self_play):
            ax.annotate(m, (xx, yy), fontsize=6.5, color=INK2, xytext=(4, -9), textcoords="offset points")
        ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
    axes[0].set_ylabel("judge accuracy (pooled over 3 judges; 95% CI)", color=INK2, fontsize=8.5)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "decisive_tournament.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:])
