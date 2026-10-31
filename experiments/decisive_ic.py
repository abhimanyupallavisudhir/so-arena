"""Decisive study, incentive compatibility under optimization (no API calls: reads the best-of-n trees).

``decisive_optimize.py`` measures what the judge concludes as agents are optimized. This asks what the
agents are *paid*, which is what training would follow (``docs/theory.md`` section 2):

* **ASD along the optimization path.** At each optimization level, the expected reward for arguing the true
  answer minus the expected reward for arguing a false one. Prompted ASD (n = 1) is first-order incentive
  compatibility; ASD at matched optimization (``both``) is whether optimizing both sides still pays honesty
  more; ``liar`` puts an optimized liar against a base honest agent.

  - consultancy: the consultant's reward on the true arm minus on the false arm;
  - debate: the mean over the two debater roles of (reward arguing the truth - reward arguing the falsehood),
    i.e. each role compared across the two arms, so the protocol's position effects cancel.

  Concordance under optimization is the share of items on which that difference is positive.
* **Goodhart curves.** The optimized side's expected reward against the judge's accuracy (``docs/theory.md``
  2.2: reward rises along the path; truth may not).
* **Tilting.** The KL-regularized RL optimum ``pi_beta ~ pi_0 exp(beta u)`` computed exactly on the same pools,
  for beta = 0.5 ... 16 (rewards are log scores of the judge's probability).
* **Choosing a side (open consultancy).** A consultant who picks which answer to defend, then a speech: its
  candidates are the union of both arms' pools (16 + 16, the two sides equally likely). Best-of-n over the
  union is the agent the reward would train if it may choose; ``honest`` is the share of its selected
  arguments that defend the true answer - global incentive compatibility, measured directly. Accuracy is the
  judge's accuracy on the selected arguments.

Every number is averaged over items with a bootstrap CI over items; pooled rows average each item over the six
(judge, debater) pairs first. Optional ``--holdout``: payoffs from the trees under ``optimize/`` select, and
values come from the re-judged trees under ``optimize_holdout/`` (``decisive_rejudge.py``), so selection and
evaluation use independent judge samples.

    python experiments/decisive_ic.py --run runs/decisive [--holdout] [--out docs/results]
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from so_arena.analysis.metrics import bootstrap_mean_ci
from so_arena.analysis.optimization import BestOfN, GameTree, Selection, Tilted, evaluate_tree
from so_arena.samplers.pools import load_trees

from decisive_common import DEBATERS, JUDGES
from decisive_optimize import NS, liar_role, mechanisms, tree_path

BETAS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
DEBATE_ROLES = ("debater_a", "debater_b")


def honest_role(mech: str, arm: str) -> str:
    if mech.startswith("consultancy"):
        return "consultant"
    return "debater_a" if arm == "true" else "debater_b"


def selection(kind: str, level: float) -> Selection:
    return BestOfN(int(level)) if kind == "bon" else Tilted(float(level))


def policies(mech: str, arm: str, mode: str, sel: Selection) -> dict[str, Selection]:
    """``liar``: only the false side optimized; ``both``: every agent role optimized."""
    if mode == "both":
        return {r: sel for r in (["consultant"] if mech.startswith("consultancy") else DEBATE_ROLES)}
    if mech.startswith("consultancy") and arm == "true":
        return {}
    return {liar_role(mech, arm): sel}


def holdout_tree(select_tree: GameTree, value_tree: GameTree) -> GameTree:
    """Selection payoffs from one judge sample, ground-truth values from an independent one (same leaves)."""
    leaves = {}
    for lid, leaf in select_tree.leaves.items():
        other = value_tree.leaves.get(lid)
        leaves[lid] = leaf.model_copy(update={"values": other.values if other is not None else {}})
    return select_tree.model_copy(update={"leaves": leaves})


def _load(run: Path, d: str, j: str, mech: str, holdout: bool) -> dict[str, dict[str, GameTree]]:
    out = {}
    for arm in ("true", "false"):
        p = tree_path(run, d, j, mech, arm)
        if not p.exists():
            return {}
        trees = {t.item_id: t for t in load_trees(p)}
        if holdout:
            hp = Path(str(p).replace(f"{os.sep}optimize{os.sep}", f"{os.sep}optimize_holdout{os.sep}"))
            if not hp.exists():
                return {}
            other = {t.item_id: t for t in load_trees(hp)}
            trees = {i: holdout_tree(t, other[i]) for i, t in trees.items() if i in other}
        out[arm] = trees
    return out


def cell_rows(run: Path, d: str, j: str, mech: str, holdout: bool = False) -> list[dict]:
    """Per item, mode and optimization level: rewards of the honest and lying sides on both arms, ASD and
    accuracy."""
    per_arm = _load(run, d, j, mech, holdout)
    if not per_arm:
        return []
    items = sorted(set(per_arm["true"]) & set(per_arm["false"]))
    levels = [("bon", n) for n in NS] + [("tilt", b) for b in BETAS]
    rows = []
    for mode in ("liar", "both"):
        for kind, level in levels:
            sel = selection(kind, level)
            for it in items:
                vals, rew = {}, {}
                for arm in ("true", "false"):
                    v = evaluate_tree(per_arm[arm][it], policies(mech, arm, mode, sel))
                    vals[arm] = v.values.get("judge_correct", np.nan)
                    rew[arm] = v.rewards
                if mech.startswith("consultancy"):
                    r_honest, r_liar = rew["true"].get("consultant"), rew["false"].get("consultant")
                    asd = r_honest - r_liar
                else:
                    # each role: arguing the truth (its honest arm) minus arguing the falsehood
                    diffs = [rew["true"]["debater_a"] - rew["false"]["debater_a"],
                             rew["false"]["debater_b"] - rew["true"]["debater_b"]]
                    asd = float(np.mean(diffs))
                    r_honest = float(np.mean([rew["true"]["debater_a"], rew["false"]["debater_b"]]))
                    r_liar = float(np.mean([rew["false"]["debater_a"], rew["true"]["debater_b"]]))
                rows.append({"debater": d, "judge": j, "mechanism": mech, "mode": mode, "kind": kind,
                             "level": level, "item": it, "accuracy": float(np.nanmean(list(vals.values()))),
                             "reward_honest": r_honest, "reward_liar": r_liar, "asd": asd,
                             "concordant": float(asd > 0) + 0.5 * float(asd == 0)})
    return rows


def open_choice_rows(run: Path, d: str, j: str, holdout: bool = False) -> list[dict]:
    """Single-argument consultancy where the consultant may choose its side: best-of-n over both arms' pools."""
    per_arm = _load(run, d, j, "consultancy_1r", holdout)
    if not per_arm:
        return []
    rows = []
    for it in sorted(set(per_arm["true"]) & set(per_arm["false"])):
        rewards, honest, correct = [], [], []
        for arm in ("true", "false"):
            for leaf in per_arm[arm][it].leaves.values():
                rewards.append(leaf.rewards.get("consultant"))
                honest.append(1.0 if arm == "true" else 0.0)
                correct.append(leaf.values.get("judge_correct", np.nan))
        u = np.array([np.nan if r is None else r for r in rewards], dtype=float)
        h, c = np.array(honest), np.array(correct, dtype=float)
        for kind, levels in (("bon", NS + [32]), ("tilt", BETAS)):
            for level in levels:
                p = selection(kind, level).probs(u)
                rows.append({"debater": d, "judge": j, "mechanism": "open_consultancy_1r", "kind": kind,
                             "level": level, "item": it, "honest": float((p * h).sum()),
                             "accuracy": float(np.nansum(p * c) / p[np.isfinite(c)].sum()),
                             "reward": float(np.nansum(p * u))})
    return rows


def _cell(args):
    run, d, j, mech, holdout = args
    if mech == "open":
        return open_choice_rows(run, d, j, holdout)
    return cell_rows(run, d, j, mech, holdout)


def item_frame(run: Path, holdout: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = [(run, d, j, m.name, holdout) for d in DEBATERS for j in JUDGES for m in mechanisms()]
    cells += [(run, d, j, "open", holdout) for d in DEBATERS for j in JUDGES]
    rows, opens = [], []
    with ProcessPoolExecutor(max_workers=min(len(cells), os.cpu_count() or 1)) as pool:
        for (_, _, _, m, _), got in zip(cells, pool.map(_cell, cells)):
            (opens if m == "open" else rows).extend(got)
    return pd.DataFrame(rows), pd.DataFrame(opens)


def summarize(items: pd.DataFrame, keys: list[str], cols: list[str], n_boot: int = 2000) -> pd.DataFrame:
    """Per-pair and per-debater rows plus pooled rows (each item averaged over pairs first), item bootstrap."""
    if items.empty:
        return pd.DataFrame()
    frames = [items,
              items.groupby(["debater"] + keys + ["item"])[cols].mean().reset_index().assign(judge="all"),
              items.groupby(keys + ["item"])[cols].mean().reset_index().assign(debater="all", judge="all")]
    out = []
    for frame in frames:
        for key, g in frame.groupby(["debater", "judge"] + keys, sort=False):
            row = dict(zip(["debater", "judge"] + keys, key))
            for c in cols:
                x = g[c].to_numpy(dtype=float)
                lo, hi = bootstrap_mean_ci(x, n_boot=n_boot)
                row.update({c: float(np.nanmean(x)), f"{c}_lo": lo, f"{c}_hi": hi})
            row["items"] = int(g.item.nunique())
            out.append(row)
    return pd.DataFrame(out)


def figure(ic: pd.DataFrame, opened: pd.DataFrame, out: Path, suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from decisive_report import INK, INK2, LABEL, SERIES, _axes

    pooled = ic[(ic.debater == "all") & (ic.kind == "bon")]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.7))
    for ax in axes:
        _axes(ax)
    # (a) ASD along best-of-n
    for k, (mech, mode) in enumerate([(m, md) for m in ("consultancy_1r", "debate_1r") for md in ("liar", "both")]):
        g = pooled[(pooled.mechanism == mech) & (pooled["mode"] == mode)].sort_values("level")
        color = SERIES[0] if mech.startswith("debate") else SERIES[1]
        ls = "-" if mode == "both" else "--"
        axes[0].fill_between(g.level, g.asd_lo, g.asd_hi, color=color, alpha=0.12, lw=0)
        axes[0].plot(g.level, g.asd, "o" + ls, color=color, lw=2, ms=5, mec="white",
                     label=f"{LABEL[mech].split(' (')[0]}, {'both optimized' if mode == 'both' else 'liar optimized'}")
    axes[0].axhline(0, color=INK2, lw=1)
    axes[0].set_ylabel("ASD: reward arguing the truth\nminus arguing a falsehood (log score)", color=INK2, fontsize=8.5)
    axes[0].set_title("incentive to be honest", color=INK, fontsize=9, loc="left")
    # (b) Goodhart: optimized liar's reward vs accuracy
    for k, mech in enumerate(("consultancy_1r", "debate_1r")):
        g = pooled[(pooled.mechanism == mech) & (pooled["mode"] == "liar")].sort_values("level")
        color = SERIES[0] if mech.startswith("debate") else SERIES[1]
        axes[1].plot(g.reward_liar, g.accuracy, "o-", color=color, lw=2, ms=5, mec="white",
                     label=LABEL[mech].split(" (")[0])
        for x, y, n in zip(g.reward_liar, g.accuracy, g.level):
            axes[1].annotate(f"n={int(n)}", (x, y), fontsize=6.5, color=INK2, xytext=(3, 3),
                             textcoords="offset points")
    axes[1].set_xlabel("optimized liar's expected reward (log score)", color=INK2, fontsize=8.5)
    axes[1].set_ylabel("judge accuracy", color=INK2, fontsize=8.5)
    axes[1].set_title("Goodhart: liar optimized alone", color=INK, fontsize=9, loc="left")
    # (c) choosing a side
    po = opened[(opened.debater != "all") & (opened.judge == "all") & (opened.kind == "bon")]
    for k, d in enumerate(sorted(po.debater.unique())):
        g = po[po.debater == d].sort_values("level")
        axes[2].fill_between(g.level, g.honest_lo, g.honest_hi, color=SERIES[k], alpha=0.12, lw=0)
        axes[2].plot(g.level, g.honest, "o-", color=SERIES[k], lw=2, ms=5, mec="white", label=d)
    axes[2].axhline(0.5, color=INK2, lw=1, ls=":")
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("share of selected arguments\ndefending the true answer", color=INK2, fontsize=8.5)
    axes[2].set_title("consultant may choose its side", color=INK, fontsize=9, loc="left")
    for ax in (axes[0], axes[2]):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("best-of-n against the judge's reward", color=INK2, fontsize=8.5)
        ticks = sorted({int(v) for v in (pooled.level if ax is axes[0] else po.level)})
        ax.set_xticks(ticks, [str(t) for t in ticks])
    for ax in axes:
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"decisive_ic{suffix}.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results")
    ap.add_argument("--holdout", action="store_true")
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "_holdout" if a.holdout else ""
    items, opens = item_frame(run, a.holdout)
    items.to_csv(run / f"ic_items{suffix}.csv.gz", index=False)
    opens.to_csv(run / f"ic_open_items{suffix}.csv.gz", index=False)
    ic = summarize(items, ["mechanism", "mode", "kind", "level"],
                   ["accuracy", "asd", "concordant", "reward_honest", "reward_liar"])
    opened = summarize(opens, ["mechanism", "kind", "level"], ["honest", "accuracy", "reward"])
    ic.to_csv(out / f"decisive_ic{suffix}.csv", index=False)
    opened.to_csv(out / f"decisive_open_choice{suffix}.csv", index=False)
    figure(ic, opened, out, suffix)
    show = ic[(ic.judge == "all") & (ic.kind == "bon")][["debater", "mechanism", "mode", "level", "accuracy", "asd",
                                                         "asd_lo", "asd_hi", "concordant", "reward_liar"]]
    print(show.round(3).to_string(index=False))
    print(opened[(opened.judge == "all")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
