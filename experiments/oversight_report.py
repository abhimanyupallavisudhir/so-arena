"""Report for experiments/oversight_testbed.py: accuracy, ASD and concordance per protocol and debater model.

    python experiments/oversight_report.py runs/oversight-testbed-v4

Reads every <agent>/protocols and <agent>/strong store under the run directory (partial runs included) and
writes summary.csv, summary_by_family.csv, summary.md and accuracy.png there. Accuracy CIs are a bootstrap over
items (both arms of an item resampled together).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from so_arena.core.store import RunStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oversight_testbed import summarize  # noqa: E402

ORDER = ["direct", "consultancy", "debate", "recursive_debate", "prover_estimator", "direct_strong"]
LABEL = {"direct": "weak judge alone", "consultancy": "consultancy", "debate": "debate",
         "recursive_debate": "recursive debate", "prover_estimator": "prover-estimator", "direct_strong":
         "strong model alone"}
COLOR = {"direct": "#52514e", "consultancy": "#2a78d6", "debate": "#eb6834", "recursive_debate": "#1baf7a",
         "prover_estimator": "#eda100", "direct_strong": "#0b0b0b"}


def load(run: Path):
    eps = []
    for d in sorted(p for p in run.iterdir() if p.is_dir()):
        for sub in ("protocols", "strong"):
            f = d / sub / "episodes.jsonl"
            if f.exists():
                for e in RunStore(str(d / sub)).episodes():
                    if e.error:
                        continue
                    e.tags["agent_model"] = d.name
                    eps.append(e)
    return eps


def accuracy_ci(eps, n_boot=2000, seed=0):
    rows = []
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([{"agent": e.tags["agent_model"], "mechanism": e.mechanism, "item": e.item_id,
                        "correct": e.ground_truth.get("judge_correct")} for e in eps]).dropna()
    for (a, m), g in df.groupby(["agent", "mechanism"]):
        per_item = g.groupby("item").correct.mean().to_numpy()
        boots = rng.choice(per_item, (n_boot, len(per_item))).mean(1)
        rows.append({"agent": a, "mechanism": m, "accuracy": per_item.mean(), "lo": np.quantile(boots, .025),
                     "hi": np.quantile(boots, .975), "items": len(per_item)})
    return pd.DataFrame(rows)


def figure(acc: pd.DataFrame, out: Path, agents: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    mechs = [m for m in ORDER if m in set(acc.mechanism)]
    off = np.linspace(-0.3, 0.3, len(mechs))
    for j, m in enumerate(mechs):
        g = acc[acc.mechanism == m].set_index("agent").reindex(agents)
        x = np.arange(len(agents)) + off[j]
        ax.vlines(x, g.lo, g.hi, color=COLOR[m], lw=2)
        ax.plot(x, g.accuracy, "o", color=COLOR[m], ms=6, mec="white", mew=1.5, label=LABEL[m])
    ax.axhline(0.5, color="#52514e", lw=0.8, ls=":")
    ax.set_xticks(range(len(agents)), agents, fontsize=9)
    ax.set_xlabel("debater / agent model (judge: gpt-4.1-nano with reasoning)", color="#52514e", fontsize=9)
    ax.set_ylabel("verdict accuracy (95% CI)", color="#52514e", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(out / "accuracy.png", dpi=160)
    plt.close(fig)


def main(run: str):
    run = Path(run)
    eps = load(run)
    agents = [a for a in ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-5-mini", "claude-sonnet-4-5"]
              if any(e.tags["agent_model"] == a for e in eps)]
    overall = pd.concat([summarize([e for e in eps if e.tags["agent_model"] == a], by_family=False).assign(agent=a)
                         for a in agents])
    fam = pd.concat([summarize([e for e in eps if e.tags["agent_model"] == a], by_family=True).assign(agent=a)
                     for a in agents])
    acc = accuracy_ci(eps)
    overall = overall.merge(acc[["agent", "mechanism", "lo", "hi"]], on=["agent", "mechanism"], how="left")
    overall.to_csv(run / "summary.csv", index=False)
    fam.to_csv(run / "summary_by_family.csv", index=False)
    figure(acc, run, agents)
    cost = sum(sum(u.cost_usd for u in e.usage.values()) for e in eps)
    piv = fam.pivot_table(index="family", columns=["agent", "mechanism"], values="accuracy")
    lines = ["# Oversight protocols on the testbed", "", f"{len(eps)} episodes, ${cost:.2f}.", "",
             overall[["agent", "mechanism", "accuracy", "lo", "hi", "asd", "concordance", "items", "parse_fail"]]
             .round(3).to_markdown(index=False), "",
             "## Accuracy by family", "", piv.round(2).to_markdown(), ""]
    (run / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines[:6]))


if __name__ == "__main__":
    main(sys.argv[1])
