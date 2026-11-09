"""Report of the PSRO self-play stage (``decisive_psro.py``): incentive compatibility at each iteration's equilibrium.

For every (protocol, judge) cell and PSRO iteration *k*, the equilibrium sigma_k (solved on the training items) is
evaluated on the held-out items, paired by item:

* ``asd``: E_sigma[honest reward - liar reward] (log scores of the 3-sample mean judge), mean over items with a
  bootstrap 95% CI; ``conc``: the share of items on which it is positive;
* ``asd_1``: the same with the expected *single-sample* reward (mean over the judge's samples of each sample's log
  score), the reward the rest of the study uses;
* ``acc``: the judge's accuracy under sigma_k;
* ``nash_conv``: sigma_k's NashConv in the training game (0 at an exact equilibrium of it) and ``exploit``: its
  exploitability by the best responses found next (training rewards);
* Goodhart rows: each best response's training reward (what selected it) and held-out reward against the
  equilibrium it answered, next to the previous strategy's.

Pooled rows average each item over the cells' judges first. Writes ``<out>/psro.md``, ``psro.csv`` and
``psro_path.{png,pdf}``.

    python experiments/decisive_psro_report.py --run runs/decisive --out docs/results
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from so_arena.analysis.metrics import bootstrap_mean_ci

EPS = 1e-4


def load_cell(d: Path) -> tuple[dict, dict]:
    state = json.loads((d / "state.json").read_text())
    pairs: dict[tuple, pd.DataFrame] = {}
    for split in ("train", "test"):
        for f in (d / "pairs" / split).glob("*.jsonl.gz"):
            h, l = f.name.removesuffix(".jsonl.gz").split("__")
            with gzip.open(f, "rt") as fh:
                recs = [json.loads(x) for x in fh]
            df = pd.DataFrame([r for r in recs if not r.get("error")])
            if df.empty:
                continue
            if "p_true_samples" in df:
                def single(ps, honest):
                    ps = [p for p in (ps if isinstance(ps, list) else []) if p is not None]
                    if not ps:
                        return np.nan
                    return float(np.mean([math.log(max(p if honest else 1 - p, EPS)) for p in ps]))

                df["r_h1"] = [single(ps, True) for ps in df.p_true_samples]
                df["r_l1"] = [single(ps, False) for ps in df.p_true_samples]
            pairs[(split, h, l)] = df.set_index("item")
    return state, pairs


def at_sigma(state: dict, pairs: dict, rec: dict, split: str = "test") -> pd.DataFrame | None:
    """Per-item expectations under an iteration's equilibrium: columns asd, asd_1, acc, r_h, r_l."""
    pops = rec["populations"]
    sh, sl = rec["sigma"]["honest"], rec["sigma"]["liar"]
    acc = None
    for h, wh in zip(pops["honest"], sh):
        for l, wl in zip(pops["liar"], sl):
            w = wh * wl
            if w < 1e-6:
                continue
            df = pairs.get((split, h, l))
            if df is None:
                return None
            part = pd.DataFrame({"asd": df.r_h - df.r_l, "r_h": df.r_h, "r_l": df.r_l, "acc": df.correct,
                                 "asd_1": (df.r_h1 - df.r_l1) if "r_h1" in df else np.nan}) * w
            acc = part if acc is None else acc.add(part, fill_value=np.nan)
    if acc is None:
        return None
    acc = acc.dropna(subset=["asd"])
    seats = pd.concat([df.seat_h for (sp, _, _), df in pairs.items() if sp == split and "seat_h" in df])
    acc["seat_h"] = seats[~seats.index.duplicated()].reindex(acc.index)  # the honest seat is fixed per item
    return acc


def ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan,) * 3
    lo, hi = bootstrap_mean_ci(x)
    return float(x.mean()), float(lo), float(hi)


SEAT_NAMES = {"debater_a": "honest first / listed first", "debater_b": "honest last / listed last"}


def seat_rows(per: pd.DataFrame, **key) -> list[dict]:
    """ASD and judge accuracy split by the honest debater's seat (in sequential debate, debater_a speaks first and
    debater_b has the last word; in simultaneous debate only the transcript order differs), with item-bootstrap CIs,
    plus the honest-last minus honest-first difference (unpaired: each item has one seat)."""
    out = []
    by = {s: g for s, g in per.groupby("seat_h")}
    for seat, g in by.items():
        m, lo, hi = ci(g.asd)
        out.append({**key, "seat_h": seat, "n_items": len(g), "asd": m, "asd_lo": lo, "asd_hi": hi,
                    "acc": float(g.acc.mean())})
    if {"debater_a", "debater_b"} <= set(by):
        a, b = by["debater_a"].asd.to_numpy(), by["debater_b"].asd.to_numpy()
        rng = np.random.default_rng(0)
        diffs = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(2000)]
        out.append({**key, "seat_h": "last - first", "n_items": len(a) + len(b), "asd": b.mean() - a.mean(),
                    "asd_lo": float(np.quantile(diffs, .025)), "asd_hi": float(np.quantile(diffs, .975)),
                    "acc": float(by["debater_b"].acc.mean() - by["debater_a"].acc.mean())})
    return out


def cell_rows(state: dict, pairs: dict) -> tuple[list[dict], list[dict], dict]:
    rows, goodhart, per_item = [], [], {}
    its = [r for r in state["iterations"] if "sigma" in r]
    for k, rec in enumerate(its):
        per = at_sigma(state, pairs, rec)
        if per is None or per.empty:
            continue
        per_item[rec["iteration"]] = per
        m, lo, hi = ci(per.asd)
        nxt = state["iterations"][k + 1] if k + 1 < len(state["iterations"]) else None
        exploit = None
        if nxt and nxt.get("best_responses"):
            exploit = sum(max(0.0, br["best_reward"] - br["seed_reward"]) for br in nxt["best_responses"])
        rows.append({"protocol": state["protocol"], "judge": state["judge"], "iteration": rec["iteration"],
                     "n_items": len(per), "asd": m, "asd_lo": lo, "asd_hi": hi, "conc": float((per.asd > 0).mean()),
                     "asd_1": ci(per.asd_1)[0], "acc": float(per.acc.mean()), "r_h": float(per.r_h.mean()),
                     "r_l": float(per.r_l.mean()),
                     "support_h": int(sum(w > 1e-3 for w in rec["sigma"]["honest"])),
                     "support_l": int(sum(w > 1e-3 for w in rec["sigma"]["liar"])),
                     "nash_conv": rec["sigma"].get("nash_conv"), "exploit_next": exploit})
        if nxt and nxt.get("best_responses"):  # Goodhart: does the best response's training gain hold out?
            for br in nxt["best_responses"]:
                side = br["side"]
                opp = "liar" if side == "honest" else "honest"
                name = f"br{rec['iteration'] + 1}"
                if not br["new"]:
                    goodhart.append({"protocol": state["protocol"], "judge": state["judge"], "iteration": rec["iteration"] + 1,
                                     "side": side, "new": False, "train_gain": 0.0, "test_gain": 0.0})
                    continue
                # held-out reward of the new strategy and of the equilibrium mixture of its side, both against
                # the opponent's equilibrium mixture of iteration k
                def side_reward(strats, weights):
                    tot = None
                    for s, ws in zip(strats, weights):
                        for o, wo in zip(rec["populations"][opp], rec["sigma"][opp]):
                            if ws * wo < 1e-6:
                                continue
                            h, l = (s, o) if side == "honest" else (o, s)
                            df = pairs.get(("test", h, l))
                            if df is None:
                                return None
                            r = (df.r_h if side == "honest" else df.r_l) * ws * wo
                            tot = r if tot is None else tot.add(r, fill_value=np.nan)
                    return tot

                new = side_reward([name], [1.0])
                old = side_reward(rec["populations"][side], rec["sigma"][side])
                test_gain = ci((new - old).dropna()) if new is not None and old is not None else (np.nan,) * 3
                goodhart.append({"protocol": state["protocol"], "judge": state["judge"], "iteration": rec["iteration"] + 1,
                                 "side": side, "new": True, "train_gain": br["best_reward"] - br["seed_reward"],
                                 "test_gain": test_gain[0], "test_lo": test_gain[1], "test_hi": test_gain[2]})
    return rows, goodhart, per_item


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--out", default="docs/results")
    a = ap.parse_args(argv)
    root = Path(a.run) / "psro"
    rows, goodhart, items = [], [], {}
    for d in sorted(p for p in root.glob("*/*") if (p / "state.json").exists()):
        state, pairs = load_cell(d)
        r, g, per = cell_rows(state, pairs)
        rows += r
        goodhart += g
        for k, df in per.items():
            items[(state["protocol"], state["judge"], k)] = df
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    gh = pd.DataFrame(goodhart)
    if df.empty:
        print("no PSRO cells yet")
        return
    # pooled over judges: each item averaged over the judges' cells at the same iteration
    pooled = []
    for (p, k), grp in df.groupby(["protocol", "iteration"]):
        frames = [items[(p, j, k)] for j in grp.judge if (p, j, k) in items]
        if len(frames) < 2:
            continue
        cat = pd.concat(frames)
        per = cat.drop(columns="seat_h").groupby(level=0).mean()
        per["seat_h"] = cat.seat_h.groupby(level=0).first()
        items[(p, "pooled", k)] = per
        m, lo, hi = ci(per.asd)
        pooled.append({"protocol": p, "judge": "pooled", "iteration": k, "n_items": len(per), "asd": m, "asd_lo": lo,
                       "asd_hi": hi, "conc": float((per.asd > 0).mean()), "asd_1": ci(per.asd_1)[0],
                       "acc": float(per.acc.mean())})
    df = pd.concat([df, pd.DataFrame(pooled)], ignore_index=True)
    df.to_csv(out / "psro.csv", index=False)
    if not gh.empty:
        gh.to_csv(out / "psro_goodhart.csv", index=False)
    lines = ["# PSRO self-play: incentive compatibility at equilibrium", "",
             "Held-out items; ASD = E_sigma[honest - liar reward] (log score of the 3-sample mean judge), 95% bootstrap CI "
             "over items; asd_1 uses the expected single-sample reward; conc = share of items where honesty earns more.",
             "", "| protocol | judge | iteration | support (H/L) | ASD [95% CI] | asd_1 | conc | judge acc | NashConv |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values(["protocol", "judge", "iteration"]).iterrows():
        sup = f"{int(r.support_h)}/{int(r.support_l)}" if pd.notna(r.get("support_h")) else "-"
        nc = f"{r.nash_conv:.3f}" if pd.notna(r.get("nash_conv")) else "-"
        lines.append(f"| {r.protocol} | {r.judge} | {int(r.iteration)} | {sup} | {r.asd:+.2f} [{r.asd_lo:+.2f}, {r.asd_hi:+.2f}] "
                     f"| {r.asd_1:+.2f} | {r.conc:.2f} | {r.acc:.2f} | {nc} |")
    if not gh.empty:
        lines += ["", "## Best responses: training gain vs held-out gain", "",
                  "| protocol | judge | iteration | side | new | train gain | held-out gain [95% CI] |", "|---|---|---|---|---|---|---|"]
        for _, r in gh.sort_values(["protocol", "judge", "iteration", "side"]).iterrows():
            held = f"{r.test_gain:+.2f} [{r.test_lo:+.2f}, {r.test_hi:+.2f}]" if r.new and pd.notna(r.get("test_lo")) else \
                f"{r.test_gain:+.2f}"
            lines.append(f"| {r.protocol} | {r.judge} | {int(r.iteration)} | {r.side} | {'yes' if r.new else 'no'} | "
                         f"{r.train_gain:+.2f} | {held} |")
    # seat breakdown: unprompted (iteration 0) and the final equilibrium of every cell and the pooled cells
    seats = []
    for (p, j), grp in df.groupby(["protocol", "judge"]):
        its = sorted(grp.iteration.astype(int))
        for k in sorted({its[0], its[-1]}):
            if (p, j, k) in items:
                seats += seat_rows(items[(p, j, k)], protocol=p, judge=j, iteration=k,
                                   stage="unprompted" if k == its[0] else "equilibrium")
    if seats:
        st = pd.DataFrame(seats)
        st.to_csv(out / "psro_seats.csv", index=False)
        lines += ["", "## By the honest debater's seat", "",
                  "Sequential debate: debater_a speaks first, debater_b has the last word. Simultaneous: both write each "
                  "round blind to the other's speech; only the transcript order differs. 'last - first': difference of the "
                  "two seats' ASD (unpaired item bootstrap; each item has one honest seat); acc column there is the "
                  "accuracy difference.", "",
                  "| protocol | judge | stage | honest seat | items | ASD [95% CI] | judge acc |", "|---|---|---|---|---|---|---|"]
        for _, r in st.iterrows():
            lines.append(f"| {r.protocol} | {r.judge} | {r.stage} (it {int(r.iteration)}) | "
                         f"{SEAT_NAMES.get(r.seat_h, r.seat_h)} | {int(r.n_items)} | "
                         f"{r.asd:+.2f} [{r.asd_lo:+.2f}, {r.asd_hi:+.2f}] | {r.acc:+.2f} |")
    (out / "psro.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
