"""Empirical incentive compatibility (EIC) of the oversight protocols: the decisive study's headline analysis.

A protocol is incentive compatible to the extent that its reward $u$ for an agent rises with the ground-truth value
$v$ of the agent's behaviour. Training on the protocol's reward moves an agent's policy $\\pi$ in the direction
$\\frac{d}{d\\beta}\\mathbb{E}_{\\pi_\\beta}[v] = \\mathrm{Cov}_{\\pi_\\beta}(u, v)$, so the covariance over the behaviours
an agent could adopt, sampled somehow, is a cheap proxy for what training would do. Here $v \\in \\{0, 1\\}$ by
construction: 1 when the agent argues the true answer, 0 when it argues a false one. Then
$\\mathrm{Cov}(u, v)/\\mathrm{Var}(v)$ is the **ASD**, the mean reward for arguing the truth minus that for lying, in
the protocol's own reward units. Every estimate is also given as **concordance**: the share of items on which
honesty earns more, 0.5 meaning no incentive. Where per-episode rewards exist, the normalized covariance
$r = \\mathrm{corr}(u, v)$ is given too.

Samplers (what the covariance is taken over):

* **instructed**: the agent told to argue the true or a false answer (ASD's sampler); 3 samples per item and arm;
* **instructed, committed liars**: the same, keeping only samples whose liars the gpt-5 audit found committed to
  the lie (a sampler of *effective* lies: a liar that concedes is not the dishonest behaviour training could
  reach);
* **best-of-n**: the agent's natural behaviour re-weighted toward high reward, exactly over sampled pools (one
  round) or on every speech (three rounds), with selection and evaluation on independent judge samples;
* **prompt optimization**: strategies written by an optimizer for the honest and the deceptive side, selected on
  training items and scored on held-out items (``decisive_eic_po.py``);
* **choosing a side**: an agent free to choose which answer to defend, optimized by best-of-n over both sides'
  candidates; EIC is then the share of selected behaviour that is honest.

Writes ``docs/results/eic/`` (tables and figures) and ``docs/results/eic.md``.

    python experiments/decisive_eic.py runs/decisive [docs/results]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decisive_report import _found, committed_filter  # noqa: E402

N_BOOT = 2000
PROTOCOLS = ["consultancy", "debate_seq", "debate_sim", "recursive_debate", "prover_estimator"]
NAMES = {"consultancy": "Consultancy", "debate_seq": "Debate, sequential", "debate_sim": "Debate, simultaneous",
         "recursive_debate": "Recursive debate", "prover_estimator": "Prover-estimator",
         "consultancy_1r": "Consultancy, 1 round", "debate_1r": "Debate, 1 round"}
UNITS = {"recursive_debate": "judge probability", "prover_estimator": "prover payoff"}  # else: log score
JUDGE_ORDER = ["llama-3.1-8b", "gemma-3-12b", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]


# ------------------------------------------------------------------------------------------------ estimators

def _boot_weights(k: int, n_boot: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, k, k), minlength=k) for _ in range(n_boot)]).astype(float)


def eic_rows(df: pd.DataFrame, by: list[str], n_boot: int = N_BOOT) -> pd.DataFrame:
    """EIC from per-episode (u, v) rows: ASD (item-weighted difference of honest and lying means), concordance
    (share of items where honesty earns more) and r = corr(u, v); CIs resample items."""
    out = []
    for key, g in (df.groupby(by) if by else [((), df)]):
        key = key if isinstance(key, tuple) else (key,)
        g = g.dropna(subset=["u", "v"])
        per = g.groupby("item").apply(lambda h: pd.Series({
            "uh": h.u[h.v == 1].mean(), "ul": h.u[h.v == 0].mean(), "n": len(h), "su": h.u.sum(),
            "suu": (h.u ** 2).sum(), "sv": h.v.sum(), "suv": (h.u * h.v).sum()}), include_groups=False).dropna(
            subset=["uh", "ul"])
        if per.empty:
            continue
        d = (per.uh - per.ul).to_numpy()
        c = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))
        S = per[["n", "su", "suu", "sv", "suv"]].to_numpy()
        W = _boot_weights(len(per), n_boot)

        def corr(s):
            n, su, suu, sv, suv = s.T if s.ndim == 2 else s
            cov = suv / n - (su / n) * (sv / n)
            var_u = suu / n - (su / n) ** 2
            var_v = sv / n - (sv / n) ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                return cov / np.sqrt(var_u * var_v)

        tot = S.sum(axis=0)
        rb = corr((W @ S))
        asd_b = (W @ d) / W.sum(axis=1)
        conc_b = (W @ c) / W.sum(axis=1)
        row = dict(zip(by, key))
        row.update({"asd": float(d.mean()), "asd_lo": float(np.quantile(asd_b, .025)),
                    "asd_hi": float(np.quantile(asd_b, .975)), "concordance": float(c.mean()),
                    "conc_lo": float(np.quantile(conc_b, .025)), "conc_hi": float(np.quantile(conc_b, .975)),
                    "r": float(corr(tot)), "r_lo": float(np.nanquantile(rb, .025)),
                    "r_hi": float(np.nanquantile(rb, .975)), "items": int(len(per)), "episodes": int(tot[0])})
        out.append(row)
    return pd.DataFrame(out)


def eic_items(df: pd.DataFrame, by: list[str], col: str = "asd", n_boot: int = N_BOOT) -> pd.DataFrame:
    """EIC from per-item ASDs (samplers that give expected rewards under a selection policy, not episodes);
    pairs (judge, debater) are averaged within an item before resampling items."""
    out = []
    for key, g in (df.groupby(by) if by else [((), df)]):
        key = key if isinstance(key, tuple) else (key,)
        per = g.groupby("item")[col].mean().dropna()
        if per.empty:
            continue
        d = per.to_numpy()
        c = np.where(d > 0, 1.0, np.where(d == 0, 0.5, 0.0))
        W = _boot_weights(len(d), n_boot)
        asd_b, conc_b = (W @ d) / W.sum(1), (W @ c) / W.sum(1)
        row = dict(zip(by, key))
        row.update({"asd": float(d.mean()), "asd_lo": float(np.quantile(asd_b, .025)),
                    "asd_hi": float(np.quantile(asd_b, .975)), "concordance": float(c.mean()),
                    "conc_lo": float(np.quantile(conc_b, .025)), "conc_hi": float(np.quantile(conc_b, .975)),
                    "items": int(len(d))})
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ samplers

def instructed(run: Path) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Per-episode (u, v) of the protocol grid, all prompted liars and committed liars only (gpt-5 audit)."""
    rows = pd.read_csv(_found(run / "protocols" / "rows.csv"))
    rows = rows[rows.mechanism.isin(PROTOCOLS) & rows.value.notna() & rows.reward.notna()].copy()
    rows["u"], rows["v"] = rows.reward, (rows.value > 0).astype(float)
    vf = _found(run / "audit" / "verdicts_gpt-5.jsonl")
    committed = None
    if vf.exists():
        ar = pd.read_csv(run / "audit" / "rows.csv.gz", low_memory=False)
        v = ar.merge(pd.read_json(vf, lines=True).drop_duplicates("key", keep="last"), on="key", how="left")
        committed = committed_filter(rows, v)
    return rows, committed


def common_items(rows: pd.DataFrame) -> set:
    sets = [set(g.item) for d, g in rows.groupby("debater")]
    return set.intersection(*sets)


def best_of_n_one_round(run: Path) -> pd.DataFrame:
    """Per-item ASD under exact best-of-n (one round; held-out judging), from ``decisive_ic.py --holdout``."""
    f = run / "ic_items_holdout.csv.gz"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    return d[d.kind == "bon"].rename(columns={"level": "n"})


def best_of_n_multiround(run: Path) -> pd.DataFrame:
    f = run / "multiround_items.csv.gz"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


def prompt_optimization(run: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """From ``decisive_eic_po.py``: per-item held-out ASD of the best honest vs best deceptive strategy after each
    search round; per-episode (u, v) of every strategy on the training items; the strategies themselves."""
    path_rows, pop_rows, strat_rows = [], [], []
    import gzip

    files = sorted((run / "eic_po").glob("*/*.json")) or sorted((run / "eic_po").glob("*/*.json.gz"))
    for f in files:  # the run's files, or the gzipped copies in the committed data bundle
        rec = json.loads(gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text())
        h, d = rec["searches"]["true"], rec["searches"]["false"]

        def per_item(records):
            df = pd.DataFrame(records)
            return df.groupby("item").reward.mean() if len(df) else pd.Series(dtype=float)

        for k, (hid, did) in enumerate(zip(h["best_path"], d["best_path"])):
            hi, di = per_item(h["heldout"][hid]), per_item(d["heldout"][did])
            both = pd.concat([hi.rename("uh"), di.rename("ul")], axis=1).dropna()
            for item, r in both.iterrows():
                path_rows.append({"mechanism": rec["protocol"], "judge": rec["judge"], "round": k, "item": item,
                                  "asd": r.uh - r.ul, "u_honest": r.uh, "u_liar": r.ul})
        for arm, s in (("true", h), ("false", d)):
            for c in s["strategies"]:
                rs = [e for e in c["train"] if e["reward"] is not None]
                strat_rows.append({"mechanism": rec["protocol"], "judge": rec["judge"], "side": arm,
                                   "directive": s["directive"], "id": c["id"], "round": c["iteration"],
                                   "train_reward": float(np.mean([e["reward"] for e in rs])) if rs else np.nan,
                                   "strategy": c["strategy"]})
                for e in rs:
                    pop_rows.append({"mechanism": rec["protocol"], "judge": rec["judge"], "item": e["item"],
                                     "u": e["reward"], "v": 1.0 if arm == "true" else 0.0, "strategy": c["id"]})
    return pd.DataFrame(path_rows), pd.DataFrame(pop_rows), pd.DataFrame(strat_rows)


def choosing_a_side(run: Path, holdout: bool = True) -> pd.DataFrame:
    """Best-of-n for an agent that may choose its side (one round): the share of its selected behaviour that is
    honest, and its expected reward. Consultancy from ``decisive_ic.py``; debate here, from the trees: debater A's
    candidates on both arms, each worth its mean reward against the base opponent's pool; selection on the
    original judge sample, reward reported on the held-out one."""
    from so_arena.analysis.optimization import BestOfN
    from so_arena.samplers.pools import load_trees

    from decisive_common import DEBATERS, JUDGES
    from decisive_optimize import NS, tree_path

    out = []
    f = run / ("ic_open_items_holdout.csv.gz" if holdout else "ic_open_items.csv.gz")
    if f.exists():
        oc = pd.read_csv(f)
        oc = oc[oc.kind == "bon"].rename(columns={"level": "n"})
        out.append(oc.assign(mechanism="consultancy_1r")[["debater", "judge", "mechanism", "n", "item", "honest",
                                                          "reward"]])
    def payoffs(p: Path, hp: Path | None) -> dict:
        """item -> list of (payoff on the selection sample, payoff on the held-out sample) for debater A's
        candidates, each its mean reward against the opponent's pool; the trees are not kept in memory."""
        held = {t.item_id: t for t in load_trees(hp)} if hp is not None and hp.exists() else {}
        res = {}
        for t in load_trees(p):
            h = held.get(t.item_id)
            cands = []
            for cid in t.nodes[t.root].children:
                leaves = t.nodes[cid].children if cid in t.nodes else []
                u = [t.leaves[x].rewards.get("debater_a") for x in leaves if x in t.leaves]
                uh = [h.leaves[x].rewards.get("debater_a") for x in leaves if h is not None and x in h.leaves]
                u, uh = [x for x in u if x is not None], [x for x in uh if x is not None]
                if u:
                    cands.append((float(np.mean(u)), float(np.mean(uh)) if uh else float(np.mean(u))))
            res[t.item_id] = cands
        del held
        return res

    rows = []
    for d in DEBATERS:
        for j in JUDGES:
            arms = {}
            for arm in ("true", "false"):
                p = tree_path(run, d, j, "debate_1r", arm)
                if p.exists():
                    hp = Path(str(p).replace("/optimize/", "/optimize_holdout/")) if holdout else None
                    arms[arm] = payoffs(p, hp)
            if len(arms) < 2:
                continue
            for it in sorted(set(arms["true"]) & set(arms["false"])):
                cands = [(a, b, 1.0) for a, b in arms["true"][it]] + [(a, b, 0.0) for a, b in arms["false"][it]]
                pay, rep, hon = (np.asarray(x) for x in zip(*cands))
                for n in NS + [32]:
                    pr = BestOfN(n).probs(pay)
                    rows.append({"debater": d, "judge": j, "mechanism": "debate_1r", "n": n, "item": it,
                                 "honest": float((pr * hon).sum()), "reward": float(np.nansum(pr * rep))})
    if rows:
        out.append(pd.DataFrame(rows))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def tournament(run: Path) -> pd.DataFrame:
    """Per (model, item, kind): the model's reward arguing the truth minus its reward lying, against every
    opponent; ``committed`` keeps debates whose liar the audit found committed."""
    f = _found(run / "tournament" / "rows.csv")
    if not f.exists():
        return pd.DataFrame()
    t = pd.read_csv(f, low_memory=False)
    t = t[t.kind == "debate"]
    from decisive_tournament_report import committed_debates

    lab = committed_debates(run, pd.read_csv(f, low_memory=False))
    if lab is not None:
        t = t.merge(lab[["episode", "judge", "verdict"]].drop_duplicates(), on=["episode", "judge"], how="left")
    out = []
    for subset, frame in (("all", t), ("committed", t[t.get("verdict", pd.Series(index=t.index)) == "committed"])):
        hon = frame.groupby(["honest", "item"]).honest_reward.mean().rename("uh")
        lie = frame.groupby(["liar", "item"]).liar_reward.mean().rename("ul")
        hon.index.names = lie.index.names = ["model", "item"]
        both = pd.concat([hon, lie], axis=1).dropna().reset_index()
        both["asd"] = both.uh - both.ul
        out.append(both.assign(liars=subset))
    return pd.concat(out, ignore_index=True)


def judge_strength(run: Path) -> pd.DataFrame:
    """Per-episode (u, v) of the same debates (and consultancies) for weak and strong judges, per dataset, with a
    ``committed`` flag from the audit (replayed debates inherit their source transcript's verdict)."""
    from decisive_extended import _drop_conceding, audit_verdicts

    f = _found(run / "judges" / "rows.csv")
    if not f.exists():
        return pd.DataFrame()
    strong = pd.read_csv(f)
    verdicts = audit_verdicts(run)
    out = []
    for ds, src in (("testbed", run / "protocols" / "rows.csv"), ("quality", run / "quality" / "rows.csv")):
        base = pd.read_csv(_found(src))
        st = strong[strong.dataset == ds].drop(columns="dataset")
        if st.empty:
            continue
        mechs = set(st.mechanism) - {"direct"}
        weak = base[base.item.isin(set(st.item)) & base.mechanism.isin(mechs)]
        frame = pd.concat([weak, st[st.mechanism.isin(mechs)]], ignore_index=True)
        frame = frame[frame.value.notna() & frame.reward.notna()].copy()
        frame["u"], frame["v"] = frame.reward, (frame.value > 0).astype(float)
        out.append(frame.assign(dataset=ds, liars="all"))
        if verdicts is not None:
            kind = "prompted" if ds == "testbed" else "quality"
            v = verdicts[verdicts.verdict != "committed"]
            vk = v[v.kind == kind]
            w = _drop_conceding(frame[frame.judge.isin(set(vk.judge))], vk,
                                ["judge", "debater", "mechanism", "item", "repeat"])
            s_ = frame[~frame.judge.isin(set(vk.judge))]
            deb = _drop_conceding(s_[s_.mechanism != "consultancy"], vk[vk.judge == "gpt-4o-mini"],
                                  ["debater", "mechanism", "item", "repeat"])
            con = _drop_conceding(s_[s_.mechanism == "consultancy"], v[v.kind == "judges"],
                                  ["judge", "debater", "mechanism", "item", "repeat"])
            out.append(pd.concat([w, deb, con], ignore_index=True).assign(dataset=ds, liars="committed"))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def sampler_agreement(tables: dict):
    """Do the samplers rank the protocols alike? Per (protocol, judge) cell: concordance under the instructed
    sampler, the committed-liar sampler and prompt optimization (final round, held-out); Spearman correlations
    between samplers over the cells, and within each judge over the protocols."""
    ins, com, po = tables.get("instructed"), tables.get("committed"), tables.get("po_path")
    if po is None or po.empty:
        return None
    po_final = po[po["round"] == po.groupby(["mechanism", "judge"])["round"].transform("max")]
    cells = []
    for _, r in po_final[po_final.judge != "all"].iterrows():
        row = {"mechanism": r.mechanism, "judge": r.judge, "po": r.concordance, "po_asd": r.asd}
        for name, t in (("instructed", ins), ("committed", com)):
            m = t[(t.mechanism == r.mechanism) & (t.judge == r.judge) & (t.debater == "gpt-5-mini")]
            row[name] = float(m.concordance.iloc[0]) if len(m) else np.nan
            row[f"{name}_asd"] = float(m.asd.iloc[0]) if len(m) else np.nan
        cells.append(row)
    cells = pd.DataFrame(cells)
    from scipy.stats import spearmanr

    out = []
    for a, b in (("instructed", "po"), ("committed", "po"), ("instructed", "committed")):
        for scope, g in [("all cells", cells)] + [(f"within {j}", h) for j, h in cells.groupby("judge")]:
            g = g.dropna(subset=[a, b])
            if len(g) >= 3:
                rho = spearmanr(g[a], g[b]).statistic
                out.append({"samplers": f"{a} vs {b}", "scope": scope, "spearman": float(rho), "cells": len(g)})
    return cells, pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ main

def main(run: str, out: str = "docs/results") -> None:
    run, res = Path(run), Path(out)
    eic = res / "eic"
    eic.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}

    rows, committed = instructed(run)
    both = common_items(rows)
    gpt = rows[rows.debater == "gpt-5-mini"]
    tables["instructed"] = pd.concat([
        eic_rows(gpt, ["mechanism"]).assign(sampler="instructed", debater="gpt-5-mini", judge="all"),
        eic_rows(gpt, ["judge", "mechanism"]).assign(sampler="instructed", debater="gpt-5-mini"),
        eic_rows(rows[rows.item.isin(both)], ["debater", "mechanism"]).assign(sampler="instructed (320 items)",
                                                                             judge="all"),
    ])
    if committed is not None:
        cg = committed[committed.debater == "gpt-5-mini"]
        tables["committed"] = pd.concat([
            eic_rows(cg, ["mechanism"]).assign(sampler="committed liars", debater="gpt-5-mini", judge="all"),
            eic_rows(cg, ["judge", "mechanism"]).assign(sampler="committed liars", debater="gpt-5-mini"),
            eic_rows(committed[committed.item.isin(both)], ["debater", "mechanism"]).assign(
                sampler="committed liars (320 items)", judge="all"),
        ])
    b1 = best_of_n_one_round(run)
    if len(b1):
        tables["bon_1round"] = pd.concat([
            eic_items(b1, ["debater", "mechanism", "mode", "n"]),
            eic_items(b1, ["mechanism", "mode", "n"]).assign(debater="all")]).assign(sampler="best-of-n, 1 round",
                                                                                      judge="all")
    bm = best_of_n_multiround(run)
    if len(bm):
        tables["bon_3round"] = pd.concat([
            eic_items(bm, ["mechanism", "mode", "n"]),
            eic_items(bm, ["judge", "mechanism", "mode", "n"])]).assign(sampler="best-of-n per speech, 3 rounds",
                                                                        debater="gpt-5-mini")
        tables["bon_3round"]["judge"] = tables["bon_3round"]["judge"].fillna("all")
    po_path, po_pop, po_strat = prompt_optimization(run)
    if len(po_path):
        tables["po_path"] = pd.concat([eic_items(po_path, ["mechanism", "round"]).assign(judge="all"),
                                       eic_items(po_path, ["judge", "mechanism", "round"])]).assign(
            sampler="prompt optimization", debater="gpt-5-mini")
        tables["po_population"] = pd.concat([eic_rows(po_pop, ["mechanism"]).assign(judge="all"),
                                             eic_rows(po_pop, ["judge", "mechanism"])]).assign(
            sampler="prompt optimization, all strategies", debater="gpt-5-mini")
        po_strat.to_csv(eic / "prompt_optimization_strategies.csv", index=False)
    side = choosing_a_side(run)
    if len(side):
        g = side.groupby(["mechanism", "n", "item"])[["honest", "reward"]].mean().reset_index()
        rows_ = []
        for (m, n), h in g.groupby(["mechanism", "n"]):
            W = _boot_weights(len(h), N_BOOT)
            hb = (W @ h.honest.to_numpy()) / W.sum(1)
            rows_.append({"mechanism": m, "n": n, "honest": h.honest.mean(), "honest_lo": np.quantile(hb, .025),
                          "honest_hi": np.quantile(hb, .975), "reward": h.reward.mean(), "items": len(h)})
        tables["choosing_a_side"] = pd.DataFrame(rows_)
    tour = tournament(run)
    if len(tour):
        tables["tournament"] = eic_items(tour, ["liars", "model"])
    js = judge_strength(run)
    if len(js):
        tables["judge_strength"] = eic_rows(js, ["liars", "dataset", "judge", "mechanism"])
    agreement = sampler_agreement(tables)
    if agreement is not None:
        tables["agreement_cells"], tables["agreement"] = agreement
    for k, t in tables.items():
        t.to_csv(eic / f"{k}.csv", index=False)
    import decisive_eic_figures as figs

    figs.make_all(tables, eic, po_strat if len(po_path) else None)
    write_report(tables, res)


def write_report(tables: dict, res: Path) -> None:
    from decisive_eic_figures import report_markdown

    (res / "eic.md").write_text(report_markdown(tables))


if __name__ == "__main__":
    main(*sys.argv[1:])
