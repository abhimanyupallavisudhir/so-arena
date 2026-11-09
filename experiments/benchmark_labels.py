"""Label audit for the benchmark: how often is an experiment-labelled testbed item's label the true effect?

Most testbed families outside decision theory, social choice and number theory are labelled by **one measurement**
of a stochastic experiment (LLM agents at temperature 0.7, Q-learning runs, a few seeds), resolved by the sign of a
difference with a small void gap. Such a label is a correct record of what was measured, but not necessarily of the
effect a perfect forecaster would predict: a rerun can flip it. Two checks:

* ``reliability()``: from each measurement's own spread (session values, binomial counts, per-seed values), the
  posterior probability (flat prior) that the label has the sign of the true expected difference,
  ``P = Phi(|d| / SE_d)``, and that an independent rerun gives the same label, ``Phi(|d| / (sqrt 2 SE_d))``. For
  ``rewardhacking/*`` (outcome questions: does the greedy policy visit the bug in at least 4 of 6 runs, void for 2 to
  4) the same quantities come from a Jeffreys Beta posterior on the per-run probability. The mean of ``P`` over a
  family is the accuracy a forecaster who knew every true effect would expect: the family's label-noise ceiling.
* ``replicate()``: reruns the measurements behind chosen items with fresh samples (a fresh LLM cache; new seeds for
  the Q-learning runs) and reports whether the label replicates, flips or comes out void. Conditions on retired
  models (OpenAI withdrew gpt-4.1-nano) cannot be rerun. Seeded CPU templates (``toyml/*``) are deterministic and not
  rerun.

    python experiments/benchmark_labels.py replicate --items docs/results/benchmark/consensus_misses.csv --budget 25
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAIRED = {"multiagent/auction"}


def _phi(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def labelled_questions() -> dict[str, dict]:
    """Every labelled experiment question: id -> family, style, the question's params and data and the measurement
    record(s) behind its label."""
    import cvtestbed
    from cvtestbed.experiment import cond_id

    out = {}
    for name, t in cvtestbed.templates().items():
        res = t.results()
        for q in t.questions():
            if q.status != "labelled":
                continue
            if t.style == "pairwise":
                recs = [res.get(cond_id(q.params["condition_1"])), res.get(cond_id(q.params["condition_2"]))]
            else:
                recs = [res.get(cond_id({k: v for k, v in q.params.items() if k not in ("version", "yes_option")}))]
            out[q.id] = {"family": name, "style": t.style, "params": q.params, "data": q.data, "answer": q.answer,
                         "recs": recs, "min_gap": t.min_gap}
    return out


def _spread(rec: dict) -> tuple[str, float | None, int | None]:
    """What the record says about its own sampling noise: ('sd', sd, k) for repeated sessions or seeds,
    ('binom', p, n) for a share of n trials, ('balanced', None, None) for balanced accuracy, else ('none', ...)."""
    if rec.get("sessions"):
        v = np.asarray(rec["sessions"], float)
        return "sd", float(v.std(ddof=1)) if len(v) > 1 else None, len(v)
    if rec.get("per_seed"):
        v = [x.get("gain", x.get("metric")) if isinstance(x, dict) else x for x in rec["per_seed"]]
        v = np.asarray(v, float)
        return "sd", float(v.std(ddof=1)) if len(v) > 1 else None, len(v)
    if rec.get("tpr") is not None and rec.get("tnr") is not None:
        return "balanced", None, None
    n = rec.get("n") or rec.get("eligible") or (rec.get("flips") if rec.get("template") == "cot/hint_verbalization"
                                                 else None)
    if n:
        return "binom", rec.get("metric"), int(n)
    return "none", None, None


def _se(rec: dict, pool_sd: float | None) -> float | None:
    kind, a, k = _spread(rec)
    if kind == "sd":
        # shrink the condition's own SD (k - 1 degrees of freedom) toward the family's pooled SD (2 pseudo-dof)
        if pool_sd is None and a is None:
            return None
        s2 = ((k - 1) * (a or 0) ** 2 + 2 * (pool_sd if pool_sd is not None else a) ** 2) / ((k - 1) + 2)
        return math.sqrt(s2 / k)
    if kind == "binom":
        p = (a * k + 1) / (k + 2)  # Agresti-Coull-style: never a zero SE at 0% or 100%
        return math.sqrt(p * (1 - p) / k)
    if kind == "balanced":
        n = 20  # control/backdoor_monitor: 20 backdoored and 20 clean functions
        tp, tn = (rec["tpr"] * n + 1) / (n + 2), (rec["tnr"] * n + 1) / (n + 2)
        return 0.5 * math.sqrt(tp * (1 - tp) / n + tn * (1 - tn) / n)
    return None


def _beta_side(k: int, n: int = 6) -> tuple[float, float]:
    """Outcome questions over n runs (yes iff >= 4 visit; 2-4 void): with a Jeffreys posterior on the per-run
    probability p, P(p on the labelled side of 1/2) and P(a rerun resolves the same way | it is not void)."""
    from scipy.stats import beta, betabinom

    a, b = k + 0.5, n - k + 0.5
    yes = k >= 4
    p_true = float(beta.sf(0.5, a, b)) if yes else float(beta.cdf(0.5, a, b))
    pred = betabinom(n, a, b)
    p_yes, p_no = float(pred.sf(4)), float(pred.cdf(1))  # k' >= 5 resolves yes, k' <= 1 no
    return p_true, (p_yes if yes else p_no) / (p_yes + p_no)


def reliability(qs: dict[str, dict] | None = None) -> pd.DataFrame:
    qs = qs or labelled_questions()
    pools: dict[str, float] = {}
    by_fam: dict[str, list[float]] = {}
    for q in qs.values():
        for r in q["recs"]:
            if r and _spread(r)[0] == "sd" and _spread(r)[1] is not None:
                by_fam.setdefault(q["family"], []).append(_spread(r)[1] ** 2)
    pools = {f: math.sqrt(float(np.mean(v))) for f, v in by_fam.items()}
    # paired designs (multiagent/auction: the bidders' values are the same across conditions within a session): the
    # session-by-session differences carry the noise of the comparison; pool their variance over the family
    paired: dict[str, list[float]] = {}
    for q in qs.values():
        if q["family"] in PAIRED and all(q["recs"]) and all(r.get("sessions") for r in q["recs"]):
            dd = np.subtract(q["recs"][0]["sessions"], q["recs"][1]["sessions"])
            scale = (q["recs"][0]["metric"] + q["recs"][1]["metric"]) / 2
            if len(dd) > 1 and scale > 0:  # relative to the pair's level: revenue ratios run 0.3 to 3
                paired.setdefault(q["family"], []).append(float(dd.var(ddof=1)) / scale ** 2)
    paired_pool = {f: float(np.mean(v)) for f, v in paired.items()}
    rows = []
    for qid, q in qs.items():
        fam = q["family"]
        row = {"item": qid, "family": fam, "d": None, "se_d": None, "p_label_true": None, "p_replicate": None}
        if fam.startswith("rewardhacking/"):
            r = q["recs"][0]
            if r and r.get("exploits") is not None:
                row["p_label_true"], row["p_replicate"] = _beta_side(int(r["exploits"]), int(r.get("runs", 6)))
        elif q["style"] == "pairwise" and all(q["recs"]):
            ses = [_se(r, pools.get(fam)) for r in q["recs"]]
            d = q["recs"][0]["metric"] - q["recs"][1]["metric"]
            row["d"] = d
            se = math.sqrt(ses[0] ** 2 + ses[1] ** 2) if all(s is not None for s in ses) else None
            if fam in paired_pool:
                dd = np.subtract(q["recs"][0]["sessions"], q["recs"][1]["sessions"])
                k = len(dd)
                scale = (q["recs"][0]["metric"] + q["recs"][1]["metric"]) / 2
                v = ((k - 1) * dd.var(ddof=1) + 2 * paired_pool[fam] * scale ** 2) / ((k - 1) + 2)
                se = math.sqrt(v / k)
            if se is not None:
                row["se_d"] = se
                row["p_label_true"] = _phi(abs(d) / se) if se > 0 else 1.0
                row["p_replicate"] = _phi(abs(d) / (math.sqrt(2) * se)) if se > 0 else 1.0
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------- replication


async def _remeasure(q: dict, llm, pool, seed_shift: int) -> dict:
    import cvtestbed

    t = cvtestbed.templates()[q["family"]]
    if t.cpu and not q["family"].startswith("rewardhacking/"):
        return {"status": "deterministic"}
    if q["style"] == "pairwise":
        conds = [q["params"]["condition_1"], q["params"]["condition_2"]]
    else:
        conds = [{k: v for k, v in q["params"].items() if k not in ("version", "yes_option")}]
    if q["family"].startswith("rewardhacking/"):
        conds = [{**c, "seed": int(c["seed"]) + seed_shift} for c in conds]
    try:
        recs = await asyncio.gather(*[t._measure_one(c, llm, pool) for c in conds])
    except Exception as e:  # a retired model, a provider error
        return {"status": "not rerunnable", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    cost = sum(r.get("cost") or 0 for r in recs)
    if q["style"] == "pairwise":
        m1, m2 = recs[0]["metric"], recs[1]["metric"]
        if m1 is None or m2 is None:
            return {"status": "void", "cost": cost, "metric_1": m1, "metric_2": m2}
        if abs(m1 - m2) < t.min_gap:
            status = "void"
        else:
            status = "same" if (0 if m1 > m2 else 1) == q["answer"] else "flipped"
        return {"status": status, "cost": cost, "metric_1": m1, "metric_2": m2}
    verdict = t.resolve(recs[0])
    if verdict is None:
        return {"status": "void", "cost": cost, "metric": recs[0]["metric"]}
    yes_first = q["params"]["yes_option"] == 0
    ans = (0 if yes_first else 1) if verdict else (1 if yes_first else 0)
    return {"status": "same" if ans == q["answer"] else "flipped", "cost": cost, "metric": recs[0]["metric"]}


def replicate(ids: list[str], out: Path, *, budget: float, seed_shift: int = 104729, concurrency: int = 8
              ) -> pd.DataFrame:
    """Rerun the measurements behind ``ids`` once each (fresh LLM cache), appending to ``out`` (jsonl; ids already
    there are skipped). Stops before the reruns' recorded cost passes ``budget``."""
    from concurrent.futures import ProcessPoolExecutor

    from cvtestbed.llm import LLM

    qs = labelled_questions()
    done = {}
    if out.exists():
        done = {json.loads(x)["item"]: json.loads(x) for x in out.read_text().splitlines() if x.strip()}
    todo = [i for i in ids if i in qs and i not in done]
    (out.parent / "replication_cache").mkdir(parents=True, exist_ok=True)
    llm = LLM(cache=out.parent / "replication_cache")
    pool = ProcessPoolExecutor(4)
    spent = sum(r.get("cost") or 0 for r in done.values())

    async def go():
        nonlocal spent
        sem = asyncio.Semaphore(concurrency)

        async def one(i):
            nonlocal spent
            async with sem:
                if spent > budget:
                    return
                r = await _remeasure(qs[i], llm, pool, seed_shift)
                r = {"item": i, "family": qs[i]["family"], **r}
                spent += r.get("cost") or 0
                with open(out, "a") as f:
                    f.write(json.dumps(r, default=str) + "\n")

        await asyncio.gather(*[one(i) for i in todo])

    try:
        asyncio.run(go())
    finally:
        pool.shutdown()
    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()] if out.exists() else []
    return pd.DataFrame(rows)


def main(argv=None):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from decisive_common import map_keys

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reliability")
    r.add_argument("--out", default=str(ROOT / "docs" / "results" / "benchmark" / "label_reliability.csv"))
    p = sub.add_parser("replicate")
    p.add_argument("--items", required=True, help="csv with an 'item' column")
    p.add_argument("--run", default="runs/benchmark")
    p.add_argument("--budget", type=float, default=25.0)
    a = ap.parse_args(argv)
    if a.cmd == "reliability":
        df = reliability()
        df.to_csv(a.out, index=False, float_format="%.4g")
        print(df.groupby("family")[["p_label_true", "p_replicate"]].mean().round(3).to_string())
        return
    map_keys()
    ids = pd.read_csv(a.items).item.tolist()
    out = Path(a.run) / "replications.jsonl"
    df = replicate(ids, out, budget=a.budget)
    spend_path = Path(a.run) / "spend.json"
    spend = json.loads(spend_path.read_text()) if spend_path.exists() else {}
    spend["label replication"] = round(float(df.get("cost", pd.Series(dtype=float)).fillna(0).sum()), 4)
    spend_path.write_text(json.dumps(spend, indent=1, sort_keys=True))
    print(df.groupby(["family", "status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
