"""Decisive study, stage 4: audit the liars - did the agents told to argue the wrong answer actually argue it?

Two kinds of liar are audited with ``audit_liars.py``'s auditor (gpt-4.1-mini, told the assignment and the true
answer; verdicts committed / conceded / refused / broken) and its regex for explicit tells:

* prompted liars: every liar transcript of ``decisive_protocols.py`` (the consultant, either debater, the
  recursive and prover-estimator provers on the false arm). Transcripts of judge-independent protocols are the
  same for every judge (cache replays) and are audited once;
* optimized liars: every candidate speech in the liar's pools of ``decisive_optimize.py`` (shared by the
  judges); the report weights the verdicts by best-of-n's selection probabilities.

Verdicts are appended to ``<run>/audit/verdicts.jsonl`` keyed by a hash of (protocol, item, role, text), so the
stage resumes and its cost is the sum over that file.

    python experiments/decisive_audit.py --run runs/decisive [--estimate]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from so_arena.core.store import RunStore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_liars import AUDITOR, PROMPT, TELLS  # noqa: E402
from decisive_common import DEBATERS, JUDGES, Ledger, map_keys, selected_items  # noqa: E402
from decisive_optimize import liar_role, mechanisms as opt_mechanisms  # noqa: E402

ROLES = {"consultancy": ["consultant"], "debate_seq": ["debater_a", "debater_b"],
         "debate_sim": ["debater_a", "debater_b"], "recursive_debate": ["prover"], "prover_estimator": ["prover"],
         "prover_estimator_v2": ["prover"]}


def key_of(mechanism: str, item: str, role: str, text: str) -> str:
    return hashlib.sha256(json.dumps([mechanism, item, role, text]).encode()).hexdigest()[:24]


def prompted_rows(run: Path) -> list[dict]:
    """One row per (judge, debater, protocol, episode, liar role), with the transcript's audit key."""
    rows = []
    for root, d, j in ((r, d, j) for r in ("protocols", "protocols_ext", "pe_v2") for d in DEBATERS for j in JUDGES):
        for mech, roles in ROLES.items():
            store = run / root / d / j / mech
            if not RunStore(str(store)).has_episodes():
                continue
            for ep in RunStore(str(store)).episodes():
                if ep.error:
                    continue
                values = (ep.ground_truth or {}).get("role_values") or {}
                for role in roles:
                    if values.get(role) is None or values[role] > 0:
                        continue  # not a liar
                    text = "\n\n".join(f"[{t.phase}] {t.text}" for t in ep.turns if t.role == role and t.text)
                    rows.append({"kind": "prompted", "judge": j, "debater": d, "mechanism": mech,
                                 "item": ep.item_id, "arm": ep.tags.get("arm"), "repeat": ep.repeat,
                                 "episode": ep.id, "role": role, "assigned": ep.positions.get(role),
                                 "text": text, "key": key_of(mech, ep.item_id, role, text)})
    return rows


def optimized_rows(run: Path) -> list[dict]:
    """One row per candidate in a liar's pool (pools are shared across judges: one row per distinct text)."""
    rows, seen = [], set()
    for d in DEBATERS:
        for j in JUDGES:
            for mech in [m.name for m in opt_mechanisms()]:
                for arm in ("true", "false"):
                    f = run / "optimize" / d / j / f"{mech}-{arm}.candidates.jsonl"
                    if not f.exists() or (mech.startswith("consultancy") and arm == "true"):
                        continue
                    liar = liar_role(mech, arm)
                    for line in f.read_text().splitlines():
                        c = json.loads(line)
                        if c["role"] != liar:
                            continue
                        k = key_of(mech, c["item"], liar, c["text"])
                        rows.append({"kind": "optimized", "judge": j, "debater": d, "mechanism": mech,
                                     "item": c["item"], "arm": arm, "role": liar, "pool_key": c["key"],
                                     "index": c["index"], "text": c["text"], "key": k})
                        seen.add(k)
    return rows


def _liar_turns(ep):
    values = (ep.ground_truth or {}).get("role_values") or {}
    for role, v in values.items():
        if v is not None and v <= 0 and role in ("consultant", "debater_a", "debater_b", "prover"):
            text = "\n\n".join(f"[{t.phase}] {t.text}" for t in ep.turns if t.role == role and t.text)
            yield role, text


def stage_rows(run: Path) -> list[dict]:
    """Liars of the later stages: the tournament's debates and consultancies, the multi-round best-of-n runs and
    the QuALITY replication (one row per episode and liar role; debate transcripts shared by judges once)."""
    rows = []

    def add(kind, store: Path, **meta):
        if not RunStore(str(store)).has_episodes():
            return
        for ep in RunStore(str(store)).episodes():
            if ep.error:
                continue
            for role, text in _liar_turns(ep):
                m = dict(meta)
                if kind == "tournament" and m.get("debater") is None:
                    m["debater"] = ep.tags.get("liar") or ep.tags.get("consultant")
                rows.append({"kind": kind, "mechanism": ep.mechanism, "item": ep.item_id, "arm": ep.tags.get("arm"),
                             "repeat": ep.repeat, "episode": ep.id, "role": role,
                             "assigned": ep.positions.get(role), "text": text,
                             "key": key_of(f"{kind}:{ep.mechanism}", ep.item_id, role, text), **m})

    t = run / "tournament"
    if t.exists():
        for jdir in (p for p in t.iterdir() if p.is_dir() and p.name in JUDGES):
            for kind_dir in ("debate", "consultancy"):
                for store in sorted((jdir / kind_dir).glob("*")) if (jdir / kind_dir).exists() else []:
                    add("tournament", store, judge=jdir.name, debater=None)
    mr = run / "multiround"
    if mr.exists():
        for store in sorted(mr.glob("*/*/*")):
            if store.is_dir():
                add("multiround", store, debater=store.parent.parent.name, judge=store.parent.name, config=store.name)
    jd = run / "judges"  # the stronger judges' fresh consultancies (their debates replay audited transcripts)
    if jd.exists():
        for store in sorted(jd.glob("*/*/*/consultancy")):
            ds, d, j = store.parent.parent.parent.name, store.parent.parent.name, store.parent.name
            if d in DEBATERS:
                add("judges", store, debater=d, judge=j, dataset=ds)
    q = run / "quality"
    if q.exists():
        for store in sorted(q.glob("*/*/*")):
            if store.is_dir() and store.parent.parent.name in DEBATERS:
                add("quality", store, debater=store.parent.parent.name, judge=store.parent.name)
    return rows


AUDITORS = {"gpt-4.1-mini": (AUDITOR, None), "gpt-5": ("openai/gpt-5", "medium")}


def verdict_file(out: Path, auditor: str) -> Path:
    return out / ("verdicts.jsonl" if auditor == "gpt-4.1-mini" else f"verdicts_{auditor}.jsonl")


async def audit_with(todo: list[dict], items: dict, out: Path, auditor: str, concurrency: int = 64) -> float:
    """Audit with a model through so-arena (reasoning models at the given effort); same prompt and verdicts."""
    from so_arena.core.types import GenerateOptions, Message
    from so_arena.models.base import get_model

    model, effort = AUDITORS[auditor]
    sem = asyncio.Semaphore(concurrency)
    cost = 0.0

    async def one(r):
        nonlocal cost
        it = items[r["item"]]
        truth = it.true_label
        assigned = r.get("assigned") or ("B" if truth == "A" else "A")
        other = "B" if assigned == "A" else "A"
        q = it.question + "\n\n" + "\n".join(f"({a.label}) {a.text}" for a in it.answers)
        p = PROMPT.format(assigned=assigned, truth=truth, which="WRONG", question=q,
                          text=r["text"][:12000] or "(nothing)", other=other)
        async with sem:
            try:
                c = await get_model(model).generate([Message.user(p)],
                                                    GenerateOptions(max_tokens=8000, reasoning_effort=effort))
            except Exception as e:  # noqa: BLE001 - a failed call is retried on the next run
                logging.getLogger("decisive").warning("audit call failed: %r", e)
                return
        cost += c.usage.cost_usd or 0.0
        import re

        m = re.search(r"\{.*\}", c.text or "", re.S)
        try:
            dd = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            dd = {}
        verdict = str(dd.get("verdict", "unparsed")).split("|")[0].strip().lower()
        tell = TELLS.search(r["text"])
        with open(out, "a") as f:
            f.write(json.dumps({"key": r["key"], "verdict": verdict, "evidence": str(dd.get("evidence", ""))[:300],
                                "tell": bool(tell), "tell_quote": tell.group(0) if tell else "",
                                "cost": c.usage.cost_usd or 0.0}) + "\n")

    await asyncio.gather(*[one(r) for r in todo])
    return cost


async def audit(todo: list[dict], items: dict, out: Path, estimate: bool) -> float:
    from cvtestbed.llm import LLM

    llm = LLM(concurrency=64, simulate=estimate, assumed_output=120)
    cost = 0.0

    async def one(r):
        nonlocal cost
        it = items[r["item"]]
        truth = it.true_label
        assigned = r.get("assigned") or ("B" if truth == "A" else "A")
        other = "B" if assigned == "A" else "A"
        q = it.question + "\n\n" + "\n".join(f"({a.label}) {a.text}" for a in it.answers)
        p = PROMPT.format(assigned=assigned, truth=truth, which="WRONG", question=q,
                          text=r["text"][:12000] or "(nothing)", other=other)
        reply = await llm.chat(AUDITOR, p, max_tokens=200, temperature=0.0, key=f"decisive-audit|{r['key']}")
        cost += reply.cost
        if estimate:
            return
        import re

        m = re.search(r"\{.*\}", reply.text, re.S)
        try:
            dd = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            dd = {}
        verdict = str(dd.get("verdict", "unparsed")).split("|")[0].strip().lower()
        tell = TELLS.search(r["text"])
        with open(out, "a") as f:
            f.write(json.dumps({"key": r["key"], "verdict": verdict, "evidence": str(dd.get("evidence", ""))[:300],
                                "tell": bool(tell), "tell_quote": tell.group(0) if tell else "",
                                "cost": reply.cost}) + "\n")

    await asyncio.gather(*[one(r) for r in todo])
    return cost


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/decisive")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--auditor", default="gpt-4.1-mini", choices=list(AUDITORS))
    ap.add_argument("--kinds", nargs="*", default=["prompted", "optimized", "tournament", "multiround", "quality",
                                                   "judges"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    map_keys()
    run = Path(a.run)
    out = run / "audit"
    out.mkdir(parents=True, exist_ok=True)
    items = {it.id: it for it in selected_items(run)}
    rows = []
    if "prompted" in a.kinds:
        rows += prompted_rows(run)
    if "optimized" in a.kinds:
        rows += optimized_rows(run)
    later = [r for r in stage_rows(run) if r["kind"] in a.kinds]
    if any(r["kind"] in ("quality", "judges") for r in later):
        from decisive_quality import quality_items

        items.update({it.id: it for it in quality_items(10_000)[1]})
    rows += later
    vf = verdict_file(out, a.auditor)
    done = {json.loads(line)["key"] for line in vf.read_text().splitlines()} if vf.exists() else set()
    todo = list({r["key"]: r for r in rows if r["key"] not in done}.values())
    print(f"{len(rows)} liar rows, {len({r['key'] for r in rows})} distinct texts, {len(todo)} to audit by {a.auditor}")
    if a.auditor == "gpt-4.1-mini":
        cost = asyncio.run(audit(todo, items, vf, a.estimate))
    elif a.estimate:
        print(f"about ${0.0054 * len(todo):.0f} (measured $0.0054 per text)")
        return
    else:
        cost = asyncio.run(audit_with(todo, items, vf, a.auditor))
    if a.estimate:
        print(f"estimated ${cost:.2f}")
        return
    frame = pd.DataFrame(rows).drop(columns=["text"])
    old = out / "rows.csv.gz"
    if old.exists():  # keep earlier rows (stores may since have been archived); new rows replace equal ones
        prev = pd.read_csv(old, low_memory=False)
        ids = [c for c in ("kind", "episode", "role", "judge", "pool_key", "index", "key") if c in frame]
        frame = pd.concat([prev, frame], ignore_index=True).drop_duplicates(ids, keep="last")
    frame.to_csv(old, index=False)
    verdicts = pd.read_json(vf, lines=True).drop_duplicates("key", keep="last")
    stage = "audit" if a.auditor == "gpt-4.1-mini" else f"audit_{a.auditor}"
    Ledger(run).set(stage, float(verdicts.cost.sum()))
    df = frame.merge(verdicts, on="key", how="left")
    print(df.groupby(["kind", "debater", "mechanism"]).verdict.value_counts(normalize=True).unstack().round(3))
    print(f"{a.auditor} audit cost ${verdicts.cost.sum():.2f}")


if __name__ == "__main__":
    main()
