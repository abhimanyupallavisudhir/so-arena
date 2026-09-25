"""Releasing mechanism results *before* ground truth is known — and resolving them later.

Workflow::

    rel = create_release(results, "releases/forecast-debate-2026-09", title="...")
    # publish rel.root (e.g. in a git commit / post) to timestamp the commitment
    ...                                  # months later, questions resolve
    tasks = ManifoldForecasting.resolve(results.tasks.values())
    report = resolve_release("releases/forecast-debate-2026-09", tasks)

A release contains, per episode, what the *mechanism* said (outcome, decision, judge
probabilities, rewards, strategy labels, optionally transcripts) — and nothing about ground
truth. Items are hash-committed (Merkle root in ``manifest.json``); with ``sealed=True`` only
salted commitments are published and contents are revealed later (``reveal.json``), which
lets you prove *what* the mechanism ranked highly without influencing the outcome.
Resolution attaches ground truth and computes incentive-compatibility and accuracy metrics
retroactively, rendering everything in a static, self-contained ``index.html``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.episode import EpisodeRecord
from ..core.task import Task
from ..core.util import now_iso
from .commit import leaf_hash, merkle_proof, merkle_root, new_salt, verify_proof


def _names_safe(rec: EpisodeRecord) -> bool:
    """Strategy names and profile labels may encode ground truth (``argue_incorrect``,
    ``debater_a=correct``) when strategies have stances; publish them only if none does."""
    from ..core.strategy import Stance

    return all(b.stance == Stance.FREE for b in rec.bound.values())


def _item(rec: EpisodeRecord, transcripts: bool, names: str = "auto") -> dict[str, Any]:
    o = rec.outcome
    show = names == "always" or (names == "auto" and _names_safe(rec))
    item: dict[str, Any] = {
        "episode": rec.id,
        "task": rec.task_id,
        "mechanism": rec.mechanism,
        "mechanism_hash": rec.mechanism_hash,
        "reward_rule": rec.reward_rule,
        "profile": (rec.profile.label or rec.profile.id) if show else rec.profile.id,
        "strategies": {r: (b.strategy_name if show else b.strategy_id.rsplit("-", 1)[-1]) for r, b in rec.bound.items()},
        "agents": {r: b.agent for r, b in rec.bound.items()},
        "positions": o.get("positions"),
        "decision": o.get("decision") if o.get("decision") is not None else o.get("verdict"),
        "probs": o.get("probs"),
        "forecasts": o.get("forecasts"),
        "rewards": rec.rewards,
        "rewards_pending": bool(rec.meta.get("rewards_pending")),
        "created_at": rec.created_at,
    }
    if transcripts:
        item["transcript"] = [
            {"role": e.role, "kind": e.kind, "content": e.content, "step": e.step,
             "evidence": [ev.render() for ev in e.evidence]}
            for e in rec.transcript.entries
        ]
    return item


def _task_item(t: Task) -> dict[str, Any]:
    v = t.view()
    return {"id": v.id, "domain": v.domain, "question": v.question,
            "options": [{"id": o.id, "text": o.text} for o in v.options],
            "metadata": {k: x for k, x in v.metadata.items() if isinstance(x, (str, int, float, bool)) or x is None}}


class Release:
    def __init__(self, path: Path, manifest: dict[str, Any]):
        self.path = path
        self.manifest = manifest

    @property
    def root(self) -> str:
        return self.manifest["root"]

    def __repr__(self) -> str:
        return f"Release({str(self.path)!r}, items={self.manifest['n_items']}, root={self.root[:16]}…)"


def create_release(
    results: Any,
    out_dir: str | Path,
    *,
    title: str = "Mechanism results",
    description: str = "",
    transcripts: bool = True,
    sealed: bool = False,
    private_dir: str | Path | None = None,
    tasks: Sequence[Task] | None = None,
    names: str = "auto",
) -> Release:
    """Publishable release of mechanism outputs (no ground truth). ``sealed``: publish only
    salted commitments; the items and salts go to ``private_dir`` (default
    ``<out_dir>.private``, *outside* the directory you publish) until :func:`reveal_release`.
    ``names``: publish strategy names and profile labels ``"always"``, ``"never"`` (hashed ids),
    or ``"auto"`` — only for episodes whose strategies have no stance, since names like
    ``argue_incorrect`` would reveal the answer."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    recs = [r for r in results.records if r.error is None]
    items = [_item(r, transcripts, names) for r in recs]
    task_map = {t.id: t for t in (tasks or results.tasks.values())}
    task_items = [_task_item(task_map[tid]) for tid in sorted({i["task"] for i in items}) if tid in task_map]
    salts = {i["episode"]: new_salt() for i in items} if sealed else {}
    leaves = {i["episode"]: leaf_hash(i, salts.get(i["episode"], "")) for i in items}
    root = merkle_root(list(leaves.values()))
    from .. import __version__

    manifest = {
        "title": title, "description": description, "created_at": now_iso(), "library": f"oversight-arena {__version__}",
        "n_items": len(items), "root": root, "sealed": sealed, "leaves": leaves,
        "mechanisms": sorted({i["mechanism"] for i in items}), "status": "unresolved",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "tasks.json").write_text(json.dumps(task_items, indent=1))
    if sealed:
        priv = Path(private_dir) if private_dir is not None else out.with_name(out.name + ".private")
        priv.mkdir(parents=True, exist_ok=True)
        (out / "sealed_items.json").write_text(json.dumps([{"episode": k} for k in leaves]))
        (priv / "reveal.json").write_text(json.dumps({"salts": salts, "items": items}))  # keep private until reveal!
    else:
        (out / "items.json").write_text(json.dumps(items))
    render_html(out)
    return Release(out, manifest)


def reveal_release(release_dir: str | Path, private_dir: str | Path | None = None) -> dict[str, Any]:
    """Open a sealed release: publish its items and salts, then verify them against the
    commitments made at creation time."""
    d = Path(release_dir)
    priv = Path(private_dir) if private_dir is not None else d.with_name(d.name + ".private")
    data = json.loads((priv / "reveal.json").read_text())
    (d / "reveal.json").write_text(json.dumps(data))
    rep = verify_release(d)
    if rep["ok"]:
        (d / "items.json").write_text(json.dumps(data["items"]))
        render_html(d)
    return rep


def load_items(release_dir: str | Path) -> list[dict[str, Any]]:
    d = Path(release_dir)
    if (d / "items.json").exists():
        return json.loads((d / "items.json").read_text())
    if (d / "reveal.json").exists():
        return json.loads((d / "reveal.json").read_text())["items"]
    raise FileNotFoundError("release has no items.json (sealed and not yet revealed)")


def verify_release(release_dir: str | Path) -> dict[str, Any]:
    """Recompute every leaf and the Merkle root; report tampered, missing and extra items.
    (A sealed, unrevealed release can only have its root checked.)"""
    d = Path(release_dir)
    man = json.loads((d / "manifest.json").read_text())
    root_ok = merkle_root(list(man["leaves"].values())) == man["root"]
    try:
        items = load_items(d)
    except FileNotFoundError:
        return {"root_ok": root_ok, "n_items": 0, "sealed": True, "tampered": [], "missing": [], "extra": [], "ok": root_ok}
    salts = json.loads((d / "reveal.json").read_text())["salts"] if man.get("sealed") and (d / "reveal.json").exists() else {}
    bad = []
    for it in items:
        lh = leaf_hash(it, salts.get(it["episode"], ""))
        if man["leaves"].get(it["episode"]) != lh:
            bad.append(it["episode"])
    present = {it["episode"] for it in items}
    missing = sorted(set(man["leaves"]) - present)
    extra = sorted(present - set(man["leaves"]))
    return {"root_ok": root_ok, "n_items": len(items), "tampered": bad, "missing": missing, "extra": extra,
            "ok": root_ok and not bad and not missing and not extra}


def inclusion_proof(release_dir: str | Path, episode: str) -> dict[str, Any]:
    man = json.loads((Path(release_dir) / "manifest.json").read_text())
    leaves = list(man["leaves"].values())
    leaf = man["leaves"][episode]
    proof = merkle_proof(leaves, leaf)
    return {"leaf": leaf, "proof": proof, "root": man["root"], "valid": verify_proof(leaf, proof, man["root"])}


def resolve_release(
    release_dir: str | Path,
    tasks: Sequence[Task] | dict[str, Task],
    records: Any = None,
    scorers: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Attach ground truth to a release and compute retroactive metrics.

    ``tasks`` must include ground truth (e.g. refreshed forecasting questions). If the full
    ``records`` (Results) are given, per-role GT scorers are run on them; otherwise the item
    positions/decisions/forecasts are scored directly.
    """
    from ..analysis.ic import alignment, asd
    from ..mechanisms.forecasting import proper_score

    d = Path(release_dir)
    ver = verify_release(d)
    items = load_items(d)
    tmap = tasks if isinstance(tasks, dict) else {t.id: t for t in tasks}
    rows = []
    for it in items:
        t = tmap.get(it["task"])
        if t is None or not t.resolved:
            continue
        correct = set(t.correct_ids()) if t.has_values() else set()
        y = t.gt.get("outcome")
        rewards = it.get("rewards") or {}
        roles = sorted(set(rewards) | set(it.get("positions") or {}) | set(it.get("forecasts") or {}))
        for role in roles:
            pos = (it.get("positions") or {}).get(role)
            fc = (it.get("forecasts") or {}).get(role)
            rew = rewards.get(role)  # None for delayed rewards: the released output is the forecast itself
            row = {"episode": it["episode"], "task": it["task"], "mechanism": it["mechanism"], "role": role,
                   "strategy_name": (it.get("strategies") or {}).get(role), "reward": rew, "error": False}
            if pos is not None and correct:
                row["gt_correct"] = float(pos in correct)
            if fc is not None and y is not None:
                row["gt_forecast_log"] = proper_score(float(fc), float(y), "log")
                row["gt_forecast_brier"] = proper_score(float(fc), float(y), "brier")
            dec = it.get("decision")
            if dec is not None and correct:
                row["gt_decision_correct"] = float(dec in correct)
            probs = it.get("probs") or {}
            if probs and correct:
                row["gt_judge_p_correct"] = sum(float(probs.get(o, 0)) for o in correct)
            rows.append(row)
    df = pd.DataFrame(rows)
    report: dict[str, Any] = {"resolved_at": now_iso(), "verification": ver, "n_resolved_rows": len(df)}
    if not df.empty:
        if "gt_correct" in df:
            report["asd"] = asd(df.dropna(subset=["gt_correct"]), gt="correct").to_dict("records")
            report["alignment_correct"] = alignment(df.dropna(subset=["gt_correct"]), gt="correct").to_dict("records")
        for g in ("forecast_log", "forecast_brier"):
            col = f"gt_{g}"
            if col in df and df[col].notna().any():
                report[f"alignment_{g}"] = alignment(df.dropna(subset=[col]), gt=g).to_dict("records")
        acc = {}
        for col in ("gt_decision_correct", "gt_judge_p_correct"):
            if col in df:
                acc[col] = df.drop_duplicates("episode").groupby("mechanism")[col].mean().to_dict()
        report["outcome_accuracy"] = acc
        # released ranking vs resolved truth, per strategy
        gcols = [c for c in df.columns if c.startswith("gt_")]
        agg = df.groupby(["mechanism", "role", "strategy_name"], dropna=False).agg(
            reward=("reward", "mean"), n=("episode", "nunique"), **{c: (c, "mean") for c in gcols}).reset_index()
        report["by_strategy"] = agg.to_dict("records")
    (d / "resolution.json").write_text(json.dumps(report, indent=1, default=str))
    df.to_csv(d / "resolution_rows.csv", index=False)
    man = json.loads((d / "manifest.json").read_text())
    if not ver["ok"]:
        man["status"] = "resolution failed verification"
    elif len(df):
        man["status"] = "resolved"
    (d / "manifest.json").write_text(json.dumps(man, indent=1))
    render_html(d)
    return report


# --------------------------------------------------------------------------- HTML
_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1d1d1f;--mut:#6b6b70;--line:#e5e5e2;--acc:#3b6fd4;--good:#2e8b57;--bad:#c0392b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16161a;--fg:#ececef;--mut:#9a9aa3;--line:#2c2c33;--acc:#7aa2ff;--good:#57c28a;--bad:#ff7b6b}}}}
*{{box-sizing:border-box}}body{{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}}
main{{max-width:1100px;margin:0 auto;padding:20px}}h1{{font-size:20px;margin:0 0 4px}}.mut{{color:var(--mut)}}
.cards{{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}}.card{{border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:120px}}
.card b{{display:block;font-size:18px}}table{{border-collapse:collapse;width:100%;margin:8px 0 18px}}th,td{{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}}
th{{cursor:pointer;font-weight:600;white-space:nowrap}}tr.ep{{cursor:pointer}}tr.ep:hover{{background:rgba(127,127,127,.08)}}
.tx{{display:none}}.tx td{{background:rgba(127,127,127,.05)}}.msg{{margin:4px 0;white-space:pre-wrap}}.who{{font-weight:600}}
.g{{color:var(--good)}}.b{{color:var(--bad)}}h2[title],th[title],span[title]{{text-decoration:underline dotted;text-decoration-thickness:1px;cursor:help}}.card{{cursor:default}}
input{{padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);width:220px}}
code{{font-size:12px}}h2{{font-size:16px;margin:18px 0 4px}}
</style></head><body><main>
<h1>{title}</h1><div class="mut">{description}</div>
<div class="cards" id="cards"></div>
<h2 title="Mean reward each strategy received from the mechanism">Leaderboard</h2><table id="lb"></table>
<div id="res"></div>
<h2>Episodes <input id="q" placeholder="filter…"></h2><table id="eps"></table>
<div class="mut">Root <code title="SHA-256 Merkle root over all items; publish it to timestamp this release">{root}</code></div>
</main>
<script>
const M={manifest};const T={tasks};const I={items};const R={resolution};
const tq=Object.fromEntries(T.map(t=>[t.id,t]));
const fmt=x=>x==null?'':(typeof x==='number'?x.toFixed(3):x);
function esc(s){{return String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c])}}
function card(k,v,tip){{return `<div class="card"><span class="mut" ${{tip?`title="${{esc(tip)}}"`:''}}>${{esc(k)}}</span><b>${{esc(v)}}</b></div>`}}
let c=card('items',M.n_items)+card('mechanisms',M.mechanisms.length)+card('status',M.status,'unresolved = ground truth not yet attached');
if(R&&R.outcome_accuracy&&R.outcome_accuracy.gt_decision_correct){{for(const [m,v] of Object.entries(R.outcome_accuracy.gt_decision_correct))c+=card(m+' accuracy',(100*v).toFixed(1)+'%','Share of final decisions that were correct')}}
if(R&&R.asd){{for(const r of R.asd)c+=card(r.mechanism+' ASD',fmt(r.asd),'Agent Score Difference: reward for arguing the true answer minus reward for arguing a false one')}}
document.getElementById('cards').innerHTML=c;
if(R&&R.by_strategy&&R.by_strategy.length){{const gk=Object.keys(R.by_strategy[0]).filter(k=>k.startsWith('gt_'));
 const nice=k=>k.slice(3).replace(/_/g,' ');
 let h='<h2 title="After resolution: the reward the mechanism gave each strategy, next to its ground-truth scores">Released ranking vs resolved truth</h2><table><tr><th>mechanism</th><th>role</th><th>strategy</th><th>mean reward</th>'+gk.map(k=>`<th>${{esc(nice(k))}}</th>`).join('')+'<th>n</th></tr>';
 for(const r of [...R.by_strategy].sort((a,b)=>(b.reward??-1e9)-(a.reward??-1e9)))h+=`<tr><td>${{esc(r.mechanism)}}</td><td>${{esc(r.role)}}</td><td>${{esc(r.strategy_name)}}</td><td>${{fmt(r.reward)}}</td>`+gk.map(k=>`<td>${{fmt(r[k])}}</td>`).join('')+`<td>${{r.n}}</td></tr>`;
 document.getElementById('res').innerHTML=h+'</table>'}}
// leaderboard: mean reward per (mechanism, role, strategy)
const agg={{}};for(const it of I){{for(const [r,v] of Object.entries(it.rewards||{{}})){{const s=(it.strategies||{{}})[r]||'';const k=it.mechanism+'|'+r+'|'+s;(agg[k]=agg[k]||[]).push(v)}}}}
let rows=Object.entries(agg).map(([k,v])=>{{const [m,r,s]=k.split('|');return [m,r,s,v.reduce((a,b)=>a+b,0)/v.length,v.length]}}).sort((a,b)=>b[3]-a[3]);
function table(el,head,rows,fmtRow){{el.innerHTML='<tr>'+head.map((h,i)=>`<th data-i="${{i}}">${{h}}</th>`).join('')+'</tr>'+rows.map(fmtRow).join('');
 el.querySelectorAll('th').forEach(th=>th.onclick=()=>{{const i=+th.dataset.i;rows.sort((a,b)=>(a[i]>b[i]?-1:1));table(el,head,rows,fmtRow)}})}}
table(document.getElementById('lb'),['mechanism','role','strategy','mean reward','n'],rows,r=>`<tr><td>${{esc(r[0])}}</td><td>${{esc(r[1])}}</td><td>${{esc(r[2])}}</td><td>${{fmt(r[3])}}</td><td>${{r[4]}}</td></tr>`);
const eps=document.getElementById('eps');
function renderEps(f){{f=(f||'').toLowerCase();let h='<tr><th>task</th><th>mechanism</th><th>profile</th><th title="Mechanism decision">decision</th><th>rewards</th></tr>';
 I.forEach((it,i)=>{{const t=tq[it.task]||{{question:it.task,options:[]}};const txt=(t.question+' '+it.mechanism+' '+it.profile).toLowerCase();if(f&&!txt.includes(f))return;
  const opt=(t.options||[]).find(o=>o.id===it.decision);const dec=it.decision==null?'':(opt?(opt.text.includes(it.decision)?opt.text.slice(0,50):it.decision+': '+opt.text.slice(0,40)):String(it.decision));
  h+=`<tr class="ep" onclick="tg(${{i}})"><td>${{esc(t.question.slice(0,90))}}</td><td>${{esc(it.mechanism)}}</td><td>${{esc(it.profile)}}</td><td>${{esc(dec)}}</td><td>${{Object.entries(it.rewards||{{}}).map(([r,v])=>esc(r)+': '+fmt(v)).join('<br>')}}</td></tr>`;
  h+=`<tr class="tx" id="tx${{i}}"><td colspan="5">${{(it.transcript||[]).map(m=>`<div class="msg"><span class="who">${{esc(m.role||'moderator')}}</span> ${{esc(m.content)}}${{(m.evidence||[]).map(e=>'<div class=mut>'+esc(e)+'</div>').join('')}}</div>`).join('')||'<span class=mut>no transcript</span>'}}</td></tr>`}});eps.innerHTML=h}}
function tg(i){{const e=document.getElementById('tx'+i);e.style.display=e.style.display==='table-row'?'none':'table-row'}}
document.getElementById('q').oninput=e=>renderEps(e.target.value);renderEps('');
</script></body></html>"""


def render_html(release_dir: str | Path) -> Path:
    d = Path(release_dir)
    man = json.loads((d / "manifest.json").read_text())
    tasks = json.loads((d / "tasks.json").read_text()) if (d / "tasks.json").exists() else []
    try:
        items = load_items(d) if not man.get("sealed") else []
    except FileNotFoundError:
        items = []
    res = json.loads((d / "resolution.json").read_text()) if (d / "resolution.json").exists() else None
    slim = {k: v for k, v in man.items() if k != "leaves"}

    def js(x: Any) -> str:  # no raw '<' inside <script> (e.g. '</script>' or '<!--' in item text)
        return json.dumps(x, default=str).replace("<", "\\u003c")

    import html as _html

    page = _TEMPLATE.format(
        title=_html.escape(man.get("title", "Release")), description=_html.escape(man.get("description", "")),
        root=_html.escape(man["root"]), manifest=js(slim), tasks=js(tasks), items=js(items), resolution=js(res),
    )
    p = d / "index.html"
    p.write_text(page)
    return p
