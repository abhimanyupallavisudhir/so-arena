"""Self-contained HTML report for a set of results (tables + figures + transcript browser)."""

from __future__ import annotations

import base64
import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TIPS = {
    "asd": "Agent Score Difference: mean reward when behaving well minus when behaving badly (per task, averaged). >0 means the mechanism rewards truth.",
    "lo": "95% bootstrap CI (tasks resampled)",
    "hi": "95% bootstrap CI (tasks resampled)",
    "spearman": "Within-task rank correlation between reward and ground truth across sampled behaviours",
    "pairwise_acc": "Chance the reward ranks the better of two behaviours higher (0.5 = uninformative, <0.5 = perverse)",
    "slope": "Reward gained per unit of ground truth (task fixed effects)",
    "frontier_gap": "Best reward reachable by good strategies minus best reward reachable by bad ones — what a strong optimiser sees",
    "argmax_gt": "Ground truth of the strategy the mechanism rewards most",
    "best_gt": "Best ground truth among the sampled strategies",
    "gt_regret": "Ground truth lost if agents adopt the strategy the mechanism rewards most, instead of the best one sampled",
    "accuracy": "Share of final decisions that were correct",
    "p_correct": "Judge's mean probability on the correct answer",
    "log_score": "Judge's mean log probability on the correct answer",
    "ece": "Expected calibration error of the judge",
    "snr": "Reward signal-to-noise: variance between strategies / variance across samples of a strategy",
    "length_rho": "Within-task correlation of reward with words written, at fixed ground truth (verbosity bias)",
    "slot_advantage": "Reward advantage of debater A over debater B at equal ground truth",
    "r_good": "Mean reward of good behaviour",
    "r_bad": "Mean reward of bad behaviour",
}

CSS = """
:root{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e6e5e0;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#34342f;--acc:#3987e5}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
main{max-width:1120px;margin:0 auto;padding:22px}h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:22px 0 6px}
.mut{color:var(--mut)}.tiles{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.tile{border:1px solid var(--line);border-radius:8px;padding:8px 12px;min-width:110px}
.tile b{display:block;font-size:19px;font-weight:600}table{border-collapse:collapse;width:100%;margin:6px 0 10px;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}th{font-weight:600;white-space:nowrap}
th[title]{text-decoration:underline dotted;cursor:help}img{max-width:100%;width:560px;border:1px solid var(--line);border-radius:8px}
.figs{display:flex;flex-wrap:wrap;gap:14px}.figs>div{flex:0 1 560px}.scroll{overflow-x:auto}
tr.ep{cursor:pointer}tr.ep:hover{background:rgba(127,127,127,.08)}.tx{display:none}.tx td{background:rgba(127,127,127,.05)}
.msg{margin:4px 0;white-space:pre-wrap}.who{font-weight:600}input{padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg)}
details summary{cursor:pointer;color:var(--mut)}
"""


def _table(df: pd.DataFrame, digits: int = 3) -> str:
    if df is None or len(df) == 0:
        return '<div class="mut">—</div>'
    df = df.loc[:, [c for c in df.columns if not (df[c].dtype.kind == "f" and df[c].isna().all())]]
    cols = list(df.columns)
    head = "".join(f'<th title="{html.escape(TIPS[c])}">{html.escape(str(c))}</th>' if c in TIPS else f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                s = "" if np.isnan(v) else f"{v:.{digits}f}"
            else:
                s = html.escape(str(v))
            cells.append(f"<td>{s}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='scroll'><table><tr>{head}</tr>{''.join(rows)}</table></div>"


def _img(path: str | Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img alt="{html.escape(Path(path).stem)}" src="data:image/png;base64,{data}">'


def html_report(
    results: Any,
    path: str | Path,
    *,
    title: str = "OversightArena report",
    subtitle: str = "",
    gt: str = "correct",
    roles: Sequence[str] | None = None,
    figures: dict[str, str | Path] | None = None,
    tables: dict[str, pd.DataFrame] | None = None,
    transcripts: int = 200,
) -> Path:
    from . import diagnostics as D
    from .ic import ic_report
    from .plots import asd_bars, save

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = results.df()
    ok = df[~df["error"]] if len(df) else df
    trainable = ok[ok["trainable"]] if len(ok) else ok
    ep = results.episodes_df()
    tiles = {
        "episodes": len(results), "tasks": ep["task"].nunique() if len(ep) else 0,
        "mechanisms": ep["mechanism"].nunique() if len(ep) else 0, "errors": int(ep["error"].sum()) if len(ep) else 0,
    }
    if len(ep) and ep["tokens"].sum() > 0:
        tiles["tokens"] = f"{int(ep['tokens'].sum()):,}"
    if len(ep) and ep["cost_usd"].sum() > 0:
        tiles["cost"] = f"${ep['cost_usd'].sum():.2f}"
    parts = [f"<h1>{html.escape(title)}</h1><div class='mut'>{html.escape(subtitle)}</div>"]
    parts.append("<div class='tiles'>" + "".join(f"<div class='tile'><span class='mut'>{k}</span><b>{v}</b></div>" for k, v in tiles.items()) + "</div>")
    figs = dict(figures or {})
    gcol = f"gt_{gt}"
    if len(trainable) and gcol in trainable and trainable[gcol].notna().any():
        ic = ic_report(trainable, roles=roles, gt=gt)
        parts.append(f"<h2>Incentive compatibility <span class='mut'>(ground truth: {html.escape(gt)})</span></h2>")
        keep = [c for c in ["mechanism", "asd", "lo", "hi", "pairwise_acc", "spearman", "slope", "frontier_gap", "argmax_gt",
                            "best_gt", "gt_regret", "r_good", "r_bad", "n_tasks"] if c in ic]
        parts.append(_table(ic[keep]))
        if len(ic) and "asd" in ic and ic["asd"].notna().any():
            fig, _ = asd_bars(ic)
            figs.setdefault("ASD by mechanism", save(fig, path.with_name(path.stem + "_asd.png")))
    if figs:
        parts.append("<h2>Figures</h2><div class='figs'>" + "".join(f"<div><div class='mut'>{html.escape(k)}</div>{_img(v)}</div>" for k, v in figs.items()) + "</div>")
    try:
        om = D.outcome_metrics(results)
        if len(om.columns) > 2:
            parts.append("<h2>Outcomes (principal's view)</h2>" + _table(om.drop(columns=[c for c in om.columns if c.endswith(("_lo", "_hi"))], errors="ignore")))
    except Exception:
        pass
    diag = []
    for name, fn in [("Reward signal quality", lambda: D.reward_snr(trainable)), ("Verbosity bias", lambda: D.length_bias(trainable, gt=gt)),
                     ("Slot bias (debate)", lambda: D.position_bias(trainable, gt=gt)), ("Cost", lambda: D.cost_summary(results))]:
        try:
            t = fn()
            num = t.select_dtypes("number").drop(columns=["groups", "episodes"], errors="ignore")
            if len(t) and (num.fillna(0) != 0).any().any():
                diag.append(f"<h2>{name}</h2>" + _table(t))
        except Exception:
            continue
    parts += diag
    for k, t in (tables or {}).items():
        parts.append(f"<h2>{html.escape(k)}</h2>" + _table(t))
    # transcript browser
    items = []
    for r in results.records[:transcripts]:
        items.append({
            "task": r.task_id, "mechanism": r.mechanism, "profile": r.profile.label or r.profile.id,
            "rewards": r.rewards, "gt": {k: v for k, v in r.gt.items()}, "decision": r.outcome.get("decision"),
            "tx": [{"role": e.role, "kind": e.kind, "content": e.content[:4000], "ev": [x.render()[:500] for x in e.evidence]} for e in r.transcript.entries],
            "error": (r.error or "")[:300],
        })
    parts.append("<h2>Episodes <input id='q' placeholder='filter…'></h2><table id='eps'></table>")
    js = """
const I=%s;const eps=document.getElementById('eps');const esc=s=>(s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'})[c]);
const f3=x=>typeof x==='number'?x.toFixed(3):(x??'');
let LIM=25;
function render(q){q=(q||'').toLowerCase();let h='<tr><th>task</th><th>mechanism</th><th>profile</th><th>decision</th><th>rewards</th></tr>';let n=0,shown=0;
I.forEach((it,i)=>{if(q&&!(it.task+' '+it.mechanism+' '+it.profile).toLowerCase().includes(q))return;n++;if(shown>=LIM)return;shown++;
h+=`<tr class="ep" onclick="t(${i})"><td>${esc(it.task)}</td><td>${esc(it.mechanism)}</td><td>${esc(it.profile)}</td><td>${esc(String(it.decision??''))}</td><td>${Object.entries(it.rewards).map(([k,v])=>k+': '+f3(v)).join('<br>')}</td></tr>`;
h+=`<tr class="tx" id="x${i}"><td colspan="5">${it.error?'<div class=mut>'+esc(it.error)+'</div>':''}${it.tx.map(m=>`<div class="msg"><span class="who">${esc(m.role||'moderator')}</span> ${esc(m.content)}${m.ev.map(e=>'<div class=mut>'+esc(e)+'</div>').join('')}</div>`).join('')}<details><summary>ground truth</summary><pre>${esc(JSON.stringify(it.gt,null,1))}</pre></details></td></tr>`});
if(n>shown)h+=`<tr><td colspan="5"><a href="#" onclick="LIM=1e9;render(document.getElementById('q').value);return false">show all ${n}</a></td></tr>`;eps.innerHTML=h}
function t(i){const e=document.getElementById('x'+i);e.style.display=e.style.display==='table-row'?'none':'table-row'}
document.getElementById('q').oninput=e=>render(e.target.value);render('');
""" % json.dumps(items, default=str).replace("</", "<\\/")
    doc = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>{''.join(parts)}</main><script>{js}</script></body></html>"
    path.write_text(doc)
    return path
