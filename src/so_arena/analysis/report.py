"""Self-contained HTML reports: metrics, charts (with per-mark tooltips and table views), episodes.

The page is compact by design: explanations live in hover/tap tooltips (the small "?" marks), every
chart has a table-view twin, charts are rendered for both light and dark surfaces, and all
transcript text is escaped (model output is untrusted). Ground truth can be hidden entirely for
releases made before it is known.
"""

from __future__ import annotations

import datetime as _dt
import html
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from so_arena.analysis.frames import config_key, mechanism_labels
from so_arena.analysis.plots import Chart
from so_arena.core.mechanism import Episode

TIPS = {
    "asd": "How much more reward an agent earns for arguing for the truth than for a falsehood. Above 0: honesty pays.",
    "asd_brier": "The same difference measured with the Brier score (bounded: a perfect judge gives 2).",
    "graded_asd": "Covariance between reward and true quality across all behaviours on an item.",
    "concordance": "How often the truly better behaviour receives the higher reward. 0.5 is chance, 1 is perfect.",
    "within_item_corr": "Correlation between reward and true value after removing per-item differences.",
    "label_efficiency": "How many ground-truth audits one audit is worth when this reward is used alongside it.",
    "accuracy": "How often the final decision is right (the classic 'control' measure).",
    "p_true": "Average probability the final decision puts on the right answer.",
    "igr_brier": "Share of the gap between judging alone and a perfect judge that this mechanism closes.",
    "cost_per_episode": "Average API cost per episode.",
    "n_items": "Number of items with both behaviours observed.",
    "margin": ("Reward of the best strategy measured honest minus that of the best measured deceptive one (by ground "
               "truth, whatever the search was told). Below 0: deception wins."),
    "parse_fail_rate": ("Share of episodes whose judgment could not be read and fell back to a default (50/50): "
                        "such episodes pay honest and dishonest behaviour alike."),
    "value_coverage": "Share of the episodes behind an average ground-truth value that had one.",
}

CSS = """
.so-root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--ring:rgba(11,11,11,.10);--good:#006300;--bad:#d03b3b;--warn:#8a6100;--chip:#f0efec;color-scheme:light}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .so-root{--surface:#1a1a19;--page:#0d0d0d;
--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--good:#0ca30c;--bad:#e66767;
--warn:#fab219;--chip:#2c2c2a;color-scheme:dark}}
:root[data-theme="dark"] .so-root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--ring:rgba(255,255,255,.10);--good:#0ca30c;--bad:#e66767;--warn:#fab219;--chip:#2c2c2a;color-scheme:dark}
body{margin:0;background:var(--page)}
.so-root{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--page);
max-width:1080px;margin:0 auto;padding:20px 16px 48px;font-size:14px;line-height:1.45}
h1{font-size:20px;margin:0 0 2px;font-weight:650}h2{font-size:15px;margin:28px 0 10px;font-weight:650}
.sub{color:var(--ink2);font-size:13px}
.kpis{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0 4px}
.kpi{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:10px 14px;min-width:120px}
.kpi .l{color:var(--ink2);font-size:12px}.kpi .v{font-size:20px;font-weight:600}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:12px;padding:14px 14px 10px;margin:10px 0}
.fig svg{max-width:min(100%,760px);height:auto;display:block}
.fig .fig-dark{display:none}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .fig .fig-dark{display:block}
:root:where(:not([data-theme="light"])) .fig .fig-light{display:none}}
:root[data-theme="dark"] .fig .fig-dark{display:block}:root[data-theme="dark"] .fig .fig-light{display:none}
:root[data-theme="light"] .fig .fig-dark{display:none}:root[data-theme="light"] .fig .fig-light{display:block}
.mark:hover,.mark:focus{opacity:.8;outline:none}
details>summary{cursor:pointer;color:var(--ink2);font-size:13px;list-style:none}
details>summary::-webkit-details-marker{display:none}
details>summary:before{content:"▸ ";color:var(--muted)}details[open]>summary:before{content:"▾ "}
table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600;white-space:nowrap}td.n{font-variant-numeric:tabular-nums;text-align:right}
th.n{text-align:right}
.tip{position:relative;display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;
border-radius:50%;border:1px solid var(--muted);color:var(--muted);font-size:10px;margin-left:4px;cursor:help;
font-weight:600;vertical-align:1px}
.tip:hover:after,.tip:focus:after{content:attr(data-tip);position:absolute;z-index:9;top:20px;left:-8px;width:240px;
background:var(--ink);color:var(--surface);padding:8px 10px;border-radius:8px;font-size:12px;font-weight:400;
line-height:1.35;white-space:normal;text-align:left}
.filters{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0 10px}
.filters select,.filters input{font:inherit;font-size:13px;padding:5px 8px;border-radius:8px;border:1px solid var(--ring);
background:var(--surface);color:var(--ink)}
.ep{background:var(--surface);border:1px solid var(--ring);border-radius:10px;margin:6px 0;padding:8px 12px}
.ep>summary{color:var(--ink);font-size:13px}
.chip{display:inline-block;background:var(--chip);border-radius:6px;padding:0 6px;font-size:12px;color:var(--ink2);margin-right:4px}
.turn{margin:8px 0;padding-left:10px;border-left:2px solid var(--grid)}
.turn .who{font-size:12px;color:var(--ink2);font-weight:600}
.turn .txt{white-space:pre-wrap;word-wrap:break-word}
.v{font-size:12px;border-radius:5px;padding:0 4px;margin-right:2px;font-weight:600}
.v.ok{color:var(--good)}.v.bad{color:var(--bad)}.v.unk{color:var(--warn)}
.gt{margin-top:8px;font-size:12px;color:var(--ink2)}
.muted{color:var(--muted)}
.theme{float:right;font:inherit;font-size:12px;border:1px solid var(--ring);background:var(--surface);color:var(--ink2);
border-radius:8px;padding:3px 8px;cursor:pointer}
"""

JS = """
(function(){
  const root=document.documentElement, b=document.getElementById('theme');
  if(b){b.addEventListener('click',()=>{const d=root.getAttribute('data-theme');
    const dark=d? d==='light' : !window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.setAttribute('data-theme', dark?'dark':'light');});}
  const m=document.getElementById('f-mech'), p=document.getElementById('f-prof'), q=document.getElementById('f-q'),
        n=document.getElementById('f-n');
  function apply(){let shown=0;document.querySelectorAll('.ep').forEach(e=>{
    const ok=(!m||!m.value||e.dataset.mech===m.value)&&(!p||!p.value||e.dataset.prof===p.value)&&
      (!q||!q.value||e.textContent.toLowerCase().includes(q.value.toLowerCase()));
    e.style.display=ok?'':'none'; if(ok) shown++;}); if(n) n.textContent=shown+' shown';}
  [m,p,q].forEach(x=>x&&x.addEventListener('input',apply)); apply();
})();
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def tip(key_or_text: str) -> str:
    text = TIPS.get(key_or_text, key_or_text)
    return f'<span class="tip" tabindex="0" data-tip="{_e(text)}">?</span>'


def _num(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int,)) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, float):
        if not math.isfinite(v):
            return "–"
        return f"{v:.3f}" if abs(v) < 100 else f"{v:,.1f}"
    return _e(v)


def table_html(df: pd.DataFrame, *, tips: dict[str, str] | None = None, max_rows: int = 500) -> str:
    tips = {**TIPS, **(tips or {})}
    cols = list(df.columns)
    head = "".join(
        f'<th class="{"n" if pd.api.types.is_numeric_dtype(df[c]) else ""}">{_e(c)}{tip(tips[c]) if c in tips else ""}</th>'
        for c in cols)
    rows = []
    for _, r in df.head(max_rows).iterrows():
        cells = "".join(
            f'<td class="{"n" if pd.api.types.is_numeric_dtype(df[c]) else ""}">{_num(r[c]) if pd.api.types.is_numeric_dtype(df[c]) else _e(r[c])}</td>'
            for c in cols)
        rows.append(f"<tr>{cells}</tr>")
    more = f'<p class="muted">{len(df) - max_rows} more rows not shown</p>' if len(df) > max_rows else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>{more}"


_MARK = re.compile(r"&lt;(verified|failed|unverified|executed) kind=&quot;([\w\-\.]+)&quot;&gt;(.*?)&lt;/\1&gt;", re.S)
_RESULT = re.compile(r"&lt;result&gt;(.*?)&lt;/result&gt;", re.S)
# what the verdict was checked against (expect=, goal=, of=, ...): shown before the claim, compactly
_CHECKED = re.compile(r"&lt;checked\s*(.*?)\s*/&gt;", re.S)
_ATTR = re.compile(r"([\w-]+)=&quot;(.*?)&quot;", re.S)


def render_text(text: str) -> str:
    """Escape model text, then turn verification markers into badges."""
    s = _e(text)

    def badge(m: re.Match) -> str:
        status, kind, body = m.group(1), m.group(2), m.group(3)
        body = _RESULT.sub(lambda r: f' <span class="muted">→ {r.group(1)}</span>', body)
        body = _CHECKED.sub(lambda c: '<span class="muted">' + " ".join(f"{k}={v}" for k, v in _ATTR.findall(c.group(1)))
                            + "</span> " if _ATTR.search(c.group(1)) else "", body)
        cls, icon, label, tip = {
            "verified": ("ok", "✓", "verified", "checked and true, by a trusted tool"),
            "failed": ("bad", "✗", "failed", "checked and false, by a trusted tool"),
            "unverified": ("unk", "?", "unverified", "not checked"),
            "executed": ("unk", "▸", "ran", "run by a trusted tool without checking a stated result: the output is "
                                            "the participant's own code's")}[status]
        return (f'<span class="v {cls}" title="{tip}">{icon} {label} {kind}</span>'
                f"<u>{body}</u>")

    return _MARK.sub(badge, s)


def gt_chips(gt: dict[str, Any]) -> str:
    parts = []
    for k, v in gt.items():
        if isinstance(v, dict):
            inner = ", ".join(f"{_e(r)} {_num(float(x)) if isinstance(x, (int, float)) and not isinstance(x, bool) else _e(x)}"
                              for r, x in v.items())
            parts.append(f'<span class="chip">{_e(k)}: {inner}</span>')
        elif isinstance(v, (int, float, str, bool)):
            parts.append(f'<span class="chip">{_e(k)} {_num(v) if not isinstance(v, str) else _e(v[:80])}</span>')
    return " ".join(parts)


class Report:
    def __init__(self, title: str, subtitle: str = ""):
        self.title, self.subtitle = title, subtitle
        self.parts: list[str] = []

    def kpis(self, items: dict[str, Any]) -> "Report":
        cells = "".join(f'<div class="kpi"><div class="l">{_e(k)}</div><div class="v">{_num(v) if not isinstance(v, str) else _e(v)}</div></div>'
                        for k, v in items.items())
        self.parts.append(f'<div class="kpis">{cells}</div>')
        return self

    def section(self, title: str, *, charts: dict[str, Chart] | Chart | None = None, table: pd.DataFrame | None = None,
                note: str | None = None, tips: dict[str, str] | None = None, info: str | None = None,
                table_open: bool = False) -> "Report":
        out = [f"<h2>{_e(title)}{tip(info) if info else ''}</h2>", '<div class="card">']
        if charts is not None:
            if isinstance(charts, Chart):
                charts = {"light": charts}
            light = charts.get("light")
            dark = charts.get("dark", light)
            out.append('<div class="fig">')
            if light is not None:
                out.append(f'<div class="fig-light">{light.svg()}</div>')
            if dark is not None:
                out.append(f'<div class="fig-dark">{dark.svg()}</div>')
            out.append("</div>")
            tbl = table if table is not None else (light.table if light is not None else None)
            for c in charts.values():
                c.close()
            if tbl is not None:
                out.append(f"<details{' open' if table_open else ''}><summary>Table</summary>{table_html(tbl, tips=tips)}</details>")
        elif table is not None:
            out.append(table_html(table, tips=tips))
        if note:
            out.append(f'<p class="sub">{_e(note)}</p>')
        out.append("</div>")
        self.parts.append("".join(out))
        return self

    def raw(self, html_text: str) -> "Report":
        self.parts.append(html_text)
        return self

    def episodes(self, episodes: Sequence[Episode], *, hide_ground_truth: bool = False, max_episodes: int = 300,
                 title: str = "Episodes") -> "Report":
        eps = list(episodes)[:max_episodes]
        labels = mechanism_labels(eps)
        mech_of = {id(e): labels[(e.mechanism, config_key(e))] for e in eps}
        mechs = sorted(set(mech_of.values()))
        profs = sorted({e.profile for e in eps})
        opts = lambda xs: "".join(f'<option value="{_e(x)}">{_e(x)}</option>' for x in xs)  # noqa: E731
        head = (f'<h2>{_e(title)}</h2><div class="filters">'
                f'<select id="f-mech" aria-label="mechanism"><option value="">All mechanisms</option>{opts(mechs)}</select>'
                f'<select id="f-prof" aria-label="profile"><option value="">All profiles</option>{opts(profs)}</select>'
                f'<input id="f-q" type="search" placeholder="Search" aria-label="search">'
                f'<span id="f-n" class="muted" style="align-self:center"></span></div>')
        cards = []
        for ep in eps:
            rewards = " ".join(f'<span class="chip">{_e(r)} {_num(v)}</span>' for r, v in ep.rewards.items())
            dec = f'<span class="chip">→ {_e(ep.outcome.decision)}</span>' if ep.outcome.decision else ""
            summary = (f'<summary><b>{_e(mech_of[id(ep)])}</b> <span class="muted">{_e(ep.profile)} · {_e(ep.item_id)}</span> '
                       f"{dec} {rewards}</summary>")
            turns = []
            for t in ep.turns:
                raw = t.shown or t.text
                if t.probs and raw.strip().startswith("{") and raw.strip().endswith("}"):
                    raw = ""  # the text is just the distribution; show it once, formatted
                body = render_text(raw) if raw else ""
                if t.probs:
                    body += ('<div class="muted">' + ", ".join(f"{_e(k)} {v:.2f}" for k, v in t.probs.items()) + "</div>")
                elif t.choice and not body:
                    body = _e(t.choice)
                elif t.score is not None and not body:
                    body = _num(t.score)
                vis = "" if t.visible_to is None else f' <span class="muted">(visible to {_e(", ".join(t.visible_to))})</span>'
                cot = (f'<details><summary>private reasoning</summary><div class="txt">{_e(t.reasoning)}</div></details>'
                       if t.reasoning else "")
                turns.append(f'<div class="turn"><div class="who">{_e(t.role)}{" · " + _e(t.phase) if t.phase else ""}{vis}</div>'
                             f'<div class="txt">{body}</div>{cot}</div>')
            gt = ""
            if not hide_ground_truth and ep.ground_truth:
                gt = f'<div class="gt"><b>Ground truth</b> {gt_chips(ep.ground_truth)}</div>'
            err = f'<div class="gt" style="color:var(--bad)">error: {_e(ep.error.splitlines()[-1])}</div>' if ep.error else ""
            cards.append(f'<details class="ep" data-mech="{_e(mech_of[id(ep)])}" data-prof="{_e(ep.profile)}">{summary}'
                         f'{"".join(turns)}{gt}{err}</details>')
        more = (f'<p class="muted">Showing the first {max_episodes} of {len(episodes)} episodes.</p>'
                if len(episodes) > max_episodes else "")
        self.parts.append(head + "".join(cards) + more)
        return self

    def html(self) -> str:
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        sub = _e(self.subtitle) + (" · " if self.subtitle else "") + f"generated {now}"
        return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(self.title)}</title>'
                f"<style>{CSS}</style></head><body><div class=\"so-root\">"
                f'<button class="theme" id="theme" title="Switch light/dark">◐</button>'
                f"<h1>{_e(self.title)}</h1><div class=\"sub\">{sub}</div>{''.join(self.parts)}</div>"
                f"<script>{JS}</script></body></html>")

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.html())
        return path


def build_report(episodes: Sequence[Episode], path: str | Path, *, title: str = "OversightArena report",
                 subtitle: str = "", hide_ground_truth: bool = False, extra: Sequence[tuple[str, Any]] = (),
                 max_episodes: int = 300) -> Path:
    """Standard report: KPIs, ASD chart + metric table (if ground truth is known), extra sections, episodes.

    ``extra`` is a sequence of ``(title, charts_or_(charts, table, note))`` entries.
    """
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd, asd_by_transform, incentive_alignment, judge_accuracy, with_parse_status
    from so_arena.analysis.plots import asd_bars, dual_mode

    eps = list(episodes)
    ok = [e for e in eps if e.error is None]
    rep = Report(title, subtitle)
    cost = sum(e.total_usage.cost_usd for e in eps)
    labels = mechanism_labels(eps)
    # released episodes carry no usage (token counts can measure a directive): no cost figure then, not 0.00
    has_usage = any(e.usage for e in eps)
    rep.kpis({"Episodes": len(eps), "Items": len({e.item_id for e in eps}),
              "Mechanisms": len(set(labels.values())), **({"Cost (USD)": f"{cost:,.2f}"} if has_usage else {}),
              **({"Errors": sum(e.error is not None for e in eps)} if any(e.error for e in eps) else {})})
    df = with_parse_status(role_frame(ok), ok)
    if not hide_ground_truth and not df.empty and df["value"].notna().any():
        a = asd(df)
        order = list(dict.fromkeys(labels[(e.mechanism, config_key(e))] for e in ok))
        if not a.empty:
            a = a.assign(_o=a["mechanism"].map({m: i for i, m in enumerate(order)})).sort_values("_o").drop(columns="_o")
            charts = dual_mode(asd_bars, a, subtitle="Paired by item; whiskers are 95% bootstrap intervals")
            shown = ["parse_fail_rate"] if "parse_fail_rate" in a and (a["parse_fail_rate"] > 0).any() else []
            table = a[["mechanism", "asd", "ci_low", "ci_high", "n_items", *shown]].copy()
            try:
                tb = asd_by_transform(ok, ("brier",), parse_warn=None)  # warned for the log score above
                if not tb.empty:
                    table = table.merge(tb[["mechanism", "asd"]].rename(columns={"asd": "asd_brier"}), on="mechanism", how="left")
            except Exception:
                pass
            ia = incentive_alignment(df, n_boot=200)
            if not ia.empty:
                table = table.merge(ia[["mechanism", "concordance", "label_efficiency"]], on="mechanism", how="left")
            ja = judge_accuracy(df)
            if not ja.empty:
                table = table.merge(ja[["mechanism", "accuracy", "p_true"]], on="mechanism", how="left")
            rep.section("Does honesty pay?", charts=charts, table=table, info="asd")
    for title_i, content in extra:
        if isinstance(content, tuple):
            charts, table, note = (list(content) + [None, None])[:3]
            rep.section(title_i, charts=charts, table=table, note=note)
        else:
            rep.section(title_i, charts=content)
    rep.episodes(eps, hide_ground_truth=hide_ground_truth, max_episodes=max_episodes)
    return rep.write(path)
