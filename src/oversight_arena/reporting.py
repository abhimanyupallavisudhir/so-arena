"""Portable reports; absent labels remain absent."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from pathlib import Path

from .core import Episode
from .optimization import OptimizationPoint
from .storage import public_result


def write_report(
    records: Sequence[Episode], path: str | Path, *, title: str = "OversightArena results"
) -> Path:
    rows = []
    for record in records:
        public = public_result(record)
        for role, reward in public["rewards"].items():
            evaluations = [
                e
                for e in public["evaluations"]
                if e["status"] == "resolved" and role in e["per_agent"]
            ]
            quality = (
                "; ".join(
                    f"{e['scorer']}: {json.dumps(e['per_agent'][role], sort_keys=True)}"
                    for e in evaluations
                )
                or "Unresolved / unavailable"
            )
            transcript = html.escape(json.dumps(public["transcript"], indent=2, ensure_ascii=False))
            rows.append(
                f"<tr><td>{html.escape(record.task.id)}</td>"
                f"<td>{html.escape(record.mechanism)}</td><td>{html.escape(role)}</td>"
                f"<td>{reward:.4g}</td><td>{html.escape(quality)}</td>"
                f"<td><details><summary>View</summary><pre>{transcript}</pre>"
                f"<small>{record.id}</small></details></td></tr>"
            )
    content = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>"""
        + html.escape(title)
        + """</title><style>
:root{font:15px system-ui;color:#172b38;background:#f7f9fb}body{max-width:1200px;margin:3rem auto;padding:0 1rem}
h1{font-size:1.6rem}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:.7rem;border-bottom:1px solid #dce4eb;vertical-align:top}
th{font-size:.85rem;color:#415567}pre{white-space:pre-wrap;max-width:40rem;font-size:.8rem}small{overflow-wrap:anywhere}input{padding:.6rem;margin:1rem 0;width:18rem;max-width:90%}summary{cursor:pointer}p{color:#415567}
</style><h1>"""
        + html.escape(title)
        + """</h1>
<p>Mechanism rewards and independent evaluations. Unresolved results make no claim about correctness.</p>
<label>Filter <input id="filter" type="search" placeholder="Task, mechanism or role"></label>
<table><thead><tr><th>Task</th><th>Mechanism</th><th>Agent</th><th>Reward</th><th>Evaluation</th><th>Transcript</th></tr></thead><tbody>
"""
        + "\n".join(rows)
        + """</tbody></table><script>
document.getElementById('filter').addEventListener('input', e => {
const q=e.target.value.toLowerCase(); document.querySelectorAll('tbody tr').forEach(r=>{
r.hidden=!Array.from(r.cells).slice(0,3).some(c=>c.textContent.toLowerCase().includes(q)); });});
</script></html>"""
    )
    path = Path(path)
    path.write_text(content)
    return path


def plot_optimization(
    series: dict[str, Sequence[OptimizationPoint]], role: str, path: str | Path
) -> Path:
    """Standalone parametric reward/quality figure with optimization pressure labels."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("Install oversight-arena[plots] for figures") from error
    figure, axis = plt.subplots(figsize=(7, 5), layout="constrained")
    plotted = False
    try:
        for label, points in series.items():
            valid = [p for p in points if p.truth.get(role) is not None]
            if not valid:
                continue
            plotted = True
            x = [p.rewards[role] for p in valid]
            y = [p.truth[role] for p in valid]
            axis.plot(x, y, "o-", label=label)
            for point, a, b in zip(valid, x, y, strict=False):
                axis.annotate(
                    ", ".join(f"{r}={n}" for r, n in point.pressure.items()),
                    (a, b),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )
        if not plotted:
            raise ValueError("No resolved quality values to plot; publish a reward report instead")
        axis.set(xlabel=f"Mechanism reward · {role}", ylabel=f"Independent quality · {role}")
        axis.legend()
        axis.grid(alpha=0.2)
        figure.savefig(path)
    finally:
        plt.close(figure)
    return Path(path)
