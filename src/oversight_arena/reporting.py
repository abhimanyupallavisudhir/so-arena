"""Small static reports: mechanism rankings and separately labelled evaluation evidence."""

from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict
from pathlib import Path

from .types import Evaluation, Run


def write_report(
    runs: list[Run],
    path: str | Path,
    *,
    evaluations: list[Evaluation] | None = None,
    title: str = "OversightArena",
    note: str = "",
) -> Path:
    """No implicit evaluator revision selection. Multiple measurements remain distinct."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels: dict[str, list[Evaluation]] = {}
    for evaluation in evaluations or []:
        labels.setdefault(evaluation.run_id, []).append(evaluation)
    rows = []
    entries = [(run, role) for run in runs for role in run.roles if role.trainable]
    entries.sort(
        key=lambda entry: (
            entry[0].mechanism,
            entry[0].task.id,
            entry[1].id,
            -entry[0].reward(entry[1].id) if entry[0].status == "complete" else float("inf"),
            entry[0].id,
        )
    )
    for run, role in entries:
        reward = f"{run.reward(role.id):.4f}" if run.status == "complete" else "Failed"
        measurements = []
        for ev in labels.get(run.id, []):
            for name, m in ev.scores.get(role.id, {}).items():
                value = f"{m.value:.4f}" if m.value is not None else m.status.title()
                measurements.append(f"{name}: {value} · {m.kind} · {ev.scorer}@{ev.version}")
        quality = "; ".join(measurements) or "Not evaluated"
        transcript = json.dumps([asdict(e) for e in run.events if e.audience is None], indent=2)
        content = f"<details><summary>Trace</summary><pre>{html.escape(transcript)}</pre></details>"
        cells = [run.mechanism, run.task.id, role.id, reward, quality]
        rows.append(
            f'<tr data-run-id="{html.escape(run.id)}">'
            + "".join(f"<td>{html.escape(c)}</td>" for c in cells)
            + f"<td>{content}</td></tr>"
        )
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root{{font-family:system-ui,sans-serif;color:#152c39;background:#f4f7f8}}
body{{max-width:1200px;margin:48px auto;padding:0 24px}}h1{{font-size:32px;letter-spacing:-1px}}
.label{{text-transform:uppercase;letter-spacing:2px;font-size:12px;color:#446674}}
p{{color:#48606b;max-width:820px;line-height:1.5}}.table{{overflow:auto;background:white;border-radius:12px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th,td{{padding:14px;text-align:left;border-bottom:1px solid #e4ebee}}
th{{color:#446674;font-weight:600}}summary{{cursor:pointer;color:#007c78}}
pre{{white-space:pre-wrap;max-width:500px;max-height:320px;overflow:auto}}
</style><p class="label">OversightArena · Experiment results</p><h1>{html.escape(title)}</h1>
<p>{html.escape(note)}</p><p>{len(runs)} runs · {sum(r.status == "failed" for r in runs)} failed</p>
<div class="table"><table><thead><tr><th>Mechanism</th><th>Task</th><th>Agent</th>
<th title="Highest first within each mechanism, task and agent">Reward ↓</th>
<th>Independent evaluation</th><th></th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div></html>"""
    path.write_text(body)
    return path


def plot_curves(
    curves: dict[str, list[dict]], path: str | Path, *, title: str = "Optimization paths"
):
    """Publication/export plot and CSV; missing quality never becomes a zero on an axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    rows = []
    for name, curve in curves.items():
        observed = [p for p in curve if p["quality"] is not None]
        if observed:
            ax.plot(
                [p["reward"] for p in observed],
                [p["quality"] for p in observed],
                marker="o",
                label=name,
            )
            for index, p in enumerate(observed):
                near_previous = (
                    index > 0 and abs(p["quality"] - observed[index - 1]["quality"]) < 0.02
                )
                ax.annotate(
                    f"{p['budget']:g}",
                    (p["reward"], p["quality"]),
                    xytext=(-12, -18) if near_previous else (5, 6),
                    textcoords="offset points",
                    fontsize=8,
                )
        rows.extend({"series": name, **point} for point in curve)
    ax.set(
        xlabel="Expected mechanism reward",
        ylabel="Expected independent quality",
        title=title + "\nLabels: best-of-N budget",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.15)
    if ax.lines:
        ax.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    if rows:
        with path.with_suffix(".csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
