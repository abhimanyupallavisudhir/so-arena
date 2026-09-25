"""Releasing mechanism results before ground truth exists, and resolving them later.

Some ground truth arrives late (forecasting questions resolve months later) or never (open
research questions judged in public). A *release* publishes what the mechanisms concluded - each
item's outcome and every behaviour's rewards, i.e. what scored highly and what scored lowly - with
no claim about which was right. A SHA-256 manifest commits to the exact contents: publish the
digest (or timestamp it, e.g. with OpenTimestamps) and anyone can later verify the results were
not edited after the truth came out.

``resolve`` takes the release plus ground truth when it arrives, verifies the manifest, fills in
deferred rewards (e.g. market scoring rules), scores every episode, and writes metrics and a report.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from so_arena.core.ground_truth import GroundTruthScorer, default_scorers
from so_arena.core.items import GroundTruth, TaskItem
from so_arena.core.mechanism import Episode
from so_arena.core.rewards import RewardRule
from so_arena.core.runner import run_sync, score_episode
from so_arena.core.store import RunStore

FILES = ("items.jsonl", "episodes.jsonl", "rankings.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip(ep: Episode) -> Episode:
    e = ep.model_copy(deep=True)
    e.ground_truth = {}
    e.gt_status = "pending" if ep.gt_status in ("pending", "unscored") else "unknown"
    return e


def rankings(episodes: Sequence[Episode]) -> dict[str, Any]:
    """Mechanism-only summaries: per item outcomes, and behaviours ranked by mean reward."""
    per_item: dict[str, dict[str, Any]] = {}
    for ep in episodes:
        if ep.error:
            continue
        d = per_item.setdefault(ep.item_id, {})
        d.setdefault(ep.mechanism, []).append({
            "profile": ep.profile, "decision": ep.outcome.decision, "probs": ep.outcome.probs,
            "rewards": ep.rewards, "labels": {r: p.label for r, p in ep.players.items()},
        })
    rows = []
    for ep in episodes:
        if ep.error:
            continue
        for r, v in ep.rewards.items():
            if v is None:
                continue
            p = ep.players.get(r)
            rows.append({"mechanism": ep.mechanism, "role": r, "behaviour": p.label if p else r, "reward": v})
    board = []
    if rows:
        df = pd.DataFrame(rows)
        g = df.groupby(["mechanism", "role", "behaviour"])["reward"].agg(["mean", "count", "std"]).reset_index()
        g = g.sort_values(["mechanism", "role", "mean"], ascending=[True, True, False])
        board = [{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in g.to_dict("records")]
    return {"per_item": per_item, "leaderboard": board}


class Manifest(BaseModel):
    title: str
    created_at: str
    files: dict[str, str]
    digest: str
    n_items: int
    n_episodes: int
    notes: str = ""
    mechanisms: list[str] = Field(default_factory=list)


def _digest(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def release(episodes: Sequence[Episode], items: Sequence[TaskItem], out_dir: str | Path, *, title: str = "Release",
            notes: str = "", html: bool = True) -> Manifest:
    """Write a ground-truth-free release bundle and return its manifest (``manifest.digest`` is the commitment)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    eps = [_strip(e) for e in episodes]
    with open(out / "items.jsonl", "w") as f:
        for it in items:
            f.write(it.censored().model_dump_json() + "\n")
    with open(out / "episodes.jsonl", "w") as f:
        for e in eps:
            f.write(e.model_dump_json() + "\n")
    (out / "rankings.json").write_text(json.dumps(rankings(eps), indent=1, default=str))
    files = {name: _sha(out / name) for name in FILES}
    man = Manifest(title=title, created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(), files=files,
                   digest=_digest(files), n_items=len(items), n_episodes=len(eps), notes=notes,
                   mechanisms=sorted({e.mechanism for e in eps}))
    (out / "MANIFEST.json").write_text(man.model_dump_json(indent=2))
    if html:
        from so_arena.analysis.report import build_report

        build_report(eps, out / "index.html", title=title, subtitle=f"release {man.digest[:12]} · ground truth not included",
                     hide_ground_truth=True)
    return man


def release_run(run_dir: str | Path, out_dir: str | Path, **kwargs: Any) -> Manifest:
    store = RunStore(run_dir)
    return release(store.episodes(), store.items(), out_dir, **kwargs)


def verify(release_dir: str | Path) -> bool:
    """True iff the release files match the manifest (the published digest commits to them)."""
    d = Path(release_dir)
    man = Manifest.model_validate_json((d / "MANIFEST.json").read_text())
    files = {name: _sha(d / name) for name in FILES}
    return files == man.files and _digest(files) == man.digest


class Resolution(BaseModel):
    release_digest: str
    resolved_at: str
    n_resolved: int
    n_unresolved: int
    metrics: dict[str, Any] = Field(default_factory=dict)


Truth = dict[str, Any] | Callable[[list[TaskItem]], dict[str, Any]]


def _apply_truth(item: TaskItem, t: Any) -> TaskItem:
    if t is None:
        return item
    if isinstance(t, GroundTruth):
        return item.model_copy(update={"ground_truth": t})
    if isinstance(t, str):
        answers = [a.model_copy(update={"value": 1.0 if a.label == t else -1.0}) for a in item.answers or []]
        return item.model_copy(update={"ground_truth": GroundTruth(status="known", correct=t), "answers": answers or item.answers})
    if isinstance(t, dict):
        answers = [a.model_copy(update={"value": float(t[a.label])}) for a in item.answers or [] if a.label in t]
        return item.model_copy(update={"ground_truth": GroundTruth(status="known", values={k: float(v) for k, v in t.items()}),
                                       "answers": answers or item.answers})
    raise TypeError(f"unsupported ground truth {t!r}")


def resolve(release_dir: str | Path, truth: Truth, *, out_dir: str | Path | None = None,
            ground_truth: Sequence[GroundTruthScorer] | None = None, reward_rule: RewardRule | None = None,
            require_valid: bool = True, html: bool = True) -> Resolution:
    """Resolve a release with ground truth (labels, value dicts, GroundTruth objects, or a callable).

    Deferred rewards are recomputed with ``reward_rule`` (defaults to each episode's stored rewards
    unless ``outcome.data`` needs a resolution, in which case pass e.g. ``MarketScoringReward()``).
    """
    d = Path(release_dir)
    if require_valid and not verify(d):
        raise ValueError("release files do not match MANIFEST.json - refusing to resolve an altered release")
    man = Manifest.model_validate_json((d / "MANIFEST.json").read_text())
    items = [TaskItem.model_validate_json(x) for x in (d / "items.jsonl").read_text().splitlines() if x.strip()]
    eps = [Episode.model_validate_json(x) for x in (d / "episodes.jsonl").read_text().splitlines() if x.strip()]
    truths = truth(items) if callable(truth) else truth
    resolved_items = {it.id: _apply_truth(it, truths.get(it.id)) for it in items}
    scorers = list(ground_truth) if ground_truth is not None else default_scorers()

    async def go() -> list[Episode]:
        out = []
        for ep in eps:
            it = resolved_items.get(ep.item_id)
            t = truths.get(ep.item_id)
            if it is None or t is None:
                out.append(ep)
                continue
            e = ep.model_copy(deep=True)
            if isinstance(t, str):
                e.outcome.data["resolution"] = t
            if reward_rule is not None:
                e.rewards = await reward_rule.acompute(e, None)
                e.reward_status = "pending" if any(v is None for r, v in e.rewards.items() if r in e.trainable_roles) else "final"
            e.gt_status = "unscored"
            out.append(await score_episode(e, it, scorers))
        return out

    scored = run_sync(go())
    n_res = sum(1 for e in scored if e.gt_status == "known")
    metrics: dict[str, Any] = {}
    from so_arena.analysis.frames import role_frame
    from so_arena.analysis.metrics import asd, incentive_alignment, judge_accuracy

    df = role_frame([e for e in scored if e.gt_status == "known"])
    if not df.empty:
        for name, fn in (("asd", asd), ("alignment", incentive_alignment), ("judge_accuracy", judge_accuracy)):
            try:
                t = fn(df)
                metrics[name] = json.loads(t.to_json(orient="records"))
            except Exception as e:  # metrics that need arms that are absent are skipped
                metrics[name] = {"error": repr(e)}
    res = Resolution(release_digest=man.digest, resolved_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
                     n_resolved=n_res, n_unresolved=len(scored) - n_res, metrics=metrics)
    out = Path(out_dir) if out_dir is not None else d / "resolved"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "episodes.jsonl", "w") as f:
        for e in scored:
            f.write(e.model_dump_json() + "\n")
    (out / "resolution.json").write_text(res.model_dump_json(indent=2))
    if html:
        from so_arena.analysis.report import build_report

        build_report(scored, out / "report.html", title=f"{man.title} (resolved)",
                     subtitle=f"release {man.digest[:12]} · {n_res} episodes resolved")
    return res
