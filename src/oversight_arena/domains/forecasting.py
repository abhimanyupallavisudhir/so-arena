"""Forecasting: delayed ground truth (question resolution).

``ManifoldForecasting`` pulls binary questions from Manifold Markets' public API: *resolved*
questions (GT available now — pick ones resolved after your models' training cutoff to avoid
leakage) or *open* ones (GT pending: run mechanisms now, release results, resolve later with
:meth:`resolve` / :func:`oversight_arena.release.resolve_release`). ``FileForecasting`` loads
your own JSON/CSV questions. Options are ``YES`` / ``NO``; ``gt['outcome']`` ∈ {0, 1}.
"""

from __future__ import annotations

import csv
import json
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

from ..core.episode import EpisodeRecord
from ..core.task import Answer, InfoBlock, Task
from ..ground_truth.base import OUTCOME, GTScorer
from .base import Domain

MANIFOLD = "https://api.manifold.markets/v0"


def _iso(ms: float | None) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def forecast_task(qid: str, question: str, outcome: float | None, *, description: str = "", close: str | None = None,
                  resolved_at: str | None = None, market_p: float | None = None, url: str | None = None,
                  source: str = "custom", show_market: bool = False) -> Task:
    pending = outcome is None
    info = []
    if description:
        info.append(InfoBlock(key="background", title="Background", content=description[:4000]))
    meta = f"Question closes: {close or 'unknown'}."
    if show_market and market_p is not None:
        meta += f" Current market probability: {market_p:.2f}."
    info.append(InfoBlock(key="meta", title="Details", content=meta))
    opts = [
        Answer(id="YES", text="Resolves YES", value=None if pending else (1.0 if outcome else -1.0)),
        Answer(id="NO", text="Resolves NO", value=None if pending else (-1.0 if outcome else 1.0)),
    ]
    gt: dict[str, Any] = {"pending": True} if pending else {"outcome": float(outcome)}  # type: ignore[arg-type]
    return Task(
        id=f"{source}-{qid}", domain="forecasting", question=question, options=opts, info=info,
        answer_type="probability", gt=gt,
        metadata={"close": close, "resolved_at": resolved_at, "market_p": market_p, "url": url, "source": source, "qid": qid},
    )


class ManifoldForecasting(Domain):
    name: ClassVar[str] = "forecasting"
    status: Literal["resolved", "open"] = "resolved"
    resolved_after: str | None = None  # ISO date; e.g. your model's knowledge cutoff
    term: str = ""
    n: int = 100
    min_volume: float = 500.0
    show_market: bool = False
    expert_clearance: list[str] = []
    judge_clearance: list[str] = []

    def _search(self) -> list[dict]:
        from ..data import get_json

        out: list[dict] = []
        offset = 0
        while len(out) < self.n and offset < 2000:
            q = urllib.parse.urlencode({
                "term": self.term, "filter": self.status, "contractType": "BINARY", "limit": 100,
                "offset": offset, "sort": "most-popular",
            })
            page = get_json(f"{MANIFOLD}/search-markets?{q}")
            if not page:
                break
            for m in page:
                if m.get("outcomeType") != "BINARY" or m.get("volume", 0) < self.min_volume:
                    continue
                if self.status == "resolved":
                    if m.get("resolution") not in ("YES", "NO"):
                        continue
                    if self.resolved_after and (_iso(m.get("resolutionTime")) or "") < self.resolved_after:
                        continue
                out.append(m)
            offset += 100
        return out[: self.n]

    def load(self) -> list[Task]:
        from ..data import data_dir

        cache = data_dir() / f"manifold_{self.status}_{self.term}_{self.resolved_after}_{self.n}_{int(self.min_volume)}.json"
        if cache.exists():
            ms = json.loads(cache.read_text())
        else:
            ms = self._search()
            cache.write_text(json.dumps(ms))
        tasks = []
        for m in ms:
            res = m.get("resolution")
            outcome = None if self.status == "open" or res not in ("YES", "NO") else (1.0 if res == "YES" else 0.0)
            tasks.append(forecast_task(
                m["id"], m["question"], outcome, description=m.get("textDescription", "") or "",
                close=_iso(m.get("closeTime")), resolved_at=_iso(m.get("resolutionTime")),
                market_p=m.get("probability"), url=m.get("url"), source="manifold", show_market=self.show_market,
            ))
        return tasks

    @staticmethod
    def resolve(tasks: list[Task]) -> list[Task]:
        """Re-fetch markets and return updated tasks (resolved ones get GT)."""
        from ..data import get_json

        out = []
        for t in tasks:
            if t.metadata.get("source") != "manifold":
                out.append(t)
                continue
            m = get_json(f"{MANIFOLD}/market/{t.metadata['qid']}")
            res = m.get("resolution") if m.get("isResolved") else None
            outcome = None if res not in ("YES", "NO") else (1.0 if res == "YES" else 0.0)
            nt = forecast_task(t.metadata["qid"], t.question, outcome, close=t.metadata.get("close"),
                               resolved_at=_iso(m.get("resolutionTime")), market_p=m.get("probability"),
                               url=t.metadata.get("url"), source="manifold")
            nt = nt.model_copy(update={"info": t.info})
            out.append(nt)
        return out


class FileForecasting(Domain):
    """Questions from a JSON list / CSV with columns id, question, [outcome], [description], [close]."""

    name: ClassVar[str] = "forecasting"
    path: str

    def load(self) -> list[Task]:
        p = Path(self.path)
        rows: list[dict] = json.loads(p.read_text()) if p.suffix == ".json" else list(csv.DictReader(p.open()))
        tasks = []
        for r in rows:
            o = r.get("outcome")
            outcome = None if o in (None, "", "null") else float(o)
            tasks.append(forecast_task(str(r["id"]), r["question"], outcome, description=r.get("description", ""),
                                       close=r.get("close"), source="file"))
        return tasks


class ForecastScore(GTScorer):
    """Per-forecaster proper scores vs the resolution (log and Brier) + the principal's log score."""

    name: str = "forecast_log"
    rule: Literal["log", "brier"] = "log"

    def score(self, task: Task, record: EpisodeRecord) -> dict[str, float | None]:
        from ..mechanisms.forecasting import proper_score

        y = task.gt.get("outcome")
        if y is None:
            return {}
        out: dict[str, float | None] = {}
        for r, p in (record.outcome.get("forecasts") or {}).items():
            if p is not None:
                out[r] = proper_score(float(p), float(y), self.rule)
        probs = record.outcome.get("probs")
        if probs and "YES" in probs:
            out[OUTCOME] = proper_score(float(probs["YES"]), float(y), self.rule)
        return out


def default_forecast_gt() -> list[GTScorer]:
    from ..ground_truth.common import DecisionCorrect, JudgeProbCorrect, TargetCorrect

    return [ForecastScore(), ForecastScore(name="forecast_brier", rule="brier"), TargetCorrect(), DecisionCorrect(), JudgeProbCorrect()]


ManifoldForecasting.gt_scorers = lambda self: default_forecast_gt()  # type: ignore[method-assign]
FileForecasting.gt_scorers = lambda self: default_forecast_gt()  # type: ignore[method-assign]
