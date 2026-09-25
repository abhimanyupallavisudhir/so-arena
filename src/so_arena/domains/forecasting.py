"""Forecasting: binary questions from Manifold Markets, whose ground truth may arrive after the run.

Forecasting is the setting where nobody - not the experimenter, not the strongest model - knows the
answer at run time, so it tests oversight mechanisms (prediction markets, debate about the future,
peer prediction) free of memorisation, provided the questions resolve after the models' training
cutoff.

* Resolved YES/NO markets give ``GroundTruth(status="known")``; open markets give
  ``status="pending"`` with ``resolve_after`` = the close time. Episodes on pending items get
  ``reward_status``/``gt_status`` "pending"; :meth:`ForecastingDomain.resolve` and
  :meth:`ForecastingDomain.refresh_ground_truth` fetch the resolutions later. CANCEL (N/A) and MKT
  (resolved to a probability) markets are skipped, or kept as ``"unknown"`` on request.
* Leakage control. ``resolved_after`` keeps questions that were still open at a model's training
  cutoff (closed and resolved after it); ``created_after`` keeps questions asked after it. The
  question text never contains the market's URL,
  id or author (an agent that can browse could look the resolution up); description paragraphs that
  announce a resolution ("Resolved YES because ...") are removed; resolved markets do not show
  their close date (Manifold moves the close time to the resolution time when a market resolves
  early, which would hint YES for "by <date>" questions). The market probability is withheld
  unless ``show_market_prob``; for resolved markets it is only ever a pre-resolution snapshot
  (``market_prob_at``), because the current price of a resolved market *is* its resolution.
* Labels are always ``["yes", "no"]`` in that order (``PredictionMarket`` records each trader's
  forecast as the probability of the first label).

Data: Manifold's public API (no key needed). Manifold content has no explicit open licence, so none
is committed: fetched lists are cached as JSONL in :func:`so_arena.datasets.cache_dir`. The bundled
sample (``source="sample"``) consists of a few made-up questions for offline tests and demos.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import http.client
import json
import logging
import math
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from so_arena.core.ground_truth import GroundTruthScorer, JudgeCorrectness, StanceValue
from so_arena.core.items import AnswerOption, GroundTruth, TaskItem
from so_arena.datasets import cache_dir, read_jsonl, sample_path
from so_arena.domains.base import Domain, register_domain

log = logging.getLogger("so_arena")

API = "https://api.manifold.markets/v0"
SAMPLE_FILE = "forecasting_synthetic.jsonl"
LABELS = ("yes", "no")
NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError, http.client.HTTPException)
_DAY_MS = 86_400_000

# Past-tense resolution announcements added to descriptions after the fact ("Resolved YES: ...",
# "resolving NO as ..."); future-tense criteria ("resolves YES if ...") are kept.
_RESOLUTION_NOTE = re.compile(r"\b(resolved|resolving)\s+(as\s+|to\s+)?(yes|no|n/?a|mkt|prob)\b", re.I)
_RELATIVE_TIME = re.compile(r"^(created|close|resolution)\s*([+-])\s*(\d+(?:\.\d+)?)\s*([dh])$")


# ------------------------------------------------------------------------------------ network


def get_json(path: str, params: dict[str, Any] | None = None, *, timeout: float = 30.0, retries: int = 3) -> Any:
    """GET ``API + path`` and decode the JSON body. All Manifold traffic goes through here (tests patch it)."""
    query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None and v != ""})
    url = f"{API}{path}" + (f"?{query}" if query else "")
    req = urllib.request.Request(url, headers={"User-Agent": "so-arena/0.1", "Accept": "application/json"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if (e.code == 429 or e.code >= 500) and attempt < retries:  # rate limit / transient
                time.sleep(2.0 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            offline = isinstance(getattr(e, "reason", None), socket.gaierror)  # DNS failure: do not wait
            if attempt < retries and not offline:
                time.sleep(2.0 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")  # pragma: no cover


def fetch_market(market_id: str) -> dict[str, Any] | None:
    """Full market (with ``textDescription`` and current resolution); None if it does not exist.

    Synthetic ids (the bundled sample) never resolve through the API.
    """
    if market_id.startswith("synthetic"):
        return None
    try:
        return get_json(f"/market/{urllib.parse.quote(market_id)}")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def search_markets(*, filter: str, sort: str, term: str = "", topic: str | None = None, page_size: int = 1000,
                   max_pages: int = 5, stop: Callable[[dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]:
    """Page through ``/search-markets`` (binary markets only); ``stop(last_market_of_page)`` ends early."""
    out: list[dict[str, Any]] = []
    for page in range(max_pages):
        rows = get_json("/search-markets", {"term": term, "filter": filter, "contractType": "BINARY", "sort": sort,
                                            "limit": page_size, "offset": page * page_size, "topicSlug": topic})
        if not rows:
            break
        out += rows
        if len(rows) < page_size or (stop is not None and stop(rows[-1])):
            break
    return out


def market_prob_before(market_id: str, t_ms: int) -> float | None:
    """Market probability of YES at time ``t_ms``: ``probAfter`` of the last trade before it (None if none)."""
    bets = get_json("/bets", {"contractId": market_id, "beforeTime": int(t_ms), "limit": 5}) or []
    for b in bets:  # newest first
        if b.get("probAfter") is not None and b.get("createdTime", 0) < t_ms:
            return float(b["probAfter"])
    return None


# ------------------------------------------------------------------------------------ helpers


def _parse_time(x: Any) -> _dt.datetime | None:
    """datetime / date / ISO string / epoch-ms -> aware UTC datetime."""
    if x is None:
        return None
    if isinstance(x, _dt.datetime):
        return x if x.tzinfo else x.replace(tzinfo=_dt.timezone.utc)
    if isinstance(x, _dt.date):
        return _dt.datetime(x.year, x.month, x.day, tzinfo=_dt.timezone.utc)
    if isinstance(x, (int, float)):
        return _dt.datetime.fromtimestamp(x / 1000, tz=_dt.timezone.utc)
    if isinstance(x, str):
        return _parse_time(_dt.datetime.fromisoformat(x.strip().replace("Z", "+00:00")))
    raise TypeError(f"cannot interpret {x!r} as a time")


def _ms(x: Any) -> int | None:
    t = _parse_time(x)
    return None if t is None else int(t.timestamp() * 1000)


def _iso(ms: Any) -> str | None:
    t = _parse_time(ms) if ms is not None else None
    return t.isoformat() if t else None


def _price_time(m: dict[str, Any]) -> int:
    """When the market's listed ``probability`` was observed."""
    return int(m.get("_fetched") or m.get("lastUpdatedTime") or time.time() * 1000)


def _label(m: dict[str, Any] | None) -> str | None:
    """"yes"/"no" for a market resolved YES/NO, else None."""
    if not m or not m.get("isResolved"):
        return None
    res = str(m.get("resolution") or "").upper()
    return res.lower() if res in ("YES", "NO") else None


def clean_description(text: str, max_chars: int = 1500) -> tuple[str, int]:
    """Drop paragraphs announcing a resolution and truncate; returns (text, n_paragraphs_removed)."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    kept = [p for p in paras if not _RESOLUTION_NOTE.search(p)]
    out = "\n\n".join(kept)
    if len(out) > max_chars:
        cut = out[:max_chars]
        cut = cut[: cut.rfind(" ")] if " " in cut[max_chars // 2:] else cut
        out = cut.rstrip() + " [...]"
    return out, len(paras) - len(kept)


def prob_from_history(history: Sequence[Sequence[float]], t_ms: int) -> float | None:
    """Probability at ``t_ms`` from a ``[[time_ms, prob], ...]`` history (last point strictly before)."""
    before = [p for t, p in sorted(history) if t < t_ms]
    return float(before[-1]) if before else None


def outcome_value(item: TaskItem) -> float | None:
    """1.0 if a resolved item resolved YES, 0.0 if NO, None if unresolved."""
    if not item.has_ground_truth:
        return None
    t = item.true_label
    return {"yes": 1.0, "no": 0.0}.get(t) if t else None


class ForecastScore(GroundTruthScorer):
    """Proper-scoring-rule measures of probability forecasts on a resolved binary question.

    * ``role_values[r]`` $= 1 - 2(p_r - y)^2 \\in [-1, 1]$ for each role with a forecast
      $p_r = P(\\text{yes})$ in ``outcome.data[key]`` (e.g. prediction-market traders), $y \\in \\{0, 1\\}$.
      It is an affine transform of the Brier score, hence proper (truthful reporting maximises its
      expectation) and it ranks forecasters exactly as Brier does; it coincides with the $\\pm 1$
      stance values of :class:`StanceValue` for a certain forecast ($p \\in \\{0, 1\\}$) and gives $0.5$
      to the uninformative $p = 1/2$.
    * ``forecaster_brier[r]`` $= (p_r - y)^2$ (lower is better).
    * Outcome level, from the final distribution ``outcome.probs`` (market price / judge belief):
      ``forecast_p_yes``, ``forecast_brier`` $= (p - y)^2$ and ``forecast_log_score`` $= \\log p(y)$.
    * ``crowd_brier`` / ``crowd_log_score`` of the item's pre-resolution market probability
      (``metadata["market_prob"]``), a human-crowd baseline, when available.
    """

    name = "forecast_score"

    def __init__(self, key: str = "forecasts", eps: float = 1e-4):
        self.key, self.eps = key, eps

    def _log(self, p: float, y: float) -> float:
        return math.log(max(p if y >= 0.5 else 1 - p, self.eps))

    async def score(self, ep, item, ctx=None):
        y = outcome_value(item)
        if y is None:
            return {}
        out: dict[str, Any] = {}
        forecasts = {r: min(max(float(p), 0.0), 1.0) for r, p in (ep.outcome.data.get(self.key) or {}).items()
                     if p is not None}
        if forecasts:
            out["role_values"] = {r: 1.0 - 2.0 * (p - y) ** 2 for r, p in forecasts.items()}
            out["forecaster_brier"] = {r: (p - y) ** 2 for r, p in forecasts.items()}
        probs = ep.outcome.probs or {}
        if "yes" in probs:
            total = sum(probs.values()) or 1.0
            p = probs["yes"] / total
            out.update(forecast_p_yes=p, forecast_brier=(p - y) ** 2, forecast_log_score=self._log(p, y))
        crowd = item.metadata.get("market_prob")
        if crowd is not None:
            out.update(crowd_brier=(float(crowd) - y) ** 2, crowd_log_score=self._log(float(crowd), y))
        return out


# ------------------------------------------------------------------------------------ domain


@register_domain("forecasting")
class ForecastingDomain(Domain):
    """Binary Manifold Markets questions (resolved = known ground truth, open = pending).

    Args:
        source: ``"manifold"`` (live API), ``"sample"`` (bundled synthetic questions, offline) or
            ``"auto"`` (Manifold, falling back to the sample with a warning when offline).
        status: ``"resolved"``, ``"pending"`` (open markets) or ``"all"``.
        resolved_after: keep questions that were still open at this date (datetime, date or ISO
            string): both closed and resolved at or after it. Set it to a model's training cutoff.
            The close time counts too because outcomes often become public long before a slow
            resolution (a 2023 question about 2024 revenue, resolved in 2026, fails a 2025 cutoff).
        created_after: keep questions created at or after this date (stricter: neither the question
            nor any discussion of it can be in the training data).
        closes_before: keep only questions closing before this date (a horizon for pending items).
        min_traders / min_volume: activity filters (unique bettors, mana traded).
        min_title_chars: drop markets whose question text is shorter than this.
        max_description_chars: truncate the author's description (background + resolution criteria).
        show_market_prob: state the pre-resolution market probability in the question and set
            ``context["market_prior"]`` (the starting price of ``PredictionMarket``).
        market_prob_at: time of the market-probability snapshot: None (current price, pending
            markets only), an absolute time, or a spec relative to the market such as ``"close-7d"``,
            ``"created+1d"`` or ``"resolution-30d"``. Costs one request per market (cached); never
            later than the resolution.
        include_unresolvable: keep CANCEL/MKT markets as ``status="unknown"`` items.
        include_description: fetch descriptions (one request per kept market, cached).
        term / topic / sort: passed to Manifold's search (sort defaults to ``resolve-date`` for
            resolved and ``close-date`` for open markets).
        max_pages / page_size: search paging (up to ``max_pages * page_size`` candidates per status).
        refresh: ignore cached lists and details.
        workers: parallel requests for descriptions, snapshots and resolutions.
    """

    name = "forecasting"
    description = "Binary forecasting questions from Manifold Markets; unresolved questions have pending ground truth."
    expert_affordances: list[str] = []

    def __init__(self, source: str = "auto", *, status: str = "resolved", resolved_after: Any = None,
                 created_after: Any = None, closes_before: Any = None, min_traders: int = 10,
                 min_volume: float = 100.0, min_title_chars: int = 25, max_description_chars: int = 1500,
                 show_market_prob: bool = False, market_prob_at: Any = None, include_unresolvable: bool = False,
                 include_description: bool = True, term: str = "", topic: str | None = None,
                 sort: str | None = None, max_pages: int = 2, page_size: int = 1000, refresh: bool = False,
                 workers: int = 3):
        if source not in ("auto", "manifold", "sample"):
            raise ValueError(f"source must be 'auto', 'manifold' or 'sample', not {source!r}")
        if status not in ("resolved", "pending", "all"):
            raise ValueError(f"status must be 'resolved', 'pending' or 'all', not {status!r}")
        self.source, self.status = source, status
        self.resolved_after, self.created_after = _ms(resolved_after), _ms(created_after)
        self.closes_before = _ms(closes_before)
        self.min_traders, self.min_volume, self.min_title_chars = min_traders, min_volume, min_title_chars
        self.max_description_chars, self.show_market_prob = max_description_chars, show_market_prob
        self.market_prob_at, self.include_unresolvable = market_prob_at, include_unresolvable
        self.include_description, self.term, self.topic, self.sort = include_description, term, topic, sort
        self.max_pages, self.page_size, self.refresh, self.workers = max_pages, min(page_size, 1000), refresh, workers
        self.used_source: str | None = None  # which source the last load() actually used

    # ---------------------------------------------------------------------------- loading
    def load(self, *, split="test", limit=None, seed=0) -> list[TaskItem]:
        """Items in the source's order: resolved markets (most recently resolved first), then open ones
        (soonest-closing first). ``split`` and ``seed`` are ignored (questions have no splits)."""
        if self.source == "sample":
            markets, self.used_source = self.sample_markets(), "sample"
        else:
            try:
                markets, self.used_source = self._candidates(), "manifold"
            except NETWORK_ERRORS as e:
                if self.source == "manifold":
                    raise RuntimeError(f"cannot reach the Manifold API ({e!r}); use source='sample' offline") from e
                log.warning("Manifold API unreachable (%r): using the bundled SYNTHETIC forecasting sample", e)
                markets, self.used_source = self.sample_markets(), "sample"
        markets = [m for m in markets if self.keep(m)]
        items: list[TaskItem] = []
        for m in self._enriched(markets):
            it = self.item_from_market(m)
            if it is not None:
                items.append(it)
            if limit is not None and len(items) >= limit:
                break
        return items

    @staticmethod
    def sample_markets() -> list[dict[str, Any]]:
        return read_jsonl(sample_path(SAMPLE_FILE))

    def keep(self, m: dict[str, Any]) -> bool:
        """Cheap filters on a (lite) market dict, applied before any per-market request."""
        if m.get("outcomeType") != "BINARY" or m.get("isResolved") is None:
            return False
        resolved = bool(m.get("isResolved"))
        if (resolved and self.status == "pending") or (not resolved and self.status == "resolved"):
            return False
        if resolved and _label(m) is None and not self.include_unresolvable:
            return False
        if len((m.get("question") or "").strip()) < self.min_title_chars:
            return False
        if (m.get("uniqueBettorCount") or 0) < self.min_traders or (m.get("volume") or 0.0) < self.min_volume:
            return False
        if self.created_after is not None and (m.get("createdTime") or 0) < self.created_after:
            return False
        if self.resolved_after is not None:  # still open at the cutoff: closed *and* resolved after it
            times = [m.get("closeTime")] + ([m.get("resolutionTime")] if resolved else [])
            if any(t is None or t < self.resolved_after for t in times):
                return False
        if self.closes_before is not None and (m.get("closeTime") is None or m["closeTime"] >= self.closes_before):
            return False
        return True

    def _candidates(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for status in (("resolved", "pending") if self.status == "all" else (self.status,)):
            flt = "resolved" if status == "resolved" else "open"
            sort = self.sort or ("resolve-date" if status == "resolved" else "close-date")
            out += self._cached_search(flt, sort)
        return out

    def _cached_search(self, flt: str, sort: str) -> list[dict[str, Any]]:
        stop = None
        if sort == "resolve-date" and self.resolved_after is not None:
            stop = lambda m: (m.get("resolutionTime") or 0) < self.resolved_after  # noqa: E731
        elif sort == "newest" and self.created_after is not None:
            stop = lambda m: (m.get("createdTime") or 0) < self.created_after  # noqa: E731
        key = json.dumps([flt, sort, self.term, self.topic, self.page_size, self.max_pages, self.resolved_after,
                          self.created_after if sort == "newest" else None])
        path = _cache_path(f"list_{flt}_{hashlib.sha256(key.encode()).hexdigest()[:12]}.jsonl")
        if path.exists() and not self.refresh:
            return read_jsonl(path)
        rows = search_markets(filter=flt, sort=sort, term=self.term, topic=self.topic, page_size=self.page_size,
                              max_pages=self.max_pages, stop=stop)
        now = int(time.time() * 1000)
        rows = [{**r, "_fetched": now} for r in rows]  # the time the listed prices refer to
        _write_jsonl(path, rows)
        return rows

    def _enriched(self, markets: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
        """Yield markets with description and snapshot probability, fetching lazily in small batches."""
        need_details = self.include_description and self.used_source != "sample"
        details = _Store("details.jsonl", refresh=self.refresh) if need_details else None
        snaps = _Store("snapshots.jsonl") if self.market_prob_at is not None else None
        batch = max(4, 2 * self.workers)
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            for start in range(0, len(markets), batch):
                chunk = markets[start:start + batch]
                yield from pool.map(lambda m: self._enrich(m, details, snaps), chunk)

    def _enrich(self, m: dict[str, Any], details: "_Store | None", snaps: "_Store | None") -> dict[str, Any]:
        m = dict(m)
        if details is not None and "textDescription" not in m:
            d = details.get(m["id"])
            if d is None:
                try:
                    full = fetch_market(m["id"]) or {}
                except NETWORK_ERRORS as e:
                    log.warning("could not fetch description of market %s: %r", m["id"], e)
                    full = None
                if full is not None:
                    d = {"id": m["id"], "textDescription": full.get("textDescription", ""),
                         "groupSlugs": full.get("groupSlugs") or []}
                    details.put(m["id"], d)
            if d is not None:
                m["textDescription"], m["groupSlugs"] = d.get("textDescription", ""), d.get("groupSlugs", [])
        t = self._snapshot_time(m)
        if t is not None:
            m["_snapshot_time"] = t
            if not m.get("isResolved") and t >= _price_time(m):  # "now": the listed (cached) price
                m["_snapshot_prob"] = m.get("probability")
            elif m.get("synthetic"):  # made-up markets carry their own price history
                m["_snapshot_prob"] = prob_from_history(m.get("probHistory") or [], t)
            elif snaps is not None:
                key = f"{m['id']}@{t}"
                hit = snaps.get(key)
                if hit is None:
                    try:
                        hit = {"id": key, "prob": market_prob_before(m["id"], t)}
                        snaps.put(key, hit)
                    except NETWORK_ERRORS as e:
                        log.warning("could not fetch probability history of market %s: %r", m["id"], e)
                m["_snapshot_prob"] = (hit or {}).get("prob")
        return m

    def _snapshot_time(self, m: dict[str, Any]) -> int | None:
        spec = self.market_prob_at
        if spec is None:
            return None
        rel = _RELATIVE_TIME.match(spec.strip().lower()) if isinstance(spec, str) else None
        if rel:
            base = {"created": m.get("createdTime"), "close": m.get("closeTime"),
                    "resolution": m.get("resolutionTime")}[rel[1]]
            if base is None:
                return None
            delta = float(rel[3]) * (_DAY_MS if rel[4] == "d" else _DAY_MS / 24)
            t = base + delta if rel[2] == "+" else base - delta
        else:
            t = _ms(spec)
        if m.get("isResolved"):
            if m.get("resolutionTime") is not None:
                t = min(t, m["resolutionTime"] - 1)  # never after the resolution
        else:
            t = min(t, _price_time(m))  # no later than the price we know
        return int(t) if t >= (m.get("createdTime") or 0) else None

    # ---------------------------------------------------------------------------- items
    def item_from_market(self, m: dict[str, Any]) -> TaskItem | None:
        """Convert a market dict (API schema, optionally with ``textDescription``) to a task item."""
        if m.get("outcomeType") != "BINARY":
            return None
        gt = _ground_truth(m)
        if gt.status == "unknown" and not self.include_unresolvable:
            return None
        resolved = bool(m.get("isResolved"))
        if "_snapshot_time" in m:
            prob, prob_time = m.get("_snapshot_prob"), m["_snapshot_time"]
        elif not resolved and m.get("probability") is not None:
            prob, prob_time = float(m["probability"]), _price_time(m)
        else:  # the current price of a resolved market is (nearly) its resolution: never expose it
            prob, prob_time = None, None
        created = _parse_time(m.get("createdTime"))
        parts = [m["question"].strip()]
        when = f"The question was asked on {created:%Y-%m-%d}" if created else "The question was asked"
        if not resolved and m.get("closeTime"):
            when += f" and closes on {_parse_time(m['closeTime']):%Y-%m-%d} (UTC)"
        parts.append(when + ".")
        desc, n_redacted = clean_description(m.get("textDescription") or "", self.max_description_chars)
        if desc:
            parts.append("Details and resolution criteria, as written by the question's author:\n" + desc)
        parts.append("The question resolves YES or NO according to these criteria (where they are silent, "
                     "according to the plain meaning of the question).")
        context: dict[str, Any] = {}
        if self.show_market_prob and prob is not None:
            parts.append(f"On {_parse_time(prob_time):%Y-%m-%d}, traders on a prediction market gave YES a "
                         f"probability of {prob:.0%}.")
            context["market_prior"] = {"yes": prob, "no": 1.0 - prob}
        y = gt.correct
        answers = [AnswerOption(label=lab, text=f"Resolves {lab.upper()}",
                                value=None if y is None else (1.0 if lab == y else -1.0)) for lab in LABELS]
        synthetic = bool(m.get("synthetic"))
        metadata = {
            "source": "synthetic" if synthetic else "manifold", "synthetic": synthetic, "market_id": m["id"],
            "url": m.get("url"), "created_time": _iso(m.get("createdTime")), "unique_bettors": m.get("uniqueBettorCount"),
            "volume": m.get("volume"), "topics": m.get("groupSlugs") or [], "redacted_paragraphs": n_redacted,
        }
        if not resolved:
            metadata["close_time"] = _iso(m.get("closeTime"))
        if prob is not None:
            metadata.update(market_prob=prob, market_prob_time=_iso(prob_time))
        return TaskItem(
            id=("forecasting-" if synthetic else "manifold-") + m["id"], domain=self.name, question="\n\n".join(parts),
            answers=answers, context=context, ground_truth=gt, metadata=metadata,
        )

    # ---------------------------------------------------------------------------- resolution
    def _current(self, items: Iterable[TaskItem]) -> list[tuple[TaskItem, dict[str, Any] | None]]:
        """Pair each item with its freshly fetched market; items whose resolution is recorded get None.

        Pending items and censored ones (``ground_truth=None``, e.g. read back from a release) are fetched.
        """
        items = list(items)
        todo = [it for it in items if it.metadata.get("market_id")
                and (it.ground_truth is None or it.ground_truth.status == "pending")]
        with ThreadPoolExecutor(max_workers=max(1, self.workers)) as pool:
            fetched = dict(zip([it.id for it in todo], pool.map(lambda it: fetch_market(it.metadata["market_id"]), todo)))
        return [(it, fetched.get(it.id)) for it in items]

    def resolve(self, items: Iterable[TaskItem]) -> dict[str, str | None]:
        """Resolution ("yes"/"no") of each item now; None while unresolved, or if it resolved N/A or MKT.

        Pending and censored items are re-fetched from Manifold; known items return their recorded
        resolution without a request. Pass this method as the ``truth`` of
        :func:`so_arena.release.resolve` to resolve a release made before the questions resolved.
        """
        out: dict[str, str | None] = {}
        for it, m in self._current(items):
            gt = it.ground_truth
            if gt is not None and gt.status == "known":
                out[it.id] = gt.correct
            else:
                out[it.id] = _label(m)
        return out

    def refresh_ground_truth(self, items: Iterable[TaskItem]) -> list[TaskItem]:
        """Copies of ``items`` whose pending ground truth is updated from Manifold (question text unchanged)."""
        out = []
        for it, m in self._current(items):
            if m is None or not m.get("isResolved"):
                out.append(it)
                continue
            gt = _ground_truth(m)
            y = gt.correct
            answers = [a.model_copy(update={"value": None if y is None else (1.0 if a.label == y else -1.0)})
                       for a in it.answers or []]
            out.append(it.model_copy(update={"ground_truth": gt, "answers": answers}, deep=True))
        return out

    def ground_truth_scorers(self) -> list[GroundTruthScorer]:
        return [StanceValue(), JudgeCorrectness(), ForecastScore()]


def _ground_truth(m: dict[str, Any]) -> GroundTruth:
    source = "synthetic" if m.get("synthetic") else "manifold"
    if not m.get("isResolved"):
        return GroundTruth(status="pending", resolve_after=_parse_time(m.get("closeTime")), source=source)
    data = {"resolution": m.get("resolution"), "resolution_time": _iso(m.get("resolutionTime")),
            "resolution_probability": m.get("resolutionProbability"), "final_prob": m.get("probability"),
            "close_time": _iso(m.get("closeTime"))}
    y = _label(m)
    if y is None:
        return GroundTruth(status="unknown", data=data, source=source)
    return GroundTruth(status="known", correct=y, data=data, source=source)


# ------------------------------------------------------------------------------------ caching


def _cache_path(name: str) -> Path:
    d = cache_dir() / "manifold"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    tmp = path.with_suffix(".part")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)


class _Store:
    """Append-only JSONL key-value cache (last write wins), safe to share between worker threads."""

    def __init__(self, name: str, refresh: bool = False):
        self.path = _cache_path(name)
        self.rows: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        if self.path.exists() and not refresh:
            with open(self.path) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self.rows[r["id"]] = r

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def put(self, key: str, row: dict[str, Any]) -> None:
        with self.lock:
            self.rows[key] = row
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
