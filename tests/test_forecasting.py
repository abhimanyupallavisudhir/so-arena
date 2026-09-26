import math
import urllib.error

import pytest

import so_arena as soa
from so_arena.core.runner import Profile, run_episodes, run_sync, score_episode
from so_arena.domains import forecasting as fc
from so_arena.domains.forecasting import ForecastingDomain, ForecastScore, clean_description
from so_arena.mechanisms import PredictionMarket


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SO_ARENA_DATA", str(tmp_path / "cache"))


def run(mech, items, profiles, ctx=None, gt=None):
    eps = run_sync(run_episodes(mech, items, profiles, ctx=ctx, ground_truth=gt))
    assert not [e.error for e in eps if e.error]
    return eps


def by_status(items):
    out = {}
    for it in items:
        out.setdefault(it.ground_truth.status, []).append(it)
    return out


# ----------------------------------------------------------------------------- bundled synthetic sample

def test_sample_known_and_pending_items():
    items = ForecastingDomain(source="sample", status="all").load()
    groups = by_status(items)
    assert len(groups["known"]) == 5 and len(groups["pending"]) == 3 and "unknown" not in groups  # CANCEL skipped
    for it in items:
        assert it.labels == ["yes", "no"] and it.metadata["synthetic"]
        assert it.metadata["market_id"] not in it.question  # no id/url that would let an agent look it up
    for it in groups["known"]:
        assert it.true_label in ("yes", "no") and it.value_of(it.true_label) == 1.0
        assert "closes on" not in it.question  # close dates of resolved markets can leak early resolution
    for it in groups["pending"]:
        assert it.ground_truth.resolve_after is not None and it.value_of("yes") is None
        assert "closes on" in it.question
    unknown = ForecastingDomain(source="sample", status="resolved", include_unresolvable=True).load()
    assert [it.ground_truth.status for it in unknown].count("unknown") == 1
    assert [it.id for it in ForecastingDomain(source="sample").load()] == [it.id for it in groups["known"]]


def test_leakage_and_activity_filters():
    ids = lambda **kw: {it.id for it in ForecastingDomain(source="sample", status="all", **kw).load()}  # noqa: E731
    everything = ids()
    assert ids(created_after="2026-01-01") == {it for it in everything if it.endswith(("06", "07", "08"))}
    # still open at the cutoff: closed *and* resolved after it (pending questions qualify)
    after = ids(resolved_after="2025-09-01")
    assert after == {f"forecasting-synthetic-0{i}" for i in (2, 4, 5, 6, 7, 8)}
    assert ids(min_traders=20) < everything and ids(min_title_chars=1000) == set()
    assert ids(closes_before="2031-07-01") == everything - {"forecasting-synthetic-06", "forecasting-synthetic-08"}


def test_market_probability_hidden_unless_requested():
    groups = by_status(ForecastingDomain(source="sample", status="all").load())
    pending, known = groups["pending"][0], groups["known"][0]
    assert pending.metadata["market_prob"] == pytest.approx(0.35)
    assert "probability" not in pending.question and "market_prior" not in pending.context
    assert "market_prob" not in known.metadata  # the price of a resolved market is its resolution
    shown = by_status(ForecastingDomain(source="sample", status="all", show_market_prob=True).load())
    assert shown["pending"][0].context["market_prior"] == pytest.approx({"yes": 0.35, "no": 0.65})
    assert "35%" in shown["pending"][0].question
    assert all("market_prior" not in it.context for it in shown["known"])
    # a pre-resolution snapshot is allowed, taken from the price history 30 days before close
    snap = by_status(ForecastingDomain(source="sample", status="all", show_market_prob=True,
                                       market_prob_at="close-30d").load())["known"]
    first = snap[0]
    assert first.metadata["market_prob"] == pytest.approx(0.58) and "58%" in first.question
    assert first.metadata["market_prob_time"] < first.ground_truth.data["resolution_time"]


def test_clean_description_removes_resolution_notes():
    text = "Resolves YES if it rains.\n\nResolved NO: it stayed dry.\n\n" + "More details. " * 200
    out, n = clean_description(text, max_chars=100)
    assert n == 1 and "Resolved NO" not in out and out.startswith("Resolves YES if it rains.")
    assert out.endswith("[...]") and len(out) <= 110


# ----------------------------------------------------------------------------- Manifold parsing (mocked network)

LITE = dict(outcomeType="BINARY", mechanism="cpmm-1", uniqueBettorCount=30, volume=5000.0, createdTime=1_750_000_000_000)
CANNED_RESOLVED = [
    dict(LITE, id="r1", question="Will the canned event number one happen before the end of the year?", isResolved=True,
         resolution="YES", closeTime=1_760_000_000_000, resolutionTime=1_760_000_000_000, probability=0.99),
    dict(LITE, id="r2", question="Will the canned event number two happen before the end of the year?", isResolved=True,
         resolution="CANCEL", closeTime=1_760_000_000_000, resolutionTime=1_760_000_000_000, probability=0.5),
    dict(LITE, id="r3", question="Will the canned event number three happen before the end of the year?", isResolved=True,
         resolution="NO", closeTime=1_760_000_000_000, resolutionTime=1_761_000_000_000, probability=0.01,
         uniqueBettorCount=2),
    dict(LITE, id="r4", question="Too short?", isResolved=True, resolution="NO", closeTime=1_760_000_000_000,
         resolutionTime=1_761_000_000_000, probability=0.01),
    dict(LITE, id="r5", outcomeType="MULTIPLE_CHOICE", question="Which of the canned multiple-choice answers will win?",
         isResolved=True, resolution="MKT", closeTime=1_760_000_000_000, resolutionTime=1_761_000_000_000),
]
CANNED_OPEN = [dict(LITE, id="o1", question="Will the canned open event happen before the year 2040 begins?",
                    isResolved=False, closeTime=2_200_000_000_000, probability=0.4)]
DETAILS = {"r1": "Resolves YES if the canned event happens.\n\nResolved YES: it happened on schedule.",
           "o1": "Resolves YES if the canned open event happens before 2040."}


def fake_api(calls):
    def get_json(path, params=None, **kw):
        calls.append((path, dict(params or {})))
        if path == "/search-markets":
            if params["offset"] > 0:
                return []
            return CANNED_RESOLVED if params["filter"] == "resolved" else CANNED_OPEN
        if path.startswith("/market/"):
            mid = path.rsplit("/", 1)[-1]
            return {"id": mid, "textDescription": DETAILS.get(mid, ""), "groupSlugs": ["canned"]}
        if path == "/bets":
            assert params["contractId"] == "r1"  # the open market's snapshot is its listed price
            assert params["beforeTime"] == 1_760_000_000_000 - 7 * 86_400_000  # "close-7d"
            return [{"probAfter": 0.7, "createdTime": params["beforeTime"] - 1000}]
        raise AssertionError(path)
    return get_json


def test_manifold_parsing_with_canned_json(monkeypatch):
    calls = []
    monkeypatch.setattr(fc, "get_json", fake_api(calls))
    dom = ForecastingDomain(source="manifold", status="all", market_prob_at="close-7d")
    items = dom.load()
    assert [it.id for it in items] == ["manifold-r1", "manifold-o1"]  # CANCEL, few traders, short, non-binary dropped
    r1, o1 = items
    assert r1.ground_truth.correct == "yes" and r1.metadata["redacted_paragraphs"] == 1
    assert "Resolved YES" not in r1.question and "Resolves YES if the canned event happens." in r1.question
    assert r1.metadata["market_prob"] == pytest.approx(0.7) and r1.ground_truth.data["final_prob"] == 0.99
    assert o1.ground_truth.status == "pending" and o1.metadata["topics"] == ["canned"]
    assert o1.metadata["market_prob"] == pytest.approx(0.4) and o1.ground_truth.resolve_after.year == 2039
    assert {p for p, _ in calls} == {"/search-markets", "/market/r1", "/market/o1", "/bets"}
    # lists, descriptions and snapshots are cached: a second load needs no network
    monkeypatch.setattr(fc, "get_json", lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    again = ForecastingDomain(source="manifold", status="all", market_prob_at="close-7d").load()
    assert [it.model_dump() for it in again] == [it.model_dump() for it in items]


def test_auto_source_falls_back_to_sample_offline(monkeypatch, caplog):
    def offline(*a, **k):
        raise urllib.error.URLError("no network")

    monkeypatch.setattr(fc, "get_json", offline)
    with pytest.raises(RuntimeError, match="source='sample'"):
        ForecastingDomain(source="manifold").load()
    dom = ForecastingDomain()  # source="auto"
    items = dom.load()
    assert dom.used_source == "sample" and items and all(it.metadata["synthetic"] for it in items)
    assert "SYNTHETIC" in caplog.text


# ----------------------------------------------------------------------------- pending -> resolved

def resolved_market(market_id, resolution="YES"):
    return {"id": market_id, "outcomeType": "BINARY", "isResolved": True, "resolution": resolution,
            "resolutionTime": 1_950_000_000_000, "closeTime": 1_950_000_000_000, "probability": 0.98}


def test_prediction_market_pending_then_resolved(monkeypatch):
    dom = ForecastingDomain(source="sample", status="pending")
    item = dom.load()[0]
    mech = PredictionMarket(n_traders=2, rounds=1)
    t1 = soa.ScriptedPolicy('{"yes": 0.7, "no": 0.3}')
    t2 = soa.ScriptedPolicy('{"yes": 0.9, "no": 0.1}')
    ep = run(mech, [item], [Profile(name="p", players={"trader_1": t1, "trader_2": t2})], ctx=dom.context(),
             gt=dom.ground_truth_scorers())[0]
    assert ep.reward_status == "pending" and ep.gt_status == "pending"
    assert ep.rewards == {"trader_1": None, "trader_2": None}
    assert ep.outcome.data["forecasts"] == {"trader_1": 0.7, "trader_2": 0.9}

    # still unresolved: synthetic markets never resolve through the (real) API
    assert dom.resolve([item]) == {item.id: None}
    monkeypatch.setattr(fc, "fetch_market", lambda mid: resolved_market(mid) if mid == item.metadata["market_id"] else None)
    resolutions = dom.resolve([item])
    assert resolutions == {item.id: "yes"}
    [updated] = dom.refresh_ground_truth([item])
    assert updated.ground_truth.status == "known" and updated.true_label == "yes" and updated.question == item.question
    assert updated.value_of("yes") == 1.0 and updated.value_of("no") == -1.0

    ep.outcome.data["resolution"] = resolutions[item.id]
    rewards = mech.reward_rule.compute(ep)
    assert rewards["trader_1"] == pytest.approx(math.log(0.7 / 0.5))
    assert rewards["trader_2"] == pytest.approx(math.log(0.9 / 0.7))
    ep.gt_status = "unscored"  # score_episode keeps a previous status, so reset it before re-scoring
    ep = run_sync(score_episode(ep, updated, dom.ground_truth_scorers()))
    assert ep.gt_status == "known"
    assert ep.ground_truth["role_values"] == pytest.approx({"trader_1": 1 - 2 * 0.3 ** 2, "trader_2": 1 - 2 * 0.1 ** 2})
    assert ep.ground_truth["forecast_brier"] == pytest.approx(0.01)
    assert ep.ground_truth["forecast_log_score"] == pytest.approx(math.log(0.9))
    assert ep.ground_truth["crowd_brier"] == pytest.approx((0.35 - 1) ** 2)  # sample price before resolution
    assert ep.ground_truth["judge_correct"] == 1.0


def test_release_then_resolve_with_domain(monkeypatch, tmp_path):
    from so_arena.mechanisms import MarketScoringReward
    from so_arena.release import release, resolve

    dom = ForecastingDomain(source="sample", status="pending")
    items = dom.load()
    players = {"trader_1": soa.ScriptedPolicy('{"yes": 0.6, "no": 0.4}'),
               "trader_2": soa.ScriptedPolicy('{"yes": 0.8, "no": 0.2}')}
    eps = run(PredictionMarket(n_traders=2, rounds=1), items, [Profile(name="p", players=players)], ctx=dom.context(),
              gt=dom.ground_truth_scorers())
    release(eps, items, tmp_path / "rel", html=False)  # released items are censored (no ground truth)
    outcomes = {"synthetic-06": "YES", "synthetic-07": "NO"}  # synthetic-08 is still open
    monkeypatch.setattr(fc, "fetch_market", lambda mid: resolved_market(mid, outcomes[mid]) if mid in outcomes else None)
    res = resolve(tmp_path / "rel", dom.resolve, reward_rule=MarketScoringReward(), ground_truth=dom.ground_truth_scorers(),
                  html=False)
    assert res.n_resolved == 2 and res.n_unresolved == 1
    lines = (tmp_path / "rel" / "resolved" / "episodes.jsonl").read_text().splitlines()
    done = {e.item_id: e for e in map(soa.Episode.model_validate_json, lines)}
    yes, no = done["forecasting-synthetic-06"], done["forecasting-synthetic-07"]
    assert yes.rewards["trader_1"] == pytest.approx(math.log(0.6 / 0.5))
    assert no.rewards["trader_2"] == pytest.approx(math.log(0.2 / 0.4))
    assert no.ground_truth["role_values"] == pytest.approx({"trader_1": 1 - 2 * 0.36, "trader_2": 1 - 2 * 0.64})
    assert done["forecasting-synthetic-08"].reward_status == "pending"


def test_resolve_known_unknown_and_cancelled(monkeypatch):
    dom = ForecastingDomain(source="sample", status="all")
    items = dom.load()
    known = [it for it in items if it.ground_truth.status == "known"]
    pending = [it for it in items if it.ground_truth.status == "pending"]
    fetched = []

    def fetch(mid):
        fetched.append(mid)
        return resolved_market(mid, "CANCEL") if mid.endswith("06") else None

    monkeypatch.setattr(fc, "fetch_market", fetch)
    res = dom.resolve(items)
    assert all(res[it.id] == it.true_label for it in known)
    assert all(res[it.id] is None for it in pending)
    assert sorted(fetched) == sorted(it.metadata["market_id"] for it in pending)  # known items need no request
    refreshed = {it.id: it for it in dom.refresh_ground_truth(pending)}
    statuses = {i[-2:]: it.ground_truth.status for i, it in refreshed.items()}
    assert statuses == {"06": "unknown", "07": "pending", "08": "pending"}


def test_forecast_score_on_known_item():
    dom = ForecastingDomain(source="sample")
    item = next(it for it in dom.load() if it.true_label == "no")
    mech = PredictionMarket(n_traders=2, rounds=1)
    players = {"trader_1": soa.ScriptedPolicy('{"yes": 0.2, "no": 0.8}'),
               "trader_2": soa.ScriptedPolicy('{"yes": 0.5, "no": 0.5}')}
    ep = run(mech, [item], [Profile(name="p", players=players)], ctx=dom.context(), gt=dom.ground_truth_scorers())[0]
    assert ep.gt_status == "known"
    assert ep.ground_truth["role_values"] == pytest.approx({"trader_1": 1 - 2 * 0.04, "trader_2": 0.5})
    assert ep.ground_truth["forecaster_brier"] == pytest.approx({"trader_1": 0.04, "trader_2": 0.25})
    # certain forecasts reproduce the +-1 stance values
    s = ForecastScore()
    ep.outcome.data["forecasts"] = {"trader_1": 0.0, "trader_2": 1.0}
    assert run_sync(s.score(ep, item))["role_values"] == {"trader_1": 1.0, "trader_2": -1.0}


# ----------------------------------------------------------------------------- live API

@pytest.mark.network
def test_live_manifold_load_and_resolve():
    dom = ForecastingDomain(source="manifold", status="all", page_size=100, max_pages=1, min_traders=5, min_volume=10)
    items = dom.load(limit=3)
    assert dom.used_source == "manifold" and 1 <= len(items) <= 3
    for it in items:
        assert it.id.startswith("manifold-") and it.labels == ["yes", "no"]
        assert it.metadata["url"] not in it.question
    res = dom.resolve(items)
    assert all(res[it.id] == it.true_label for it in items if it.ground_truth.status == "known")
