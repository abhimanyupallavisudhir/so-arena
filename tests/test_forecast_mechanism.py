"""The Forecast mechanism, the synthetic forecasting world, and immediate proxies vs. proper resolution scores
(``docs/theory.md``, Proposition 7)."""

import math

import numpy as np
import pytest

import so_arena as soa
from so_arena.core.policy import FunctionPolicy
from so_arena.core.rewards import ResolutionScore, rescore
from so_arena.core.runner import Profile, run_episodes, run_sync
from so_arena.domains import get_domain
from so_arena.domains.synthetic_forecasting import STYLES, SyntheticForecasting, rating_judge, synthetic_forecaster
from so_arena.mechanisms import Forecast, rating_reward
from so_arena.release import release, resolve

AFF = {"forecaster": ["forecast_info"], "judge": ["judge_info"]}


def run(mech, items, profiles, gt=None):
    eps = run_sync(run_episodes(mech, items, profiles, ground_truth=gt))
    assert not [e.error for e in eps if e.error]
    return eps


def by_style(eps, fn):
    return {s: float(np.mean([fn(e) for e in eps if e.profile == s])) for s in STYLES}


def test_synthetic_world_is_calibrated_and_pending():
    dom = get_domain("synthetic_forecasting", n_items=3000, seed=3)
    items = dom.load()
    assert all(it.ground_truth.status == "pending" and it.ground_truth.correct is None for it in items)
    assert all(a.value is None for it in items for a in it.answers)
    truth = dom.resolve(items)
    p = np.array([it.private["forecast_info"]["p_yes"] for it in items])
    y = np.array([truth[it.id] == "yes" for it in items], dtype=float)
    for lo, hi in ((0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)):  # the information is calibrated
        m = (p >= lo) & (p < hi)
        assert y[m].mean() == pytest.approx(p[m].mean(), abs=0.04)
    known = SyntheticForecasting(n_items=3000, seed=3, resolved=True).load()
    assert [it.true_label for it in known] == [truth[it.id] for it in items]
    assert dom.resolve([soa.TaskItem(id="other", question="?")]) == {"other": None}


def test_forecast_records_forecasts_and_ratings():
    dom = SyntheticForecasting(n_items=5, seed=1)
    items = dom.load()
    seen, inner = [], rating_judge(1.0)

    async def judge(req, ctx):
        seen.append("\n".join(m.content for m in req.prompt))
        return await inner.act(req, ctx)

    mech = Forecast(n_forecasters=2, judge=True, reward=rating_reward(), affordances={"agents": ["forecast_info"]})
    players = {"forecaster_1": synthetic_forecaster("calibrated"), "forecaster_2": synthetic_forecaster("extremizing"),
               "judge": FunctionPolicy(judge)}
    eps = run(mech, items, [Profile(name="p", players=players)], gt=dom.ground_truth_scorers())
    for ep, it in zip(eps, items):
        pi = it.private["forecast_info"]["p_yes"]
        fc = ep.outcome.data["forecasts"]
        assert fc["forecaster_1"] == pytest.approx(pi)
        assert ep.outcome.probs["yes"] == pytest.approx((fc["forecaster_1"] + fc["forecaster_2"]) / 2)
        # paid the rating now: decisiveness |2q - 1|, whatever the outcome will be
        assert ep.rewards == pytest.approx({f: abs(2 * q - 1) for f, q in fc.items()}, abs=1e-3)
        assert ep.reward_status == "final" and ep.gt_status == "pending"
        assert all(t.visible_to == ["judge"] for t in ep.turns if t.phase.startswith("rating"))
    assert not any("forecast_info" in s or "p_yes" in s for s in seen)  # the judge sees forecasts, not the information
    with pytest.raises(ValueError, match="need a judge"):
        Forecast(reward=rating_reward())


def test_unparsed_forecasts_and_ratings_are_flagged():
    item = SyntheticForecasting(n_items=1).load()[0]
    mech = Forecast(judge=True, reward=rating_reward())
    garbled = FunctionPolicy(lambda req, ctx: "I'd rather not say.")
    ep = run(mech, [item], [Profile(name="p", players={"forecaster": garbled, "judge": garbled})])[0]
    assert ep.outcome.data["forecasts"] == {"forecaster": 0.5}  # the uninformative forecast
    assert ep.outcome.data["forecast_parse_ok"] == {"forecaster": False}
    assert ep.outcome.data["ratings"] == {"forecaster": 0.5} and ep.outcome.data["rating_parse_ok"] == {"forecaster": False}


def test_rating_proxy_is_not_proper_and_resolution_is():
    # Proposition 7 on the synthetic world: the immediate rating ranks distortions above the calibrated forecast,
    # the proper resolution score ranks the calibrated forecast first - in expectation exactly (latent
    # probabilities), and on realized outcomes
    dom = SyntheticForecasting(n_items=400, seed=0)
    items = dom.load()
    mech = Forecast(judge=True, reward=rating_reward(), affordances=AFF)
    eps = run(mech, items, [Profile(name=s, players={"forecaster": synthetic_forecaster(s), "judge": rating_judge(0.7)})
                            for s in STYLES])
    rating = by_style(eps, lambda e: e.rewards["forecaster"])
    assert rating["extremizing"] > rating["overconfident"] > rating["calibrated"] > rating["underconfident"]
    latent = {it.id: it.private["forecast_info"]["p_yes"] for it in items}

    def expected_log(e):
        q, p = e.outcome.data["forecasts"]["forecaster"], latent[e.item_id]
        return p * math.log(q) + (1 - p) * math.log(1 - q)

    exp = by_style(eps, expected_log)
    assert max(exp, key=exp.get) == "calibrated"
    truth = dom.resolve(items)
    for e in eps:
        e.outcome.data["resolution"] = truth[e.item_id]
    for transform in ("log", "brier"):
        realized = by_style(rescore(eps, ResolutionScore(transform)), lambda e: e.rewards["forecaster"])
        assert max(realized, key=realized.get) == "calibrated", transform


def test_release_before_resolution_then_resolve(tmp_path):
    dom = SyntheticForecasting(n_items=20, seed=2)
    items = dom.load()
    mech = Forecast(judge=True, reward=rating_reward(), affordances=AFF)
    eps = run(mech, items, [Profile(name=s, players={"forecaster": synthetic_forecaster(s), "judge": rating_judge()})
                            for s in ("calibrated", "extremizing")])
    man = release(eps, items, tmp_path / "rel", public_labels=True)
    text = (tmp_path / "rel" / "items.jsonl").read_text()
    assert "forecast_info" not in text and "judge_info" not in text  # private information is never published
    res = resolve(tmp_path / "rel", dom.resolve, reward_rule=ResolutionScore("log"),
                  ground_truth=dom.ground_truth_scorers(), digest=man.digest)
    assert res.n_resolved == len(eps)
    resolved = [soa.Episode.model_validate_json(x) for x in (tmp_path / "rel" / "resolved" / "episodes.jsonl").read_text().splitlines()]
    truth = dom.resolve(items)
    for e in resolved:
        q = e.outcome.data["forecasts"]["forecaster"]
        y = truth[e.item_id] == "yes"
        assert e.rewards["forecaster"] == pytest.approx(math.log(q if y else 1 - q))
        assert e.ground_truth["role_values"]["forecaster"] == pytest.approx(1 - 2 * (q - y) ** 2)


def test_default_reward_waits_for_resolution():
    items = SyntheticForecasting(n_items=2).load()
    ep = run(Forecast(affordances={"agents": ["forecast_info"]}), items, [Profile(name="p", players={"forecaster": synthetic_forecaster()})])[0]
    assert ep.rewards == {"forecaster": None} and ep.reward_status == "pending"
