import json
import math

import pytest

import so_arena as soa
from so_arena.core.runner import Profile, run_episodes, run_sync
from so_arena.mechanisms import MarketScoringReward, PredictionMarket
from so_arena.release import release, resolve, verify


def forecasting_items():
    items = []
    for i in range(3):
        items.append(soa.TaskItem(id=f"q{i}", question=f"Will event {i} happen?",
                                  answers=[soa.AnswerOption(label="yes", text="yes"), soa.AnswerOption(label="no", text="no")],
                                  ground_truth=soa.GroundTruth(status="pending")))
    return items


def run_market(items):
    bull = soa.ScriptedPolicy('{"yes": 0.8, "no": 0.2}', label="bull")
    bear = soa.ScriptedPolicy('{"yes": 0.3, "no": 0.7}', label="bear")
    mech = PredictionMarket(n_traders=2, rounds=1)
    return run_sync(run_episodes(mech, items, [Profile(name="p", players={"trader_1": bull, "trader_2": bear})]))


def test_release_verify_resolve(tmp_path):
    items = forecasting_items()
    eps = run_market(items)
    assert all(e.reward_status == "pending" and e.gt_status == "pending" for e in eps)
    man = release(eps, items, tmp_path / "rel", title="Forecasts")
    assert verify(tmp_path / "rel")
    board = json.loads((tmp_path / "rel" / "rankings.json").read_text())
    assert set(board["per_item"]) == {"q0", "q1", "q2"}
    html = (tmp_path / "rel" / "index.html").read_text()
    assert "Ground truth" not in html  # releases show mechanism results only

    res = resolve(tmp_path / "rel", {"q0": "yes", "q1": "no"}, reward_rule=MarketScoringReward())
    assert res.release_digest == man.digest and res.n_resolved == 2 and res.n_unresolved == 1
    resolved = [soa.Episode.model_validate_json(x) for x in (tmp_path / "rel" / "resolved" / "episodes.jsonl").read_text().splitlines()]
    q0 = next(e for e in resolved if e.item_id == "q0")
    assert q0.rewards["trader_1"] == pytest.approx(math.log(0.8 / 0.5))
    assert q0.rewards["trader_2"] == pytest.approx(math.log(0.3 / 0.8))
    assert q0.ground_truth["judge_correct"] == 0.0  # final price 0.3 on yes, which resolved yes
    assert (tmp_path / "rel" / "resolved" / "report.html").exists()


def test_tampered_release_is_rejected(tmp_path):
    items = forecasting_items()
    release(run_market(items), items, tmp_path / "rel")
    p = tmp_path / "rel" / "episodes.jsonl"
    text = p.read_text()
    assert '"yes":0.8' in text
    p.write_text(text.replace('"yes":0.8', '"yes":0.9'))
    assert not verify(tmp_path / "rel")
    with pytest.raises(ValueError):
        resolve(tmp_path / "rel", {"q0": "yes"})
