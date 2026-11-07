"""Tests for the model panel (decisive_panel.py, decisive_panel_report.py)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import decisive_panel as dpn  # noqa: E402
import decisive_panel_report as dpr  # noqa: E402


def test_panel_config_has_aa_scores_and_distinct_names():
    cfg = dpn.panel_config()
    for group in ("agents", "judges"):
        names = [m["name"] for m in cfg[group]]
        assert len(names) == len(set(names))
        assert all(isinstance(m["aa_index"], (int, float)) for m in cfg[group])
    # the effort ladder is one model at increasing effort and increasing index
    ladder = [m for m in cfg["agents"] if m["name"].startswith("gpt-5.6-sol@")]
    assert [m["reasoning_effort"] for m in ladder] == ["none", "low", "medium", "high"]
    assert [m["aa_index"] for m in ladder] == sorted(m["aa_index"] for m in ladder)


def test_agent_policies_give_reasoning_room_only_when_reasoning():
    cfg = {m["name"]: m for m in dpn.panel_config()["agents"]}
    thinking = dpn.agent_policy(cfg["gpt-5.6-sol@low"])
    plain = dpn.agent_policy(cfg["gpt-4o-mini"])
    off = dpn.agent_policy(cfg["gpt-5.6-sol@none"])
    assert thinking.reasoning_effort == "low" and thinking.max_tokens >= 8000
    assert plain.reasoning_effort is None and plain.max_tokens <= 3000
    assert off.reasoning_effort == "none" and off.max_tokens <= 3000


def test_judge_token_budget_defaults_and_overrides():
    cfg = {m["name"]: m for m in dpn.panel_config()["judges"]}
    # DeepSeek-V3.2 works problems out in the open and was cut off at 4000 tokens in the pilot
    assert dpn.judge_policy(cfg["deepseek-v3.2"]).max_tokens >= 12000
    assert dpn.judge_policy(cfg["gpt-4o-mini"]).max_tokens == 4000


def test_deepseek_is_pinned_to_hosts_that_respect_reasoning_off():
    cfg = dpn.panel_config()
    judge = {m["name"]: m for m in cfg["judges"]}["deepseek-v3.2"]
    agent = {m["name"]: m for m in cfg["agents"]}["deepseek-v3.2"]
    for pol in (dpn.judge_policy(judge), dpn.agent_policy(agent)):
        route = pol.model._args["provider"]
        assert route["allow_fallbacks"] is False and "gmicloud/fp8" not in route["order"]
    # unrouted models are left as they are
    assert "provider" not in getattr(dpn.judge_policy(judge | {"openrouter_provider": None}).model, "_args", {})


def test_every_panel_protocol_has_lead_roles():
    # rows without a lead role carry no (u, v): consultancy_1r was silently dropped from the pilot report
    from decisive_protocols import LEAD_ROLES

    assert all(m.name in LEAD_ROLES for m in dpn.mechanisms())


def test_protocols_are_judge_independent():
    for m in dpn.mechanisms():
        assert not getattr(m, "judge_questions", False)


def test_done_items_requires_every_agent_protocol_and_judge(tmp_path):
    rows = []
    for item in ("q1", "q2"):
        for a in ("A", "B"):
            for m in ("debate_sim", "consultancy_1r"):
                for j in ("j1", "j2"):
                    if item == "q2" and a == "B" and j == "j2":
                        continue  # q2 is missing one agent's second judge
                    rows.append({"item": item, "debater": a, "mechanism": m, "judge": j})
    p = tmp_path / "rows.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    assert dpn.done_items(p, ["A", "B"], ["j1", "j2"]) == {"q1"}
    assert dpn.done_items(p, ["A"], ["j1", "j2"]) == {"q1", "q2"}
    assert dpn.done_items(tmp_path / "none.csv", ["A"], ["j1"]) == set()


def _rows():
    out = []
    for item, (rt, rf) in {"q1": (-0.1, -2.0), "q2": (-1.0, -0.5), "q3": (-0.3, -0.3)}.items():
        for arm, value, reward in (("true", 1.0, rt), ("false", -1.0, rf)):
            out.append({"judge": "j", "debater": "A", "mechanism": "consultancy_1r", "item": item, "arm": arm,
                        "repeat": 0, "value": value, "reward": reward})
    return pd.DataFrame(out)


def test_per_item_asd_and_concordance():
    it = dpr.per_item(_rows(), ["mechanism", "debater"]).set_index("item")
    assert it.asd.to_dict() == pytest.approx({"q1": 1.9, "q2": -0.5, "q3": 0.0})
    assert it.conc.to_dict() == {"q1": 1.0, "q2": 0.0, "q3": 0.5}


def test_probability_asd_uses_the_judges_probability_for_the_argued_side():
    rows = _rows()
    # the judge's probability that the true answer is right; the liar argued the false one
    rows["p_true"] = [0.9, 0.2, 0.3, 0.6, 0.5, 0.5]
    it = dpr.both_units(dpr.with_prob_reward(rows), ["mechanism", "debater"]).set_index("item")
    # honest earns p_true, liar earns 1 - p_true
    assert it.asd_p.to_dict() == pytest.approx({"q1": 0.9 - 0.8, "q2": 0.3 - 0.4, "q3": 0.0})
    assert it.asd.to_dict() == pytest.approx({"q1": 1.9, "q2": -0.5, "q3": 0.0})


def test_committed_drops_both_arms_of_a_conceding_sample_for_every_judge():
    rows = pd.concat([_rows(), _rows().assign(judge="k")])
    verdicts = pd.DataFrame([{"debater": "A", "mechanism": "consultancy_1r", "item": "q2", "repeat": 0,
                              "verdict": "conceded"}])
    kept = dpr.committed(rows, verdicts)
    assert set(kept.item) == {"q1", "q3"} and len(kept) == 8


def test_slope_recovers_a_linear_trend():
    items = []
    idx = {"a": 10.0, "b": 20.0, "c": 30.0}
    for i in range(40):
        for name, x in idx.items():
            items.append({"item": f"q{i}", "debater": name, "conc": 0.3 + 0.01 * x})
    s = dpr.slope(pd.DataFrame(items), "debater", idx, "conc")
    assert s["slope_per_10"] == pytest.approx(0.1) and s["n"] == 3
