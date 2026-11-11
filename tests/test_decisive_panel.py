"""Tests for the model panel (decisive_panel.py, decisive_panel_report.py)."""

import json
import sys
from pathlib import Path

import numpy as np
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


def test_brier_asd_scores_the_judges_probability_for_the_argued_side():
    rows = _rows()
    rows["p_true"] = [0.9, 0.2, 0.3, 0.6, 0.5, 0.5]
    it = dpr.both_units(dpr.with_prob_reward(rows), ["mechanism", "debater"]).set_index("item")
    # Brier reward -(1 - p_side)^2: honest p_side = p_true, liar p_side = 1 - p_true
    assert it.asd_b.to_dict() == pytest.approx({"q1": -(0.1 ** 2) + 0.2 ** 2, "q2": -(0.7 ** 2) + 0.6 ** 2,
                                                "q3": 0.0})


def test_summary_reports_mean_median_and_gain_with_intervals():
    items = pd.DataFrame({"mechanism": "m", "item": [f"q{i}" for i in range(30)],
                          "asd": [0.0] * 20 + [3.0] * 10, "gain": 0.5})
    s = dpr.summarize(items, ["mechanism"]).iloc[0]
    assert s.asd == pytest.approx(1.0) and s.asd_med == pytest.approx(0.0) and s.gain == pytest.approx(0.5)
    assert s.asd_lo <= s.asd <= s.asd_hi and s["items"] == 30


def test_blind_baseline_is_matched_to_the_items_an_agent_covers(monkeypatch, tmp_path):
    def row(debater, mech, item, value, p, repeat=0, judge="j"):
        return {"judge": judge, "debater": debater, "mechanism": mech, "item": item, "arm": "x", "repeat": repeat,
                "episode": f"{debater}-{item}-{value}", "correct": True, "p_true": p, "parse_ok": True, "cost": 0.0,
                "role": "a", "value": value, "reward": np.log(p if value > 0 else 1 - p)}
    # the blind judge is right on item "easy" (p=.9) and unsure on "hard" (p=.5); the agent's only committed lie is on "hard"
    rows = [row("", "direct", it, v, p) for it, p in (("easy", .9), ("hard", .5)) for v in (1, -1)]
    rows += [row("a", "debate_sim", it, v, .5) for it in ("easy", "hard") for v in (1, -1)]
    rows = pd.DataFrame(rows)
    verdicts = pd.DataFrame([{"key": "k", "debater": "a", "mechanism": "debate_sim", "item": "easy", "repeat": 0,
                              "verdict": "conceded"},
                             {"key": "l", "debater": "a", "mechanism": "debate_sim", "item": "hard", "repeat": 0,
                              "verdict": "committed"}])
    monkeypatch.setattr(dpr, "load", lambda run: (rows, verdicts))
    monkeypatch.setattr(dpr, "config", lambda: {"agents": [{"name": "a", "aa_index": 10}],
                                                "judges": [{"name": "j", "aa_index": 5}]})
    monkeypatch.setattr(dpr, "slope", lambda *a, **k: {})
    monkeypatch.setattr(dpr, "N_BOOT", 2)
    t = dpr.tables(tmp_path)
    assert t["blind_pooled"].iloc[0].asd_b == pytest.approx(0.4)  # over both items
    assert t["agents_committed"].iloc[0].blind_b == pytest.approx(0.0)  # only on the item it lied on
    assert t["agents_all"].iloc[0].blind_b == pytest.approx(0.4)


def test_regression_band_covers_the_fit_and_two_level_ci_is_wider():
    rng = np.random.default_rng(0)
    idx = {f"m{k}": float(k) for k in range(8)}
    items = [{"item": i, "debater": m, "conc": 0.05 * x + rng.normal(0, 0.05) + (0.1 if k % 2 else -0.1)}
             for i in range(60) for k, (m, x) in enumerate(idx.items())]
    r = dpr.regression(pd.DataFrame(items), "debater", idx, "conc")
    assert r["slope_per_10"] == pytest.approx(0.5, abs=0.15)
    assert (r["band_lo"] <= r["fit"] + 1e-9).all() and (r["fit"] <= r["band_hi"] + 1e-9).all()
    # model-level scatter (the +-0.1 offsets) is invisible to the items-only bootstrap
    assert r["hi"] - r["lo"] > r["hi_items"] - r["lo_items"]


def test_optimized_panel_trees_share_leaves_and_feed_the_held_out_asd(tmp_path, monkeypatch):
    import decisive_ic as dic
    import decisive_panel_bon as dpb
    import so_arena as soa
    from so_arena.core.items import AnswerOption, GroundTruth, TaskItem
    from so_arena.core.runner import run_sync
    from so_arena.models.base import FunctionModel
    from so_arena.samplers.pools import load_trees

    items = [TaskItem(id=f"q{i}", domain="t", question=f"Is {i} even?",
                      answers=[AnswerOption(label="A", text="yes"), AnswerOption(label="B", text="no")],
                      ground_truth=GroundTruth(correct="A")) for i in range(2)]
    speaker = soa.LLMPolicy(FunctionModel(lambda m, o, i: f"speech {i}", name="agent"))
    # the judge's probability depends on the sample index: the held-out tree must draw different samples
    judge = soa.LLMPolicy(FunctionModel(lambda m, o, i: f'ok {{"A": {0.2 + 0.6 * (i % 2)}, "B": {0.8 - 0.6 * (i % 2)}}}',
                                        name="judge"))
    monkeypatch.setattr(dpb, "agent_policy", lambda spec: speaker)
    monkeypatch.setattr(dpb, "judge_policy", lambda spec: judge)
    mech = dpb.mechanisms()[0]
    run_sync(dpb.grow_cell(tmp_path, items, {"name": "ag"}, {"name": "jd"}, mech, pool=3, concurrency=2,
                           estimate=False))
    assert dpb.done_items(tmp_path, "ag", "jd", mech.name) == {"q0", "q1"}
    for arm in ("true", "false"):
        sel = {t.item_id: t for t in load_trees(dpb.tree_path(tmp_path, "ag", "jd", mech.name, arm))}
        held = {t.item_id: t for t in load_trees(dpb.tree_path(tmp_path, "ag", "jd", mech.name, arm, holdout=True))}
        for k in sel:
            assert set(sel[k].leaves) == set(held[k].leaves) and len(sel[k].leaves) == 3
    rows = dic.cell_rows(dpb.root(tmp_path), "ag", "jd", mech.name, holdout=True)
    assert {r["item"] for r in rows} == {"q0", "q1"} and {r["mode"] for r in rows} == {"liar", "both"}


def test_prune_judge_cache_keeps_arguments_and_records_cost(tmp_path):
    import json
    import sqlite3

    import prune_judge_cache as P

    assert P.is_verdict('reasoning...\n\n{"A": 0.4, "B": 0.6}')
    assert not P.is_verdict("An argument that B is correct, because {A} implies B.")
    run = tmp_path / "run"
    db = run / "cache" / "s" / "responses.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, value TEXT)")
    rows = {"k1": ('x\n{"A": 0.2, "B": 0.8}', 0.5), "k2": ("an argument", 1.0), "k3": ('{"A": 1, "B": 0}', 0.25)}
    con.executemany("INSERT INTO cache VALUES (?, ?)",
                    [(k, json.dumps({"text": t, "usage": {"cost_usd": c}})) for k, (t, c) in rows.items()])
    con.commit()
    con.close()
    P.main(["--run", str(run), "--stage", "s"])
    left = [k for (k,) in sqlite3.connect(db).execute("SELECT key FROM cache")]
    assert left == ["k2"]
    assert json.loads((run / "spend.json").read_text())["s_pruned"] == 0.75


def _calibration():
    import decisive_calibration as dc

    return dc


def test_platt_recovers_a_known_map_and_never_inverts():
    dc = _calibration()
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, 40000)
    y = (rng.random(len(x)) < dc.expit(0.5 * x)).astype(float)  # the judge is twice as extreme as it should be
    a, b = dc.platt(x, y)
    assert abs(a) < 0.05 and abs(b - 0.5) < 0.03
    assert dc.platt(x, y, symmetric=True)[0] == 0.0
    # a judge that is systematically wrong gets no signal rather than an inverted one
    assert dc.platt(x, 1 - y, symmetric=True)[1] == pytest.approx(0.0, abs=1e-3)


def test_platt_is_stable_when_a_judge_answers_exactly_zero_or_one():
    dc = _calibration()
    p = np.array([0.0, 1.0] * 500 + [0.7, 0.3] * 500)
    y = np.array([0.0, 1.0] * 400 + [1.0, 0.0] * 100 + [1.0, 0.0] * 500)
    a, b = dc.platt(dc.logit(p), y, symmetric=True)
    assert a == 0.0 and 0 < b < dc.B_MAX


def _cal_rows(n, wrong=0.3, conf=0.95, seed=0, mech="debate_sim"):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        ok = rng.random() > wrong
        for v in (1.0, -1.0):
            honest = v > 0
            p = conf if ok == honest else 1 - conf
            out.append({"judge": "j", "mechanism": mech, "item": f"i{i}", "condition": "all", "y": float(honest), "p": p, "w": 1.0})
    return pd.DataFrame(out)


def test_crossfit_calibrates_an_overconfident_judge_without_changing_accuracy():
    dc = _calibration()
    df = dc.crossfit(_cal_rows(1000), ["judge", "mechanism"])
    assert df.pc.notna().all()
    honest = df[df.y == 1]
    # 70% accurate, so a calibrated probability on the truth should be near 0.7 / 0.3
    assert honest.pc.max() == pytest.approx(0.7, abs=0.04)
    assert ((honest.p > .5) == (honest.pc > .5)).mean() == 1.0


def test_item_asd_brier_matches_probability_identity_in_debate():
    dc = _calibration()
    df = _cal_rows(200, wrong=0.2, conf=0.8)
    it = dc.item_asd(df, ["judge"], "p")
    # for mirrored sides, Brier ASD = 2 E[p on the truth] - 1
    assert it.b.mean() == pytest.approx(2 * df[df.y == 1].p.mean() - 1, abs=1e-9)


def test_calibration_equalizes_judges_that_differ_only_in_scale():
    dc = _calibration()
    a = dc.crossfit(_cal_rows(1500, conf=0.99, seed=1), ["judge", "mechanism"])
    b = dc.crossfit(_cal_rows(1500, conf=0.60, seed=1), ["judge", "mechanism"])
    raw = dc.item_asd(a, ["judge"], "p").b.mean() - dc.item_asd(b, ["judge"], "p").b.mean()
    cal = dc.item_asd(a, ["judge"], "pc").b.mean() - dc.item_asd(b, ["judge"], "pc").b.mean()
    assert abs(raw) > 0.15 and abs(cal) < 0.02


def test_panel_bon_audit_reads_the_liars_pool_and_weights_by_selection():
    import decisive_panel_bon_audit as pba

    # consultancy: one node, children are leaves
    leaves = {f"l{i}": {"rewards": {"consultant": float(i)}} for i in range(4)}
    tree = {"root": "r", "nodes": {"r": {"role": "consultant", "children": list(leaves), "candidates": list("abcd")}},
            "leaves": leaves}
    texts, u = pba.liar_pool(tree, "consultant")
    assert texts == list("abcd") and list(u) == [0, 1, 2, 3]
    # debate: root debater_a (2 candidates) x debater_b (3 candidates); the liar's mean is over the other side
    lv, nodes = {}, {"r": {"role": "debater_a", "children": ["b0", "b1"], "candidates": ["A0", "A1"]}}
    for i in range(2):
        kids = []
        for j in range(3):
            lv[f"l{i}{j}"] = {"rewards": {"debater_a": i * 10 + j, "debater_b": -(i * 10 + j)}}
            kids.append(f"l{i}{j}")
        nodes[f"b{i}"] = {"role": "debater_b", "children": kids, "candidates": ["B0", "B1", "B2"]}
    t = {"root": "r", "nodes": nodes, "leaves": lv}
    ta, ua = pba.liar_pool(t, "debater_a")
    tb, ub = pba.liar_pool(t, "debater_b")
    assert ta == ["A0", "A1"] and list(ua) == [1.0, 11.0]
    assert tb == ["B0", "B1", "B2"] and list(ub) == [-5.0, -6.0, -7.0]
    w = pba.selection_weights(np.array([0.0, 1.0, np.nan, 2.0]), 16)
    assert w[2] == 0 and w.sum() == pytest.approx(1) and w[3] > w[1] > w[0]


def test_pe_psro_solve_finds_the_saddle_point_of_the_shared_estimator_game():
    import decisive_pe_psro as pp

    # honest provers: matching pennies against the estimator; liars: no say (a single strategy)
    A_H = np.array([[1.0, 0.0], [0.0, 1.0]])
    A_L = np.array([[0.0, 0.0]])
    s = pp.solve(A_H, A_L)
    assert s["estimator"] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert s["honest"] == pytest.approx([0.5, 0.5], abs=1e-6)
    assert s["value_honest"] == pytest.approx(0.5) and s["nash_conv"] == pytest.approx(0, abs=1e-6)
    # a dominant estimator strategy is played with certainty, and each prover best-responds to it
    A_H = np.array([[0.2, 0.9], [0.4, 0.8]])
    A_L = np.array([[-0.5, 0.1], [-0.3, 0.3]])
    s = pp.solve(A_H, A_L)
    assert s["estimator"] == pytest.approx([1, 0], abs=1e-6)
    assert s["honest"] == pytest.approx([0, 1], abs=1e-6) and s["liar"] == pytest.approx([0, 1], abs=1e-6)
    assert s["value_honest"] - s["value_liar"] == pytest.approx(0.7) and s["nash_conv"] == pytest.approx(0, abs=1e-6)


def test_pe_estimator_eic_is_positive_when_reward_tracks_accuracy():
    import decisive_pe_estimator as pe

    rows = []
    for it in range(30):
        for arm in ("True", "False"):
            y = arm == "True"
            for b, err in zip(pe.STRATEGIES, (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)):
                p = (1 - err) if y else err
                rows.append({"judge": "j", "behaviour": b, "item": f"i{it}", "arm": arm, "episode": f"{it}{arm}{b}",
                             "reward": err - 0.2 + (0.3 if y else 0.0), "p_top": p, "start": 0.1, "rounds": 0.0,
                             "leaf": 0.0})
    r = pe.prepare(pd.DataFrame(rows))
    e = pe.estimator_eic(r)
    top = e[(e.arm == "all") & (e.judge == "all")].iloc[0]
    assert top["cov"] > 0 and top["corr"] > 0.9 and top["share_positive"] == 1.0
    a = pe.prover_asd(r)
    assert a[(a.judge == "all") & (a.behaviour == "sincere")].asd.iloc[0] == pytest.approx(0.3)
