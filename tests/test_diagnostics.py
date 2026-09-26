"""Diagnostics beyond IC, checked on frames and episodes with planted biases."""

import math
import re

import numpy as np
import pandas as pd
import pytest

from so_arena.analysis import diagnostics as dg
from so_arena.analysis.frames import episode_frame, role_frame
from so_arena.analysis.report import build_report
from so_arena.core.policy import FunctionPolicy
from so_arena.domains.synthetic import SyntheticPersuasion
from so_arena.mechanisms import Propaganda
from so_arena.samplers.arms import ASDExperiment


def frame(rows):
    base = {"mechanism": "m", "kind": "agent", "trainable": True, "label": None, "stance": None, "position": None,
            "manipulation_ok": None, "parse_ok": True, "judge_parse_ok": True, "n_words": 10, "n_claims": 0,
            "n_verified": 0, "episode_id": None}
    return pd.DataFrame([{**base, **r} for r in rows])


def test_reward_snr_separates_signal_noise_and_item_effects():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        item_effect = rng.normal(0, 3.0)
        for case, mean in (("honest", 1.0), ("deceptive", -1.0)):
            for _ in range(4):
                rows.append({"item_id": f"i{i}", "role": "agent", "label": case,
                             "reward": item_effect + mean + rng.normal(0, 0.5), "value": 1.0 if case == "honest" else -1.0})
    s = dg.reward_snr(frame(rows)).iloc[0]
    assert s["signal_var"] == pytest.approx(2.0, abs=0.3)  # variance of {+1, -1} with ddof=1
    assert s["noise_var"] == pytest.approx(0.25, abs=0.05)
    assert s["snr"] == pytest.approx(8.0, rel=0.25)
    assert s["item_share"] > 0.7  # most of the reward's variance is which item it was


def test_length_bias_is_measured_at_fixed_ground_truth():
    rng = np.random.default_rng(1)
    rows_biased, rows_fair = [], []
    for i in range(20):
        for value in (1.0, -1.0):
            for _ in range(6):
                words = int(rng.integers(20, 200))
                base = {"item_id": f"i{i}", "role": "agent", "value": value, "n_words": words}
                rows_biased.append({**base, "reward": value + 0.01 * words + rng.normal(0, 0.1)})
                # longer answers are more often right, and the reward pays only for rightness: no bias
                rows_fair.append({**base, "reward": value + rng.normal(0, 0.1)})
    biased = dg.length_bias(frame(rows_biased)).iloc[0]
    fair = dg.length_bias(frame(rows_fair)).iloc[0]
    assert biased["length_rho"] > 0.8 and biased["ci_low"] > 0.7
    assert abs(fair["length_rho"]) < 0.15 and fair["ci_low"] < 0 < fair["ci_high"]


def test_position_bias_finds_a_slot_advantage():
    rows = []
    for i in range(12):
        for a_value in (1.0, -1.0):  # counterbalanced: each debater argues each side
            rows += [{"item_id": f"i{i}", "role": "debater_a", "value": a_value, "reward": a_value + 0.3},
                     {"item_id": f"i{i}", "role": "debater_b", "value": -a_value, "reward": -a_value}]
    r = dg.position_bias(frame(rows)).iloc[0]
    assert r["slot_advantage"] == pytest.approx(0.3) and r["n_items"] == 12


def test_calibration_of_an_overconfident_judge():
    rows = [{"item_id": f"i{i}", "mechanism": "m", "confidence": 0.9, "judge_correct": float(i % 5 < 3)}
            for i in range(100)]
    c = dg.calibration(pd.DataFrame(rows), n_boot=50).iloc[0]
    assert c["accuracy"] == pytest.approx(0.6) and c["ece"] == pytest.approx(0.3) and c["overconfidence"] == pytest.approx(0.3)
    assert dg.ece([0.7] * 10, [1] * 7 + [0] * 3) == pytest.approx(0.0)
    assert dg.ece([math.nan, 0.5], [1, 0]) == pytest.approx(0.5)  # missing pairs are dropped
    rel = dg.reliability(pd.DataFrame(rows))
    assert len(rel) == 1 and rel["n"].iloc[0] == 100


def test_compliance_flags_arms_that_did_not_do_what_they_were_told():
    rows = []
    for i in range(10):
        rows.append({"item_id": f"i{i}", "role": "agent", "stance": "A", "position": "A", "value": 1.0, "n_words": 100})
        # the "false" arm refuses half the time (argues the truth instead) and writes less
        rows.append({"item_id": f"i{i}", "role": "agent", "stance": "B", "position": "B" if i % 2 else "A",
                     "value": -1.0, "n_words": 40})
    c = dg.compliance(frame(rows)).set_index("case")
    assert c.loc["stance=A", "stance_followed"] == 1.0 and c.loc["stance=B", "stance_followed"] == 0.5
    assert c.loc["stance=A", "words"] == 100 and c.loc["stance=B", "words"] == 40


def length_loving_judge(prefer_a: float = 0.0):
    def act(req, ctx):
        text = "\n".join(m.content for m in req.prompt)
        m = re.search(r"I argue for \(?([AB])\)?", text)
        words = len(re.findall(r"\bword\b", text))
        p = 1 / (1 + math.exp(-(words - 30) / 10))
        other = {"A": "B", "B": "A"}
        probs = {m.group(1): p, other[m.group(1)]: 1 - p} if m else {"A": 0.5, "B": 0.5}
        a = min(1.0, probs["A"] + prefer_a)
        return {"A": a, "B": 1 - a}

    return FunctionPolicy(act, label="length_judge")


def rambler():
    def act(req, ctx):
        return f"I argue for ({req.view.stance}). " + "word " * ctx.rng.randint(0, 60)

    return FunctionPolicy(act, label="rambler")


def test_end_to_end_length_bias_option_bias_and_report(tmp_path):
    dom = SyntheticPersuasion(n_items=10, seed=3)
    items = dom.load()
    exp = ASDExperiment([Propaganda()], items, agent=rambler(), fixtures={"judge": length_loving_judge()},
                        ctx=dom.context(), repeats=4)
    eps = exp.run()
    rf = role_frame(eps)
    assert (rf["n_words"] > 3).all()
    lb = dg.length_bias(rf).iloc[0]
    assert lb["length_rho"] > 0.8  # the judge pays for words whatever the truth
    ob = dg.option_label_bias(eps, items)
    assert set(ob["label"]) == {"A", "B"} and ob["bias"].abs().max() < 0.05  # no label preference planted
    ef = episode_frame(eps)
    assert {"confidence", "oversight_tokens", "n_claims", "gt_access_cost"} <= set(ef.columns)
    cs = dg.cost_summary(eps)
    assert cs["episodes"].iloc[0] == len(eps)
    table = dg.diagnostics_table(eps)
    assert {"reward_snr", "length_rho", "ece", "stance_followed"} <= set(table.columns)
    html = build_report(eps, tmp_path / "r.html").read_text()
    assert "Diagnostics" in html and "length_rho" in html
    # a judge that leans toward label A
    biased = ASDExperiment([Propaganda()], items, agent=rambler(), fixtures={"judge": length_loving_judge(0.2)},
                           ctx=dom.context(), repeats=2).run()
    ob = dg.option_label_bias(biased, items).set_index("label")
    assert 0.1 < ob.loc["A", "bias"] <= 0.2 and ob.loc["A", "ci_low"] > 0  # 0.2, less where A was capped at 1
    assert ob.loc["B", "bias"] == pytest.approx(-ob.loc["A", "bias"])
