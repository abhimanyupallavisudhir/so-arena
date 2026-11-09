"""Tests for the general model benchmark (benchmark_solo.py, benchmark_report.py, benchmark_labels.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import benchmark_labels as bl  # noqa: E402
import benchmark_report as br  # noqa: E402
import benchmark_solo as bs  # noqa: E402
import decisive_panel as dpn  # noqa: E402


def test_non_reasoning_models_get_the_reasoning_token_budget_and_nothing_else_changes():
    cfg = {m["name"]: m for m in dpn.panel_config()["agents"]}
    for name, spec in cfg.items():
        pol, base = bs.solo(spec), dpn.agent_policy(spec, solo=True)
        thinks = spec.get("reasoning_effort") not in (None, "none")
        assert (name in bs.CHANGED) == (not thinks)
        assert pol.max_tokens == (base.max_tokens if thinks else bs.SOLO_MAX_TOKENS)
        assert (pol.cot, pol.reasoning_effort, pol.temperature) == (base.cot, base.reasoning_effort, base.temperature)
    # every reasoning configuration has the same budget, so the comparison is even
    assert {dpn.agent_policy(s, solo=True).max_tokens for n, s in cfg.items() if n not in bs.CHANGED} == {
        bs.SOLO_MAX_TOKENS}


def test_tiers_name_panel_models():
    names = {m["name"] for m in dpn.panel_config()["agents"]}
    assert set(br.STRONG) <= names and set(br.WEAK) <= names and set(br.LADDER) <= names
    assert not set(br.STRONG) & set(br.WEAK)


def _rows(models, items, acc):
    rng = np.random.default_rng(0)
    out = []
    for m, a in zip(models, acc):
        for i in items:
            c = float(rng.random() < a)
            out.append({"model": m, "item": i, "correct": c, "p_true": 0.9 if c else 0.2, "parse_ok": True,
                        "cost": 0.01, "source": "benchmark"})
    return pd.DataFrame(out)


def test_matrix_keeps_only_items_every_model_answered_and_bootstrap_is_unbiased():
    items = pd.DataFrame({"item": [f"q{i}" for i in range(200)], "family": "f/a", "domain": "f", "order": range(200)})
    rows = _rows(["a", "b"], items.item, [0.9, 0.6])
    rows = rows[~((rows.model == "b") & (rows.item == "q0"))]
    mx = br.Matrix(rows, ["a", "b"], items)
    assert mx.n == 199 and "q0" not in mx.items
    w = mx.weights(500)
    assert np.allclose(w.sum(1), mx.n)
    draws = br.boot_means(mx.correct, w)
    assert np.allclose(draws.mean(0), mx.correct.mean(1), atol=0.01)
    assert np.allclose(mx.brier, (1 - mx.p) ** 2)


def test_poisson_binomial_matches_binomial():
    from scipy.stats import binom

    assert np.allclose(br.poisson_binomial(np.full(6, 0.9)), binom.pmf(range(7), 6, 0.9))


def _tier(n_per=40):
    rows = []
    k = 0
    for s, w in ((6, 0), (6, 5), (3, 2), (0, 1)):
        for d in ("x", "y"):
            for _ in range(n_per):
                rows.append({"item": f"q{k}", "family": f"{d}/f", "domain": d, "order": k, "strong_correct": s,
                             "weak_correct": w, "n_correct": s + w})
                k += 1
    return pd.DataFrame(rows)


def test_stratified_order_balances_strata_in_every_prefix_and_drops_unstable_labels():
    tier = _tier()
    audit = pd.DataFrame({"item": ["q0", "q1"], "verdict": ["unstable", "replicated"]})
    out = br.stratify(tier, audit)
    assert "q0" not in set(out.item) and "q1" in set(out.item)
    assert list(out["rank"]) == list(range(1, len(out) + 1))
    first = out.head(40)
    assert first.stratum.value_counts().to_dict() == {s: 10 for s in br.STRATA}
    assert first.domain.value_counts().min() >= 18  # domains alternate within a stratum
    assert out.stratum.iloc[:4].tolist() == br.STRATA
    # deterministic
    assert out.item.tolist() == br.stratify(tier, audit).item.tolist()


def test_strata_definitions():
    out = br.stratify(_tier(1), None).set_index("item")
    assert out.loc["q0", "stratum"] == "gap" and out.loc["q2", "stratum"] == "easy"
    assert out.loc["q4", "stratum"] == "contested" and out.loc["q6", "stratum"] == "hard"


def test_label_audit_precedence():
    items = pd.DataFrame({"item": ["n1", "e1", "e2", "e3", "e4", "d1", "p1"],
                          "family": ["numbers/counts", "multiagent/auction", "multiagent/auction", "toyml/x",
                                     "sycophancy/wrong_answer", "dtbench/capabilities", "numbers/prime_race"]})
    rel = pd.DataFrame({"item": ["e1", "e2", "e3", "e4"], "p_label_true": [0.99, 0.6, 0.95, 0.97],
                        "p_replicate": [0.95, 0.55, 0.9, 0.93]})
    reps = pd.DataFrame({"item": ["e1", "e2", "e4"], "status": ["flipped", "same", "void"]})
    a = br.label_audit(items, rel, reps).set_index("item").verdict
    assert a["n1"] == "computed" and a["d1"] == "expert" and a["e1"] == "unstable" and a["e2"] == "noisy"
    assert a["e3"] == "measured" and a["e4"] == "weak"
    a2 = br.label_audit(items.replace({"p1": br.SIEVE_CHECKED[0]}), rel, reps).set_index("item").verdict
    assert a2[br.SIEVE_CHECKED[0]] == "sieve-checked"


def test_outcome_label_reliability_is_symmetric_and_sharper_at_the_extremes():
    p0, r0 = bl._beta_side(0)
    p1, r1 = bl._beta_side(1)
    p6, r6 = bl._beta_side(6)
    assert p0 == pytest.approx(p6) and r0 == pytest.approx(r6)
    assert 0.5 < p1 < p0 < 1 and 0.5 < r1 < r0 < 1


def test_binomial_standard_error_never_zero():
    se = bl._se({"metric": 1.0, "n": 30}, None)
    assert 0 < se < 0.1
    assert bl._se({"sessions": [0.5, 0.5, 0.5], "metric": 0.5}, 0.1) > 0  # pooled SD keeps a constant run honest


def test_bundle_first_300_items_are_the_panel_preview():
    bundle = br.BUNDLE / "items.csv.gz"
    if not bundle.exists():
        pytest.skip("no committed benchmark bundle")
    items = pd.read_csv(bundle)
    prev = pd.read_csv(bs.PREVIEW, usecols=["item", "mechanism"])
    assert set(items.sort_values("order").item[:300]) == set(prev[prev.mechanism == "solo"].item)
    assert len(items) == 3437 and items.item.is_unique


def test_item_set_loader_keeps_order_and_strata(tmp_path):
    if not bs.ITEM_SET.exists():
        pytest.skip("no committed item set")
    run = Path(__file__).resolve().parents[1] / "runs" / "benchmark"
    if not (run / "items.jsonl.gz").exists():
        pytest.skip("needs the saved item order (or a slow rebuild from the testbed)")
    got = bs.item_set(8, run=run)
    want = pd.read_csv(bs.ITEM_SET).sort_values("rank")
    assert [it.id for it in got] == want.item.tolist()[:8]
    gap = bs.item_set(5, strata=["gap"], run=run)
    assert [it.id for it in gap] == want[want.stratum == "gap"].item.tolist()[:5]
    assert all(it.true_label is not None for it in got)
