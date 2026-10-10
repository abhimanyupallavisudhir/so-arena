"""Budgets priced per token: token profiles, prices, totals, intervals and the dry-run prompt measurement."""

import pytest

from so_arena.budget import (
    Budget,
    Fixed,
    Item,
    Price,
    Tokens,
    agentic,
    conversation,
    measure,
    price,
    tinker_prices,
)


def test_agentic_profile_counts_each_turns_context_once():
    t = agentic(prompt=100, turns=2, reasoning=10, action=5, observation=20, final=7, final_reasoning=3, trained=True)
    # three calls, with contexts 100, 135, 170; the environment's 2 x 20 tokens and the prompt are uncached
    assert t.prefill == 140 and t.cached == 100 + 135 + 170 - 140
    assert t.sample == 2 * 15 + 10
    assert t.train == 100 + 2 * 35 + 10  # the final sequence, trained once
    assert agentic(100, 2, 10, 5, 20, 7).train == 0  # a frozen role is not trained


def test_conversation_profile_uncaches_only_what_others_add():
    t = conversation(prompt=50, calls=3, output=10, reasoning=5, others=20, trained=True)
    assert t.prefill == 50 + 2 * 20
    assert t.cached == (50 + 85 + 120) - 90
    assert t.sample == 45 and t.train == 50 + 3 * 15 + 2 * 20


def test_prices_train_only_where_training_is_sold():
    p = Price(prefill=1.0, cached_prefill=0.1, sample=2.0, train=3.0, provider="tinker")
    assert p.cost(Tokens(prefill=1e6, cached=1e6, sample=1e6, train=1e6)) == pytest.approx(6.1)
    with pytest.raises(ValueError, match="cannot be trained"):
        Price(1.0, 0.1, 2.0).cost(Tokens(train=1.0))
    tl, disc = tinker_prices(True), tinker_prices(False)
    assert "Qwen/Qwen3.5-9B" in tl and all(tl[m].sample >= disc[m].sample for m in tl)  # list prices: no discounts
    assert price("Qwen/Qwen3.5-9B").provider == "tinker" and price("anthropic/claude-haiku-4-5").train is None
    with pytest.raises(KeyError):
        price("nobody/unknown-model")


def test_budget_totals_intervals_and_sensitivity():
    ranges = {"turns": (1.0, 2.0, 6.0)}
    job = Item("job", lambda a: 1000 * a["turns"],
               lambda a: {"worker": ("Qwen/Qwen3.5-9B", Tokens(sample=1000, train=2000))})
    b = Budget([job, Fixed("people", (100.0, 200.0, 300.0), provider="people")], ranges)
    per_ep = (1000 * 1.995 + 2000 * 1.463) / 1e6
    tot = b.totals()
    assert tot.loc["tinker", "central"] == pytest.approx(2000 * per_ep)
    assert tot.loc["total", "high"] == pytest.approx(6000 * per_ep + 300)
    iv = b.interval(n=400)
    assert iv[0.1] < iv[0.5] < iv[0.9]
    assert iv[0.5] > b.total()  # right-skewed ranges: the median exceeds the all-central scenario
    s = b.sensitivity()
    assert s.loc[0, "assumption"] == "turns" and s.loc[0, "total_at_high"] > s.loc[0, "total_at_low"]


def test_measure_reports_real_prompt_sizes_without_calling_models():
    from so_arena.domains.synthetic import SyntheticPersuasion
    from so_arena.mechanisms import Debate

    dom = SyntheticPersuasion(n_items=3, seed=0)
    m = measure(Debate(rounds=2), dom.load(), ctx=dom.context(), output_tokens={"debater_a": 50, "debater_b": 50})
    assert set(m) == {"debater_a", "debater_b", "judge"}
    for r in ("debater_a", "debater_b"):
        assert m[r]["calls"] == 3 and m[r]["output"] == pytest.approx(150, rel=0.05)
        assert 0 < m[r]["first_input"] < m[r]["input"]  # later calls see the transcript
