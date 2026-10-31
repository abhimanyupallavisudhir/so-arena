"""The so-arena adapter for the conceptual validation testbed (skipped unless cvtestbed is installed)."""

import pytest

cvtestbed = pytest.importorskip("cvtestbed")


def test_testbed_questions_become_items():
    from so_arena.domains import get_domain

    dom = get_domain("testbed", families=["decisions/newcomb", "agents/auction"], per_family=6)
    items = dom.load()
    assert {it.metadata["family"] for it in items} == {"decisions/newcomb", "multiagent/auction"}
    for it in items:
        assert it.labels == ["A", "B"] and it.true_label in ("A", "B")
        c = it.censored()
        assert c.ground_truth is None and all(a.value is None for a in c.answers)
    # ids are the testbed's own
    assert {it.id for it in items} >= {q.id for q in cvtestbed.load(["decisions/newcomb"], n=6)}
