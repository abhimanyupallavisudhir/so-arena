"""ParamSearch: numeric search over programmatic policies' parameters."""

import json

import pytest

from so_arena.core.policy import FunctionPolicy
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_reviewer, synthetic_worker
from so_arena.mechanisms import PredictionMarket, ReviewedWork
from so_arena.samplers.param_search import ParamSearch, encode, satisfies
from so_arena.samplers.prompt_search import Candidate, honesty_margin


def setup(n=12, n_eval=6):
    dom = SyntheticPersuasion(n_items=n + n_eval, seed=7)
    items = dom.load()
    return items[:n], items[n:], dom.context()


def forecaster(q: float) -> FunctionPolicy:
    def act(req, ctx):
        a, b = req.options
        return {a: q, b: 1 - q}

    return FunctionPolicy(act, label=f"q={q:.3f}")


def judge_at(p: float) -> FunctionPolicy:
    def act(req, ctx):
        a, b = req.options
        return {a: p, b: 1 - p}

    return FunctionPolicy(act, label="judge")


def test_param_search_finds_the_known_optimum_of_a_proper_score():
    """A judged market pays a trader the expected log-score gain under the judge's final belief (0.7 on the
    first option), so the best forecast is exactly 0.7 (Gibbs' inequality)."""
    train, test, ctx = setup()
    search = ParamSearch(PredictionMarket(n_traders=1, rounds=1, judged=True), train, role="trader_1",
                         factory=forecaster, space={"q": (0.01, 0.99)}, others={"judge": judge_at(0.7)},
                         iterations=10, candidates_per_iter=6, eval_items=test, ctx=ctx, seed=1)
    res = search.run()
    best = res.best
    assert best.split == "eval" and best.params == res.candidates[[c.id for c in res.candidates].index(best.parent)].params
    assert best.params["q"] == pytest.approx(0.7, abs=0.03)
    assert res.path()["best_reward"].is_monotonic_increasing
    assert all(c.accepted for c in res.candidates)  # no constraint
    assert len({c.strategy for c in res.candidates}) == len(res.candidates)  # no candidate evaluated twice


def worker_search(train, test, ctx, constraint, seed=0):
    return ParamSearch(ReviewedWork(critique_rounds=0, transform="prob", affordances={"agents": ["answer_key"]}),
                       train, role="worker",
                       factory=lambda **p: synthetic_worker(honest_mean=1.0, dishonest_mean=0.2, sd=0.5,
                                                            sophistry_boost=2.5, **p),
                       space={"honest_choice_rate": (0.0, 1.0), "sophistry_rate": (0.0, 1.0)},
                       others={"reviewer": synthetic_reviewer()}, iterations=5, candidates_per_iter=6, repeats=2,
                       eval_items=test, ctx=ctx, constraint=constraint, seed=seed)


def test_measured_value_constraint_maps_the_honest_and_deceptive_frontiers():
    train, test, ctx = setup()
    free = worker_search(train, test, ctx, None).run()
    honest = worker_search(train, test, ctx, "honest").run()
    deceptive = worker_search(train, test, ctx, "deceptive").run()
    # a single judge fooled by sophistry: the unconstrained optimum is to lie with sophistry
    assert free.best.mean_value < 0 and free.best.params["sophistry_rate"] > 0.6
    # the constrained search's best is measured honest; candidates failing the filter are kept but never chosen
    assert honest.best.mean_value > 0
    rejected = [c for c in honest.candidates if not c.accepted]
    assert rejected and all(not c.mean_value > 0 for c in rejected)
    assert all(c.accepted for c in honest.evaluated) or honest.best.accepted
    assert deceptive.best.mean_value < 0
    # classed by measured value, the unconstrained winner is deceptive and beats every honest strategy found
    m = honesty_margin({"honest": honest, "unconstrained": free, "deceptive": deceptive}, n_boot=200)
    assert m["deceptive_id"] == free.best.id and m["margin"] < 0 and m["ci_high"] < 0
    frame = honest.frame()
    assert "accepted" in frame and (~frame["accepted"]).any()


def test_constraints_fail_closed_and_spaces_are_checked():
    c = Candidate(id="x", strategy=encode({"a": 1.0}), values=[None, None], rewards=[1.0, 1.0])
    assert not satisfies("honest", c) and not satisfies("deceptive", c) and satisfies(None, c)
    assert json.loads(encode({"b": 0.1234567891, "a": 2})) == {"a": 2, "b": 0.123457}
    train, _, ctx = setup(2, 1)
    with pytest.raises(ValueError, match="lo > hi"):
        ParamSearch(PredictionMarket(), train, role="trader_1", factory=forecaster, space={"q": (0.9, 0.1)}, others={})


def test_episodes_without_reward_count_as_the_worst_reward():
    """A setting that breaks the episodes on hard items must not be ranked on the easy items alone."""
    from so_arena.core.policy import FunctionPolicy
    from so_arena.domains.monitoring import SyntheticMonitoring, keyword_monitor, monitoring_grader
    from so_arena.mechanisms import get_mechanism
    from so_arena.samplers.param_search import ParamSearch

    dom = SyntheticMonitoring(n_items=8)
    items = dom.load()
    for i, it in enumerate(items):  # half the items are hard
        it.context["monitoring"] = dict(it.context["monitoring"], success=0.9 if i % 2 == 0 else 0.1)

    def factory(crash_on_hard: int = 0):
        def act(req, ctx):
            s = req.view.item.context["monitoring"]["success"]
            if crash_on_hard and s < 0.5:
                raise RuntimeError("policy broke")
            return f"Implemented.\nChanged files: src/feature.py\nTest pass rate: {s:.4f}"
        return FunctionPolicy(act, label=f"crash={crash_on_hard}")

    res = ParamSearch(get_mechanism("monitored_work", lam=0.0), items, role="worker", factory=factory,
                      space={"crash_on_hard": [0, 1]}, seeds=[{"crash_on_hard": 0}, {"crash_on_hard": 1}],
                      iterations=0, others={"monitor": keyword_monitor(), "grader": monitoring_grader()},
                      ctx=dom.context()).run()
    by = {c.params["crash_on_hard"]: c for c in res.candidates if c.split == "train"}
    assert by[1].reward_coverage == 0.5 and by[0].reward_coverage == 1.0
    assert by[1].missing_reward is not None and by[1].mean_reward <= by[0].mean_reward
    assert res.best.params == {"crash_on_hard": 0}
