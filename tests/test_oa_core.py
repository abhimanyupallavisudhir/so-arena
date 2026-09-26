import oversight_arena as oa
from oversight_arena.core.strategy import Stance
from oversight_arena.agents.parsing import parse_choice, parse_distribution, parse_json, parse_scalar, split_thinking
from conftest import make_tasks


def test_task_view_strips_ground_truth():
    t = make_tasks(1)[0]
    v = t.view()
    assert not hasattr(v, "gt")
    assert all(not hasattr(o, "value") for o in v.options)
    assert "value" not in v.model_dump_json()
    assert "hidden" not in v.model_dump_json()


def test_strategy_binding_resolves_stances():
    t = make_tasks(2)[1]  # correct is B
    assert oa.argue(Stance.CORRECT).bind(t, "x").target == "B"
    assert oa.argue(Stance.INCORRECT).bind(t, "x").target == "A"
    s = oa.Strategy(name="s", stance=Stance.OPTION, option="A", instructions="argue {target}: {target_text}")
    b = s.bind(t, "x")
    assert b.target == "A" and "argue A: 3" in b.instructions
    assert oa.Strategy(name="free").bind(t, "x").target is None


def test_profile_ids_stable_and_distinct():
    p1 = oa.Profile.of(a=oa.HONEST, b=oa.DISHONEST)
    p2 = oa.Profile.of(a=oa.HONEST, b=oa.DISHONEST)
    p3 = oa.Profile.of(a=oa.DISHONEST, b=oa.HONEST)
    assert p1.id == p2.id != p3.id


def test_parsing():
    assert parse_choice("blah\nANSWER: B", ["A", "B"]) == "B"
    assert parse_choice("I think (A) is right", ["A", "B"]) == "A"
    assert parse_choice("Final answer: **B**", ["A", "B"]) == "B"
    d = parse_distribution('reasoning... {"A": 0.8, "B": 0.2}', ["A", "B"])
    assert abs(d["A"] - 0.8) < 1e-9
    d = parse_distribution("A: 70%\nB: 30%", ["A", "B"])
    assert abs(d["A"] - 0.7) < 1e-9
    d = parse_distribution('{"A": 60}', ["A", "B"])
    assert abs(d["A"] - 0.6) < 1e-9 and abs(d["B"] - 0.4) < 1e-9
    assert parse_scalar("SUSPICION: 7", 0, 10, "suspicion") == 7
    assert parse_scalar("probability: 45%", 0, 1, "probability") == 0.45
    assert parse_json("x {'a': 1,} y") == {"a": 1}
    pub, priv = split_thinking("<thinking>secret plan</thinking>Public text")
    assert pub == "Public text" and priv == "secret plan"


def test_score_prob_transforms():
    from oversight_arena.core.rewards import score_prob
    import math

    assert abs(score_prob(0.5, "log") - math.log(0.5)) < 1e-9
    assert score_prob(0.9, "brier") > score_prob(0.1, "brier")
    assert score_prob(0.6, "win") == 1.0 and score_prob(0.4, "win") == 0.0
    assert abs(score_prob(0.5, "logit")) < 1e-9
