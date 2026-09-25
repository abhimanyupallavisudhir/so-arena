import itertools
import math
from dataclasses import replace

import pytest

from oversight_arena.analysis import binary_score, calibration, clustered_mean, paired_asd, truth
from oversight_arena.core import Evaluation
from oversight_arena.games import FiniteGame
from oversight_arena.optimization import (
    Decision,
    Leaf,
    best_of_n,
    best_of_n_weights,
    nested_best_of_n,
)


@pytest.mark.parametrize(
    "values,base,n",
    [
        ([0, 1], [0.5, 0.5], 2),
        ([1, 1, 0], [0.1, 0.3, 0.6], 3),
        ([-3, -2, -2], [0.2, 0.7, 0.1], 4),
        ([5, 5], [0.2, 0.8], 3),
        ([0, 1, 2], [0, 0.6, 0.4], 2),
        ([2], [1], 8),
    ],
)
def test_bon_matches_exhaustive_sampling(values, base, n):
    expected = [0.0] * len(values)
    for draws in itertools.product(range(len(values)), repeat=n):
        probability = math.prod(base[i] for i in draws)
        best = max(values[i] for i in draws)
        winners = [i for i in draws if values[i] == best]
        for winner in winners:
            expected[winner] += probability / len(winners)
    assert best_of_n_weights(values, n, base=base) == pytest.approx(expected)


@pytest.mark.parametrize("n", [0, -1, True, 1.5])
def test_bon_rejects_invalid_pressure(n):
    with pytest.raises(ValueError):
        best_of_n_weights([0, 1], n)


def test_bon1_and_large_n():
    assert best_of_n_weights([0.2, 0.8], 1) == [0.5, 0.5]
    assert best_of_n_weights([0.2, 0.8], 10000) == [0, 1]


def test_nested_general_sum_uses_own_reward_and_not_truth():
    tree = Decision(
        "critic",
        (
            Leaf({"critic": 0.9, "proposer": 0.9}, {"quality": 0}),
            Leaf({"critic": 0.1, "proposer": 0.1}, {"quality": 1}),
        ),
    )
    result = nested_best_of_n(tree, {"critic": 8})
    assert result.rewards["proposer"] > 0.89
    assert result.truth["quality"] < 0.01
    assert sum(result.leaf_probabilities.values()) == pytest.approx(1)


def test_nested_bon1_is_uniform_and_missing_truth_propagates():
    tree = Decision("a", (Leaf({"a": 0}, {"a": 1}), Leaf({"a": 1}, {})))
    result = nested_best_of_n(tree, {"a": 1})
    assert result.rewards["a"] == 0.5
    assert result.truth["a"] is None
    assert result.leaf_probabilities == {(0,): 0.5, (1,): 0.5}


def test_bon_selects_reward_even_when_wrong(episode_factory):
    good = episode_factory(0.2, quality=1)
    bad = episode_factory(0.9, quality=0)
    result = best_of_n([good, bad], "agent", 4)
    assert result.truth["agent"] == pytest.approx(1 / 16)
    assert result.rewards["agent"] > 0.8


def test_unresolved_never_becomes_zero(episode_factory):
    record = replace(episode_factory(), evaluations=())
    assert best_of_n([record], "agent", 4).truth["agent"] is None
    assert truth(record, "agent") is None


def test_latest_scorer_revision_and_ambiguity(episode_factory):
    record = episode_factory()
    record = replace(
        record, evaluations=(*record.evaluations, Evaluation("exact_match", "2", status="pending"))
    )
    assert truth(record, "agent") is None
    record = replace(record, evaluations=(*record.evaluations, Evaluation("other", "1")))
    with pytest.raises(ValueError, match="Multiple scorers"):
        truth(record, "agent")


def test_asd_two_worlds_and_pair_validation(episode_factory):
    good = episode_factory(0.8)
    bad = episode_factory(0.6)
    assert paired_asd([good], [bad], "agent").mean == pytest.approx(0.2)
    with pytest.raises(ValueError, match="identical"):
        paired_asd([good], [episode_factory(0.6, seed=2)], "agent")
    with pytest.raises(ValueError, match="Duplicate"):
        paired_asd([good, good], [bad, bad], "agent")


def test_task_cluster_bootstrap_not_replicate_weighted():
    result = clustered_mean([("a", 1)] * 100 + [("b", 0)], seed=10)
    assert result.mean == 0.5
    assert result.tasks == 2 and result.observations == 101
    assert result.low <= 0.5 <= result.high


def test_proper_scoring_and_calibration():
    assert binary_score(0.8) == pytest.approx(-0.08)
    assert binary_score(0.8, rule="log") - binary_score(0.6, rule="log") == pytest.approx(
        math.log(4 / 3)
    )
    with pytest.raises(ValueError, match="infinity"):
        binary_score(0, rule="log")
    assert calibration([0, 1], [0, 1])["brier"] == 0


def test_matching_pennies_regret_and_no_pure_equilibrium():
    table = {(a, b): ((1.0, -1.0) if a == b else (-1.0, 1.0)) for a in range(2) for b in range(2)}
    game = FiniteGame(("a", "b"), (("H", "T"), ("H", "T")), table)
    assert game.pure_equilibria() == []
    assert game.regrets([[0.5, 0.5], [0.5, 0.5]]) == {"a": 0, "b": 0}
    assert game.regrets([[1, 0], [1, 0]]) == {"a": 0, "b": 2}
    assert game.best_response_dynamics((0, 0))["status"] == "cycle"


def test_swarm_has_multiple_equilibria_and_coalition_deviation():
    from oversight_arena.demo import reporting_game

    game = reporting_game(0.2)
    assert game.pure_equilibria() == [(0, 0), (1, 1)]
    assert game.coalition_gain((1, 1), ["a", "b"])["gain"] == 0.8
    assert reporting_game(1.2).pure_equilibria() == [(1, 1)]


def test_game_rejects_incomplete_payoffs():
    with pytest.raises(ValueError, match="entire product"):
        FiniteGame(("a",), (("x", "y"),), {(0,): (1,)})


def test_bon_large_pressure_with_zero_probability_maximum():
    result = best_of_n_weights([1, 2, 3, 4], 100000, base=[0.1, 0.2, 0.7, 0])
    assert result == pytest.approx([0, 0, 1, 0])


def test_pool_refuses_mixed_mechanisms(episode_factory):
    a = episode_factory()
    b = replace(a, mechanism="another")
    with pytest.raises(ValueError, match="configuration"):
        best_of_n([a, b], "agent", 1)


def test_ranking_does_not_compare_incomparable_mechanism_scores(episode_factory):
    from oversight_arena.analysis import ranking_diagnostics

    a = episode_factory(0.2, 1)
    b = replace(episode_factory(0.9, 0), mechanism="different")
    assert ranking_diagnostics([a, b], "agent")["comparable_pairs"] == 0
