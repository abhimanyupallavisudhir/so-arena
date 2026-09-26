import numpy as np
import pytest

import so_arena as soa
from so_arena.analysis.metrics import fails_loudly
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.games import NormalFormGame
from so_arena.mechanisms import Debate, DirectJudge
from so_arena.samplers.arms import ASDExperiment


def test_fails_loudly_judge_uncertainty():
    dom = SyntheticPersuasion(n_items=60, hint_strength=0.3, seed=11)
    items, ctx = dom.load(), dom.context()
    exp = ASDExperiment([DirectJudge(), Debate(rounds=1, affordances={"agents": ["answer_key"]})], items,
                        agent=synthetic_arguer(honest_mean=0.8, dishonest_mean=0.2, sd=1.0),
                        fixtures={"judge": synthetic_judge()}, ctx=ctx)
    t = fails_loudly(exp.run())
    unc = t[t.signal == "judge_uncertainty"].set_index("mechanism")
    # errors concentrate where the judge is unsure: the uncertainty signal detects failures
    assert (unc["auroc"] > 0.6).all() and (unc["n_failures"] > 0).all()


def test_coalition_deviation_detects_collusion():
    # two debaters: exposing each other (E) vs staying quiet (Q); both prefer mutual quiet, but E is dominant
    A = np.array([[1.0, 3.0], [0.0, 2.0]])  # row payoffs; strategies [E, Q]
    g = NormalFormGame(["a", "b"], {"a": ["E", "Q"], "b": ["E", "Q"]}, {"a": A, "b": A.T},
                       outcomes={"truth_exposed": np.array([[1.0, 1.0], [1.0, 0.0]])})
    assert g.pure_nash() == [(0, 0)]
    devs = g.coalition_deviations((0, 0))
    assert len(devs) == 1 and devs[0]["deviation"] == {"a": "Q", "b": "Q"}
    assert devs[0]["outcomes"]["truth_exposed"] == 0.0
