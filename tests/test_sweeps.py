import math

import pytest

from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.mechanisms import Debate, Propaganda
from so_arena.models import register_model
from so_arena.samplers.arms import ASDExperiment
from so_arena.samplers.sweeps import breakdown_point, capability, sweep


def test_sweep_capability_gap_breakdown():
    """As agents out-class the judge (sophistry grows with the gap), ASD falls; find where it hits zero."""
    dom = SyntheticPersuasion(n_items=20, seed=7)
    items, ctx = dom.load(), dom.context()
    aff = {"agents": ["answer_key"]}

    def make(cfg):
        gap = cfg["gap"]
        arguer = synthetic_arguer(honest_mean=1.0, dishonest_mean=0.4 * gap, sd=0.3)
        return ASDExperiment([Propaganda(affordances=aff), Debate(rounds=1, affordances=aff)], items,
                             agent=arguer, fixtures={"judge": synthetic_judge()}, ctx=ctx)

    df = sweep([{"gap": g} for g in (0, 1, 2, 3, 4)], make)
    assert set(df["mechanism"]) == {"propaganda", "debate"} and len(df) == 10
    bp = breakdown_point(df, x="gap").set_index("mechanism")
    assert (bp["slope"] < 0).all()
    # argument strengths equalize at gap 2.5, but the judge's own prior keeps ASD positive a little longer
    assert 3.0 < bp.loc["propaganda", "breakdown"] < 5.0 and 3.0 < bp.loc["debate", "breakdown"] < 5.0
    assert bp.loc["debate", "slope"] < bp.loc["propaganda", "slope"]  # debate's incentive is more sensitive to the gap


def test_capability_from_registry():
    register_model(name="test/model-x", params_b=7, ratings={"chess_elo": 1500})
    assert capability("test/model-x", "chess_elo") == 1500
    assert capability("test/model-x", "log_params") == pytest.approx(math.log10(7e9))
    assert capability("unknown/model") is None
