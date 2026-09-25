import math

import pytest

import so_arena as soa
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.mechanisms import Debate


def run(mech, items, ctx):
    prof = Profile(name="p", players={"debater_a": PlayerSpec(policy=synthetic_arguer(), stance="true"),
                                      "debater_b": PlayerSpec(policy=synthetic_arguer(), stance="false"),
                                      "judge": synthetic_judge()})
    return run_sync(run_episodes(mech, items, [prof], ctx=ctx))


def test_trainable_judge_is_paid_an_audited_proper_score():
    dom = SyntheticPersuasion(n_items=200, seed=3)
    items, ctx = dom.load(), dom.context()
    full = Debate(rounds=1, affordances={"agents": ["answer_key"]}, trainable={"judge": True},
                  reward=soa.JudgeScore("log") + soa.JudgeAuditScore(soa.truth_oracle(items), p=1.0))
    eps = run(full, items, ctx)
    assert all(e.error is None for e in eps)
    for e in eps:
        truth = next(it for it in items if it.id == e.item_id).true_label
        assert e.rewards["judge"] == pytest.approx(math.log(max(e.outcome.probs[truth], 1e-4)))
        assert {"debater_a", "debater_b", "judge"} <= set(e.rewards) and "judge" in e.trainable_roles
    mean_full = sum(e.rewards["judge"] for e in eps) / len(eps)
    sampled = Debate(rounds=1, affordances={"agents": ["answer_key"]}, trainable={"judge": True}, name="debate_p",
                     reward=soa.JudgeScore("log") + soa.JudgeAuditScore(soa.truth_oracle(items), p=0.3))
    eps = run(sampled, items, ctx)
    r = [e.rewards["judge"] for e in eps]
    assert 0 < sum(x == 0.0 for x in r) < len(r)                 # unaudited judgments pay 0
    assert sum(r) / len(r) == pytest.approx(mean_full, abs=0.35)  # inverse-probability weighting: unbiased
    assert "trusted audit" in sampled.reward_rule.describe()
