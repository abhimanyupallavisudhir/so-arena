"""DPO preference pairs from episodes and game trees: same context only, ground-truth agreement."""

import math

import pytest

from so_arena.core.runner import PlayerSpec, Profile, build_players, run_episodes, run_sync
from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
from so_arena.integrations.rl import _agrees, preference_pairs
from so_arena.mechanisms import Debate, Propaganda
from so_arena.samplers.pools import expand_tree


def setup(n=6):
    dom = SyntheticPersuasion(n_items=n, seed=5)
    return dom.load(), dom.context()


def run(mech, items, ctx, profiles, repeats=3):
    return run_sync(run_episodes(mech, items, profiles, ctx=ctx, repeats=repeats))


def test_gt_agrees_handles_missing_and_equal_values():
    assert _agrees(1.0, -1.0) is True and _agrees(-1.0, 1.0) is False
    assert _agrees(1.0, 1.0) is None and _agrees(None, 1.0) is None and _agrees(math.nan, -1.0) is None
    assert _agrees(1.0, math.nan) is None


def test_pairs_never_cross_mechanisms_configs_opponents_items_or_stances():
    items, ctx = setup()
    agent = synthetic_arguer(honest_mean=1.0, dishonest_mean=0.2, sd=1.0, sophistry_rate=0.3)
    aff = {"agents": ["answer_key"]}
    profiles = [Profile(name=f"{arm}:{skill}", players={"agent": PlayerSpec(policy=agent, stance=arm),
                                                        "judge": synthetic_judge(skill=skill, label=f"judge{skill}")})
                for arm in ("true", "false") for skill in (0.5, 2.0)]
    eps = (run(Propaganda(affordances=aff), items, ctx, profiles)
           + run(Propaganda(affordances=aff, transform="prob"), items, ctx, profiles))
    pairs = preference_pairs(eps, "agent", items=items)
    assert len(pairs) > 0
    by_id = {e.id: e for e in eps}
    for _, p in pairs.iterrows():
        a, b = by_id[p["chosen_episode"]], by_id[p["rejected_episode"]]
        assert a.item_id == b.item_id and a.mechanism_config == b.mechanism_config
        assert a.players["judge"].policy_id == b.players["judge"].policy_id  # the same opponent
        assert a.players["agent"].stance == b.players["agent"].stance  # the same assigned stance
        assert p["reward_chosen"] > p["reward_rejected"]
        assert items[[it.id for it in items].index(p["item_id"])].question in p["prompt"]
    # 2 configs x 6 items x 2 stances x 2 judges = 48 contexts, each with 3 repeats: at most 3 pairs each
    assert pairs["context"].nunique() <= 48 and len(pairs) <= 48 * 3
    assert pairs["gt_agrees"].isna().all()  # one stance: equal values, no ground-truth preference
    # min_gap drops near-ties
    assert len(preference_pairs(eps, "agent", min_gap=10.0)) == 0


def test_open_choice_pairs_record_whether_ground_truth_agrees():
    """A worker that picks its answer: pairs of its behaviours with different values show how often the
    mechanism's preference is the right one - more often with a critic than with a single judge."""
    from so_arena.domains.synthetic import synthetic_critic, synthetic_reviewer, synthetic_worker
    from so_arena.mechanisms import ReviewedWork

    items, ctx = setup(10)
    worker = synthetic_worker(honest_choice_rate=0.5, honest_mean=1.0, dishonest_mean=0.2, sd=0.8,
                              sophistry_rate=0.4, sophistry_boost=2.5)
    aff = {"agents": ["answer_key"]}
    share = {}
    for name, mech, players in (
            ("single", ReviewedWork(critique_rounds=0, transform="prob", affordances=aff),
             {"worker": worker, "reviewer": synthetic_reviewer()}),
            ("critic", ReviewedWork(critique_rounds=1, rebuttal=False, transform="prob", affordances=aff),
             {"worker": worker, "critic": synthetic_critic(), "reviewer": synthetic_reviewer()})):
        eps = run(mech, items, ctx, [Profile(name="open", players=players)], repeats=6)
        pairs = preference_pairs(eps, "worker")
        known = pairs["gt_agrees"].dropna()
        assert len(known) > 20 and pairs["gt_agrees"].isna().any()  # equal values: no preference to agree with
        share[name] = float(known.astype(float).mean())
    assert share["critic"] > share["single"] + 0.1
    # a missing ground-truth value never counts as agreement (or disagreement)
    eps[0].ground_truth["role_values"]["worker"] = math.nan
    p = preference_pairs(eps, "worker")
    assert p[(p["chosen_episode"] == eps[0].id) | (p["rejected_episode"] == eps[0].id)]["gt_agrees"].isna().all()


def test_simultaneous_partner_moves_are_not_in_the_prompt():
    items, ctx = setup(3)
    arguer = synthetic_arguer(honest_mean=1.0, dishonest_mean=0.0, sd=1.0)
    prof = Profile(name="d", players={"debater_a": PlayerSpec(policy=arguer, stance="true"),
                                      "debater_b": PlayerSpec(policy=arguer, stance="false"), "judge": synthetic_judge()})
    eps = run(Debate(rounds=1, affordances={"agents": ["answer_key"]}), items, ctx, [prof], repeats=4)
    pairs = preference_pairs(eps, "debater_b", items=items)
    assert len(pairs) > 0  # simultaneous: b's context does not include a's argument, so repeats pair up
    assert not pairs["prompt"].str.contains("debater_a").any()


def test_tree_pairs_compare_candidates_of_one_information_set():
    items, ctx = setup(2)
    agent = synthetic_arguer(honest_mean=1.0, dishonest_mean=0.2, sd=1.0)
    mech = Propaganda(affordances={"agents": ["answer_key"]})
    prof = Profile(name="pool", players={"agent": PlayerSpec(policy=agent, stance="false"), "judge": synthetic_judge()})
    trees, leaves = [], []
    for it in items:
        tree, eps = run_sync(expand_tree(mech, it, build_players(prof, it), pool_sizes={"agent": 5}, ctx=ctx,
                                         keep_episodes=True))
        trees.append(tree)
        leaves += eps
    pairs = preference_pairs(trees, "agent", items=items, episodes=leaves)
    assert pairs["context"].nunique() == 2 and len(pairs) == 2 * 10  # 5 candidates: 10 pairs per tree
    assert (pairs["reward_chosen"] > pairs["reward_rejected"]).all()
    assert pairs["value_chosen"].eq(-1.0).all() and pairs["gt_agrees"].isna().all()  # a false stance: equal values
    assert pairs["chosen"].str.contains("I argue").all() and pairs["prompt"].notna().all()
    # without the leaf episodes: summaries, and no prompt
    bare = preference_pairs(trees, "agent")
    assert len(bare) == 20 and bare["prompt"].isna().all()
    assert bare["reward_chosen"].tolist() == pytest.approx(pairs["reward_chosen"].tolist())
