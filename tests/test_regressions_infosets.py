"""Regressions from the review of game-tree optimization: best-of-N selecting on moves the chooser never saw
(information sets), and summaries that averaged ground truth over the labelled plays without saying so."""

import logging
import math

import numpy as np
import pytest

import so_arena as soa
from so_arena.analysis.optimization import (
    BestOfN,
    GameTree,
    TreeLeaf,
    TreeNode,
    Tilted,
    evaluate_tree,
    optimization_grid,
    pool_curve,
)
from so_arena.core.mechanism import Mechanism, Outcome, RoleSpec
from so_arena.core.policy import FunctionPolicy
from so_arena.core.rewards import RewardRule
from so_arena.core.runner import run_sync
from so_arena.samplers.pools import expand_tree

# ------------------------------------------------------------------------------ information sets


class Match(RewardRule):
    """The guesser is paid 1 for matching the coin; the dealer is paid for a mismatch."""

    name = "match"

    def compute(self, ep):
        hit = float(ep.outcome.data["coin"] == ep.outcome.data["guess"])
        return {"guesser": hit, "dealer": 1.0 - hit}


class CoinGame(Mechanism):
    name = "coin"

    def __init__(self, public: bool = False, **kw):
        super().__init__(**kw)
        self.public = public

    def roles(self):
        return {"dealer": RoleSpec(name="dealer"), "guesser": RoleSpec(name="guesser")}

    def default_reward(self):
        return Match()

    async def protocol(self, g):
        d = await g.act("dealer", kind="choice", options=["H", "T"], prompt="Flip the coin.", phase="flip",
                        visible_to=None if self.public else ["dealer"])
        if not self.public:
            assert not g.visible_turns("guesser")  # the guesser is blind to the flip
        q = await g.act("guesser", kind="choice", options=["H", "T"], phase="guess",
                        prompt="Guess the coin." + (f" It shows {d.choice}." if self.public else ""))
        return Outcome(decision=q.choice, data={"coin": d.choice, "guess": q.choice})


def _coin_players():
    side = lambda req, c: soa.Action(text="HT"[c.sample_index % 2], choice="HT"[c.sample_index % 2],  # noqa: E731
                                     usage=soa.Usage(calls=1))
    return {r: soa.Player(policy=FunctionPolicy(side, label=r)) for r in ("dealer", "guesser")}


def _coin_tree(public=False):
    item = soa.TaskItem(id="coin", question="?")
    tree, eps = run_sync(expand_tree(CoinGame(public=public), item, _coin_players(),
                                     pool_sizes={"dealer": 2, "guesser": 2}, keep_episodes=True))
    return tree, eps


def test_best_of_n_cannot_select_on_a_hidden_coin():
    tree, _ = _coin_tree()
    guesser = [n for n in tree.nodes.values() if n.role == "guesser"]
    assert len(guesser) == 2 and len({n.key for n in guesser}) == 1  # one information set behind the coin
    tv = evaluate_tree(tree, {"guesser": BestOfN(2)})
    assert tv.rewards["guesser"] == pytest.approx(0.5)  # was 1.0: it picked the matching guess per branch
    assert tv.gap == pytest.approx(0.0)
    # control: when the flip is public the guesser does see it, and selection may use it
    open_tree, _ = _coin_tree(public=True)
    assert len({n.key for n in open_tree.nodes.values() if n.role == "guesser"}) == 2
    assert evaluate_tree(open_tree, {"guesser": BestOfN(2)}).rewards["guesser"] == pytest.approx(1.0)


def test_hidden_moves_against_an_optimizing_dealer_reach_the_mixed_equilibrium():
    tree, _ = _coin_tree()
    tv = evaluate_tree(tree, {"guesser": BestOfN(2), "dealer": BestOfN(2)})
    # matching pennies: neither side can do better than 1/2 - selection on the hidden coin would give 1
    assert tv.rewards["guesser"] == pytest.approx(0.5, abs=0.02)
    assert tv.rewards["dealer"] == pytest.approx(0.5, abs=0.02)
    soft = evaluate_tree(tree, {"guesser": Tilted(3.0), "dealer": Tilted(3.0)})
    assert soft.rewards["guesser"] == pytest.approx(0.5, abs=1e-6) and soft.gap < 1e-6


def test_a_shared_pool_is_sampled_and_charged_once():
    tree, eps = _coin_tree()
    assert {r: u["calls"] for r, u in tree.usage.items()} == {"dealer": 2, "guesser": 2}  # not 4 guesses
    per_role = {r: sum(e.usage[r].calls for e in eps if r in e.usage) for r in ("dealer", "guesser")}
    assert per_role == {"dealer": 2, "guesser": 2}  # the leaves still add up to the tree's usage


def test_information_sets_must_agree_on_role_and_pool():
    nodes = {"r": TreeNode(id="r", key="K1", role="a", children=["n1", "l2"]),
             "n1": TreeNode(id="n1", key="K1", role="a", children=["l1"])}
    leaves = {lid: TreeLeaf(id=lid, rewards={"a": 1.0}) for lid in ("l1", "l2")}
    with pytest.raises(ValueError, match="information set"):
        evaluate_tree(GameTree(item_id="x", mechanism="m", root="r", nodes=nodes, leaves=leaves))


# ------------------------------------------------------------------------------ label coverage

REWARDS = [1.0, 0.2, 0.1, 0.0]
CORRECT = [math.nan, 1.0, 1.0, 0.0]  # the candidate selection likes most has no label


def _pool_tree():
    leaves = {f"l{i}": TreeLeaf(id=f"l{i}", rewards={"agent": r},
                                values={} if math.isnan(v) else {"correct": v})
              for i, (r, v) in enumerate(zip(REWARDS, CORRECT))}
    nodes = {"root": TreeNode(id="root", key="K", role="agent", children=list(leaves))}
    return GameTree(item_id="x", mechanism="m", root="root", nodes=nodes, leaves=leaves)


def test_tree_values_report_their_label_coverage():
    base = evaluate_tree(_pool_tree())
    assert base.values["correct"] == pytest.approx(2 / 3) and base.coverage["correct"] == pytest.approx(0.75)
    bo64 = evaluate_tree(_pool_tree(), {"agent": BestOfN(64, "plugin")})
    assert bo64.values["correct"] == pytest.approx(1.0)  # an average over almost none of the selected mass
    assert bo64.coverage["correct"] == pytest.approx(1 - (1 - (3 / 4) ** 64), abs=1e-12)
    assert bo64.reward_coverage["agent"] == pytest.approx(1.0)


def test_optimization_grid_drops_values_without_coverage(caplog):
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        g = optimization_grid([_pool_tree()], {"agent": [1, 4, 64]}).set_index("level_agent")
    assert g.loc[1, "correct"] == pytest.approx(2 / 3) and g.loc[1, "correct_coverage"] == pytest.approx(0.75)
    for n in (4, 64):  # selection sits on the unlabelled candidate: no value, rather than 1.000 or a silent NaN
        assert math.isnan(g.loc[n, "correct"]) and g.loc[n, "correct_coverage"] < 1e-6
    assert "dropped" in caplog.text
    assert (g["gap"] == 0).all()


def test_pool_curves_report_their_label_coverage():
    import pandas as pd

    pool = pd.DataFrame({"item_id": ["q"] * 4, "reward": REWARDS, "value": CORRECT})
    out = pool_curve(pool, selections=[BestOfN(1), BestOfN(4)]).set_index("selection")
    assert out.loc["BestOfN(1, 'unbiased')", "value_coverage"] == pytest.approx(0.75)
    assert math.isnan(out.loc["BestOfN(4, 'unbiased')", "value"])
    assert out.loc["BestOfN(4, 'unbiased')", "value_coverage"] == pytest.approx(0.0)
    assert np.isfinite(out["reward"]).all()


# ------------------------------------------------------------------------------ chance moves


def test_multitask_peer_prediction_draws_peers_as_a_chance_move():
    from so_arena.mechanisms import PeerPrediction

    bundle = soa.TaskItem(id="b", question="?", context={"subitems": [
        {"id": f"s{i}", "question": f"Q{i}?", "labels": ["A", "B"]} for i in range(4)]})
    answers = {"reporter_1": "AABB", "reporter_2": "ABAB", "reporter_3": "ABBA"}
    players = {r: soa.Player(policy=FunctionPolicy(
        lambda req, c: soa.Action(text=answers[c.role][int(req.phase.split(":s")[1])],
                                  choice=answers[c.role][int(req.phase.split(":s")[1])])))
        for r in answers}
    mech = PeerPrediction(n_reporters=3, rule="multitask")
    # the random peers and comparison tasks depended on the episode id - in a game tree, on the path,
    # i.e. on the reports being rewarded; now every play of the item draws the same ones
    rewards = [run_sync(mech.run(bundle, players, episode_id=eid)).rewards for eid in ("a", "b:tree:1.0", "c")]
    assert rewards[0] == rewards[1] == rewards[2]
    other_seed = run_sync(mech.run(bundle, players, seed=1)).rewards
    assert other_seed != rewards[0]  # still random across seeds
