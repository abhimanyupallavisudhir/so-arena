"""Regressions from the second bug hunt (core): response-cache collisions between roles and between
backends, config hashes blind to closure parameters, failed ground truth frozen by resumption,
configurations pooled under one mechanism name, and silent estimator switches in optimization."""

import logging
import math
import random

import numpy as np
import pytest

import so_arena as soa
from so_arena.analysis.frames import role_frame
from so_arena.analysis.metrics import asd
from so_arena.analysis.optimization import BestOfN, GameTree, Tilted, TreeLeaf, TreeNode, evaluate_tree, optimization_grid
from so_arena.core.game import Player
from so_arena.core.ground_truth import GroundTruthScorer, StanceValue
from so_arena.core.items import binary_item
from so_arena.core.mechanism import Mechanism, Outcome, RoleSpec
from so_arena.core.policy import ActContext, BestOfNPolicy, FunctionPolicy, LLMPolicy
from so_arena.core.rewards import FunctionReward, JudgeScore, RandomAudit
from so_arena.core.runner import PlayerSpec, Profile, profile_description, run_episodes, run_sync
from so_arena.core.store import RunStore
from so_arena.core.types import GenerateOptions
from so_arena.mechanisms import Debate, Propaganda
from so_arena.models.base import FunctionModel, ProviderModel, draw_seed
from so_arena.models.cache import CachedModel, ResponseCache

# ------------------------------------------------------------------------------------ response cache


class Twins(Mechanism):
    """Two roles asked the identical question at once (as peer-prediction reporters are)."""

    name = "twins"

    def roles(self):
        return {"a": RoleSpec(name="a"), "b": RoleSpec(name="b")}

    async def protocol(self, g):
        acts = await g.simultaneous([(r, dict(kind="choice", options=["A", "B"], prompt="Which one?")) for r in "ab"])
        return Outcome(data={"answers": {r: x.choice for r, x in zip("ab", acts)}})


def test_identical_prompts_from_different_roles_are_separate_cached_draws(tmp_path):
    rng, calls = random.Random(0), {"n": 0}

    def sampler(messages, options, i):  # a temperature > 0 API: every call is a fresh draw
        calls["n"] += 1
        return f"Answer: {rng.choice('AB')}"

    model = CachedModel(FunctionModel(sampler, name="api"), ResponseCache.at(tmp_path))
    players = {r: Player(policy=LLMPolicy(model, label=r)) for r in "ab"}
    items = [soa.TaskItem(id=f"q{i}", question="?") for i in range(12)]
    cold = [run_sync(Twins().run(it, players)).outcome.data["answers"] for it in items]
    assert calls["n"] == 24
    warm = [run_sync(Twins().run(it, players)).outcome.data["answers"] for it in items]
    assert calls["n"] == 24 and warm == cold  # the cache replays each role's own draw
    assert any(a["a"] != a["b"] for a in cold)  # ... and the roles' draws were never one and the same
    # seeded backends derive the seed from sample_index: the draw offsets it, per role and decision
    assert draw_seed(GenerateOptions(draw="a|k")) != draw_seed(GenerateOptions(draw="b|k"))
    assert draw_seed(GenerateOptions()) == 0


def test_model_arguments_distinguish_cached_answers_and_episode_ids():
    step0, step900 = ProviderModel("vllm/policy", base_url="http://step0"), ProviderModel("vllm/policy", base_url="http://step900")
    assert step0.identity != step900.identity and step0.identity.startswith("vllm/policy")
    assert ProviderModel("vllm/policy", api_key="secret-1").identity == ProviderModel("vllm/policy").identity  # credentials
    p0 = Profile(name="rl", players={"agent": LLMPolicy(step0)})
    p9 = Profile(name="rl", players={"agent": LLMPolicy(step900)})
    assert profile_description(p0) != profile_description(p9)  # resume would reuse step-0 episodes for step 900


# ------------------------------------------------------------------------------------ config hashes


def test_closure_parameters_of_reward_rules_and_oracles_change_the_config_hash():
    def length_penalty(lam):
        return FunctionReward(lambda ep: {"agent": -lam * len(ep.turns[-2].text)}, name="len_pen")

    assert Propaganda(reward=length_penalty(1.0)).config_hash() != Propaganda(reward=length_penalty(100.0)).config_hash()
    assert Propaganda(reward=length_penalty(1.0)).config_hash() == Propaganda(reward=length_penalty(1.0)).config_hash()

    def noisy(noise):
        return lambda ep: {"agent": -noise}

    audits = {n: Propaganda(reward=RandomAudit(JudgeScore("log"), noisy(n), p=1.0, mode="ipw")).config_hash()
              for n in (0.0, 5.0)}
    assert audits[0.0] != audits[5.0]


# ------------------------------------------------------------------------------------ resumption


def test_failed_ground_truth_is_scored_again_when_the_store_is_resumed(tmp_path):
    state = {"broken": True}

    class Flaky(GroundTruthScorer):
        name = "stance_value"

        async def score(self, ep, item, ctx=None):
            if state["broken"] and item.id == "q0":
                raise RuntimeError("sandbox unavailable")
            return await StanceValue().score(ep, item, ctx)

    items = [binary_item(f"q{i}", f"Q{i}", "right", "wrong", shuffle_seed=i) for i in range(4)]
    judge = soa.ScriptedPolicy('{"A": 0.7, "B": 0.3}')
    profs = [Profile(name=s, players={"agent": PlayerSpec(policy=soa.ScriptedPolicy("argument"), stance=s), "judge": judge})
             for s in ("true", "false")]
    store = RunStore(tmp_path / "run")
    eps = run_sync(run_episodes(Propaganda(), items, profs, store=store, ground_truth=[Flaky()]))
    q0 = [e for e in eps if e.item_id == "q0"]
    assert {e.gt_status for e in q0} == {"error"}  # was "known", and never scored again
    state["broken"] = False
    eps = run_sync(run_episodes(Propaganda(), items, profs, store=store, ground_truth=[Flaky()]))
    q0 = [e for e in eps if e.item_id == "q0"]
    assert {e.gt_status for e in q0} == {"known"} and all("error_stance_value" not in e.ground_truth for e in q0)
    assert asd(role_frame(eps), n_boot=20)["n_items"].tolist() == [4]


# ------------------------------------------------------------------------------------ configurations


def test_configurations_of_one_mechanism_are_separate_analysis_rows():
    from so_arena.domains.synthetic import SyntheticPersuasion, synthetic_arguer, synthetic_judge
    from so_arena.samplers.arms import ASDExperiment

    dom = SyntheticPersuasion(n_items=12, seed=1)
    items = dom.load()

    def run(mechs):
        exp = ASDExperiment(mechs, items, agent=synthetic_arguer(), fixtures={"judge": synthetic_judge()}, ctx=dom.context())
        exp.run()
        return asd(exp.frame(), n_boot=20).set_index("mechanism")["asd"]

    alone = [run([Debate(rounds=1)]), run([Debate(rounds=1, transform="prob")])]
    assert list(alone[0].index) == ["debate"]  # a single configuration keeps its plain name
    both = run([Debate(rounds=1), Debate(rounds=1, transform="prob")])
    assert sorted(both.index) == ["debate(transform=log)", "debate(transform=prob)"]  # were pooled into one row
    assert both["debate(transform=log)"] == pytest.approx(alone[0]["debate"])
    assert both["debate(transform=prob)"] == pytest.approx(alone[1]["debate"])


def test_saving_one_configurations_trees_keeps_the_others(tmp_path):
    from so_arena.samplers.pools import load_trees, save_trees

    def tree(item, config):
        leaf = TreeLeaf(id="l", rewards={"a": 1.0})
        return GameTree(item_id=item, mechanism="debate", config=config, root="l", leaves={"l": leaf})

    save_trees(tmp_path, [tree("q1", "cfgA")], mechanism="debate", config="cfgA")
    save_trees(tmp_path, [tree("q1", "cfgB")], mechanism="debate", config="cfgB")  # used to delete cfgA's trees
    assert sorted(t.config for t in load_trees(tmp_path)) == ["cfgA", "cfgB"]
    save_trees(tmp_path, [tree("q2", "cfgA")], mechanism="debate", config="cfgA")  # replaces cfgA's only
    assert sorted((t.config, t.item_id) for t in load_trees(tmp_path, "debate")) == [("cfgA", "q2"), ("cfgB", "q1")]


# ------------------------------------------------------------------------------------ optimization


def test_best_of_n_beyond_the_pool_size_says_it_uses_the_plug_in_estimate(caplog):
    from so_arena.analysis import optimization

    optimization._WARNED_PLUGIN.clear()
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        p = BestOfN(5, "unbiased").probs(np.array([0.0, 0.0, 1.0]))
    assert p == pytest.approx(BestOfN(5, "plugin").probs(np.array([0.0, 0.0, 1.0])))
    assert "plug-in" in caplog.text and "pool of 3" in caplog.text


def _stage(A, B):
    nodes = {"root": TreeNode(id="root", key="kA", role="a", group="g", children=[f"B{i}" for i in range(len(A))])}
    leaves = {}
    for i in range(len(A)):
        nodes[f"B{i}"] = TreeNode(id=f"B{i}", key="kB", role="b", group="g", children=[f"L{i}{j}" for j in range(len(A[i]))])
        for j in range(len(A[i])):
            leaves[f"L{i}{j}"] = TreeLeaf(id=f"L{i}{j}", rewards={"a": float(A[i][j]), "b": float(B[i][j])})
    return GameTree(item_id="x", mechanism="m", root="root", nodes=nodes, leaves=leaves, roles=["a", "b"])


def test_fictitious_play_reports_when_it_does_not_settle(caplog):
    best = {"a": Tilted(float("inf")), "b": Tilted(float("inf"))}
    shapley = _stage(np.array([[0.01, 1, 0], [0, 0, 1], [1, 0, 0]]), np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))
    assert evaluate_tree(shapley, best).gap > 0.05  # fictitious play cycles here: the values depend on fp_iters
    pennies = _stage(np.array([[3.0, 0], [0, 1]]), -np.array([[3.0, 0], [0, 1]]))
    tv = evaluate_tree(pennies, best)
    assert tv.rewards["a"] == pytest.approx(0.75, abs=1e-3) and tv.gap < 0.01  # converges: small gap in reward units
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        optimization_grid([shapley], {"a": [3], "b": [3]})
    assert "did not settle" in caplog.text


def test_best_of_n_policy_ranks_nan_scores_last():
    scores = [math.nan, 0.9, 0.5, 0.99]  # e.g. a preview judge whose parse failed on candidate 0
    pol = BestOfNPolicy(FunctionPolicy(lambda req, c: f"candidate {c.sample_index}"), 4,
                        lambda req, a, c: scores[int(a.text.split()[-1])])
    act = soa.core.actions.ActionRequest(kind="text")
    assert run_sync(pol.act(act, ActContext(role="agent"))).text == "candidate 3"
    scores[:] = [0.2, math.nan, 0.1, 0.05]
    assert run_sync(pol.act(act, ActContext(role="agent"))).text == "candidate 0"
    scores[:] = [math.nan] * 4
    assert run_sync(pol.act(act, ActContext(role="agent"))).text == "candidate 0"  # all unscored: the first
