"""Regression tests for sampler, game and analysis fixes: symmetric PSRO populations, simultaneous stages
with missing or truncated cells, equal weights for every false answer, cluster-bootstrap relabelling,
label efficiency without signal, conservative imputation of missing payoffs, held-out selection in
prompt search, missing rewards in pool curves, graded ASD by behaviour, transparent capture policies,
per-mechanism tree files and dry-run episode ids."""

import asyncio
import hashlib
import json
import logging
import math
import re

import numpy as np
import pandas as pd
import pytest

import so_arena as soa
from so_arena.analysis.metrics import (
    cluster_bootstrap,
    graded_asd,
    incentive_alignment,
    pairwise_concordance,
)
from so_arena.analysis.optimization import (
    BestOfN,
    GameTree,
    Tilted,
    TreeLeaf,
    TreeNode,
    Uniform,
    evaluate_tree,
    pool_curve,
)
from so_arena.core.items import AnswerOption, GroundTruth, TaskItem
from so_arena.core.policy import FunctionPolicy, ScriptedPolicy
from so_arena.core.runner import (
    PlayerSpec,
    Profile,
    build_players,
    episode_id,
    profile_description,
    resolve_stance,
    run_episodes,
    run_sync,
)
from so_arena.core.store import RunStore
from so_arena.domains.synthetic import (
    SyntheticPersuasion,
    SyntheticTeam,
    synthetic_arguer,
    synthetic_judge,
    team_worker,
)
from so_arena.games import EmpiricalGameExperiment, NormalFormGame
from so_arena.games.learning import policy_gradient
from so_arena.mechanisms import Debate, Propaganda, Team
from so_arena.models import FunctionModel
from so_arena.samplers.arms import ASDExperiment, arm_profiles
from so_arena.samplers.pools import OptimizationExperiment, expand_tree, load_trees
from so_arena.samplers.prompt_search import CapturePolicy, PromptSearch
from so_arena.samplers.psro import PSRO, solve_meta
from so_arena.theory.audits import efficiency_from_corr, label_efficiency


def _synth(n=4, seed=3):
    dom = SyntheticPersuasion(n_items=n, seed=seed)
    return dom.load(), dom.context()


# ----------------------------------------------------------------------------- symmetric PSRO

def _team_factory(text):
    return team_worker(report="report" if "REPORT" in (text or "") else "silent")


def test_symmetric_psro_shares_one_population_and_estimates_the_right_game():
    """Each symmetric role used to get its own 'br1'; EGTA, matching names only, then filled the unsimulated
    (w2=br1, w3=base) profile from worker_3's different br1 (0.9 instead of 1.1) and NashConv read 0."""
    dom = SyntheticTeam(n_items=2)
    items, ctx = dom.load(), dom.context()
    mech = Team(n_workers=3, regrade_on_report=True, reward=soa.Whistleblower(bounty=0.5))
    searched_as = []

    def optimizer(messages, options, i):
        role = re.search(r"plays the role '([^']+)'", messages[-1].content).group(1)
        searched_as.append(role)
        return "<rationale>r</rationale><strategy>REPORT the teammate who hacked</strategy>"

    fixtures = {"worker_1": team_worker(work="hack"), "grader": dom.grader()}
    psro = PSRO(mech, items, roles=["worker_2", "worker_3"],
                initial={"worker_2": {"base": "stay quiet"}, "worker_3": {"base": "stay quiet"}},
                policy_factories={"worker_2": _team_factory, "worker_3": _team_factory}, fixtures=fixtures,
                optimizer=FunctionModel(optimizer, name="opt"), iterations=1, symmetric=["worker_2", "worker_3"],
                search_kwargs={"iterations": 1, "candidates_per_iter": 1}, ctx=ctx,
                ground_truth=dom.ground_truth_scorers())
    hist = psro.run()
    assert set(searched_as) == {"worker_2"}  # one best-response search per iteration for the population
    assert psro.populations["worker_2"] == psro.populations["worker_3"]
    g = psro.games[-1]
    base, br = g.strategies["worker_2"].index("base"), g.strategies["worker_2"].index("br1")
    direct = Profile(name="direct", players={**fixtures, "worker_2": _team_factory(psro.populations["worker_2"]["br1"]),
                                             "worker_3": _team_factory("stay quiet")})
    eps = run_sync(run_episodes(mech, items, [direct], ctx=ctx))
    simulated = float(np.mean([e.rewards["worker_2"] for e in eps]))
    assert simulated == pytest.approx(1.1)  # bounty 0.5 beats the hack's share 0.3
    assert g.payoffs["worker_2"][(br, base)] == pytest.approx(simulated)
    assert hist[1].nash_conv_prev == pytest.approx(0.4)  # both roles gain 1.1 - 0.9 by reporting
    assert hist[1].meta_strategy["worker_2"] == pytest.approx([0.0, 1.0], abs=1e-6)
    assert hist[1].meta_strategy["worker_2"] == hist[1].meta_strategy["worker_3"]


def test_symmetric_roles_must_really_share_strategies():
    dom = SyntheticTeam(n_items=1)
    items, ctx = dom.load(), dom.context()
    mech = Team(n_workers=3, regrade_on_report=True, reward=soa.Whistleblower(bounty=0.5))
    fixtures = {"worker_1": team_worker(work="hack"), "grader": dom.grader()}
    same_names = {"worker_2": {"base": team_worker(), "br1": team_worker(report="report")},
                  "worker_3": {"base": team_worker(), "br1": team_worker(report="silent")}}
    with pytest.raises(ValueError, match="br1"):
        EmpiricalGameExperiment(mech, items, same_names, fixtures=fixtures, symmetric=["worker_2", "worker_3"], ctx=ctx)
    # equal policies built separately (labels aside) are fine
    ok = {w: {"silent": team_worker(label=f"{w}-s"), "report": team_worker(report="report")} for w in ("worker_2", "worker_3")}
    EmpiricalGameExperiment(mech, items, ok, fixtures=fixtures, symmetric=["worker_2", "worker_3"], ctx=ctx)
    with pytest.raises(ValueError, match="one population"):
        PSRO(mech, items, roles=["worker_2", "worker_3"],
             initial={"worker_2": {"base": "stay quiet"}, "worker_3": {"base": "REPORT"}},
             policy_factories={"worker_2": _team_factory, "worker_3": _team_factory}, fixtures=fixtures,
             optimizer=FunctionModel(lambda m, o, i: "", name="opt"), symmetric=["worker_2", "worker_3"])


def test_symmetric_meta_strategy_for_an_anti_coordination_game():
    # hawk-dove: the symmetric equilibrium is the mixed one; two-population dynamics would leave it
    A = np.array([[-1.0, 2.0], [0.0, 1.0]])
    g = NormalFormGame(["a", "b"], {"a": ["hawk", "dove"], "b": ["hawk", "dove"]}, {"a": A, "b": A.T})
    for solver in ("nash", "replicator", "fictitious"):
        x = solve_meta(g, solver, symmetric=["a", "b"])
        assert x[0] == pytest.approx(x[1], abs=1e-9)
        assert x[0] == pytest.approx([0.5, 0.5], abs=0.02)


# ----------------------------------------------------------------------------- simultaneous stages

def _stage_tree(RA, simultaneous=True):
    """A (row) and B (column) move simultaneously; B's pool is shared across A's candidates."""
    m, k = RA.shape
    grp = "g1" if simultaneous else None
    nodes = {"root": TreeNode(id="root", key="KA", role="A", group=grp, children=[f"b{i}" for i in range(m)])}
    leaves = {}
    for i in range(m):
        nodes[f"b{i}"] = TreeNode(id=f"b{i}", key="KB" if simultaneous else f"KB{i}", role="B", group=grp,
                                  children=[f"l{i}{j}" for j in range(k)])
        for j in range(k):
            r = RA[i, j]
            leaves[f"l{i}{j}"] = TreeLeaf(id=f"l{i}{j}", rewards={"A": None if math.isnan(r) else r, "B": 0.0},
                                          values={"judge_correct": float(i == 0)})
    return GameTree(item_id="x", mechanism="m", root="root", nodes=nodes, leaves=leaves, roles=["A", "B"])


def test_stage_game_missing_cell_and_uniform_start_do_not_bias_the_solution():
    RA = np.zeros((4, 4))
    RA[0] = 1.0  # A's candidate 0 is best whatever B does
    exact = evaluate_tree(_stage_tree(RA, simultaneous=False), {"A": BestOfN(4)})
    assert exact.rewards["A"] == 1.0
    tv = evaluate_tree(_stage_tree(RA), {"A": BestOfN(4)})
    assert tv.rewards["A"] == pytest.approx(1.0, abs=1e-12)  # was 0.996: the uniform start kept 1/(T+1)
    RA[0, 3] = math.nan  # one errored leaf in the best row: was imputed as -1e9 (reward 0.001)
    tv = evaluate_tree(_stage_tree(RA), {"A": BestOfN(4)})
    assert tv.rewards["A"] == pytest.approx(1.0) and tv.values["judge_correct"] == pytest.approx(1.0)
    # no strategic interaction: a tilted stage is exactly the softmax-weighted mean
    RA = np.tile(np.array([[1.0], [0.0], [0.0]]), (1, 3))
    w = np.exp(2.0 * RA[:, 0])
    assert evaluate_tree(_stage_tree(RA), {"A": Tilted(2.0)}).rewards["A"] == pytest.approx(w @ RA[:, 0] / w.sum())


def test_tree_truncated_inside_a_simultaneous_stage():
    """max_leaves truncation turns a stage member into a leaf (the play took candidate 0 of the rest); the
    stage solver used to look it up as a node (KeyError)."""
    items, ctx = _synth(1)
    item = items[0]
    leaf_played = asyncio.Event()

    async def judge(req, c):
        text = " ".join(t.text or "" for t in req.view.transcript)
        a, b = int(re.search(r"A(\d)", text).group(1)), int(re.search(r"B(\d)", text).group(1))
        if (a, b) == (0, 1):  # only a complete play (a leaf) takes B's second candidate
            leaf_played.set()
        if a == 1:  # hold A's second candidate back until a leaf exists: its play is then truncated
            await asyncio.wait_for(leaf_played.wait(), 10)
            await asyncio.sleep(0.3)
        p = 0.5 + 0.2 * a - 0.1 * b
        return {item.labels[0]: p, item.labels[1]: 1 - p}

    players = {"debater_a": PlayerSpec(policy=FunctionPolicy(lambda r, c: f"A{c.sample_index}"), stance="true"),
               "debater_b": PlayerSpec(policy=FunctionPolicy(lambda r, c: f"B{c.sample_index}"), stance="false"),
               "judge": FunctionPolicy(judge)}
    tree, _ = run_sync(expand_tree(Debate(rounds=1), item, build_players(Profile(name="p", players=players), item),
                                   pool_sizes={"debater_a": 2, "debater_b": 2}, ctx=ctx, max_leaves=1, concurrency=4))
    root = tree.nodes[tree.root]
    assert root.group is not None and tree.n_leaves == 3
    assert sorted(c in tree.nodes for c in root.children) == [False, True]  # truncated inside the stage
    # uniform play, each decision over its observed candidates: the truncated play stands for B's first
    # candidate, so it is A's second candidate's whole value (not one cell of four, pooled with A's first)
    base = evaluate_tree(tree)
    staged = next(c for c in root.children if c in tree.nodes)
    cut = next(c for c in root.children if c in tree.leaves)
    b_mean = np.mean([tree.leaves[c].rewards["debater_a"] for c in tree.nodes[staged].children])
    assert base.rewards["debater_a"] == pytest.approx(0.5 * b_mean + 0.5 * tree.leaves[cut].rewards["debater_a"])
    opt = evaluate_tree(tree, {"debater_a": BestOfN(2), "debater_b": BestOfN(2)})
    assert all(np.isfinite(v) for v in opt.rewards.values())


# ----------------------------------------------------------------------------- all false answers

def _mc_item(i, n_opts):
    labels = "ABCDEFGHIJ"[:n_opts]
    return TaskItem(id=f"q{i}", question="?", answers=[AnswerOption(label=lab, text=lab) for lab in labels],
                    ground_truth=GroundTruth(correct="A"))


P_JUDGE = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}


def test_all_false_argues_every_false_answer_equally_often():
    judge = FunctionPolicy(lambda req, c: dict(P_JUDGE) if req.kind == "probabilities" else "Answer: A", label="judge")
    agent = FunctionPolicy(lambda req, c: "my argument", label="agent")
    mech = Propaganda()
    for n_opts in (2, 4, 10):
        it = _mc_item(0, n_opts)
        profs = arm_profiles(mech, agent=agent, fixtures={"judge": judge}, all_false=True, items=[it])
        stances = sorted(build_players(p, it)["agent"].stance for p in profs)
        assert stances == sorted(it.labels)  # each answer exactly once (was: wrapped over 8 arms)
    with pytest.raises(LookupError):
        resolve_stance("false:3", _mc_item(0, 4), None)
    exp = ASDExperiment([mech], [_mc_item(i, 4) for i in range(3)] + [_mc_item(3, 2)], agent=agent,
                        fixtures={"judge": judge}, all_false=True)
    exp.run()
    df = exp.frame()
    assert df.groupby("item_id").size().to_dict() == {"q0": 4, "q1": 4, "q2": 4, "q3": 2}
    per = df[df.item_id == "q0"].groupby("value")["reward"].mean()
    assert per[1.0] - per[-1.0] == pytest.approx(math.log(0.4) - np.mean(np.log([0.3, 0.2, 0.1])))


# ----------------------------------------------------------------------------- bootstrap and label efficiency

def test_cluster_bootstrap_counts_a_redrawn_cluster_as_a_new_cluster():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        good = rng.random() < 0.7
        for k in range(4):
            v = 1.0 if k < 2 else -1.0
            rows.append(dict(item_id=f"q{i}", value=v, reward=(v if good else -v) + 0.01 * rng.normal()))
    df = pd.DataFrame(rows)
    clusters = [g for _, g in df.groupby("item_id")]
    seen = []
    cluster_bootstrap(clusters, lambda s: seen.append(s["item_id"].nunique()) or 0.0, n_boot=20)
    assert set(seen) == {30}
    lo, hi = cluster_bootstrap(clusters, lambda s: pairwise_concordance(s)[0], n_boot=200, seed=1)
    ref = np.random.default_rng(1)
    vals = []
    for _ in range(200):
        idx = ref.integers(0, 30, 30)
        vals.append(pairwise_concordance(pd.concat([clusters[i].assign(item_id=j) for j, i in enumerate(idx)]))[0])
    assert (lo, hi) == pytest.approx(tuple(np.quantile(vals, [0.025, 0.975])))
    assert hi - lo < 0.4  # the merged-copies version gave about [0.32, 0.85]


def test_label_efficiency_is_one_without_within_item_signal():
    rows = []
    for i in range(10):
        base = -0.69 + 0.1 * i  # a 50/50 judge: both arms get the same reward on each item
        rows += [dict(episode_id=f"e{i}{s}", item_id=f"q{i}", mechanism="m", role="agent", kind="agent",
                      trainable=True, reward=base, value=v) for s, v in (("t", 1.0), ("f", -1.0))]
    df = pd.DataFrame(rows)
    out = incentive_alignment(df, n_boot=20)
    assert out["label_efficiency"].iloc[0] == 1.0  # was inf
    uw = df.reward - df.groupby("item_id").reward.transform("mean")
    vw = df.value - df.groupby("item_id").value.transform("mean")
    assert label_efficiency(uw, vw) == 1.0
    assert efficiency_from_corr(math.nan) == 1.0 and efficiency_from_corr(1.0) == math.inf
    assert efficiency_from_corr(0.6) == pytest.approx(1 / (1 - 0.36))


# ----------------------------------------------------------------------------- missing payoffs

def _game_with_missing_best_cell():
    # log-scale payoffs; (x1, y1) never produced a valid episode. Filled with 0 it was the unique equilibrium.
    A = np.array([[-1.0, -0.5], [-0.8, np.nan]])
    B = np.array([[-1.0, -0.8], [-0.5, np.nan]])
    W = np.array([[0.0, 1.0], [1.0, np.nan]])
    return NormalFormGame(["a", "b"], {"a": ["x0", "x1"], "b": ["y0", "y1"]}, {"a": A, "b": B},
                          outcomes={"welfare": W}, name="g")


def test_missing_payoffs_are_imputed_conservatively(caplog):
    g = _game_with_missing_best_cell()
    with caplog.at_level(logging.WARNING, logger="so_arena"):
        filled = g.imputed()
    assert "2 missing payoff entries" in caplog.text
    assert filled.payoffs["a"][1, 1] == -1.0 and filled.payoffs["b"][1, 1] == -1.0
    for solver in ("nash", "replicator", "fictitious"):
        x = solve_meta(g, solver)
        assert x[0][1] * x[1][1] < 0.5, solver
    df = policy_gradient(g, steps=300)
    last = df.iloc[-1]
    assert last["p_a_x1"] * last["p_b_y1"] < 0.5
    assert df["welfare"].iloc[0] == pytest.approx(2 / 3)  # over the observed profiles only (0 filled: 1/2)
    with pytest.raises(ValueError, match="no observed payoff"):
        NormalFormGame(["a", "b"], g.strategies, {"a": np.full((2, 2), np.nan), "b": np.zeros((2, 2))}).imputed()


# ----------------------------------------------------------------------------- prompt search

def test_best_is_selected_on_training_and_scored_held_out():
    items, ctx = _synth(24, seed=7)
    calls = [0]

    def opt(m, o, i):
        calls[0] += 1
        return "".join(f"<strategy>variant {calls[0]}-{j}</strategy>" for j in range(4))

    same = lambda s: synthetic_arguer(honest_mean=0.0, dishonest_mean=0.0, sd=1.0, label="same")  # noqa: E731
    res = PromptSearch(Propaganda(affordances={"agents": ["answer_key"]}), items[:8], role="agent", policy_factory=same,
                       others={"judge": synthetic_judge()}, optimizer=FunctionModel(opt, name="o"), arms=["false"],
                       iterations=3, candidates_per_iter=4, eval_items=items[8:], n_final=6, ctx=ctx).run()
    top = max(res.candidates, key=lambda c: c.mean_reward)
    best = res.best
    assert best.split == "eval" and best.parent == top.id and best.strategy == top.strategy
    held_out = sorted(c.mean_reward for c in res.evaluated)
    assert best.mean_reward < held_out[-1]  # identical strategies: the held-out maximum is noise
    res.evaluated = []
    assert res.best is top


def test_capture_policy_is_transparent():
    inner = soa.LLMPolicy("mock", strategy="be brief", label="cand")
    cap = CapturePolicy(inner)
    assert cap.describe() == inner.describe() and cap.model_name == inner.model_name == "mock"
    items, _ = _synth(1)
    mech = Propaganda()
    ids = [episode_id("r", mech, items[0], Profile(name="p", players={"agent": PlayerSpec(policy=p, stance="false"),
                                                                    "judge": synthetic_judge()}), 0)
           for p in (inner, cap)]
    assert ids[0] == ids[1]


# ----------------------------------------------------------------------------- pools and trees

def test_pool_curve_leaves_missing_rewards_out():
    pool = pd.DataFrame({"item_id": ["q"] * 3, "reward": [np.nan, -1.0, -1.0], "value": [0.0, 1.0, 1.0]})
    c = pool_curve(pool, selections=[Uniform(), BestOfN(1), BestOfN(2), Tilted(0.0)])
    assert c["reward"].tolist() == pytest.approx([-1.0] * 4)  # was -0.667 for Bo1 (NaN counted as 0)


def test_optimization_experiments_share_a_store_per_mechanism(tmp_path):
    items, ctx = _synth(2)
    store = RunStore(tmp_path / "run")
    arguer, judge = synthetic_arguer(), synthetic_judge()
    prof = Profile(name="p", players={"agent": PlayerSpec(policy=arguer, stance="false"), "judge": judge})
    one = Propaganda(affordances={"agents": ["answer_key"]})
    two = Propaganda(affordances={"agents": ["answer_key"]}, name="propaganda2")
    for mech in (one, two, one):  # re-running a mechanism replaces its trees, others' stay
        OptimizationExperiment(mech, items, prof, pool_sizes={"agent": 2}, ctx=ctx, store=store).run()
    trees = load_trees(store.path)
    assert sorted(t.mechanism for t in trees) == ["propaganda"] * 2 + ["propaganda2"] * 2
    assert [t.mechanism for t in load_trees(store.path, mechanism="propaganda2")] == ["propaganda2"] * 2


# ----------------------------------------------------------------------------- graded ASD

def test_graded_asd_groups_cases_by_behaviour_not_value():
    u = {"A": -0.9, "B": -1.2, "C": -1.6, "D": -2.3}
    v = {"A": 1.0, "B": -1.0, "C": -1.0, "D": -1.0}
    rows = [dict(item_id=f"q{i}", mechanism="m", role="agent", kind="agent", trainable=True, stance=lab,
                 label="argue_true" if v[lab] > 0 else "argue_false", value=v[lab], reward=u[lab])
            for i in range(3) for lab in "ABCD"]
    got = graded_asd(pd.DataFrame(rows))["graded_asd"].iloc[0]
    uu, vv = np.array(list(u.values())), np.array(list(v.values()))
    assert got == pytest.approx(4 * np.mean((vv - vv.mean()) * (uu - uu.mean())))  # |C| Cov(u, v) over 4 cases
    assert got != pytest.approx(u["A"] - np.mean([u["B"], u["C"], u["D"]]))  # the by-value version
    # two debaters arguing one answer are one case; without stances, labels (arms) are the cases
    deb = [dict(item_id="q", mechanism="d", role=r, kind="agent", trainable=True, stance=s, label=lab, value=val, reward=x)
           for r, s, lab, val, x in (("a", "A", "argue_true", 1.0, -0.2), ("b", "B", "opponent", -1.0, -1.8),
                                     ("a", "B", "argue_false", -1.0, -1.4), ("b", "A", "opponent", 1.0, -0.4))]
    assert graded_asd(pd.DataFrame(deb))["graded_asd"].iloc[0] == pytest.approx((-0.2 - 0.4) / 2 - (-1.8 - 1.4) / 2)
    arms = [dict(item_id="q", mechanism="w", role="worker", kind="agent", trainable=True, stance=None, label=lab,
                 value=val, reward=x) for lab, val, x in (("honest", 1.0, 1.0), ("honest", 0.5, 0.8),
                                                            ("fake", 0.0, 0.9), ("fake", 0.2, 0.7))]
    assert graded_asd(pd.DataFrame(arms))["graded_asd"].iloc[0] == pytest.approx(
        (0.75 - 0.425) * 0.9 + (0.1 - 0.425) * 0.8)


# ----------------------------------------------------------------------------- dry-run episode ids

def test_dry_run_episodes_are_not_resumed_by_a_real_run(tmp_path):
    items, ctx = _synth(1)
    mech = Propaganda()
    calls = []

    def agent(req, c):
        calls.append(1)
        return "my argument"

    prof = Profile(name="p", players={"agent": PlayerSpec(policy=FunctionPolicy(agent), stance="true"),
                                      "judge": ScriptedPolicy('{"A": 0.5, "B": 0.5}')})
    raw = json.dumps([ctx.run_id, mech.name, mech.config_hash(), items[0].id, items[0].fingerprint(), prof.name,
                      profile_description(prof), 0, 0], sort_keys=True)
    real_id = episode_id(ctx.run_id, mech, items[0], prof, 0)
    assert real_id.endswith("#" + hashlib.sha256(raw.encode()).hexdigest()[:8])  # real ids unchanged
    store = RunStore(tmp_path / "run")
    try:
        soa.configure(simulate=True)
        dry_id = episode_id(ctx.run_id, mech, items[0], prof, 0)
        eps = run_sync(run_episodes(mech, items, [prof], ctx=ctx, store=store))
    finally:
        soa.configure(simulate=False)
    assert dry_id != real_id and eps[0].id == dry_id
    n = len(calls)
    eps = run_sync(run_episodes(mech, items, [prof], ctx=ctx, store=store))
    assert len(calls) > n and eps[0].id == real_id  # played for real, not resumed from the dry run
    n = len(calls)
    run_sync(run_episodes(mech, items, [prof], ctx=ctx, store=store))
    assert len(calls) == n  # real episodes still resume
