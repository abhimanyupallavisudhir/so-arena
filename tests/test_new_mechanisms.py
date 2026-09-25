import asyncio
import json

import pytest

import oversight_arena as oa
from oversight_arena.analysis import asd, bon_curve, bon_kl
from oversight_arena.channels import EvidencePolicy, FunctionProbe
from oversight_arena.domains.synthetic import HiddenBits
from oversight_arena.mechanisms import Comparison, Monitoring, OpenDebate, ProposerCritic, ProverVerifier
from oversight_arena.sim import BayesianBitJudge, BitAdvocate

DOM = HiddenBits(n_tasks=16)
AGENTS = {"kind:judge": BayesianBitJudge(trust=0.8), "*": BitAdvocate(claims=3)}


def run(mech, profiles):
    res = oa.Experiment(DOM, mech, AGENTS, profiles, progress=False).run()
    assert not res.errors, res.errors[0].error
    return res


def test_open_debate_rewards_choosing_the_defensible_side():
    res = run(OpenDebate(rounds=2, evidence=EvidencePolicy(budget=4)),
              [oa.Profile.of(label="truth", debater_a=oa.HONEST), oa.Profile.of(label="lie", debater_a=oa.DISHONEST)])
    for r in res.records:
        choice = r.transcript.by_role("debater_a")[0]
        assert choice.step == "choose" and not choice.evidence  # the choice is not verified / not public
        assert r.outcome["positions"]["debater_a"] == r.bound["debater_a"].target
    assert asd(res.df(trainable_only=True), roles="debater_a").iloc[0]["asd"] > 0.5


def test_comparison_and_prover_verifier_and_proposer_critic_with_exact_judges():
    res = run(Comparison(evidence=EvidencePolicy(budget=4)), oa.Stances(roles=["candidate_1"]))
    assert all(abs(sum(r.outcome["preference"].values()) - 1) < 1e-9 for r in res.records)
    assert asd(res.df(trainable_only=True), roles="candidate_1").iloc[0]["asd"] >= 0
    res = run(ProverVerifier(evidence=EvidencePolicy(budget=4)), oa.Stances(roles=["prover"]))
    modes = {r.outcome["mode"] for r in res.records}
    assert modes == {"helpful", "sneaky"} and all({"prover", "verifier"} <= set(r.rewards) for r in res.records)
    assert all(any(c.channel == "label" for c in r.channels) for r in res.records)
    res = run(ProposerCritic(evidence=EvidencePolicy(budget=4)), oa.Stances(roles=["proposer"]))
    assert asd(res.df(trainable_only=True), roles="proposer").iloc[0]["asd"] > 0


def test_function_probe_is_not_ground_truth():
    probe = FunctionProbe(fn=lambda entries: 0.9 if any("(A)" in e.content for e in entries) else 0.1)
    res = oa.Experiment(DOM, Monitoring(monitor="probe", probe=probe, grader=False), {"*": BitAdvocate()},
                        oa.Stances(roles=["actor"]), progress=False).run()
    assert {r.outcome["probe"] for r in res.records} == {0.1, 0.9}
    assert not any(r.channels for r in res.records)


def test_sweep_and_labels():
    from oversight_arena.experiment.sweep import label
    from oversight_arena.mechanisms import Propaganda

    def make(budget):
        return oa.Experiment(HiddenBits(n_tasks=6), Propaganda(evidence=EvidencePolicy(budget=budget)), AGENTS, oa.Stances(),
                             progress=False)

    t = oa.sweep(make, {"budget": [0, 4]})
    assert list(t["budget"]) == [0, 4] and "asd" in t and "accuracy" in t
    assert label(EvidencePolicy(budget=3)) == "EvidencePolicy(budget=3.0)"


def test_bon_curve_has_kl_axis():
    import math

    res = oa.Experiment(DOM, OpenDebate(rounds=1), AGENTS, oa.Seeds(n=4, strategies={"debater_a": oa.Strategy(
        name="base", params={"sample": {"claims": (1, 4), "lie_rate": (0, 1)}})}), progress=False).run()
    c = bon_curve(res, roles="debater_a", n_values=[1, 4])
    assert list(c["kl"]) == [0.0, pytest.approx(math.log(4) - 0.75)] and bon_kl(1) == 0.0


def test_opponent_aware_proposer_sees_opponents():
    from oversight_arena.elicitation import Evaluator, LLMProposer, PromptOptimizer
    from oversight_arena.mechanisms import Debate
    from oversight_arena.models import FunctionModel

    prompts = []

    def fake(messages, config, tools, sample):
        prompts.append(messages[-1].content)
        return json.dumps({"instructions": ["be precise"]})

    opp = oa.Strategy(name="opp", instructions="OPPONENT PLAN: cite three bits")
    ev = Evaluator(DOM, Debate(rounds=1), AGENTS, "debater_a", tasks=DOM.tasks()[:2], others={"debater_b": opp})
    asyncio.run(PromptOptimizer(ev, LLMProposer(FunctionModel(fake), show_opponents=True), iterations=1, per_iter=1).run())
    assert "OPPONENT PLAN" in prompts[0]


def test_artifact_archetypes():
    from oversight_arena.elicitation import ARTIFACT_ARCHETYPES, artifact_library

    lib = artifact_library()
    assert len(lib) == len(ARTIFACT_ARCHETYPES) and {s.tags["honest"] for s in lib} == {True, False}


def test_game_tree_profiles_nest_samples():
    from collections import defaultdict

    from oversight_arena.analysis.bon import tree_value, trees_from_results
    from oversight_arena.experiment.profiles import GameTree

    base = oa.Strategy(name="base", params={"sample": {"claims": (1, 5), "lie_rate": (0.0, 1.0), "side_error": 0.4}})
    src = GameTree(levels=[("proposer", "proposal", 3), ("critic", "critique", 2), ("proposer", "rebuttal", 2)],
                   strategies={"proposer": base, "critic": base})
    dom = HiddenBits(n_tasks=4)
    res = oa.Experiment(dom, ProposerCritic(critiques=1, rebuttal=True, evidence=EvidencePolicy(budget=3)), AGENTS, src,
                        progress=False).run()
    assert len(res) == 4 * 12 and not res.errors
    proposals = defaultdict(set)
    for r in res.records:
        proposals[(r.task_id, r.bound["proposer"].seed_for("proposal"))].add(r.transcript.entries[0].content)
    assert all(len(v) == 1 for v in proposals.values())  # proposal j is shared by all its descendants
    trees = trees_from_results(res, [("proposer", "proposal"), ("critic", "critique"), ("proposer", "rebuttal")])
    assert [len(t.children) for t in trees] == [3] * 4 and all(len(c.children) == 2 for t in trees for c in t.children)
    pay1, _ = tree_value(trees[0], [1, 1, 1], [True, False, True])
    pay3, _ = tree_value(trees[0], [3, 1, 1], [True, False, True])
    assert pay3 >= pay1 - 1e-9  # proposer optimisation cannot lower its expected payoff
