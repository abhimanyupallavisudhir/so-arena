"""Mechanism zoo: k-debater debate and cross-examination, comparisons (RLHF signal), prover-verifier games,
Correlated Agreement and DMI peer prediction, and the level-k PSRO meta-solver."""

import math
import random

import numpy as np
import pytest

import so_arena as soa
from so_arena.core.policy import FunctionPolicy, stable_hash
from so_arena.core.rewards import TRANSFORMS, score_probability, truth_oracle
from so_arena.core.runner import Profile, run_episodes, run_sync
from so_arena.domains.synthetic import (SyntheticPersuasion, synthetic_arguer, synthetic_comparer, synthetic_judge,
                                        synthetic_reviewer, synthetic_worker)
from so_arena.games.normal_form import NormalFormGame
from so_arena.mechanisms import Comparison, Debate, PeerPrediction, PreferenceScore, ProverReward, ProverVerifier
from so_arena.mechanisms.preference import worst_score
from so_arena.samplers.pools import OptimizationExperiment
from so_arena.samplers.psro import PSRO, solve_meta


def run(mech, items, profiles, ctx=None, gt=None, seed=0):
    eps = run_sync(run_episodes(mech, items, profiles, ctx=ctx, ground_truth=gt, seed=seed))
    errs = [e.error for e in eps if e.error]
    assert not errs, errs[0]
    return eps


def three_options(id="t", correct="A"):
    return soa.TaskItem(id=id, question="Which is it?", answers=[soa.AnswerOption(label=x, text=f"option {x}") for x in "ABC"],
                        ground_truth=soa.GroundTruth(correct=correct))


def recorder(seen):
    """Chooses the first option offered, says who it is and records every prompt it gets."""

    def act(req, ctx):
        seen.setdefault(ctx.role, []).append((req.phase, "\n".join(m.content for m in req.prompt)))
        if req.kind == "choice":
            return f"Answer: {req.options[0]}"
        if req.kind == "probabilities":
            return {o: 1 / len(req.options) for o in req.options}
        return f"<{ctx.role} {req.phase}>"

    return FunctionPolicy(act)


# ----------------------------------------------------------------------------- debate

def test_two_debater_config_is_unchanged():
    # the new options stay out of the config at their defaults: the config hash (hence episode ids) of every
    # existing debate is what it was
    d = Debate()
    assert "n_debaters" not in d.config and "cross_examination" not in d.config
    assert d.config_hash() == Debate(n_debaters=2, cross_examination=False).config_hash()
    assert Debate(n_debaters=3).config_hash() != d.config_hash() != Debate(cross_examination=True).config_hash()
    assert list(d.roles()) == ["debater_a", "debater_b", "judge"]


def test_two_debater_prompts_are_unchanged():
    seen = {}
    item = three_options()
    run_sync(Debate(rounds=1).run(item, {r: soa.Player(policy=recorder(seen)) for r in ("debater_a", "debater_b", "judge")}))
    system = seen["debater_a"][1][1]
    assert "Your opponent defends (B)" in system and "Rebut your opponent's points." in system
    assert "simultaneously (you cannot see your opponent's current speech)" in system
    assert "Two debaters argued for different answers" in seen["judge"][0][1]


def test_k_debaters_take_distinct_positions():
    seen = {}
    mech = Debate(n_debaters=3, rounds=1)
    assert mech.debaters == ["debater_a", "debater_b", "debater_c"]
    ep = run_sync(mech.run(three_options(), {r: soa.Player(policy=recorder(seen)) for r in [*mech.debaters, "judge"]}))
    assert ep.error is None
    assert [ep.positions[d] for d in mech.debaters] == ["A", "B", "C"]  # each chose the first answer left
    assert "Your opponents defend: Debater B, (B)" in seen["debater_a"][1][1] and "Debater C, (C)" in seen["debater_a"][1][1]
    assert "3 debaters argued" in seen["judge"][0][1]
    assert set(ep.rewards) == set(mech.debaters)
    # more debaters than answers: answers repeat once none is left
    mech4 = Debate(n_debaters=4, rounds=1)
    ep4 = run_sync(mech4.run(three_options(), {r: soa.Player(policy=recorder({})) for r in [*mech4.debaters, "judge"]}))
    assert [ep4.positions[d] for d in mech4.debaters] == ["A", "B", "C", "A"]
    # assigned stances are kept (the arms of an ASD design)
    players = {r: soa.Player(policy=synthetic_arguer(), stance=s) for r, s in zip(mech.debaters, "CAB")}
    ep = run_sync(mech.run(three_options(), {**players, "judge": soa.Player(policy=synthetic_judge())}))
    assert [ep.positions[d] for d in mech.debaters] == ["C", "A", "B"]


def test_zero_sum_needs_two_debaters():
    with pytest.raises(ValueError, match="two debaters"):
        Debate(n_debaters=3, zero_sum=True)
    with pytest.raises(ValueError):
        Debate(n_debaters=1)


def test_cross_examination_phase_order_and_visibility():
    seen = {}
    mech = Debate(rounds=2, cross_examination=True)
    ep = run_sync(mech.run(three_options(), {r: soa.Player(policy=recorder(seen)) for r in ("debater_a", "debater_b", "judge")}))
    assert ep.error is None
    order = [(t.role, t.phase) for t in ep.turns if t.phase != "position"]
    assert order == [("debater_a", "round1"), ("debater_b", "round1"),
                     ("debater_a", "cross_question:debater_b"), ("debater_b", "cross_answer:debater_a"),
                     ("debater_b", "cross_question:debater_a"), ("debater_a", "cross_answer:debater_b"),
                     ("debater_a", "round2"), ("debater_b", "round2"), ("judge", "judgment")]
    # the whole cross-examination is public; the answering debater sees the question it answers
    assert all(t.visible_to is None for t in ep.turns if t.phase.startswith("cross"))
    answer_prompt = dict(seen["debater_b"])["cross_answer:debater_a"]
    assert "<debater_a cross_question:debater_b>" in answer_prompt and "answer Debater A's question" in answer_prompt
    # opening speeches stay simultaneous; closing speeches see the cross-examination
    assert "<debater_a round1>" not in dict(seen["debater_b"])["round1"]
    assert "<debater_a cross_answer:debater_b>" in dict(seen["debater_b"])["round2"]
    judge_prompt = seen["judge"][0][1]
    assert "<debater_b cross_answer:debater_a>" in judge_prompt
    assert "each debater asks each opponent one question" in dict(seen["debater_a"])["round1"]


# ----------------------------------------------------------------------------- comparison

def comparison_setup(n=6):
    dom = SyntheticPersuasion(n_items=n, seed=2)
    return dom.load(), dom.context()


def fixed_candidate(text):
    return FunctionPolicy(lambda req, ctx: text)


def test_comparison_pays_the_score_of_being_preferred():
    items, ctx = comparison_setup(1)
    item = items[0]
    wrong = next(x for x in item.labels if x != item.true_label)
    texts = {"candidate_1": f'<arg for="accept" strength="1.0"> Answer: {item.true_label}',
             "candidate_2": f'<arg for="accept" strength="0.0"> Answer: {wrong}',
             "candidate_3": f'<arg for="accept" strength="0.5"> Answer: {item.true_label}'}
    players = {**{c: fixed_candidate(t) for c, t in texts.items()}, "judge": synthetic_comparer()}
    for aggregate in (False, True):
        mech = Comparison(n_candidates=3, aggregate=aggregate, transform="log")
        ep = run(mech, [item], [Profile(name="p", players=players)], ctx=ctx, gt=soa.default_scorers())[0]
        w = {c: math.exp(s) for c, s in (("candidate_1", 1.0), ("candidate_2", 0.0), ("candidate_3", 0.5))}
        pref = {c: v / sum(w.values()) for c, v in w.items()}
        assert ep.outcome.data["preference"] == pytest.approx(pref)
        # outcome: preference summed by answer, so decision accuracy and judge correctness are defined
        assert ep.outcome.probs[item.true_label] == pytest.approx(pref["candidate_1"] + pref["candidate_3"])
        assert ep.outcome.decision == item.true_label and ep.ground_truth["judge_correct"] == 1.0
        assert ep.positions["candidate_2"] == wrong and ep.ground_truth["role_values"]["candidate_2"] == -1.0
        if aggregate:  # duplicates of the right answer share its preference instead of competing
            assert ep.rewards["candidate_1"] == ep.rewards["candidate_3"] == pytest.approx(math.log(ep.outcome.probs[item.true_label]))
        else:
            assert ep.rewards == pytest.approx({c: math.log(p) for c, p in pref.items()})


def test_comparison_open_ended_and_unparsed_answers():
    item = soa.TaskItem(id="open", question="Write a haiku.")
    players = {"candidate_1": fixed_candidate('<arg for="accept" strength="2.0"> An old silent pond'),
               "candidate_2": fixed_candidate('<arg for="accept" strength="0.0"> roses are red'),
               "judge": synthetic_comparer()}
    ep = run(Comparison(aggregate=True), [item], [Profile(name="p", players=players)])[0]
    # open-ended: every candidate is an answer of its own, so aggregation changes nothing
    assert set(ep.outcome.probs) == {"candidate_1", "candidate_2"} and ep.outcome.decision == "candidate_1"
    assert ep.outcome.output.endswith("An old silent pond")
    p1 = 1 / (1 + math.exp(-2.0))
    assert ep.rewards == pytest.approx({"candidate_1": math.log(p1), "candidate_2": math.log(1 - p1)})
    rule = PreferenceScore("brier")
    assert rule.compute(ep)["candidate_1"] == pytest.approx(score_probability(ep.outcome.data["preference"], "candidate_1", "brier"))


def test_best_of_n_against_a_comparison_judge():
    # the RLHF signal under optimization: selecting candidate_1's best of n samples by its reward raises it
    items, ctx = comparison_setup()
    aff = {"agents": ["answer_key"]}
    prof = Profile(name="p", players={"candidate_1": synthetic_worker(), "candidate_2": synthetic_worker(),
                                      "judge": synthetic_comparer()})
    exp = OptimizationExperiment(Comparison(affordances=aff), items, prof, pool_sizes={"candidate_1": 6}, ctx=ctx)
    exp.run()
    curve = exp.grid({"candidate_1": [1, 2, 4, 6]})
    assert curve["reward_candidate_1"].is_monotonic_increasing
    assert curve["reward_candidate_1"].iloc[-1] > curve["reward_candidate_1"].iloc[0] + 0.1


# ----------------------------------------------------------------------------- prover-verifier

def mode_prover(seen):
    """Gives the right answer when told to be helpful, a wrong one when told to be sneaky."""

    def act(req, ctx):
        text = "\n".join(m.content for m in req.prompt)
        seen.append(text)
        truth = req.view.item.private["answer_key"]
        answer = truth if "HELPFUL" in text else next(x for x in req.view.item.labels if x != truth)
        return f'<arg for="accept" strength="0.5"> Answer: {answer}'

    return FunctionPolicy(act)


def test_prover_mode_is_a_chance_move():
    items, ctx = comparison_setup(8)
    mech = ProverVerifier(oracle=truth_oracle(items), affordances={"agents": ["answer_key"]})
    item = items[0]
    players = {"prover": soa.Player(policy=mode_prover([])), "verifier": soa.Player(policy=synthetic_reviewer())}
    modes = {run_sync(mech.run(item, players, ctx, episode_id=eid)).outcome.data["mode"] for eid in ("a", "b:tree:1.0")}
    assert len(modes) == 1  # the same for every play of the item, whatever the path
    by_seed = {s: run_sync(mech.run(item, players, ctx, seed=s)).outcome.data["mode"] for s in range(12)}
    assert set(by_seed.values()) == {"helpful", "sneaky"}
    for rate, mode in ((0.0, "helpful"), (1.0, "sneaky")):
        m = ProverVerifier(oracle=truth_oracle(items), sneaky_rate=rate, affordances={"agents": ["answer_key"]})
        assert {e.outcome.data["mode"] for e in run(m, items, [Profile(name="p", players={
            "prover": mode_prover([]), "verifier": synthetic_reviewer()})], ctx=ctx)} == {mode}


def test_prover_reward_and_trainable_verifier():
    items, ctx = comparison_setup(8)
    seen_p, seen_v = [], []

    def verifier(req, ctx):
        seen_v.append("\n".join(m.content for m in req.prompt))
        return {"accept": 0.7, "reject": 0.3}

    mech = ProverVerifier(oracle=truth_oracle(items), affordances={"agents": ["answer_key"]}, trainable={"verifier": True})
    eps = run(mech, items, [Profile(name="p", players={"prover": mode_prover(seen_p), "verifier": FunctionPolicy(verifier)})],
              ctx=ctx, gt=soa.default_scorers() + [soa.JudgeCorrectness(subject="prover")])
    assert {e.outcome.data["mode"] for e in eps} == {"helpful", "sneaky"}
    assert not any("SNEAKY" in t or "HELPFUL" in t for t in seen_v)  # the mode is never shown to the verifier
    for e in eps:
        correct = e.outcome.data["mode"] == "helpful"
        assert e.rewards["prover"] == pytest.approx(math.log(0.7))  # the prover always fits its mode here
        # the verifier: log score of its verdict against the audited correctness
        assert e.rewards["verifier"] == pytest.approx(math.log(0.7 if correct else 0.3))
        assert e.reward_details["prover_audit_correct"] is correct
        assert e.ground_truth["judge_correct"] == (1.0 if correct else 0.0)
    # a prover that ignores its mode earns the misaligned penalty, below anything an aligned prover can get
    stubborn = FunctionPolicy(lambda req, c: f"Answer: {req.view.item.private['answer_key']}")
    eps = run(mech, items, [Profile(name="p", players={"prover": stubborn, "verifier": FunctionPolicy(verifier)})], ctx=ctx)
    for e in eps:
        expected = math.log(0.7) if e.outcome.data["mode"] == "helpful" else worst_score("log") - 1
        assert e.rewards["prover"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="oracle"):
        ProverVerifier()


@pytest.mark.parametrize("transform", TRANSFORMS)
def test_misaligned_penalty_is_below_every_aligned_score(transform):
    # docs/theory.md, section 4: complying with the mode is strictly dominant for every verifier iff the misaligned
    # pay is below the transform's worst aligned score
    worst = worst_score(transform)
    grid = [score_probability({"accept": p, "reject": 1 - p}, "accept", transform) for p in np.linspace(0, 1, 101)]
    assert min(grid) == pytest.approx(worst)
    assert ProverReward(lambda ep: None, transform).misaligned < worst
    with pytest.raises(ValueError, match="strictly below"):
        ProverReward(lambda ep: None, transform, misaligned=worst)


# ----------------------------------------------------------------------------- peer prediction

def bundle(n_tasks, seed, labels=("A", "B")):
    truth = {f"s{i}": random.Random(stable_hash("truth", seed, i)).choice(labels) for i in range(n_tasks)}
    item = soa.TaskItem(id=f"bundle{seed}", question="?", context={"subitems": [
        {"id": s, "question": f"Q{s}?", "labels": list(labels)} for s in truth]})
    return item, truth


def reporter(truth, *, accuracy=0.8, informative=1.0, constant=None, labels=("A", "B")):
    """Sees a private signal (the truth with probability ``accuracy``) per task; reports it with probability
    ``informative``, else a uniformly random answer. Draws depend on the role and task only."""

    def act(req, ctx):
        sid = req.phase.split(":", 1)[1]
        if constant is not None:
            return f"Answer: {constant}"
        rng = random.Random(stable_hash("signal", ctx.role, req.view.item.id, sid))
        signal = truth[sid] if rng.random() < accuracy else rng.choice([x for x in labels if x != truth[sid]])
        report = signal if rng.random() < informative else rng.choice(labels)
        return f"Answer: {report}"

    return FunctionPolicy(act)


def test_correlated_agreement_pays_informative_reports_and_nothing_for_constant_ones():
    mech = PeerPrediction(n_reporters=3, rule="ca")
    item, truth = bundle(12, 0)
    const = run(mech, [item], [Profile(name="c", players={r: reporter(truth, constant="A") for r in mech.reporters})])[0]
    assert const.rewards == {r: 0.0 for r in mech.reporters}
    truthful, noise = [], []
    for it, tr in (bundle(20, s) for s in range(20)):
        players = {r: reporter(tr) for r in mech.reporters}
        truthful.append(run(mech, [it], [Profile(name="t", players=players)])[0].rewards["reporter_1"])
        players["reporter_1"] = reporter(tr, informative=0.0)  # answers at random
        noise.append(run(mech, [it], [Profile(name="n", players=players)])[0].rewards["reporter_1"])
    # truthful: agreement 0.68 on the same task vs 0.5 across tasks, i.e. about 0.18 per task
    assert np.mean(truthful) == pytest.approx(0.18, abs=0.08)
    assert abs(np.mean(noise)) < 0.08 < np.mean(truthful)
    short = run_sync(run_episodes(mech, [bundle(2, 0)[0]], [Profile(name="c", players={r: reporter(truth) for r in mech.reporters})]))
    assert "at least 3 tasks" in short[0].error


def test_dmi_truthful_beats_partly_informative():
    mech = PeerPrediction(n_reporters=3, rule="dmi")
    truthful, garbled = [], []
    for s in range(60):
        item, truth = bundle(40, s)
        others = {r: reporter(truth) for r in mech.reporters[1:]}
        for informative, out in ((1.0, truthful), (0.7, garbled)):
            players = {"reporter_1": reporter(truth, informative=informative), **others}
            out.append(run(mech, [item], [Profile(name=f"i{informative}", players=players)])[0].rewards["reporter_1"])
    assert all(-1.0 <= x <= 1.0 for x in truthful + garbled)
    # a 70%-informative strategy garbles with determinant 0.7: expected pay 0.7^2 of the truthful pay (0.48 over
    # 200 bundles; noisier here)
    assert np.mean(truthful) > np.mean(garbled) > 0
    assert 0.2 < np.mean(garbled) / np.mean(truthful) < 0.8


def test_dmi_uses_a_fixed_normalising_constant():
    # the payment is det(M1) det(M2) / ((n1/C)^C (n2/C)^C), a constant of the bundle - never of the population
    mech = PeerPrediction(n_reporters=2, rule="dmi")
    answers = {"reporter_1": "AABBABAB", "reporter_2": "ABBBAAAB"}
    item = soa.TaskItem(id="b", question="?", context={"subitems": [
        {"id": f"s{i}", "question": f"Q{i}?", "labels": ["A", "B"]} for i in range(8)]})
    players = {r: FunctionPolicy(lambda req, c: f"Answer: {answers[c.role][int(req.phase.split(':s')[1])]}")
               for r in answers}
    ep = run(mech, [item], [Profile(name="p", players=players)])[0]

    def det(part):
        M = np.zeros((2, 2))
        for t in part:
            M["AB".index(answers["reporter_1"][t]), "AB".index(answers["reporter_2"][t])] += 1
        return np.linalg.det(M)

    expected = det(range(4)) * det(range(4, 8)) / (2.0 ** 2 * 2.0 ** 2)
    assert ep.rewards["reporter_1"] == pytest.approx(expected) and ep.rewards["reporter_2"] == pytest.approx(expected)
    const = {r: FunctionPolicy(lambda req, c: "Answer: B") for r in answers}
    assert run(mech, [item], [Profile(name="c", players=const)])[0].rewards == pytest.approx({r: 0.0 for r in answers})
    short = item.model_copy(update={"context": {"subitems": item.context["subitems"][:3]}})
    ep = run_sync(run_episodes(mech, [short], [Profile(name="p", players=players)]))[0]
    assert "at least 4 tasks" in ep.error  # two halves of at least C = 2 tasks


# ----------------------------------------------------------------------------- PSRO: level-k

def test_last_meta_solver_is_iterated_best_response():
    g = NormalFormGame(["a", "b"], {"a": ["x", "y", "z"], "b": ["u", "v"]},
                       {"a": np.arange(6.0).reshape(3, 2), "b": -np.arange(6.0).reshape(3, 2)})
    x = solve_meta(g, "last")
    assert x[0].tolist() == [0, 0, 1] and x[1].tolist() == [0, 1]
    with pytest.raises(ValueError, match="meta-solver"):
        solve_meta(g, "level-k")


def test_psro_with_the_last_meta_solver():
    from test_search import level_factory, optimizer_model

    dom = SyntheticPersuasion(n_items=12, seed=4)
    items, ctx = dom.load()[:4], dom.context()
    opt, seen = optimizer_model()
    psro = PSRO(Debate(rounds=1, affordances={"agents": ["answer_key"]}, zero_sum=True), items,
                roles=["debater_a", "debater_b"], initial={"debater_a": {"base": ""}, "debater_b": {"base": ""}},
                policy_factories={"debater_a": level_factory, "debater_b": level_factory},
                fixtures={"judge": synthetic_judge()}, optimizer=opt, meta_solver="last",
                stances={"debater_a": "true", "debater_b": "false"}, iterations=2,
                search_kwargs={"iterations": 1, "candidates_per_iter": 2}, ctx=ctx)
    hist = psro.run()
    for h in hist:  # each level best-responds to the others' newest strategy only
        for r, sig in h.meta_strategy.items():
            assert sig == [0.0] * (len(sig) - 1) + [1.0]
    assert "with probability 1.00" in seen[-1]
    with pytest.raises(ValueError, match="meta-solver"):
        PSRO(Debate(), items, roles=["debater_a"], initial={"debater_a": {"base": ""}},
             policy_factories={"debater_a": level_factory}, fixtures={}, optimizer=opt, meta_solver="level")
