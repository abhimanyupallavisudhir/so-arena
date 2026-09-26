"""HiddenBits and verifiable disclosure: the domain, its verifier and scripted players, the judges' posteriors
against theory.disclosure, Proposition 3, and the claim ground truth."""

import math

import pytest

import so_arena as soa
from so_arena.analysis.frames import role_frame
from so_arena.analysis.metrics import asd
from so_arena.core.game import Player
from so_arena.core.runner import PlayerSpec, Profile, run_episodes, run_sync
from so_arena.domains import get_domain
from so_arena.domains.hidden_bits import (
    BitClaims,
    HiddenBits,
    bayesian_bit_judge,
    bit_advocate,
    bits_of,
    plain_bit_claims,
    shown_bit_claims,
)
from so_arena.mechanisms import Consultancy, Debate, Propaganda
from so_arena.samplers.arms import ASDExperiment
from so_arena.theory import disclosure as td

AFF = {"agents": ["bits"]}


def world(n=40, **kw):
    dom = HiddenBits(n_items=n, **kw)
    return dom, dom.load(), dom.context()


def verified(k, **kw):
    return soa.VerificationPolicy(verifiers=["bit"], budget_per_role=k, **kw)


def run_asd(mech, items, ctx, judge, advocate=None, dom=None):
    exp = ASDExperiment([mech], items, agent=advocate or bit_advocate(claims=15), fixtures={"judge": judge}, ctx=ctx,
                        ground_truth=dom.ground_truth_scorers() if dom else None)
    eps = exp.run()
    assert not [e.error for e in eps if e.error]
    return eps, asd(role_frame(eps)).iloc[0]


def p_yes(ep, item):
    yes = item.context["claim_bit"]
    return next(p for lab, p in ep.outcome.probs.items() if yes[lab] == 1)


# ----------------------------------------------------------------------------- domain and verifier

def test_items_bits_labels_and_truth_agree():
    dom = get_domain("hidden_bits", n_items=30, length=9, bias_range=(0.2, 0.8))
    for it in dom.load():
        bits = bits_of(it)
        yes = next(lab for lab, b in it.context["claim_bit"].items() if b == 1)
        assert len(bits) == 9 and it.answer(yes).text == "Yes"
        assert it.true_label == (yes if 2 * sum(bits) >= 9 else next(iter(set(it.labels) - {yes})))
        assert it.ground_truth.data["ones"] == sum(bits) and "bits" not in it.context
    assert [it.private for it in dom.load()] == [it.private for it in dom.load()]  # deterministic


@pytest.mark.parametrize("claim,status", [('<claim kind="bit" i="0">B0</claim>', "verified"),
                                          ('<claim kind="bit" i="0">NB0</claim>', "refuted"),
                                          ('<claim kind="bit" i="99">1</claim>', "refuted"),
                                          ('<claim kind="bit">1</claim>', "unchecked"),
                                          ('<claim kind="bit" i="0">maybe</claim>', "unchecked")])
def test_bit_verifier(claim, status):
    _, items, ctx = world(1)
    b0 = bits_of(items[0])[0]
    text = claim.replace("NB0", str(1 - b0)).replace("B0", str(b0))
    players = {"agent": Player(policy=soa.FixedPolicy(text), stance=items[0].labels[0]),
               "judge": Player(policy=bayesian_bit_judge())}
    ep = run_sync(Propaganda(verification=verified(None)).run(items[0], players, ctx))
    assert [v.status for v in ep.verifications("agent")] == [status]


def test_advocate_reveals_favourable_bits_and_lies_only_when_they_run_out():
    dom, items, ctx = world(30)
    liar = bit_advocate(claims=15, lie_rate=1.0)
    eps = run_sync(run_episodes(Propaganda(affordances=AFF), items,
                                [Profile(name="false", players={"agent": PlayerSpec(policy=liar, stance="false"),
                                                                "judge": bayesian_bit_judge()})],
                                ctx=ctx, ground_truth=dom.ground_truth_scorers()))
    for e, it in zip(eps, items):
        bits, want = bits_of(it), it.context["claim_bit"][e.positions["agent"]]
        claims = [(i, v) for i, v, _ in shown_bit_claims(e.turns_of("agent")[0].text)]
        assert len({i for i, _ in claims}) == len(claims) == 15 and all(v == want for _, v in claims)
        assert e.ground_truth["lies_unchecked_agent"] == sum(bits[i] != want for i, _ in claims) == 15 - bits.count(want)


def test_plain_claims_are_cheap_talk():
    assert plain_bit_claims('bit 3 is 1; Bit #4 = 0 <claim kind="bit" i="5">1</claim>') == [(3, 1), (4, 0)]
    _, items, ctx = world(1)
    plain = bit_advocate(claims=3, markup=False)
    players = {"agent": Player(policy=plain, stance=items[0].labels[0]), "judge": Player(policy=bayesian_bit_judge())}
    ep = run_sync(Propaganda(affordances=AFF, verification=verified(None)).run(items[0], players, ctx))
    assert ep.verifications() == [] and len(plain_bit_claims(ep.turns_of("agent")[0].text)) == 3
    assert ep.outcome.probs == pytest.approx({lab: 0.5 for lab in items[0].labels})  # a rational judge ignores them


# ----------------------------------------------------------------------------- judges vs theory

@pytest.mark.parametrize("k", [0, 3, 8])
@pytest.mark.parametrize("sceptical", [False, True])
def test_judge_posterior_equals_the_disclosure_formulas(k, sceptical):
    _, items, ctx = world(40)
    judge = bayesian_bit_judge(sceptical=sceptical, disclosure_limit=k)
    for mech in (Propaganda(affordances=AFF, verification=verified(k)),
                 Debate(rounds=1, affordances=AFF, verification=verified(k))):
        eps, _ = run_asd(mech, items, ctx, judge)
        by_id = {it.id: it for it in items}
        for e in eps:
            it = by_id[e.item_id]
            ones, L = sum(bits_of(it)), 15
            if mech.name == "debate":
                want = td.debate_value(L, k, ones, sceptical)
            else:
                side_yes = it.context["claim_bit"][e.positions["agent"]] == 1
                want = td.propaganda_value(L, k, ones, side_yes, sceptical)
            assert p_yes(e, it) == pytest.approx(want, abs=1e-9)


def test_naive_propaganda_asd_at_budget_8_is_about_004_in_theory_and_simulation():
    row = next(r for r in td.asd_table(15, [8]) if r["judge"] == "naive")
    assert row["propaganda_asd"] == pytest.approx(0.0366, abs=5e-4)
    _, items, ctx = world(200)
    _, a = run_asd(Propaganda(affordances=AFF, verification=verified(8)), items, ctx, bayesian_bit_judge())
    assert a["ci_low"] <= row["propaganda_asd"] <= a["ci_high"] and 0.02 < a["asd"] < 0.06


@pytest.mark.parametrize("mech", [Propaganda(affordances=AFF), Consultancy(rounds=2, affordances=AFF),
                                  Debate(rounds=2, affordances=AFF)])
def test_a_rational_judge_without_verification_gives_zero_asd(mech):
    _, items, ctx = world(20)
    eps, a = run_asd(mech, items, ctx, bayesian_bit_judge(trust=0.5), bit_advocate(claims=5, lie_rate=1.0))
    assert a["asd"] == 0.0 and all(p == pytest.approx(0.5) for e in eps for p in e.outcome.probs.values())


def test_a_credulous_judge_is_moved_by_cheap_talk_and_can_be_misled():
    _, items, ctx = world(60)
    honest_liar = bit_advocate(claims=10, lie_rate=1.0)
    _, credulous = run_asd(Propaganda(affordances=AFF), items, ctx, bayesian_bit_judge(trust=0.8), honest_liar)
    _, checked = run_asd(Propaganda(affordances=AFF, verification=verified(10)), items, ctx,
                         bayesian_bit_judge(trust=0.8), honest_liar)
    # cheap talk from both arms says the same thing (10 favourable bits): the credulous judge cannot tell them apart
    assert abs(credulous["asd"]) < 0.05 and checked["asd"] > credulous["asd"] + 0.3


def test_hiding_verdicts_from_the_judge_turns_verified_claims_into_cheap_talk():
    _, items, ctx = world(20)
    hidden = verified(8, show_to=["agent"])
    eps, a = run_asd(Propaganda(affordances=AFF, verification=hidden), items, ctx, bayesian_bit_judge())
    statuses = {v.status for e in eps for v in e.verifications()}
    assert a["asd"] == 0.0 and {"verified", "over_budget"} <= statuses <= {"verified", "refuted", "over_budget"}


def test_proposition_3():
    L, m = 15, 8
    for ones in range(L + 1):
        truth = 1.0 if ones >= m else 0.0
        # (1) debate, budget >= (L+1)/2: any judge treating verified bits as facts decides correctly
        for aware in (False, True):
            assert td.debate_value(L, m, ones, aware) == truth
        # (2) one advocate, sceptical judge, budget >= (L+1)/2: either advocate reveals the truth
        for side_yes in (False, True):
            assert td.propaganda_value(L, m, ones, side_yes, aware=True) == truth
    # (3) naive judge: selective disclosure keeps propaganda's ASD near 0 against debate's clipped certainty
    rows = {r["judge"]: r for r in td.asd_table(L, [m], eps=1e-6)}
    assert rows["naive"]["propaganda_asd"] < 0.04 and rows["naive"]["debate_asd"] == pytest.approx(math.log(1e6), 1e-3)
    # without verification nothing is learned
    assert all(r["propaganda_asd"] == r["debate_asd"] == 0.0 for r in td.asd_table(L, [0]))


def test_theory_posteriors_are_probabilities_and_selection_awareness_bounds_counts():
    assert td.posterior_majority_naive(15, 0, 0) == pytest.approx(0.5)
    assert td.posterior_majority_selection_aware(15, 3, 0, exact_ones=True) == 0.0
    assert td.posterior_majority_selection_aware(15, 8, 0) == 1.0
    assert td.posterior_majority_selection_aware(15, 3, 13, exact_ones=True, exact_zeros=True) == 0.5  # no mass
    with pytest.raises(ValueError):
        td.posterior_majority_naive(3, 2, 2)


# ----------------------------------------------------------------------------- ground truth of claims

def test_bit_claims_separate_checked_and_unchecked_lies():
    _, items, ctx = world(1)
    it = items[0]
    bits = bits_of(it)
    one, zero = bits.index(1), bits.index(0)
    text = (f'<claim kind="bit" i="{zero}">1</claim> <claim kind="bit" i="{one}">1</claim> '
            f'<claim kind="bit" i="{zero}">1</claim> bit {zero} is 1')
    players = {"agent": PlayerSpec(policy=soa.FixedPolicy(text), stance=it.labels[0]), "judge": bayesian_bit_judge()}
    for noise in (0.0, 1.0):  # what was checked, not what the (erring) verifier said
        ep = run_sync(run_episodes(Propaganda(verification=verified(2, noise=noise)), items,
                                   [Profile(name="p", players=players)], ctx=ctx, ground_truth=[BitClaims()]))[0]
        gt = ep.ground_truth
        assert (gt["claims_agent"], gt["lies_checked_agent"], gt["lies_unchecked_agent"]) == (4.0, 1.0, 2.0)
        assert gt["bit_honesty"] == {"agent": 0.25}
