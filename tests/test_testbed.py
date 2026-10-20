"""The conceptual-reasoning testbed: conventions shared by all families, and ground truth checked independently.

Every closed-form answer is checked against a separate simulation of the process the question describes, so a
mistake in a formula (or a question text that describes a different process than the formula computes) shows
up as a disagreement.
"""

from __future__ import annotations

import random
from collections import Counter

import numpy as np
import pytest

from so_arena.domains import get_domain
from so_arena.domains.testbed.base import by_family, split_of
from so_arena.domains.testbed.decisions import TEMPLATES


# ------------------------------------------------------------------------- conventions (every domain)
TESTBED_DOMAINS = ["decisions", "socialchoice", "numbers", "rlhack", "agents"]


@pytest.mark.parametrize("name", TESTBED_DOMAINS)
def test_items_follow_conventions(name):
    dom = get_domain(name, per_family=12)
    items = dom.load()
    fams = by_family(items)
    assert set(fams) == set(dom.families())
    for fam, its in fams.items():
        assert len(its) == 12
        # true answer at position A on exactly half the items of each family
        assert Counter(it.true_label for it in its) == {"A": 6, "B": 6}
        for it in its:
            assert it.labels == ["A", "B"] and it.metadata["tier"] in ("A", "B")
            assert it.metadata["split"] == split_of(it.id)
            assert sorted(a.value for a in it.answers) == [-1.0, 1.0]
            c = it.censored()
            assert c.ground_truth is None and all(a.value is None for a in c.answers) and not c.metadata
    # deterministic in the seed, different across seeds
    again = get_domain(name, per_family=12).load()
    assert [it.fingerprint() for it in items] == [it.fingerprint() for it in again]
    other = get_domain(name, per_family=12, seed=1).load()
    assert [it.question for it in items] != [it.question for it in other]


@pytest.mark.parametrize("name", TESTBED_DOMAINS)
def test_cards_reveal_no_outcomes(name):
    for fam in get_domain(name).families().values():
        assert len(fam.card) > 100 and fam.tier in ("A", "B")


# ------------------------------------------------------------------------- decisions: Monte Carlo
def _sim(template: str, p: dict, prog: str, rng: np.random.Generator, n: int = 200_000) -> float:
    """Simulate the process described in the question, n times, for the given program."""
    u = lambda: rng.random(n)  # noqa: E731
    if template == "newcomb":
        reads = u() < p["r"]
        correct = u() < p["q"]
        predicted_one = np.where(reads, np.where(correct, prog == "one", prog != "one"), u() < p["b"])
        box2 = np.where(predicted_one, p["M"], 0.0)
        return float(np.mean(box2 + (p["k"] if prog == "two" else 0.0)))
    if template == "lesion":
        gene = u() < p["g"]
        risk = np.where(gene, p["c1"], p["c0"]) + (p["delta"] if prog == "smoke" else 0.0)
        cancer = u() < np.minimum(risk, 1.0)
        return float(np.mean((p["s"] if prog == "smoke" else 0.0) - p["C"] * cancer))
    if template == "twin":
        copy = u() < p["rho"]
        y_coop = np.where(copy, prog == "coop", u() < p["c"])
        x_coop = prog == "coop"
        pay = np.where(x_coop, np.where(y_coop, p["R"], p["S"]), np.where(y_coop, p["T"], p["P"]))
        return float(np.mean(pay))
    if template == "mugging":
        heads = u() < p["h"]
        pays = prog == "pay"
        return float(np.mean(np.where(heads, p["y"] * pays, -p["x"] * pays)))
    if template == "beauty":
        heads = u() < p["h"]
        stake, price = p["stake"], p["c"] * p["stake"]
        if prog == "reject":
            return 0.0
        per_bet = np.where(heads, -price, stake - price)
        n_settled = np.where(heads, p["nH"], p["nT"]) if p["mode"] == "every" else 1
        return float(np.mean(per_bet * n_settled))
    if template == "driver":
        q = p["p1"] if prog == "lo" else p["p2"]
        first = u() < q
        second = u() < q
        pay = np.where(first, p["a"], np.where(second, p["b"], p["c"])) * p["unit"]
        return float(np.mean(pay))
    if template == "damascus":
        if prog == "coin":
            x_aleppo = u() < 0.5
            death_aleppo = u() < 0.5
        else:
            x_aleppo = np.full(n, prog == "aleppo")
            correct = u() < p["q"]
            death_aleppo = np.where(correct, x_aleppo, ~x_aleppo)
        survive = x_aleppo != death_aleppo
        return float(np.mean(p["V"] * survive - p["t"] * x_aleppo))
    if template == "hitchhiker":
        correct = u() < p["q"]
        predicted_pay = np.where(correct, prog == "pay", prog != "pay")
        return float(np.mean(np.where(predicted_pay, p["V"] - (p["P"] if prog == "pay" else 0.0), 0.0)))
    raise KeyError(template)


PROGRAMS = {"newcomb": ("one", "two"), "lesion": ("smoke", "abstain"), "twin": ("coop", "defect"),
            "mugging": ("pay", "refuse"), "beauty": ("accept", "reject"), "driver": ("lo", "hi"),
            "hitchhiker": ("pay", "nopay")}


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_decision_payoffs_match_simulation(template):
    rng = random.Random(7)
    gen = np.random.default_rng(7)
    checked = 0
    while checked < 6:
        d = TEMPLATES[template](rng)
        if d is None:
            continue
        progs = d.params["pair"] if template == "damascus" else PROGRAMS[template]
        eu = d.data["expected_payoff"]
        texts = dict(zip(progs, eu))  # insertion order of programs in the draft
        for prog in progs:
            closed = eu[texts[prog]]
            sim = _sim(template, d.params, prog, gen)
            scale = max(1.0, max(abs(v) for v in eu.values()), d.data["gap"])
            assert abs(sim - closed) < 0.02 * scale + 1e-9, (template, prog, d.params, sim, closed)
        # the simulated winner is the draft's true answer
        sims = {prog: _sim(template, d.params, prog, gen) for prog in progs}
        assert max(sims, key=sims.get) == d.params["winner"], (template, d.params, sims)
        checked += 1


# ------------------------------------------------------------------------- social choice: known theorems
@pytest.mark.parametrize("args,expected", [
    (("rankings", 2, 3, "full", "strict", ("iia", "nondictatorship", "pareto")), False),          # Arrow
    (("rankings", 3, 3, "full", "weak", ("iia", "nondictatorship", "pareto")), False),            # Arrow, ties allowed
    (("rankings", 3, 3, "full", "strict", ("iia", "pareto")), True),                               # dictatorship
    (("rankings", 3, 3, "single_peaked", "strict", ("anonymity", "iia", "nondictatorship", "pareto")), True),  # majority
    (("winners", 3, 3, "full", "winner", ("nondictatorship", "onto", "strategyproof")), False),   # Gibbard-Satterthwaite
    (("winners", 2, 4, "full", "winner", ("nondictatorship", "onto", "strategyproof")), False),
    (("winners", 3, 3, "single_peaked", "winner", ("anonymity", "nondictatorship", "onto", "strategyproof")), True),  # median
    (("winners", 3, 3, "full", "winner", ("monotonic", "nondictatorship", "unanimity")), False),  # Muller-Satterthwaite
    (("winners", 3, 3, "full", "winner", ("anonymity", "neutrality")), False),                    # Moulin: 3 | 3
    (("winners", 2, 3, "full", "winner", ("anonymity", "neutrality")), True),                     # 3 is no sum of 2s
    (("winners", 2, 3, "full", "winner", ("anonymity", "neutrality", "pareto")), False),
])
def test_social_choice_matches_theorems(args, expected):
    from so_arena.domains.testbed.socialchoice import satisfiable

    assert satisfiable(*args) is expected


# ------------------------------------------------------------------------- numbers: brute force
def _isprime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n ** 0.5) + 1))


@pytest.mark.parametrize("pattern,N", [("twin", 3000), ("sexy", 3000), ("sophie", 3000), ("square_plus_one", 300),
                                       ("goldbach", 2468), ("two_squares", 2000)])
def test_number_counts_match_brute_force(pattern, N):
    from so_arena.domains.testbed.numbers import _count

    brute = {
        "twin": lambda: sum(_isprime(p) and _isprime(p + 2) for p in range(N + 1)),
        "sexy": lambda: sum(_isprime(p) and _isprime(p + 6) for p in range(N + 1)),
        "sophie": lambda: sum(_isprime(p) and _isprime(2 * p + 1) for p in range(N + 1)),
        "square_plus_one": lambda: sum(_isprime(n * n + 1) for n in range(1, N + 1)),
        "goldbach": lambda: sum(_isprime(p) and _isprime(N - p) for p in range(2, N // 2 + 1)),
        "two_squares": lambda: sum(any(int((m - a * a) ** 0.5 + 1e-9) ** 2 == m - a * a for a in range(int(m ** 0.5) + 1))
                                   for m in range(1, N + 1)),
    }[pattern]()
    assert _count(pattern, N) == brute


def test_collatz_escape_matches_python_ints():
    from so_arena.domains.testbed.numbers import escape_share

    def slow(a, b, L, steps, bound):
        esc = 0
        for n0 in range(1, L + 1):
            n = n0
            for _ in range(steps):
                n = n // 2 if n % 2 == 0 else a * n + b
                if n > bound:
                    esc += 1
                    break
        return esc / L

    for args in [(3, 1, 200, 300, 10 ** 6), (5, 1, 200, 300, 10 ** 9), (7, 3, 100, 100, 10 ** 12)]:
        assert escape_share(*args) == pytest.approx(slow(*args))


# ------------------------------------------------------------------------- measured families
def test_rlhack_records_reproduce():
    """A committed experiment re-run gives the recorded number of exploiting runs (numba seeds are fixed)."""
    import json

    from so_arena.datasets import sample_path
    from so_arena.domains.testbed.rlhack import SAMPLE_FILE, run_experiment

    with open(sample_path(SAMPLE_FILE)) as f:
        recs = [json.loads(line) for line in f][:: 97]
    for r in recs:
        assert run_experiment(r["params"]) == r["exploits"]


def test_agent_pairs_differ_in_one_factor_and_clear_the_gap():
    from so_arena.domains.testbed.agents import MIN_GAP

    dom = get_domain("agents", per_family=20)
    for fam, pairs in dom.pairs().items():
        kind = fam.split("/", 1)[1]
        assert pairs, fam
        for p in pairs:
            a, b = p["a"]["condition"], p["b"]["condition"]
            assert [k for k in a if a[k] != b[k]] == [p["factor"]]
            assert abs(p["a"]["metric"] - p["b"]["metric"]) >= MIN_GAP[kind]


def test_suite_mixes_families_and_keeps_ids():
    dom = get_domain("testbed", families=["decisions/newcomb", "numbers", "rlhack/sensor"], per_family=6)
    items = dom.load()
    fams = by_family(items)
    assert set(fams) == {"decisions/newcomb", "numbers/prime_race", "numbers/counts", "numbers/collatz",
                         "rlhack/sensor"}
    alone = get_domain("decisions", families=["newcomb"], per_family=6).load()
    assert [it.id for it in fams["decisions/newcomb"]] == [it.id for it in alone]


@pytest.mark.network
def test_dtbench_loads():
    items = get_domain("dtbench").load()
    assert len(items) == 340 and Counter(it.true_label for it in items) == {"A": 170, "B": 170}
