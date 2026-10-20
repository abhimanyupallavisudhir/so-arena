"""Social choice: is a set of axioms jointly satisfiable in a small voting setting? Decided by a SAT solver.

Impossibility theorems (Arrow, Gibbard-Satterthwaite, Muller-Satterthwaite, Moulin's anonymity/neutrality
conflict) are the cleanest examples of philosophical arguments with checkable conclusions: a list of
individually reasonable desiderata, and a question of whether anything satisfies them all. Each item fixes a
finite setting - 2 or 3 voters, 3 (sometimes 4) alternatives, the allowed ballots (all strict rankings, or
only rankings single-peaked on a fixed axis), and whether a rule outputs a social ranking or a single winner -
and lists properties, each defined precisely in the question. The answer (does some rule satisfy all of
them?) is computed exactly by encoding every possible rule as a SAT instance.

Recalling the famous theorems is not enough: their hypotheses (number of voters, alternatives, domain,
resoluteness, which axioms) are varied, so a reasoner must check whether a theorem's proof actually applies
or construct a rule. Families: ``socialchoice/rankings`` (rules that output a social ranking, strict or with
ties) and ``socialchoice/winners`` (rules that pick one winner).
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Sequence
from functools import lru_cache

from so_arena.domains.base import register_domain
from so_arena.domains.testbed.base import Draft, Family, TestbedDomain, choose

ALTS = "abcd"

FAMILIES = {
    "socialchoice/rankings": Family(
        "socialchoice/rankings", "Which axioms can a social welfare function satisfy together?",
        "Each item fixes a small voting setting: 2 or 3 voters, 3 alternatives, the ballots voters may submit "
        "(all strict rankings, or only rankings single-peaked on a fixed left-right axis), and whether the "
        "social ranking a rule outputs must be strict or may contain ties. It then lists several properties of "
        "rules (Pareto, independence of irrelevant alternatives, non-dictatorship, anonymity, neutrality, "
        "unanimity, non-imposition, ...), each defined precisely. The question is whether any rule satisfies all "
        "of them at once.",
        tags=("social choice", "impossibility theorems")),
    "socialchoice/winners": Family(
        "socialchoice/winners", "Which axioms can a voting rule that picks one winner satisfy together?",
        "Each item fixes a small voting setting: 2 or 3 voters, 3 or 4 alternatives, and the ballots voters may "
        "submit (all strict rankings, or only rankings single-peaked on a fixed axis). A rule picks exactly one "
        "winning alternative for every profile of ballots. The item lists several properties (strategy-"
        "proofness, onto, non-dictatorship, unanimity, Pareto efficiency, Maskin monotonicity, Condorcet "
        "consistency, anonymity, neutrality, ...), each defined precisely. The question is whether any rule "
        "satisfies all of them at once.",
        tags=("social choice", "impossibility theorems", "mechanism design")),
}

# ------------------------------------------------------------------------------------- setting
Ranking = tuple[int, ...]  # alternatives from best to worst


def rankings(m: int) -> list[Ranking]:
    return list(itertools.permutations(range(m)))


def single_peaked(m: int) -> list[Ranking]:
    """Strict rankings single-peaked on the axis 0 < 1 < ... < m-1."""
    out = []
    for r in rankings(m):
        pos = [r.index(a) for a in range(m)]
        peak = r[0]
        ok = all(pos[a] > pos[a + 1] for a in range(peak)) and all(pos[a] < pos[a + 1] for a in range(peak, m - 1))
        if ok:
            out.append(r)
    return out


def weak_orders(m: int) -> list[tuple[int, ...]]:
    """Weak orders as rank vectors (rank[a] = number of alternatives strictly above a), normalized."""
    out = set()
    for vals in itertools.product(range(m), repeat=m):
        # rank vector: a's level; normalize to dense levels
        levels = sorted(set(vals))
        out.add(tuple(levels.index(v) for v in vals))
    return sorted(out)


def prefers(r: Ranking, x: int, y: int) -> bool:
    return r.index(x) < r.index(y)


class CNF:
    def __init__(self):
        self.n = 0
        self.clauses: list[list[int]] = []

    def var(self) -> int:
        self.n += 1
        return self.n

    def add(self, *lits: int) -> None:
        self.clauses.append(list(lits))

    def exactly_one(self, xs: Sequence[int]) -> None:
        self.add(*xs)
        for a, b in itertools.combinations(xs, 2):
            self.add(-a, -b)

    def iff_or(self, v: int, xs: Sequence[int]) -> None:
        """v <-> OR(xs)."""
        self.add(-v, *xs)
        for x in xs:
            self.add(-x, v)

    def equal(self, a: int, b: int) -> None:
        self.add(-a, b)
        self.add(a, -b)

    def solve(self) -> bool:
        from pysat.solvers import Solver

        with Solver(name="cadical153", bootstrap_with=self.clauses) as s:
            return bool(s.solve())


# ------------------------------------------------------------------------------------- axioms
RANKING_AXIOMS = {
    "pareto": ("Pareto", "whenever every voter ranks an alternative x above an alternative y, the social ranking "
                         "puts x strictly above y."),
    "iia": ("Independence of irrelevant alternatives", "for any two alternatives x and y, whether the social "
            "ranking puts x above y, y above x, or ties them depends only on how each voter ranks x relative to y "
            "(two profiles in which every voter ranks x and y the same way get the same social verdict on x vs "
            "y)."),
    "nondictatorship": ("Non-dictatorship", "there is no voter whose ranking the social ranking always copies "
                        "exactly (for every profile)."),
    "anonymity": ("Anonymity", "permuting the voters' ballots among the voters never changes the social "
                  "ranking."),
    "neutrality": ("Neutrality", "relabelling the alternatives in every ballot relabels the social ranking in the "
                   "same way."),
    "unanimity": ("Unanimity", "if all voters submit the same ranking, the social ranking is that ranking."),
    "nonimposition": ("Non-imposition", "every ranking that the rule is allowed to output is the social ranking "
                      "of at least one profile."),
    "nonconstant": ("Non-constancy", "the rule does not output the same social ranking for every profile."),
}

WINNER_AXIOMS = {
    "strategyproof": ("Strategy-proofness", "no voter can ever obtain a winner they rank strictly higher (by their "
                      "true ballot) by submitting a different allowed ballot, the other ballots being fixed."),
    "onto": ("Onto", "every alternative is the winner for at least one profile."),
    "nondictatorship": ("Non-dictatorship", "there is no voter whose top-ranked alternative always wins."),
    "unanimity": ("Unanimity", "if every voter ranks the same alternative first, that alternative wins."),
    "pareto": ("Pareto efficiency", "the winner is never an alternative y for which some other alternative x is "
               "ranked above y by every voter."),
    "monotonic": ("Maskin monotonicity", "if x wins at a profile, and the profile changes so that every "
                  "alternative a voter ranked below x is still ranked below x by that voter, then x still wins."),
    "condorcet": ("Condorcet consistency", "if some alternative beats every other alternative in pairwise "
                  "majority comparisons (more voters rank it above the other than the other above it), it wins."),
    "anonymity": ("Anonymity", "permuting the voters' ballots among the voters never changes the winner."),
    "neutrality": ("Neutrality", "relabelling the alternatives in every ballot relabels the winner in the same "
                   "way (whenever the relabelled ballots are allowed)."),
    "condorcet_loser": ("Never a Condorcet loser", "an alternative that loses every pairwise majority comparison "
                        "never wins."),
}


@lru_cache(maxsize=512)
def satisfiable(kind: str, n: int, m: int, domain: str, output: str, axioms: tuple[str, ...]) -> bool:
    ballots = single_peaked(m) if domain == "single_peaked" else rankings(m)
    profiles = list(itertools.product(ballots, repeat=n))
    index = {P: i for i, P in enumerate(profiles)}
    cnf = CNF()
    alt_perms = [p for p in itertools.permutations(range(m))]

    def relabel_ballot(r: Ranking, pi) -> Ranking:
        return tuple(pi[a] for a in r)

    if kind == "rankings":
        outs = rankings(m) if output == "strict" else weak_orders(m)
        X = [[cnf.var() for _ in outs] for _ in profiles]
        for row in X:
            cnf.exactly_one(row)

        def above(o, x, y) -> bool:  # o puts x strictly above y
            return (o.index(x) < o.index(y)) if output == "strict" else (o[x] < o[y])

        # A[P][x][y]: social ranking at P puts x strictly above y
        A = {}
        for pi_, P in enumerate(profiles):
            for x, y in itertools.permutations(range(m), 2):
                v = cnf.var()
                cnf.iff_or(v, [X[pi_][k] for k, o in enumerate(outs) if above(o, x, y)])
                A[pi_, x, y] = v
        if "pareto" in axioms:
            for pi_, P in enumerate(profiles):
                for x, y in itertools.permutations(range(m), 2):
                    if all(prefers(r, x, y) for r in P):
                        cnf.add(A[pi_, x, y])
        if "iia" in axioms:
            for x, y in itertools.combinations(range(m), 2):
                groups: dict[tuple, list[int]] = {}
                for pi_, P in enumerate(profiles):
                    groups.setdefault(tuple(prefers(r, x, y) for r in P), []).append(pi_)
                for g in groups.values():
                    for a, b in zip(g, g[1:]):
                        cnf.equal(A[a, x, y], A[b, x, y])
                        cnf.equal(A[a, y, x], A[b, y, x])
        if "nondictatorship" in axioms:
            for i in range(n):
                lits = []
                for pi_, P in enumerate(profiles):
                    # society differs from voter i at P: some pair where i strictly prefers x>y but society does not
                    for x, y in itertools.permutations(range(m), 2):
                        if prefers(P[i], x, y):
                            lits.append(-A[pi_, x, y])
                cnf.add(*lits)
        if "anonymity" in axioms:
            for pi_, P in enumerate(profiles):
                for sigma in itertools.permutations(range(n)):
                    Q = tuple(P[s] for s in sigma)
                    for k in range(len(outs)):
                        cnf.equal(X[pi_][k], X[index[Q]][k])
        if "neutrality" in axioms:
            oi = {o: k for k, o in enumerate(outs)}
            for pi_, P in enumerate(profiles):
                for pi in alt_perms:
                    Q = tuple(relabel_ballot(r, pi) for r in P)
                    if Q not in index:
                        continue
                    for k, o in enumerate(outs):
                        if output == "strict":
                            o2 = tuple(pi[a] for a in o)
                        else:
                            o2 = tuple(o[pi.index(a)] for a in range(m))
                        cnf.add(-X[pi_][k], X[index[Q]][oi[o2]])
        if "unanimity" in axioms:
            for r in ballots:
                P = tuple([r] * n)
                o = r if output == "strict" else tuple(r.index(a) for a in range(m))
                cnf.add(X[index[P]][outs.index(o)])
        if "nonimposition" in axioms:
            for k in range(len(outs)):
                cnf.add(*[X[pi_][k] for pi_ in range(len(profiles))])
        if "nonconstant" in axioms:
            for k in range(len(outs)):
                cnf.add(*[-X[pi_][k] for pi_ in range(len(profiles))])
        return cnf.solve()

    # winners
    W = [[cnf.var() for _ in range(m)] for _ in profiles]
    for row in W:
        cnf.exactly_one(row)
    if "strategyproof" in axioms:
        for pi_, P in enumerate(profiles):
            for i in range(n):
                for r2 in ballots:
                    if r2 == P[i]:
                        continue
                    Q = P[:i] + (r2,) + P[i + 1:]
                    qi = index[Q]
                    for x, y in itertools.permutations(range(m), 2):
                        if prefers(P[i], y, x):  # true ballot prefers y to x: winning x at P, y at Q is a manipulation
                            cnf.add(-W[pi_][x], -W[qi][y])
    if "onto" in axioms:
        for x in range(m):
            cnf.add(*[W[pi_][x] for pi_ in range(len(profiles))])
    if "nondictatorship" in axioms:
        for i in range(n):
            cnf.add(*[-W[pi_][P[i][0]] for pi_, P in enumerate(profiles)])
    if "unanimity" in axioms:
        for pi_, P in enumerate(profiles):
            tops = {r[0] for r in P}
            if len(tops) == 1:
                cnf.add(W[pi_][tops.pop()])
    if "pareto" in axioms:
        for pi_, P in enumerate(profiles):
            for x, y in itertools.permutations(range(m), 2):
                if all(prefers(r, x, y) for r in P):
                    cnf.add(-W[pi_][y])
    if "monotonic" in axioms:
        below = {(r, x): frozenset(r[r.index(x) + 1:]) for r in ballots for x in range(m)}
        for pi_, P in enumerate(profiles):
            for x in range(m):
                for qi, Q in enumerate(profiles):
                    if qi != pi_ and all(below[P[i], x] <= below[Q[i], x] for i in range(n)):
                        cnf.add(-W[pi_][x], W[qi][x])

    def majority(P, x, y) -> int:
        return sum(prefers(r, x, y) for r in P) - sum(prefers(r, y, x) for r in P)

    if "condorcet" in axioms:
        for pi_, P in enumerate(profiles):
            for x in range(m):
                if all(majority(P, x, y) > 0 for y in range(m) if y != x):
                    cnf.add(W[pi_][x])
    if "condorcet_loser" in axioms:
        for pi_, P in enumerate(profiles):
            for x in range(m):
                if all(majority(P, x, y) < 0 for y in range(m) if y != x):
                    cnf.add(-W[pi_][x])
    if "anonymity" in axioms:
        for pi_, P in enumerate(profiles):
            for sigma in itertools.permutations(range(n)):
                Q = tuple(P[s] for s in sigma)
                for x in range(m):
                    cnf.equal(W[pi_][x], W[index[Q]][x])
    if "neutrality" in axioms:
        for pi_, P in enumerate(profiles):
            for pi in alt_perms:
                Q = tuple(relabel_ballot(r, pi) for r in P)
                if Q in index:
                    for x in range(m):
                        cnf.add(-W[pi_][x], W[index[Q]][pi[x]])
    return cnf.solve()


# ------------------------------------------------------------------------------------- questions
def _setting_text(kind: str, n: int, m: int, domain: str, output: str) -> str:
    names = ", ".join(ALTS[:m - 1]) + f" and {ALTS[m - 1]}"
    nb = len(single_peaked(m) if domain == "single_peaked" else rankings(m))
    if domain == "single_peaked":
        axis = " < ".join(ALTS[:m])
        ballots = (f"Each voter submits a strict ranking of the alternatives that is single-peaked on the axis "
                   f"{axis}: the voter has a favourite, and alternatives further from the favourite along the axis, "
                   f"on either side, are ranked lower ({nb} such rankings exist). Every combination of such ballots "
                   f"is possible ({nb ** n} profiles).")
    else:
        ballots = (f"Each voter submits a strict ranking of the alternatives ({nb} possible rankings), and every "
                   f"combination of ballots is possible ({nb ** n} profiles).")
    if kind == "rankings":
        what = ("A rule assigns to every profile a social ranking, which must be a strict ranking of all the "
                "alternatives (no ties)." if output == "strict" else
                "A rule assigns to every profile a social ranking, which is a ranking of all the alternatives that "
                "may contain ties (any weak order).")
    else:
        what = "A rule assigns to every profile exactly one winning alternative."
    return (f"There are {n} voters and {m} alternatives ({names}). {ballots} {what}")


def social_choice_draft(rng: random.Random, kind: str) -> Draft | None:
    n = choose(rng, [2, 3, 3])
    if kind == "rankings":
        m, output = 3, choose(rng, ["strict", "strict", "weak"])
        domain = choose(rng, ["full", "full", "full", "single_peaked"])
        pool = list(RANKING_AXIOMS)
        core = choose(rng, [("pareto", "iia", "nondictatorship"), ("pareto", "iia"), ("iia", "nondictatorship"),
                            ("anonymity", "neutrality"), ()])
    else:
        m = choose(rng, [3, 3, 3, 4]) if n == 2 else 3
        output = "winner"
        domain = choose(rng, ["full", "full", "full", "single_peaked"])
        pool = list(WINNER_AXIOMS)
        core = choose(rng, [("strategyproof", "onto", "nondictatorship"), ("unanimity", "monotonic", "nondictatorship"),
                            ("anonymity", "neutrality"), ("condorcet", "strategyproof"), ()])
    k = rng.randint(max(2, len(core)), min(len(pool), len(core) + 2))
    rest = [a for a in pool if a not in core]
    rng.shuffle(rest)
    axioms = tuple(sorted(set(core) | set(rest[: k - len(core)])))
    if len(axioms) < 2:
        return None
    sat = satisfiable(kind, n, m, domain, output, axioms)
    table = RANKING_AXIOMS if kind == "rankings" else WINNER_AXIOMS
    order = list(axioms)
    rng.shuffle(order)
    props = "\n".join(f"{i + 1}. {table[a][0]}: {table[a][1]}" for i, a in enumerate(order))
    question = (_setting_text(kind, n, m, domain, output) + "\n\nConsider the following properties of a rule:\n"
                + props + "\n\nDoes there exist a rule (for exactly this setting) that has all of these properties?")
    yes, no = "Yes: some rule has all of these properties", "No: no rule has all of these properties"
    return Draft(question=question, options=(yes, no) if sat else (no, yes),
                 params=dict(kind=kind, n=n, m=m, domain=domain, output=output, axioms=list(axioms),
                             winner="yes" if sat else "no"),
                 data={"satisfiable": sat}, source="SAT (CaDiCaL) over all rules")


@register_domain("socialchoice")
class SocialChoice(TestbedDomain):
    """Axiom-consistency questions about voting rules in small settings, decided by SAT."""

    name = "socialchoice"
    description = "Social choice: is a list of axioms jointly satisfiable in a small voting setting? (SAT)"
    FAMILIES = FAMILIES

    def generate(self, family, rng, index):
        return social_choice_draft(rng, family.split("/", 1)[1])

    def balance_keys(self, family):
        return ("yes", "no")
