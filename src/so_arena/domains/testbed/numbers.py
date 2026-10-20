"""Heuristic number theory: statements settled by computation that a reasoner must settle by heuristic argument.

Mathematicians believe many statements on heuristic grounds long before proofs exist: the Hardy-Littlewood
prime-tuple conjectures, Chebyshev's bias in prime races, the expected growth of Collatz-like maps. Such
arguments - probabilistic models of the primes, singular series, expected drift - are conceptual reasoning
with a checkable answer. Each item states a concrete finite claim that a computer decides in seconds but a
reasoner (or a debater, or a judge) without one must argue about:

* ``numbers/prime_race``: are there more primes up to N in one residue class than in another? Chebyshev's
  bias favours quadratic non-residues, but not always, and races between two residues (or two non-residues)
  are subtler.
* ``numbers/counts``: is the number of twin primes, prime pairs (p, p + 6), Sophie Germain primes, primes of
  the form n^2 + 1, Goldbach representations of an even number, or sums of two squares up to N above a
  threshold? Thresholds sit 4-30% above or below the true count, so a crude estimate (N / log^2 N) fails
  where the Hardy-Littlewood estimate with its singular series succeeds.
* ``numbers/collatz``: under the map n -> n/2 (n even), a n + b (n odd), what share of starting values up to L
  escape past a bound within a number of steps? The expected drift log(a) - 2 log 2 per odd step separates
  contracting from expanding maps; cycles and small cases complicate it.

Answers are balanced within each family except ``prime_race``, whose base rate (non-residues usually lead) is
part of what is being tested.
"""

from __future__ import annotations

import math
import random
from functools import lru_cache

import numpy as np

from so_arena.domains.base import register_domain
from so_arena.domains.testbed.base import Draft, Family, TestbedDomain, choose, pct

FAMILIES = {
    "numbers/prime_race": Family(
        "numbers/prime_race", "Prime number races between residue classes",
        "Each item names a modulus q, two residue classes r1 and r2 coprime to q, and a bound N, and asks "
        "whether there are more primes p <= N with p = r1 (mod q) or with p = r2 (mod q). Items vary q (3 to "
        "13), the classes (quadratic residues and non-residues) and N (10^3 to 10^7).",
        tags=("number theory", "heuristics", "Chebyshev bias")),
    "numbers/counts": Family(
        "numbers/counts", "Counts of prime patterns and sums of two squares against thresholds",
        "Each item asks whether a count up to a bound N is above a stated threshold: twin primes (p, p+2), pairs "
        "(p, p+6), Sophie Germain primes (p, 2p+1), primes of the form n^2+1, the number of ways to write an even "
        "number as a sum of two primes, or integers that are sums of two squares. Thresholds lie some percent "
        "above or below the true count, so answering requires a quantitatively accurate estimate.",
        tags=("number theory", "heuristics", "Hardy-Littlewood")),
    "numbers/collatz": Family(
        "numbers/collatz", "Escape rates of generalized Collatz maps",
        "Each item defines a map n -> n/2 for even n and n -> a*n + b for odd n (a and b odd), a range of starting "
        "values 1..L, a step budget and an escape bound, and asks whether more than a stated share of starting "
        "values exceed the bound within the step budget. Items vary a (3 to 9), b, L, the budget and the share.",
        tags=("number theory", "heuristics", "dynamics")),
}

SIEVE_LIMIT = 20_000_000


@lru_cache(maxsize=1)
def sieve() -> np.ndarray:
    is_p = np.ones(SIEVE_LIMIT + 1, dtype=bool)
    is_p[:2] = False
    for i in range(2, int(SIEVE_LIMIT ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i:: i] = False
    return is_p


@lru_cache(maxsize=1)
def primes() -> np.ndarray:
    return np.flatnonzero(sieve())


def legendre_residue(r: int, q: int) -> bool:
    """r is a square modulo q (r coprime to q)."""
    return any((x * x - r) % q == 0 for x in range(q))


# ----------------------------------------------------------------------------------------- races
def prime_race(rng: random.Random) -> Draft | None:
    q = choose(rng, [3, 4, 5, 7, 8, 11, 12, 13])
    units = [r for r in range(1, q) if math.gcd(r, q) == 1]
    r1, r2 = rng.sample(units, 2)
    N = int(choose(rng, [10 ** e for e in (3, 4, 5, 6, 7)]) * choose(rng, [1, 2, 3, 5]))
    N = min(N, SIEVE_LIMIT)
    ps = primes()
    ps = ps[ps <= N]
    c1, c2 = int(np.sum(ps % q == r1)), int(np.sum(ps % q == r2))
    if c1 == c2:
        return None
    q1, q2 = legendre_residue(r1, q), legendre_residue(r2, q)
    first, second = (r1, r2) if c1 > c2 else (r2, r1)
    question = (f"Consider the primes p with p <= {N:,}. Are there more such primes with p = {r1} (mod {q}) or "
                f"with p = {r2} (mod {q})? (Exact counts are compared; they are not equal.)")
    opts = (f"More primes with p = {first} (mod {q})", f"More primes with p = {second} (mod {q})")
    kind = "residue_vs_nonresidue" if q1 != q2 else ("both_residues" if q1 else "both_nonresidues")
    winner_nonres = (not legendre_residue(first, q)) if q1 != q2 else None
    return Draft(question=question, options=opts,
                 params=dict(q=q, r1=r1, r2=r2, N=N, kind=kind, nonresidue_leads=winner_nonres, winner=None),
                 data={"counts": {str(r1): c1, str(r2): c2}}, source="sieve")


# ----------------------------------------------------------------------------------------- counts
def _count(pattern: str, N: int) -> int:
    is_p = sieve()
    if pattern == "twin":
        p = primes()
        p = p[p <= N]
        return int(np.sum(is_p[p + 2]))
    if pattern == "sexy":
        p = primes()
        p = p[p <= N]
        return int(np.sum(is_p[p + 6]))
    if pattern == "sophie":
        p = primes()
        p = p[p <= N]
        return int(np.sum(is_p[2 * p + 1]))
    if pattern == "square_plus_one":
        n = np.arange(1, N + 1, dtype=np.int64)
        return int(np.sum(is_p[n * n + 1]))
    if pattern == "goldbach":
        p = primes()
        p = p[p <= N // 2]
        return int(np.sum(is_p[N - p]))
    if pattern == "two_squares":
        n = np.arange(1, N + 1)
        ok = np.zeros(N + 1, dtype=bool)
        a = 0
        while a * a <= N:
            b = np.arange(a, int(math.isqrt(N - a * a)) + 1)
            ok[a * a + b * b] = True
            a += 1
        return int(np.sum(ok[1:]))
    raise KeyError(pattern)


PATTERNS = {
    "twin": ("primes p <= {N:,} such that p + 2 is also prime", (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7)),
    "sexy": ("primes p <= {N:,} such that p + 6 is also prime", (10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7)),
    "sophie": ("primes p <= {N:,} such that 2p + 1 is also prime", (10 ** 4, 10 ** 5, 10 ** 6, 5 * 10 ** 6)),
    "square_plus_one": ("integers n with 1 <= n <= {N:,} such that n^2 + 1 is prime", (100, 1000, 3000, 4000)),
    "goldbach": ("ways to write {N:,} as p + q with p <= q both prime", None),
    "two_squares": ("integers m with 1 <= m <= {N:,} that are a sum of two squares of non-negative integers "
                    "(m = a^2 + b^2)", (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6)),
}


def counts(rng: random.Random) -> Draft | None:
    pattern = choose(rng, list(PATTERNS))
    text, sizes = PATTERNS[pattern]
    if pattern == "goldbach":
        N = 2 * rng.randint(5_000, 1_000_000)
    else:
        N = int(choose(rng, sizes) * choose(rng, [1, 1, 2, 3]))
        if pattern == "square_plus_one":
            N = min(N, 4000)
        if pattern in ("sophie",):
            N = min(N, SIEVE_LIMIT // 2 - 1)
        if pattern in ("twin", "sexy"):
            N = min(N, SIEVE_LIMIT - 10)
    true = _count(pattern, N)
    if true < 20:
        return None
    delta = choose(rng, [-1, 1]) * rng.uniform(0.04, 0.30)
    X = int(round(true * (1 + delta)))
    if X == true:
        return None
    above = true > X
    what = text.format(N=N)
    question = f"Let K be the number of {what}. Is K greater than {X:,}?"
    yes, no = f"Yes: K > {X:,}", f"No: K <= {X:,}"
    return Draft(question=question, options=(yes, no) if above else (no, yes),
                 params=dict(pattern=pattern, N=N, threshold=X, delta=round(delta, 4), winner="above" if above else "below"),
                 data={"count": true}, source="exact count")


# ----------------------------------------------------------------------------------------- collatz
def escape_share(a: int, b: int, L: int, steps: int, bound: int) -> float:
    n = np.arange(1, L + 1, dtype=np.int64)
    escaped = np.zeros(L, dtype=bool)
    for _ in range(steps):
        odd = (n & 1) == 1
        n = np.where(odd, a * n + b, n >> 1)
        escaped |= n > bound
        n = np.where(escaped, 1, n)  # stop tracking escaped values (keeps int64 safe)
    return float(escaped.mean())


def collatz(rng: random.Random) -> Draft | None:
    a = choose(rng, [3, 3, 5, 5, 7, 9])
    b = choose(rng, [1, 1, 3, 5, 7])
    L = choose(rng, [100, 1000, 10_000])
    steps = choose(rng, [100, 300, 1000])
    bound = choose(rng, [10 ** 9, 10 ** 12, 10 ** 15])
    share = escape_share(a, b, L, steps, bound)
    f = choose(rng, [0.01, 0.1, 0.25, 0.5, 0.75, 0.9])
    if abs(share - f) < 0.03:
        return None
    above = share > f
    question = (f"Define the map T(n) = n/2 if n is even, and T(n) = {a}n + {b} if n is odd. For each starting value "
                f"n = 1, 2, ..., {L:,}, apply T repeatedly, at most {steps:,} times. Say a starting value escapes if "
                f"some value in its trajectory (within those {steps:,} steps) exceeds {bound:,}. Do more than "
                f"{pct(f)} of the {L:,} starting values escape?")
    yes, no = f"Yes: more than {pct(f)} escape", f"No: at most {pct(f)} escape"
    return Draft(question=question, options=(yes, no) if above else (no, yes),
                 params=dict(a=a, b=b, L=L, steps=steps, bound=bound, f=f, winner="above" if above else "below"),
                 data={"escape_share": share}, source="direct iteration")


TEMPLATES = {"prime_race": prime_race, "counts": counts, "collatz": collatz}


@register_domain("numbers")
class HeuristicNumbers(TestbedDomain):
    """Finite number-theoretic claims decided by computation, to be argued heuristically (module docstring)."""

    name = "numbers"
    description = "Heuristic number theory: prime races, prime-pattern counts, Collatz-like escape rates."
    FAMILIES = FAMILIES

    def generate(self, family, rng, index):
        return TEMPLATES[family.split("/", 1)[1]](rng)

    def balance_keys(self, family):
        return () if family.endswith("prime_race") else ("above", "below")
