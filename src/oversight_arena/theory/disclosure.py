"""Verifiable disclosure: why verified claims change everything (a numerical companion).

With *cheap talk* (unverifiable claims), a biased advocate's message is uninformative to a
rational judge (Crawford & Sobel 1982 with maximally biased senders). With *verifiable*
evidence, a sceptical judge treats silence as bad news and full disclosure *unravels*
(Grossman 1981; Milgrom 1981); with two opposed advocates and verifiable evidence, even a
naive judge can reach the full-information decision (Milgrom & Roberts 1986) — the
economic core of debate.

These functions compute a judge's posterior in the HiddenBits setting (L i.i.d. fair bits;
question: majority of ones?) under different judge models, given the verified bits revealed.
"""

from __future__ import annotations

from math import comb


def _prior_count(L: int) -> list[float]:
    return [comb(L, k) / 2**L for k in range(L + 1)]


def posterior_majority_naive(L: int, ones_shown: int, zeros_shown: int) -> float:
    """Naive judge: treats revealed bits as a random sample (ignores selection)."""
    rest = L - ones_shown - zeros_shown
    need = (L + 1) // 2 - ones_shown
    return sum(comb(rest, k) / 2**rest for k in range(max(need, 0), rest + 1)) if rest >= 0 else float(ones_shown * 2 > L)


def posterior_majority_selection_aware(L: int, ones_shown: int, zeros_shown: int, exact_ones: bool = False,
                                       exact_zeros: bool = False) -> float:
    """Sceptical judge: revealed favourable bits only prove *lower bounds* on counts
    (#ones ≥ ones_shown, #zeros ≥ zeros_shown); ``exact_*`` when an advocate revealed fewer
    than it could have (so no more favourable bits exist)."""
    pri = _prior_count(L)
    lo = ones_shown
    hi = L - zeros_shown
    if exact_ones:
        hi = min(hi, ones_shown)
    if exact_zeros:
        lo = max(lo, L - zeros_shown)
    mass = sum(pri[k] for k in range(lo, hi + 1))
    if mass == 0:
        return 0.5
    maj = sum(pri[k] for k in range(max(lo, (L + 1) // 2), hi + 1))
    return maj / mass


def propaganda_value(L: int, k: int, true_ones: int, side_yes: bool, aware: bool) -> float:
    """Judge's P(yes) after ONE advocate for ``side`` reveals up to k favourable verified bits."""
    fav = true_ones if side_yes else L - true_ones
    shown = min(k, fav)
    exact = shown < k
    if side_yes:
        return posterior_majority_selection_aware(L, shown, 0, exact_ones=exact) if aware else posterior_majority_naive(L, shown, 0)
    return posterior_majority_selection_aware(L, 0, shown, exact_zeros=exact) if aware else posterior_majority_naive(L, 0, shown)


def debate_value(L: int, k: int, true_ones: int, aware: bool) -> float:
    """Judge's P(yes) after both advocates reveal up to k favourable verified bits each."""
    o = min(k, true_ones)
    z = min(k, L - true_ones)
    if aware:
        return posterior_majority_selection_aware(L, o, z, exact_ones=o < k, exact_zeros=z < k)
    return posterior_majority_naive(L, o, z)


def asd_table(L: int = 15, ks: list[int] | None = None) -> list[dict]:
    """Exact expected ASD (log-score) over the uniform prior for propaganda vs debate, naive vs
    selection-aware judges, as a function of the verification budget k."""
    import math

    ks = ks or list(range(0, L + 1))
    pri = _prior_count(L)
    rows = []
    for k in ks:
        for aware in (False, True):
            asd_p = asd_d = 0.0
            for ones, w in enumerate(pri):
                yes = ones * 2 > L
                # propaganda: truthful advocate for the true side vs advocate for the false side
                py_true = propaganda_value(L, k, ones, side_yes=yes, aware=aware)
                py_false = propaganda_value(L, k, ones, side_yes=not yes, aware=aware)
                p_true_world = py_true if yes else 1 - py_true
                p_false_world = (1 - py_false) if yes else py_false
                asd_p += w * (math.log(max(p_true_world, 1e-6)) - math.log(max(p_false_world, 1e-6)))
                pd_yes = debate_value(L, k, ones, aware)
                p_true = pd_yes if yes else 1 - pd_yes
                asd_d += w * (math.log(max(p_true, 1e-6)) - math.log(max(1 - p_true, 1e-6)))
            rows.append({"k": k, "judge": "selection-aware" if aware else "naive", "propaganda_asd": asd_p, "debate_asd": asd_d})
    return rows
