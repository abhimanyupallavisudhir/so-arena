r"""Verifiable disclosure in the HiddenBits game (``docs/theory.md``, section 5, Proposition 3).

$L$ i.i.d. fair hidden bits; the question is whether at least half are 1 (at least $m = \lceil L/2 \rceil$
ones). An advocate for "yes" can reveal ones, an advocate for "no" zeros, each claim checked by a trusted
verifier, up to $k$ checked claims per advocate. The functions below give the judge's exact posterior
$P(\text{yes})$ after the advocates reveal verified favourable bits, for two judge models:

* **naive** - the revealed bits are a random sample: the unrevealed ones are still fair coins;
* **sceptical (selection-aware)** - revealed favourable bits only bound the counts from below
  ($\#\text{ones} \ge$ ones shown), and an advocate that showed fewer than $k$ had no more to show
  (the unraveling inference: silence is bad news).

Together with the log score they give the exact expected ASD of propaganda (one advocate) and debate (two
opposed advocates) as the verification budget $k$ grows (:func:`asd_table`). The policies
:func:`so_arena.domains.hidden_bits.bayesian_bit_judge` compute the same posteriors from what they are shown.
"""

from __future__ import annotations

import math
from math import comb


def need(L: int) -> int:
    """Ones needed for "at least half"."""
    return (L + 1) // 2


def count_prior(L: int) -> list[float]:
    """$P(\\#\\text{ones} = n)$ for $L$ fair bits."""
    return [comb(L, n) / 2**L for n in range(L + 1)]


def posterior_majority_naive(L: int, ones_shown: int, zeros_shown: int) -> float:
    """A naive judge's $P(\\text{yes})$: the unrevealed bits are fair coins (selection is ignored)."""
    rest = L - ones_shown - zeros_shown
    if rest < 0:
        raise ValueError("more bits shown than exist")
    missing = need(L) - ones_shown
    return sum(comb(rest, n) for n in range(max(missing, 0), rest + 1)) / 2**rest


def posterior_majority_selection_aware(L: int, ones_shown: int, zeros_shown: int, *, exact_ones: bool = False,
                                       exact_zeros: bool = False) -> float:
    """A sceptical judge's $P(\\text{yes})$: $\\#\\text{ones} \\ge$ ``ones_shown`` and $\\#\\text{zeros} \\ge$
    ``zeros_shown``, with equality where an advocate showed fewer than it could have (``exact_*``). Only the
    count matters under the exchangeable prior. Contradictory constraints (no mass) give 1/2."""
    pri = count_prior(L)
    lo, hi = ones_shown, L - zeros_shown
    if exact_ones:
        hi = min(hi, ones_shown)
    if exact_zeros:
        lo = max(lo, L - zeros_shown)
    mass = sum(pri[n] for n in range(lo, hi + 1))
    if mass <= 0:
        return 0.5
    return sum(pri[n] for n in range(max(lo, need(L)), hi + 1)) / mass


def propaganda_value(L: int, k: int, ones: int, side_yes: bool, aware: bool) -> float:
    """$P(\\text{yes})$ after one advocate for ``side_yes`` reveals as many favourable verified bits as it has,
    up to ``k``, when ``ones`` of the ``L`` bits are 1."""
    shown = min(k, ones if side_yes else L - ones)
    exact = shown < k
    if side_yes:
        return (posterior_majority_selection_aware(L, shown, 0, exact_ones=exact) if aware
                else posterior_majority_naive(L, shown, 0))
    return (posterior_majority_selection_aware(L, 0, shown, exact_zeros=exact) if aware
            else posterior_majority_naive(L, 0, shown))


def debate_value(L: int, k: int, ones: int, aware: bool) -> float:
    """$P(\\text{yes})$ after both advocates reveal up to ``k`` favourable verified bits each."""
    o, z = min(k, ones), min(k, L - ones)
    if aware:
        return posterior_majority_selection_aware(L, o, z, exact_ones=o < k, exact_zeros=z < k)
    return posterior_majority_naive(L, o, z)


def _log(p: float, eps: float) -> float:
    return math.log(min(max(p, eps), 1 - eps))


def asd_table(L: int = 15, ks: list[int] | None = None, *, eps: float = 1e-4) -> list[dict]:
    """Exact expected ASD with log-score rewards (clipped at ``eps``, as ``JudgeScore``) over the fair prior,
    for propaganda and debate, naive and sceptical judges, by verification budget ``k``.

    Propaganda: the reward of an advocate for the true answer minus that of an advocate for the false one.
    Debate: the true side's reward minus the false side's in the same debate."""
    ks = list(range(L + 1)) if ks is None else list(ks)
    pri = count_prior(L)
    rows = []
    for k in ks:
        for aware in (False, True):
            asd_p = asd_d = 0.0
            for ones, w in enumerate(pri):
                yes = ones >= need(L)

                def on_truth(py: float, yes: bool = yes) -> float:  # the judge's probability on the true answer
                    return py if yes else 1 - py

                asd_p += w * (_log(on_truth(propaganda_value(L, k, ones, yes, aware)), eps)
                              - _log(1 - on_truth(propaganda_value(L, k, ones, not yes, aware)), eps))
                p = on_truth(debate_value(L, k, ones, aware))
                asd_d += w * (_log(p, eps) - _log(1 - p, eps))
            rows.append({"k": k, "judge": "sceptical" if aware else "naive", "propaganda_asd": asd_p,
                         "debate_asd": asd_d})
    return rows
