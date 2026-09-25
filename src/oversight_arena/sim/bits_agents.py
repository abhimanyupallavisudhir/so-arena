"""Programmatic agents for HiddenBits: parameterised advocates and Bayesian judges.

Advocate strategy parameters (``Strategy.params``):
    claims (int): bits revealed per turn.
    lie_rate (float): probability that a claim is a fabricated favourable bit when truthful
        favourable bits run out (or always, if ``lie_first``).
    lie_first (bool): fabricate even when true favourable bits are available.
    markup (bool): use verifiable ``<bit>`` markup (else plain-text claims, never verified).
    rebut (bool): after round 0, spend claims exposing bits that hurt the opponent's side.

Judge parameters: ``trust`` — credence that an *unverified* claim is true (0.5 = a rational
judge who knows advocates are biased ignores cheap talk; >0.5 = credulous). Refuted claims
make the judge distrust that speaker's remaining unverified claims (``punish_liars``).
"""

from __future__ import annotations

import math
from typing import Any

from ..agents.base import Action, Agent, Observation
from ..core.util import rng_for
from ..domains.synthetic import parse_bit_claims, poisson_binomial_tail


class BitAdvocate(Agent):
    def __init__(self, id: str = "bit_advocate", **defaults: Any):
        self.id = id
        self.defaults = {"claims": 3, "lie_rate": 0.0, "lie_first": False, "markup": True, "rebut": True, **defaults}

    def describe(self) -> dict[str, Any]:
        return {"type": "BitAdvocate", "id": self.id, **self.defaults}

    async def act(self, obs: Observation) -> Action:
        p = {**self.defaults, **obs.params}
        bits: list[int] = obs.private.get("bits") or []
        target = obs.target
        if target is None or not bits:  # e.g. open protocols: pick the honest answer
            ones = sum(bits)
            want = 1 if ones * 2 > len(bits) else 0
            target = next((o.id for o in obs.task.options if o.data.get("claim") == want), obs.task.option_ids[0])
        want = next(o.data.get("claim") for o in obs.task.options if o.id == target)
        rng = obs.rng("advocate")
        revealed = set()
        for e in obs.entries:
            for i, _, _ in parse_bit_claims(e):
                revealed.add(i)
        n = int(p["claims"])
        true_fav = [i for i, b in enumerate(bits) if b == want and i not in revealed]
        unfav = [i for i, b in enumerate(bits) if b != want and i not in revealed]
        rng.shuffle(true_fav)
        rng.shuffle(unfav)
        claims: list[tuple[int, int]] = []
        for _ in range(n):
            lie = (p["lie_first"] or not true_fav) and unfav and rng.random() < float(p["lie_rate"])
            if lie:
                claims.append((unfav.pop(), want))
            elif true_fav:
                claims.append((true_fav.pop(), want))
        if p["markup"]:
            body = " ".join(f'<bit i="{i}">{v}</bit>' for i, v in claims)
        else:
            body = " ".join(f"bit {i} is {v};" for i, v in claims)
        ans = next(o.text for o in obs.task.options if o.id == target)
        text = f"The answer is ({target}) {ans}. Evidence: {body}" if claims else f"The answer is ({target}) {ans}."
        parsed: dict[str, Any] = {}
        if obs.response.kind == "choice":
            parsed = {"choice": target}
        return Action(text=text, parsed=parsed)


class BayesianBitJudge(Agent):
    """Exact Bayesian judge over the hidden bits (prior Bernoulli(0.5) i.i.d.)."""

    def __init__(self, trust: float = 0.5, punish_liars: bool = True, prior: float = 0.5, id: str | None = None):
        self.trust = trust
        self.punish_liars = punish_liars
        self.prior = prior
        self.id = id or f"bayes_judge(trust={trust})"

    def describe(self) -> dict[str, Any]:
        return {"type": "BayesianBitJudge", "trust": self.trust, "punish_liars": self.punish_liars, "prior": self.prior}

    def posterior_yes(self, obs: Observation) -> float:
        L = int(obs.task.data.get("length", 15))
        known: dict[int, int] = {}
        liars: set[str] = set()
        cheap: list[tuple[str, int, int]] = []
        for e in obs.entries:
            for i, v, status in parse_bit_claims(e):
                if status == "VERIFIED":
                    known[i] = v
                elif status == "REFUTED":
                    known[i] = 1 - v
                    liars.add(e.role or "")
                else:
                    cheap.append((e.role or "", i, v))
            for m in _plain_claims(e.content):
                cheap.append((e.role or "", *m))
        ps = []
        for i in range(L):
            if i in known:
                ps.append(float(known[i]))
                continue
            logit = math.log(self.prior / (1 - self.prior))
            for who, j, v in cheap:
                if j != i:
                    continue
                t = 0.5 if (self.punish_liars and who in liars) else self.trust
                t = min(max(t, 1e-6), 1 - 1e-6)
                lr = math.log(t / (1 - t))
                logit += lr if v == 1 else -lr
            ps.append(1 / (1 + math.exp(-logit)))
        need = (L + 1) // 2
        return poisson_binomial_tail(ps, need)

    async def act(self, obs: Observation) -> Action:
        py = min(max(self.posterior_yes(obs), 1e-4), 1 - 1e-4)
        probs = {}
        for o in obs.task.options:
            probs[o.id] = py if o.data.get("claim") == 1 else 1 - py
        if obs.response.kind == "distribution":
            return Action(text=f"P(yes)={py:.3f}", parsed={"probs": probs, "choice": max(probs, key=probs.get)})
        if obs.response.kind == "choice":
            return Action(text="", parsed={"choice": max(probs, key=probs.get)})
        if obs.response.kind == "scalar":
            return Action(text="", parsed={obs.response.scalar_name: py * (obs.response.hi - obs.response.lo) + obs.response.lo})
        return Action(text="What do the remaining bits say?")  # questions (consultancy)


def _plain_claims(text: str) -> list[tuple[int, int]]:
    import re

    return [(int(a), int(b)) for a, b in re.findall(r"bit (\d+) is ([01])", text)]
