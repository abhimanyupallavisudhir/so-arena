"""Exact finite-game diagnostics over an explicitly enumerated strategy support."""

from __future__ import annotations

import itertools
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from .core import finite
from .optimization import probabilities


@dataclass(frozen=True)
class FiniteGame:
    players: tuple[str, ...]
    strategies: tuple[tuple[str, ...], ...]
    payoffs: dict[tuple[int, ...], tuple[float, ...]]

    def __post_init__(self):
        if not self.players or len(set(self.players)) != len(self.players):
            raise ValueError("Unique players required")
        if len(self.strategies) != len(self.players):
            raise ValueError("Each player needs strategies")
        if any(not s or len(set(s)) != len(s) for s in self.strategies):
            raise ValueError("Nonempty unique strategies required")
        if set(self.payoffs) != set(self.profiles()):
            raise ValueError("Payoff table must cover the entire product support")
        for payoff in self.payoffs.values():
            if len(payoff) != len(self.players):
                raise ValueError("Missing player payoff")
            for value in payoff:
                finite(value)

    def profiles(self):
        return itertools.product(*(range(len(s)) for s in self.strategies))

    def expected(self, mixture: Sequence[Sequence[float]]) -> tuple[float, ...]:
        if len(mixture) != len(self.players):
            raise ValueError("Missing player mixture")
        weights = [probabilities(w, len(s)) for w, s in zip(mixture, self.strategies, strict=False)]
        totals = [0.0] * len(self.players)
        for profile, payoff in self.payoffs.items():
            mass = 1.0
            for player, action in enumerate(profile):
                mass *= weights[player][action]
            for player, value in enumerate(payoff):
                totals[player] += mass * value
        return tuple(totals)

    def regrets(self, mixture: Sequence[Sequence[float]]) -> dict[str, float]:
        """Unilateral regret for an independent mixed profile, not correlated equilibrium."""
        baseline = self.expected(mixture)
        result = {}
        for player, strategies in enumerate(self.strategies):
            gains = []
            for action in range(len(strategies)):
                alternative = [list(m) for m in mixture]
                alternative[player] = [float(j == action) for j in range(len(strategies))]
                gains.append(self.expected(alternative)[player] - baseline[player])
            result[self.players[player]] = max(0.0, max(gains))
        return result

    def pure_equilibria(self, tolerance: float = 1e-9) -> list[tuple[int, ...]]:
        if finite(tolerance) < 0:
            raise ValueError("Negative tolerance")
        return [
            p
            for p in self.profiles()
            if max(self.regrets(self.pure_mixture(p)).values()) <= tolerance
        ]

    def pure_mixture(self, profile: tuple[int, ...]) -> list[list[float]]:
        if profile not in self.payoffs:
            raise ValueError("Unknown strategy profile")
        return [
            [float(j == a) for j in range(len(s))]
            for a, s in zip(profile, self.strategies, strict=False)
        ]

    def coalition_gain(
        self, profile: tuple[int, ...], coalition: Sequence[str], *, transferable: bool = False
    ) -> dict:
        """Pure deviations; max minimum individual gain, or sum with explicit transfers."""
        if not coalition or len(set(coalition)) != len(coalition):
            raise ValueError("Coalition must contain distinct players")
        indices = [self.players.index(p) for p in coalition]
        baseline = self.payoffs[profile]
        best_gain, best = 0.0, profile
        for candidate, payoff in self.payoffs.items():
            if any(candidate[i] != profile[i] for i in range(len(profile)) if i not in indices):
                continue
            gains = [payoff[i] - baseline[i] for i in indices]
            gain = sum(gains) if transferable else min(gains)
            if gain > best_gain:
                best_gain, best = gain, candidate
        return {
            "gain": best_gain,
            "profile": best,
            "transferable": transferable,
            "support_only": True,
        }

    def best_response_dynamics(
        self,
        initial: tuple[int, ...],
        *,
        steps: int = 100,
        simultaneous: bool = True,
        seed: int = 0,
    ) -> dict:
        """No convergence claim: records cycles explicitly."""
        self.pure_mixture(initial)
        if steps < 1:
            raise ValueError("steps must be positive")
        rng = random.Random(seed)
        path, current = [initial], initial
        for _ in range(steps):
            updated = list(current)
            for i, strategies in enumerate(self.strategies):
                reference = current if simultaneous else tuple(updated)
                values = []
                for action in range(len(strategies)):
                    alternative = list(reference)
                    alternative[i] = action
                    values.append(self.payoffs[tuple(alternative)][i])
                maxima = [a for a, value in enumerate(values) if value == max(values)]
                updated[i] = reference[i] if reference[i] in maxima else rng.choice(maxima)
            new = tuple(updated)
            if new == current:
                return {"path": path + [new], "status": "fixed_point"}
            if new in path:
                return {"path": path + [new], "status": "cycle"}
            path.append(new)
            current = new
        return {"path": path, "status": "budget_exhausted"}


async def empirical_game(
    strategies: Mapping[str, Sequence[str]],
    payoff: Callable[[dict[str, str]], Awaitable[Mapping[str, float]]],
) -> FiniteGame:
    """The payoff callback can average run_episode rewards over tasks and seeds."""
    players = tuple(strategies)
    choices = tuple(tuple(strategies[p]) for p in players)
    table = {}
    for profile in itertools.product(*(range(len(s)) for s in choices)):
        result = await payoff({p: choices[i][profile[i]] for i, p in enumerate(players)})
        table[profile] = tuple(result[p] for p in players)
    return FiniteGame(players, choices, table)
