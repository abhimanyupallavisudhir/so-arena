"""Empirical normal-form games: regret, equilibria, coalitions and learning dynamics.

All guarantees are restricted to the supplied action sets and measured payoffs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from .optimization import distribution
from .types import finite


@dataclass
class EmpiricalGame:
    players: tuple[str, ...]
    actions: tuple[tuple[str, ...], ...]
    payoffs: Mapping[tuple[str, ...], tuple[float, ...]]

    def __post_init__(self):
        if (
            not self.players
            or len(set(self.players)) != len(self.players)
            or len(self.actions) != len(self.players)
            or any(not a or len(set(a)) != len(a) for a in self.actions)
        ):
            raise ValueError("Invalid players/action spaces")
        if set(self.payoffs) != set(product(*self.actions)):
            raise ValueError("Payoff table must cover the full Cartesian product")
        for payoff in self.payoffs.values():
            if len(payoff) != len(self.players):
                raise ValueError("Every payoff needs one reward per player")
            for value in payoff:
                finite(value)

    def expected(self, policies: Sequence[Sequence[float]]) -> tuple[float, ...]:
        if len(policies) != len(self.players):
            raise ValueError("One mixed policy is required per player")
        policies = [distribution(p) for p in policies]
        if any(len(p) != len(a) for p, a in zip(policies, self.actions, strict=True)):
            raise ValueError("Policy size mismatch")
        result = [0.0] * len(self.players)
        for indices in product(*(range(len(a)) for a in self.actions)):
            mass = 1.0
            for i, j in enumerate(indices):
                mass *= policies[i][j]
            profile = tuple(self.actions[i][j] for i, j in enumerate(indices))
            for i, reward in enumerate(self.payoffs[profile]):
                result[i] += mass * reward
        return tuple(result)

    def regrets(self, policies: Sequence[Sequence[float]]) -> dict[str, float]:
        base = self.expected(policies)
        regrets = {}
        for i, player in enumerate(self.players):
            utilities = []
            for j in range(len(self.actions[i])):
                deviation = list(policies)
                deviation[i] = [float(k == j) for k in range(len(self.actions[i]))]
                utilities.append(self.expected(deviation)[i])
            regrets[player] = max(0.0, max(utilities) - base[i])
        return regrets

    def pure_equilibria(self, epsilon: float = 0) -> list[tuple[str, ...]]:
        if finite(epsilon) < 0:
            raise ValueError("epsilon must be nonnegative")
        return [
            profile
            for profile, payoffs in self.payoffs.items()
            if all(
                self.payoffs[profile[:i] + (action,) + profile[i + 1 :]][i] <= payoffs[i] + epsilon
                for i, actions in enumerate(self.actions)
                for action in actions
            )
        ]

    def coalition_gain(self, profile: tuple[str, ...], coalition: Sequence[str]) -> float:
        """Largest minimum member gain from a joint deviation, without transferable utility."""
        if not coalition or len(set(coalition)) != len(coalition):
            raise ValueError("Need a nonempty distinct coalition")
        indices = [self.players.index(p) for p in coalition]
        base = self.payoffs[profile]
        gains = []
        for choices in product(*(self.actions[i] for i in indices)):
            deviation = list(profile)
            for i, action in zip(indices, choices, strict=True):
                deviation[i] = action
            gains.append(min(self.payoffs[tuple(deviation)][i] - base[i] for i in indices))
        return max(gains)

    def coarse_correlated_regret(self, joint: Mapping[tuple[str, ...], float]) -> dict[str, float]:
        """External regret of a joint distribution; preserves correlated opponent behavior."""
        if not set(joint) <= self.payoffs.keys():
            raise ValueError("Unknown profile")
        weights = distribution(list(joint.values()))
        joint = dict(zip(joint, weights, strict=True))
        result = {}
        for i, player in enumerate(self.players):
            base = sum(p * self.payoffs[s][i] for s, p in joint.items())
            result[player] = max(
                0.0,
                max(
                    sum(
                        p * self.payoffs[s[:i] + (action,) + s[i + 1 :]][i]
                        for s, p in joint.items()
                    )
                    - base
                    for action in self.actions[i]
                ),
            )
        return result

    def best_response_dynamics(
        self, start: tuple[str, ...], steps: int, schedule: str = "simultaneous"
    ) -> dict:
        if (
            start not in self.payoffs
            or steps < 0
            or schedule not in {"simultaneous", "alternating"}
        ):
            raise ValueError("Invalid dynamics configuration")
        current = start
        history = [current]
        for _ in range(steps):
            updated = list(current)
            for i, actions in enumerate(self.actions):
                reference = current if schedule == "simultaneous" else tuple(updated)
                best = max(
                    actions,
                    key=lambda a: self.payoffs[reference[:i] + (a,) + reference[i + 1 :]][i],
                )
                # Retain current action on a tie; avoid artificial cycles from tie ordering.
                if (
                    self.payoffs[reference[:i] + (best,) + reference[i + 1 :]][i]
                    > self.payoffs[reference][i]
                ):
                    updated[i] = best
            current = tuple(updated)
            history.append(current)
            if current in history[:-1]:
                return {
                    "history": history,
                    "cycle_start": history.index(current),
                    "equilibrium": current in self.pure_equilibria(),
                }
        return {
            "history": history,
            "cycle_start": None,
            "equilibrium": current in self.pure_equilibria(),
        }
