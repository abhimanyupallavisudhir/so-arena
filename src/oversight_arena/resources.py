"""Research covariates and cost planning; callers supply dated prices, never guessed rates."""

from dataclasses import dataclass, field

from .types import finite


@dataclass(frozen=True)
class ModelSpec:
    id: str
    revision: str
    family: str
    parameters: int | None = None
    training_flops: float | None = None
    inference: dict = field(default_factory=dict)
    domain_capability: dict[str, float] = field(default_factory=dict)
    source: str = "experimenter supplied"


@dataclass(frozen=True)
class Price:
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float
    source: str
    as_of: str

    def cost(self, *, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        rates = (self.input_per_million, self.output_per_million, self.cached_input_per_million)
        if any(finite(r) < 0 for r in rates) or not self.source or not self.as_of:
            raise ValueError("Nonnegative prices need a source and date")
        if min(input_tokens, output_tokens, cached_tokens) < 0 or cached_tokens > input_tokens:
            raise ValueError("Invalid token accounting")
        return (
            (input_tokens - cached_tokens) * self.input_per_million
            + cached_tokens * self.cached_input_per_million
            + output_tokens * self.output_per_million
        ) / 1e6


def tree_calls(branching: list[int]) -> int:
    """Model generations for a fully expanded debate tree, excluding leaf judges."""
    total, width = 0, 1
    for count in branching:
        if type(count) is not int or count < 1:
            raise ValueError("Branching factors must be positive integers")
        width *= count
        total += width
    return total
