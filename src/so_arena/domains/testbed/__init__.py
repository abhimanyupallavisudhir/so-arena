"""The conceptual-reasoning testbed: families of questions settled by computation, proof or experiment.

See :mod:`so_arena.domains.testbed.base` for the conventions and ``docs/testbed.md`` for the families.
"""

from so_arena.domains.testbed.base import Family, TestbedDomain, by_family, family_of, split_of
from so_arena.domains.testbed import agents, decisions, dtbench, numbers, rlhack, socialchoice, suite  # noqa: F401  (register)

__all__ = ["Family", "TestbedDomain", "by_family", "family_of", "split_of"]
