"""The testbed as one domain: any mix of families from the testbed's domains.

    get_domain("testbed")                                   # every tier-A family, 40 items each
    get_domain("testbed", families=["decisions/newcomb", "rlhack/sensor"], per_family=60)
    get_domain("testbed", families=["decisions"])           # all families of a domain

Item ids, splits and ground truth are those of the underlying domain, so results are comparable whichever way
an item was loaded.
"""

from __future__ import annotations

from collections.abc import Sequence

from so_arena.core.items import TaskItem
from so_arena.domains.base import Domain, get_domain, register_domain
from so_arena.domains.testbed.base import Family

# domains whose families make up the testbed, and whether loading them needs no network or API calls
DOMAINS = ("decisions", "socialchoice", "numbers", "rlhack", "agents", "dtbench")
DEFAULT_FAMILIES = (
    "decisions/newcomb", "decisions/lesion", "decisions/twin", "decisions/mugging", "decisions/beauty",
    "decisions/driver", "decisions/damascus", "decisions/hitchhiker",
    "socialchoice/rankings", "socialchoice/winners",
    "numbers/prime_race", "numbers/counts", "numbers/collatz",
    "rlhack/sensor", "rlhack/fake_goal", "rlhack/watched_sensor",
    "agents/auction", "agents/sycophancy", "agents/persona",
)


def all_families() -> dict[str, Family]:
    out: dict[str, Family] = {}
    for d in DOMAINS:
        cls = type(get_domain(d))
        out.update(cls.FAMILIES)
    return out


def expand(families: Sequence[str] | None) -> list[str]:
    """Family names, with a bare domain name standing for all of its families."""
    fams = all_families()
    if not families:
        return list(DEFAULT_FAMILIES)
    out: list[str] = []
    for f in families:
        if f in fams:
            out.append(f)
        elif f in DOMAINS:
            out += [k for k in fams if k.startswith(f + "/")]
        else:
            raise KeyError(f"unknown testbed family {f!r}; available: {sorted(fams)}")
    return out


@register_domain("testbed")
class Testbed(Domain):
    """Any mix of testbed families (module docstring)."""

    name = "testbed"
    description = "The conceptual-reasoning testbed: families of questions with ground truth from computation or experiment."

    def __init__(self, families: Sequence[str] | None = None, per_family: int = 40, seed: int = 0):
        self.family_names = expand(families)
        self.per_family, self.seed = per_family, seed

    def families(self) -> dict[str, Family]:
        fams = all_families()
        return {f: fams[f] for f in self.family_names}

    def load(self, *, split: str = "all", limit: int | None = None, seed: int | None = None) -> list[TaskItem]:
        seed = self.seed if seed is None else seed
        by_domain: dict[str, list[str]] = {}
        for f in self.family_names:
            by_domain.setdefault(f.split("/", 1)[0], []).append(f)
        items: list[TaskItem] = []
        for d, fams in by_domain.items():
            dom = get_domain(d, families=fams, seed=seed, per_family=self.per_family)
            items += dom.load(split=split, seed=seed)
        return items[:limit] if limit is not None else items
