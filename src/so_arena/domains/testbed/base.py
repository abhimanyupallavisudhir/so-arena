"""Shared machinery of the conceptual-reasoning testbed.

The testbed is a set of *families* of binary questions whose answers are fixed by computation, proof or
experiment, but which a reasoner without that computation has to settle by argument - the situation of
conceptual reasoning, with the answer key kept by the experimenter. It serves two experiments:

* **oversight evaluations**: run a mechanism (debate, consultancy, prover-estimator debate, ...) on items
  whose answer it never sees, and score its verdicts against the hidden truth;
* **essays as latents**: an essay about a family is scored by how it changes a frozen reader's predictions
  of the family's items. That needs *families* - several items that share one latent structure - and a
  family *card* that describes the setting without revealing outcomes.

Conventions every family follows:

* items have two answers ``A``/``B`` with values +1/-1 and ``GroundTruth.correct``; the true answer is
  ``A`` on (as close as possible to) half of each family's items, at positions drawn independently of
  anything in the question;
* ``metadata`` carries ``family`` (``"<domain>/<template>"``), ``tier`` (``"A"``: objective ground truth,
  ``"B"``: expert judgment), ``split`` (``"train"``/``"test"``, a stable hash of the item id) and the
  generating parameters; ground-truth quantities (expected payoffs, counts, SAT witnesses) live in
  ``ground_truth.data``;
* generation is deterministic in ``(seed, index)`` and needs no network.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from so_arena.core.items import AnswerOption, GroundTruth, TaskItem
from so_arena.domains.base import Domain

SPLITS = ("train", "test")


def split_of(item_id: str, test_fraction: float = 0.5) -> str:
    """The split an item belongs to: a stable hash of its id, so it never depends on what else is loaded."""
    h = int(hashlib.sha256(f"split|{item_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_fraction else "train"


@dataclass(frozen=True)
class Family:
    """A family of items sharing one latent structure.

    ``card`` is what an essay writer (or a debater preparing in advance) may read about the family: the kind
    of process the items describe and what is asked, never an outcome. ``tier`` is ``"A"`` when answers are
    objective (computed, proved or measured) and ``"B"`` when they are expert judgments.
    """

    name: str
    title: str
    card: str
    tier: str = "A"
    tags: tuple[str, ...] = ()


@dataclass
class Draft:
    """One generated question before it becomes an item: the two answers, which is true, and its record."""

    question: str
    options: tuple[str, str]  # (text of the answer that is TRUE, text of the answer that is FALSE)
    params: dict[str, Any]
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""


def make_item(domain: str, family: str, index: int, draft: Draft, *, seed: int, tier: str = "A",
              true_first: bool | None = None) -> TaskItem:
    """Turn a draft into an item: answers in a seeded random order (or ``true_first``), values +1/-1."""
    if true_first is None:
        true_first = random.Random(f"order|{domain}|{family}|{seed}|{index}").random() < 0.5
    true_text, false_text = draft.options
    texts = (true_text, false_text) if true_first else (false_text, true_text)
    correct = "A" if true_first else "B"
    answers = [AnswerOption(label=lab, text=t, value=1.0 if lab == correct else -1.0) for lab, t in zip("AB", texts)]
    iid = f"{family.replace('/', '-')}-s{seed}-{index:04d}"
    return TaskItem(
        id=iid, domain=domain, question=draft.question, answers=answers,
        ground_truth=GroundTruth(correct=correct, data=dict(draft.data), source=draft.source or family),
        metadata={"family": family, "tier": tier, "split": split_of(iid), "params": dict(draft.params)},
    )


def balanced_positions(n: int, seed: int, key: str) -> list[bool]:
    """``true_first`` flags for ``n`` items: exactly half (rounded) true-first, in a seeded random order."""
    flags = [i < (n + 1) // 2 for i in range(n)]
    random.Random(f"positions|{key}|{seed}").shuffle(flags)
    return flags


class TestbedDomain(Domain):
    """A domain whose items come from named families of generators.

    Subclasses define ``FAMILIES`` (name -> :class:`Family`) and ``generate(family, rng, index)``, which returns
    a :class:`Draft` or ``None`` to reject a draw (e.g. a near-tie). ``load`` draws ``per_family`` accepted
    drafts per family, balances which answer is true within each family (``balance="truth"``: the generator's
    own answer, e.g. "one-boxing wins", is the true one on half the items, by rejection; ``"none"`` keeps the
    natural base rate) and places the true answer first on half of them.
    """

    FAMILIES: dict[str, Family] = {}
    max_tries: int = 200

    def __init__(self, families: Sequence[str] | None = None, per_family: int = 40, seed: int = 0,
                 balance: str = "truth"):
        unknown = [f for f in families or [] if self._qualify(f) not in self.FAMILIES]
        if unknown:
            raise KeyError(f"{self.name}: unknown families {unknown}; available: {sorted(self.FAMILIES)}")
        self.family_names = [self._qualify(f) for f in families] if families else list(self.FAMILIES)
        self.per_family, self.seed, self.balance = per_family, seed, balance

    def _qualify(self, f: str) -> str:
        return f if "/" in f else f"{self.name}/{f}"

    # ---------------------------------------------------------------------------------------- API
    def families(self) -> dict[str, Family]:
        return {f: self.FAMILIES[f] for f in self.family_names}

    def generate(self, family: str, rng: random.Random, index: int) -> Draft | None:  # pragma: no cover
        raise NotImplementedError

    def truth_key(self, draft: Draft) -> Any:
        """What balancing balances: by default which of the generator's two canonical answers is true
        (``params["winner"]``)."""
        return draft.params.get("winner")

    def drafts(self, family: str, n: int, seed: int) -> list[Draft]:
        rng = random.Random(f"{self.name}|{family}|{seed}")
        out: list[Draft] = []
        counts: dict[Any, int] = {}
        tries = 0
        while len(out) < n and tries < n * self.max_tries:
            tries += 1
            d = self.generate(family, rng, len(out))
            if d is None:
                continue
            if self.balance == "truth":
                k = self.truth_key(d)
                if k is not None:
                    keys = self.balance_keys(family)
                    cap = -(-n // max(1, len(keys))) if keys else n
                    if counts.get(k, 0) >= cap:
                        continue
                    counts[k] = counts.get(k, 0) + 1
            out.append(d)
        if len(out) < n:
            raise RuntimeError(f"{family}: generated only {len(out)} of {n} items in {tries} draws")
        return out

    def balance_keys(self, family: str) -> tuple[Any, ...]:
        """The values ``truth_key`` takes in ``family`` (each gets an equal share of items)."""
        return ()

    def load(self, *, split: str = "all", limit: int | None = None, seed: int | None = None) -> list[TaskItem]:
        seed = self.seed if seed is None else seed
        items: list[TaskItem] = []
        for fam in self.family_names:
            drafts = self.drafts(fam, self.per_family, seed)
            firsts = balanced_positions(len(drafts), seed, fam)
            items += [make_item(self.name, fam, i, d, seed=seed, tier=self.FAMILIES[fam].tier, true_first=tf)
                      for i, (d, tf) in enumerate(zip(drafts, firsts))]
        if split in SPLITS:
            items = [it for it in items if it.metadata["split"] == split]
        elif split not in ("all", "test_all"):
            raise ValueError(f"split must be one of {SPLITS + ('all',)}, got {split!r}")
        return items[:limit] if limit is not None else items


def family_of(item: TaskItem) -> str:
    return str(item.metadata.get("family", item.domain))


def by_family(items: Iterable[TaskItem]) -> dict[str, list[TaskItem]]:
    out: dict[str, list[TaskItem]] = {}
    for it in items:
        out.setdefault(family_of(it), []).append(it)
    return out


def money(x: float) -> str:
    """$1,250 / $0.50 style rendering."""
    if abs(x - round(x)) < 1e-9:
        return f"${int(round(x)):,}"
    return f"${x:,.2f}"


def pct(p: float) -> str:
    s = f"{100 * p:.1f}".rstrip("0").rstrip(".")
    return f"{s}%"


def choose(rng: random.Random, xs: Sequence[Any]) -> Any:
    return xs[rng.randrange(len(xs))]


def uniform_grid(rng: random.Random, lo: float, hi: float, step: float) -> float:
    k = int(round((hi - lo) / step))
    return round(lo + step * rng.randint(0, k), 10)


Generator = Callable[[random.Random, int], "Draft | None"]
