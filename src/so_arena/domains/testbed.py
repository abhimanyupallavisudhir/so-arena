"""The conceptual validation testbed as an so-arena domain.

The testbed - questions that must be argued about, with hidden ground truth from computation, proof or
experiment - lives in its own package, ``conceptual-validation-testbed`` (``pip install
"so-arena[testbed]"``). This adapter turns its questions into task items:

    get_domain("testbed")                                         # every tier-A family, 40 questions each
    get_domain("testbed", families=["decisions/newcomb", "multiagent"], per_family=16)

Item ids, splits and ground truth are the testbed's own. Questions whose experiments have not been run are
loaded only with ``status="unlabelled"`` and get pending ground truth.
"""

from __future__ import annotations

from collections.abc import Sequence

from so_arena.core.items import AnswerOption, GroundTruth, TaskItem
from so_arena.domains.base import Domain, register_domain


def to_item(q) -> TaskItem:
    labelled = q.status == "labelled"
    answers = [AnswerOption(label=lab, text=t, value=(1.0 if i == q.answer else -1.0) if labelled else None)
               for i, (lab, t) in enumerate(zip("AB", q.options))]
    gt = (GroundTruth(correct=q.label, data=dict(q.data), source=q.source) if labelled else
          GroundTruth(status="pending" if q.status == "unlabelled" else "unknown", data=dict(q.data), source=q.source))
    return TaskItem(id=q.id, domain=q.family.split("/", 1)[0], question=q.text, answers=answers, ground_truth=gt,
                    metadata={"family": q.family, "tier": q.tier, "split": q.split, "params": dict(q.params),
                              "status": q.status})


@register_domain("testbed")
class Testbed(Domain):
    """Questions from the conceptual validation testbed (``cvtestbed``) as task items."""

    name = "testbed"
    description = "The conceptual validation testbed: questions with hidden ground truth from computation or experiment."

    def __init__(self, families: Sequence[str] | None = None, per_family: int | None = 40, seed: int = 0,
                 status: str = "labelled"):
        import cvtestbed

        self.cvt = cvtestbed
        self.family_names = cvtestbed.expand(families)
        self.per_family, self.seed, self.status = per_family, seed, status

    def families(self):
        fams = self.cvt.families()
        return {f: fams[f] for f in self.family_names}

    def load(self, *, split: str = "all", limit: int | None = None, seed: int | None = None) -> list[TaskItem]:
        qs = self.cvt.load(self.family_names, n=self.per_family, seed=self.seed if seed is None else seed,
                           split=split, status=self.status)
        items = [to_item(q) for q in qs]
        return items[:limit] if limit is not None else items
