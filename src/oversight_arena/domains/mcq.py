"""Generic multiple-choice domains from Hugging Face (knowledge / expertise gaps).

``HFMultipleChoice`` adapts any MCQ dataset; presets for MMLU-Pro and GPQA (gated: set
``HF_TOKEN`` and accept the terms). In ASD style, each task keeps the correct option and one
(seeded) distractor unless ``two_options=False``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from pydantic import Field

from ..core.task import Answer, Task
from ..core.util import rng_for
from .base import Domain


class HFMultipleChoice(Domain):
    name: ClassVar[str] = "hf_mcq"
    dataset: str
    split: str = "test"
    config: str = "default"
    question_field: str = "question"
    options_field: str = "options"
    answer_field: str = "answer_index"  # int index, or letter
    id_field: str | None = None
    category_field: str | None = None
    max_rows: int | None = 2000
    two_options: bool = True
    transform: Callable[[dict], dict] | None = Field(default=None, exclude=True)

    def load(self) -> list[Task]:
        from ..data import hf_rows

        rows = hf_rows(self.dataset, self.split, self.config, limit=self.max_rows)
        tasks = []
        for i, r in enumerate(rows):
            if self.transform:
                r = self.transform(r)
            opts = list(r[self.options_field])
            ans: Any = r[self.answer_field]
            gold = "ABCDEFGHIJKLMNOP".index(ans) if isinstance(ans, str) and len(ans) == 1 else int(ans)
            rng = rng_for(self.dataset, i)
            idx = list(range(len(opts)))
            if self.two_options:
                wrong = [j for j in idx if j != gold]
                idx = [gold, rng.choice(wrong)]
            rng.shuffle(idx)
            letters = "ABCDEFGHIJKLMNOP"
            options = [Answer(id=letters[k], text=str(opts[j]), value=1.0 if j == gold else -1.0) for k, j in enumerate(idx)]
            tid = str(r[self.id_field]) if self.id_field else str(i)
            tasks.append(Task(
                id=f"{self.dataset.split('/')[-1]}-{tid}", domain=self.dataset, question=r[self.question_field],
                options=options, metadata={"category": r.get(self.category_field) if self.category_field else None},
            ))
        return tasks


def MMLUPro(**kw: Any) -> HFMultipleChoice:
    return HFMultipleChoice(dataset="TIGER-Lab/MMLU-Pro", split="test", options_field="options",
                            answer_field="answer_index", id_field="question_id", category_field="category", **kw)


def GPQA(**kw: Any) -> HFMultipleChoice:
    """GPQA diamond (gated on HF: requires HF_TOKEN with access)."""

    def tf(r: dict) -> dict:
        opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        return {"question": r["Question"], "options": opts, "answer_index": 0}

    return HFMultipleChoice(dataset="Idavidrein/gpqa", config="gpqa_diamond", split="train", transform=tf, **kw)
