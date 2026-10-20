"""DTBench: expert-written questions on decision-theoretic reasoning in Newcomb-like problems.

Oesterheld, Cooper, Kodama, Nguyen and Perez (2024), *A dataset of questions on decision-theoretic reasoning
in Newcomb-like problems* (MIT licence), the question set behind the Conceptual Reasoning Index's DTBench. It has

* 407 **capability** questions with a single correct answer, written by a decision theorist and checked by a
  second expert ("What does CDT recommend here?", "What is the evidential expected utility of one-boxing?");
* 130 **attitude** questions, whose answers are labelled by which theory (EDT or CDT) they agree with ("For the
  purpose of decision making, what is the expected utility of one-boxing?") - no correct answer, but a
  measurable disposition.

Capability questions are tier-A-like items (expert answers rather than computation; family
``dtbench/capabilities``). By default each becomes a two-option item, the correct answer against one distractor
drawn at random (``binary=False`` keeps every option). Attitude questions (``dtbench/attitudes``) have unknown
ground truth and carry the theory labels in ``ground_truth.data["theories"]``; they serve as a behavioural
battery (e.g. does a persona shift a model's EDT share?).

The data is distributed as a password-protected zip so that it stays out of web scrapes and training corpora.
It is downloaded on first use into the dataset cache and never committed; please do not republish it in plain
text.
"""

from __future__ import annotations

import random
import zipfile
from pathlib import Path
from typing import Any

from so_arena.core.items import AnswerOption, GroundTruth, TaskItem
from so_arena.datasets import cache_dir, download
from so_arena.domains.base import register_domain
from so_arena.domains.testbed.base import Family, TestbedDomain, balanced_positions, split_of

URL = "https://github.com/casparoe/newcomblike_questions_dataset/raw/main/benchmark/data.zip"
PASSWORD = b"onebox"  # published in the dataset's README

FAMILIES = {
    "dtbench/capabilities": Family(
        "dtbench/capabilities", "Expert questions on decision theory in Newcomb-like problems",
        "Questions written and checked by decision theorists about Newcomb-like problems: what causal or "
        "evidential decision theory recommends in a scenario, expected-utility calculations under each theory, "
        "which principles an argument relies on, multi-agent problems with copies and predictors, randomization, "
        "imperfect recall and ratifiability. Each has a single correct answer.",
        tier="A", tags=("decision theory", "expert-written")),
    "dtbench/attitudes": Family(
        "dtbench/attitudes", "Attitude questions: does the answer side with EDT or CDT?",
        "Questions about Newcomb-like problems whose answers are labelled by which decision theory (evidential or "
        "causal) they agree with. There is no correct answer; the share of EDT-consistent answers measures a "
        "disposition.",
        tier="B", tags=("decision theory", "attitudes")),
}


def extracted_dir() -> Path:
    d = cache_dir() / "dtbench"
    if not any(d.glob("setting*.json")):
        z = download(URL, "dtbench_data.zip")
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as f:
            f.extractall(d, pwd=PASSWORD)
    return d


def _flatten(prefix: str, setting: dict[str, Any], tags: list[str]) -> list[dict[str, Any]]:
    tags = tags + list(setting.get("tags", []))
    if "setup" in setting:
        out: list[dict[str, Any]] = []
        for s in setting["questions"]:
            out += _flatten(prefix + setting["setup"], s, tags)
        return out
    return [dict(setting, text=prefix + setting["question_text"], all_tags=tags)]


def questions() -> list[dict[str, Any]]:
    import json5

    out: list[dict[str, Any]] = []
    for f in sorted(extracted_dir().glob("setting*.json")):
        with open(f) as fh:
            out += _flatten("", json5.load(fh), [])
    return out


@register_domain("dtbench")
class DTBench(TestbedDomain):
    """The DTBench / Newcomb-like questions dataset (module docstring)."""

    name = "dtbench"
    description = "Expert decision-theory questions (DTBench): capabilities with answers, attitudes with theory labels."
    FAMILIES = FAMILIES

    EXCLUDED_TAGS = ("trivia", "rote memorization sufficient")

    def __init__(self, families=None, per_family: int | None = None, seed: int = 0, binary: bool = True,
                 exclude_tags: tuple[str, ...] | None = EXCLUDED_TAGS):
        """``exclude_tags`` drops questions answerable by recall rather than reasoning (by default those tagged
        trivia or rote memorization; ``()`` keeps all 407 capability questions)."""
        super().__init__(families=families or ["dtbench/capabilities"], per_family=per_family or 0, seed=seed,
                         balance="none")
        self.binary, self.exclude_tags = binary, tuple(exclude_tags or ())

    def load(self, *, split: str = "all", limit: int | None = None, seed: int | None = None) -> list[TaskItem]:
        seed = self.seed if seed is None else seed
        items: list[TaskItem] = []
        qs = questions()
        for fam in self.family_names:
            att = fam.endswith("attitudes")
            chosen = [q for q in qs if bool(q.get("attitude_q")) == att
                      and not set(q["all_tags"]) & set(self.exclude_tags)]
            firsts = balanced_positions(len(chosen), seed, fam)
            items += [self._item(q, fam, seed, tf) for q, tf in zip(chosen, firsts)]
        if self.per_family:
            rng = random.Random(f"dtbench|{seed}")
            rng.shuffle(items)
            kept: list[TaskItem] = []
            counts: dict[str, int] = {}
            for it in items:
                fam = it.metadata["family"]
                if counts.get(fam, 0) < self.per_family:
                    counts[fam] = counts.get(fam, 0) + 1
                    kept.append(it)
            items = kept
        if split in ("train", "test"):
            items = [it for it in items if it.metadata["split"] == split]
        return items[:limit] if limit is not None else items

    def _item(self, q: dict[str, Any], family: str, seed: int, true_first: bool) -> TaskItem:
        rng = random.Random(f"dtbench-item|{q['qid']}|{seed}")
        opts = list(q["permissible_answers"])
        iid = f"dtbench-{q['qid']}"
        meta = {"family": family, "tier": FAMILIES[family].tier, "split": split_of(iid), "qid": q["qid"],
                "tags": q["all_tags"], "n_options_original": len(opts)}
        if q.get("attitude_q"):
            order = list(range(len(opts)))
            rng.shuffle(order)
            labels = [chr(65 + i) for i in range(len(opts))]
            theories = {}
            for th, idx in q["correct_answer"].items():
                idx = idx if isinstance(idx, list) else [idx]
                theories[th] = [labels[order.index(i)] for i in idx]
            answers = [AnswerOption(label=labels[k], text=opts[i]) for k, i in enumerate(order)]
            return TaskItem(id=iid, domain="dtbench", question=q["text"].strip(), answers=answers,
                            ground_truth=GroundTruth(status="unknown", data={"theories": theories}), metadata=meta)
        correct = q["correct_answer"]
        if self.binary and len(opts) > 2:
            wrong = rng.choice([i for i in range(len(opts)) if i != correct])
            chosen = [correct, wrong] if true_first else [wrong, correct]
        elif len(opts) == 2:
            chosen = [correct, 1 - correct] if true_first else [1 - correct, correct]
        else:
            chosen = list(range(len(opts)))
            rng.shuffle(chosen)
        labels = [chr(65 + i) for i in range(len(chosen))]
        answers = [AnswerOption(label=lab, text=opts[i], value=1.0 if i == correct else -1.0)
                   for lab, i in zip(labels, chosen)]
        true = labels[chosen.index(correct)]
        return TaskItem(id=iid, domain="dtbench", question=q["text"].strip(), answers=answers,
                        ground_truth=GroundTruth(correct=true, source="expert (Oesterheld et al. 2024)"), metadata=meta)
