"""Question-answering datasets as oversight domains: GSM8K, MMLU, TruthfulQA, GPQA and QuALITY.

All share :class:`QADomain`: a dataset-specific :meth:`QADomain.records` yields :class:`QARecord`
objects (question, correct answer, distractors, private information) and the base class turns them
into task items, either

* binary (``binary=True``, the default): the correct answer vs one distractor with values +1/-1,
  the classic ASD / debate setup. The distractor is the dataset's own "best distractor" where it has
  one (TruthfulQA's best incorrect answer, QuALITY's annotator-voted distractor; ``distractor="best"``),
  else a deterministic random one; or
* full multiple choice (``binary=False``): +1 for the correct option, -1 for the others.

Option order is shuffled deterministically by ``(seed, item id)``, ids are stable (they identify the
underlying question, e.g. ``gsm8k-test-0042``), and ``limit`` takes a seeded random subset (kept in
dataset order), because dataset order is often not random (MMLU is grouped by subject).

Questions with an option that refers to other options ("All of the above", "Both A and B", "None of
these", ...) are dropped (:func:`refers_to_options`): once options are shuffled or reduced to a pair
such an option means something else (a true option becomes "wrong" next to "All of the above"; "A and
B only" can point at itself). That is 684 of the 14,042 MMLU test questions and 11 of the 2,086
QuALITY dev questions (2 of 2,523 train); ``QADomain.dropped_option_references`` counts them per load.

Data is downloaded on demand into :func:`so_arena.datasets.cache_dir`. GSM8K (MIT) ships a 50-item
sample, used automatically when offline. GPQA is gated: it needs ``HF_TOKEN``, and its authors ask
that examples not be published, so none are bundled and its items carry ``metadata["do_not_publish"]``.
"""

from __future__ import annotations

import ast
import csv
import http.client
import io
import json
import logging
import operator
import os
import random
import re
import string
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from so_arena.core.items import AnswerOption, GroundTruth, TaskItem
from so_arena.core.policy import stable_hash
from so_arena.core.verification import PythonExecVerifier, QuoteVerifier, Verifier
from so_arena.datasets import cache_dir, download, read_jsonl, sample_path
from so_arena.domains.base import Domain, register_domain

log = logging.getLogger("so_arena")

NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError, http.client.HTTPException)


class DatasetUnavailable(RuntimeError):
    """The dataset could not be fetched and no bundled sample covers the request."""


class GatedDatasetError(DatasetUnavailable):
    """A gated dataset was requested without (valid) credentials."""


_QUANTIFIER = r"(?:all|none|both|neither|either|any|each|one|two|three|some|more\s+than\s+one)"
_VERDICT = r"(?:correct|true|right|false|incorrect|wrong)"
_LETTER = r"\(?[a-e]\)?"
OPTION_REFERENCE = re.compile("|".join([
    _QUANTIFIER + r"\s+(?:\d+\s+)?of\s+(?:the\s+)?(?:above|below|foregoing)\b",  # "none of the above"
    r"\bthe\s+(?:above|foregoing)\b(?!-)",  # "... all the above" (not "the above-mentioned")
    _QUANTIFIER + r"\s+of\s+the\s+(?:options|choices|answers|alternatives)\b",
    # "these"/"those" refer to the options at the start or end of an option ("All of these", "... either
    # of these"), not in the middle ("each of these families")
    r"^\W*" + _QUANTIFIER + r"\s+of\s+(?:these|those)\b",
    r"\b" + _QUANTIFIER + r"\s+of\s+(?:these|those)\W*$",
    r"^\W*(?:all|none|both|neither|either|any)\s+of\s+them\W*$",
    r"^\W*(?:all|none|both|neither)\s+(?:the\s+)?(?:above|below|options?|choices?|answers?|alternatives?)\b",
    r"^\W*(?:all|none|both|neither)\s+(?:are|is)\s+" + _VERDICT + r"\b",
    r"\b(?:both|neither)\s+" + _LETTER + r"\s*(?:and|nor|&)\s*" + _LETTER
    + r"(?:\s+(?:are|is)\s+" + _VERDICT + r")?(?=\s*(?:[.;:,)]|$))",  # "both A and B", "neither (a) nor (b)."
    r"^\W*(?:both|neither)\W*$",
    r"^\W*(?:(?:only|options?|choices?|answers?)\s+)?" + _LETTER + r"(?:(?:\s*[,&+]\s*|\s+(?:and|or)\s+)"
    + _LETTER + r")+(?:\s+(?:only|are\s+" + _VERDICT + r"))?[\s.]*$",  # "A and C only", "(A) and (C)"
]), re.I)


def refers_to_options(text: str) -> bool:
    """Whether an answer option refers to other options ("All of the above", "Both A and B", "None of
    these", "Neither", "A and C only", ...). Such options are only meaningful with the full, original
    option list in its original order."""
    return bool(OPTION_REFERENCE.search(text.strip()))


@dataclass
class QARecord:
    """One question before it becomes a task item."""

    id: str
    question: str
    correct: str
    incorrect: list[str]
    best_distractor: str | None = None  # the dataset's own choice of distractor for binary items
    private: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)  # experimenter-side ground-truth data
    metadata: dict[str, Any] = field(default_factory=dict)


class QADomain(Domain):
    """Shared loader: dataset records -> binary or multiple-choice task items.

    Subclasses set ``splits`` (accepted name -> canonical split), ``default_split``, ``license``,
    optionally ``sample_file``/``sample_split`` (bundled offline sample), and implement :meth:`records`.
    """

    license: ClassVar[str] = ""
    splits: ClassVar[dict[str, str]] = {"test": "test"}
    default_split: ClassVar[str] = "test"
    sample_file: ClassVar[str | None] = None
    sample_split: ClassVar[str] = "test"

    def __init__(self, *, binary: bool = True, distractor: str = "best", seed: int = 0, offline: bool = False):
        """Args:
            binary: correct answer vs one distractor (+1/-1), or all options (+1 / -1 each).
            distractor: ``"best"`` (the dataset's best distractor when it has one) or ``"random"``.
            seed: default seed for option shuffles, distractor choice and ``limit`` subsets.
            offline: never touch the network (use the bundled sample, if any).
        """
        if distractor not in ("best", "random"):
            raise ValueError(f"distractor must be 'best' or 'random', not {distractor!r}")
        self.binary, self.distractor, self.seed, self.offline = binary, distractor, seed, offline
        self.used_source: str | None = None  # "download" or "sample", set by load()
        self.dropped_option_references: int | None = None  # questions dropped by the last load(), see below

    # ---------------------------------------------------------------------------- to implement
    def records(self, split: str) -> list[QARecord]:
        """All records of ``split``, fetched from the network (or the cache)."""
        raise NotImplementedError

    def sample_records(self, split: str) -> list[QARecord]:
        """Records from the bundled sample (only called if ``sample_file`` is set)."""
        raise NotImplementedError

    def keep(self, rec: QARecord) -> bool:
        """Dataset-specific filters (e.g. QuALITY's ``hard_only``)."""
        return True

    # ---------------------------------------------------------------------------- loading
    def canonical_split(self, split: str | None) -> str:
        if split is None:
            return self.default_split
        if split not in self.splits:
            raise ValueError(f"{self.name}: unknown split {split!r}; available: {sorted(self.splits)}")
        return self.splits[split]

    def load(self, *, split: str | None = None, limit: int | None = None, seed: int | None = None) -> list[TaskItem]:
        """Items of ``split``. Questions with an option that refers to other options are dropped (see
        :func:`refers_to_options`; the count is in ``dropped_option_references``): shuffling or pairing
        options changes what such an option means, so its value would be wrong."""
        seed = self.seed if seed is None else seed
        split = self.canonical_split(split)
        recs = self._fetch(split)
        recs = [r for r in recs if self.keep(r) and r.incorrect]
        n = len(recs)
        recs = [r for r in recs if not any(refers_to_options(o) for o in [r.correct, *r.incorrect])]
        self.dropped_option_references = n - len(recs)
        if n > len(recs):
            log.info("%s: dropped %d of %d questions with options that refer to other options", self.name,
                     n - len(recs), n)
        if limit is not None and limit < len(recs):
            rng = random.Random(stable_hash("qa-subset", self.name, split, seed))
            recs = [recs[i] for i in sorted(rng.sample(range(len(recs)), limit))]
        return [self.to_item(r, seed=seed, split=split) for r in recs]

    def _fetch(self, split: str) -> list[QARecord]:
        has_sample = self.sample_file is not None and split == self.sample_split
        if self.offline:
            if not has_sample:
                raise DatasetUnavailable(f"{self.name}: offline=True but there is no bundled sample for split {split!r}")
            self.used_source = "sample"
            return self.sample_records(split)
        try:
            recs = self.records(split)
            self.used_source = "download"
            return recs
        except NETWORK_ERRORS as e:
            if not has_sample:
                raise DatasetUnavailable(f"{self.name}: could not download split {split!r} ({e!r}) and no bundled "
                                         "sample covers it; check your connection or cache") from e
            log.warning("%s: download failed (%r); using the bundled %s sample", self.name, e, self.sample_file)
            self.used_source = "sample"
            return self.sample_records(split)

    def to_item(self, rec: QARecord, *, seed: int = 0, split: str | None = None) -> TaskItem:
        """Build the item: choose the distractor(s) and shuffle options, deterministically by (seed, id)."""
        rng = random.Random(stable_hash("qa-options", self.name, seed, rec.id))
        incorrect = [w for w in dict.fromkeys(rec.incorrect) if w.strip() != rec.correct.strip()]
        best = rec.best_distractor if rec.best_distractor and rec.best_distractor.strip() != rec.correct.strip() else None
        if self.binary:
            use_best = self.distractor == "best" and best is not None
            wrong = [best] if use_best else [rng.choice(incorrect)]
        else:
            wrong = incorrect
        opts = [(rec.correct, True)] + [(w, False) for w in wrong]
        rng.shuffle(opts)
        labels = string.ascii_uppercase
        answers = [AnswerOption(label=labels[i], text=t, value=1.0 if ok else -1.0) for i, (t, ok) in enumerate(opts)]
        correct = next(labels[i] for i, (_, ok) in enumerate(opts) if ok)
        metadata = {"dataset": self.name, "license": self.license, "split": split,
                    "variant": "binary" if self.binary else "multiple_choice", "option_seed": seed, **rec.metadata}
        if self.binary:
            metadata["distractor"] = "best" if (self.distractor == "best" and best is not None) else "random"
        return TaskItem(
            id=rec.id, domain=self.name, question=rec.question, answers=answers, context=dict(rec.context),
            private=dict(rec.private), metadata=metadata,
            ground_truth=GroundTruth(correct=correct, data={"answer": rec.correct, **rec.data}, source=self.name),
        )


# ------------------------------------------------------------------------------------ download helpers


def _fetch_to_cache(url: str, name: str, *, headers: dict[str, str] | None = None, timeout: float = 120.0) -> Path:
    """Like :func:`so_arena.datasets.download`, with request headers (for authenticated downloads)."""
    path = cache_dir() / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "so-arena/0.1", **(headers or {})})
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 16):
            f.write(chunk)
    tmp.replace(path)
    return path


def hf_url(repo: str, path: str, revision: str = "main") -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(path)}"


def hf_rows(repo: str, config: str, split: str, *, parquet_path: str | None = None) -> list[dict[str, Any]]:
    """Rows of a public Hugging Face dataset split, without the ``datasets`` library.

    Reads the repo's parquet file with ``pyarrow`` when installed, else pages through the
    datasets-server JSON API (100 rows per request). Both are cached.
    """
    if parquet_path is not None:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            pq = None
        if pq is not None:
            return pq.read_table(download(hf_url(repo, parquet_path))).to_pylist()
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", f"hfrows_{repo}_{config}_{split}.jsonl")
    path = cache_dir() / safe
    if path.exists():
        return read_jsonl(path)
    rows: list[dict[str, Any]] = []
    while True:
        q = urllib.parse.urlencode({"dataset": repo, "config": config, "split": split, "offset": len(rows), "length": 100})
        req = urllib.request.Request(f"https://datasets-server.huggingface.co/rows?{q}", headers={"User-Agent": "so-arena/0.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.loads(r.read().decode("utf-8"))
        rows += [x["row"] for x in page.get("rows", [])]
        if not page.get("rows") or len(rows) >= page.get("num_rows_total", 0):
            break
    tmp = path.with_suffix(".part")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(path)
    return rows


# ------------------------------------------------------------------------------------ GSM8K

GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"
_CALC = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_OP_SYMBOLS = "+-*/"
DISTRACTOR_KINDS = ("intermediate", "wrong_op", "factor", "perturb", "transpose")


def _eval_arith(expr: str) -> float | None:
    """Evaluate + - * / arithmetic safely (None if it is anything else)."""
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError(n)

    try:
        return ev(ast.parse(expr.replace(",", ""), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, RecursionError):
        return None


def _num(s: str) -> float | None:
    try:
        return float(s.replace(",", "").replace("$", "").strip())
    except ValueError:
        return None


def format_number(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.2f}".rstrip("0").rstrip(".")


def gsm8k_answer(solution: str) -> str:
    """The final answer after ``####`` (commas removed)."""
    return solution.rsplit("####", 1)[-1].strip().replace(",", "")


def gsm8k_clean_solution(solution: str) -> str:
    """Solution text without calculator annotations, ending in 'Answer: X'."""
    body, _, ans = solution.rpartition("####")
    return _CALC.sub("", body).strip() + f"\nAnswer: {ans.strip()}"


def gsm8k_wrong_answers(solution: str, correct: str) -> dict[str, list[str]]:
    """Plausible wrong answers by kind of slip, from the reference solution.

    * ``intermediate``: a dropped step - the result of an earlier calculation in the solution;
    * ``wrong_op``: the last calculation with one operator swapped (e.g. added instead of multiplied);
    * ``factor``: off by a factor of 2 or 10; ``perturb``: off by a little (+-1, +-2, +-10%);
    * ``transpose``: two adjacent digits swapped.

    Candidates keep the answer's sign and integrality and are within a factor of 10 of it, so none
    is absurd at a glance.
    """
    y = _num(correct)
    if y is None:
        return {}
    integral = abs(y - round(y)) < 1e-9

    def ok(v: float | None) -> bool:
        if v is None or abs(v - y) < 1e-9:
            return False
        if (y > 0 and v <= 0) or (y >= 0 and v < 0) or (y < 0 and v >= 0):
            return False
        if y != 0 and not 0.1 - 1e-9 <= v / y <= 10 + 1e-9:  # a slip rarely changes the magnitude more
            return False
        if y == 0 and abs(v) > 10:
            return False
        return not integral or abs(v - round(v)) < 1e-9

    out: dict[str, list[str]] = {}

    def add(kind: str, v: float | None) -> None:
        if ok(v):
            s = format_number(v)  # type: ignore[arg-type]
            if s != format_number(y) and s not in out.setdefault(kind, []):
                out[kind].append(s)

    calcs = _CALC.findall(solution)
    for _, res in calcs[:-1]:
        add("intermediate", _num(res))
    if calcs:
        expr = calcs[-1][0]
        for m in re.finditer(r"(?<=[\d.)\s])[-+*/](?=[\s\d.(])", expr):
            for op in _OP_SYMBOLS:
                if op != m.group():
                    add("wrong_op", _eval_arith(expr[: m.start()] + op + expr[m.end():]))
    for f in (2, 10):
        add("factor", y * f)
        add("factor", y / f)
    for d in (1, -1, 2, -2):
        add("perturb", y + d)
    tenth = round(abs(y) * 0.1)
    if tenth >= 3:
        add("perturb", y + tenth)
        add("perturb", y - tenth)
    digits = format_number(abs(y))
    if integral and len(digits) >= 2:
        for i in range(len(digits) - 1):
            if digits[i] != digits[i + 1] and not (i == 0 and digits[1] == "0"):
                sw = digits[:i] + digits[i + 1] + digits[i] + digits[i + 2:]
                add("transpose", float(sw) * (1 if y >= 0 else -1))
    return {k: v for k, v in out.items() if v}


@register_domain("gsm8k")
class GSM8K(QADomain):
    """Grade-school maths word problems (Cobbe et al. 2021): the numeric answer vs a plausible slip.

    Wrong answers are synthetic (see :func:`gsm8k_wrong_answers`): each item draws a kind of slip
    uniformly among those available for it, then a candidate, deterministically by (seed, id), and
    records it in ``metadata["distractor_kinds"]`` so results can be split by slip type. The
    reference solution is in ``private["solution"]`` (grant it to model informed experts); the
    ``python`` verifier lets agents make execution-checked arithmetic claims.
    """

    name = "gsm8k"
    description = "GSM8K maths word problems: the correct numeric answer vs a synthetic plausible wrong answer."
    license = "MIT (Copyright (c) 2021 OpenAI)"
    splits = {"test": "test", "train": "train"}
    default_split = "test"
    sample_file = "gsm8k_sample.jsonl"
    expert_affordances: list[str] = []

    def __init__(self, *, binary: bool = True, n_wrong: int = 3, kinds: Sequence[str] = DISTRACTOR_KINDS,
                 seed: int = 0, offline: bool = False):
        """``n_wrong``: wrong answers per item when ``binary=False`` (fewer if the solution offers fewer
        candidates); ``kinds``: allowed kinds of slip (see :data:`DISTRACTOR_KINDS`)."""
        super().__init__(binary=binary, distractor="random", seed=seed, offline=offline)
        unknown = set(kinds) - set(DISTRACTOR_KINDS)
        if unknown:
            raise ValueError(f"unknown distractor kinds {sorted(unknown)}; available: {DISTRACTOR_KINDS}")
        self.n_wrong, self.kinds = (1 if binary else n_wrong), tuple(kinds)

    def records(self, split):
        return self.parse_rows(read_jsonl(download(GSM8K_URL.format(split=split))), split)

    def sample_records(self, split):
        return self.parse_rows(read_jsonl(sample_path(self.sample_file)), split)

    def parse_rows(self, rows: Iterable[dict[str, Any]], split: str) -> list[QARecord]:
        """Records with every candidate wrong answer (``data["candidates"]``); :meth:`to_item` picks."""
        out = []
        for i, row in enumerate(rows):
            correct = gsm8k_answer(row["answer"])
            cands = {k: v for k, v in gsm8k_wrong_answers(row["answer"], correct).items() if k in self.kinds}
            if not cands:
                continue
            out.append(QARecord(
                id=f"gsm8k-{split}-{i:04d}", question=row["question"].strip(), correct=correct,
                incorrect=sorted({w for ws in cands.values() for w in ws}),
                private={"solution": gsm8k_clean_solution(row["answer"])},
                data={"numeric_answer": _num(correct), "candidates": cands}, metadata={"index": i},
            ))
        return out

    def choose_wrong(self, rec: QARecord, seed: int) -> tuple[list[str], list[str]]:
        """Wrong answers and their kinds: kinds of slip in random order (distinct kinds first), a random
        candidate of each."""
        cands = {k: list(v) for k, v in rec.data["candidates"].items()}
        rng = random.Random(stable_hash("gsm8k-wrong", seed, rec.id))
        order = sorted(cands)
        rng.shuffle(order)
        picks = [(k, rng.choice(cands[k])) for k in order]  # one per kind of slip first ...
        rest = [(k, v) for k in order for v in cands[k] if (k, v) not in picks]
        rng.shuffle(rest)  # ... then any remaining candidate
        wrong: list[str] = []
        kinds: list[str] = []
        for k, v in picks + rest:
            if len(wrong) < self.n_wrong and v not in wrong:
                wrong.append(v)
                kinds.append(k)
        return wrong, kinds

    def to_item(self, rec, *, seed=0, split=None):
        wrong, kinds = self.choose_wrong(rec, seed)
        data = {k: v for k, v in rec.data.items() if k != "candidates"}
        rec = QARecord(id=rec.id, question=rec.question, correct=rec.correct, incorrect=wrong, private=rec.private,
                       data={**data, "wrong_answers": wrong, "distractor_kinds": kinds},
                       metadata={**rec.metadata, "distractor_kinds": kinds})
        return super().to_item(rec, seed=seed, split=split)

    def verifiers(self) -> dict[str, Verifier]:
        return {"python": PythonExecVerifier()}


# ------------------------------------------------------------------------------------ MMLU

MMLU_SUBJECTS = (
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics",
    "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history",
    "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning",
    "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine",
    "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology",
    "world_religions",
)


@register_domain("mmlu")
class MMLU(QADomain):
    """MMLU (Hendrycks et al. 2021) from ``cais/mmlu``; ``subjects`` selects a subset of the 57 subjects.

    Questions with options such as "All of the above" or "Both A and B" are dropped (684 of the 14,042
    test questions), see :func:`refers_to_options`.
    """

    name = "mmlu"
    description = "MMLU multiple-choice questions (57 subjects); binary by default (answer vs one distractor)."
    license = "MIT"
    splits = {"test": "test", "validation": "validation", "val": "validation", "dev": "dev"}
    default_split = "test"
    repo = "cais/mmlu"

    def __init__(self, *, subjects: Sequence[str] | str | None = None, binary: bool = True, seed: int = 0,
                 offline: bool = False):
        super().__init__(binary=binary, distractor="random", seed=seed, offline=offline)
        subjects = [subjects] if isinstance(subjects, str) else list(subjects or MMLU_SUBJECTS)
        unknown = [s for s in subjects if s not in MMLU_SUBJECTS]
        if unknown:
            raise ValueError(f"unknown MMLU subjects {unknown}; available: {MMLU_SUBJECTS}")
        self.subjects = subjects

    def records(self, split):
        out = []
        for subject in self.subjects:
            rows = hf_rows(self.repo, subject, split, parquet_path=f"{subject}/{split}-00000-of-00001.parquet")
            out += self.parse_rows(rows, subject, split)
        return out

    @staticmethod
    def parse_rows(rows: Iterable[dict[str, Any]], subject: str, split: str) -> list[QARecord]:
        out = []
        for i, row in enumerate(rows):
            choices = [str(c) for c in row["choices"]]
            a = int(row["answer"])
            out.append(QARecord(
                id=f"mmlu-{subject}-{split}-{i:04d}",
                question=f"Subject: {subject.replace('_', ' ')}.\n\n{row['question'].strip()}",
                correct=choices[a], incorrect=[c for j, c in enumerate(choices) if j != a],
                metadata={"subject": subject, "index": i},
            ))
        return out


# ------------------------------------------------------------------------------------ TruthfulQA

# Pinned to the last commit that changed the file (2025-01-15: binary ``mc0_targets`` added, 817 -> 790
# questions), so item ids - row indices - always name the same questions.
TRUTHFULQA_COMMIT = "f6be04e52bbcb41d4d20daee6358231d4a5015d2"
TRUTHFULQA_URL = f"https://raw.githubusercontent.com/sylinrl/TruthfulQA/{TRUTHFULQA_COMMIT}/data/mc_task.json"


@register_domain("truthfulqa")
class TruthfulQA(QADomain):
    """TruthfulQA multiple choice (Lin et al. 2022), from the authors' ``mc_task.json`` at a pinned
    commit (:data:`TRUTHFULQA_COMMIT`, 790 questions); ids (``truthfulqa-0042``) are row indices there.

    Full multiple choice uses the MC1 targets (one true answer). Binary items use the authors'
    recommended binary format (``mc0_targets``: best answer vs best incorrect answer) unless
    ``distractor="random"``, which draws a random MC1 distractor.
    """

    name = "truthfulqa"
    description = "TruthfulQA (MC1): questions that some humans answer falsely due to misconceptions."
    license = "Apache-2.0"
    splits = {"validation": "validation", "test": "validation"}
    default_split = "validation"

    def __init__(self, *, binary: bool = True, distractor: str = "best", seed: int = 0, offline: bool = False):
        super().__init__(binary=binary, distractor=distractor, seed=seed, offline=offline)

    def records(self, split):
        with open(download(TRUTHFULQA_URL)) as f:
            return self.parse_rows(json.load(f))

    @staticmethod
    def parse_rows(rows: Iterable[dict[str, Any]]) -> list[QARecord]:
        out = []
        for i, row in enumerate(rows):
            mc1 = row["mc1_targets"]
            correct = [a for a, v in mc1.items() if v == 1]
            if len(correct) != 1:
                continue
            best = [a for a, v in (row.get("mc0_targets") or {}).items() if v == 0]
            out.append(QARecord(
                id=f"truthfulqa-{i:04d}", question=row["question"].strip(), correct=correct[0],
                incorrect=[a for a, v in mc1.items() if v == 0], best_distractor=best[0] if best else None,
                data={"true_answers": [a for a, v in (row.get("mc2_targets") or {}).items() if v == 1]},
                metadata={"index": i},
            ))
        return out


# ------------------------------------------------------------------------------------ GPQA

GPQA_REPO = "Idavidrein/gpqa"
GPQA_SUBSETS = ("diamond", "main", "extended")


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


@register_domain("gpqa")
class GPQA(QADomain):
    """GPQA (Rein et al. 2023): graduate-level science questions that are hard to look up.

    The dataset is gated on Hugging Face: accept its terms at
    https://huggingface.co/datasets/Idavidrein/gpqa and export ``HF_TOKEN``. Its authors ask that
    examples not be revealed online (to keep them out of training data), so nothing is bundled and
    every item carries ``metadata["do_not_publish"] = True``: redact these items when releasing
    transcripts.
    """

    name = "gpqa"
    description = "GPQA graduate-level science questions (gated; requires HF_TOKEN)."
    license = "CC BY 4.0 (gated; do not publish examples)"
    splits = {"train": "train", "test": "train"}
    default_split = "train"

    def __init__(self, *, subset: str = "diamond", binary: bool = True, seed: int = 0, offline: bool = False):
        super().__init__(binary=binary, distractor="random", seed=seed, offline=offline)
        if subset not in GPQA_SUBSETS:
            raise ValueError(f"GPQA subset must be one of {GPQA_SUBSETS}, not {subset!r}")
        self.subset = subset

    def records(self, split):
        token = hf_token()
        if not token:
            raise GatedDatasetError(
                "GPQA is a gated dataset: accept its terms at https://huggingface.co/datasets/Idavidrein/gpqa, "
                "then set the HF_TOKEN environment variable to a Hugging Face access token")
        fname = f"gpqa_{self.subset}.csv"
        try:
            path = _fetch_to_cache(hf_url(GPQA_REPO, fname), f"gated/{fname}", headers={"Authorization": f"Bearer {token}"})
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise GatedDatasetError(
                    f"Hugging Face refused HF_TOKEN for {GPQA_REPO} (HTTP {e.code}): make sure the token is valid "
                    "and that its account has accepted the dataset's terms") from e
            raise
        with open(path, encoding="utf-8-sig", newline="") as f:
            return self.parse_rows(csv.DictReader(f), self.subset)

    @staticmethod
    def parse_rows(rows: Iterable[dict[str, str]], subset: str) -> list[QARecord]:
        out = []
        for i, row in enumerate(rows):
            wrong = [row.get(f"Incorrect Answer {k}", "").strip() for k in (1, 2, 3)]
            rid = (row.get("Record ID") or "").strip() or f"{i:04d}"
            out.append(QARecord(
                id=f"gpqa-{subset}-{rid}", question=row["Question"].strip(), correct=row["Correct Answer"].strip(),
                incorrect=[w for w in wrong if w], private={"explanation": (row.get("Explanation") or "").strip()},
                metadata={"subset": subset, "index": i, "subdomain": row.get("Subdomain"),
                          "high_level_domain": row.get("High-level domain"), "do_not_publish": True},
            ))
        return out


# ------------------------------------------------------------------------------------ QuALITY

QUALITY_URL = "https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.zip"


@register_domain("quality")
class QuALITY(QADomain):
    """QuALITY v1.0.1 (Pang et al. 2022): long-story reading comprehension, for information-asymmetry debate.

    The story is private (``private["passage"]``, the experts' affordance); the question shows only
    the question and options, so a judge must rely on the agents - and on verified quotes
    (``<claim kind="quote">``, checked by :class:`~so_arena.core.verification.QuoteVerifier`), as in
    Michael et al. (2023) and Khan et al. (2024). Binary items pair the answer with the distractor
    most often chosen as "best distractor" by the dataset's untimed annotators.

    Filters: ``hard_only`` keeps the dataset's ``difficult`` subset (most time-limited annotators
    failed); ``min_untimed_accuracy`` (e.g. 1.0: every untimed annotator was right, so the answer is
    unambiguous) and ``max_speed_accuracy`` use the per-question annotations. The test split's labels
    are hidden, so ``dev`` (default; also used when ``"test"`` is requested) and ``train`` are available.
    Questions with options such as "All of the options are correct" are dropped (11 of the 2,086 dev
    questions, 2 of the 2,523 train questions), see :func:`refers_to_options`.
    """

    name = "quality"
    description = "QuALITY long-story questions; the story is private to the experts, quotes are verifiable."
    license = "CC BY 4.0"
    splits = {"dev": "dev", "validation": "dev", "train": "train"}
    default_split = "dev"
    expert_affordances = ["passage"]

    def __init__(self, *, binary: bool = True, distractor: str = "best", hard_only: bool = False,
                 min_untimed_accuracy: float | None = None, max_speed_accuracy: float | None = None,
                 seed: int = 0, offline: bool = False):
        super().__init__(binary=binary, distractor=distractor, seed=seed, offline=offline)
        self.hard_only, self.min_untimed_accuracy, self.max_speed_accuracy = hard_only, min_untimed_accuracy, max_speed_accuracy

    def canonical_split(self, split):
        if split == "test":  # generic callers ask for "test"; its labels are hidden, dev is the usual stand-in
            log.info("QuALITY's test labels are hidden: using the dev split")
            return "dev"
        return super().canonical_split(split)

    def records(self, split):
        out: list[QARecord] = []
        with zipfile.ZipFile(download(QUALITY_URL)) as z:
            name = next((n for n in z.namelist() if n.endswith(f"QuALITY.v1.0.1.htmlstripped.{split}")), None)
            if name is None:
                raise DatasetUnavailable(f"QuALITY archive has no {split!r} file; members: {z.namelist()}")
            with z.open(name) as f:
                for line in io.TextIOWrapper(f, encoding="utf-8"):
                    if line.strip():
                        out += self.parse_article(json.loads(line))
        return out

    @staticmethod
    def parse_article(article: dict[str, Any]) -> list[QARecord]:
        """Records for every labelled question of one QuALITY article (the dataset's JSON schema)."""
        passage = article["article"]
        kind = "a short story" if article.get("source") == "Gutenberg" else "an article"  # Slate etc. are non-fiction
        out = []
        for q in article.get("questions", []):
            gold = q.get("gold_label")
            if not gold:  # unlabelled (test split)
                continue
            opts = [str(o) for o in q["options"]]
            untimed = [v.get("untimed_answer") for v in q.get("validation", [])]
            speed = [v.get("speed_answer") for v in q.get("speed_validation", [])]
            votes = Counter(v.get("untimed_best_distractor") for v in q.get("validation", [])
                            if v.get("untimed_best_distractor") not in (None, gold))
            best = min(votes, key=lambda k: (-votes[k], k)) if votes else None
            out.append(QARecord(
                id=f"quality-{q['question_unique_id']}",
                question=f"This question is about {kind}.\n\n" + q["question"].strip(),
                correct=opts[gold - 1], incorrect=[o for j, o in enumerate(opts, 1) if j != gold],
                best_distractor=opts[best - 1] if best and 1 <= best <= len(opts) else None,
                private={"passage": passage},
                data={"writer_label": q.get("writer_label")},
                metadata={
                    "article_id": article.get("article_id"), "title": article.get("title"),
                    "author": article.get("author"), "source": article.get("source"),
                    "difficult": bool(q.get("difficult")), "passage_words": len(passage.split()),
                    "untimed_accuracy": sum(a == gold for a in untimed) / len(untimed) if untimed else None,
                    "speed_accuracy": sum(a == gold for a in speed) / len(speed) if speed else None,
                },
            ))
        return out

    def keep(self, rec):
        m = rec.metadata
        if self.hard_only and not m.get("difficult"):
            return False
        ua, sa = m.get("untimed_accuracy"), m.get("speed_accuracy")
        if self.min_untimed_accuracy is not None and (ua is None or ua < self.min_untimed_accuracy):
            return False
        if self.max_speed_accuracy is not None and (sa is None or sa > self.max_speed_accuracy):
            return False
        return True

    def verifiers(self) -> dict[str, Verifier]:
        return {"quote": QuoteVerifier(source_key="passage")}
