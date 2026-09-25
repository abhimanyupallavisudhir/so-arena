"""QuALITY reading comprehension with information asymmetry and verified quotes.

The classic debate setting (Khan et al. 2024; Michael et al. 2023): experts can read the
story, the judge cannot, and quotes are checked against the text. Included as a baseline for
comparability — it simulates the capability gap with an *information* gap. Distractors use
annotators' "best distractor" votes.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from typing import ClassVar

from ..channels.evidence import Claim, Verifier, VerifyEnv
from ..core.task import Answer, InfoBlock, Task
from ..core.transcript import Evidence
from ..core.util import rng_for
from .base import Domain

QUALITY_ZIP = "https://github.com/nyu-mll/quality/raw/main/data/v1.0.1/QuALITY.v1.0.1.zip"


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


class QuoteVerifier(Verifier):
    name = "quote_checker"
    tag = "quote"
    help = "an exact quote from the story, e.g. <quote>he never returned</quote>; checked verbatim (whitespace/quote-style insensitive)"

    def __init__(self, source_key: str = "article", cost: float = 0.0, max_words: int | None = 60):
        self.source_key = source_key
        self.cost = cost
        self.max_words = max_words

    async def verify(self, claim: Claim, env: VerifyEnv) -> Evidence:
        text = env.resources.get(self.source_key, "")
        q = claim.content
        if self.max_words and len(q.split()) > self.max_words:
            return Evidence(verifier=self.name, kind=self.tag, claim=q[:200], result=f"quote too long (>{self.max_words} words)", verified=False)
        ok = _norm(q) in _norm(text) if q.strip() else False
        return Evidence(verifier=self.name, kind=self.tag, claim=q, result="found verbatim in the source" if ok else "NOT found in the source", verified=ok)


class QuALITY(Domain):
    name: ClassVar[str] = "quality"
    split: str = "dev"
    hard_only: bool = False
    two_options: bool = True  # correct + best distractor (else all four options)
    max_article_words: int | None = None
    expert_clearance: list[str] = ["article"]
    judge_clearance: list[str] = []

    def load(self) -> list[Task]:
        from ..data import data_dir, download

        z = download(QUALITY_ZIP, "QuALITY.v1.0.1.zip")
        member = f"QuALITY.v1.0.1/QuALITY.v1.0.1.htmlstripped.{self.split}"
        target = data_dir() / f"quality.{self.split}.jsonl"
        if not target.exists():
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.endswith(f"htmlstripped.{self.split}")]
                target.write_bytes(zf.read(names[0] if names else member))
        tasks = []
        for line in target.read_text().splitlines():
            art = json.loads(line)
            article = art["article"]
            if self.max_article_words and len(article.split()) > self.max_article_words:
                continue
            for q in art["questions"]:
                if self.hard_only and not q.get("difficult"):
                    continue
                gold = q["gold_label"] - 1
                opts_text = q["options"]
                if self.two_options:
                    votes = Counter(v["untimed_best_distractor"] - 1 for v in q.get("validation", []) if v.get("untimed_best_distractor"))
                    votes.pop(gold, None)
                    distractor = votes.most_common(1)[0][0] if votes else next(i for i in range(len(opts_text)) if i != gold)
                    chosen = [gold, distractor]
                else:
                    chosen = list(range(len(opts_text)))
                rng = rng_for("quality", q["question_unique_id"])
                rng.shuffle(chosen)
                ids = "ABCD"
                opts = [Answer(id=ids[k], text=opts_text[i], value=1.0 if i == gold else -1.0) for k, i in enumerate(chosen)]
                dist_id = next(a.id for a in opts if a.value < 0)
                tasks.append(Task(
                    id=f"quality-{q['question_unique_id']}", domain=self.name, question=q["question"], options=opts,
                    info=[InfoBlock(key="article", title=f"Story: {art['title']}", content=article, access="article")],
                    resources={"article": article}, resource_access={"article": "article"},
                    gt={"distractor": dist_id, "difficult": q.get("difficult", 0)},
                    metadata={"title": art["title"], "difficult": q.get("difficult", 0)},
                ))
        return tasks

    def verifiers(self, task: Task) -> list[Verifier]:
        return [QuoteVerifier()]
