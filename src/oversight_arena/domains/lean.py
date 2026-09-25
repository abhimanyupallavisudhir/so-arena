"""Lean 4 / miniF2F: formal proofs checked by the kernel — and the question the kernel *can't*
answer: does the formal statement mean what the English says?

Formats:
- ``faithfulness`` (default): given an informal problem and a Lean statement, is the
  formalisation faithful? Half the items use the original miniF2F formalisation (YES), half
  a *perturbed* one (constant changed, relation flipped, hypothesis dropped, ∧/∨ swapped: NO).
  GT is by construction; no Lean installation needed. Experts with a Lean checker can try to
  prove/disprove things; ``<lean>`` claims are kernel-checked.
- ``proof``: the proposer writes a proof (artifact); GT = kernel check (needs a checker).

Checkers: :class:`LocalLean` (``lake env lean`` inside a Mathlib project) or
:class:`KiminaLean` (a Kimina Lean Server URL). Without a checker, ``<lean>`` claims are
reported as unverifiable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from ..channels.evidence import Claim, Verifier, VerifyEnv
from ..core.episode import EpisodeRecord
from ..core.task import Answer, InfoBlock, Task
from ..core.tools import Tool, tool
from ..core.transcript import Evidence
from ..core.util import rng_for
from ..ground_truth.base import GTScorer
from .base import Domain


class LeanChecker(ABC):
    @abstractmethod
    def check(self, code: str, timeout: float = 60.0) -> tuple[bool, str]: ...


class LocalLean(LeanChecker):
    """Run ``lake env lean`` on a temp file inside a Lean project that has Mathlib."""

    def __init__(self, project_dir: str):
        self.project_dir = project_dir

    def check(self, code: str, timeout: float = 120.0) -> tuple[bool, str]:
        with tempfile.NamedTemporaryFile("w", suffix=".lean", dir=self.project_dir, delete=False) as f:
            f.write(code)
            path = f.name
        try:
            p = subprocess.run(["lake", "env", "lean", path], cwd=self.project_dir, capture_output=True, text=True, timeout=timeout)
            msg = (p.stdout + p.stderr).strip()
            ok = p.returncode == 0 and "error" not in msg.lower() and "sorry" not in msg.lower()
            return ok, msg[-3000:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        finally:
            os.unlink(path)


class KiminaLean(LeanChecker):
    """Client for a Kimina Lean Server (https://github.com/project-numina/kimina-lean-server)."""

    def __init__(self, url: str, api_key: str | None = None):
        self.url = url.rstrip("/")
        self.api_key = api_key or os.environ.get("KIMINA_API_KEY")

    def check(self, code: str, timeout: float = 60.0) -> tuple[bool, str]:
        body = json.dumps({"codes": [{"custom_id": "0", "proof": code}], "infotree_type": "original"}).encode()
        req = urllib.request.Request(f"{self.url}/verify", data=body, headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read().decode())
        item = (res.get("results") or [{}])[0]
        msgs = (item.get("response") or {}).get("messages", []) or []
        errs = [m for m in msgs if m.get("severity") == "error"]
        text = "\n".join(m.get("data", "") for m in msgs)[-3000:]
        sorry = any("sorry" in m.get("data", "") for m in msgs)
        return (not errs and not item.get("error") and not sorry), text or str(item.get("error") or "ok")


class LeanVerifier(Verifier):
    name = "lean_kernel"
    tag = "lean"
    help = "a complete Lean 4 snippet (e.g. a theorem with proof, using Mathlib), checked by the Lean kernel"

    def __init__(self, checker: LeanChecker | None, header: str = "", cost: float = 2.0):
        self.checker = checker
        self.header = header
        self.cost = cost

    async def verify(self, claim: Claim, env: VerifyEnv) -> Evidence:
        import asyncio

        if self.checker is None:
            return Evidence(verifier=self.name, kind=self.tag, claim=claim.content[:300], result="no Lean checker configured", verified=None)
        code = (env.resources.get("header", self.header) + "\n\n" + claim.content).strip()
        ok, msg = await asyncio.to_thread(self.checker.check, code)
        return Evidence(verifier=self.name, kind=self.tag, claim=claim.content[:300],
                        result="kernel accepted" if ok else f"rejected: {msg[:400]}", verified=ok)


def perturb_statement(stmt: str, seed: int) -> tuple[str, str] | None:
    """A meaning-changing perturbation of a Lean statement: (new statement, description).

    Constants / relations / connectives are only changed in the *conclusion* (so the change
    is material); hypotheses may be dropped. GT-by-construction is approximate — exactly the
    kind of label that, at scale, needs (expensive, possibly delayed) human checking.
    """
    rng = rng_for("lean-perturb", seed)
    m_concl = re.search(r"\)\s*:\s*\n?(.*?):=\s*(?:by\s+)?sorry\s*$", stmt, re.S)
    if m_concl is None:
        m_concl = re.search(r":\s*(.*?):=\s*(?:by\s+)?sorry\s*$", stmt, re.S)
    if m_concl is None:
        return None
    c0, c1 = m_concl.start(1), m_concl.end(1)
    concl = stmt[c0:c1]
    cands: list[tuple[str, list]] = []
    nums = list(re.finditer(r"(?<![\w.₀-₉^])(\d+)(?![\w.₀-₉])", concl))
    if nums:
        cands.append(("constant changed", nums))
    rels = list(re.finditer(r"≤|≥|<|>|≠", concl))
    if rels:
        cands.append(("relation changed", rels))
    conj = list(re.finditer(r"∧|∨", concl))
    if conj:
        cands.append(("connective changed", conj))
    hyps = list(re.finditer(r"\n\s*\((h[^\s:]*)\s*:[^\n]*\)", stmt[:c0]))
    if len(hyps) >= 2:
        cands.append(("hypothesis dropped", hyps))
    if not cands:
        return None
    kind, ms = rng.choice(cands)
    m = rng.choice(ms)
    if kind == "hypothesis dropped":
        out = stmt[: m.start()] + stmt[m.end():]
    else:
        if kind == "constant changed":
            v = int(m.group(1))
            rep = str(v + rng.choice([1, 2]) if v > 1 else v + 1)
        elif kind == "relation changed":
            rep = {"≤": "<", "<": "≤", "≥": ">", ">": "≥", "≠": "="}[m.group(0)]
        else:
            rep = "∨" if m.group(0) == "∧" else "∧"
        concl2 = concl[: m.start()] + rep + concl[m.end():]
        out = stmt[:c0] + concl2 + stmt[c1:]
    return (out, kind) if out != stmt else None


class LeanProofGT(GTScorer):
    name: str = "proof_valid"

    def score(self, task: Task, record: EpisodeRecord) -> dict[str, float | None]:
        checker: LeanChecker | None = task.resources.get("_checker")
        if checker is None:
            return {}
        out: dict[str, float | None] = {}
        for role, art in (record.outcome.get("artifacts") or {}).items():
            m = re.search(r"```(?:lean4?|)\n(.*?)```", str(art), re.S)
            code = m.group(1) if m else str(art)
            ok, _ = checker.check(task.resources.get("header", "") + "\n\n" + code)
            out[role] = float(ok)
        return out


class MiniF2F(Domain):
    name: ClassVar[str] = "minif2f"
    split: Literal["test", "validation"] = "test"
    format: Literal["faithfulness", "proof"] = "faithfulness"
    checker: Any = None  # LeanChecker instance
    expert_clearance: list[str] = ["lean"]
    judge_clearance: list[str] = []

    def load(self) -> list[Task]:
        from ..data import hf_rows

        rows = hf_rows("cat-searcher/minif2f-lean4", self.split)
        tasks = []
        for i, r in enumerate(rows):
            stmt = r["formal_statement"]
            header = r.get("header", "")
            if self.format == "faithfulness":
                rng = rng_for("minif2f", r["id"])
                faithful = rng.random() < 0.5
                shown, desc = stmt, None
                if not faithful:
                    p = perturb_statement(stmt, i)
                    if p is None:
                        faithful = True
                    else:
                        shown, desc = p
                opts = [Answer(id="YES", text="The formal statement is a faithful formalisation", value=1.0 if faithful else -1.0),
                        Answer(id="NO", text="The formal statement does not match the informal problem", value=-1.0 if faithful else 1.0)]
                tasks.append(Task(
                    id=f"minif2f-{r['id']}-faith", domain=self.name,
                    question="Does the Lean statement faithfully formalise the informal problem?",
                    options=opts,
                    info=[InfoBlock(key="informal", title="Informal problem", content=r["informal_stmt"]),
                          InfoBlock(key="formal", title="Lean 4 statement", content=f"```lean\n{shown}\n```")],
                    resources={"header": header, "statement": shown, "_checker": self.checker},
                    gt={"faithful": faithful, "perturbation": desc, "original": stmt},
                ))
            else:
                tasks.append(Task(
                    id=f"minif2f-{r['id']}-proof", domain=self.name,
                    question="Prove the following Lean 4 theorem. Give the complete theorem with proof in a ```lean block.",
                    info=[InfoBlock(key="informal", title="Informal problem", content=r["informal_stmt"]),
                          InfoBlock(key="formal", title="Lean 4 statement", content=f"```lean\n{stmt}\n```")],
                    answer_type="proof",
                    resources={"header": header, "statement": stmt, "_checker": self.checker},
                    gt={"informal_proof": r.get("informal_proof")},
                ))
        return tasks

    def tools(self, task: Task) -> list[Tool]:
        checker: LeanChecker | None = self.checker
        header = task.resources.get("header", "")

        @tool(name="lean_check", group="lean")
        def lean_check(code: str) -> str:
            """Check Lean 4 code (Mathlib available) with the Lean kernel; returns errors or 'ok'.

            Args:
                code: Lean 4 source (without the import header)
            """
            if checker is None:
                return "no Lean checker configured"
            ok, msg = checker.check(header + "\n\n" + code)
            return "ok" if ok else msg[:1500]

        return [lean_check]

    def verifiers(self, task: Task) -> list[Verifier]:
        return [LeanVerifier(self.checker, task.resources.get("header", ""))]

    def gt_scorers(self) -> list[GTScorer]:
        if self.format == "proof":
            from ..ground_truth.common import AcceptCorrect

            return [LeanProofGT(), AcceptCorrect(correctness="proof_valid")]
        return super().gt_scorers()
