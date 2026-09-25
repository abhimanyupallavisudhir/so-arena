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
from ..core.util import rng_for, stable_hash
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
            errors = re.search(r":\d+:\d+: error\b|^error\b", msg, re.M)  # Lean's message format, not names
            ok = p.returncode == 0 and not errors and "declaration uses 'sorry'" not in msg
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
        ok, msg = await asyncio.to_thread(kernel_check, self.checker, env.resources.get("header", self.header), claim.content)
        return Evidence(verifier=self.name, kind=self.tag, claim=claim.content[:300],
                        result="kernel accepted" if ok else f"rejected: {msg[:400]}", verified=ok)

    def forge(self, claim: Claim, shown: Evidence, env: VerifyEnv) -> Evidence:
        if shown.verified is None:
            return shown
        return shown.model_copy(update={"result": "kernel accepted" if shown.verified else
                                        "rejected: error: unsolved goals"})


_SUB = "₀₁₂₃₄₅₆₇₈₉"
_HYP = re.compile(r"(?<![\w'.])h([₀-₉]+)(?![\w'])")


def renumber_hypotheses(stmt: str) -> str:
    """Hypotheses renamed h₀, h₁, ... in order of appearance, everywhere in the statement, so a
    gap in the numbering (e.g. left by a dropped hypothesis) cannot give an item away."""
    names = []
    for m in re.finditer(r"\(\s*h([₀-₉]+)\s*:", stmt):
        if m.group(1) not in names:
            names.append(m.group(1))
    new = {old: "".join(_SUB[int(c)] for c in str(i)) for i, old in enumerate(names)}
    return _HYP.sub(lambda m: "h" + new.get(m.group(1), m.group(1)), stmt) if new else stmt


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


# Escape hatches refused before the kernel runs: unproved goals, new axioms, code the kernel
# trusts without checking, disabling kernel checks, ending the file early, and metaprogramming
# (which can add axioms without the keyword). Matched on the raw text, comments included:
# stripping comments would let string literals hide code.
FORBIDDEN = re.compile(
    r"\b(sorry|sorryAx|admit|axiom|axioms|unsafe|implemented_by|extern|native_decide|ofReduceBool|ofReduceNat"
    r"|opaque|run_cmd|run_elab|run_meta|elab|elab_rules|macro|macro_rules|syntax|initialize|builtin_initialize"
    r"|addDecl|skipKernelTC)\b|#exit|@\[\s*(implemented_by|extern|csimp)")
STANDARD_AXIOMS = frozenset({"propext", "Classical.choice", "Quot.sound"})
_DECL = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)?(?:(?:private|protected|noncomputable)\s+)*(?:theorem|lemma)\s+([^\s:({\[]+)", re.M)


def nonstandard_axioms(msg: str) -> set[str]:
    """Axioms reported by ``#print axioms`` beyond Lean's standard three (e.g. ``sorryAx``)."""
    out: set[str] = set()
    for m in re.finditer(r"depends on axioms:\s*\[([^\]]*)\]", msg):
        out |= {a.strip() for a in m.group(1).split(",") if a.strip()} - STANDARD_AXIOMS
    return out


def kernel_check(checker: LeanChecker, header: str, code: str, before: str = "", after: str = "") -> tuple[bool, str]:
    """Static screen (:data:`FORBIDDEN`) of the untrusted ``code``, then the kernel on
    ``header + before + code + after`` (``before``/``after``: trusted checks), then an axiom audit:
    every theorem the code declares must depend only on the standard axioms (``#print axioms``)."""
    bad = FORBIDDEN.search(code)
    if bad:
        return False, f"uses {bad.group(0)!r}, which is not allowed"
    names = _DECL.findall(code)
    audit = "".join(f"\n#print axioms {n}" for n in names)
    parts = [header, before, code, after]
    ok, msg = checker.check("\n\n".join(x.strip() for x in parts if x.strip()) + "\n" + audit)
    if ok and nonstandard_axioms(msg):
        return False, f"depends on non-standard axioms: {', '.join(sorted(nonstandard_axioms(msg)))}"
    return ok, msg


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def statement_head(stmt: str) -> str:
    """The statement without its placeholder proof (``:= sorry`` / ``:= by sorry``)."""
    return _ws(re.sub(r":=\s*(?:by\s+)?sorry\s*$", "", stmt.strip()))


_OPEN, _CLOSE = "([{⦃", ")]}⦄"


def split_statement(stmt: str) -> tuple[str, str, str] | None:
    """``theorem NAME BINDERS : GOAL := sorry`` → (NAME, BINDERS, GOAL); None if unparseable."""
    m = re.match(r"\s*(?:theorem|lemma)\s+(\S+)(.*)$", statement_head(stmt), re.S)
    if not m:
        return None
    name, rest = m.group(1), m.group(2)
    depth = 0
    for i, ch in enumerate(rest):
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth -= 1
        elif ch == ":" and depth == 0 and rest[i + 1:i + 2] != "=":
            return name, rest[:i].strip(), rest[i + 1:].strip()
    return None


def statement_checks(stmt: str, tag: str) -> tuple[str, str] | None:
    """Trusted Lean text around a proof of ``stmt``: the statement elaborated as a definition
    *before* the proof (so notations or instances the proof declares cannot change its meaning),
    and an ``example`` after it that the declared theorem has exactly that type."""
    parts = split_statement(stmt)
    if parts is None:
        return None
    name, binders, goal = parts
    prop = f"∀ {binders}, {goal}" if binders else goal
    ref = f"_root_.OAStatement_{tag}"
    return f"def {ref} : Prop :=\n  {prop}", f"example : {ref} := @{name}"


def proves_statement(code: str, stmt: str) -> tuple[bool, str]:
    """Static screen before the kernel: no escape hatches (sorry, new axioms, unsafe/extern code,
    metaprogramming, ...) and the given theorem is declared. That the declared theorem has the
    given statement is then checked by the kernel (:func:`statement_checks`), not by matching
    text: a weakened theorem, or the statement in a comment or string, would pass a text match."""
    if FORBIDDEN.search(code):
        return False, f"uses a forbidden construct ({FORBIDDEN.search(code).group(0)!r})"
    parts = split_statement(stmt)
    if parts is None:
        return False, "could not parse the given statement"
    if parts[0] not in _DECL.findall(code):
        return False, f"does not declare the theorem {parts[0]}"
    return True, ""


class LeanProofGT(GTScorer):
    """1 iff the artifact proves *the given statement* (verbatim, no sorry/axioms) and the Lean
    kernel accepts it. Without the statement check a proof of ``True`` or of a weakened
    theorem would pass."""

    name: str = "proof_valid"

    def score(self, task: Task, record: EpisodeRecord) -> dict[str, float | None]:
        checker: LeanChecker | None = task.resources.get("_checker")
        if checker is None:
            return {}
        out: dict[str, float | None] = {}
        for role, art in (record.outcome.get("artifacts") or {}).items():
            m = re.search(r"```(?:lean4?|)\n(.*?)```", str(art), re.S)
            code = m.group(1) if m else str(art)
            stmt = task.resources.get("statement", "")
            ok, _ = proves_statement(code, stmt)
            checks = statement_checks(stmt, stable_hash(stmt, getattr(record, "id", ""), role, length=12))
            if ok and checks is not None:
                ok, _ = kernel_check(checker, task.resources.get("header", ""), code, before=checks[0], after=checks[1])
            out[role] = float(ok and checks is not None)
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
                shown = renumber_hypotheses(shown)  # faithful and perturbed alike: numbering is no tell
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
