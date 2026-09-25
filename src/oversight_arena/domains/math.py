"""Math word problems with tool asymmetry (the ASD paper's GSM8K setting).

Correct answers come from GSM8K; plausible *incorrect* answers (with worked "proofs") are the
ones generated for the SOlib/ASD benchmark (downloaded on demand). Experts get a calculator /
Python tool; the judge does not — a tool-use capability gap. ``<calc>`` claims are verified.
"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from typing import ClassVar

from ..channels.evidence import Claim, Verifier, VerifyEnv, perturb_output
from ..core.task import Answer, InfoBlock, Task
from ..core.tools import Tool, tool
from ..core.transcript import Evidence
from ..core.util import rng_for
from .base import Domain

SOLIB_GSM8K = "https://raw.githubusercontent.com/ArjunPanickssery/math_problems_debate/master/solib/data/math/{split}_expanded.json"

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max, "floor": math.floor, "ceil": math.ceil}


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression safely (numbers, + - * / // % **, a few functions)."""

    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression: {ast.dump(n)[:60]}")

    return ev(ast.parse(_drop_thousands_separators(expr.strip().replace("^", "**")), mode="eval"))


def _drop_thousands_separators(expr: str) -> str:
    """``1,000`` → ``1000`` outside function calls; commas separating call arguments stay."""
    out: list[str] = []
    calls: list[bool] = []  # per open parenthesis: does it belong to a function call?
    for i, ch in enumerate(expr):
        if ch == "(":
            prev = "".join(out).rstrip()
            calls.append(bool(prev) and (prev[-1].isalnum() or prev[-1] == "_"))
        elif ch == ")" and calls:
            calls.pop()
        elif ch == "," and not (calls and calls[-1]) and expr[i - 1:i].isdigit() and re.match(r"\d{3}(?!\d)", expr[i + 1:]):
            continue
        out.append(ch)
    return "".join(out)


def fmt_number(v: float) -> str:
    """Numbers as a person would write them: integers without exponent, others to 10 significant digits."""
    if isinstance(v, bool):
        return str(v)
    if float(v).is_integer() and abs(v) < 1e15:
        return str(int(v))
    return f"{v:.10g}"


def calculator_tool() -> Tool:
    @tool(name="calculator", group="tools")
    def calc(expression: str) -> str:
        """Evaluate an arithmetic expression, e.g. '(16 - 3 - 4) * 2'.

        Args:
            expression: the arithmetic expression to evaluate
        """
        try:
            return f"{expression} = {fmt_number(safe_eval(expression))}"
        except Exception as e:
            return f"error: {e}"

    return calc


class CalcVerifier(Verifier):
    """Verifies ``<calc>expr = value</calc>`` claims (or just shows ``<calc>expr</calc>``)."""

    name = "calculator"
    tag = "calc"
    help = "an arithmetic claim, e.g. <calc>12 * 7 = 84</calc>; checked by a trusted calculator"

    def __init__(self, cost: float = 0.0):
        self.cost = cost

    async def verify(self, claim: Claim, env: VerifyEnv) -> Evidence:
        text = claim.content.strip().replace("==", "=")
        if "=" in text:
            lhs, rhs = (x.strip() for x in text.rsplit("=", 1))
            try:
                val, claimed = safe_eval(lhs), safe_eval(rhs)
                ok = math.isclose(val, claimed, rel_tol=1e-6, abs_tol=1e-9)
                return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"{lhs} = {fmt_number(val)}", verified=ok)
            except Exception as e:
                return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"error: {e}", verified=False)
        try:
            return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"= {fmt_number(safe_eval(text))}", verified=None)
        except Exception as e:
            return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"error: {e}", verified=False)

    def forge(self, claim: Claim, shown: Evidence, env: VerifyEnv) -> Evidence:
        text = claim.content.strip().replace("==", "=")
        if "=" not in text:
            if shown.verified is None:  # a computed value, shown slightly wrong
                return shown.model_copy(update={"result": perturb_output(shown.result, text)})
            if shown.verified:  # an expression that failed, shown as a value
                return shown.model_copy(update={"verified": None, "result": "= 0"})
            return shown.model_copy(update={"result": "error: could not evaluate"})
        lhs, rhs = (x.strip() for x in text.rsplit("=", 1))
        if shown.verified:  # a false claim reported as correct: echo the claimed value
            return shown.model_copy(update={"result": f"{lhs} = {rhs}"})
        try:  # a true claim reported as wrong: report a nearby wrong value
            v = safe_eval(lhs)
            wrong = v + 1 if float(v).is_integer() else v * 1.1
            return shown.model_copy(update={"result": f"{lhs} = {fmt_number(wrong)}"})
        except Exception:
            return shown.model_copy(update={"result": "error: could not evaluate"})


class GSM8K(Domain):
    """GSM8K questions with one correct and one plausible incorrect answer (+ worked solutions)."""

    name: ClassVar[str] = "gsm8k"
    split: str = "test"
    show_solutions: bool = False  # if True, experts also see both worked solutions (privileged)
    expert_clearance: list[str] = ["private", "tools"]
    judge_clearance: list[str] = []

    def load(self) -> list[Task]:
        from ..data import download

        path = download(SOLIB_GSM8K.format(split=self.split), f"solib_gsm8k_{self.split}_expanded.json")
        rows = json.loads(path.read_text())
        tasks = []
        for i, r in enumerate(rows):
            inc = r.get("answer_incorrect")
            if not inc or not isinstance(inc, dict):
                continue
            cor = r["answer_correct"]
            if inc.get("numeric") == cor.get("numeric"):
                continue
            rng = rng_for("gsm8k", i)
            ids = ["A", "B"]
            rng.shuffle(ids)
            fmt = lambda x: (f"{x:g}" if isinstance(x, float) else str(x))  # noqa: E731
            opts = sorted(
                [
                    Answer(id=ids[0], text=fmt(cor["numeric"]), value=1.0),
                    Answer(id=ids[1], text=fmt(inc["numeric"]), value=-1.0),
                ],
                key=lambda a: a.id,
            )
            info = []
            if self.show_solutions:  # keyed by option id only: the key must not reveal which is correct
                sols = {ids[0]: cor.get("proof", ""), ids[1]: inc.get("proof", "")}
                for oid in sorted(sols):
                    info.append(InfoBlock(key=f"solution_{oid}", title=f"Worked solution for ({oid})", content=sols[oid], access="private"))
            tasks.append(Task(
                id=f"gsm8k-{self.split}-{i}", domain=self.name, question=r["question"], options=opts, info=info,
                gt={"distractor": ids[1], "proof_correct": cor.get("proof"), "proof_incorrect": inc.get("proof")},
            ))
        return tasks

    def tools(self, task: Task) -> list[Tool]:
        return [calculator_tool()]

    def verifiers(self, task: Task) -> list[Verifier]:
        return [CalcVerifier()]
