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
from decimal import Decimal
from fractions import Fraction
from typing import Any, ClassVar

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
_MAX_BITS = 10_000  # integers up to ~3000 digits; larger ones are refused before they are computed


def safe_eval(expr: str) -> float:
    """Evaluate an arithmetic expression safely (numbers, + - * / // % **, a few functions).
    Results are real numbers of bounded size: exponents above 100, integers beyond ~3000 digits
    (also through nested powers or products) and complex results raise ``ValueError``."""

    def big(v: Any) -> bool:
        return isinstance(v, int) and v.bit_length() > _MAX_BITS

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            v = n.value
        elif isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)  # each operand once (nested exponents must not double the work)
            if isinstance(n.op, ast.Pow):
                if abs(b) > 100:
                    raise ValueError("exponent too large")
                if isinstance(a, int) and isinstance(b, int) and b > 0 and a.bit_length() * b > _MAX_BITS:
                    raise ValueError("number too large")
            if isinstance(n.op, ast.Mult) and isinstance(a, int) and isinstance(b, int) \
                    and a.bit_length() + b.bit_length() > _MAX_BITS:
                raise ValueError("number too large")
            v = _OPS[type(n.op)](a, b)
        elif isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            v = _OPS[type(n.op)](ev(n.operand))
        elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            v = _FUNCS[n.func.id](*[ev(a) for a in n.args])
        else:
            raise ValueError(f"unsupported expression: {ast.dump(n)[:60]}")
        if isinstance(v, complex):
            raise ValueError("not a real number")
        if big(v):
            raise ValueError("number too large")
        return v

    return ev(ast.parse(normalize_arithmetic(expr), mode="eval"))


def same_number(a: Any, b: Any) -> bool:
    """Equal up to rounding (relative 1e-6, absolute 1e-9); exact for integers beyond float range."""
    try:
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)
    except OverflowError:
        fa, fb = Fraction(a), Fraction(b)
        return abs(fa - fb) <= max(Fraction(1, 10**9), Fraction(1, 10**6) * max(abs(fa), abs(fb)))


def normalize_arithmetic(expr: str) -> str:
    """Normalise how people write arithmetic into Python: ``^``→``**``, ``×``/``·``/``x``→``*``,
    thousands separators removed (``1,000``→``1000``, also inside calls like ``max(1,000, 5)``),
    currency symbols dropped, and a trailing ``%`` turned into ``/100`` (``50%``→``(50/100)``).
    A bare ``%`` with spaces around it stays modulo."""
    e = expr.strip().replace("^", "**").replace("×", "*").replace("·", "*")
    e = re.sub(r"[$£€₹]", "", e)
    # a percent sign directly after a number (no space) is 'percent'; "a % b" stays modulo
    e = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", e)
    # 'x' as multiplication between numbers: 12 x 7, 12x7 (not inside identifiers like 'max')
    e = re.sub(r"(?<=[\d)\s])[x](?=[\s(]*[\d.])", "*", e)
    # thousands separators: a comma between a digit and exactly three digits (not a 4th)
    e = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", e)
    return e


def fmt_number(v: float) -> str:
    """Numbers as a person would write them: integers without exponent, others to 10 significant digits."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int) and v.bit_length() > 1000:  # beyond float range
        return f"{Decimal(v):.10g}"
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
        # an expression the calculator cannot evaluate is *unverified* (None), never *refuted*:
        # refuting means "computed, and the stated value is wrong", not "could not parse".
        if "=" in text:
            lhs, rhs = (x.strip() for x in text.rsplit("=", 1))
            try:
                val, claimed = safe_eval(lhs), safe_eval(rhs)
            except Exception as e:
                return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"could not evaluate: {e}", verified=None)
            try:
                ok = same_number(val, claimed)
            except (ArithmeticError, TypeError, ValueError) as e:  # e.g. infinity against a huge integer
                return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"could not compare: {e}", verified=None)
            return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"{lhs} = {fmt_number(val)}", verified=ok)
        try:
            return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"= {fmt_number(safe_eval(text))}", verified=None)
        except Exception as e:
            return Evidence(verifier=self.name, kind=self.tag, claim=text, result=f"could not evaluate: {e}", verified=None)

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
            wrong = v + 1 if isinstance(v, int) or float(v).is_integer() else v * 1.1
            return shown.model_copy(update={"result": f"{lhs} = {fmt_number(wrong)}"})
        except Exception:
            return shown.model_copy(update={"result": "error: could not evaluate"})


def _form(x: Any) -> tuple[bool, bool]:
    """(integer?, negative?) of an answer: options that differ in form can be told apart blind."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return (False, False)
    return (v.is_integer(), v < 0)


class GSM8K(Domain):
    """GSM8K questions with one correct and one plausible incorrect answer (+ worked solutions).

    ``match_form`` (default): keep only items whose two answers have the same form (integer or
    not, sign); otherwise "pick the integer, non-negative one" is right 60% of the time without
    reading the question."""

    name: ClassVar[str] = "gsm8k"
    split: str = "test"
    show_solutions: bool = False  # if True, experts also see both worked solutions (privileged)
    match_form: bool = True
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
            if self.match_form and _form(inc.get("numeric")) != _form(cor.get("numeric")):
                continue  # e.g. an integer answer against a fractional or negative distractor: guessable blind
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
