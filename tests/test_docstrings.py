"""Docstrings keep their LaTeX: no escape sequence turned into a control character (``\\frac`` read as a form
feed, ``\\beta`` as a backspace) and no invalid escape (``\\Delta`` in a non-raw string)."""

import ast
import re
from pathlib import Path

ROOTS = [Path(__file__).resolve().parent.parent / d for d in ("src", "scripts")]
# an odd number of backslashes before a letter or LaTeX spacing, other than the escapes docstrings may mean
ESCAPE = re.compile(r"(?<!\\)(?:\\\\)*\\(?![nt\"'uUxN\\])[A-Za-z,;!]")
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _docstrings():
    for root in ROOTS:
        for path in sorted(root.rglob("*.py")):
            src = path.read_text()
            for node in ast.walk(ast.parse(src)):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
                    d = node.body[0]
                    if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str):
                        yield path, d.value, ast.get_source_segment(src, d.value) or ""


def test_docstrings_have_no_control_characters_or_invalid_escapes():
    bad = []
    for path, node, seg in _docstrings():
        if CONTROL.search(node.value):
            bad.append(f"{path}:{node.lineno}: control character {CONTROL.search(node.value).group(0)!r}")
        raw = "r" in re.match(r"[a-zA-Z]*", seg).group(0).lower()
        if not raw and ESCAPE.search(seg):
            bad.append(f"{path}:{node.lineno}: escape {ESCAPE.search(seg).group(0)!r} in a non-raw docstring (use r\"\"\")")
    assert not bad, "\n".join(bad)
