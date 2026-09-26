"""Robust parsing of structured answers out of free-form LLM text."""

from __future__ import annotations

import json
import math
import re
from typing import Any

_THINK_RE = re.compile(r"<(thinking|scratchpad|think)>(.*?)</\1>", re.S | re.I)


def split_thinking(text: str) -> tuple[str, str | None]:
    """Remove <thinking>...</thinking> (or scratchpad/think) blocks; return (public, private)."""
    parts = [m.group(2).strip() for m in _THINK_RE.finditer(text)]
    public = _THINK_RE.sub("", text)
    # unterminated thinking block: treat the rest as private
    m = re.search(r"<(thinking|scratchpad|think)>", public, re.I)
    if m:
        parts.append(public[m.end():].strip())
        public = public[: m.start()]
    return public.strip(), ("\n\n".join(p for p in parts if p) or None)


def _opt_alt(options: list[str]) -> str:
    """Regex alternation over options: single characters (letters) match case-*sensitively*
    (so the article "a" is never read as option A); longer labels (YES/NO) case-insensitively."""
    parts = []
    for o in sorted(options, key=len, reverse=True):
        parts.append(f"(?-i:{re.escape(o)})" if len(o) == 1 else re.escape(o))
    return "|".join(parts)


def _match_option(cand: str, options: list[str]) -> str | None:
    for o in options:
        if cand == o or (len(o) > 1 and cand.lower() == o.lower()):
            return o
    return None


def parse_choice(text: str, options: list[str], option_texts: dict[str, str] | None = None,
                 strict: bool = False) -> str | None:
    """The option an answer commits to. Explicit forms first (``ANSWER: B``, "the answer is
    (B)", "(B)"), then an exact option-text mention. Unless ``strict``, fall back to the last
    standalone option token (weak: prefer asking again, see :class:`LLMAgent`)."""
    if not text:
        return None
    opts = list(options)
    alt = _opt_alt(opts)
    alt_ci = "|".join(re.escape(o) for o in sorted(opts, key=len, reverse=True))
    pats = [
        rf"ANSWER\s*[:=]\s*[\(\[\*]*\s*({alt})\s*[\)\]\*]*(?![A-Za-z0-9])",
        rf"ANSWER\s*[:=]\s*[\(\[\*]*\s*({alt_ci})\s*[\)\]\*\.]*\s*$",  # "ANSWER: b" ending a line
        rf"(?:final answer|answer|choice|verdict)\s*(?:is|:)?\s*[\(\[]?\s*\**\s*({alt})\s*\**\s*[\)\]]?(?![A-Za-z0-9])",
        rf"\(({alt})\)",
    ]
    for p in pats:
        ms = list(re.finditer(p, text, re.I | re.M))
        if ms:
            cand = ms[-1].group(1)
            got = _match_option(cand, opts) or next((o for o in opts if o.lower() == cand.lower()), None)
            if got:
                return got
    stripped = text.strip().strip(".*()[] ")
    got = _match_option(stripped, opts)
    if got:
        return got
    if option_texts:
        low = text.lower()
        hits = [o for o, t in option_texts.items() if t and t.lower() in low]
        if len(hits) == 1:
            return hits[0]
    if strict:
        return None
    ms = re.findall(rf"(?<![A-Za-z0-9])({alt})(?![A-Za-z0-9])", text, re.I)
    if ms:
        return _match_option(ms[-1], opts)
    return None


def _extract_json_objects(text: str) -> list[str]:
    objs, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(text[start : i + 1])
    return objs


def parse_json(text: str) -> dict[str, Any] | None:
    for blob in reversed(_extract_json_objects(text)):
        for attempt in (blob, blob.replace("'", '"'), re.sub(r",\s*}", "}", blob.replace("'", '"'))):
            try:
                val = json.loads(attempt)
                if isinstance(val, dict):
                    return val
            except Exception:
                continue
    return None


_NUM = r"(?:\d+\.?\d*|\.\d+)"


def _raw_prob(v: Any) -> tuple[float, bool] | None:
    """A number and whether it was written as a percentage ("70%")."""
    pct = False
    try:
        if isinstance(v, str):
            v = v.strip()
            if v.endswith("%"):
                pct, v = True, v[:-1]
        f = float(v)
    except Exception:
        return None
    if math.isnan(f) or f < 0:
        return None
    return f, pct


def _to_prob(v: Any) -> float | None:
    """A single probability: "70%" → 0.7, a bare number > 1 → percent."""
    r = _raw_prob(v)
    if r is None:
        return None
    f, pct = r
    return f / 100 if (pct or f > 1.0) else f


def _scale(raw: dict[str, tuple[float, bool]]) -> dict[str, float]:
    """Choose the scale once per distribution: values are percentages if any is written with
    "%", any bare value exceeds 1, or the bare values sum to about 100."""
    bare = [f for f, pct in raw.values() if not pct]
    percent = any(pct for _, pct in raw.values()) or any(f > 1.0 for f in bare) or (len(bare) > 1 and 90 <= sum(bare) <= 110)
    return {k: (f / 100 if (pct or percent) else f) for k, (f, pct) in raw.items()}


def _json_key(k: Any) -> str:
    return str(k).strip().strip("()*[]\"'").lower()


def parse_distribution(text: str, options: list[str]) -> dict[str, float] | None:
    """Parse a probability distribution over options. Accepts JSON (``{"A": 0.7}``, nested under
    "probabilities", keys like "Option A", "(A)" or "A (Paris)"), and lines such as ``A: 70%``,
    ``**A**: 0.7``, ``- A (Paris): 0.8``, ``A) 70%``, ``A - 70%``, ``P(A) = 0.7``,
    ``"A": .65``. Percentages are detected per distribution. Returns None if nothing parses."""
    raw: dict[str, tuple[float, bool]] = {}
    obj = parse_json(text)
    if obj:
        if isinstance(obj.get("probabilities"), dict):
            obj = obj["probabilities"]
        keys = {_json_key(k): v for k, v in obj.items()}
        first = {_json_key(k).split()[0]: v for k, v in obj.items() if _json_key(k).split()}
        for o in options:
            v = keys.get(o.lower(), keys.get(f"option {o}".lower(), first.get(o.lower())))
            r = _raw_prob(v)
            if r is not None:
                raw[o] = r
    if len(raw) < len(options):
        for o in options:
            if o in raw:
                continue
            oa = _opt_alt([o])
            pats = [
                rf"P\(\s*(?:{oa})\s*\)\s*[:=]\s*({_NUM}\s*%?)",
                rf"(?<![A-Za-z0-9])[\"']?(?:{oa})(?:\*\*|\*|__)?[\"']?[\)\]]?\s*(?:\([^()\n]{{0,80}}\))?\s*(?:\*\*|\*)?\s*[:=–—\)-]\s*\**\s*({_NUM}\s*%?)",
            ]
            for pat in pats:
                m = list(re.finditer(pat, text, re.I))
                if m:
                    r = _raw_prob(m[-1].group(1).replace(" ", ""))
                    if r is not None:
                        raw[o] = r
                        break
    if not raw:
        return None
    probs = _scale(raw)
    if len(probs) == len(options) - 1:  # infer the missing one
        missing = [o for o in options if o not in probs][0]
        probs[missing] = max(0.0, 1.0 - sum(probs.values()))
    for o in options:
        probs.setdefault(o, 0.0)
    s = sum(max(v, 0.0) for v in probs.values())
    if s <= 0:
        return None
    return {o: max(probs[o], 0.0) / s for o in options}


_NUM = r"-?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d*\.?\d+)(?:[eE][+-]?\d+)?"  # 1,200 · 12.5 · .5 · 1e-3


def _on_scale(v: float, lo: float, hi: float, pct: bool = False, den: float | None = None) -> float | None:
    """A number as a value on [lo, hi]. Fractions ("3/10", "8 out of 10") and percentages map
    proportionally; "x out of hi" is already on-scale; a bare integer in (1, 100] on a sub-unit
    probability scale is a percentage (70 → 0.7). An out-of-range value that is none of these is
    rejected (None) rather than silently rescaled (1.5 on [0,1] is not 0.015)."""
    if den is not None:
        if den <= 0:
            return None
        v = v if abs(den - hi) < 1e-9 else lo + (v / den) * (hi - lo)  # "x out of hi" is on-scale; else a fraction
        return min(max(v, lo), hi)
    if pct:
        return min(max(lo + (v / 100) * (hi - lo), lo), hi)
    if lo <= v <= hi:
        return v
    if lo >= 0 and hi <= 1 and float(v).is_integer() and 1 < v <= 100:  # "70" on a probability scale means 70%
        return v / 100
    return None


def _number_at(text: str, pos: int = 0) -> tuple[float, bool, float | None] | None:
    """(value, percent?, denominator) for the number starting at ``pos`` (after markup). Handles a
    percentage, a fraction ``x/y`` and ``x out of y``."""
    m = re.match(rf"\s*[*_`$£€₹]*\s*({_NUM})\s*[*_`]*\s*(%|(?:/|out\s+of\s+)\s*[*_`]*\s*({_NUM}))?", text[pos:])
    if not m:
        return None
    den = float(m.group(3).replace(",", "")) if m.group(3) else None
    return float(m.group(1).replace(",", "")), m.group(2) == "%", den


def parse_scalar(text: str, lo: float, hi: float, name: str = "value") -> float | None:
    """A number on the scale [lo, hi] from free text or JSON: a labelled value (``NAME: 7``,
    ``**NAME**: 3/10``, ``NAME = 70%``, ``NAME: 8 out of 10``), else a JSON field, else the last
    number already in range. Out-of-range values that are not a recognisable percentage or fraction
    are ignored rather than rescaled."""
    for label in (re.escape(name), r"(?:score|probability|rating|suspicion|value|forecast|confidence)"):
        ms = list(re.finditer(rf"[*_`]*\b{label}\b[*_`]*\s*(?:[:=]|\bis\b)\s*", text, re.I))
        for m in reversed(ms):
            got = _number_at(text, m.end())
            if got is not None:
                s = _on_scale(got[0], lo, hi, got[1], got[2])
                if s is not None:
                    return s
    obj = parse_json(text)
    if isinstance(obj, dict):
        for k in (name, "value", "score", "probability", "suspicion", "rating"):
            v = obj.get(k)
            if isinstance(v, bool) or v is None:
                continue
            got = (float(v), False, None) if isinstance(v, (int, float)) else _number_at(str(v))
            if got is not None:
                s = _on_scale(got[0], lo, hi, got[1], got[2])
                if s is not None:
                    return s
    for m in reversed(list(re.finditer(_NUM, text))):
        v = float(m.group(0).replace(",", ""))
        if lo <= v <= hi:
            return v
    return None
