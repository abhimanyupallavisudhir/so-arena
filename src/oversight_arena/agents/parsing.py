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


def parse_choice(text: str, options: list[str], option_texts: dict[str, str] | None = None) -> str | None:
    if not text:
        return None
    opts = [o for o in options]
    esc = "|".join(re.escape(o) for o in sorted(opts, key=len, reverse=True))
    pats = [
        rf"ANSWER\s*[:=]\s*\(?\s*({esc})\s*\)?(?![A-Za-z0-9])",
        rf"(?:final answer|answer|choice|verdict)\s*(?:is|:)?\s*\(?\s*\**\s*({esc})\s*\**\s*\)?(?![A-Za-z0-9])",
        rf"\(({esc})\)",
    ]
    for p in pats:
        ms = list(re.finditer(p, text, re.I))
        if ms:
            cand = ms[-1].group(1)
            for o in opts:
                if o.lower() == cand.lower():
                    return o
    stripped = text.strip().strip(".*()[] ")
    for o in opts:
        if stripped.lower() == o.lower():
            return o
    if option_texts:
        low = text.lower()
        hits = [o for o, t in option_texts.items() if t and t.lower() in low]
        if len(hits) == 1:
            return hits[0]
    ms = re.findall(rf"(?<![A-Za-z0-9])({esc})(?![A-Za-z0-9])", text)
    if ms:
        return ms[-1]
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


def _to_prob(v: Any) -> float | None:
    try:
        if isinstance(v, str):
            v = v.strip()
            if v.endswith("%"):
                return float(v[:-1]) / 100
            v = float(v)
        v = float(v)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def parse_distribution(text: str, options: list[str]) -> dict[str, float] | None:
    """Parse a probability distribution over options. Accepts JSON, 'A: 0.7' lines, percents."""
    probs: dict[str, float] = {}
    obj = parse_json(text)
    if obj:
        lower = {str(k).strip().strip("()").lower(): v for k, v in obj.items()}
        if "probabilities" in lower and isinstance(lower["probabilities"], dict):
            lower = {str(k).strip().lower(): v for k, v in lower["probabilities"].items()}
        for o in options:
            v = lower.get(o.lower())
            if v is None:
                v = lower.get(f"option {o}".lower())
            p = _to_prob(v)
            if p is not None:
                probs[o] = p
    if len(probs) < len(options):
        for o in options:
            if o in probs:
                continue
            m = list(
                re.finditer(
                    rf"(?<![A-Za-z0-9]){re.escape(o)}\)?\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", text
                )
            )
            if m:
                val = float(m[-1].group(1))
                probs[o] = val / 100 if m[-1].group(2) == "%" else val
    if not probs:
        return None
    if any(v > 1.0 for v in probs.values()):  # percentages without % sign
        probs = {k: v / 100 for k, v in probs.items()}
    if len(probs) == len(options) - 1:  # infer the missing one
        missing = [o for o in options if o not in probs][0]
        probs[missing] = max(0.0, 1.0 - sum(probs.values()))
    for o in options:
        probs.setdefault(o, 0.0)
    s = sum(max(v, 0.0) for v in probs.values())
    if s <= 0:
        return None
    return {o: max(probs[o], 0.0) / s for o in options}


def parse_scalar(text: str, lo: float, hi: float, name: str = "value") -> float | None:
    pats = [
        rf"{re.escape(name)}\s*[:=]\s*(-?[0-9]*\.?[0-9]+)\s*(%?)",
        r"(?:score|probability|rating|suspicion|value|forecast)\s*[:=]\s*(-?[0-9]*\.?[0-9]+)\s*(%?)",
    ]
    for p in pats:
        ms = list(re.finditer(p, text, re.I))
        if ms:
            v = float(ms[-1].group(1))
            if ms[-1].group(2) == "%":
                v /= 100
            return min(max(v, lo), hi)
    obj = parse_json(text)
    if obj:
        for k in (name, "value", "score", "probability"):
            if k in obj:
                v = _to_prob(obj[k])
                if v is not None:
                    return min(max(v, lo), hi)
    nums = re.findall(r"-?[0-9]*\.?[0-9]+", text)
    for n in reversed(nums):
        try:
            v = float(n)
        except ValueError:
            continue
        if lo <= v <= hi:
            return v
    return None
