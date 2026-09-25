"""Robust parsing of model outputs into structured actions (probabilities, choices, scores, JSON).

All parsers return ``None`` on failure rather than raising; callers fall back to a neutral action and
flag ``parse_ok=False`` so failures are visible in the logs and in manipulation checks.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

from so_arena.core.types import TokenLogprob

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def extract_tag(text: str, tag: str) -> str | None:
    m = re.findall(rf"<{tag}\b[^>]*>(.*?)</{tag}>", text, flags=re.S | re.I)
    return m[-1].strip() if m else None


def split_reasoning(text: str, tags: Sequence[str] = ("thinking", "reasoning", "scratchpad")) -> tuple[str, str | None]:
    """Separate private reasoning (inside e.g. <thinking> tags) from the public part of a response."""
    reasoning_parts = []
    public = text
    for tag in tags:
        pattern = re.compile(rf"<{tag}\b[^>]*>(.*?)</{tag}>", re.S | re.I)
        reasoning_parts += [m.strip() for m in pattern.findall(public)]
        public = pattern.sub("", public)
        # unterminated opening tag: everything after it is reasoning
        m = re.search(rf"<{tag}\b[^>]*>", public, re.I)
        if m:
            reasoning_parts.append(public[m.end():].strip())
            public = public[: m.start()]
    reasoning = "\n\n".join(p for p in reasoning_parts if p) or None
    return public.strip(), reasoning


def truncate_words(text: str, limit: int | None) -> str:
    """Keep the first ``limit`` words, preserving the original whitespace (so code survives)."""
    if not limit:
        return text
    matches = list(re.finditer(r"\S+", text))
    if len(matches) <= limit:
        return text
    return text[: matches[limit - 1].end()] + " [...]"


def parse_json_object(text: str) -> dict | None:
    """Return the last JSON object in ``text`` (tolerates code fences and trailing prose)."""
    candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, flags=re.S)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            try:
                obj = json.loads(cand.replace("'", '"'))
            except json.JSONDecodeError:
                continue
        if isinstance(obj, dict):
            return obj
    return None


def _norm_label(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _match_label(key: str, options: Sequence[str]) -> str | None:
    nk = _norm_label(key)
    for o in options:
        if _norm_label(o) == nk:
            return o
    return None


def _prefix_label(token: str, options: Sequence[str]) -> str | None:
    """Map a first token that is a prefix of exactly one (multi-token) option label, e.g. "susp" -> "suspicious"."""
    t = _norm_label(token)
    if len(t) < 2:
        return None
    hits = [o for o in options if _norm_label(o).startswith(t) and _norm_label(o) != t]
    return hits[0] if len(hits) == 1 else None


def normalize_probs(probs: dict[str, float], options: Sequence[str], floor: float = 0.0) -> dict[str, float] | None:
    vals = {o: max(float(probs.get(o, 0.0)), 0.0) for o in options}
    if any(v > 1.0 for v in vals.values()) and all(v <= 100.0 for v in vals.values()):
        # looks like percentages
        vals = {o: v / 100.0 for o, v in vals.items()}
    total = sum(vals.values())
    if total <= 0 or not math.isfinite(total):
        return None
    out = {o: v / total for o, v in vals.items()}
    if floor > 0:
        out = {o: (1 - floor * len(options)) * v + floor for o, v in out.items()}
    return out


def parse_probabilities(text: str, options: Sequence[str]) -> dict[str, float] | None:
    """Parse a distribution over ``options`` from free text.

    Accepts JSON (``{"A": 0.7, "B": 0.3}``), ``A: 70%`` style lines, and a single probability for a
    binary question (``Probability: 0.8`` -> first option 0.8).
    """
    obj = parse_json_object(text)
    if obj:
        found = {}
        for k, v in obj.items():
            lab = _match_label(str(k), options)
            if lab is not None:
                try:
                    found[lab] = float(str(v).rstrip("%"))
                except ValueError:
                    pass
        if found and (len(found) == len(options) or len(options) == 2):
            if len(found) == 1 and len(options) == 2:
                (lab, p), = found.items()
                p = p / 100 if p > 1 else p
                other = [o for o in options if o != lab][0]
                found = {lab: p, other: 1 - p}
            res = normalize_probs(found, options)
            if res:
                return res
    found = {}
    for o in options:
        pat = rf"(?:^|[\s(\[\"']){re.escape(o)}[)\]\"']?\s*[:=\-]\s*({_NUM})\s*(%)?"
        ms = re.findall(pat, text, flags=re.I | re.M)
        if ms:
            num, pct = ms[-1]
            val = float(num)
            found[o] = val / 100 if pct else val
    if len(found) == len(options):
        return normalize_probs(found, options)
    if len(options) == 2:
        m = re.findall(rf"probability[^0-9\n]{{0,40}}?({_NUM})\s*(%)?", text, flags=re.I)
        if m:
            num, pct = m[-1]
            p = float(num) / (100 if pct or float(num) > 1 else 1)
            if 0 <= p <= 1:
                return {options[0]: p, options[1]: 1 - p}
    return None


def parse_choice(text: str, options: Sequence[str]) -> str | None:
    """Parse one of ``options`` from free text, preferring explicit ``Answer: X`` / ``<answer>X</answer>``."""
    tagged = extract_tag(text, "answer") or extract_tag(text, "choice")
    if tagged:
        lab = _match_label(tagged, options) or _match_label(tagged.strip("()[] ."), options)
        if lab:
            return lab
    m = re.findall(r"(?:final\s+)?(?:answer|choice|decision|verdict)\s*(?:is)?\s*[:=]?\s*\(?\s*([A-Za-z0-9_\-]+)\s*\)?", text, flags=re.I)
    for cand in reversed(m):
        lab = _match_label(cand, options)
        if lab:
            return lab
    stripped = text.strip().strip("()[]. \n\"'")
    lab = _match_label(stripped, options)
    if lab:
        return lab
    # last standalone mention of an option label
    best, best_pos = None, -1
    for o in options:
        for mm in re.finditer(rf"(?<![A-Za-z0-9]){re.escape(o)}(?![A-Za-z0-9])", text, flags=re.I if len(o) > 1 else 0):
            if mm.start() > best_pos:
                best, best_pos = o, mm.start()
    return best


def parse_score(text: str, lo: float, hi: float) -> float | None:
    tagged = extract_tag(text, "score")
    cands = []
    if tagged:
        cands += re.findall(_NUM, tagged)
    cands += [x for x in re.findall(rf"score\s*[:=]?\s*({_NUM})", text, flags=re.I)]
    cands += re.findall(_NUM, text)[-3:][::-1]
    for c in cands:
        try:
            v = float(c)
        except ValueError:
            continue
        if lo <= v <= hi:
            return v
    return None


def probs_from_logprobs(logprobs: list[TokenLogprob] | None, options: Sequence[str]) -> dict[str, float] | None:
    """Distribution over option labels from the top-logprobs of the first informative token.

    Token variants (" A", "A", "(A", "a") are pooled per option; the result is renormalized over
    options (probability mass on other tokens is discarded, as in "tip-of-tongue" judges).
    """
    if not logprobs:
        return None
    for tok in logprobs[:5]:
        cands = tok.top or []
        if not cands:
            cands = [tok]
        mass: dict[str, float] = {}
        for c in cands:
            tok = c.token.strip().strip("()[]*\"'.")
            lab = _match_label(tok, options) or _prefix_label(tok, options)
            if lab is not None:
                mass[lab] = mass.get(lab, 0.0) + math.exp(c.logprob)
        if mass:
            floor = min(mass.values()) * 1e-3 if mass else 0.0
            filled = {o: mass.get(o, floor) for o in options}
            total = sum(filled.values())
            return {o: v / total for o, v in filled.items()}
    return None
