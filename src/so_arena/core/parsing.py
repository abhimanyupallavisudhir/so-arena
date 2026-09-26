"""Robust parsing of model outputs into structured actions (probabilities, choices, scores, JSON).

All parsers return ``None`` on failure rather than raising; callers fall back to a neutral action and
flag ``parse_ok=False`` so failures are visible in the logs and in manipulation checks.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence

from so_arena.core.types import TokenLogprob

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_POS = r"(?:\d+(?:\.\d*)?|\.\d+)"
# a score term: a number, or "N/M" / "N out of M" (N on a scale up to M)
_TERM = rf"{_NUM}(?:\s*(?:/|\bout\s+of\b)\s*{_POS})?"
_TERM_RE = re.compile(rf"(?P<n>{_NUM})(?:\s*(?:/|\bout\s+of\b)\s*(?P<m>{_POS}))?", re.I)


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


def complete_probs(stated: Mapping[str, float], options: Sequence[str]) -> dict[str, float] | None:
    """A distribution over ``options`` from probabilities stated for some of them.

    The mass the stated options leave goes to the unstated ones in equal parts (``{"A": 0.8}`` over A, B
    means B: 0.2); stated values that are percentages are rescaled, and ones summing to more than one
    are normalized.
    """
    vals = {o: max(float(stated[o]), 0.0) for o in options if o in stated}
    if not vals:
        return None
    if any(v > 1.0 for v in vals.values()) and all(v <= 100.0 for v in vals.values()):
        vals = {o: v / 100.0 for o, v in vals.items()}
    missing = [o for o in options if o not in vals]
    rest = 1.0 - sum(vals.values())
    for o in missing:
        vals[o] = rest / len(missing) if rest > 0 else 0.0
    return normalize_probs(vals, options)


def probs_from_mapping(obj: Mapping[object, object], options: Sequence[str]) -> dict[str, float] | None:
    """A distribution from a mapping such as ``{"A": 0.7, "b": "30%"}`` (keys matched to option labels
    loosely, unstated options completed as in :func:`complete_probs`). Shared by the text parser and by
    policies returning dicts, so a distribution means the same whichever way it arrives."""
    stated: dict[str, float] = {}
    for k, v in obj.items():
        lab = _match_label(str(k), options)
        if lab is None:
            continue
        s = str(v).strip()
        try:
            x = float(s.rstrip("%").strip())
        except ValueError:
            continue
        stated[lab] = x / 100.0 if s.endswith("%") else x
    return complete_probs(stated, options) if stated else None


def mentioned_options(text: str, options: Sequence[str], option_texts: Mapping[str, str] | None = None) -> list[str]:
    """Options that ``text`` names, by label or answer text, in order of first mention.

    Longer names win ("no violation" names ``no_violation``, not ``violation``); one-letter labels match
    case-sensitively (the article "a" is not option "A"); names of four or more letters also match their
    inflections ("rejection" names ``reject``).
    """
    names: list[tuple[str, str]] = []
    for o in options:
        for n in dict.fromkeys([o, re.sub(r"[_-]+", " ", o), (option_texts or {}).get(o) or ""]):
            if n.strip():
                names.append((n.strip(), o))
    if not names:
        return []
    names.sort(key=lambda t: -len(t[0]))
    parts = []
    for i, (n, _) in enumerate(names):
        body = re.escape(n)
        pat = body if len(n) == 1 else f"(?i:{body})" + (r"\w*" if len(n) >= 4 else "")
        parts.append(rf"(?P<n{i}>{pat})(?!\w)")
    found: list[str] = []
    for m in re.finditer(r"(?<!\w)(?:" + "|".join(parts) + ")", text):
        o = names[int(str(m.lastgroup)[1:])][1]
        if o not in found:
            found.append(o)
    return found


def parse_probabilities(text: str, options: Sequence[str], option_texts: Mapping[str, str] | None = None
                        ) -> dict[str, float] | None:
    """Parse a distribution over ``options`` from free text.

    Accepts JSON (``{"A": 0.7, "B": 0.3}``; unstated options share the remaining mass), ``A: 70%`` style
    lines, and a single probability for a binary question, which goes to the option the statement names
    by label or answer text (``Probability of rejection: 90%`` -> reject 0.9), else to the first option
    (``Probability: 0.8``); a statement naming both options is ambiguous and not parsed.
    """
    obj = parse_json_object(text)
    if obj:
        res = probs_from_mapping(obj, options)
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
        ms = list(re.finditer(rf"probability(?P<about>[^0-9\n]{{0,40}}?)(?P<num>{_NUM})\s*(?P<pct>%)?", text, flags=re.I))
        if ms:
            m = ms[-1]
            num = float(m.group("num"))
            p = num / (100 if m.group("pct") or num > 1 else 1)
            named = mentioned_options(m.group("about"), options, option_texts)
            if 0 <= p <= 1 and len(named) < 2:
                first = named[0] if named else options[0]
                return {o: p if o == first else 1 - p for o in options}
    return None


def parse_choice(text: str, options: Sequence[str]) -> str | None:
    """Parse one of ``options`` from free text, preferring explicit ``Answer: X`` / ``<answer>X</answer>``
    (a lowercase one-letter word after "answer is" that a word follows - "the answer is a tricky one" - is
    an article, not a label)."""
    tagged = extract_tag(text, "answer") or extract_tag(text, "choice")
    if tagged:
        lab = _match_label(tagged, options) or _match_label(tagged.strip("()[] ."), options)
        if lab:
            return lab
    found = list(re.finditer(r"(?:final\s+)?(?:answer|choice|decision|verdict)\s*(?:is)?\s*[:=]?\s*(\()?\s*([A-Za-z0-9_\-]+)\s*(\))?",
                             text, flags=re.I))
    for m in reversed(found):
        cand = m.group(2)
        # "the answer is a tricky one", "my decision: a clear B": a lowercase letter that a word follows is the
        # article (or "i"), not option A - unless it is bracketed like a label
        if (len(cand) == 1 and cand.islower() and cand not in options and not (m.group(1) or m.group(3))
                and re.match(r"\s+[A-Za-z]", text[m.end(2):])):
            continue
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


def _score_terms(text: str, lo: float, hi: float) -> list[float]:
    """The numbers in ``text`` read as scores on ``[lo, hi]``, in order.

    "N/M" and "N out of M" are one term, N on a scale up to M (rescaled linearly when M is not ``hi``), so
    the denominator is never read as a score; a description of the requested scale itself ("0-10",
    "from 0 to 10") is skipped.
    """
    def scale(m: re.Match) -> str:
        a, b = (m.group(1), m.group(2)) if m.group(1) is not None else (m.group(3), m.group(4))
        return " " if (float(a), float(b)) == (lo, hi) else m.group(0)

    text = re.sub(rf"(?<![\w.])(?:(?:from\s+)?({_NUM})\s*(?:-|–|\bto\b)\s*({_NUM})|between\s+({_NUM})\s+and\s+({_NUM}))"
                  r"(?![\w.])", scale, text, flags=re.I)
    out = []
    for m in _TERM_RE.finditer(text):
        v = float(m.group("n"))
        d = float(m.group("m")) if m.group("m") is not None else None
        out.append(lo + (v - lo) * (hi - lo) / (d - lo) if d is not None and d != hi and d > lo else v)
    return out


def parse_score(text: str, lo: float, hi: float) -> float | None:
    """A score in ``[lo, hi]``: from a ``<score>`` tag, else the last ``Score: N``, else one of the last numbers.

    "Suspicion: 3/10" and "3 out of 10" are 3 (not 10); see :func:`_score_terms`.
    """
    tagged = extract_tag(text, "score")
    cands: list[float] = []
    if tagged:
        cands += _score_terms(tagged, lo, hi)
    for m in reversed(list(re.finditer(rf"score\s*[:=]?\s*({_TERM})", text, flags=re.I))):
        cands += _score_terms(m.group(1), lo, hi)[:1]
    cands += _score_terms(text, lo, hi)[-3:][::-1]
    for v in cands:
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
