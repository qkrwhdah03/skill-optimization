"""Shared multiple-choice prompt formatting, answer extraction, and scoring helpers."""
from __future__ import annotations

import re
from typing import Sequence

MCQ_LETTERS: str = "ABCDEFGHIJ"

_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_PREFIX_RE = re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_PAREN_LETTER_RE = re.compile(r"\(([A-Za-z])\)")
_LEAD_LETTER_RE = re.compile(r"^\s*(?:(?:option|choice)\s+)?\(?([A-Za-z])\)?(?![A-Za-z0-9])", re.IGNORECASE)
_EXISTING_OPT_RE = re.compile(r"^\(([A-Za-z])\)\s*(.*)$", re.DOTALL)


def format_mcq_options(choices: Sequence[str]) -> str:
    """Format choices as ``(A) ...`` lines without double-prefixing already-labeled options."""
    lines: list[str] = []
    for idx, choice in enumerate(choices):
        letter = MCQ_LETTERS[idx] if idx < len(MCQ_LETTERS) else chr(ord("A") + idx)
        text = str(choice).strip()
        m = _EXISTING_OPT_RE.match(text)
        if m and m.group(1).upper() == letter:
            rest = m.group(2).strip()
            lines.append(f"({letter}) {rest}" if rest else f"({letter})")
        else:
            lines.append(f"({letter}) {text}")
    return "\n".join(lines)


def mcq_answer_format(num_choices: int = 10) -> str:
    """Standard multiple-choice answer-format instruction clamped to 2..10 options."""
    n = max(2, min(len(MCQ_LETTERS), int(num_choices)))
    last = MCQ_LETTERS[n - 1]
    return f'End your response with "The answer is (X)." where X is the letter of the correct option (A-{last}).'


def _strip_markup(text: str) -> str:
    cleaned = _TAG_RE.sub(" ", text)
    return cleaned.replace("*", "").replace("`", "")


def _pick_from_boxed(inner: str, valid: set[str]) -> str | None:
    s = inner.strip()
    m = _PAREN_LETTER_RE.search(s)
    if m and m.group(1).lower() in valid:
        return m.group(1).lower()
    m = _LEAD_LETTER_RE.match(s)
    if m and m.group(1).lower() in valid:
        tail = s[m.end() :].strip(" .):,-")
        if not tail:
            return m.group(1).lower()
    return None


def _pick_from_segment(segment: str, valid: set[str], *, prefer_first: bool) -> str | None:
    boxed = _BOXED_RE.findall(segment)
    for inner in reversed(boxed):
        hit = _pick_from_boxed(inner, valid)
        if hit is not None:
            return hit

    paren_hits = [m.group(1).lower() for m in _PAREN_LETTER_RE.finditer(segment) if m.group(1).lower() in valid]
    if paren_hits:
        return paren_hits[0] if prefer_first else paren_hits[-1]

    lead = _LEAD_LETTER_RE.match(segment)
    if lead and lead.group(1).lower() in valid:
        return lead.group(1).lower()
    return None


def extract_mcq_answer(text: str | None, valid_letters: str = "abcdefghij") -> str | None:
    """Extract a normalized lowercase multiple-choice letter (e.g. ``'a'``..``'j'``) from model output."""
    if text is None or not text.strip():
        return None
    valid = {ch.lower() for ch in valid_letters if ch.isalpha()}
    if not valid:
        return None

    cleaned = _strip_markup(text)

    prefix_matches = list(_PREFIX_RE.finditer(cleaned))
    if prefix_matches:
        rest = cleaned[prefix_matches[-1].end() :]
        for raw_line in rest.splitlines():
            line = raw_line.strip()
            if line:
                return _pick_from_segment(line, valid, prefer_first=True)
        return None

    for inner in reversed(_BOXED_RE.findall(cleaned)):
        hit = _pick_from_boxed(inner, valid)
        if hit is not None:
            return hit

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return None
    return _pick_from_segment(lines[-1], valid, prefer_first=False)


def score_mcq(pred: str | None, gold: str) -> bool:
    """Case-insensitive exact match on normalized option letter."""
    if pred is None:
        return False
    return pred.strip().lower() == str(gold).strip().lower()
