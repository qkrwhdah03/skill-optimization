"""Shared math-answer handling: ``\\boxed{}`` extraction, normalization and equivalence.

Rule-based comparison uses the stdlib only; an optional symbolic check uses
``math-verify`` (``pip install -e .[math]``), imported lazily on first use.
"""
from __future__ import annotations

import functools
import importlib.metadata
import logging
import re
import threading
from fractions import Fraction
from typing import Any

logger = logging.getLogger(__name__)

MATH_ANSWER_FORMAT: str = "Put your final answer within \\boxed{}."
SYMBOLIC_TIMEOUT_SECONDS: float = 5.0

_BOX_CMD_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_ANSWER_IS_RE = re.compile(r"answer is", re.IGNORECASE)
_UNWRAP_RE = re.compile(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}")
_ASSIGN_RE = re.compile(r"^[A-Za-z]\s*=")
_FRAC_DD_RE = re.compile(r"\\frac(\d)(\d)")
_FRAC_D_BRACE_RE = re.compile(r"\\frac(\d)\{")
_SQRT_RE = re.compile(r"\\sqrt([0-9A-Za-z])")
_WS_RE = re.compile(r"\s+")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_LEADING_DOT_RE = re.compile(r"^(-?)\.(\d)")
_INT_RE = re.compile(r"^-?\d+$")
_DEC_RE = re.compile(r"^-?\d*\.\d+$")
_SLASH_RE = re.compile(r"^(-?)(\d+)/(\d+)$")
_TEX_FRAC_RE = re.compile(r"^(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}$")

_REMOVE_TOKENS = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\displaystyle")
_REMOVE_UNITS = ("^{\\circ}", "^\\circ", "\u00b0", "\\%", "%", "\\$", "$")


def _balanced(text: str, open_idx: int) -> int | None:
    """Index of the ``}`` closing the ``{`` at ``open_idx`` (None if unbalanced)."""
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def last_boxed(text: str | None) -> str | None:
    """Content of the last non-blank ``\\boxed{...}`` / ``\\fbox{...}`` (balanced braces), or None."""
    if text is None or not text.strip():
        return None
    contents: list[str] = []
    for m in _BOX_CMD_RE.finditer(text):
        open_idx = m.end() - 1
        close = _balanced(text, open_idx)
        if close is not None:
            contents.append(text[open_idx + 1 : close])
    for content in reversed(contents):
        if content.strip():
            return content.strip()
    return None


def extract_math_answer(text: str | None) -> str | None:
    """Last ``\\boxed{}`` content, else the text after the last "answer is"; no last-number fallback."""
    if text is None or not text.strip():
        return None
    boxed = last_boxed(text)
    if boxed is not None:
        return boxed
    matches = list(_ANSWER_IS_RE.finditer(text))
    if not matches:
        return None
    rest = text[matches[-1].end():].lstrip()
    if rest.startswith(":"):
        rest = rest[1:]
    segment = ""
    for line in rest.splitlines():
        if line.strip():
            segment = line.strip()
            break
    if segment.endswith("."):
        segment = segment[:-1]
    segment = segment.strip().strip("$").strip()
    return segment or None


def normalize_math_answer(s: str | None) -> str | None:
    """Canonical string form of a math answer (case preserved)."""
    if s is None:
        return None
    t = str(s).strip()
    if t.startswith("\\boxed{"):
        open_idx = len("\\boxed{") - 1
        if _balanced(t, open_idx) == len(t) - 1:
            t = t[open_idx + 1 : -1].strip()
    t = t.strip("$").strip()
    for tok in _REMOVE_TOKENS:
        t = t.replace(tok, "")
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    prev = None
    while t != prev:
        prev = t
        t = _UNWRAP_RE.sub(lambda m: m.group(1), t)
    for tok in _REMOVE_UNITS:
        t = t.replace(tok, "")
    if t.count("=") == 1 and _ASSIGN_RE.match(t.strip()):
        t = _ASSIGN_RE.sub("", t.strip(), count=1)
    t = _FRAC_DD_RE.sub(r"\\frac{\1}{\2}", t)
    t = _FRAC_D_BRACE_RE.sub(r"\\frac{\1}{", t)
    t = _SQRT_RE.sub(r"\\sqrt{\1}", t)
    t = _WS_RE.sub("", t)
    if t.endswith("."):
        t = t[:-1]
    if _THOUSANDS_RE.match(t):
        t = t.replace(",", "")
    t = _LEADING_DOT_RE.sub(r"\g<1>0.\2", t)
    return t


def _as_fraction(s: str) -> Fraction | None:
    try:
        if _INT_RE.match(s):
            return Fraction(int(s))
        if _DEC_RE.match(s):
            return Fraction(s)
        m = _SLASH_RE.match(s)
        if m:
            sign, num, den = m.groups()
            if int(den) == 0:
                return None
            value = Fraction(int(num), int(den))
            return -value if sign else value
        m = _TEX_FRAC_RE.match(s)
        if m:
            sign, num, den = m.groups()
            if int(den) == 0:
                return None
            value = Fraction(int(num), int(den))
            return -value if sign else value
    except (ValueError, ZeroDivisionError):
        return None
    return None


def _outer_brackets(s: str) -> str | None:
    if len(s) >= 2 and s[0] in "([" and s[-1] in ")]":
        return s[0] + s[-1]
    return None


def _split_top_level(s: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(s[start:i])
            start = i + 1
    parts.append(s[start:])
    return parts


def _rule_equal(pred: str | None, gold: str | None) -> bool:
    """Rule-based equivalence: normalized string, exact rational, or element-wise tuple."""
    a = normalize_math_answer(pred)
    b = normalize_math_answer(gold)
    if not a or not b:
        return False
    if a == b:
        return True
    fa, fb = _as_fraction(a), _as_fraction(b)
    if fa is not None and fb is not None:
        return fa == fb
    ba, bb = _outer_brackets(a), _outer_brackets(b)
    if ba == bb and ba is not None:
        a, b = a[1:-1], b[1:-1]
    parts_a, parts_b = _split_top_level(a), _split_top_level(b)
    if len(parts_a) > 1 and len(parts_b) > 1:
        if len(parts_a) != len(parts_b):
            return False
        return all(_rule_equal(x, y) for x, y in zip(parts_a, parts_b))
    return False


MATH_VERIFY_DIST = "math-verify"


def math_verify_version() -> str | None:
    """Installed version of the optional ``math-verify`` distribution, or None if not installed.

    Reads package metadata only (``importlib.metadata.version``); never imports math_verify or sympy.
    """
    try:
        return importlib.metadata.version(MATH_VERIFY_DIST)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def load_math_verify() -> Any | None:
    """The ``math_verify`` module if installed and importable, else None (cached)."""
    try:
        import math_verify
    except Exception:
        return None
    return math_verify


def _as_boxed(s: str) -> str:
    return "\\boxed{" + s.strip().strip("$").strip() + "}"


def symbolic_math_equal(pred: str | None, gold: str | None, timeout: float | None = None) -> bool:
    """Symbolic equivalence via math-verify within ``timeout`` seconds; False if unavailable, failed or timed out."""
    if timeout is None:
        timeout = SYMBOLIC_TIMEOUT_SECONDS
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    if pred is None or gold is None or not pred.strip() or not gold.strip():
        return False
    mv = load_math_verify()
    if mv is None:
        return False
    boxed_pred, boxed_gold = _as_boxed(pred), _as_boxed(gold)
    result: dict[str, bool] = {}

    def work() -> None:
        try:
            g = mv.parse(boxed_gold, parsing_timeout=None)
            p = mv.parse(boxed_pred, parsing_timeout=None)
            if not g or not p:
                result["value"] = False
                return
            result["value"] = bool(mv.verify(g, p, timeout_seconds=None))
        except Exception as exc:
            logger.debug("symbolic check failed: %s: %s", type(exc).__name__, exc)
            result["value"] = False

    # math-verify's built-in timeouts break off the main thread and on Windows,
    # so the limit is enforced here with a daemon thread (left running on timeout).
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        logger.debug("symbolic check timed out after %ss", timeout)
        return False
    return result.get("value", False)


def math_answers_equal(pred: str | None, gold: str | None) -> bool:
    """Rule-based equivalence first; otherwise an optional symbolic check (math-verify, time-limited)."""
    a = normalize_math_answer(pred)
    b = normalize_math_answer(gold)
    if not a or not b:
        return False
    if _rule_equal(pred, gold):
        return True
    if _as_fraction(a) is not None and _as_fraction(b) is not None:
        return False
    return symbolic_math_equal(pred, gold)
