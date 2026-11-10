"""Helpers for Qwen3-style ``<think>...</think>`` reasoning blocks."""
from __future__ import annotations

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def split_thinking(text: str) -> tuple[str, str]:
    """Split model output into ``(reasoning_text, visible_text)``."""
    if text is None:
        return "", ""
    if THINK_CLOSE in text:
        idx = text.rfind(THINK_CLOSE)
        reasoning = text[:idx].lstrip()
        if reasoning.startswith(THINK_OPEN):
            reasoning = reasoning[len(THINK_OPEN):]
        visible = text[idx + len(THINK_CLOSE):]
        return reasoning.strip(), visible.strip()
    if THINK_OPEN in text:
        idx = text.find(THINK_OPEN)
        return text[idx + len(THINK_OPEN):].strip(), ""
    return "", text.strip()
