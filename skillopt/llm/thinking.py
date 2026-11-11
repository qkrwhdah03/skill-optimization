"""Helpers for Qwen3-style ``<think>...</think>`` reasoning blocks."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

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


# tokenizer class names already warned about (warn at most once per class)
_WARNED_TOKENIZERS: set[str] = set()


def render_chat_prompt(tokenizer: Any, prompt: str, enable_thinking: bool) -> str:
    """Render ``prompt`` as one user turn with ``enable_thinking`` passed directly to the chat template.

    Tokenizers whose ``apply_chat_template`` rejects the kwarg (``TypeError``) are rendered without it and a
    WARNING is logged once per tokenizer class (the toggle is then ignored).
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
        )
    except TypeError:
        name = type(tokenizer).__name__
        if name not in _WARNED_TOKENIZERS:
            _WARNED_TOKENIZERS.add(name)
            logger.warning(
                f"tokenizer {name} rejected enable_thinking in apply_chat_template; "
                f"the thinking toggle (enable_thinking={enable_thinking}) is ignored"
            )
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
