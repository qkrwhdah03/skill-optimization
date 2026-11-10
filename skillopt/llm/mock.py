"""Deterministic mock backend (no deps, no GPU) for end-to-end runs and tests."""
from __future__ import annotations

import hashlib
import re
from typing import ClassVar

from .base import BaseLLM, RawGeneration
from .registry import register_llm
from .thinking import split_thinking

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


@register_llm("mock")
class MockLLM(BaseLLM):
    """Config keys: ``responses`` (cycled if set), ``think_words`` (default 20), ``seed`` (default 0)."""

    uses_gpu: ClassVar[bool] = False

    def __init__(self, config: dict, role: str = "default") -> None:
        super().__init__(config, role=role)
        responses = config.get("responses")
        self.responses: list[str] | None = list(responses) if responses else None
        self.think_words = int(config.get("think_words", 20))
        self.seed = int(config.get("seed", 0))
        self.calls: list[str] = []
        self._counter = 0

    def _default_output(self, prompt: str, gen: dict) -> str:
        digest = hashlib.sha256(f"{self.seed}|{prompt}".encode("utf-8")).digest()
        lo = self.think_words // 2
        hi = self.think_words
        n = lo + (int.from_bytes(digest[:4], "big") % (hi - lo + 1))
        k = int.from_bytes(digest[4:8], "big") % 21
        answer = f"The answer is {k}."
        if gen.get("enable_thinking", True) is False:
            return answer
        return "<think>" + " ".join(["step"] * n) + "</think>" + answer

    def _generate(self, prompt: str, gen: dict) -> RawGeneration:
        self.calls.append(prompt)
        if self.responses:
            out = self.responses[self._counter % len(self.responses)]
            self._counter += 1
        else:
            out = self._default_output(prompt, gen)
        reasoning, visible = split_thinking(out)
        return RawGeneration(text=visible, reasoning_text=reasoning, prompt_text=prompt, raw=out)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(_TOKEN_RE.findall(text))
