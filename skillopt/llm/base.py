"""LLM interface. All token accounting lives here so every backend is measured identically."""
from __future__ import annotations

import contextlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Iterator


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0  # INCLUDES reasoning/thinking tokens
    reasoning_tokens: int = 0
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.total_tokens is None:
            self.total_tokens = self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=(self.total_tokens or 0) + (other.total_tokens or 0),
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class RawGeneration:
    """What a backend's ``_generate`` returns. Token fields are raw reported counts or None."""

    text: str
    reasoning_text: str = ""
    prompt_text: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None  # reported count INCLUDING reasoning
    reasoning_tokens: int | None = None
    raw: Any = None


@dataclass
class LLMResponse:
    text: str
    reasoning_text: str
    usage: TokenUsage
    latency: float
    raw: Any = None


@dataclass
class UsageSnapshot:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "latency": self.latency,
        }


class UsageTracker:
    """Accumulates usage over many calls."""

    def __init__(self) -> None:
        self._snap = UsageSnapshot()

    def record(self, usage: TokenUsage, latency: float) -> None:
        s = self._snap
        s.calls += 1
        s.input_tokens += usage.input_tokens
        s.output_tokens += usage.output_tokens
        s.reasoning_tokens += usage.reasoning_tokens
        s.total_tokens += usage.total_tokens or 0
        s.latency += latency

    def reset(self) -> None:
        self._snap = UsageSnapshot()

    def snapshot(self) -> UsageSnapshot:
        s = self._snap
        return UsageSnapshot(
            calls=s.calls,
            input_tokens=s.input_tokens,
            output_tokens=s.output_tokens,
            reasoning_tokens=s.reasoning_tokens,
            total_tokens=s.total_tokens,
            latency=s.latency,
        )


class BaseLLM(ABC):
    """Base class for every LLM backend.

    Backends implement ``_generate`` (and optionally ``_generate_batch``) and report raw
    token counts if they have them; ``generate``/``generate_batch`` turn those into
    ``TokenUsage`` and record them into the per-instance tracker and any active ``track()``.
    """

    uses_gpu: ClassVar[bool] = True

    def __init__(self, config: dict, role: str = "default") -> None:
        self.config = dict(config)
        self.role = role
        self.model = config["model"]
        self.gen_defaults = dict(config.get("generation", {}) or {})
        self.usage = UsageTracker()
        self._active_trackers: list[UsageTracker] = []
        self._hf_tokenizer: Any = None

    # ------------------------------------------------------------------ backend hooks
    @abstractmethod
    def _generate(self, prompt: str, gen: dict) -> RawGeneration:
        ...

    def _generate_batch(self, prompts: list[str], gen: dict) -> list[RawGeneration]:
        return [self._generate(p, gen) for p in prompts]

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._hf_tokenizer is None:
            try:
                from transformers import AutoTokenizer  # lazy heavy import
            except ImportError as e:  # pragma: no cover - depends on env
                raise RuntimeError(
                    f"count_tokens for backend '{self.config.get('backend')}' needs the 'transformers' "
                    "package (pip install -e .[hf]) or a backend that reports token counts"
                ) from e
            name = self.config.get("tokenizer", self.model)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(name)
        return len(self._hf_tokenizer.encode(text, add_special_tokens=False))

    # ------------------------------------------------------------------ accounting
    def _usage_from_raw(self, raw: RawGeneration, prompt: str) -> TokenUsage:
        if raw.input_tokens is not None:
            input_tokens = int(raw.input_tokens)
        else:
            input_tokens = self.count_tokens(raw.prompt_text if raw.prompt_text is not None else prompt)
        if raw.reasoning_tokens is not None:
            reasoning_tokens = int(raw.reasoning_tokens)
        else:
            reasoning_tokens = self.count_tokens(raw.reasoning_text) if raw.reasoning_text else 0
        if raw.output_tokens is not None:
            output_tokens = int(raw.output_tokens)
        else:
            output_tokens = (self.count_tokens(raw.text) if raw.text else 0) + reasoning_tokens
        if output_tokens < reasoning_tokens:
            reasoning_tokens = output_tokens
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    def _record(self, usage: TokenUsage, latency: float) -> None:
        self.usage.record(usage, latency)
        for tracker in list(self._active_trackers):
            tracker.record(usage, latency)

    def generate(self, prompt: str, **gen_kwargs) -> LLMResponse:
        gen = {**self.gen_defaults, **gen_kwargs}
        start = time.perf_counter()
        raw = self._generate(prompt, gen)
        latency = time.perf_counter() - start
        usage = self._usage_from_raw(raw, prompt)
        self._record(usage, latency)
        return LLMResponse(
            text=raw.text, reasoning_text=raw.reasoning_text, usage=usage, latency=latency, raw=raw.raw
        )

    def generate_batch(self, prompts: list[str], **gen_kwargs) -> list[LLMResponse]:
        prompts = list(prompts)
        if not prompts:
            return []
        gen = {**self.gen_defaults, **gen_kwargs}
        start = time.perf_counter()
        raws = self._generate_batch(prompts, gen)
        wall = time.perf_counter() - start
        if len(raws) != len(prompts):
            raise RuntimeError(
                f"backend returned {len(raws)} generations for {len(prompts)} prompts"
            )
        latency = wall / len(prompts)
        out: list[LLMResponse] = []
        for prompt, raw in zip(prompts, raws):
            usage = self._usage_from_raw(raw, prompt)
            self._record(usage, latency)
            out.append(
                LLMResponse(
                    text=raw.text,
                    reasoning_text=raw.reasoning_text,
                    usage=usage,
                    latency=latency,
                    raw=raw.raw,
                )
            )
        return out

    @contextlib.contextmanager
    def track(self) -> Iterator[UsageTracker]:
        tracker = UsageTracker()
        self._active_trackers.append(tracker)
        try:
            yield tracker
        finally:
            # remove this exact tracker (identity, not equality)
            for i in range(len(self._active_trackers) - 1, -1, -1):
                if self._active_trackers[i] is tracker:
                    del self._active_trackers[i]
                    break

    # ------------------------------------------------------------------ resources
    @classmethod
    def estimate_vram_gb(cls, config: dict, vram_budget_gb: float) -> float:
        if config.get("est_vram_gb") is not None:
            return float(config["est_vram_gb"])
        if not cls.uses_gpu:
            return 0.0
        raise ValueError(
            f"est_vram_gb required for backend '{config.get('backend')}' (model '{config.get('model')}')"
        )
