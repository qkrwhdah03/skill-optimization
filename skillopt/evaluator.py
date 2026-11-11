"""Evaluator: runs the fixed target LLM with skills on a list of benchmarks; measures accuracy + tokens.

The Evaluator only measures; it never aggregates, compares or judges. Its ONE fixed result format:

    Evaluator.evaluate(skills, split) -> list[SkillEvaluation]      (same order as ``skills``)
    SkillEvaluation.skill            the evaluated Skill (``.skill_id`` == ``skill.id``)
    SkillEvaluation.split            e.g. "train" (the split evaluated every loop round)
    SkillEvaluation.results          {benchmark key (config order): EvalResult}; benchmarks without the split
                                     are skipped
    EvalResult                       n, accuracy, avg_input_tokens, avg_output_tokens (thinking INCLUDED),
                                     avg_reasoning_tokens, avg_total_tokens, avg_latency, num_truncated,
                                     truncation_rate, failures(k),
                                     items: list[ItemResult] (question, prediction, gold, correct, usage,
                                     latency, response_text, reasoning_text, truncated = hit max_tokens)
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass

from .benchmarks import Benchmark, build_benchmark
from .llm import BaseLLM, TokenUsage
from .skills import Skill

logger = logging.getLogger(__name__)


@dataclass
class ItemResult:
    example_id: str
    prediction: str | None
    gold: str
    correct: bool
    usage: TokenUsage
    latency: float
    response_text: str
    reasoning_text: str
    question: str = ""
    truncated: bool = False  # generation stopped because it hit max_tokens

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "prediction": self.prediction,
            "gold": self.gold,
            "correct": self.correct,
            "usage": self.usage.to_dict(),
            "latency": self.latency,
            "response_text": self.response_text,
            "reasoning_text": self.reasoning_text,
            "question": self.question,
            "truncated": self.truncated,
        }


@dataclass
class EvalResult:
    skill_id: str
    benchmark: str
    split: str
    items: list[ItemResult]

    def _mean(self, values: list[float]) -> float:
        return float(sum(values)) / len(values) if values else 0.0

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def accuracy(self) -> float:
        return self._mean([1.0 if it.correct else 0.0 for it in self.items])

    @property
    def avg_output_tokens(self) -> float:
        return self._mean([it.usage.output_tokens for it in self.items])

    @property
    def avg_reasoning_tokens(self) -> float:
        return self._mean([it.usage.reasoning_tokens for it in self.items])

    @property
    def avg_input_tokens(self) -> float:
        return self._mean([it.usage.input_tokens for it in self.items])

    @property
    def avg_total_tokens(self) -> float:
        return self._mean([it.usage.total_tokens or 0 for it in self.items])

    @property
    def avg_latency(self) -> float:
        return self._mean([it.latency for it in self.items])

    @property
    def num_truncated(self) -> int:
        return sum(1 for it in self.items if it.truncated)

    @property
    def truncation_rate(self) -> float:
        return self.num_truncated / self.n if self.n else 0.0

    def failures(self, k: int | None = None) -> list[ItemResult]:
        bad = [it for it in self.items if not it.correct]
        return bad if k is None else bad[:k]

    def summary(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "benchmark": self.benchmark,
            "split": self.split,
            "n": self.n,
            "accuracy": self.accuracy,
            "avg_input_tokens": self.avg_input_tokens,
            "avg_output_tokens": self.avg_output_tokens,
            "avg_reasoning_tokens": self.avg_reasoning_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "avg_latency": self.avg_latency,
            "num_truncated": self.num_truncated,
            "truncation_rate": self.truncation_rate,
        }




_KEY_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")

_MAPPING_EXAMPLE = "benchmarks: {gsm8k: {source: hf, train: 200, val: 200}}"

# old option key -> explanation of the replacement (raised before the benchmark is built)
_OLD_KEYS: dict[str, str] = {
    "dev_size": "renamed to 'train'",
    "val_size": "renamed to 'val'",
    "heldout_tasks": "renamed to 'heldout'",
    "splits": "removed: the loop always evaluates 'train'; use --eval-only --split <split> for other splits",
    "id": "removed: the mapping key is the benchmark key; use 'name' to choose the benchmark class",
}


@dataclass
class BenchmarkSpec:
    """One benchmark entry: a unique key and the benchmark object."""

    key: str
    benchmark: Benchmark


def build_benchmark_specs(configs: dict, seed: int = 0) -> list[BenchmarkSpec]:
    """Build specs from the ``benchmarks:`` mapping ``{key: {name?, ...constructor kwargs} | None}``."""
    if not isinstance(configs, dict):
        raise ValueError(
            f"benchmarks must be a mapping keyed by benchmark name, e.g. {_MAPPING_EXAMPLE}; got {configs!r}"
        )
    if not configs:
        raise ValueError(f"benchmarks must not be empty, e.g. {_MAPPING_EXAMPLE}")
    specs: list[BenchmarkSpec] = []
    for key, raw in configs.items():
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise ValueError(f"invalid benchmark key {key!r}: must match {_KEY_RE.pattern}")
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"benchmark '{key}': options must be a mapping (or empty), got {raw!r}")
        for old, hint in _OLD_KEYS.items():
            if old in raw:
                raise ValueError(f"benchmark '{key}': option '{old}' is no longer supported ({hint})")
        options = dict(raw)
        cls_name = options.pop("name", key)
        benchmark = build_benchmark({"name": cls_name, **options}, seed)
        specs.append(BenchmarkSpec(key=key, benchmark=benchmark))
    return specs


@dataclass
class SkillEvaluation:
    """Raw results of one skill on every benchmark providing one split (no aggregation)."""

    skill: Skill
    split: str
    results: dict[str, EvalResult]

    @property
    def skill_id(self) -> str:
        return self.skill.id

    @property
    def benchmarks(self) -> list[str]:
        return list(self.results)

    def summary(self) -> dict:
        return {
            "skill_id": self.skill_id,
            "split": self.split,
            "benchmarks": {key: r.summary() for key, r in self.results.items()},
        }


class Evaluator:
    """Measures a skill on a list of benchmarks with the fixed target LLM. Never judges."""

    def __init__(self, llm: BaseLLM, benchmarks: list[BenchmarkSpec], batch_size: int = 16) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        specs = list(benchmarks or [])
        if not specs:
            raise ValueError("Evaluator needs at least one benchmark spec")
        self._specs: dict[str, BenchmarkSpec] = {}
        for spec in specs:
            if spec.key in self._specs:
                raise ValueError(f"duplicate benchmark key '{spec.key}'")
            self._specs[spec.key] = spec
        self.llm = llm
        self.benchmarks: list[BenchmarkSpec] = specs
        self.batch_size = int(batch_size)

    @property
    def benchmark_keys(self) -> list[str]:
        return list(self._specs)

    def _spec(self, key: str) -> BenchmarkSpec:
        try:
            return self._specs[key]
        except KeyError:
            raise KeyError(f"unknown benchmark key '{key}'; available: {', '.join(self._specs)}") from None

    def evaluate_split(self, skill: Skill, key: str, split: str, limit: int | None = None) -> EvalResult:
        benchmark = self._spec(key).benchmark
        examples = benchmark.load(split)
        if limit is not None:
            examples = examples[:limit]
        started = time.perf_counter()
        total = len(examples)
        items: list[ItemResult] = []
        for start in range(0, len(examples), self.batch_size):
            chunk = examples[start : start + self.batch_size]
            prompts = [benchmark.build_prompt(skill, ex) for ex in chunk]
            responses = self.llm.generate_batch(prompts)
            for ex, resp in zip(chunk, responses):
                pred = benchmark.extract_answer_for(resp.text, ex)
                items.append(
                    ItemResult(
                        example_id=ex.id,
                        prediction=pred,
                        gold=ex.answer,
                        correct=bool(benchmark.score(pred, ex.answer)),
                        usage=resp.usage,
                        latency=resp.latency,
                        response_text=resp.text,
                        reasoning_text=resp.reasoning_text,
                        question=ex.question,
                        truncated=bool(resp.truncated),
                    )
                )
            logger.info(f"eval {skill.id:<24} {key:<10} {split:<8} {len(items):>5}/{total}")
        result = EvalResult(skill_id=skill.id, benchmark=key, split=split, items=items)
        secs = time.perf_counter() - started
        logger.info(
            f"eval {skill.id:<24} {key:<10} {split:<8} done n={result.n} acc={result.accuracy:.3f} "
            f"out_tok={result.avg_output_tokens:.1f} think_tok={result.avg_reasoning_tokens:.1f} "
            f"trunc={result.num_truncated} trunc_rate={result.truncation_rate:.3f} time={secs:.1f}s"
        )
        return result

    def evaluate_skill(self, skill: Skill, split: str = "train", limit: int | None = None) -> SkillEvaluation:
        """Evaluate one skill on ``split`` of every benchmark providing it (config order).

        Benchmarks whose ``splits`` lack ``split`` are skipped; ``limit`` caps items per benchmark (None = all).
        """
        keys = [key for key, spec in self._specs.items() if split in spec.benchmark.splits]
        if not keys:
            available = "; ".join(
                f"{key}: {', '.join(spec.benchmark.splits)}" for key, spec in self._specs.items()
            )
            raise ValueError(f"no benchmark provides split '{split}' (available splits: {available})")
        results: dict[str, EvalResult] = {}
        for key in keys:
            results[key] = self.evaluate_split(skill, key, split, limit)
        return SkillEvaluation(skill=skill, split=split, results=results)

    def evaluate(self, skills: list[Skill], split: str = "train", limit: int | None = None) -> list[SkillEvaluation]:
        """Evaluate every skill on every benchmark; one ``SkillEvaluation`` per skill, same order as ``skills``."""
        if not isinstance(skills, (list, tuple)):
            raise TypeError(f"evaluate() expects a list of Skill, got {type(skills).__name__}; use evaluate_skill()")
        return [self.evaluate_skill(skill, split, limit) for skill in skills]


__all__ = [
    "ItemResult",
    "EvalResult",
    "BenchmarkSpec",
    "build_benchmark_specs",
    "SkillEvaluation",
    "Evaluator",
]
