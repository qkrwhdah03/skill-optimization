"""Benchmark interface: load splits, build prompts from a skill, extract and score answers."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar

from skillopt.registry import Registry
from skillopt.skills import Skill, render_prompt

BENCHMARK_REGISTRY = Registry("benchmark")

# project root = parent of the skillopt package (same rule as skillopt.config.resolve_path;
# duplicated here to keep benchmarks independent of config)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@dataclass
class Example:
    id: str
    question: str
    answer: str  # gold, already normalized
    meta: dict = field(default_factory=dict)


class Benchmark(ABC):
    name: ClassVar[str]
    answer_format: ClassVar[str]
    splits: ClassVar[tuple[str, ...]] = ("train", "val", "test")  # split names load() accepts

    @abstractmethod
    def load(self, split: str) -> list[Example]:
        ...

    def build_prompt(self, skill: Skill, example: Example) -> str:
        return render_prompt(skill, example.question, self.answer_format)

    @abstractmethod
    def extract_answer(self, text: str) -> str | None:
        ...

    @abstractmethod
    def score(self, pred: str | None, gold: str) -> bool:
        ...


def register_benchmark(name: str) -> Callable[[type[Benchmark]], type[Benchmark]]:
    return BENCHMARK_REGISTRY.register(name)


def get_benchmark_class(name: str) -> type[Benchmark]:
    return BENCHMARK_REGISTRY.get(name)


def build_benchmark(config: dict, seed: int = 0) -> Benchmark:
    cls = get_benchmark_class(config["name"])
    kwargs = {k: v for k, v in config.items() if k != "name"}
    kwargs.setdefault("seed", seed)
    return cls(**kwargs)
