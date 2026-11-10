"""Baseline reasoning styles expressed as skills (direct, CoT, Chain of Draft): ready-made skills an Optimizer may use."""
from __future__ import annotations

from .skills import Skill

BASELINE_NAMES: tuple[str, ...] = ("direct", "cot", "cod")

_INSTRUCTIONS: dict[str, str] = {
    "direct": "Answer the question directly. Do not explain.",
    "cot": "Let's think step by step.",
    "cod": "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most.",
}


def get_baseline(name: str) -> Skill:
    if name not in _INSTRUCTIONS:
        raise KeyError(f"unknown baseline '{name}'; available: {', '.join(BASELINE_NAMES)}")
    return Skill(id=name, instruction=_INSTRUCTIONS[name], few_shot=[], metadata={"baseline": True})
