"""The plain runner: initial_prompts() -> Evaluator -> update(results, history) -> Evaluator -> ...

Every round evaluates the ``train`` split of every benchmark (no limit).

No scoring, ranking, selection or final evaluation: it stops when ``update`` returns ``[]`` or after
``max_iterations`` updates. Every decision belongs to the user's ``Optimizer``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from skillopt.evaluator import Evaluator, SkillEvaluation
from skillopt.skills import Skill

from .base import Optimizer

TRAIN_SPLIT: str = "train"


@dataclass
class LoopResult:
    history: list[list[SkillEvaluation]]  # every evaluated round, index 0 = initial prompts
    stop_reason: str  # "max_iterations" | "empty_update"


def _check_prompts(prompts, where: str) -> list[Skill]:
    if not isinstance(prompts, (list, tuple)):
        raise TypeError(f"{where} must return a list of Skill, got {type(prompts).__name__}")
    seen: set[str] = set()
    for i, skill in enumerate(prompts):
        if not isinstance(skill, Skill):
            raise TypeError(f"{where} returned a non-Skill at index {i}: {type(skill).__name__}")
        if skill.id in seen:
            raise ValueError(f"{where} returned duplicate skill id '{skill.id}'")
        seen.add(skill.id)
    return list(prompts)


def optimize(
    optimizer: Optimizer,
    evaluator: Evaluator,
    max_iterations: int,
    *,
    on_round: Callable[[int, list[SkillEvaluation]], None] | None = None,
) -> LoopResult:
    """Run the loop on the ``train`` split; ``update`` is called at most ``max_iterations`` times."""
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError(f"max_iterations must be an int >= 0, got {max_iterations!r}")

    prompts = _check_prompts(optimizer.initial_prompts(), "initial_prompts()")
    if not prompts:
        raise ValueError("initial_prompts() returned no skills")

    history: list[list[SkillEvaluation]] = []
    index = 0
    while True:
        results = evaluator.evaluate(prompts, TRAIN_SPLIT)
        if on_round is not None:
            on_round(index, results)
        if index >= max_iterations:
            history.append(results)
            return LoopResult(history=history, stop_reason="max_iterations")
        prompts = _check_prompts(optimizer.update(results, list(history)), "update()")
        history.append(results)
        if not prompts:
            return LoopResult(history=history, stop_reason="empty_update")
        index += 1
