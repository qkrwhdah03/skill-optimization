"""The ``Optimizer`` interface (the user's algorithm), its registry and auto-discovery.

The framework only runs ``initial_prompts()`` -> Evaluator -> ``update(results, history)`` -> ... and never
judges. Everything inside an ``Optimizer`` (how to make skills, which to keep, when to stop) is the algorithm's choice.
"""
from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Callable

from skillopt.evaluator import SkillEvaluation
from skillopt.llm import BaseLLM
from skillopt.registry import Registry
from skillopt.skills import Skill

OPTIMIZER_REGISTRY = Registry("optimizer")


class Optimizer(ABC):
    """The user's algorithm. The framework never judges; everything inside is the algorithm's choice."""

    def __init__(self, llm: BaseLLM, **params) -> None:
        self.llm = llm
        self.params = dict(params)

    @abstractmethod
    def initial_prompts(self) -> list[Skill]:
        """Skills evaluated in round 0 (must be non-empty)."""

    @abstractmethod
    def update(self, results: list[SkillEvaluation], history: list[list[SkillEvaluation]]) -> list[Skill]:
        """Next skills to evaluate; ``[]`` stops the loop.

        ``results``: this round's evaluations (same order as the skills evaluated).
        ``history``: every EARLIER round's evaluations, oldest first (``results`` not included).
        """


def register_optimizer(name: str) -> Callable[[type[Optimizer]], type[Optimizer]]:
    return OPTIMIZER_REGISTRY.register(name)


def build_optimizer(config: dict, llm: BaseLLM) -> Optimizer:
    """Build from the ``optimizer:`` config section ``{name: ..., **params}`` -> ``cls(llm, **params)``."""
    if not isinstance(config, dict) or "name" not in config:
        raise ValueError(f"optimizer config must be a mapping with a 'name' key, got {config!r}")
    cls = OPTIMIZER_REGISTRY.get(config["name"])
    params = {k: v for k, v in config.items() if k != "name"}
    return cls(llm, **params)


def discover_optimizers(package: str = "skillopt.optimizer.algorithms") -> list[str]:
    """Import every non-private module of ``package`` so it self-registers; returns the module names (sorted)."""
    pkg = importlib.import_module(package)
    names: list[str] = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        full = f"{package}.{info.name}"
        importlib.import_module(full)
        names.append(full)
    return sorted(names)
