"""Shared engine cache and VRAM budget check (all in-process models must fit the budget)."""
from __future__ import annotations

import json
from typing import Any, Callable

_ENGINE_CACHE: dict[tuple, Any] = {}


class VRAMBudgetError(ValueError):
    """Raised when the configured LLM roles do not fit in the VRAM budget."""


def engine_key(config: dict) -> tuple:
    return (
        config.get("backend"),
        config.get("model"),
        json.dumps(config.get("engine", {}) or {}, sort_keys=True),
    )


def get_or_create_engine(key: tuple, factory: Callable[[], Any]) -> Any:
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = factory()
    return _ENGINE_CACHE[key]


def clear_engine_cache() -> None:
    _ENGINE_CACHE.clear()


def check_vram_budget(llm_configs: dict[str, dict], vram_budget_gb: float) -> float:
    from .registry import get_llm_class

    engines: dict[tuple, tuple[list[str], float]] = {}
    for role, cfg in llm_configs.items():
        key = engine_key(cfg)
        if key in engines:
            engines[key][0].append(role)
            continue
        est = float(get_llm_class(cfg["backend"]).estimate_vram_gb(cfg, vram_budget_gb))
        engines[key] = ([role], est)
    total = sum(est for _, est in engines.values())
    if total > vram_budget_gb + 1e-9:
        lines = [
            f"  - {key[0]}:{key[1]} (roles: {', '.join(roles)}): {est:.2f} GB"
            for key, (roles, est) in engines.items()
        ]
        raise VRAMBudgetError(
            f"estimated VRAM {total:.2f} GB exceeds budget {vram_budget_gb:.2f} GB:\n" + "\n".join(lines)
        )
    return total
