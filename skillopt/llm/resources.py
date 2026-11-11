"""Shared engine cache and VRAM budget check (all in-process models must fit the budget)."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

_ENGINE_CACHE: dict[tuple, Any] = {}

logger = logging.getLogger(__name__)


class VRAMBudgetError(ValueError):
    """Raised when the configured LLM roles do not fit in the VRAM budget."""


def engine_key(config: dict) -> tuple:
    return (
        config.get("backend"),
        config.get("model"),
        json.dumps(config.get("engine", {}) or {}, sort_keys=True),
    )


def _describe_key(key: Any) -> str:
    """``{backend}:{model} dtype={dtype}`` for an ``engine_key`` tuple, else ``repr(key)``; never raises."""
    try:
        if isinstance(key, tuple) and len(key) == 3 and isinstance(key[2], str):
            engine = json.loads(key[2])
            if isinstance(engine, dict):
                dtype = engine.get("dtype") or "-"
                return f"{key[0]}:{key[1]} dtype={dtype}"
        return repr(key)
    except Exception:
        try:
            return repr(key)
        except Exception:
            return "<engine>"


def get_or_create_engine(key: tuple, factory: Callable[[], Any]) -> Any:
    if key not in _ENGINE_CACHE:
        desc = _describe_key(key)
        logger.info(f"loading engine {desc} ...")
        started = time.perf_counter()
        engine = factory()
        secs = time.perf_counter() - started
        _ENGINE_CACHE[key] = engine
        logger.info(f"engine {desc} loaded in {secs:.1f}s")
    else:
        logger.debug(f"reusing engine {_describe_key(key)}")
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
