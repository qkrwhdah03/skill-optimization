"""LLM backend registry and role construction."""
from __future__ import annotations

import logging
from typing import Callable

from skillopt.registry import Registry

from .base import BaseLLM

LLM_REGISTRY = Registry("llm backend")

logger = logging.getLogger(__name__)


def register_llm(name: str) -> Callable[[type[BaseLLM]], type[BaseLLM]]:
    return LLM_REGISTRY.register(name)


def get_llm_class(name: str) -> type[BaseLLM]:
    return LLM_REGISTRY.get(name)


def build_llm(config: dict, role: str = "default") -> BaseLLM:
    if "backend" not in config:
        raise ValueError(f"llm config for role '{role}' has no 'backend'")
    cls = get_llm_class(config["backend"])
    return cls(config, role=role)


def build_role_llms(llm_configs: dict[str, dict], vram_budget_gb: float | None = None) -> dict[str, BaseLLM]:
    """Build one LLM instance per role. Runs the VRAM budget check before building anything.

    Local backends share heavy engines through ``resources.get_or_create_engine`` so two roles
    with the same backend/model/engine settings occupy VRAM only once.
    """
    from .resources import check_vram_budget

    if vram_budget_gb is not None:
        total = check_vram_budget(llm_configs, vram_budget_gb)
        logger.debug(f"vram estimate {total:.2f} GB of {vram_budget_gb} GB")
    llms: dict[str, BaseLLM] = {}
    for name, cfg in llm_configs.items():
        llms[name] = build_llm(cfg, role=name)
        dtype = (cfg.get("engine") or {}).get("dtype") or "-"
        logger.info(f"llm {name}: backend={cfg.get('backend')} model={cfg.get('model')} dtype={dtype}")
    return llms
