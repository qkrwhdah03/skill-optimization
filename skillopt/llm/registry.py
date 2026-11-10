"""LLM backend registry and role construction."""
from __future__ import annotations

from typing import Callable

from skillopt.registry import Registry

from .base import BaseLLM

LLM_REGISTRY = Registry("llm backend")


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
        check_vram_budget(llm_configs, vram_budget_gb)
    return {name: build_llm(cfg, role=name) for name, cfg in llm_configs.items()}
