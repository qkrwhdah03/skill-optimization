"""LLM interface, token accounting, backends and VRAM resources."""
from .base import BaseLLM, LLMResponse, RawGeneration, TokenUsage, UsageSnapshot, UsageTracker
from .registry import build_llm, build_role_llms, get_llm_class, register_llm
from .resources import (
    VRAMBudgetError,
    check_vram_budget,
    clear_engine_cache,
    engine_key,
    get_or_create_engine,
)
from .thinking import split_thinking

# import backends so they self-register
from .mock import MockLLM
from .vllm_backend import VLLMBackend
from .hf_backend import HFTransformersLLM
from .openai_backend import OpenAICompatLLM

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "RawGeneration",
    "TokenUsage",
    "UsageSnapshot",
    "UsageTracker",
    "build_llm",
    "build_role_llms",
    "get_llm_class",
    "register_llm",
    "VRAMBudgetError",
    "check_vram_budget",
    "clear_engine_cache",
    "engine_key",
    "get_or_create_engine",
    "split_thinking",
    "MockLLM",
    "VLLMBackend",
    "HFTransformersLLM",
    "OpenAICompatLLM",
]
