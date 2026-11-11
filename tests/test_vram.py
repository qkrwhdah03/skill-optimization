"""D. VRAM budget & engine sharing (24 GB constraint)."""
import pytest
import yaml

from helpers import CONFIGS, uniq
from skillopt.config import load_config
from skillopt.llm import (
    BaseLLM,
    HFTransformersLLM,
    MockLLM,
    OpenAICompatLLM,
    RawGeneration,
    VLLMBackend,
    VRAMBudgetError,
    build_role_llms,
    check_vram_budget,
    clear_engine_cache,
    engine_key,
    get_llm_class,
    get_or_create_engine,
    register_llm,
)


def _vllm(model="m", util=0.85, **extra):
    cfg = {"backend": "vllm", "model": model, "engine": {"gpu_memory_utilization": util}}
    cfg.update(extra)
    return cfg


def _openai():
    return {"backend": "openai", "model": "remote-model",
            "engine": {"base_url": "http://localhost:9/v1", "api_key_env": "SKILLOPT_UNUSED_KEY"}}


# --- AC14 ------------------------------------------------------------------

def test_vllm_estimate_from_gpu_memory_utilization():
    assert VLLMBackend.estimate_vram_gb(_vllm(), 24) == pytest.approx(20.4)


def test_vllm_estimate_default_utilization():
    assert VLLMBackend.estimate_vram_gb({"backend": "vllm", "model": "m"}, 24) == pytest.approx(0.9 * 24)


def test_explicit_est_vram_wins():
    assert VLLMBackend.estimate_vram_gb(_vllm(est_vram_gb=10), 24) == pytest.approx(10)
    assert HFTransformersLLM.estimate_vram_gb({"backend": "hf", "model": "m", "est_vram_gb": 18}, 24) == pytest.approx(18)


def test_non_gpu_backends_estimate_zero():
    assert MockLLM.estimate_vram_gb({"backend": "mock", "model": "m"}, 24) == 0.0
    assert OpenAICompatLLM.estimate_vram_gb(_openai(), 24) == 0.0


def test_hf_without_estimate_raises():
    with pytest.raises(ValueError):
        HFTransformersLLM.estimate_vram_gb({"backend": "hf", "model": "m"}, 24)


# --- AC15 ------------------------------------------------------------------

def test_shared_engine_counted_once():
    target = _vllm(generation={"max_tokens": 512, "temperature": 0.0})
    optimizer = _vllm(generation={"max_tokens": 2048, "temperature": 1.0})
    total = check_vram_budget({"target": target, "optimizer": optimizer}, 24)
    assert total == pytest.approx(20.4)


def test_exactly_at_budget_passes():
    total = check_vram_budget({"target": _vllm(util=1.0)}, 24)
    assert total == pytest.approx(24.0)
    total = check_vram_budget({"target": {"backend": "hf", "model": "m", "est_vram_gb": 24.0}}, 24)
    assert total == pytest.approx(24.0)


# --- AC16 ------------------------------------------------------------------

def test_two_distinct_vllm_engines_over_budget():
    with pytest.raises(VRAMBudgetError):
        check_vram_budget({"target": _vllm(model="a"), "optimizer": _vllm(model="b")}, 24)
    assert issubclass(VRAMBudgetError, ValueError)


def test_hf_plus_vllm_over_budget():
    with pytest.raises(VRAMBudgetError):
        check_vram_budget({"target": _vllm(),
                           "optimizer": {"backend": "hf", "model": "h", "est_vram_gb": 18}}, 24)


def test_vllm_plus_remote_openai_passes():
    total = check_vram_budget({"target": _vllm(), "optimizer": _openai()}, 24)
    assert total == pytest.approx(20.4)


def test_hf_missing_estimate_in_budget_check_raises_value_error():
    with pytest.raises(ValueError):
        check_vram_budget({"target": {"backend": "hf", "model": "h"}}, 24)


# --- AC17 ------------------------------------------------------------------

_CONSTRUCTED = []


class _FlagGPU(BaseLLM):
    uses_gpu = True

    def __init__(self, config, role="default"):
        _CONSTRUCTED.append(role)
        super().__init__(config, role=role)

    def _generate(self, prompt, gen):
        return RawGeneration(text="x")

    def count_tokens(self, text):
        return len(text.split())


_FLAG_NAME = uniq("flaggpu")
register_llm(_FLAG_NAME)(_FlagGPU)


def test_build_role_llms_checks_budget_before_constructing():
    _CONSTRUCTED.clear()
    cfgs = {
        "target": {"backend": _FLAG_NAME, "model": "a", "est_vram_gb": 14},
        "optimizer": {"backend": _FLAG_NAME, "model": "b", "est_vram_gb": 14},
    }
    with pytest.raises(VRAMBudgetError):
        build_role_llms(cfgs, vram_budget_gb=24)
    assert _CONSTRUCTED == []


def test_build_role_llms_within_budget_constructs():
    _CONSTRUCTED.clear()
    cfgs = {
        "target": {"backend": _FLAG_NAME, "model": "a", "est_vram_gb": 10},
        "optimizer": {"backend": _FLAG_NAME, "model": "b", "est_vram_gb": 10},
    }
    llms = build_role_llms(cfgs, vram_budget_gb=24)
    assert set(llms) == {"target", "optimizer"}
    assert sorted(_CONSTRUCTED) == ["optimizer", "target"]


# --- AC18 ------------------------------------------------------------------

def test_engine_key_ignores_generation():
    a = _vllm(generation={"max_tokens": 1})
    b = _vllm(generation={"max_tokens": 999, "temperature": 1.0})
    assert engine_key(a) == engine_key(b)


def test_engine_key_differs_on_model_or_engine():
    assert engine_key(_vllm(model="a")) != engine_key(_vllm(model="b"))
    assert engine_key(_vllm(util=0.5)) != engine_key(_vllm(util=0.85))


def test_get_or_create_engine_caches_per_key(clean_engines):
    calls = []

    def factory():
        calls.append(1)
        return object()

    key = ("test", uniq("model"), "{}")
    e1 = get_or_create_engine(key, factory)
    e2 = get_or_create_engine(key, factory)
    assert e1 is e2
    assert len(calls) == 1

    other = ("test", uniq("model"), "{}")
    get_or_create_engine(other, factory)
    assert len(calls) == 2

    clear_engine_cache()
    e3 = get_or_create_engine(key, factory)
    assert len(calls) == 3
    assert e3 is not e1


# --- AC19 ------------------------------------------------------------------

def _model_presets():
    return sorted((CONFIGS / "models").glob("*.yaml"))


def _top_configs():
    return [CONFIGS / "config.yaml"]


def test_model_presets_exist():
    assert _model_presets(), "configs/models/*.yaml missing"
    assert all(p.is_file() for p in _top_configs()), "configs/config.yaml missing"


@pytest.mark.parametrize("path", _model_presets(), ids=lambda p: p.name)
def test_model_preset_fits_24gb(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    cls = get_llm_class(cfg["backend"])
    assert cls.estimate_vram_gb(cfg, 24) <= 24


@pytest.mark.parametrize("path", _top_configs(), ids=lambda p: p.name)
def test_top_level_config_fits_24gb(path):
    cfg = load_config(path)
    assert cfg["hardware"]["vram_gb"] == 24
    check_vram_budget(cfg["llm"], 24)


def test_qwen3_8b_vllm_preset():
    with open(CONFIGS / "models" / "qwen3-8b-vllm.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["model"] == "Qwen/Qwen3-8B"
    assert cfg["backend"] == "vllm"
    assert cfg["engine"]["dtype"] == "bfloat16"


def test_config_targets_qwen3_8b_shared_engine():
    cfg = load_config(CONFIGS / "config.yaml")
    assert cfg["llm"]["target"]["model"] == "Qwen/Qwen3-8B"
    assert check_vram_budget(cfg["llm"], 24) == pytest.approx(20.4, abs=0.5)
