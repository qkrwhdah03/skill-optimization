"""Model presets: fp16 Qwen3-8B vLLM default; bundled presets trimmed (loop 2026-11-11-001)."""
import pytest
import yaml

from helpers import CONFIGS, PROJECT_ROOT
from skillopt.config import load_config, read_raw_config
from skillopt.llm import check_vram_budget, engine_key, get_llm_class

MODELS = CONFIGS / "models"
EXPECTED_PRESETS = ["mock.yaml", "qwen3-8b-fp16-vllm.yaml", "qwen3-8b-vllm.yaml"]
REMOVED_PRESETS = ("qwen3-8b-fp8-vllm", "qwen3-8b-awq-vllm", "qwen3-4b-vllm", "qwen3-8b-hf",
                   "qwen3-8b-hf-4bit", "qwen3-8b-server", "openai-api")
README_REQUIRED = ("qwen3-8b-fp16-vllm", "qwen3-8b-vllm", "mock", "compute capability", "8.0")
FORBIDDEN_NAMES = ("qwen3-8b-fp8-vllm", "-fp8-vllm", "-awq-vllm", "qwen3-4b-vllm", "qwen3-8b-hf",
                   "-hf-4bit", "qwen3-8b-server", "openai-api")


def _load_preset(name):
    with open(MODELS / f"{name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    return cfg


def _without_dtype(cfg):
    out = dict(cfg)
    engine = dict(out.get("engine") or {})
    engine.pop("dtype", None)
    out["engine"] = engine
    return out


# --- AC1 / AC2: preset file layout ------------------------------------------------

def test_model_presets_are_exactly_three():
    assert sorted(p.name for p in MODELS.glob("*.yaml")) == EXPECTED_PRESETS


@pytest.mark.parametrize("name", REMOVED_PRESETS)
def test_removed_preset_file_absent(name):
    assert not (MODELS / f"{name}.yaml").exists(), name


# --- AC3 / AC4: fp16 preset contents ----------------------------------------------

def test_fp16_preset_contents():
    cfg = _load_preset("qwen3-8b-fp16-vllm")
    assert cfg["backend"] == "vllm"
    assert cfg["model"] == "Qwen/Qwen3-8B"
    assert cfg["engine"]["dtype"] == "float16"
    assert cfg["engine"]["gpu_memory_utilization"] == 0.85


def test_bf16_preset_dtype_unchanged():
    assert _load_preset("qwen3-8b-vllm")["engine"]["dtype"] == "bfloat16"


def test_fp16_preset_differs_only_in_dtype():
    fp16 = _load_preset("qwen3-8b-fp16-vllm")
    bf16 = _load_preset("qwen3-8b-vllm")
    assert _without_dtype(fp16) == _without_dtype(bf16)
    assert fp16["engine"]["dtype"] != bf16["engine"]["dtype"]


# --- AC5 / AC6: config.yaml defaults to fp16 --------------------------------------

def test_raw_config_selects_fp16_preset_for_both_roles():
    raw = read_raw_config(CONFIGS / "config.yaml")
    assert raw["llm"]["target"]["preset"] == "qwen3-8b-fp16-vllm"
    assert raw["llm"]["optimizer"]["preset"] == "qwen3-8b-fp16-vllm"


def test_raw_config_roles_do_not_override_engine():
    raw = read_raw_config(CONFIGS / "config.yaml")
    for role in ("target", "optimizer"):
        assert "engine" not in raw["llm"][role], role


def test_resolved_config_fp16_shared_engine():
    cfg = load_config(CONFIGS / "config.yaml")
    target, optimizer = cfg["llm"]["target"], cfg["llm"]["optimizer"]
    for role_cfg in (target, optimizer):
        assert role_cfg["model"] == "Qwen/Qwen3-8B"
        assert role_cfg["backend"] == "vllm"
        assert role_cfg["engine"]["dtype"] == "float16"
    # optimizer has a generation override; engine_key ignores generation, so the engine is shared
    assert engine_key(target) == engine_key(optimizer)
    assert check_vram_budget(cfg["llm"], 24) == pytest.approx(20.4, abs=0.5)


# --- AC7: switching back to bf16 via overrides -------------------------------------

def test_override_to_bf16_preset_shares_engine():
    cfg = load_config(CONFIGS / "config.yaml", overrides=["llm.target.preset=qwen3-8b-vllm",
                                                          "llm.optimizer.preset=qwen3-8b-vllm"])
    target, optimizer = cfg["llm"]["target"], cfg["llm"]["optimizer"]
    assert target["engine"]["dtype"] == "bfloat16"
    assert optimizer["engine"]["dtype"] == "bfloat16"
    assert engine_key(target) == engine_key(optimizer)
    assert check_vram_budget(cfg["llm"], 24) == pytest.approx(20.4, abs=0.5)


# --- AC8: removed preset names no longer resolve -----------------------------------

@pytest.mark.parametrize("name", REMOVED_PRESETS)
def test_removed_preset_override_raises(name):
    with pytest.raises(FileNotFoundError):
        load_config(CONFIGS / "config.yaml", overrides=[f"llm.target.preset={name}"])


# --- AC9: all backends kept --------------------------------------------------------

@pytest.mark.parametrize("backend", ["mock", "vllm", "hf", "openai"])
def test_backend_classes_kept(backend):
    assert isinstance(get_llm_class(backend), type)


# --- AC10: README ------------------------------------------------------------------

def _readme():
    return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("s", README_REQUIRED)
def test_readme_mentions_preset_info(s):
    assert s in _readme(), s


@pytest.mark.parametrize("s", FORBIDDEN_NAMES)
def test_readme_has_no_removed_preset_names(s):
    assert s not in _readme(), s


# --- AC11: config.yaml raw text ----------------------------------------------------

@pytest.mark.parametrize("s", FORBIDDEN_NAMES)
def test_config_yaml_has_no_removed_preset_names(s):
    assert s not in (CONFIGS / "config.yaml").read_text(encoding="utf-8"), s
