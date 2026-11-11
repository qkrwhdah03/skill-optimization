"""H. Packaging, configs, scripts & docs."""
import os
import subprocess
import sys

import pytest
import yaml

from helpers import CONFIGS, PROJECT_ROOT, read_json

OPT_DIR = PROJECT_ROOT / "skillopt" / "optimizer"


# --- AC28: README -----------------------------------------------------------------

README_REQUIRED = ("```mermaid", "run.py", "configs/config.yaml", "--mock", "--eval-only", "initial_prompts",
                   "update", "register_optimizer", "my_optimizer", "SkillEvaluation", "EvalResult",
                   "max_iterations", "register_llm", "register_benchmark", "Qwen3-8B", "benchmarks:",
                   "## Benchmarks", "openai/gsm8k", "lukaemon/bbh", "https://arxiv.org/abs/2110.14168",
                   "https://aclanthology.org/2023.findings-acl.824/", "7,473", "1,319", "6,511", "--split",
                   "benchmarks.gsm8k.train", "heldout")
README_FORBIDDEN = ("register_generator", "SkillGenerator", "EvaluatedSkill", "GenerationContext",
                    "reference_baseline", "search_stage", "selection.json", "aggregator", "selector",
                    "register_strategy", "run_optimize.py", "run_eval.py",
                    "dev_size", "val_size", "heldout_tasks", "--stage", "splits:", "benchmarks.0")


def _readme():
    readme = PROJECT_ROOT / "README.md"
    assert readme.is_file()
    return readme.read_text(encoding="utf-8")


@pytest.mark.parametrize("s", README_REQUIRED)
def test_readme_contains(s):
    assert s in _readme(), s


@pytest.mark.parametrize("s", README_FORBIDDEN)
def test_readme_does_not_contain(s):
    assert s not in _readme(), s


# --- AC29: optimizer package layout -----------------------------------------------

@pytest.mark.parametrize("name", ["generation.py", "scoring.py", "selection.py", "stopping.py", "final.py"])
def test_removed_optimizer_modules(name):
    assert not (OPT_DIR / name).exists(), name


def test_generators_folder_removed():
    assert not (OPT_DIR / "generators").exists()


@pytest.mark.parametrize("rel", ["base.py", "loop.py", "algorithms/my_optimizer.py", "algorithms/opro.py"])
def test_new_optimizer_files_exist(rel):
    assert (OPT_DIR / rel).is_file(), rel


# --- AC44 ---------------------------------------------------------------------

@pytest.mark.parametrize("name", ["run.py", "pyproject.toml"])
def test_root_files_exist(name):
    assert (PROJECT_ROOT / name).is_file()


@pytest.mark.parametrize("name", ["run_optimize.py", "run_eval.py"])
def test_old_entry_points_removed(name):
    assert not (PROJECT_ROOT / name).exists()


# --- AC43 ---------------------------------------------------------------------

def test_only_one_top_level_config():
    assert sorted(p.name for p in CONFIGS.glob("*.yaml")) == ["config.yaml"]
    for preset in ("mock.yaml", "qwen3-8b-vllm.yaml", "qwen3-8b-fp16-vllm.yaml"):
        assert (CONFIGS / "models" / preset).is_file(), preset


def test_config_yaml_contents():
    """AC27."""
    from skillopt.config import load_config
    from skillopt.llm import check_vram_budget
    from skillopt.optimizer import OPTIMIZER_REGISTRY

    cfg = load_config(CONFIGS / "config.yaml")
    assert cfg["hardware"]["vram_gb"] == 24
    total = check_vram_budget(cfg["llm"], 24)
    assert total == pytest.approx(20.4, abs=0.5)
    assert cfg["llm"]["target"]["model"] == "Qwen/Qwen3-8B"
    assert cfg["llm"]["optimizer"]["model"] == "Qwen/Qwen3-8B"
    assert "benchmark" not in cfg
    assert isinstance(cfg["benchmarks"], dict)
    assert list(cfg["benchmarks"]) == ["gsm8k", "bbh"]
    assert "eval" in cfg
    assert "optimize" not in cfg
    assert isinstance(cfg["optimizer"], dict)
    assert cfg["optimizer"]["name"] == "my_optimizer"
    assert cfg["optimizer"]["name"] in OPTIMIZER_REGISTRY
    assert isinstance(cfg["max_iterations"], int) and not isinstance(cfg["max_iterations"], bool)
    assert cfg["max_iterations"] >= 0


def test_config_yaml_raw_top_level_keys():
    with open(CONFIGS / "config.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert "optimize" not in raw
    assert "optimizer" in raw and "max_iterations" in raw
    assert raw["optimizer"]["name"] == "my_optimizer"


def test_config_yaml_mock_builds_specs():
    from skillopt.cli import mock_overrides
    from skillopt.config import load_config
    from skillopt.evaluator import build_benchmark_specs

    with open(CONFIGS / "config.yaml", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = load_config(CONFIGS / "config.yaml", overrides=mock_overrides(raw))
    specs = build_benchmark_specs(cfg["benchmarks"], seed=0)
    assert [s.key for s in specs] == ["gsm8k", "bbh"]


@pytest.mark.parametrize("rel,min_lines", [
    ("data/sample/gsm8k/train.jsonl", 16),
    ("data/sample/gsm8k/test.jsonl", 6),
    ("data/sample/bbh/boolean_expressions.jsonl", 8),
    ("data/sample/bbh/date_understanding.jsonl", 8),
    ("data/sample/bbh/web_of_lies.jsonl", 8),
])
def test_sample_data_present(rel, min_lines):
    from helpers import count_lines

    p = PROJECT_ROOT / rel
    assert p.is_file()
    assert count_lines(p) >= min_lines


# --- AC42a --------------------------------------------------------------------

def test_run_py_subprocess_mock_eval_only(tmp_path):
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        [sys.executable, "run.py", "--mock", "--eval-only", "--limit", "1",
         "--set", f"run.root={tmp_path.as_posix()}"],
        cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300,
    )
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-2000:])
    runs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(runs) == 1
    assert len(read_json(runs[0] / "summary.json")) == 3


# --- loop 2026-11-10-004: mapping-form benchmarks, train split, sizes documented ---

EXPECTED_BENCHMARKS = {
    "gsm8k": {"source": "hf", "train": 200, "val": 200},
    "bbh": {"source": "hf", "tasks": "all", "heldout": ["web_of_lies"], "train": 10, "val": 10},
}
OLD_KEYS = ("dev_size", "val_size", "splits", "heldout_tasks")


def _raw_config():
    with open(CONFIGS / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_config_yaml_benchmarks_mapping_exact():
    raw = _raw_config()
    assert isinstance(raw["benchmarks"], dict)
    assert raw["benchmarks"] == EXPECTED_BENCHMARKS
    assert list(raw["benchmarks"]) == ["gsm8k", "bbh"]


def test_config_yaml_eval_split_not_stage():
    raw = _raw_config()
    assert raw["eval"]["split"] == "test"
    assert "stage" not in raw["eval"]


@pytest.mark.parametrize("old", OLD_KEYS)
def test_config_yaml_has_no_old_keys(old):
    assert old not in set(_walk_keys(_raw_config())), old


@pytest.mark.parametrize("size", ["7,473", "1,319", "6,511"])
def test_config_yaml_size_comments(size):
    assert size in (CONFIGS / "config.yaml").read_text(encoding="utf-8"), size


def test_readme_search_stage_absent_case_insensitive():
    assert "search_stage" not in _readme().lower()


@pytest.mark.parametrize("rel,hf_id", [
    ("skillopt/benchmarks/gsm8k.py", "openai/gsm8k"),
    ("skillopt/benchmarks/bbh.py", "lukaemon/bbh"),
])
def test_readme_hf_ids_match_code(rel, hf_id):
    assert hf_id in (PROJECT_ROOT / rel).read_text(encoding="utf-8")
    assert hf_id in _readme()


@pytest.mark.parametrize("s", ["--stage", "dev_size"])
def test_run_py_has_no_old_flags(s):
    assert s not in (PROJECT_ROOT / "run.py").read_text(encoding="utf-8"), s
