"""H. Packaging, configs, scripts & docs."""
import os
import subprocess
import sys

import pytest
import yaml

from helpers import CONFIGS, PROJECT_ROOT, read_json

OPT_DIR = PROJECT_ROOT / "skillopt" / "optimizer"
ALL_BENCHMARK_KEYS = [
    "gsm8k", "bbh", "math500", "bbeh", "aime2024", "aime2025", "amc2023",
    "musr", "folio", "lsat_ar", "strategyqa", "mmlu_pro",
]


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
    assert total == pytest.approx(22.08, abs=0.05)
    assert total <= 24
    assert cfg["llm"]["target"]["model"] == "Qwen/Qwen3-8B"
    assert cfg["llm"]["optimizer"]["model"] == "Qwen/Qwen3-8B"
    assert "benchmark" not in cfg
    assert isinstance(cfg["benchmarks"], dict)
    assert list(cfg["benchmarks"]) == ALL_BENCHMARK_KEYS
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
    assert [s.key for s in specs] == ALL_BENCHMARK_KEYS


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
    "math500": {"source": "hf", "train": 50, "val": 50},
    "bbeh": {"source": "hf", "tasks": "all", "mini": False, "train": 2, "val": 2},
    "aime2024": {"source": "hf"},
    "aime2025": {"source": "hf"},
    "amc2023": {"source": "hf"},
    "musr": {"source": "hf", "tasks": "all", "train": 10, "val": 10},
    "folio": {"source": "hf", "train": 40, "val": 40},
    "lsat_ar": {"source": "hf", "train": 20, "val": 20},
    "strategyqa": {"source": "hf", "train": 40, "val": 40},
    "mmlu_pro": {"source": "hf", "categories": "non_math", "train": 4, "val": 4},
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
    assert list(raw["benchmarks"]) == ALL_BENCHMARK_KEYS


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


# --- loop 2026-11-16-003: long-reasoning benchmarks (MATH-500, AIME, AMC, BBEH) ---

def test_config_yaml_optimizer_generation_kept():
    raw = _raw_config()
    gen = raw["llm"]["optimizer"]["generation"]
    assert gen["max_tokens"] == 1024
    assert gen["enable_thinking"] is False


def test_config_yaml_mock_builds_all_specs_local():
    from skillopt.cli import mock_overrides
    from skillopt.config import load_config
    from skillopt.evaluator import build_benchmark_specs

    raw = _raw_config()
    cfg = load_config(CONFIGS / "config.yaml", overrides=mock_overrides(raw))
    for key in ALL_BENCHMARK_KEYS:
        assert cfg["benchmarks"][key]["source"] == "local", key
    assert cfg["benchmarks"]["math500"]["train"] == 2
    assert cfg["benchmarks"]["math500"]["val"] == 2
    assert cfg["benchmarks"]["bbeh"]["tasks"] == "all"
    assert cfg["benchmarks"]["bbeh"]["train"] == 2
    assert cfg["benchmarks"]["bbeh"]["val"] == 2
    specs = build_benchmark_specs(cfg["benchmarks"], seed=0)
    assert [s.key for s in specs] == ALL_BENCHMARK_KEYS


NEW_SAMPLE_FILES = [
    ("data/sample/math500/data.jsonl", 8),
    ("data/sample/aime2024/data.jsonl", 4),
    ("data/sample/aime2025/data.jsonl", 4),
    ("data/sample/amc2023/data.jsonl", 4),
    ("data/sample/bbeh/multistep_arithmetic.jsonl", 6),
    ("data/sample/bbeh/web_of_lies.jsonl", 6),
    ("data/sample/bbeh/geometric_shapes.jsonl", 6),
]

SAMPLE_REQUIRED_FIELDS = {
    "math500": ("problem", "answer"),
    "aime2024": ("problem", "answer"),
    "aime2025": ("problem", "answer"),
    "amc2023": ("question", "answer"),
    "bbeh": ("input", "target", "mini"),
}


@pytest.mark.parametrize("rel,lines", NEW_SAMPLE_FILES)
def test_new_sample_data_line_counts(rel, lines):
    from helpers import count_lines

    p = PROJECT_ROOT / rel
    assert p.is_file(), rel
    assert count_lines(p) == lines


@pytest.mark.parametrize("rel,lines", NEW_SAMPLE_FILES)
def test_new_sample_data_required_fields(rel, lines):
    from helpers import read_jsonl

    bench = rel.split("/")[2]
    rows = read_jsonl(PROJECT_ROOT / rel)
    for row in rows:
        for field in SAMPLE_REQUIRED_FIELDS[bench]:
            assert field in row, (rel, field)
            assert str(row[field]).strip() != "" or field == "mini", (rel, field)


@pytest.mark.parametrize("task", ["multistep_arithmetic", "web_of_lies", "geometric_shapes"])
def test_bbeh_sample_mini_rows(task):
    from helpers import read_jsonl

    rows = read_jsonl(PROJECT_ROOT / "data" / "sample" / "bbeh" / f"{task}.jsonl")
    assert sum(1 for r in rows if r["mini"] == 1) == 2


@pytest.mark.parametrize("name,dirname,n", [
    ("math500", "math500", 8),
    ("aime2024", "aime2024", 4),
    ("aime2025", "aime2025", 4),
    ("amc2023", "amc2023", 4),
])
def test_new_sample_data_loads_pooled(name, dirname, n):
    from skillopt.benchmarks import build_benchmark

    bench = build_benchmark({"name": name, "source": "local", "data_dir": f"data/sample/{dirname}"})
    test = bench.load("test")
    assert len(test) == n
    assert all(e.question.strip() for e in test)
    assert all(str(e.answer).strip() for e in test)


@pytest.mark.parametrize("name,n", [("math500", 8), ("aime2024", 4), ("aime2025", 4), ("amc2023", 4)])
def test_new_sample_data_default_data_dir(name, n):
    from skillopt.benchmarks import build_benchmark

    assert len(build_benchmark({"name": name, "source": "local"}).load("test")) == n


def test_bbeh_sample_data_loads():
    from skillopt.benchmarks import build_benchmark

    bench = build_benchmark({"name": "bbeh", "source": "local", "data_dir": "data/sample/bbeh"})
    test = bench.load("test")
    assert len(test) == 18
    assert {e.meta["task"] for e in test} == {"multistep_arithmetic", "web_of_lies", "geometric_shapes"}
    assert all(e.question.strip() for e in test)
    mini = build_benchmark({"name": "bbeh", "source": "local", "data_dir": "data/sample/bbeh", "mini": True})
    assert len(mini.load("test")) == 6


README_NEW_REQUIRED = ("HuggingFaceH4/MATH-500", "HuggingFaceH4/aime_2024", "math-ai/aime25", "math-ai/amc23",
                       "BBEH/bbeh", "https://arxiv.org/abs/2502.19187", "math500", "bbeh")


@pytest.mark.parametrize("s", README_NEW_REQUIRED)
def test_readme_contains_new_benchmarks(s):
    assert s in _readme(), s


def test_readme_mentions_32k_context():
    text = _readme()
    assert "32768" in text or "32,768" in text


# --- loop 2026-11-16-004: GPQA removed ---

KEPT_BENCHMARK_KEYS = [
    "gsm8k", "bbh", "math500", "bbeh", "aime2024", "aime2025", "amc2023",
    "musr", "folio", "lsat_ar", "strategyqa", "mmlu_pro",
]


def test_benchmarks_package_still_imports():
    r = subprocess.run([sys.executable, "-c", "import skillopt.benchmarks"], cwd=PROJECT_ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]


@pytest.mark.parametrize("name", ["gpqa", "GPQA", "Gpqa", "gpqa_diamond"])
def test_gpqa_not_registered(name):
    from skillopt.benchmarks import get_benchmark_class

    with pytest.raises(KeyError):
        get_benchmark_class(name)


def test_gpqa_build_benchmark_raises():
    from skillopt.benchmarks import build_benchmark

    with pytest.raises(KeyError):
        build_benchmark({"name": "gpqa", "source": "local"})


def test_gpqa_module_not_importable():
    import importlib

    sys.modules.pop("skillopt.benchmarks.gpqa", None)
    with pytest.raises(ImportError):
        importlib.import_module("skillopt.benchmarks.gpqa")


@pytest.mark.parametrize("attr", ["GPQA", "GPQA_SUBSETS"])
def test_gpqa_not_exported(attr):
    import skillopt.benchmarks as b

    assert not hasattr(b, attr), attr
    assert attr not in b.__all__, attr


@pytest.mark.parametrize("rel", ["skillopt/benchmarks/gpqa.py", "data/sample/gpqa"])
def test_gpqa_files_removed(rel):
    assert not (PROJECT_ROOT / rel).exists(), rel


def test_config_yaml_benchmark_keys_without_gpqa():
    assert list(_raw_config()["benchmarks"]) == KEPT_BENCHMARK_KEYS


def test_config_yaml_text_has_no_gpqa():
    assert "gpqa" not in (CONFIGS / "config.yaml").read_text(encoding="utf-8").lower()


def test_mock_overrides_have_no_gpqa():
    from skillopt.cli import _MOCK_BENCHMARK_OVERRIDES

    assert "gpqa" not in _MOCK_BENCHMARK_OVERRIDES
    assert set(_MOCK_BENCHMARK_OVERRIDES) == {
        "gsm8k", "bbh", "math500", "aime2024", "aime2025", "amc2023", "bbeh",
        "musr", "folio", "lsat_ar", "strategyqa", "mmlu_pro",
    }


@pytest.mark.parametrize("s", ["gpqa", "idavidrein", "2311.12022"])
def test_readme_has_no_gpqa(s):
    assert s not in _readme().lower(), s


NON_MATH_SAMPLE_FILES = [
    ("data/sample/musr/murder_mysteries.jsonl", 6),
    ("data/sample/musr/object_placements.jsonl", 6),
    ("data/sample/musr/team_allocation.jsonl", 6),
    ("data/sample/folio/train.jsonl", 10),
    ("data/sample/folio/test.jsonl", 4),
    ("data/sample/lsat_ar/data.jsonl", 8),
    ("data/sample/strategyqa/train.jsonl", 10),
    ("data/sample/strategyqa/test.jsonl", 4),
    ("data/sample/mmlu_pro/law.jsonl", 6),
    ("data/sample/mmlu_pro/philosophy.jsonl", 6),
    ("data/sample/mmlu_pro/biology.jsonl", 6),
]


@pytest.mark.parametrize("rel,lines", NON_MATH_SAMPLE_FILES)
def test_non_math_sample_data_line_counts(rel, lines):
    from helpers import count_lines

    p = PROJECT_ROOT / rel
    assert p.is_file(), rel
    assert count_lines(p) == lines


@pytest.mark.parametrize("s", [
    "TAUR-Lab/MuSR", "tasksource/folio", "hails/agieval-lsat-ar", "ChilleD/StrategyQA", "TIGER-Lab/MMLU-Pro",
    "musr", "folio", "lsat_ar", "strategyqa", "mmlu_pro", "Non-math reasoning benchmarks",
])
def test_readme_contains_non_math_benchmarks(s):
    assert s in _readme(), s
