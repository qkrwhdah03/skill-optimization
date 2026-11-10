"""J. Config utilities and RunDir."""
import copy
from pathlib import Path

import pytest
import yaml

from helpers import PROJECT_ROOT, read_json
from skillopt.config import PROJECT_ROOT as PKG_ROOT
from skillopt.config import apply_overrides, deep_merge, load_config, resolve_path
from skillopt.runlog import RunDir, set_seed
from skillopt.skills import Skill


def _write_yaml(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)
    return path


def test_project_root_is_repo_root():
    assert Path(PKG_ROOT).resolve() == PROJECT_ROOT


def test_resolve_path():
    assert resolve_path("configs") == PKG_ROOT / "configs"
    abs_p = PROJECT_ROOT / "x" / "y"
    assert resolve_path(abs_p) == abs_p
    assert resolve_path(str(abs_p)) == abs_p


# --- AC41 ------------------------------------------------------------------

def test_deep_merge_nested_without_mutation():
    base = {"a": {"b": 1, "c": {"d": 2}}, "x": [1, 2], "keep": True}
    over = {"a": {"c": {"e": 3}, "b": 10}, "x": [9], "new": "n"}
    base0, over0 = copy.deepcopy(base), copy.deepcopy(over)
    out = deep_merge(base, over)
    assert out == {"a": {"b": 10, "c": {"d": 2, "e": 3}}, "x": [9], "keep": True, "new": "n"}
    assert base == base0
    assert over == over0
    out["a"]["c"]["d"] = 99
    assert base["a"]["c"]["d"] == 2


def test_apply_overrides_typed_values():
    cfg = {"optimize": {"iterations": 5}, "run": {"name": "old"}}
    out = apply_overrides(cfg, ["optimize.iterations=1", "run.name=x", "a.new.key=[1, 2]"])
    assert out["optimize"]["iterations"] == 1
    assert isinstance(out["optimize"]["iterations"], int)
    assert out["run"]["name"] == "x"
    assert out["a"]["new"]["key"] == [1, 2]


def test_apply_overrides_float_bool_and_equals_in_value():
    out = apply_overrides({}, ["x.f=0.85", "x.b=false", "x.s=a=b"])
    assert out["x"]["f"] == pytest.approx(0.85)
    assert out["x"]["b"] is False
    assert out["x"]["s"] == "a=b"


def test_apply_overrides_without_equals():
    with pytest.raises(ValueError):
        apply_overrides({}, ["optimize.iterations"])


# --- AC42 ------------------------------------------------------------------

def _preset_setup(tmp_path):
    _write_yaml(tmp_path / "models" / "p1.yaml", {
        "backend": "mock", "model": "preset-model",
        "generation": {"max_tokens": 100, "temperature": 0.6},
        "engine": {"gpu_memory_utilization": 0.5},
    })
    return _write_yaml(tmp_path / "cfg.yaml", {
        "hardware": {"vram_gb": 24},
        "llm": {
            "target": {"preset": "p1", "generation": {"temperature": 0.0}},
            "optimizer": {"backend": "mock", "model": "plain"},
        },
    })


def test_load_config_resolves_presets(tmp_path):
    cfg = load_config(_preset_setup(tmp_path))
    t = cfg["llm"]["target"]
    assert "preset" not in t
    assert t["backend"] == "mock"
    assert t["model"] == "preset-model"
    assert t["generation"] == {"max_tokens": 100, "temperature": 0.0}
    assert t["engine"] == {"gpu_memory_utilization": 0.5}
    assert cfg["llm"]["optimizer"] == {"backend": "mock", "model": "plain"}


def test_load_config_presets_dir(tmp_path):
    _write_yaml(tmp_path / "elsewhere" / "p2.yaml", {"backend": "mock", "model": "m2"})
    path = _write_yaml(tmp_path / "sub" / "cfg.yaml", {"llm": {"target": {"preset": "p2"}}})
    cfg = load_config(path, presets_dir=tmp_path / "elsewhere")
    assert cfg["llm"]["target"]["model"] == "m2"
    assert "preset" not in cfg["llm"]["target"]


def test_load_config_missing_preset(tmp_path):
    path = _write_yaml(tmp_path / "cfg.yaml", {"llm": {"target": {"preset": "nope"}}})
    with pytest.raises(FileNotFoundError):
        load_config(path)


def test_load_config_overrides_apply_after_presets(tmp_path):
    cfg = load_config(_preset_setup(tmp_path), overrides=["llm.target.generation.max_tokens=7"])
    assert cfg["llm"]["target"]["generation"]["max_tokens"] == 7
    assert cfg["llm"]["target"]["generation"]["temperature"] == 0.0


# --- AC49 / RunDir ---------------------------------------------------------

def test_rundir_same_second_distinct(tmp_path):
    a = RunDir(tmp_path, "same", {"x": 1})
    b = RunDir(tmp_path, "same", {"x": 1})
    assert a.path != b.path
    assert a.path.is_dir() and b.path.is_dir()
    assert a.path.parent == tmp_path and b.path.parent == tmp_path
    assert "same" in a.path.name and "same" in b.path.name


def test_rundir_writes_config_and_files(tmp_path):
    rd = RunDir(tmp_path, "w", {"seed": 3, "nested": {"k": "v"}})
    cfg_path = rd.path / "config.yaml"
    assert cfg_path.exists()
    with open(cfg_path, encoding="utf-8") as f:
        assert yaml.safe_load(f)["nested"] == {"k": "v"}
    sp = rd.save_skill(Skill(id="sk-1", instruction="Hi."))
    assert sp == rd.path / "skills" / "sk-1.json"
    assert Skill.from_dict(read_json(sp)) == Skill(id="sk-1", instruction="Hi.")
    jp = rd.save_json("selection.json", {"a": [1, 2]})
    assert jp == rd.path / "selection.json"
    assert read_json(jp) == {"a": [1, 2]}


def test_rundir_save_eval_and_usage(tmp_path):
    from helpers import make_result
    from skillopt.llm import UsageSnapshot

    rd = RunDir(tmp_path, "e", {})
    res = make_result("sk", n=3, n_correct=2, output_tokens=5, split="train", benchmark="gsm8k")
    summary_path = rd.save_eval(res, "train")
    assert summary_path == rd.path / "results" / "train" / "gsm8k" / "sk.summary.json"
    summary = read_json(summary_path)
    assert summary["n"] == 3
    assert summary["split"] == "train"
    assert summary["benchmark"] == "gsm8k"
    jl = rd.path / "results" / "train" / "gsm8k" / "sk.jsonl"
    lines = jl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    up = rd.save_usage("target", UsageSnapshot(calls=2, input_tokens=1))
    assert up == rd.path / "usage" / "target.json"
    assert read_json(up)["calls"] == 2


def test_rundir_save_evaluation_one_pair_per_benchmark(tmp_path):
    from helpers import count_lines, make_evaluation

    rd = RunDir(tmp_path, "ev", {})
    ev = make_evaluation("sk2", {"bbh_alg": (0.5, 10, 4), "bbh_b": (1.0, 20, 2), "empty": (0.0, 1, 0)},
                         split="val")
    assert ev.split == "val"
    paths = rd.save_evaluation(ev)
    base = rd.path / "results" / "val"
    assert paths == [base / "bbh_alg" / "sk2.summary.json", base / "bbh_b" / "sk2.summary.json",
                     base / "empty" / "sk2.summary.json"]
    assert read_json(paths[0])["n"] == 4
    assert read_json(paths[1])["n"] == 2
    assert read_json(paths[2])["n"] == 0
    assert count_lines(base / "bbh_alg" / "sk2.jsonl") == 4
    assert count_lines(base / "bbh_b" / "sk2.jsonl") == 2
    assert (base / "empty" / "sk2.jsonl").is_file()
    assert count_lines(base / "empty" / "sk2.jsonl") == 0


def test_rundir_save_eval_test_split(tmp_path):
    from helpers import make_result

    rd = RunDir(tmp_path, "t", {})
    res = make_result("sk", n=2, n_correct=1, split="test", benchmark="bbh")
    summary_path = rd.save_eval(res, "test")
    base = rd.path / "results" / "test" / "bbh"
    assert summary_path == base / "sk.summary.json"
    assert (base / "sk.jsonl").is_file()
    assert read_json(summary_path)["split"] == "test"
    assert not (rd.path / "results" / "final").exists()


# --- AC33: list indexing in overrides ----------------------------------------

def _bench_cfg():
    return {"benchmarks": [{"name": "gsm8k", "dev_size": 4}, {"name": "bbh"}]}


def test_apply_overrides_indexes_lists_without_mutation():
    cfg = _bench_cfg()
    before = copy.deepcopy(cfg)
    out = apply_overrides(cfg, ["benchmarks.0.dev_size=100", "benchmarks.1.splits.search=val"])
    assert out["benchmarks"][0]["dev_size"] == 100
    assert isinstance(out["benchmarks"][0]["dev_size"], int)
    assert out["benchmarks"][0]["name"] == "gsm8k"
    assert out["benchmarks"][1]["splits"] == {"search": "val"}
    assert out["benchmarks"][1]["name"] == "bbh"
    assert cfg == before


def test_apply_overrides_list_index_at_final_component():
    out = apply_overrides(_bench_cfg(), ["benchmarks.0={name: other, dev_size: 1}"])
    assert out["benchmarks"][0] == {"name": "other", "dev_size": 1}
    assert out["benchmarks"][1] == {"name": "bbh"}
    assert len(out["benchmarks"]) == 2


@pytest.mark.parametrize("bad", ["benchmarks.5.x=1", "benchmarks.a.x=1"])
def test_apply_overrides_bad_list_index(bad):
    with pytest.raises(ValueError):
        apply_overrides(_bench_cfg(), [bad])


def test_apply_overrides_dict_behavior_unchanged():
    out = apply_overrides({"a": {"b": 1}}, ["a.c=2", "new.deep.k=x", "a.b=null"])
    assert out == {"a": {"b": None, "c": 2}, "new": {"deep": {"k": "x"}}}


# --- AC33a: overrides applied before preset resolution ----------------------

def test_load_config_override_selects_preset(tmp_path):
    _write_yaml(tmp_path / "models" / "p1.yaml", {"backend": "mock", "model": "m-one",
                                                  "generation": {"max_tokens": 1}})
    _write_yaml(tmp_path / "models" / "p2.yaml", {"backend": "mock", "model": "m-two",
                                                  "generation": {"max_tokens": 2}})
    path = _write_yaml(tmp_path / "cfg.yaml", {"llm": {"target": {"preset": "p1"}}})
    cfg = load_config(path, overrides=["llm.target.preset=p2"])
    t = cfg["llm"]["target"]
    assert "preset" not in t
    assert t["model"] == "m-two"
    assert t["generation"] == {"max_tokens": 2}
    cfg1 = load_config(path)
    assert cfg1["llm"]["target"]["model"] == "m-one"


def test_load_config_override_explicit_key_beats_preset(tmp_path):
    _write_yaml(tmp_path / "models" / "p2.yaml", {"backend": "mock", "model": "m-two",
                                                  "generation": {"max_tokens": 2, "temperature": 0.5}})
    path = _write_yaml(tmp_path / "cfg.yaml", {"llm": {"target": {"preset": "p2"}}})
    cfg = load_config(path, overrides=["llm.target.generation.max_tokens=7"])
    assert cfg["llm"]["target"]["generation"] == {"max_tokens": 7, "temperature": 0.5}


def test_load_config_override_missing_preset(tmp_path):
    _write_yaml(tmp_path / "models" / "p1.yaml", {"backend": "mock", "model": "m-one"})
    path = _write_yaml(tmp_path / "cfg.yaml", {"llm": {"target": {"preset": "p1"}}})
    with pytest.raises(FileNotFoundError):
        load_config(path, overrides=["llm.target.preset=missing"])


def test_set_seed_runs():
    import random

    set_seed(5)
    a = random.random()
    set_seed(5)
    assert random.random() == a
