"""G. End-to-end CLI runs: `main` with `--mock` on configs/config.yaml; round-based run dir (AC19-26)."""
import argparse
import copy
import shutil

import pytest
import yaml

from helpers import CONFIGS, make_mock, read_json, read_jsonl, uniq
from skillopt.baselines import get_baseline
from skillopt.cli import DEFAULT_CONFIG, build_parser, main, mock_overrides, run_eval_only, run_loop
from skillopt.config import load_config
from skillopt.evaluator import Evaluator, build_benchmark_specs
from skillopt.llm import BaseLLM, VRAMBudgetError

CONFIG = CONFIGS / "config.yaml"
SNAPSHOT_KEYS = {"calls", "input_tokens", "output_tokens", "reasoning_tokens", "total_tokens", "latency"}
EVAL_SUMMARY_KEYS = {"skill_id", "benchmark", "split", "n", "accuracy", "avg_input_tokens",
                     "avg_output_tokens", "avg_reasoning_tokens", "avg_total_tokens", "avg_latency",
                     "num_truncated", "truncation_rate"}
BENCH_KEYS = ["gsm8k", "bbh"]
OPRO = "optimizer.name=opro"


def _base(root):
    return ["--config", str(CONFIG), "--mock", "--set", f"run.root={root.as_posix()}"]


def _run(root, *sets, extra=()):
    argv = _base(root)
    for s in sets:
        argv += ["--set", s]
    return main(argv + list(extra))


def _raw_config():
    with open(CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _mock_cfg(root, *extra):
    """Fully loaded mock config (as main would build it), for calling run_loop / run_eval_only directly."""
    return load_config(CONFIG, overrides=mock_overrides(_raw_config()) +
                       [f"run.root={root.as_posix()}", *extra])


def _rounds(run):
    return sorted(p.name for p in (run / "rounds").iterdir() if p.is_dir())


def _round_skill_ids(run, rnd):
    return sorted(p.stem for p in (run / "rounds" / rnd / "skills").glob("*.json"))


def _strip_latency(summary):
    out = copy.deepcopy(summary)
    for entry in out:
        for r in entry["benchmarks"].values():
            r.pop("avg_latency", None)
    return out


@pytest.fixture(scope="module")
def opro_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("opro_root")
    return root, _run(root, OPRO)


@pytest.fixture
def no_llm_built(monkeypatch):
    """Records every LLM construction; config errors must be raised before any is built."""
    built = []
    orig = BaseLLM.__init__

    def spy(self, *a, **kw):
        built.append(type(self).__name__)
        orig(self, *a, **kw)

    monkeypatch.setattr(BaseLLM, "__init__", spy)
    return built


# --- AC19 ---------------------------------------------------------------------

def test_cli_constants_and_parser():
    assert DEFAULT_CONFIG == "configs/config.yaml"
    assert isinstance(build_parser(), argparse.ArgumentParser)


def test_one_run_dir_under_root(opro_run):
    root, run = opro_run
    assert run.is_dir()
    assert root in run.parents
    assert [p for p in root.rglob("run.json")] == [run / "run.json"]


def test_run_basic_files(opro_run):
    _, run = opro_run
    assert (run / "config.yaml").is_file()
    with open(run / "config.yaml", encoding="utf-8") as f:
        saved = yaml.safe_load(f)
    assert saved["llm"]["target"]["backend"] == "mock"
    assert saved["optimizer"]["name"] == "opro"
    assert "seed" in read_json(run / "meta.json")


def test_run_json(opro_run):
    _, run = opro_run
    assert read_json(run / "run.json") == {"rounds": 3, "stop_reason": "max_iterations", "max_iterations": 2}


def test_round_dirs(opro_run):
    _, run = opro_run
    assert _rounds(run) == ["000", "001", "002"]


def test_round0_is_initial_prompts(opro_run):
    _, run = opro_run
    assert _round_skill_ids(run, "000") == ["cod", "cot", "direct"]
    assert [s["skill_id"] for s in read_json(run / "rounds" / "000" / "summary.json")] == ["direct", "cot", "cod"]


def test_later_rounds_are_opro_proposals(opro_run):
    _, run = opro_run
    for i, rnd in ((1, "001"), (2, "002")):
        ids = _round_skill_ids(run, rnd)
        assert ids, rnd
        assert all(sid.startswith(f"opro-i{i}-") for sid in ids), ids


# --- AC20 ---------------------------------------------------------------------

def test_round_files_complete(opro_run):
    _, run = opro_run
    for rnd in _rounds(run):
        d = run / "rounds" / rnd
        summary = read_json(d / "summary.json")
        assert isinstance(summary, list) and summary
        ids = [s["skill_id"] for s in summary]
        assert sorted(ids) == _round_skill_ids(run, rnd)
        for entry in summary:
            sid = entry["skill_id"]
            assert set(entry) == {"skill_id", "split", "benchmarks"}
            assert entry["split"] == "train"
            assert list(entry["benchmarks"]) == BENCH_KEYS
            skill = read_json(d / "skills" / f"{sid}.json")
            assert skill["id"] == sid
            assert "instruction" in skill
            for key in BENCH_KEYS:
                jl = d / "results" / key / f"{sid}.jsonl"
                sm = d / "results" / key / f"{sid}.summary.json"
                assert jl.is_file(), jl
                assert sm.is_file(), sm
                s = read_json(sm)
                assert set(s) == EVAL_SUMMARY_KEYS
                assert s == entry["benchmarks"][key]
                assert s["skill_id"] == sid
                assert s["benchmark"] == key
                assert s["split"] == "train"
                assert len(read_jsonl(jl)) == s["n"]


def test_round0_train_sizes(opro_run):
    _, run = opro_run
    d = run / "rounds" / "000" / "results"
    for b in ("direct", "cot", "cod"):
        assert read_json(d / "gsm8k" / f"{b}.summary.json")["n"] == 4
        assert read_json(d / "bbh" / f"{b}.summary.json")["n"] == 4


def test_round_usage_cumulative(opro_run):
    _, run = opro_run
    prev_t = prev_o = -1
    total_n = 0
    for rnd in _rounds(run):
        d = run / "rounds" / rnd
        usage = read_json(d / "usage.json")
        assert set(usage) == {"target", "optimizer"}
        assert set(usage["target"]) == SNAPSHOT_KEYS
        assert set(usage["optimizer"]) == SNAPSHOT_KEYS
        total_n += sum(e["benchmarks"][k]["n"] for e in read_json(d / "summary.json") for k in BENCH_KEYS)
        assert usage["target"]["calls"] >= prev_t
        assert usage["optimizer"]["calls"] >= prev_o
        assert usage["target"]["calls"] >= total_n
        prev_t, prev_o = usage["target"]["calls"], usage["optimizer"]["calls"]
    u0 = read_json(run / "rounds" / "000" / "usage.json")
    u1 = read_json(run / "rounds" / "001" / "usage.json")
    assert u0["optimizer"]["calls"] == 0
    assert u1["optimizer"]["calls"] > 0
    assert u1["target"]["calls"] > u0["target"]["calls"]


def test_top_level_usage_files(opro_run):
    _, run = opro_run
    tgt = read_json(run / "usage" / "target.json")
    opt = read_json(run / "usage" / "optimizer.json")
    assert set(tgt) == SNAPSHOT_KEYS and set(opt) == SNAPSHOT_KEYS
    last = read_json(run / "rounds" / "002" / "usage.json")
    assert tgt["calls"] >= last["target"]["calls"] > 0
    assert opt["calls"] >= last["optimizer"]["calls"] > 0


def test_no_judgment_artifacts(opro_run):
    _, run = opro_run
    assert not list(run.rglob("selection.json"))
    assert not (run / "results").exists()
    assert not (run / "skills").exists()


# --- AC21 ---------------------------------------------------------------------

def test_deterministic(tmp_path, opro_run):
    _, run1 = opro_run
    run2 = _run(tmp_path / "second", OPRO)
    assert _rounds(run1) == _rounds(run2)
    for rnd in _rounds(run1):
        s1 = read_json(run1 / "rounds" / rnd / "summary.json")
        s2 = read_json(run2 / "rounds" / rnd / "summary.json")
        # avg_latency is wall-clock timing, not a model output; everything else must be identical
        assert _strip_latency(s1) == _strip_latency(s2), rnd
        assert _round_skill_ids(run1, rnd) == _round_skill_ids(run2, rnd)


# --- AC22 ---------------------------------------------------------------------

def test_zero_iterations(tmp_path):
    run = _run(tmp_path, OPRO, "max_iterations=0")
    assert read_json(run / "run.json") == {"rounds": 1, "stop_reason": "max_iterations", "max_iterations": 0}
    assert _rounds(run) == ["000"]
    assert _round_skill_ids(run, "000") == ["cod", "cot", "direct"]
    assert read_json(run / "rounds" / "000" / "usage.json")["optimizer"]["calls"] == 0


def test_one_iteration(tmp_path):
    run = _run(tmp_path, OPRO, "max_iterations=1")
    assert read_json(run / "run.json")["rounds"] == 2
    assert _rounds(run) == ["000", "001"]


def test_benchmark_override_by_key(tmp_path):
    run = _run(tmp_path, OPRO, "benchmarks.gsm8k.train=2", "max_iterations=0")
    d = run / "rounds" / "000" / "results"
    for b in ("direct", "cot", "cod"):
        assert read_json(d / "gsm8k" / f"{b}.summary.json")["n"] == 2
        assert read_json(d / "bbh" / f"{b}.summary.json")["n"] == 4


# --- AC23 ---------------------------------------------------------------------

def test_default_config_my_optimizer_not_implemented(tmp_path):
    with pytest.raises(NotImplementedError):
        _run(tmp_path)


# --- AC24 ---------------------------------------------------------------------

def test_config_with_optimize_section_rejected(tmp_path, no_llm_built):
    cfg = _mock_cfg(tmp_path, OPRO)
    cfg["optimize"] = {"generator": {"name": "opro"}, "stop": {"max_iterations": 2}}
    with pytest.raises(ValueError) as ei:
        run_loop(cfg)
    msg = str(ei.value)
    assert "optimizer" in msg and "max_iterations" in msg
    assert no_llm_built == []


def test_config_file_with_optimize_section_rejected(tmp_path, no_llm_built):
    raw = _raw_config()
    raw["optimize"] = {"generator": {"name": "opro"}}
    path = _write_cfg(tmp_path, raw)
    with pytest.raises(ValueError):
        main(["--config", str(path), "--mock", "--set", f"run.root={(tmp_path / 'runs').as_posix()}",
              "--set", OPRO])
    assert no_llm_built == []


def test_missing_optimizer_rejected(tmp_path, no_llm_built):
    cfg = _mock_cfg(tmp_path)
    cfg.pop("optimizer", None)
    with pytest.raises(ValueError):
        run_loop(cfg)
    assert no_llm_built == []


@pytest.mark.parametrize("bad", [{}, {"num_candidates": 2}, "opro", None, ["opro"]],
                         ids=["empty", "no_name", "str", "none", "list"])
def test_optimizer_without_name_rejected(tmp_path, no_llm_built, bad):
    cfg = _mock_cfg(tmp_path)
    cfg["optimizer"] = bad
    with pytest.raises(ValueError):
        run_loop(cfg)
    assert no_llm_built == []


def test_missing_max_iterations_rejected(tmp_path, no_llm_built):
    cfg = _mock_cfg(tmp_path, OPRO)
    cfg.pop("max_iterations", None)
    with pytest.raises(ValueError):
        run_loop(cfg)
    assert no_llm_built == []


@pytest.mark.parametrize("bad", [-1, True, "two", None, 1.5])
def test_invalid_max_iterations_rejected(tmp_path, no_llm_built, bad):
    cfg = _mock_cfg(tmp_path, OPRO)
    cfg["max_iterations"] = bad
    with pytest.raises(ValueError):
        run_loop(cfg)
    assert no_llm_built == []


def test_max_iterations_negative_via_cli(tmp_path, no_llm_built):
    with pytest.raises(ValueError):
        _run(tmp_path, OPRO, "max_iterations=-1")
    assert no_llm_built == []


def test_unknown_optimizer_name_key_error(tmp_path):
    with pytest.raises(KeyError):
        _run(tmp_path, f"optimizer.name={uniq('nope')}")


def _write_cfg(tmp_path, raw):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir(exist_ok=True)
    if not (cfg_dir / "models").exists():
        shutil.copytree(CONFIGS / "models", cfg_dir / "models")
    path = cfg_dir / "config.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(raw, f)
    return path


def test_singular_benchmark_key_rejected(tmp_path, no_llm_built):
    raw = _raw_config()
    raw["benchmark"] = dict(raw.pop("benchmarks")["gsm8k"], name="gsm8k")
    path = _write_cfg(tmp_path, raw)
    with pytest.raises(ValueError):
        main(["--config", str(path), "--mock", "--set", f"run.root={(tmp_path / 'runs').as_posix()}",
              "--set", OPRO, "--set", "llm.target.preset=mock", "--set", "llm.optimizer.preset=mock"])
    assert no_llm_built == []


def test_vram_guard(tmp_path):
    with pytest.raises(VRAMBudgetError):
        _run(
            tmp_path, OPRO,
            "llm.target.backend=vllm", "llm.target.model=a",
            "llm.target.engine.gpu_memory_utilization=0.85",
            "llm.optimizer.backend=vllm", "llm.optimizer.model=b",
            "llm.optimizer.engine.gpu_memory_utilization=0.85",
        )


# --- bbh heldout ----------------------------------------------------------------

def test_bbh_no_heldout_leakage(opro_run):
    _, run = opro_run
    jsonls = list((run / "rounds").glob("*/results/*/*.jsonl"))
    assert jsonls
    for jl in jsonls:
        for row in read_jsonl(jl):
            assert not str(row["example_id"]).startswith("web_of_lies"), (jl, row["example_id"])


def test_bbh_heldout_split_only_web_of_lies():
    cfg = load_config(CONFIG, overrides=mock_overrides(_raw_config()))
    specs = build_benchmark_specs(cfg["benchmarks"], seed=0)
    ev = Evaluator(make_mock(), specs)
    res = ev.evaluate_split(get_baseline("cot"), "bbh", "heldout")
    assert res.n > 0
    assert all(it.example_id.startswith("web_of_lies") for it in res.items)


# --- AC25: --eval-only unchanged ----------------------------------------------------

def _summary(run):
    return read_json(run / "summary.json")


def test_eval_only_default(tmp_path):
    run = main(_base(tmp_path) + ["--eval-only"])
    assert tmp_path in run.parents
    assert "mock_eval" in run.name
    summary = _summary(run)
    assert isinstance(summary, list)
    assert [s["skill_id"] for s in summary] == ["direct", "cot", "cod"]
    for s in summary:
        assert set(s) == {"skill_id", "split", "benchmarks"}
        assert s["split"] == "test"
        g, b = s["benchmarks"]["gsm8k"], s["benchmarks"]["bbh"]
        assert g["split"] == "test" and g["n"] == 6
        assert b["split"] == "test" and b["n"] == 8
        jl = run / "results" / "test" / "gsm8k" / f"{s['skill_id']}.jsonl"
        assert jl.is_file()
        assert len(read_jsonl(jl)) == g["n"]
        assert (run / "skills" / f"{s['skill_id']}.json").is_file()
    assert not (run / "results" / "final").exists()
    assert (run / "usage" / "target.json").is_file()
    assert not (run / "usage" / "optimizer.json").exists()
    assert not (run / "selection.json").exists()
    assert not (run / "rounds").exists()
    assert not (run / "run.json").exists()


def test_eval_only_split_val(tmp_path):
    summary = _summary(main(_base(tmp_path) + ["--eval-only", "--split", "val"]))
    for s in summary:
        assert s["split"] == "val"
        assert s["benchmarks"]["gsm8k"]["split"] == "val"
        assert s["benchmarks"]["gsm8k"]["n"] == 2
        assert s["benchmarks"]["bbh"]["split"] == "val"
        assert s["benchmarks"]["bbh"]["n"] == 4


def test_eval_only_split_train(tmp_path):
    summary = _summary(main(_base(tmp_path) + ["--eval-only", "--split", "train"]))
    for s in summary:
        assert s["split"] == "train"
        assert all(r["split"] == "train" for r in s["benchmarks"].values())
        assert s["benchmarks"]["gsm8k"]["n"] == 4


def test_eval_only_split_test_explicit(tmp_path):
    summary = _summary(main(_base(tmp_path) + ["--eval-only", "--split", "test"]))
    for s in summary:
        assert s["split"] == "test"
        assert all(r["split"] == "test" for r in s["benchmarks"].values())


def test_eval_only_split_heldout(tmp_path):
    run = main(_base(tmp_path) + ["--eval-only", "--split", "heldout"])
    summary = _summary(run)
    assert [s["skill_id"] for s in summary] == ["direct", "cot", "cod"]
    for s in summary:
        assert s["split"] == "heldout"
        assert list(s["benchmarks"]) == ["bbh"]
        b = s["benchmarks"]["bbh"]
        assert b["split"] == "heldout"
        assert b["n"] == 8
        jl = run / "results" / "heldout" / "bbh" / f"{s['skill_id']}.jsonl"
        assert jl.is_file()
        assert all(str(r["example_id"]).startswith("web_of_lies") for r in read_jsonl(jl))
    assert not (run / "results" / "heldout" / "gsm8k").exists()


def test_eval_only_split_unknown_rejected(tmp_path):
    with pytest.raises(ValueError):
        main(_base(tmp_path) + ["--eval-only", "--split", "nope"])


def test_eval_only_limit(tmp_path):
    summary = _summary(main(_base(tmp_path) + ["--eval-only", "--limit", "1"]))
    assert len(summary) == 3
    for s in summary:
        assert set(s["benchmarks"]) == set(BENCH_KEYS)
        assert all(r["n"] == 1 for r in s["benchmarks"].values())


def test_eval_only_with_saved_round_skill(tmp_path, opro_run):
    _, opt = opro_run
    saved = sorted((opt / "rounds" / "001" / "skills").glob("opro-*.json"))
    assert saved
    run = main(_base(tmp_path) + ["--eval-only", "--limit", "1", "--skills", "cot", str(saved[0])])
    assert [s["skill_id"] for s in _summary(run)] == ["cot", saved[0].stem]


def test_eval_only_unknown_skill_spec(tmp_path):
    with pytest.raises(ValueError):
        main(_base(tmp_path) + ["--eval-only", "--skills", "definitely_not_a_skill"])


def test_eval_only_without_optimizer_or_max_iterations(tmp_path):
    cfg = _mock_cfg(tmp_path)
    cfg.pop("optimizer", None)
    cfg.pop("max_iterations", None)
    cfg["eval"] = {"skills": ["cot"], "split": "val", "limit": 1}
    run = run_eval_only(cfg)
    summary = _summary(run)
    assert [s["skill_id"] for s in summary] == ["cot"]
    assert summary[0]["split"] == "val"
    assert all(r["n"] == 1 for r in summary[0]["benchmarks"].values())
    assert all(r["split"] == "val" for r in summary[0]["benchmarks"].values())


def test_eval_only_default_split_when_eval_split_missing(tmp_path):
    cfg = _mock_cfg(tmp_path)
    cfg["eval"] = {"skills": ["cot"], "limit": 1}
    summary = _summary(run_eval_only(cfg))
    assert summary[0]["split"] == "test"


def test_eval_only_old_stage_key_rejected(tmp_path, no_llm_built):
    cfg = _mock_cfg(tmp_path)
    cfg["eval"] = {"stage": "val"}
    with pytest.raises(ValueError, match=r"eval\.split"):
        run_eval_only(cfg)
    assert no_llm_built == []


def test_eval_only_old_stage_key_via_cli_rejected(tmp_path, no_llm_built):
    with pytest.raises(ValueError, match=r"eval\.split"):
        main(_base(tmp_path) + ["--set", "eval.stage=final", "--eval-only"])
    assert no_llm_built == []


def test_eval_only_config_file_without_optimizer(tmp_path):
    raw = _raw_config()
    raw.pop("optimizer", None)
    raw.pop("max_iterations", None)
    raw.pop("optimize", None)
    path = _write_cfg(tmp_path, raw)
    run = main(["--config", str(path), "--mock", "--set", f"run.root={(tmp_path / 'runs').as_posix()}",
                "--eval-only", "--limit", "1"])
    assert len(_summary(run)) == 3


@pytest.mark.parametrize("extra", [["--split", "val"], ["--skills", "cot"], ["--limit", "1"]])
def test_eval_flags_require_eval_only(tmp_path, extra):
    with pytest.raises(SystemExit) as ei:
        main(_base(tmp_path) + extra)
    assert ei.value.code == 2


def test_stage_flag_removed(tmp_path):
    with pytest.raises(SystemExit) as ei:
        main(_base(tmp_path) + ["--eval-only", "--stage", "test"])
    assert ei.value.code == 2


def test_parser_has_split_not_stage():
    opts = {o for a in build_parser()._actions for o in a.option_strings}
    assert "--split" in opts
    assert "--stage" not in opts


# --- old benchmark keys / list form rejected before any LLM is built -----------------

@pytest.mark.parametrize("eval_only", [False, True], ids=["loop", "eval_only"])
@pytest.mark.parametrize("override", ["benchmarks.gsm8k.dev_size=4", "benchmarks.gsm8k.val_size=2",
                                      "benchmarks.bbh.heldout_tasks=[web_of_lies]"],
                         ids=["dev_size", "val_size", "heldout_tasks"])
def test_old_benchmark_keys_via_cli_rejected(tmp_path, no_llm_built, eval_only, override):
    extra = ["--eval-only"] if eval_only else []
    with pytest.raises(ValueError):
        _run(tmp_path, OPRO, override, extra=extra)
    assert no_llm_built == []


def _old_list_form_raw():
    raw = _raw_config()
    raw["benchmarks"] = [
        {"name": "gsm8k", "source": "hf", "dev_size": 200, "val_size": 200},
        {"name": "bbh", "source": "hf", "tasks": ["algorithmic"], "heldout_tasks": ["web_of_lies"]},
    ]
    return raw


@pytest.mark.parametrize("eval_only", [False, True], ids=["loop", "eval_only"])
def test_old_list_form_config_file_rejected(tmp_path, no_llm_built, eval_only):
    path = _write_cfg(tmp_path, _old_list_form_raw())
    argv = ["--config", str(path), "--mock", "--set", f"run.root={(tmp_path / 'runs').as_posix()}",
            "--set", OPRO]
    if eval_only:
        argv.append("--eval-only")
    with pytest.raises(ValueError):
        main(argv)
    assert no_llm_built == []


# --- AC26: mock_overrides ----------------------------------------------------------

def test_mock_overrides_on_real_config():
    raw = _raw_config()
    before = copy.deepcopy(raw)
    ovs = mock_overrides(raw)
    assert raw == before
    assert isinstance(ovs, list) and ovs
    assert all(isinstance(o, str) and "=" in o for o in ovs)
    assert "max_iterations=2" in ovs
    assert not [o for o in ovs if o.startswith("optimize.")]
    assert not [o for o in ovs if o.startswith("optimizer.name")]
    assert not [o for o in ovs if o.startswith("optimizer=")]
    assert "llm.target.preset=mock" in ovs
    assert "llm.optimizer.preset=mock" in ovs
    for o in ("benchmarks.gsm8k.source=local", "benchmarks.gsm8k.train=4", "benchmarks.gsm8k.val=2",
              "benchmarks.bbh.source=local", "benchmarks.bbh.tasks=all", "benchmarks.bbh.train=2",
              "benchmarks.bbh.val=2"):
        assert o in ovs, o
    assert not [o for o in ovs if "dev_size" in o or "val_size" in o]
    assert not [o for o in ovs if o.startswith("benchmarks.0") or o.startswith("benchmarks.1")]


def test_mock_overrides_alias_entry():
    ovs = mock_overrides({"benchmarks": {"g2": {"name": "gsm8k"}}})
    assert "benchmarks.g2.source=local" in ovs


def test_mock_overrides_alias_entry_does_not_mutate():
    raw = {"benchmarks": {"g2": {"name": "gsm8k"}, "gsm8k": None}}
    before = copy.deepcopy(raw)
    mock_overrides(raw)
    assert raw == before


def test_mock_overrides_list_form_emits_no_benchmark_overrides():
    ovs = mock_overrides(_old_list_form_raw())
    assert not [o for o in ovs if o.startswith("benchmarks.")]


def test_mock_overrides_applied_config():
    cfg = load_config(CONFIG, overrides=mock_overrides(_raw_config()))
    assert cfg["llm"]["target"]["backend"] == "mock"
    assert cfg["llm"]["optimizer"]["backend"] == "mock"
    assert cfg["max_iterations"] == 2
    assert cfg["optimizer"]["name"] == "my_optimizer"
    assert "optimize" not in cfg
    specs = build_benchmark_specs(cfg["benchmarks"], seed=0)
    assert [s.key for s in specs] == ["gsm8k", "bbh"]
    assert cfg["benchmarks"]["gsm8k"]["train"] == 4
    assert cfg["benchmarks"]["gsm8k"]["val"] == 2
    assert cfg["benchmarks"]["bbh"]["train"] == 2
    assert cfg["benchmarks"]["bbh"]["val"] == 2
    assert cfg["benchmarks"]["bbh"]["heldout"] == ["web_of_lies"]


def test_mock_overrides_unknown_benchmark_only():
    raw = _raw_config()
    raw["benchmarks"] = {"not_a_known_benchmark_xyz": {}}
    ovs = mock_overrides(raw)
    assert not [o for o in ovs if o.startswith("benchmarks.")]


def test_mock_overrides_without_benchmarks():
    raw = _raw_config()
    raw.pop("benchmarks", None)
    ovs = mock_overrides(raw)
    assert isinstance(ovs, list)


# --- loop 2026-11-11-003: truncation fields in saved outputs ----------------------

ITEM_ROW_KEYS = {"example_id", "prediction", "gold", "correct", "usage", "latency",
                 "response_text", "reasoning_text", "question", "truncated"}


def test_saved_outputs_include_truncation_fields(opro_run):
    _, run = opro_run
    checked = 0
    for rnd in _rounds(run):
        for key in BENCH_KEYS:
            for jl in (run / "rounds" / rnd / "results" / key).glob("*.jsonl"):
                rows = read_jsonl(jl)
                for row in rows:
                    assert ITEM_ROW_KEYS <= set(row), (jl, set(row))
                    assert isinstance(row["truncated"], bool)
                sm = read_json(jl.with_name(jl.name[: -len(".jsonl")] + ".summary.json"))
                assert EVAL_SUMMARY_KEYS <= set(sm)
                assert sm["num_truncated"] == sum(1 for r in rows if r["truncated"])
                expected_rate = sm["num_truncated"] / sm["n"] if sm["n"] else 0.0
                assert sm["truncation_rate"] == pytest.approx(expected_rate)
                checked += 1
    assert checked
