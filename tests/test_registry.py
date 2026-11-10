"""C. Registries (+ lazy heavy imports), Optimizer interface (AC6-10)."""
import importlib
import os
import subprocess
import sys

import pytest

from helpers import PROJECT_ROOT, make_mock, uniq
from skillopt.benchmarks import Benchmark, get_benchmark_class, register_benchmark
from skillopt.llm import BaseLLM, RawGeneration, get_llm_class, register_llm
from skillopt.optimizer import (
    OPTIMIZER_REGISTRY,
    MyOptimizer,
    OPROOptimizer,
    Optimizer,
    build_optimizer,
    discover_optimizers,
    register_optimizer,
)
from skillopt.registry import Registry


# --- Registry class --------------------------------------------------------

def test_registry_register_get_names():
    reg = Registry("thing")

    @reg.register("b")
    class B:
        pass

    @reg.register("a")
    def a():
        return 1

    assert reg.get("b") is B
    assert reg.get("a") is a
    assert reg.names() == ["a", "b"]


def test_registry_duplicate_raises_value_error():
    reg = Registry("thing")
    reg.register("x")(object)
    with pytest.raises(ValueError):
        reg.register("x")(int)


def test_registry_unknown_raises_key_error_listing_names():
    reg = Registry("thing")
    reg.register("alpha_name")(object)
    with pytest.raises(KeyError) as ei:
        reg.get("missing")
    assert "alpha_name" in str(ei.value)


# --- AC11: llm -------------------------------------------------------------

class _DummyLLM(BaseLLM):
    uses_gpu = False

    def _generate(self, prompt, gen):
        return RawGeneration(text="ok")

    def count_tokens(self, text):
        return len(text.split())


def test_register_llm_roundtrip_duplicate_unknown():
    name = uniq("llm")
    register_llm(name)(_DummyLLM)
    assert get_llm_class(name) is _DummyLLM
    with pytest.raises(ValueError):
        register_llm(name)(_DummyLLM)
    with pytest.raises(KeyError):
        get_llm_class(uniq("nope"))


# --- AC11: benchmark -------------------------------------------------------

class _DummyBench(Benchmark):
    name = "dummy"
    answer_format = "Answer."

    def load(self, split):
        return []

    def extract_answer(self, text):
        return text or None

    def score(self, pred, gold):
        return pred == gold


def test_register_benchmark_roundtrip_duplicate_unknown():
    name = uniq("bench")
    register_benchmark(name)(_DummyBench)
    assert get_benchmark_class(name) is _DummyBench
    with pytest.raises(ValueError):
        register_benchmark(name)(_DummyBench)
    with pytest.raises(KeyError):
        get_benchmark_class(uniq("nope"))


# --- AC12 ------------------------------------------------------------------

@pytest.mark.parametrize("name", ["mock", "vllm", "hf", "openai"])
def test_builtin_llms_registered(name):
    cls = get_llm_class(name)
    assert issubclass(cls, BaseLLM)


@pytest.mark.parametrize("name", ["gsm8k", "bbh"])
def test_builtin_benchmarks_registered(name):
    assert issubclass(get_benchmark_class(name), Benchmark)


def test_optimizer_package_exports_exactly():
    import skillopt.optimizer as opt

    expected = {"Optimizer", "OPTIMIZER_REGISTRY", "register_optimizer", "build_optimizer",
                "discover_optimizers", "optimize", "LoopResult", "TRAIN_SPLIT", "MyOptimizer",
                "OPROOptimizer"}
    for name in expected:
        assert hasattr(opt, name), name
    assert set(opt.__all__) == expected


def test_evaluator_module_exports():
    import skillopt.evaluator as ev

    for name in ("ItemResult", "EvalResult", "BenchmarkSpec",
                 "build_benchmark_specs", "SkillEvaluation", "Evaluator"):
        assert hasattr(ev, name), name
    for name in ("SplitSpec", "DEFAULT_SPLITS"):
        assert not hasattr(ev, name), name


# --- AC13 ------------------------------------------------------------------

def test_import_does_not_pull_heavy_dependencies():
    code = (
        "import sys\n"
        "heavy = ('vllm', 'torch', 'transformers', 'datasets')\n"
        "pre = {m for m in heavy if m in sys.modules}\n"
        "import skillopt.llm, skillopt.benchmarks, skillopt.optimizer, skillopt.cli\n"
        "bad = [m for m in heavy if m in sys.modules and m not in pre]\n"
        "print('HEAVY:', bad)\n"
        "sys.exit(1 if bad else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --- AC6: Optimizer is abstract ----------------------------------------------

class _OnlyInitial(Optimizer):
    def initial_prompts(self):
        return []


class _OnlyUpdate(Optimizer):
    def update(self, results, history):
        return []


class _Full(Optimizer):
    def __init__(self, llm, **params):
        super().__init__(llm, **params)
        self.got_params = params

    def initial_prompts(self):
        return []

    def update(self, results, history):
        return []


def test_optimizer_base_is_abstract():
    llm = make_mock(["x"], role="optimizer")
    with pytest.raises(TypeError):
        Optimizer(llm)
    with pytest.raises(TypeError):
        _OnlyInitial(llm)
    with pytest.raises(TypeError):
        _OnlyUpdate(llm)
    o = _Full(llm, a=1)
    assert o.llm is llm
    assert o.params == {"a": 1}


def test_optimizer_base_init_sets_llm_and_params():
    class Plain(Optimizer):
        def initial_prompts(self):
            return []

        def update(self, results, history):
            return []

    llm = make_mock(["x"], role="optimizer")
    o = Plain(llm, x=1, y="z")
    assert o.llm is llm
    assert o.params == {"x": 1, "y": "z"}
    assert isinstance(o.params, dict)


# --- AC7: register / build ----------------------------------------------------

def test_register_and_build_optimizer():
    name = uniq("opt")
    returned = register_optimizer(name)(_Full)
    assert returned is _Full
    assert name in OPTIMIZER_REGISTRY
    llm = make_mock(["x"], role="optimizer")
    o = build_optimizer({"name": name, "a": 1, "b": "two"}, llm)
    assert isinstance(o, _Full)
    assert o.llm is llm
    assert o.got_params == {"a": 1, "b": "two"}
    assert o.params == {"a": 1, "b": "two"}


def test_build_optimizer_does_not_mutate_config():
    name = uniq("opt")
    register_optimizer(name)(_Full)
    cfg = {"name": name, "a": 1}
    build_optimizer(cfg, make_mock(["x"], role="optimizer"))
    assert cfg == {"name": name, "a": 1}


def test_register_optimizer_duplicate_value_error():
    name = uniq("opt")
    register_optimizer(name)(_Full)
    with pytest.raises(ValueError):
        register_optimizer(name)(_Full)


def test_build_optimizer_unknown_key_error_lists_available():
    with pytest.raises(KeyError) as ei:
        build_optimizer({"name": uniq("nope")}, make_mock(["x"], role="optimizer"))
    msg = str(ei.value)
    assert "my_optimizer" in msg
    assert "opro" in msg


@pytest.mark.parametrize("cfg", [{}, {"num_candidates": 2}, "opro", None, ["opro"]],
                         ids=["empty", "no_name", "str", "none", "list"])
def test_build_optimizer_bad_config_value_error(cfg):
    with pytest.raises(ValueError):
        build_optimizer(cfg, make_mock(["x"], role="optimizer"))


# --- AC8: built-ins + discovery -------------------------------------------------

def test_builtin_optimizers_registered():
    import skillopt.optimizer  # noqa: F401

    assert "my_optimizer" in OPTIMIZER_REGISTRY
    assert "opro" in OPTIMIZER_REGISTRY
    llm = make_mock(["<skill>x</skill>"], role="optimizer")
    assert isinstance(build_optimizer({"name": "opro"}, llm), OPROOptimizer)
    assert isinstance(build_optimizer({"name": "my_optimizer"}, llm), MyOptimizer)
    assert issubclass(MyOptimizer, Optimizer)
    assert issubclass(OPROOptimizer, Optimizer)


def test_discover_default_package():
    found = discover_optimizers()
    assert found == sorted(found)
    assert "skillopt.optimizer.algorithms.my_optimizer" in found
    assert "skillopt.optimizer.algorithms.opro" in found
    assert all(not m.rsplit(".", 1)[-1].startswith("_") for m in found)
    assert discover_optimizers() == found  # idempotent: no duplicate registration


def test_discover_optimizers_one_file_plugin(tmp_path, monkeypatch):
    pkg = uniq("optpkg")
    opt_name = uniq("plugged")
    d = tmp_path / pkg
    d.mkdir()
    (d / "__init__.py").write_text("", encoding="utf-8")
    (d / "zeta.py").write_text("X = 1\n", encoding="utf-8")
    (d / "mine.py").write_text(
        "from skillopt.optimizer import Optimizer, register_optimizer\n\n"
        f"@register_optimizer({opt_name!r})\n"
        "class Mine(Optimizer):\n"
        "    def initial_prompts(self):\n"
        "        return []\n\n"
        "    def update(self, results, history):\n"
        "        return []\n",
        encoding="utf-8",
    )
    (d / "_private.py").write_text("raise RuntimeError('private module must not be imported')\n",
                                   encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        found = discover_optimizers(pkg)
        assert found == [f"{pkg}.mine", f"{pkg}.zeta"]
        assert f"{pkg}._private" not in sys.modules
        o = build_optimizer({"name": opt_name, "k": 2}, make_mock(["x"], role="optimizer"))
        assert isinstance(o, Optimizer)
        assert o.params == {"k": 2}
        assert discover_optimizers(pkg) == found
    finally:
        for mod in [m for m in sys.modules if m == pkg or m.startswith(pkg + ".")]:
            monkeypatch.delitem(sys.modules, mod, raising=False)


# --- AC9: the empty template -------------------------------------------------------

def test_my_optimizer_is_unimplemented_template():
    o = MyOptimizer(make_mock(["x"], role="optimizer"))
    with pytest.raises(NotImplementedError):
        o.initial_prompts()
    with pytest.raises(NotImplementedError):
        o.update([], [])


# --- AC10: old names are gone ---------------------------------------------------------

@pytest.mark.parametrize("name", ["SkillGenerator", "register_generator", "EvaluatedSkill", "GenerationContext",
                                  "get_aggregator", "get_selector", "build_stop", "get_final_policy",
                                  "OPROGenerator"])
def test_old_optimizer_names_removed(name):
    import skillopt.optimizer as opt

    assert not hasattr(opt, name), name


@pytest.mark.parametrize("mod", ["generation", "scoring", "selection", "stopping", "final", "generators"])
def test_old_optimizer_modules_not_importable(mod):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"skillopt.optimizer.{mod}")
