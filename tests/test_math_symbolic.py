"""Optional symbolic equivalence (math-verify, time-limited) for math answers (loop 2026-11-16-005)."""
import os
import subprocess
import sys
import time
import tomllib

import pytest

from helpers import PROJECT_ROOT
from skillopt.benchmarks import MATH500
from skillopt.benchmarks import math_answer
from skillopt.benchmarks.math_answer import (
    SYMBOLIC_TIMEOUT_SECONDS,
    load_math_verify,
    math_answers_equal,
    symbolic_math_equal,
)
from test_math_answer import EQUAL_PAIRS, UNEQUAL_PAIRS


class FakeMV:
    """Stand-in for the math_verify module that records every call."""

    def __init__(self, result=True, parse_result=None, parse_exc=None, verify_exc=None, verify_sleep=0.0):
        self.result = result
        self.parse_result = parse_result  # callable(text) -> list, or None for [text]
        self.parse_exc = parse_exc
        self.verify_exc = verify_exc
        self.verify_sleep = verify_sleep
        self.parse_calls = []   # (text, kwargs)
        self.verify_calls = []  # (gold, pred, kwargs)

    def parse(self, text, **kwargs):
        self.parse_calls.append((text, kwargs))
        if self.parse_exc is not None:
            raise self.parse_exc
        if self.parse_result is not None:
            return self.parse_result(text)
        return [text]

    def verify(self, gold, pred, **kwargs):
        self.verify_calls.append((gold, pred, kwargs))
        if self.verify_sleep:
            time.sleep(self.verify_sleep)
        if self.verify_exc is not None:
            raise self.verify_exc
        return self.result

    @property
    def n_calls(self):
        return len(self.parse_calls) + len(self.verify_calls)


@pytest.fixture
def no_backend(monkeypatch):
    monkeypatch.setattr(math_answer, "load_math_verify", lambda: None)


@pytest.fixture
def fake_backend(monkeypatch):
    def install(**kwargs):
        fake = FakeMV(**kwargs)
        monkeypatch.setattr(math_answer, "load_math_verify", lambda: fake)
        return fake
    return install


# --- crit 1: no backend -> behaviour unchanged ------------------------------------

@pytest.mark.parametrize("a,b", EQUAL_PAIRS)
def test_no_backend_equal_pairs(no_backend, a, b):
    assert math_answers_equal(a, b) is True
    assert math_answers_equal(b, a) is True


@pytest.mark.parametrize("a,b", UNEQUAL_PAIRS)
def test_no_backend_unequal_pairs(no_backend, a, b):
    assert math_answers_equal(a, b) is False


def test_no_backend_commutative_polynomial_false(no_backend):
    assert math_answers_equal("x^2+1", "1+x^2") is False


# --- crit 2: symbolic_math_equal without backend ----------------------------------

def test_symbolic_no_backend_false(no_backend):
    assert symbolic_math_equal("1+x^2", "x^2+1") is False


# --- crit 3: backend decides rule-unequal pairs -----------------------------------

@pytest.mark.parametrize("verdict", [True, False])
def test_backend_decides_rule_unequal(fake_backend, verdict):
    fake = fake_backend(result=verdict)
    assert math_answers_equal("1+x^2", "x^2+1") is verdict
    assert len(fake.verify_calls) == 1


# --- crit 4: rule-equal pairs never touch the backend ---------------------------

@pytest.mark.parametrize("a,b", [("0.5", "\\frac{1}{2}"), ("25", "025")] + EQUAL_PAIRS)
def test_rule_equal_skips_backend(fake_backend, a, b):
    fake = fake_backend(result=False)
    assert math_answers_equal(a, b) is True
    assert math_answers_equal(b, a) is True
    assert fake.n_calls == 0


# --- crit 5: distinct exact rationals are final ---------------------------------

@pytest.mark.parametrize("a,b", [("5", "6"), ("\\frac{1}{3}", "0.33"), ("0.333333", "1/3"),
                                 ("1/2", "\\frac{1}{3}"), ("2", "2.5")])
def test_distinct_rationals_skip_backend(fake_backend, a, b):
    fake = fake_backend(result=True)
    assert math_answers_equal(a, b) is False
    assert math_answers_equal(b, a) is False
    assert fake.n_calls == 0


# --- crit 6: empty / None answers ---------------------------------------------------

@pytest.mark.parametrize("a,b", [(None, "5"), ("5", None), ("", "0"), ("  ", "x")])
def test_empty_answers_skip_backend(fake_backend, a, b):
    fake = fake_backend(result=True)
    assert math_answers_equal(a, b) is False
    assert symbolic_math_equal(a, b) is False
    assert fake.n_calls == 0


# --- crit 7: argument order (gold first) ------------------------------------------

def test_verify_gold_first(fake_backend):
    fake = fake_backend(result=True)
    assert math_answers_equal(pred="1+x^2", gold="x^2+1") is True
    assert len(fake.verify_calls) == 1
    gold_arg, pred_arg, _ = fake.verify_calls[0]
    assert gold_arg == ["\\boxed{x^2+1}"]
    assert pred_arg == ["\\boxed{1+x^2}"]


def test_symbolic_verify_gold_first(fake_backend):
    fake = fake_backend(result=True)
    assert symbolic_math_equal("1+x^2", "x^2+1") is True
    gold_arg, pred_arg, _ = fake.verify_calls[0]
    assert gold_arg == ["\\boxed{x^2+1}"]
    assert pred_arg == ["\\boxed{1+x^2}"]


# --- crit 8: parse input is the boxed raw answer --------------------------------

def test_parse_receives_boxed_raw(fake_backend):
    fake = fake_backend(result=False)
    math_answers_equal("$\\sqrt{3}/2$", " 2\\pi r ")
    texts = sorted(t for t, _ in fake.parse_calls)
    assert texts == sorted(["\\boxed{\\sqrt{3}/2}", "\\boxed{2\\pi r}"])


def test_parse_receives_boxed_raw_symbolic(fake_backend):
    fake = fake_backend(result=False)
    symbolic_math_equal(" $x + y$ ", "$$ y+x $$")
    texts = sorted(t for t, _ in fake.parse_calls)
    assert texts == sorted(["\\boxed{x + y}", "\\boxed{y+x}"])


# --- crit 9: built-in timeouts disabled -----------------------------------------

def test_backend_timeouts_disabled(fake_backend):
    fake = fake_backend(result=True)
    assert symbolic_math_equal("1+x^2", "x^2+1") is True
    assert len(fake.parse_calls) == 2
    for _, kwargs in fake.parse_calls:
        assert "parsing_timeout" in kwargs and kwargs["parsing_timeout"] is None
    assert len(fake.verify_calls) == 1
    kwargs = fake.verify_calls[0][2]
    assert "timeout_seconds" in kwargs and kwargs["timeout_seconds"] is None


# --- crit 10: empty parse result ------------------------------------------------

@pytest.mark.parametrize("empty_side", ["pred", "gold"])
def test_empty_parse_no_verify(fake_backend, empty_side):
    empty_text = "\\boxed{1+x^2}" if empty_side == "pred" else "\\boxed{x^2+1}"
    fake = fake_backend(result=True, parse_result=lambda t: [] if t == empty_text else [t])
    assert symbolic_math_equal("1+x^2", "x^2+1") is False
    assert math_answers_equal("1+x^2", "x^2+1") is False
    assert fake.verify_calls == []


# --- crit 11: backend exceptions swallowed --------------------------------------

@pytest.mark.parametrize("where", ["parse", "verify"])
@pytest.mark.parametrize("exc", [RuntimeError("boom"), ValueError("bad")], ids=["runtime", "value"])
def test_backend_exception_false(fake_backend, where, exc):
    if where == "parse":
        fake_backend(result=True, parse_exc=exc)
    else:
        fake_backend(result=True, verify_exc=exc)
    assert symbolic_math_equal("1+x^2", "x^2+1") is False
    assert math_answers_equal("1+x^2", "x^2+1") is False


# --- crit 12: wall-clock timeout ------------------------------------------------

def test_verify_timeout_returns_false(fake_backend):
    fake_backend(result=True, verify_sleep=2.0)
    start = time.monotonic()
    assert symbolic_math_equal("a", "b", timeout=0.2) is False
    assert time.monotonic() - start < 1.5


# --- crit 13: timeout validation / default --------------------------------------

@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5])
def test_nonpositive_timeout_raises(fake_backend, bad):
    fake_backend(result=True)
    with pytest.raises(ValueError):
        symbolic_math_equal("a", "b", timeout=bad)


def test_default_timeout_constant():
    assert SYMBOLIC_TIMEOUT_SECONDS == 5.0


def test_timeout_none_uses_default(fake_backend):
    fake_backend(result=True)
    assert symbolic_math_equal("1+x^2", "x^2+1", timeout=None) is True


# --- crit 14: tuple pairs checked once as a whole ------------------------------

def test_tuple_pair_single_symbolic_call(fake_backend):
    fake = fake_backend(result=True)
    assert math_answers_equal("(1+x, 2)", "(x+1, 2)") is True
    assert len(fake.verify_calls) <= 1


# --- crit 15: load_math_verify -------------------------------------------------

@pytest.fixture
def clean_loader_cache():
    load_math_verify.cache_clear()
    yield
    load_math_verify.cache_clear()


def test_load_math_verify_missing_returns_none(monkeypatch, clean_loader_cache):
    assert hasattr(load_math_verify, "cache_clear")
    monkeypatch.setitem(sys.modules, "math_verify", None)
    load_math_verify.cache_clear()
    assert load_math_verify() is None
    assert load_math_verify() is None


def test_load_math_verify_cached(monkeypatch, clean_loader_cache):
    monkeypatch.setitem(sys.modules, "math_verify", None)
    load_math_verify.cache_clear()
    first = load_math_verify()
    assert first is None
    # Cached: even with a module now importable, the cached value is returned until cleared.
    import types
    monkeypatch.setitem(sys.modules, "math_verify", types.ModuleType("math_verify"))
    assert load_math_verify() is first


# --- crit 16: import is lazy ------------------------------------------------------

def test_import_does_not_load_math_verify_or_sympy():
    code = (
        "import sys\n"
        "import skillopt.benchmarks\n"
        "import skillopt.benchmarks.math_answer\n"
        "bad = [m for m in ('math_verify', 'sympy') if m in sys.modules]\n"
        "print('LOADED', bad)\n"
        "raise SystemExit(1 if bad else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --- crit 17: MathBenchmark.score contract ------------------------------------

def test_math500_score_uses_symbolic(fake_backend):
    fake_backend(result=True)
    bench = MATH500(source="local", data_dir=str(PROJECT_ROOT / "data" / "sample" / "math500"))
    assert bench.load("test")
    assert bench.score("1+x^2", "x^2+1") is True
    assert bench.score("1+x^2", "x^2+1") == math_answers_equal("1+x^2", "x^2+1")
    assert bench.score("5", "6") is False


def test_math500_score_no_backend(no_backend):
    bench = MATH500(source="local", data_dir=str(PROJECT_ROOT / "data" / "sample" / "math500"))
    for pred, gold in [("1+x^2", "x^2+1"), ("25", "025"), ("5", "6"), (None, "5")]:
        assert bench.score(pred, gold) == math_answers_equal(pred, gold)
    assert bench.score("1+x^2", "x^2+1") is False


# --- crit 18: packaging / docs ------------------------------------------------

def test_pyproject_math_extra():
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    extras = data["project"]["optional-dependencies"]
    assert "math" in extras
    assert any(dep.replace("_", "-").lower().startswith("math-verify") for dep in extras["math"])


def test_readme_mentions_math_extra():
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install -e .[math]" in text
    assert "math-verify" in text


# --- crit 19: real math-verify (skipped when not installed) ---------------------

@pytest.fixture
def real_backend(monkeypatch):
    pytest.importorskip("math_verify")
    load_math_verify.cache_clear()
    yield
    load_math_verify.cache_clear()


@pytest.mark.parametrize("pred,gold", [
    ("1+x^2", "x^2+1"),
    ("\\frac{\\sqrt{3}}{2}", "\\sqrt{3}/2"),
    ("[1,2)", "[1, 2)"),
])
def test_real_math_verify_equal(real_backend, pred, gold):
    assert load_math_verify() is not None
    assert math_answers_equal(pred, gold) is True


def test_real_math_verify_unequal(real_backend):
    assert math_answers_equal("x^2", "x^3") is False
