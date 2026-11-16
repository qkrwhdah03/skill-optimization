"""Single-pool math benchmarks: MATH-500, AIME 2024/2025, AMC 2023 (loop 2026-11-16-003)."""
import json
import random
import sys
import types

import pytest

from skillopt.benchmarks import (
    AIME2024,
    AIME2025,
    AMC2023,
    MATH500,
    BBEH,
    Example,
    MathBenchmark,
    PooledBenchmark,
    build_benchmark,
    get_benchmark_class,
)
from skillopt.benchmarks.math_answer import MATH_ANSWER_FORMAT, math_answers_equal

MATH_CLASSES = {"math500": MATH500, "aime2024": AIME2024, "aime2025": AIME2025, "amc2023": AMC2023}
QUESTION_FIELD = {"math500": "problem", "aime2024": "problem", "aime2025": "problem", "amc2023": "question"}


def _write_pool(path, name, n, answers=None):
    """Write <path>/data.jsonl with n rows (HF field names); returns the rows."""
    rows = []
    for i in range(n):
        ans = answers[i] if answers is not None else str(i * 7)
        rows.append({QUESTION_FIELD[name]: f"Synthetic {name} problem number {i}?", "answer": ans})
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "data.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def _expected_split(n, train, val, seed):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return (sorted(order[:train]), sorted(order[train:train + val]), sorted(order[train + val:]))


def _idx(examples, name):
    out = []
    for e in examples:
        assert e.id.startswith(f"{name}-"), e.id
        out.append(int(e.id[len(name) + 1:]))
    return out


class _FakeDatasets:
    def __init__(self, rows=None, exc=None):
        self.calls = []
        self.rows = rows or []
        self.exc = exc

    def module(self):
        mod = types.ModuleType("datasets")

        def load_dataset(*args, **kwargs):
            self.calls.append((args, kwargs))
            if self.exc is not None:
                raise self.exc
            return list(self.rows)

        mod.load_dataset = load_dataset
        return mod


@pytest.fixture
def fake_datasets(monkeypatch):
    def install(rows=None, exc=None):
        fake = _FakeDatasets(rows, exc)
        monkeypatch.setitem(sys.modules, "datasets", fake.module())
        return fake
    return install


# --- crit 7: registration ------------------------------------------------------------

@pytest.mark.parametrize("name,cls", [("math500", MATH500), ("aime2024", AIME2024), ("aime2025", AIME2025),
                                      ("amc2023", AMC2023), ("bbeh", BBEH)])
def test_registered(name, cls):
    assert get_benchmark_class(name) is cls


@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_math_classes_hierarchy(name):
    cls = MATH_CLASSES[name]
    assert issubclass(cls, MathBenchmark)
    assert issubclass(cls, PooledBenchmark)


@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_build_benchmark_math(tmp_path, name):
    _write_pool(tmp_path, name, 5)
    bench = build_benchmark({"name": name, "source": "local", "data_dir": str(tmp_path), "train": 1, "val": 1})
    assert isinstance(bench, MATH_CLASSES[name])
    assert len(bench.load("test")) == 3


# --- crit 8: splits per instance --------------------------------------------------------

@pytest.mark.parametrize("name", list(MATH_CLASSES))
@pytest.mark.parametrize("train,val,expected", [
    (0, 0, ("test",)),
    (2, 0, ("train", "test")),
    (2, 1, ("train", "val", "test")),
    (0, 2, ("val", "test")),
])
def test_splits_per_instance(tmp_path, name, train, val, expected):
    _write_pool(tmp_path, name, 6)
    bench = MATH_CLASSES[name](source="local", data_dir=str(tmp_path), train=train, val=val)
    assert tuple(bench.splits) == expected


@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_default_constructor_is_test_only(tmp_path, name):
    _write_pool(tmp_path, name, 3)
    bench = MATH_CLASSES[name](data_dir=str(tmp_path))
    assert tuple(bench.splits) == ("test",)
    assert len(bench.load("test")) == 3


# --- crit 9: split rule ---------------------------------------------------------------

@pytest.mark.parametrize("name", list(MATH_CLASSES))
@pytest.mark.parametrize("n,train,val,seed", [(10, 3, 2, 0), (12, 4, 4, 7), (9, 0, 3, 5), (7, 2, 0, 11)])
def test_split_rule(tmp_path, name, n, train, val, seed):
    _write_pool(tmp_path, name, n)
    bench = MATH_CLASSES[name](source="local", data_dir=str(tmp_path), train=train, val=val, seed=seed)
    exp_train, exp_val, exp_test = _expected_split(n, train, val, seed)
    got_train = _idx(bench.load("train"), name)
    got_val = _idx(bench.load("val"), name)
    got_test = _idx(bench.load("test"), name)
    assert got_train == exp_train
    assert got_val == exp_val
    assert got_test == exp_test
    assert not (set(got_train) & set(got_val)) and not (set(got_train) & set(got_test))
    assert not (set(got_val) & set(got_test))
    assert sorted(got_train + got_val + got_test) == list(range(n))


@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_split_overflow_no_error(tmp_path, name):
    _write_pool(tmp_path, name, 4)
    bench = MATH_CLASSES[name](source="local", data_dir=str(tmp_path), train=3, val=3, seed=1)
    exp_train, exp_val, exp_test = _expected_split(4, 3, 3, 1)
    assert _idx(bench.load("train"), name) == exp_train
    assert _idx(bench.load("val"), name) == exp_val
    assert bench.load("test") == [] and exp_test == []


@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_split_same_seed_identical(tmp_path, name):
    _write_pool(tmp_path, name, 10)
    a = MATH_CLASSES[name](source="local", data_dir=str(tmp_path), train=3, val=3, seed=4)
    b = MATH_CLASSES[name](source="local", data_dir=str(tmp_path), train=3, val=3, seed=4)
    for split in ("train", "val", "test"):
        assert [e.id for e in a.load(split)] == [e.id for e in b.load(split)]


@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_ids_and_examples(tmp_path, name):
    rows = _write_pool(tmp_path, name, 3)
    bench = MATH_CLASSES[name](source="local", data_dir=str(tmp_path))
    test = bench.load("test")
    assert [e.id for e in test] == [f"{name}-{i}" for i in range(3)]
    assert all(isinstance(e, Example) for e in test)
    for e, r in zip(test, rows):
        assert e.question.strip() == r[QUESTION_FIELD[name]]
        assert e.answer == r["answer"]


# --- crit 10: errors -------------------------------------------------------------------

@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_load_train_with_zero_train_is_empty(tmp_path, name):
    _write_pool(tmp_path, name, 4)
    bench = MATH_CLASSES[name](source="local", data_dir=str(tmp_path))
    assert bench.load("train") == []


@pytest.mark.parametrize("name", list(MATH_CLASSES))
@pytest.mark.parametrize("split", ["heldout", "nope"])
def test_load_unknown_split_raises(tmp_path, name, split):
    _write_pool(tmp_path, name, 4)
    bench = MATH_CLASSES[name](source="local", data_dir=str(tmp_path), train=1, val=1)
    with pytest.raises(ValueError):
        bench.load(split)


@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_unknown_source_raises(tmp_path, name):
    with pytest.raises(ValueError):
        MATH_CLASSES[name](source="s3", data_dir=str(tmp_path))


@pytest.mark.parametrize("name", list(MATH_CLASSES))
@pytest.mark.parametrize("kw", [{"train": -1}, {"val": -1}])
def test_negative_sizes_raise(tmp_path, name, kw):
    _write_pool(tmp_path, name, 4)
    with pytest.raises(ValueError):
        MATH_CLASSES[name](source="local", data_dir=str(tmp_path), **kw)


# --- crit 11: local gold ---------------------------------------------------------------

@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_gold_from_json_int_and_leading_zero(tmp_path, name):
    tmp_path.mkdir(parents=True, exist_ok=True)
    q = QUESTION_FIELD[name]
    with open(tmp_path / "data.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({q: "P0", "answer": 70}) + "\n")
        f.write(json.dumps({q: "P1", "answer": "025"}) + "\n")
        f.write(json.dumps({q: "P2", "answer": "  \\frac{1}{2} "}) + "\n")
    bench = MATH_CLASSES[name](source="local", data_dir=str(tmp_path))
    test = bench.load("test")
    assert [e.answer for e in test] == ["70", "025", "\\frac{1}{2}"]
    assert bench.score("25", test[1].answer) is True
    assert bench.score("70", test[0].answer) is True


def test_amc_reads_question_field(tmp_path):
    with open(tmp_path / "data.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"question": "AMC question text", "answer": 3.0}) + "\n")
    e = AMC2023(source="local", data_dir=str(tmp_path)).load("test")[0]
    assert e.question.strip() == "AMC question text"
    assert e.answer == "3.0"


@pytest.mark.parametrize("name", ["math500", "aime2024", "aime2025"])
def test_math_aime_read_problem_field(tmp_path, name):
    with open(tmp_path / "data.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"problem": "Problem field text", "answer": "1"}) + "\n")
    e = MATH_CLASSES[name](source="local", data_dir=str(tmp_path)).load("test")[0]
    assert e.question.strip() == "Problem field text"


# --- crit 12: format / extraction / score -----------------------------------------

@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_math_format_extract_score(tmp_path, name):
    _write_pool(tmp_path, name, 2)
    bench = MATH_CLASSES[name](source="local", data_dir=str(tmp_path))
    ex = bench.load("test")[0]
    assert bench.answer_format_for(ex) == MATH_ANSWER_FORMAT
    assert bench.extract_answer_for("so the result is \\boxed{204}", ex) == "204"
    for pred, gold in [("25", "025"), ("0.5", "\\frac{1}{2}"), ("5", "6"), (None, "5"), ("14/3", "\\dfrac{14}{3}")]:
        assert bench.score(pred, gold) == math_answers_equal(pred, gold)


# --- crit 13: HF source ---------------------------------------------------------------

HF_CALLS = [
    ("math500", ("HuggingFaceH4/MATH-500",), {"split": "test"}),
    ("aime2024", ("HuggingFaceH4/aime_2024",), {"split": "train"}),
    ("aime2025", ("math-ai/aime25",), {"split": "test"}),
    ("amc2023", ("math-ai/amc23",), {"split": "test"}),
]


@pytest.mark.parametrize("name,args,kwargs", HF_CALLS)
def test_hf_load_call(fake_datasets, name, args, kwargs):
    rows = [{QUESTION_FIELD[name]: f"hf problem {i}", "answer": str(i)} for i in range(3)]
    fake = fake_datasets(rows)
    bench = MATH_CLASSES[name](source="hf")
    assert fake.calls == []
    test = bench.load("test")
    assert len(fake.calls) == 1
    assert fake.calls[0] == (args, kwargs)
    assert [e.id for e in test] == [f"{name}-{i}" for i in range(3)]
    assert [e.answer for e in test] == ["0", "1", "2"]
    bench.load("test")
    bench.load("train")
    assert len(fake.calls) == 1


@pytest.mark.parametrize("name", list(MATH_CLASSES))
def test_hf_construction_does_not_import_datasets(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "datasets", None)  # any import would raise ImportError
    bench = MATH_CLASSES[name](source="hf", train=1, val=1)
    assert tuple(bench.splits) == ("train", "val", "test")
