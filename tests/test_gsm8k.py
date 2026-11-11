"""G. GSM8K benchmark (local sample data)."""
import random

import pytest

from helpers import GSM8K_DIR, count_lines
from skillopt.baselines import get_baseline
from skillopt.benchmarks import GSM8K, Example, build_benchmark


def _bench(seed=0, train=4, val=2, data_dir="data/sample/gsm8k"):
    return GSM8K(source="local", data_dir=data_dir, train=train, val=val, seed=seed)


@pytest.fixture(scope="module")
def bench():
    return _bench()


# --- AC26 ------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Some reasoning here. The answer is 1,234.", "1234"),
    ("work work\n#### 42", "42"),
    ("so we get \\boxed{7}", "7"),
    ("I got 3 then 5", "5"),
    ("The answer is 12. Because 13 apples", "12"),
    ("$18", "18"),
    ("-3.50", "-3.5"),
    ("no numbers", None),
    ("", None),
])
def test_gsm8k_extract_answer(bench, text, expected):
    assert bench.extract_answer(text) == expected


@pytest.mark.parametrize("pred,gold,ok", [
    ("42", "42.0", True),
    (None, "42", False),
    ("41", "42", False),
    ("1,234", "1234", True),
])
def test_gsm8k_score(bench, pred, gold, ok):
    assert bench.score(pred, gold) is ok


# --- AC27 ------------------------------------------------------------------

def test_gsm8k_train_val_sizes_and_disjoint(bench):
    train = bench.load("train")
    val = bench.load("val")
    assert len(train) == 4
    assert len(val) == 2
    assert all(isinstance(e, Example) for e in train + val)
    assert not ({e.id for e in train} & {e.id for e in val})
    assert len({e.id for e in train}) == 4


def test_gsm8k_deterministic_same_seed():
    a, b = _bench(seed=0), _bench(seed=0)
    assert [e.id for e in a.load("train")] == [e.id for e in b.load("train")]
    assert [e.id for e in a.load("val")] == [e.id for e in b.load("val")]
    assert [e.answer for e in a.load("train")] == [e.answer for e in b.load("train")]


def test_gsm8k_other_seed_still_valid():
    c = _bench(seed=123)
    train, val = c.load("train"), c.load("val")
    assert len(train) == 4 and len(val) == 2
    assert not ({e.id for e in train} & {e.id for e in val})


def test_gsm8k_test_split_loads_test_file(bench):
    test = bench.load("test")
    assert len(test) == count_lines(GSM8K_DIR / "test.jsonl")
    assert len(test) >= 4


def test_gsm8k_gold_answers_normalized(bench):
    for split in ("train", "val", "test"):
        for e in bench.load(split):
            assert "####" not in e.answer
            assert "," not in e.answer
            assert e.answer.strip() == e.answer
            assert e.answer != ""
            assert e.question


def test_gsm8k_unknown_split(bench):
    with pytest.raises(ValueError):
        bench.load("bogus")


def test_gsm8k_train_larger_than_available_takes_all():
    b = _bench(train=1000, val=2)
    train = b.load("train")
    assert len(train) == count_lines(GSM8K_DIR / "train.jsonl") == 16
    assert len({e.id for e in train}) == 16
    assert b.load("val") == []


def test_gsm8k_relative_data_dir_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    b = _bench()
    assert len(b.load("train")) == 4


def test_gsm8k_absolute_data_dir():
    b = _bench(data_dir=str(GSM8K_DIR))
    assert len(b.load("train")) == 4


# --- AC30 ------------------------------------------------------------------

def test_build_benchmark_gsm8k_and_build_prompt():
    b = build_benchmark({"name": "gsm8k", "source": "local", "data_dir": "data/sample/gsm8k",
                         "train": 4, "val": 2}, seed=0)
    assert isinstance(b, GSM8K)
    assert b.train == 4 and b.val == 2
    ex = b.load("train")[0]
    skill = get_baseline("cot")
    p = b.build_prompt(skill, ex)
    assert skill.instruction in p
    assert ex.question in p


def test_build_benchmark_minimal_config():
    b = build_benchmark({"name": "gsm8k", "source": "local", "data_dir": "data/sample/gsm8k"}, seed=0)
    assert isinstance(b, GSM8K)


# --- loop 2026-11-10-004: dev -> train, splits classvar, attributes ----------

def _ref_order(seed, n=16):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


@pytest.mark.parametrize("seed", [0, 1, 7, 123, 2026])
def test_gsm8k_train_val_match_reference_sampling(seed):
    b = _bench(seed=seed, train=4, val=2)
    order = _ref_order(seed)
    train_ids = [e.id for e in b.load("train")]
    val_ids = [e.id for e in b.load("val")]
    assert train_ids == ["train-%d" % i for i in order[:4]]
    assert val_ids == ["train-%d" % i for i in order[4:6]]
    assert not (set(train_ids) & set(val_ids))


def test_gsm8k_test_is_all_rows_in_file_order(bench):
    assert [e.id for e in bench.load("test")] == ["test-%d" % i for i in range(6)]


@pytest.mark.parametrize("split", ["dev", "heldout", "search", "final"])
def test_gsm8k_old_or_foreign_split_raises(bench, split):
    with pytest.raises(ValueError):
        bench.load(split)


def test_gsm8k_unknown_split_message_names_train(bench):
    with pytest.raises(ValueError, match="train, val or test"):
        bench.load("dev")


def test_gsm8k_splits_classvar():
    assert GSM8K.splits == ("train", "val", "test")


def test_benchmark_default_splits_classvar():
    from skillopt.benchmarks.base import Benchmark

    class _Mini(Benchmark):
        name = "mini_test_bench"
        answer_format = "x"

        def load(self, split):
            return []

        def extract_answer(self, text):
            return None

        def score(self, pred, gold):
            return False

    assert _Mini.splits == ("train", "val", "test")
    assert _Mini().splits == ("train", "val", "test")


def test_gsm8k_constructor_attributes():
    b = _bench(train=3, val=5)
    assert b.train == 3
    assert b.val == 5
    assert not hasattr(b, "dev_size")
    assert not hasattr(b, "val_size")


def test_gsm8k_default_sizes():
    b = GSM8K(source="local")
    assert b.train == 200 and b.val == 200


@pytest.mark.parametrize("kw", [{"dev_size": 4}, {"val_size": 2}])
def test_gsm8k_old_kwargs_type_error(kw):
    with pytest.raises(TypeError):
        GSM8K(source="local", data_dir="data/sample/gsm8k", **kw)


# --- loop 2026-11-11-003: answer format without angle brackets ----------------

def test_gsm8k_answer_format_has_no_angle_brackets():
    fmt = GSM8K.answer_format
    assert "<" not in fmt and ">" not in fmt
    assert "The answer is" in fmt


def test_gsm8k_extract_answer_inside_answer_tag(bench):
    assert bench.extract_answer("... The answer is <answer>42</answer>.") == "42"


def test_gsm8k_build_prompt_uses_answer_format(bench):
    ex = bench.load("train")[0]
    assert bench.answer_format_for(ex) == GSM8K.answer_format
    assert GSM8K.answer_format in bench.build_prompt(get_baseline("cot"), ex)
