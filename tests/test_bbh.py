"""G. BBH benchmark (local sample data)."""
import random
from collections import Counter

import pytest

from helpers import BBH_DIR, count_lines
from skillopt.benchmarks import BBH, BBH_TASK_GROUPS, BBH_TASKS


def _bench(**kw):
    args = dict(source="local", data_dir="data/sample/bbh", heldout=["web_of_lies"],
                train=2, val=2, seed=0)
    args.update(kw)
    return BBH(**args)


@pytest.fixture(scope="module")
def bench():
    return _bench()


# --- AC28 ------------------------------------------------------------------

def test_bbh_extract_option_letter(bench):
    assert bench.extract_answer("So the answer is (B).") == "b"


@pytest.mark.parametrize("pred,gold,ok", [
    ("b", "(B)", True),
    ("(B)", "b", True),
    ("B.", "(B)", True),
    ("a", "(B)", False),
    (None, "(B)", False),
])
def test_bbh_score_options(bench, pred, gold, ok):
    assert bench.score(pred, gold) is ok


def test_bbh_answer_is_true(bench):
    pred = bench.extract_answer("answer is True")
    assert bench.score(pred, "True") is True
    assert bench.score(pred, "False") is False


def test_bbh_case_insensitive(bench):
    assert bench.score(bench.extract_answer("So the answer is yes."), "Yes") is True


def test_bbh_no_answer_is_uses_last_nonempty_line(bench):
    pred = bench.extract_answer("Some reasoning\nmore reasoning\nYes\n\n  \n")
    assert pred is not None
    assert bench.score(pred, "Yes") is True
    assert bench.score(pred, "No") is False


def test_bbh_empty_text(bench):
    assert bench.extract_answer("") is None


# --- AC29 ------------------------------------------------------------------

def test_bbh_heldout_never_in_train_val_test(bench):
    for split in ("train", "val", "test"):
        for e in bench.load(split):
            assert not e.id.startswith("web_of_lies"), (split, e.id)
            assert e.meta.get("task") != "web_of_lies"


def test_bbh_heldout_contains_all_web_of_lies(bench):
    held = bench.load("heldout")
    assert len(held) == count_lines(BBH_DIR / "web_of_lies.jsonl")
    assert all(e.id.startswith("web_of_lies") for e in held)
    assert all(e.meta["task"] == "web_of_lies" for e in held)


def test_bbh_train_has_two_per_discovery_task(bench):
    train = bench.load("train")
    counts = Counter(e.meta["task"] for e in train)
    assert counts == {"boolean_expressions": 2, "date_understanding": 2}


def test_bbh_val_disjoint_from_train(bench):
    train_ids = {e.id for e in bench.load("train")}
    val_ids = {e.id for e in bench.load("val")}
    assert not (train_ids & val_ids)
    assert val_ids


def test_bbh_meta_task_set(bench):
    for split in ("train", "val", "test", "heldout"):
        for e in bench.load(split):
            assert e.meta["task"] in BBH_TASKS


def test_bbh_custom_task_group_restricts_train():
    b = _bench(tasks=["g"], task_groups={"g": ["boolean_expressions"]}, heldout=None)
    train = b.load("train")
    assert train
    assert {e.meta["task"] for e in train} == {"boolean_expressions"}


def test_bbh_unknown_task_raises():
    with pytest.raises(ValueError):
        b = _bench(tasks=["definitely_not_a_task"])
        b.load("train")


def test_bbh_unknown_split_raises(bench):
    with pytest.raises(ValueError):
        bench.load("bogus")


def test_bbh_task_constants():
    assert len(BBH_TASKS) == 27
    assert len(set(BBH_TASKS)) == 27
    for t in ("boolean_expressions", "date_understanding", "web_of_lies"):
        assert t in BBH_TASKS
    assert "algorithmic" in BBH_TASK_GROUPS
    assert "nlu" in BBH_TASK_GROUPS
    for group in BBH_TASK_GROUPS.values():
        assert all(t in BBH_TASKS for t in group)


# --- loop 2026-11-10-004: dev -> train, heldout attr, splits classvar --------

def _ref_order(seed, n):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return order


@pytest.mark.parametrize("seed", [0, 1, 7, 123])
def test_bbh_splits_match_reference_sampling(seed):
    b = _bench(seed=seed)
    assert b.tasks == ["boolean_expressions", "date_understanding"]
    exp_train, exp_val, exp_test = [], [], []
    for task in b.tasks:
        n = count_lines(BBH_DIR / f"{task}.jsonl")
        order = _ref_order(seed, n)
        exp_train += [f"{task}-{i}" for i in order[:2]]
        exp_val += [f"{task}-{i}" for i in order[2:4]]
        exp_test += [f"{task}-{i}" for i in order[4:]]
    assert [e.id for e in b.load("train")] == exp_train
    assert [e.id for e in b.load("val")] == exp_val
    assert [e.id for e in b.load("test")] == exp_test
    held = [e.id for e in b.load("heldout")]
    assert held == [f"web_of_lies-{i}" for i in range(8)]
    for split in ("train", "val", "test"):
        assert not any(i.startswith("web_of_lies") for i in [e.id for e in b.load(split)])


@pytest.mark.parametrize("split", ["dev", "search", "final"])
def test_bbh_old_or_foreign_split_raises(bench, split):
    with pytest.raises(ValueError):
        bench.load(split)


def test_bbh_unknown_split_message_names_train(bench):
    with pytest.raises(ValueError, match="train, val, test or heldout"):
        bench.load("dev")


def test_bbh_splits_classvar():
    assert BBH.splits == ("train", "val", "test", "heldout")


def test_bbh_constructor_attributes(bench):
    assert bench.train == 2
    assert bench.val == 2
    assert bench.heldout == ["web_of_lies"]
    assert "web_of_lies" not in bench.tasks
    assert not hasattr(bench, "heldout_tasks")
    assert not hasattr(bench, "dev_size")
    assert not hasattr(bench, "val_size")


def test_bbh_default_sizes():
    b = BBH(source="local")
    assert b.train == 10 and b.val == 10


def test_bbh_heldout_group_expands():
    b = _bench(heldout=["g"], task_groups={"g": ["web_of_lies", "date_understanding"]})
    assert b.heldout == ["web_of_lies", "date_understanding"]
    assert b.tasks == ["boolean_expressions"]
    held = b.load("heldout")
    assert {e.meta["task"] for e in held} == {"web_of_lies", "date_understanding"}
    assert len(held) == 16


def test_bbh_heldout_none_is_empty():
    b = _bench(heldout=None)
    assert b.heldout == []
    assert b.load("heldout") == []
    assert "web_of_lies" in b.tasks


@pytest.mark.parametrize("kw", [{"dev_size": 2}, {"val_size": 2}, {"heldout_tasks": ["web_of_lies"]}])
def test_bbh_old_kwargs_type_error(kw):
    with pytest.raises(TypeError):
        BBH(source="local", data_dir="data/sample/bbh", **kw)
