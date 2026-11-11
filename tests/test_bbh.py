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


# --- loop 2026-11-11-003: per-task answer formats and extraction -----------------

from skillopt.baselines import get_baseline  # noqa: E402
from skillopt.benchmarks import Example  # noqa: E402
from skillopt.benchmarks.bbh import (  # noqa: E402
    BBH_BINARY_CHOICES,
    BBH_TASK_KINDS,
    bbh_answer_format,
    extract_bbh_answer,
)

_KINDS = {"multiple_choice", "binary", "numeric", "dyck", "word_sorting"}
_BINARY = {"boolean_expressions": ("True", "False"), "causal_judgement": ("Yes", "No"),
           "navigate": ("Yes", "No"), "web_of_lies": ("Yes", "No"),
           "sports_understanding": ("yes", "no"), "formal_fallacies": ("valid", "invalid")}
_NUMERIC = {"object_counting", "multistep_arithmetic_two"}
_MC_TASKS = [t for t in BBH_TASKS
             if t not in _BINARY and t not in _NUMERIC and t not in ("dyck_languages", "word_sorting")]


def _tasks_of(kind):
    return [t for t in BBH_TASKS if BBH_TASK_KINDS[t] == kind]


def test_bbh_task_kinds_cover_all_tasks():
    assert set(BBH_TASK_KINDS) == set(BBH_TASKS)
    assert len(BBH_TASK_KINDS) == 27
    assert set(BBH_TASK_KINDS.values()) <= _KINDS
    counts = Counter(BBH_TASK_KINDS.values())
    assert counts == {"multiple_choice": 17, "binary": 6, "numeric": 2, "dyck": 1, "word_sorting": 1}
    assert set(_tasks_of("binary")) == set(_BINARY)
    assert set(_tasks_of("numeric")) == _NUMERIC
    assert _tasks_of("dyck") == ["dyck_languages"]
    assert _tasks_of("word_sorting") == ["word_sorting"]
    assert _tasks_of("multiple_choice") == _MC_TASKS


def test_bbh_binary_choices_exact():
    assert BBH_BINARY_CHOICES == _BINARY


@pytest.mark.parametrize("task", list(BBH_TASKS) + [None, "not_a_bbh_task"])
def test_bbh_answer_format_properties(task):
    fmt = bbh_answer_format(task)
    assert isinstance(fmt, str) and fmt
    assert "<" not in fmt and ">" not in fmt
    assert "So the answer is" in fmt
    assert "\n" not in fmt


@pytest.mark.parametrize("task", _MC_TASKS)
def test_bbh_answer_format_multiple_choice(task):
    fmt = bbh_answer_format(task)
    assert "(X)" in fmt
    assert "option" in fmt.lower()


@pytest.mark.parametrize("task", sorted(_BINARY))
def test_bbh_answer_format_binary(task):
    a, b = _BINARY[task]
    fmt = bbh_answer_format(task)
    assert f'"So the answer is {a}."' in fmt
    assert f'"So the answer is {b}."' in fmt
    assert "(X)" not in fmt


@pytest.mark.parametrize("task", sorted(_NUMERIC))
def test_bbh_answer_format_numeric(task):
    assert "number" in bbh_answer_format(task).lower()


def test_bbh_answer_format_dyck_and_word_sorting():
    dyck = bbh_answer_format("dyck_languages")
    assert "closing brackets" in dyck
    assert "spaces" in dyck
    assert "sorted words" in bbh_answer_format("word_sorting")


def test_bbh_unknown_task_format_is_free_form():
    assert bbh_answer_format("not_a_bbh_task") == bbh_answer_format(None)


def test_bbh_class_answer_format_is_free_form():
    assert BBH.answer_format == bbh_answer_format(None)


def _train_example(b, task):
    for e in b.load("train"):
        if e.meta["task"] == task:
            return e
    raise AssertionError(f"no {task} example in train")


def test_bbh_build_prompt_uses_per_task_format(bench):
    du = _train_example(bench, "date_understanding")
    be = _train_example(bench, "boolean_expressions")
    skill = get_baseline("cot")
    p_du = bench.build_prompt(skill, du)
    p_be = bench.build_prompt(skill, be)
    assert bench.answer_format_for(du) == bbh_answer_format("date_understanding")
    assert bench.answer_format_for(be) == bbh_answer_format("boolean_expressions")
    assert bbh_answer_format("date_understanding") in p_du
    assert bbh_answer_format("boolean_expressions") in p_be
    assert '"So the answer is True."' in p_be
    assert "(X)" in p_du


MC_CASES = [
    ("So the answer is (A).", "a"),
    ("So the answer is A.", "a"),
    ("So the answer is <answer>(B)</answer>.", "b"),
    ("So the answer is **(C)**", "c"),
    ("So the answer is (F) 07/31/2021.", "f"),
    ("So the answer is:\n(D)", "d"),
    ("The answer is (A) at first, but on reflection the answer is (E)", "e"),
    ("So the answer is about time.", None),
    ("So the answer is a dog.", None),
    ("", None),
    (None, None),
]

BINARY_CASES = [
    ("boolean_expressions", "So the answer is <answer>True</answer>.", "true"),
    ("web_of_lies", "So the answer is Yes, Dan tells the truth.", "yes"),
    ("sports_understanding", "So the answer is no.", "no"),
    ("formal_fallacies", "So the answer is **invalid**.", "invalid"),
    ("causal_judgement", "Thinking about causation here.\nNo", "no"),
]

NUMERIC_CASES = [
    ("object_counting", "So the answer is 7.", "7"),
    ("object_counting", "So the answer is <answer>1,234</answer>.", "1234"),
    ("multistep_arithmetic_two", "So the answer is -42.", "-42"),
    ("multistep_arithmetic_two", "So the answer is \\boxed{15}.", "15"),
]

DYCK_CASES = [
    ("So the answer is ] ) }.", "] ) }"),
    ("So the answer is <answer>])}</answer>", "] ) }"),
    ("So the answer is > ]", "> ]"),
]

WORD_CASES = [
    ("So the answer is apple, banana, cherry.", "apple banana cherry"),
    ('So the answer is "apple  banana cherry"', "apple banana cherry"),
]

FREE_CASES = [
    ("So the answer is (B).", "b"),
    ("So the answer is <answer>True</answer>.", "true"),
    ("", None),
]


@pytest.mark.parametrize("text,expected", MC_CASES)
def test_extract_bbh_multiple_choice(text, expected):
    assert extract_bbh_answer(text, "date_understanding") == expected


@pytest.mark.parametrize("task,text,expected", BINARY_CASES)
def test_extract_bbh_binary(task, text, expected):
    assert extract_bbh_answer(text, task) == expected


@pytest.mark.parametrize("task,text,expected", NUMERIC_CASES)
def test_extract_bbh_numeric(task, text, expected):
    assert extract_bbh_answer(text, task) == expected


@pytest.mark.parametrize("text,expected", DYCK_CASES)
def test_extract_bbh_dyck(text, expected):
    assert extract_bbh_answer(text, "dyck_languages") == expected


def test_bbh_dyck_scores_against_gold(bench):
    pred = extract_bbh_answer("So the answer is <answer>])}</answer>", "dyck_languages")
    assert bench.score(pred, "] ) }") is True


@pytest.mark.parametrize("text,expected", WORD_CASES)
def test_extract_bbh_word_sorting(text, expected):
    assert extract_bbh_answer(text, "word_sorting") == expected


def test_bbh_word_sorting_scores_against_gold(bench):
    for text, _ in WORD_CASES:
        assert bench.score(extract_bbh_answer(text, "word_sorting"), "apple banana cherry") is True


@pytest.mark.parametrize("text,expected", FREE_CASES)
def test_extract_bbh_free_form(bench, text, expected):
    assert extract_bbh_answer(text, None) == expected
    assert bench.extract_answer(text) == expected


def test_extract_bbh_free_form_last_line_and_answer_is(bench):
    assert bench.score(extract_bbh_answer("answer is True", None), "True") is True
    assert bench.score(extract_bbh_answer("reasoning\nmore\nYes\n\n  \n", None), "Yes") is True
    assert extract_bbh_answer(None, None) is None


def test_extract_bbh_default_task_is_free_form():
    assert extract_bbh_answer("So the answer is (B).") == "b"


def test_extract_bbh_unknown_task_is_free_form():
    for text, expected in FREE_CASES:
        assert extract_bbh_answer(text, "not_a_bbh_task") == expected


def test_extract_bbh_empty_remainder_uses_next_line():
    assert extract_bbh_answer("So the answer is\n\n(C)\n", "date_understanding") == "c"
    assert extract_bbh_answer("So the answer is\n  \nTrue", "boolean_expressions") == "true"


def _ex(task):
    return Example(id=f"{task}-0", question="q", answer="x", meta={"task": task})


@pytest.mark.parametrize("task,text,expected",
                         [("date_understanding", t, e) for t, e in MC_CASES]
                         + BINARY_CASES + NUMERIC_CASES
                         + [("dyck_languages", t, e) for t, e in DYCK_CASES]
                         + [("word_sorting", t, e) for t, e in WORD_CASES])
def test_bbh_extract_answer_for_dispatches_on_meta_task(bench, task, text, expected):
    assert bench.extract_answer_for(text, _ex(task)) == expected
