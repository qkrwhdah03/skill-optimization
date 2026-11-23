"""Tests for non-math reasoning benchmarks: mcq helper, MuSR, FOLIO, LSAT-AR, StrategyQA, MMLU-Pro."""
from __future__ import annotations

import json
import random
import sys
import types

import pytest

from skillopt.benchmarks import (
    FOLIO,
    FOLIO_LABELS,
    LSATAR,
    MMLU_PRO_CATEGORIES,
    MMLU_PRO_GROUPS,
    MUSR_TASKS,
    MMLUPro,
    MuSR,
    StrategyQA,
    build_benchmark,
    get_benchmark_class,
)
from skillopt.benchmarks.folio import FOLIO_ANSWER_FORMAT, extract_folio_answer, normalize_folio_label
from skillopt.benchmarks.mcq import extract_mcq_answer, format_mcq_options, mcq_answer_format, score_mcq
from skillopt.benchmarks.strategyqa import (
    STRATEGYQA_ANSWER_FORMAT,
    extract_yes_no_answer,
    normalize_yes_no,
)


def _fake_datasets(monkeypatch, loader):
    calls = []
    mod = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return loader(*args, **kwargs)

    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)
    return calls


# --- mcq helper -----------------------------------------------------------------------


def test_format_mcq_options_and_answer_format():
    assert format_mcq_options(["alpha", "(B) beta", "(c) gamma"]) == "(A) alpha\n(B) beta\n(C) gamma"
    assert "(A-E)" in mcq_answer_format(5)
    assert "(A-J)" in mcq_answer_format(10)
    assert "(A-B)" in mcq_answer_format(1)


@pytest.mark.parametrize(
    "text,valid,expected",
    [
        ("The answer is (B).", "abcde", "b"),
        ("The answer is B.", "abcde", "b"),
        ("The answer is: **D**.", "abcde", "d"),
        ("The answer is <answer>(D)</answer>.", "abcde", "d"),
        ("The final answer is Option C.", "abcde", "c"),
        ("First thought (A). Actually the answer is (E).", "abcde", "e"),
        ("The answer is (B) because (A) contradicts.", "abcde", "b"),
        ("Therefore \\boxed{C}", "abcde", "c"),
        ("Final choice:\n\\boxed{(D)}", "abcde", "d"),
        ("Looking at the options\n(J)", "abcdefghij", "j"),
        ("The answer is (Z).", "abcde", None),
        ("cat", "abcde", None),
        ("   ", "abcde", None),
        (None, "abcde", None),
    ],
)
def test_extract_mcq_answer_variants(text, valid, expected):
    assert extract_mcq_answer(text, valid) == expected


def test_score_mcq():
    assert score_mcq("b", "B") is True
    assert score_mcq("a", "b") is False
    assert score_mcq(None, "a") is False


# --- MuSR -----------------------------------------------------------------------------


def test_musr_registered_and_local_sample_splits():
    assert get_benchmark_class("musr") is MuSR
    assert MUSR_TASKS == ("murder_mysteries", "object_placements", "team_allocation")

    bench = build_benchmark({"name": "musr", "source": "local", "train": 2, "val": 2, "seed": 0})
    assert tuple(bench.splits) == ("train", "val", "test")
    assert len(bench.load("train")) == 6
    assert len(bench.load("val")) == 6
    assert len(bench.load("test")) == 6

    ex = bench.load("train")[0]
    assert "Options:" in ex.question
    assert "(A)" in ex.question
    assert ex.answer in ("a", "b", "c", "d", "e")
    assert "(A-" in bench.answer_format_for(ex)
    assert bench.score(bench.extract_answer_for(f"The answer is ({ex.answer.upper()}).", ex), ex.answer)


def test_musr_heldout_and_list_choices(tmp_path):
    rows = [
        {"narrative": "Story", "question": "Where?", "choices": ["x", "y", "z"], "answer_index": 2},
        {"narrative": "Story2", "question": "Who?", "choices": "['p', 'q']", "answer_index": 0},
    ]
    for task in ("murder_mysteries", "object_placements"):
        (tmp_path / f"{task}.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    bench = MuSR(source="local", data_dir=str(tmp_path), tasks="all", heldout=["murder_mysteries"], train=1)
    assert tuple(bench.splits) == ("train", "test", "heldout")
    assert len(bench.load("train")) == 1
    assert len(bench.load("heldout")) == 2
    assert bench.extract_answer_for("The answer is (C).", bench.load("heldout")[0]) == "c"
    assert bench.extract_answer_for("The answer is (C).", bench.load("heldout")[1]) is None


def test_musr_hf_lazy_and_mocked(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    b = MuSR(source="hf", tasks=["murder_mysteries"], train=1, val=1)

    calls = _fake_datasets(
        monkeypatch,
        lambda path, split: [
            {"narrative": "N0", "question": "Q0?", "choices": "['A', 'B']", "answer_index": 0},
            {"narrative": "N1", "question": "Q1?", "choices": "['A', 'B']", "answer_index": 1},
            {"narrative": "N2", "question": "Q2?", "choices": "['A', 'B']", "answer_index": 0},
        ],
    )
    assert len(b.load("train")) == 1
    assert len(b.load("val")) == 1
    assert len(b.load("test")) == 1
    assert calls == [(("TAUR-Lab/MuSR",), {"split": "murder_mysteries"})]


# --- FOLIO ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,norm",
    [("True", "true"), (" false.", "false"), ("Uncertain!", "uncertain"), ("maybe", None), (None, None)],
)
def test_folio_normalize(raw, norm):
    assert normalize_folio_label(raw) == norm


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Reasoning...\nThe answer is True.", "true"),
        ("The final answer is: **Uncertain**.", "uncertain"),
        ("The answer is <answer>False</answer>.", "false"),
        ("Conclusion is \\boxed{Uncertain}", "uncertain"),
        ("Based on premises:\nFalse", "false"),
        ("The answer is unknown.", None),
        ("", None),
    ],
)
def test_folio_extract(text, expected):
    assert extract_folio_answer(text) == expected


def test_folio_splits_and_hf(monkeypatch):
    assert get_benchmark_class("folio") is FOLIO
    assert FOLIO_LABELS == ("true", "false", "uncertain")
    assert "Uncertain" in FOLIO_ANSWER_FORMAT

    local = FOLIO(source="local", train=3, val=2, seed=4)
    tr, va, te = local.load("train"), local.load("val"), local.load("test")
    assert len(tr) == 3 and len(va) == 2 and len(te) == 4
    assert not ({e.id for e in tr} & {e.id for e in va})
    assert all(e.answer in FOLIO_LABELS for e in tr + va + te)
    assert "Premises:" in tr[0].question and "Conclusion:" in tr[0].question
    assert local.score(local.extract_answer("The answer is Uncertain."), "uncertain") is True

    monkeypatch.setitem(sys.modules, "datasets", None)
    hf = FOLIO(source="hf", train=2, val=1, seed=0)
    calls = _fake_datasets(
        monkeypatch,
        lambda path, split: [
            {"premises": f"P {split} {i}", "conclusion": "C", "label": "True", "story_id": i, "example_id": i}
            for i in range(4)
        ],
    )
    assert len(hf.load("train")) == 2
    assert len(hf.load("val")) == 1
    assert len(hf.load("test")) == 4
    assert calls == [
        (("tasksource/folio",), {"split": "train"}),
        (("tasksource/folio",), {"split": "validation"}),
    ]


# --- LSAT-AR --------------------------------------------------------------------------


def test_lsat_ar_splits_stripping_and_fallback(tmp_path, monkeypatch):
    assert get_benchmark_class("lsat_ar") is LSATAR

    local = LSATAR(source="local", train=2, val=2, seed=0)
    tr, va, te = local.load("train"), local.load("val"), local.load("test")
    assert (len(tr), len(va), len(te)) == (2, 2, 4)
    assert "Among A through E" not in tr[0].question
    assert "(A)" in tr[0].question
    assert tr[0].answer in ("a", "b", "c", "d", "e")
    assert local.score(local.extract_answer("The answer is (D)."), "d") is True

    # Fallback formatting when query does not already contain options
    row = {"query": "Logic puzzle stem.", "choices": ["opt1", "opt2", "opt3", "opt4", "opt5"], "gold": [3]}
    (tmp_path / "data.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    custom = LSATAR(source="local", data_dir=str(tmp_path))
    ex = custom.load("test")[0]
    assert "(D) opt4" in ex.question
    assert ex.answer == "d"

    monkeypatch.setitem(sys.modules, "datasets", None)
    hf = LSATAR(source="hf", train=1, val=1)
    calls = _fake_datasets(
        monkeypatch,
        lambda path, split: [
            {"query": f"Q{i}\n(A) x\nA: Among A through E, the answer is", "choices": ["x"] * 5, "gold": [1]}
            for i in range(3)
        ],
    )
    assert len(hf.load("train")) == 1
    assert "Among A through E" not in hf.load("train")[0].question
    assert calls == [(("hails/agieval-lsat-ar",), {"split": "test"})]


# --- StrategyQA -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [(True, "yes"), (False, "no"), ("Yes.", "yes"), ("NO!", "no"), ("maybe", None), (None, None)],
)
def test_strategyqa_normalize(raw, expected):
    assert normalize_yes_no(raw) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Step 1... The answer is Yes.", "yes"),
        ("The final answer is: **No**.", "no"),
        ("Therefore \\boxed{Yes}", "yes"),
        ("Conclusion:\nNo", "no"),
        ("The answer is maybe.", None),
    ],
)
def test_strategyqa_extract(text, expected):
    assert extract_yes_no_answer(text) == expected


def test_strategyqa_splits_and_hf(monkeypatch):
    assert get_benchmark_class("strategyqa") is StrategyQA
    assert "Yes" in STRATEGYQA_ANSWER_FORMAT

    local = StrategyQA(source="local", train=3, val=2, seed=1)
    tr, va, te = local.load("train"), local.load("val"), local.load("test")
    assert (len(tr), len(va), len(te)) == (3, 2, 4)
    assert not ({e.id for e in tr} & {e.id for e in va})
    assert all(e.answer in ("yes", "no") for e in tr + va + te)
    assert local.score("yes", "yes") is True and local.score("no", "yes") is False

    monkeypatch.setitem(sys.modules, "datasets", None)
    hf = StrategyQA(source="hf", train=2, val=1)
    calls = _fake_datasets(
        monkeypatch,
        lambda path, split: [{"qid": f"{split}-{i}", "question": f"Q{i}?", "answer": (i % 2 == 0)} for i in range(4)],
    )
    assert len(hf.load("train")) == 2
    assert len(hf.load("test")) == 4
    assert calls == [
        (("ChilleD/StrategyQA",), {"split": "train"}),
        (("ChilleD/StrategyQA",), {"split": "test"}),
    ]


# --- MMLU-Pro -------------------------------------------------------------------------


def test_mmlu_pro_categories_groups_and_local_sample():
    assert get_benchmark_class("mmlu_pro") is MMLUPro
    assert len(MMLU_PRO_CATEGORIES) == 14
    assert "math" not in MMLU_PRO_GROUPS["non_math"]
    assert len(MMLU_PRO_GROUPS["non_math"]) == 13

    bench = MMLUPro(source="local", categories="non_math", heldout=["law"], train=2, val=2, seed=0)
    assert bench.categories == ["biology", "philosophy"]
    assert bench.heldout == ["law"]
    assert tuple(bench.splits) == ("train", "val", "test", "heldout")
    assert len(bench.load("train")) == 4
    assert len(bench.load("val")) == 4
    assert len(bench.load("test")) == 4
    assert len(bench.load("heldout")) == 6

    ex = bench.load("heldout")[0]
    assert "(J)" in ex.question
    assert "(A-J)" in bench.answer_format_for(ex)
    assert bench.score(bench.extract_answer_for("The answer is (B).", ex), "b") is True


def test_mmlu_pro_hf_normalizes_computer_science_and_short_options(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    bench = MMLUPro(source="hf", categories=["computer science", "math"], train=1, val=0, seed=0)
    assert bench.categories == ["computer_science", "math"]

    calls = _fake_datasets(
        monkeypatch,
        lambda path, split: [
            {"question": "CS1?", "options": ["x", "y", "z"], "answer": "C", "answer_index": 2, "category": "computer science"},
            {"question": "CS2?", "options": ["x", "y", "z"], "answer": "A", "answer_index": 0, "category": "computer science"},
            {"question": "M1?", "options": ["1", "2", "3", "4"], "answer": "D", "answer_index": 3, "category": "math"},
        ],
    )
    tr = bench.load("train")
    te = bench.load("test")
    assert len(tr) == 2
    assert len(te) == 1
    assert "(A-C)" in bench.answer_format_for(tr[0])
    assert bench.extract_answer_for("The answer is (D).", tr[0]) is None  # only 3 options (A-C)
    assert calls == [(("TIGER-Lab/MMLU-Pro",), {"split": "test"})]

    # Verify unknown category raises ValueError
    with pytest.raises(ValueError):
        MMLUPro(source="local", categories=["not_a_real_category"])

    # Verify seed determinism
    a = MMLUPro(source="local", categories="all", train=2, val=2, seed=7)
    b = MMLUPro(source="local", categories="all", train=2, val=2, seed=7)
    order = [random.Random(7).randint(0, 100)]
    assert order
    assert [e.id for e in a.load("train")] == [e.id for e in b.load("train")]
