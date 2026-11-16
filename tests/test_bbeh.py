"""BIG-Bench Extra Hard (BBEH) (loop 2026-11-16-003). Synthetic rows only."""
import json
import random
import re
import sys
import types

import pytest

from skillopt.benchmarks import BBEH, BBEH_TASKS, build_benchmark
from skillopt.benchmarks.bbeh import (
    BBEH_ANSWER_FORMAT,
    bbeh_fuzzy_match,
    extract_bbeh_answer,
    normalize_bbeh_reference,
)


def _write_task(d, task, n, mini=None):
    rows = []
    for i in range(n):
        rows.append({"input": f"Synthetic {task} input {i}", "target": str(i),
                     "mini": int(bool(mini and i in mini))})
    with open(d / f"{task}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


@pytest.fixture
def data(tmp_path):
    _write_task(tmp_path, "boardgame_qa", 10, mini={1, 4, 7})
    _write_task(tmp_path, "web_of_lies", 8, mini={0, 5})
    _write_task(tmp_path, "zebra_puzzles", 6, mini={2})
    return tmp_path


def _idx_by_task(examples):
    out = {}
    for e in examples:
        task = e.meta["task"]
        assert e.id.startswith(f"{task}-"), e.id
        out.setdefault(task, []).append(int(e.id[len(task) + 1:]))
    return out


def _expected(n, train, val, seed):
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return sorted(order[:train]), sorted(order[train:train + val]), sorted(order[train + val:])


@pytest.fixture(scope="module")
def scorer(tmp_path_factory):
    d = tmp_path_factory.mktemp("bbeh_scorer")
    _write_task(d, "web_of_lies", 2)
    return BBEH(source="local", data_dir=str(d))


# --- crit 19: task list --------------------------------------------------------------

def test_bbeh_tasks():
    assert len(BBEH_TASKS) == 23
    assert len(set(BBEH_TASKS)) == 23
    for t in BBEH_TASKS:
        assert re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", t), t
    for t in ("boardgame_qa", "multistep_arithmetic", "web_of_lies", "zebra_puzzles", "sarc_triples", "nycc",
              "linguini"):
        assert t in BBEH_TASKS, t


# --- crit 20: answer format ---------------------------------------------------------

def test_bbeh_answer_format():
    assert "The answer is:" in BBEH_ANSWER_FORMAT
    assert "<" not in BBEH_ANSWER_FORMAT and ">" not in BBEH_ANSWER_FORMAT


def test_bbeh_answer_format_for_every_task(data):
    bench = BBEH(source="local", data_dir=str(data))
    for e in bench.load("test"):
        assert bench.answer_format_for(e) == BBEH_ANSWER_FORMAT


# --- crit 21: official evaluator examples -------------------------------------------

@pytest.mark.parametrize("text,ref,ok", [
    ("Ok The final answer is: \\boxed{4}.", "4", True),
    ("[Reasoning] The final answer is: \\boxed{4}.", "3", False),
    ("Alright! The final answer is: 2, 3, 4", "2,3,4", True),
    ("blah blah The final answer is: 2, 3, 4", "2,3,5", False),
    ("Ok The answer is: (A)", "a", True),
    ("Ok The answer is: (A)", "b", False),
    ("Ok The answer is: **25**\nHere's why.", "25.0", True),
    ("Ok The answer is: **25**\nHere's why.", "26.0", False),
])
def test_official_examples(scorer, text, ref, ok):
    assert scorer.score(extract_bbeh_answer(text), normalize_bbeh_reference(ref)) is ok


# --- crit 22: case-insensitive prefix --------------------------------------------

@pytest.mark.parametrize("text,ref", [
    ("So the answer is (B).", "(B)"),
    ("The answer is: yes, no, yes", "yes, no, yes"),
    ("The answer is: E", "(E)"),
    ("The answer is: [thursday]", "[thursday]"),
    ("THE ANSWER IS: 7", "7"),
    ("The answer is: 3. Hmm, actually the answer is: 5", "5"),
    ("The answer is: $12$", "12"),
], ids=["so_answer_is", "list", "letter_vs_paren", "brackets", "upper", "last_prefix", "dollar"])
def test_prefix_variants(scorer, text, ref):
    assert scorer.score(extract_bbeh_answer(text), normalize_bbeh_reference(ref)) is True


def test_last_prefix_wins_negative(scorer):
    pred = extract_bbeh_answer("The answer is: 3. Hmm, actually the answer is: 5")
    assert scorer.score(pred, normalize_bbeh_reference("3")) is False


def test_multiline_first_line_only(scorer):
    pred = extract_bbeh_answer("The answer is: 42\n43 is wrong")
    assert scorer.score(pred, normalize_bbeh_reference("42")) is True


# --- crit 23: fuzzy match / normalize ------------------------------------------------

@pytest.mark.parametrize("pred,ref", [("(a)", "a"), ("a", "(a)"), ("25", "25.0"), ("[x]", "x"), ("x?", "x"),
                                      ("it's", "its")])
def test_fuzzy_match_true(pred, ref):
    assert bbeh_fuzzy_match(pred, ref) is True


def test_fuzzy_match_false():
    assert bbeh_fuzzy_match("a", "b") is False


def test_normalize_reference():
    assert normalize_bbeh_reference("Yes, No") == "yes,no"


@pytest.mark.parametrize("text", [None, "   ", ""])
def test_extract_none(text):
    assert extract_bbeh_answer(text) is None


def test_score_none(scorer):
    assert scorer.score(None, "4") is False


# --- crit 24: local data / tasks / heldout -------------------------------------------

def test_tasks_all_is_file_stems(data):
    bench = BBEH(source="local", data_dir=str(data))
    assert set(_idx_by_task(bench.load("test"))) == {"boardgame_qa", "web_of_lies", "zebra_puzzles"}


def test_tasks_subset(data):
    bench = BBEH(source="local", data_dir=str(data), tasks=["web_of_lies"])
    got = _idx_by_task(bench.load("test"))
    assert got == {"web_of_lies": list(range(8))}


@pytest.mark.parametrize("kw", [{"tasks": ["not_a_task"]}, {"heldout": ["not_a_task"]}])
def test_unknown_task_raises(data, kw):
    with pytest.raises(ValueError):
        BBEH(source="local", data_dir=str(data), **kw)


def test_heldout_removed_from_tasks(data):
    bench = BBEH(source="local", data_dir=str(data), tasks="all", heldout=["web_of_lies"], train=2, val=1)
    for split in ("train", "val", "test"):
        assert "web_of_lies" not in _idx_by_task(bench.load(split)), split
    held = _idx_by_task(bench.load("heldout"))
    assert held == {"web_of_lies": list(range(8))}


def test_heldout_overlap_with_explicit_tasks(data):
    bench = BBEH(source="local", data_dir=str(data), tasks=["web_of_lies", "zebra_puzzles"],
                 heldout=["web_of_lies"])
    assert set(_idx_by_task(bench.load("test"))) == {"zebra_puzzles"}
    assert set(_idx_by_task(bench.load("heldout"))) == {"web_of_lies"}


def test_build_benchmark_bbeh(data):
    bench = build_benchmark({"name": "bbeh", "source": "local", "data_dir": str(data), "train": 1, "val": 1})
    assert isinstance(bench, BBEH)
    assert len(bench.load("train")) == 3


# --- crit 25: splits ---------------------------------------------------------------

@pytest.mark.parametrize("train,val,heldout,expected", [
    (0, 0, None, ("test",)),
    (2, 0, None, ("train", "test")),
    (0, 2, None, ("val", "test")),
    (2, 2, None, ("train", "val", "test")),
    (2, 2, ["web_of_lies"], ("train", "val", "test", "heldout")),
    (0, 0, ["web_of_lies"], ("test", "heldout")),
])
def test_splits_per_instance(data, train, val, heldout, expected):
    bench = BBEH(source="local", data_dir=str(data), train=train, val=val, heldout=heldout)
    assert tuple(bench.splits) == expected


@pytest.mark.parametrize("train,val,seed", [(2, 2, 0), (3, 1, 5), (0, 2, 9), (5, 5, 1)])
def test_per_task_split_rule(data, train, val, seed):
    sizes = {"boardgame_qa": 10, "web_of_lies": 8, "zebra_puzzles": 6}
    bench = BBEH(source="local", data_dir=str(data), train=train, val=val, seed=seed)
    got = {s: _idx_by_task(bench.load(s)) for s in ("train", "val", "test")}
    all_ids = []
    for task, n in sizes.items():
        et, ev, es = _expected(n, train, val, seed)
        assert got["train"].get(task, []) == et, task
        assert got["val"].get(task, []) == ev, task
        assert got["test"].get(task, []) == es, task
    for s in ("train", "val", "test"):
        all_ids += [e.id for e in bench.load(s)]
    assert len(all_ids) == len(set(all_ids)) == sum(sizes.values())


def test_split_overflow_per_task(data):
    bench = BBEH(source="local", data_dir=str(data), tasks=["zebra_puzzles"], train=4, val=4, seed=0)
    assert len(bench.load("train")) == 4
    assert len(bench.load("val")) == 2
    assert bench.load("test") == []


def test_no_id_in_two_splits_with_heldout(data):
    bench = BBEH(source="local", data_dir=str(data), heldout=["zebra_puzzles"], train=2, val=2)
    seen = {}
    for s in ("train", "val", "test", "heldout"):
        for e in bench.load(s):
            assert e.id not in seen, (e.id, seen.get(e.id), s)
            seen[e.id] = s


def test_examples_ids_and_meta(data):
    bench = BBEH(source="local", data_dir=str(data), tasks=["web_of_lies"])
    test = bench.load("test")
    assert [e.id for e in test] == [f"web_of_lies-{i}" for i in range(8)]
    assert all(e.meta["task"] == "web_of_lies" for e in test)
    assert test[3].question.strip() == "Synthetic web_of_lies input 3"


def test_load_train_zero_and_unknown_split(data):
    bench = BBEH(source="local", data_dir=str(data))
    assert bench.load("train") == []
    with pytest.raises(ValueError):
        bench.load("nope")


def test_same_seed_same_ids(data):
    a = BBEH(source="local", data_dir=str(data), train=2, val=2, seed=3)
    b = BBEH(source="local", data_dir=str(data), train=2, val=2, seed=3)
    for s in ("train", "val", "test"):
        assert [e.id for e in a.load(s)] == [e.id for e in b.load(s)]


# --- crit 26: mini -------------------------------------------------------------------

def test_mini_keeps_original_ids(data):
    bench = BBEH(source="local", data_dir=str(data), mini=True)
    got = _idx_by_task(bench.load("test"))
    assert got == {"boardgame_qa": [1, 4, 7], "web_of_lies": [0, 5], "zebra_puzzles": [2]}


def test_mini_filter_before_split(data):
    bench = BBEH(source="local", data_dir=str(data), tasks=["boardgame_qa"], mini=True, train=1, val=1, seed=2)
    pool = [1, 4, 7]
    order = list(range(len(pool)))
    random.Random(2).shuffle(order)
    exp_train = sorted(pool[k] for k in order[:1])
    exp_val = sorted(pool[k] for k in order[1:2])
    exp_test = sorted(pool[k] for k in order[2:])
    assert _idx_by_task(bench.load("train")) == {"boardgame_qa": exp_train}
    assert _idx_by_task(bench.load("val")) == {"boardgame_qa": exp_val}
    assert _idx_by_task(bench.load("test")) == {"boardgame_qa": exp_test}


# --- crit 13 / 27: HF source ---------------------------------------------------------

def _fake(monkeypatch, rows):
    calls = []
    mod = types.ModuleType("datasets")

    def load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return list(rows)

    mod.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", mod)
    return calls


HF_ROWS = [
    {"input": "hf q0", "target": "1", "task": "boardgame qa", "mini": 0},
    {"input": "hf q1", "target": "2", "task": "web of lies", "mini": 1},
    {"input": "hf q2", "target": "3", "task": "boardgame qa", "mini": 1},
    {"input": "hf q3", "target": "4", "task": "boardgame qa", "mini": 0},
]


def test_hf_load_call_and_task_mapping(monkeypatch):
    calls = _fake(monkeypatch, HF_ROWS)
    bench = BBEH(source="hf", tasks=["boardgame_qa", "web_of_lies"])
    assert calls == []
    test = bench.load("test")
    assert calls == [(("BBEH/bbeh",), {"split": "train"})]
    got = _idx_by_task(test)
    assert got == {"boardgame_qa": [0, 1, 2], "web_of_lies": [0]}
    bench.load("test")
    bench.load("train")
    assert len(calls) == 1


def test_hf_construction_does_not_import_datasets(monkeypatch):
    monkeypatch.setitem(sys.modules, "datasets", None)
    BBEH(source="hf", tasks="all", heldout=["web_of_lies"], train=2, val=2)
