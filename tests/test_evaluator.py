"""A. Benchmark specs & Evaluator (raw per-benchmark results only)."""
import re
from statistics import mean

import pytest

from helpers import GSM8K_DIR, BBH_DIR, make_mock, make_result, make_skill, uniq
from skillopt.baselines import get_baseline
from skillopt.benchmarks import Benchmark, Example, register_benchmark
from skillopt.evaluator import (
    BenchmarkSpec,
    EvalResult,
    Evaluator,
    ItemResult,
    SkillEvaluation,
    build_benchmark_specs,
)

THINK_RESP = "<think>a b c</think>The answer is 3."

EVAL_SUMMARY_KEYS = {"skill_id", "benchmark", "split", "n", "accuracy", "avg_input_tokens",
                     "avg_output_tokens", "avg_reasoning_tokens", "avg_total_tokens", "avg_latency",
                     "num_truncated", "truncation_rate"}


class ListBench(Benchmark):
    name = "listbench"
    answer_format = "Answer with a number."

    def __init__(self, golds=(), **kw):
        self._examples = [Example(id=f"ex{i}", question=f"Question number {i}?", answer=str(g))
                          for i, g in enumerate(golds)]
        self.loaded = []

    def load(self, split):
        self.loaded.append(split)
        if split not in self.splits:
            raise ValueError(split)
        return list(self._examples)

    def extract_answer(self, text):
        nums = re.findall(r"-?\d+", text or "")
        return nums[-1] if nums else None

    def score(self, pred, gold):
        return pred is not None and pred == gold


LISTBENCH_NAME = uniq("listbench")
register_benchmark(LISTBENCH_NAME)(ListBench)

def _spec(key, golds):
    return BenchmarkSpec(key=key, benchmark=ListBench(golds))


def _multi(specs, responses=(THINK_RESP,), batch_size=16, **mock_kw):
    mock = make_mock(list(responses) if responses is not None else None, **mock_kw)
    return mock, Evaluator(mock, specs, batch_size=batch_size)


def _eval(golds, responses=(THINK_RESP,), batch_size=16, **mock_kw):
    """Single-benchmark evaluator under key 'lb'."""
    return _multi([_spec("lb", golds)], responses=responses, batch_size=batch_size, **mock_kw)


def _gsm8k_entry(**kw):
    e = {"source": "local", "data_dir": str(GSM8K_DIR), "train": 4, "val": 2}
    e.update(kw)
    return e


def _bbh_entry(**kw):
    e = {"source": "local", "data_dir": str(BBH_DIR), "heldout": ["web_of_lies"], "train": 2, "val": 2}
    e.update(kw)
    return e


def _sample_specs():
    return build_benchmark_specs({"gsm8k": _gsm8k_entry(), "bbh": _bbh_entry()}, seed=0)


# --- build_benchmark_specs (mapping form) ----------------------------------

def test_build_specs_keys_order_and_classes():
    from skillopt.benchmarks import BBH, GSM8K

    specs = _sample_specs()
    assert [s.key for s in specs] == ["gsm8k", "bbh"]
    assert isinstance(specs[0].benchmark, GSM8K)
    assert isinstance(specs[1].benchmark, BBH)
    for s in specs:
        assert isinstance(s, BenchmarkSpec)
        assert isinstance(s.benchmark, Benchmark)
        assert not hasattr(s, "splits")
    assert specs[0].benchmark.train == 4 and specs[0].benchmark.val == 2
    assert specs[1].benchmark.heldout == ["web_of_lies"]


def test_build_specs_insertion_order_kept():
    specs = build_benchmark_specs({"bbh": _bbh_entry(), "gsm8k": _gsm8k_entry()}, seed=0)
    assert [s.key for s in specs] == ["bbh", "gsm8k"]


def test_build_specs_none_entry_uses_defaults():
    from skillopt.benchmarks import GSM8K

    specs = build_benchmark_specs({"gsm8k": None}, seed=0)
    assert [s.key for s in specs] == ["gsm8k"]
    b = specs[0].benchmark
    assert isinstance(b, GSM8K)
    assert b.source == "local"
    assert b.train == 200 and b.val == 200
    assert len(b.load("test")) == 6


def test_build_specs_empty_dict_entry_uses_defaults():
    from skillopt.benchmarks import GSM8K

    specs = build_benchmark_specs({"gsm8k": {}}, seed=0)
    assert isinstance(specs[0].benchmark, GSM8K)


def test_build_specs_seed_passed_through():
    a = build_benchmark_specs({"gsm8k": _gsm8k_entry()}, seed=5)[0].benchmark
    import random
    order = list(range(16))
    random.Random(5).shuffle(order)
    assert [e.id for e in a.load("train")] == ["train-%d" % i for i in order[:4]]


def test_build_specs_name_alias_two_bbh():
    from skillopt.benchmarks import BBH

    specs = build_benchmark_specs({
        "bbh_a": _bbh_entry(name="bbh", tasks=["boolean_expressions"], heldout=[]),
        "bbh_b": _bbh_entry(name="bbh", tasks=["date_understanding"], heldout=[]),
    })
    assert [s.key for s in specs] == ["bbh_a", "bbh_b"]
    assert all(isinstance(s.benchmark, BBH) for s in specs)
    assert specs[0].benchmark is not specs[1].benchmark
    assert specs[0].benchmark.tasks == ["boolean_expressions"]
    assert specs[1].benchmark.tasks == ["date_understanding"]


@pytest.mark.parametrize("key", ["g.v2", "gsm-8k", "gsm8k_alt", "G2"])
def test_build_specs_key_with_dots_hyphens(key):
    specs = build_benchmark_specs({key: _gsm8k_entry(name="gsm8k")})
    assert [s.key for s in specs] == [key]


def test_build_specs_with_test_local_benchmark():
    specs = build_benchmark_specs({LISTBENCH_NAME: {"golds": ["1", "2"]}}, seed=0)
    assert specs[0].key == LISTBENCH_NAME
    assert isinstance(specs[0].benchmark, ListBench)


# --- build_benchmark_specs errors ------------------------------------------

def test_build_specs_list_form_rejected_with_hint():
    with pytest.raises(ValueError, match="mapping keyed by benchmark name"):
        build_benchmark_specs([{"name": "gsm8k", "source": "local"}])


@pytest.mark.parametrize("configs", [
    {},
    {"gsm8k": ["source", "local"]},
    {"gsm8k": 3},
    {"gsm8k": "local"},
    {"bad key!": _gsm8k_entry(name="gsm8k")},
    {"a/b": _gsm8k_entry(name="gsm8k")},
], ids=["empty", "entry_list", "entry_int", "entry_str", "bad_key_space_bang", "slash_key"])
def test_build_specs_value_errors(configs):
    with pytest.raises(ValueError):
        build_benchmark_specs(configs)


@pytest.mark.parametrize("key,entry,needle", [
    ("gsm8k", {"source": "local", "dev_size": 4}, "'train'"),
    ("gsm8k", {"source": "local", "val_size": 2}, "'val'"),
    ("bbh", {"source": "local", "heldout_tasks": ["web_of_lies"]}, "'heldout'"),
    ("gsm8k", {"source": "local", "splits": {"search": "dev"}}, "--split"),
    ("gsm8k", {"source": "local", "id": "g2"}, "'name'"),
], ids=["dev_size", "val_size", "heldout_tasks", "splits", "id"])
def test_build_specs_old_option_names_new_key(key, entry, needle):
    with pytest.raises(ValueError) as ei:
        build_benchmark_specs({key: entry})
    msg = str(ei.value)
    assert needle in msg, msg
    assert key in msg, msg


def test_build_specs_old_key_mixed_with_valid_still_raises():
    with pytest.raises(ValueError, match="'train'"):
        build_benchmark_specs({"gsm8k": _gsm8k_entry(train=4, dev_size=4)})


def test_build_specs_old_key_message_names_alias_key():
    with pytest.raises(ValueError, match="g2"):
        build_benchmark_specs({"g2": _gsm8k_entry(name="gsm8k", val_size=3)})


def test_build_specs_unknown_benchmark_name():
    with pytest.raises(KeyError):
        build_benchmark_specs({uniq("nobench"): None})


# --- AC3: Evaluator construction -------------------------------------------

def test_evaluator_requires_benchmarks():
    with pytest.raises(ValueError):
        Evaluator(make_mock(), [])


def test_evaluator_batch_size_zero():
    with pytest.raises(ValueError):
        Evaluator(make_mock(), [_spec("a", ["1"])], batch_size=0)


def test_evaluator_duplicate_keys():
    with pytest.raises(ValueError):
        Evaluator(make_mock(), [_spec("a", ["1"]), _spec("a", ["2"])])


def test_evaluator_benchmark_keys_order_and_llm():
    mock = make_mock()
    ev = Evaluator(mock, [_spec("zeta", ["1"]), _spec("alpha", ["2"]), _spec("mid", ["3"])])
    assert ev.benchmark_keys == ["zeta", "alpha", "mid"]
    assert ev.llm is mock


# --- AC4: evaluate across benchmarks ---------------------------------------

def test_evaluate_two_benchmarks_raw_results():
    mock, ev = _multi([_spec("a", ["3", "4"]), _spec("b", ["3"])])
    out = ev.evaluate_skill(get_baseline("cot"), "train")
    assert isinstance(out, SkillEvaluation)
    assert out.skill_id == "cot"
    assert out.split == "train"
    assert out.benchmarks == ["a", "b"]
    assert list(out.results) == ["a", "b"]
    ra, rb = out.results["a"], out.results["b"]
    assert isinstance(ra, EvalResult) and isinstance(rb, EvalResult)
    assert ra.benchmark == "a" and rb.benchmark == "b"
    assert ra.split == "train" and rb.split == "train"
    assert ra.skill_id == "cot" and rb.skill_id == "cot"
    assert ra.accuracy == pytest.approx(0.5)
    assert rb.accuracy == pytest.approx(1.0)
    assert [it.example_id for it in ra.items] == ["ex0", "ex1"]
    assert [it.prediction for it in ra.items] == ["3", "3"]
    assert [it.gold for it in ra.items] == ["3", "4"]
    assert [it.correct for it in ra.items] == [True, False]
    for it in ra.items + rb.items:
        assert isinstance(it, ItemResult)
        assert it.usage.reasoning_tokens == 3
        assert it.usage.output_tokens == 3 + mock.count_tokens("The answer is 3.")
        assert it.reasoning_text == "a b c"
        assert it.response_text == "The answer is 3."
        assert it.question.startswith("Question number")


def test_evaluate_requested_split_on_every_benchmark():
    specs = [_spec("a", ["3"]), _spec("b", ["3"])]
    _, ev = _multi(specs)
    for split in ("train", "val", "test"):
        out = ev.evaluate_skill(get_baseline("cot"), split)
        assert out.split == split
        assert out.results["a"].split == split
        assert out.results["b"].split == split
    assert specs[0].benchmark.loaded == ["train", "val", "test"]
    assert specs[1].benchmark.loaded == ["train", "val", "test"]


def test_evaluate_averages_are_item_means():
    _, ev = _multi([_spec("a", ["3", "4", "5"]), _spec("b", ["1", "2"])], responses=None)
    out = ev.evaluate_skill(get_baseline("direct"), "train")
    for res in out.results.values():
        items = res.items
        assert res.avg_output_tokens == pytest.approx(mean(i.usage.output_tokens for i in items))
        assert res.avg_input_tokens == pytest.approx(mean(i.usage.input_tokens for i in items))
        assert res.avg_reasoning_tokens == pytest.approx(mean(i.usage.reasoning_tokens for i in items))
        assert res.avg_total_tokens == pytest.approx(mean(i.usage.total_tokens for i in items))
        assert res.avg_latency == pytest.approx(mean(i.latency for i in items))
        assert res.avg_total_tokens == pytest.approx(res.avg_input_tokens + res.avg_output_tokens)
        assert res.avg_input_tokens > 0


def test_evaluate_zero_example_benchmark_alongside_nonempty():
    mock, ev = _multi([_spec("empty", []), _spec("full", ["3", "3"])])
    out = ev.evaluate_skill(get_baseline("cot"), "train")
    assert out.benchmarks == ["empty", "full"]
    assert out.results["empty"].n == 0
    assert out.results["empty"].accuracy == 0.0
    assert out.results["full"].n == 2
    assert mock.usage.snapshot().calls == 2


# --- AC5: SkillEvaluation is raw only --------------------------------------

def test_skill_evaluation_has_no_aggregate():
    _, ev = _multi([_spec("a", ["3"]), _spec("b", ["4"])])
    out = ev.evaluate_skill(get_baseline("cot"), "train")
    for attr in ("accuracy", "score", "tokens", "avg_output_tokens"):
        assert not hasattr(out, attr), attr


def test_skill_evaluation_summary():
    _, ev = _multi([_spec("a", ["3", "4"]), _spec("b", ["3"])])
    out = ev.evaluate_skill(get_baseline("cot"), "train")
    s = out.summary()
    assert set(s) == {"skill_id", "split", "benchmarks"}
    assert s["skill_id"] == "cot"
    assert s["split"] == "train"
    assert list(s["benchmarks"]) == ["a", "b"]
    for key in ("a", "b"):
        assert s["benchmarks"][key] == out.results[key].summary()
        assert set(s["benchmarks"][key]) == EVAL_SUMMARY_KEYS


def test_skill_evaluation_constructed_directly():
    r1 = make_result("s", 4, 2, benchmark="k1")
    r2 = make_result("s", 2, 2, benchmark="k2")
    sk = make_skill("s")
    se = SkillEvaluation(skill=sk, split="val", results={"k1": r1, "k2": r2})
    assert se.skill is sk
    assert se.split == "val"
    assert se.summary()["split"] == "val"
    assert se.skill_id == "s"
    assert se.benchmarks == ["k1", "k2"]
    assert se.summary()["benchmarks"]["k2"]["accuracy"] == pytest.approx(1.0)


# --- AC6: limits -------------------------------------------------------------

def test_evaluate_limit_applies_to_every_benchmark():
    specs = [_spec("a", ["3", "4", "5", "6"]), _spec("b", ["3", "4", "5", "6", "7"])]
    mock, ev = _multi(specs)
    out = ev.evaluate_skill(get_baseline("cot"), "train", limit=3)
    assert out.results["a"].n == 3
    assert out.results["b"].n == 3
    assert [it.example_id for it in out.results["b"].items] == ["ex0", "ex1", "ex2"]
    assert mock.usage.snapshot().calls == 6


def test_evaluate_limit_zero_no_calls():
    mock, ev = _multi([_spec("a", ["3", "4"]), _spec("b", ["5"])])
    out = ev.evaluate_skill(get_baseline("cot"), "train", limit=0)
    assert out.results["a"].n == 0
    assert out.results["b"].n == 0
    assert mock.usage.snapshot().calls == 0
    assert mock.calls == []


def test_call_count_equals_items_evaluated():
    mock, ev = _multi([_spec("a", ["3", "4"]), _spec("b", ["5", "6", "7"])])
    before = mock.usage.snapshot().calls
    out = ev.evaluate_skill(get_baseline("cot"), "train")
    total = sum(r.n for r in out.results.values())
    assert total == 5
    assert mock.usage.snapshot().calls - before == total
    ev.evaluate_skill(get_baseline("cot"), "train", limit=1)
    assert mock.usage.snapshot().calls - before == total + 2


# --- removed stage machinery / split skip rule -------------------------------

def test_stage_machinery_removed():
    import skillopt.evaluator as ev_mod

    assert not hasattr(ev_mod, "SplitSpec")
    assert not hasattr(ev_mod, "DEFAULT_SPLITS")
    assert not hasattr(Evaluator, "resolve_split")
    with pytest.raises(ImportError):
        from skillopt.evaluator import SplitSpec  # noqa: F401
    with pytest.raises(ImportError):
        from skillopt.evaluator import DEFAULT_SPLITS  # noqa: F401


def test_benchmark_spec_has_only_key_and_benchmark():
    import dataclasses

    spec = _spec("a", ["1"])
    assert [f.name for f in dataclasses.fields(BenchmarkSpec)] == ["key", "benchmark"]
    assert not hasattr(spec, "splits")
    with pytest.raises(TypeError):
        BenchmarkSpec(key="a", benchmark=ListBench(["1"]), splits={})


def test_skill_evaluation_has_no_stage():
    import dataclasses

    names = [f.name for f in dataclasses.fields(SkillEvaluation)]
    assert "split" in names and "stage" not in names
    with pytest.raises(TypeError):
        SkillEvaluation(skill=make_skill("s"), stage="search", results={})


def test_heldout_skips_benchmarks_without_it():
    mock = make_mock()
    ev = Evaluator(mock, _sample_specs())
    out = ev.evaluate_skill(get_baseline("cot"), "heldout")
    assert out.split == "heldout"
    assert list(out.results) == ["bbh"]
    assert out.benchmarks == ["bbh"]
    r = out.results["bbh"]
    assert r.split == "heldout"
    assert r.n == 8
    assert all(it.example_id.startswith("web_of_lies") for it in r.items)
    assert mock.usage.snapshot().calls == 8


def test_heldout_skip_rule_in_evaluate_list():
    ev = Evaluator(make_mock(), _sample_specs())
    out = ev.evaluate([get_baseline("cot"), get_baseline("direct")], "heldout", limit=2)
    for e in out:
        assert list(e.results) == ["bbh"]
        assert e.results["bbh"].n == 2


def test_heldout_only_gsm8k_raises():
    specs = build_benchmark_specs({"gsm8k": _gsm8k_entry()}, seed=0)
    mock = make_mock()
    ev = Evaluator(mock, specs)
    with pytest.raises(ValueError):
        ev.evaluate_skill(get_baseline("cot"), "heldout")
    with pytest.raises(ValueError):
        ev.evaluate([get_baseline("cot")], "heldout")
    assert mock.usage.snapshot().calls == 0


def test_unknown_split_on_all_benchmarks_raises():
    mock = make_mock()
    ev = Evaluator(mock, _sample_specs())
    with pytest.raises(ValueError):
        ev.evaluate_skill(get_baseline("cot"), "nope")
    with pytest.raises(ValueError):
        ev.evaluate_skill(get_baseline("cot"), "dev")
    assert mock.usage.snapshot().calls == 0


def test_unknown_split_listbench_raises():
    _, ev = _multi([_spec("a", ["3"]), _spec("b", ["4"])])
    with pytest.raises(ValueError):
        ev.evaluate_skill(get_baseline("cot"), "heldout")


def test_evaluate_split_heldout_on_gsm8k_raises():
    ev = Evaluator(make_mock(), _sample_specs())
    with pytest.raises(ValueError):
        ev.evaluate_split(get_baseline("cot"), "gsm8k", "heldout")


def test_heldout_limit_zero():
    ev = Evaluator(make_mock(), _sample_specs())
    out = ev.evaluate_skill(get_baseline("cot"), "heldout", limit=0)
    assert list(out.results) == ["bbh"]
    assert out.results["bbh"].n == 0


def test_unknown_key_raises_key_error():
    _, ev = _eval(["3"])
    with pytest.raises(KeyError):
        ev.evaluate_split(get_baseline("cot"), "nope", "train")


# --- AC8: evaluate_split (old single-benchmark behavior) --------------------

def test_evaluate_split_accuracy_items_and_usage():
    mock, ev = _eval(["3", "4"])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train")
    assert isinstance(res, EvalResult)
    assert res.skill_id == "cot"
    assert res.benchmark == "lb"
    assert res.split == "train"
    assert res.n == 2
    assert res.accuracy == pytest.approx(0.5)
    assert [it.example_id for it in res.items] == ["ex0", "ex1"]
    assert [it.correct for it in res.items] == [True, False]
    for it in res.items:
        assert it.usage.output_tokens == 3 + mock.count_tokens("The answer is 3.")


def test_evaluate_split_limit_and_call_count():
    mock, ev = _eval(["3", "4", "5", "6", "7"])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train", limit=2)
    assert res.n == 2
    assert [it.example_id for it in res.items] == ["ex0", "ex1"]
    assert mock.usage.snapshot().calls == 2
    assert ev.evaluate_split(get_baseline("cot"), "lb", "train").n == 5
    assert mock.usage.snapshot().calls == 7


def test_evaluate_split_limit_zero():
    mock, ev = _eval(["3", "4"])
    assert ev.evaluate_split(get_baseline("cot"), "lb", "train", limit=0).n == 0
    assert mock.usage.snapshot().calls == 0


def test_batch_size_does_not_change_predictions():
    golds = [str(i) for i in range(7)]
    _, ev1 = _eval(golds, responses=None, batch_size=1, seed=0)
    _, ev16 = _eval(golds, responses=None, batch_size=16, seed=0)
    r1 = ev1.evaluate_split(get_baseline("cot"), "lb", "train")
    r16 = ev16.evaluate_split(get_baseline("cot"), "lb", "train")
    assert [i.prediction for i in r1.items] == [i.prediction for i in r16.items]
    assert [i.usage.output_tokens for i in r1.items] == [i.usage.output_tokens for i in r16.items]
    assert [i.example_id for i in r1.items] == [i.example_id for i in r16.items]


def test_batch_size_one_calls_per_item():
    mock, ev = _eval(["3", "4", "5"], batch_size=1)
    ev.evaluate_split(get_baseline("cot"), "lb", "train")
    assert mock.usage.snapshot().calls == 3


def test_evaluate_split_zero_examples():
    mock, ev = _eval([])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train")
    assert res.n == 0
    assert res.items == []
    assert res.accuracy == 0.0
    for attr in ("avg_output_tokens", "avg_input_tokens", "avg_reasoning_tokens",
                 "avg_total_tokens", "avg_latency"):
        assert getattr(res, attr) == 0.0, attr
    assert mock.usage.snapshot().calls == 0
    assert mock.calls == []


def test_evaluate_split_truncated_thinking():
    _, ev = _eval(["3"], responses=["<think>long never closed"])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train")
    it = res.items[0]
    assert it.prediction is None
    assert it.correct is False
    assert it.usage.output_tokens > 0
    assert res.accuracy == 0.0


def test_failures_returns_incorrect_in_order_at_most_k():
    _, ev = _eval(["3", "4", "5", "3", "6"])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train")
    assert [i.example_id for i in res.failures()] == ["ex1", "ex2", "ex4"]
    assert [i.example_id for i in res.failures(2)] == ["ex1", "ex2"]
    assert res.failures(0) == []
    assert all(not i.correct for i in res.failures())


def test_eval_result_summary_keys_and_values():
    _, ev = _eval(["3", "4"])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train")
    s = res.summary()
    assert set(s) == EVAL_SUMMARY_KEYS
    assert s["skill_id"] == "cot"
    assert s["benchmark"] == "lb"
    assert s["split"] == "train"
    assert s["n"] == 2
    assert s["accuracy"] == pytest.approx(0.5)
    assert s["avg_output_tokens"] == pytest.approx(res.avg_output_tokens)


def test_item_result_to_dict_usage_is_dict():
    _, ev = _eval(["3"])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train")
    d = res.items[0].to_dict()
    assert isinstance(d["usage"], dict)
    assert d["usage"]["reasoning_tokens"] == 3
    assert d["example_id"] == "ex0"
    assert d["prediction"] == "3"
    assert d["correct"] is True


def test_eval_result_constructed_directly():
    r = make_result("s", n=4, n_correct=1, output_tokens=10, input_tokens=5)
    assert r.accuracy == pytest.approx(0.25)
    assert r.avg_output_tokens == pytest.approx(10)
    assert r.avg_total_tokens == pytest.approx(15)
    assert len(r.failures()) == 3


# --- sample data through specs ----------------------------------------------

def test_sample_specs_evaluate_end_to_end():
    specs = _sample_specs()
    mock = make_mock()
    ev = Evaluator(mock, specs)
    out = ev.evaluate_skill(get_baseline("cot"), "train")
    assert out.benchmarks == ["gsm8k", "bbh"]
    assert out.results["gsm8k"].n == 4
    assert out.results["bbh"].n == 4
    assert all(not it.example_id.startswith("web_of_lies") for it in out.results["bbh"].items)
    assert mock.usage.snapshot().calls == 8


# --- loop 2026-11-10-003: evaluate(list) is THE fixed result format ---------------

def test_evaluate_list_two_skills_two_benchmarks():
    """AC1: one SkillEvaluation per skill, input order, raw per-benchmark EvalResults."""
    mock, ev = _multi([_spec("a", ["3", "4"]), _spec("b", ["3"])])
    s1, s2 = get_baseline("cot"), get_baseline("direct")
    out = ev.evaluate([s1, s2], "train")
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].skill is s1 and out[1].skill is s2
    assert [e.skill_id for e in out] == ["cot", "direct"]
    for e, sk in zip(out, (s1, s2)):
        assert isinstance(e, SkillEvaluation)
        assert e.skill_id == sk.id
        assert e.split == "train"
        assert list(e.results) == ["a", "b"]
        assert e.benchmarks == ["a", "b"]
        for key, res in e.results.items():
            assert isinstance(res, EvalResult)
            assert res.skill_id == sk.id
            assert res.benchmark == key
            items = res.items
            assert res.n == len(items)
            assert res.accuracy == pytest.approx(mean(1.0 if i.correct else 0.0 for i in items))
            assert res.avg_input_tokens == pytest.approx(mean(i.usage.input_tokens for i in items))
            assert res.avg_output_tokens == pytest.approx(mean(i.usage.output_tokens for i in items))
            assert res.avg_reasoning_tokens == pytest.approx(mean(i.usage.reasoning_tokens for i in items))
            assert res.avg_total_tokens == pytest.approx(mean(i.usage.total_tokens for i in items))
            for it in items:
                assert isinstance(it, ItemResult)
                assert it.prediction == "3"
                assert it.gold in ("3", "4")
                assert isinstance(it.correct, bool)
                assert it.question.startswith("Question number")
                assert it.response_text == "The answer is 3."
                assert it.reasoning_text == "a b c"
    assert out[0].results["a"].accuracy == pytest.approx(0.5)
    assert out[0].results["b"].accuracy == pytest.approx(1.0)
    assert mock.usage.snapshot().calls == 6


def test_evaluate_list_default_split_is_train():
    specs = [_spec("a", ["3"])]
    _, ev = _multi(specs)
    out = ev.evaluate([get_baseline("cot")])
    assert out[0].split == "train"
    assert out[0].results["a"].split == "train"
    assert specs[0].benchmark.loaded == ["train"]


def test_evaluate_list_val_split_and_limit():
    specs = [_spec("a", ["3", "4", "5"]), _spec("b", ["3", "4"])]
    mock, ev = _multi(specs)
    out = ev.evaluate([get_baseline("cot"), get_baseline("cod")], "val", limit=1)
    for e in out:
        assert e.split == "val"
        assert e.results["a"].split == "val"
        assert e.results["b"].split == "val"
        assert all(r.n == 1 for r in e.results.values())
    assert mock.usage.snapshot().calls == 4


def test_evaluate_list_val_limit_on_sample_data():
    ev = Evaluator(make_mock(), _sample_specs())
    out = ev.evaluate([get_baseline("cot")], "val", limit=1)
    for r in out[0].results.values():
        assert r.n <= 1
        assert r.split == "val"


def test_evaluate_list_accepts_tuple():
    _, ev = _multi([_spec("a", ["3"])])
    out = ev.evaluate((get_baseline("cot"), get_baseline("direct")), "train")
    assert [e.skill_id for e in out] == ["cot", "direct"]


def test_evaluate_list_same_as_evaluate_skill():
    _, ev1 = _multi([_spec("a", ["3", "4"]), _spec("b", ["5"])], responses=None, seed=0)
    _, ev2 = _multi([_spec("a", ["3", "4"]), _spec("b", ["5"])], responses=None, seed=0)
    batch = ev1.evaluate([get_baseline("cot")], "train")[0]
    single = ev2.evaluate_skill(get_baseline("cot"), "train")
    for key in ("a", "b"):
        assert [i.prediction for i in batch.results[key].items] == \
            [i.prediction for i in single.results[key].items]
        assert [i.usage.output_tokens for i in batch.results[key].items] == \
            [i.usage.output_tokens for i in single.results[key].items]


def test_evaluate_thinking_counted_in_output_tokens():
    """AC2: avg_output_tokens includes thinking tokens."""
    _, ev = _multi([_spec("a", ["3", "4"]), _spec("b", ["3"])])
    out = ev.evaluate([get_baseline("cot")], "train")
    for res in out[0].results.values():
        assert res.avg_reasoning_tokens > 0
        assert res.avg_output_tokens >= res.avg_reasoning_tokens
        assert res.avg_output_tokens > res.avg_reasoning_tokens  # visible answer adds tokens too


def test_evaluate_empty_list_no_calls():
    """AC3."""
    mock, ev = _multi([_spec("a", ["3"]), _spec("b", ["4"])])
    assert ev.evaluate([], "train") == []
    assert mock.calls == []
    assert mock.usage.snapshot().calls == 0


def test_evaluate_bare_skill_type_error():
    """AC3."""
    mock, ev = _multi([_spec("a", ["3"])])
    with pytest.raises(TypeError):
        ev.evaluate(get_baseline("cot"), "train")
    assert mock.calls == []


def test_evaluate_list_zero_example_benchmark():
    """Edge: benchmark with 0 examples alongside a non-empty one."""
    mock, ev = _multi([_spec("empty", []), _spec("full", ["3", "3"])])
    out = ev.evaluate([get_baseline("cot"), get_baseline("direct")], "train")
    for e in out:
        r = e.results["empty"]
        assert r.n == 0 and r.items == []
        for attr in ("accuracy", "avg_output_tokens", "avg_input_tokens", "avg_reasoning_tokens",
                     "avg_total_tokens"):
            assert getattr(r, attr) == 0.0, attr
        assert e.results["full"].n == 2
    assert mock.usage.snapshot().calls == 4


def test_evaluate_skill_default_split_train():
    """evaluate_skill default split is train."""
    _, ev = _multi([_spec("a", ["3"]), _spec("b", ["4"])])
    out = ev.evaluate_skill(get_baseline("cot"))
    assert isinstance(out, SkillEvaluation)
    assert out.split == "train"
    assert out.skill_id == "cot"
    assert out.skill.id == "cot"
    assert all(r.split == "train" for r in out.results.values())


def test_skill_evaluation_no_aggregate_attrs_on_list_result():
    """AC5."""
    _, ev = _multi([_spec("a", ["3"]), _spec("b", ["4"])])
    for e in ev.evaluate([get_baseline("cot")], "train"):
        for attr in ("accuracy", "score", "tokens", "avg_output_tokens"):
            assert not hasattr(e, attr), attr
        s = e.summary()
        assert set(s) == {"skill_id", "split", "benchmarks"}
        assert s["benchmarks"] == {k: r.summary() for k, r in e.results.items()}


# --- loop 2026-11-11-003: truncation fields, per-example hooks -------------------

_ITEM_KEYS_BEFORE = {"example_id", "prediction", "gold", "correct", "usage", "latency",
                     "response_text", "reasoning_text", "question"}


def test_item_result_truncated_field_default_and_order():
    import dataclasses

    from skillopt.evaluator import ItemResult
    from skillopt.llm import TokenUsage

    names = [f.name for f in dataclasses.fields(ItemResult)]
    assert "truncated" in names
    assert names.index("truncated") == names.index("question") + 1
    it = ItemResult("e", "1", "1", True, TokenUsage(), 0.0, "r", "")
    assert it.truncated is False
    d = it.to_dict()
    assert set(d) == _ITEM_KEYS_BEFORE | {"truncated"}
    assert d["truncated"] is False
    it2 = ItemResult("e", "1", "1", True, TokenUsage(), 0.0, "r", "", "Q?", True)
    assert it2.question == "Q?"
    assert it2.to_dict()["truncated"] is True


def test_eval_result_truncation_counts():
    r = make_result("s", n=4, n_correct=2)
    r.items[0].truncated = True
    r.items[3].truncated = True
    assert r.num_truncated == 2
    assert r.truncation_rate == pytest.approx(0.5)
    s = r.summary()
    assert s["num_truncated"] == 2
    assert s["truncation_rate"] == pytest.approx(0.5)


def test_eval_result_truncation_zero_items():
    r = EvalResult(skill_id="s", benchmark="b", split="train", items=[])
    assert r.num_truncated == 0
    assert r.truncation_rate == 0.0
    s = r.summary()
    assert s["num_truncated"] == 0
    assert s["truncation_rate"] == 0.0


def test_eval_result_summary_previous_values_unchanged():
    r = make_result("s", n=4, n_correct=1, output_tokens=10, input_tokens=5)
    s = r.summary()
    assert set(s) == EVAL_SUMMARY_KEYS
    assert s["skill_id"] == "s" and s["benchmark"] == "testbench" and s["split"] == "train"
    assert s["n"] == 4
    assert s["accuracy"] == pytest.approx(0.25)
    assert s["avg_input_tokens"] == pytest.approx(5)
    assert s["avg_output_tokens"] == pytest.approx(10)
    assert s["avg_reasoning_tokens"] == pytest.approx(0)
    assert s["avg_total_tokens"] == pytest.approx(15)
    assert s["avg_latency"] == pytest.approx(0.01)
    assert s["num_truncated"] == 0
    assert s["truncation_rate"] == 0.0


def test_evaluate_split_copies_truncated_flag():
    _, ev = _eval(["3", "3"], responses=["<think>long never closed", "3"])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train")
    assert [it.truncated for it in res.items] == [True, False]
    assert res.num_truncated == 1
    assert res.truncation_rate == pytest.approx(0.5)
    assert [it.to_dict()["truncated"] for it in res.items] == [True, False]


class _ForBench(ListBench):
    """Overrides the per-example hooks and records what it received."""

    def __init__(self, golds=(), **kw):
        super().__init__(golds, **kw)
        self.seen = []

    def answer_format_for(self, example):
        return f"FORMAT-FOR-{example.id}"

    def extract_answer_for(self, text, example):
        self.seen.append(example)
        return example.answer  # always "correct"


def test_extract_answer_default_hook_delegates():
    b = ListBench(["7"])
    ex = b.load("train")[0]
    assert b.extract_answer_for("the answer is 7", ex) == "7"
    assert b.extract_answer_for("nothing", ex) is None


def test_answer_format_for_default_and_build_prompt():
    b = ListBench(["7"])
    ex = b.load("train")[0]
    assert b.answer_format_for(ex) == ListBench.answer_format
    assert ListBench.answer_format in b.build_prompt(get_baseline("cot"), ex)


def test_build_prompt_uses_answer_format_for_override():
    b = _ForBench(["7"])
    ex = b.load("train")[0]
    prompt = b.build_prompt(get_baseline("cot"), ex)
    assert "FORMAT-FOR-ex0" in prompt
    assert ListBench.answer_format not in prompt


def test_evaluator_uses_extract_answer_for_with_example():
    b = _ForBench(["11", "12"])
    mock = make_mock(["no number here"])
    ev = Evaluator(mock, [BenchmarkSpec(key="fb", benchmark=b)])
    res = ev.evaluate_split(get_baseline("cot"), "fb", "train")
    assert [e.id for e in b.seen] == ["ex0", "ex1"]
    assert all(isinstance(e, Example) for e in b.seen)
    assert [it.prediction for it in res.items] == ["11", "12"]
    assert res.accuracy == pytest.approx(1.0)
    assert all("FORMAT-FOR-" in p for p in mock.calls)


def test_evaluator_custom_benchmark_with_only_extract_answer_still_works():
    _, ev = _eval(["3", "4"])
    res = ev.evaluate_split(get_baseline("cot"), "lb", "train")
    assert [it.prediction for it in res.items] == ["3", "3"]
    assert [it.correct for it in res.items] == [True, False]


def test_evaluator_bbh_sample_answer_tag_option_scored_correct():
    mock = make_mock(["So the answer is <answer>(F)</answer>."])
    ev = Evaluator(mock, build_benchmark_specs({"bbh": _bbh_entry()}, seed=0))
    hits = []
    for split in ("train", "val", "test"):
        res = ev.evaluate_split(get_baseline("cot"), "bbh", split)
        hits += [it for it in res.items if it.example_id.startswith("date_understanding") and it.gold == "f"]
    assert hits, "expected a date_understanding sample with gold (F)"
    for it in hits:
        assert it.prediction == "f"
        assert it.correct is True
