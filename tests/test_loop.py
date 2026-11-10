"""E. The plain runner: optimizer.initial_prompts() -> evaluate -> optimizer.update(...) -> ... (AC11-16)."""
import re

import pytest

from helpers import make_mock, make_result, make_skill, read_json, read_jsonl
from skillopt.benchmarks import Benchmark, Example
from skillopt.evaluator import BenchmarkSpec, Evaluator, SkillEvaluation
from skillopt.optimizer import TRAIN_SPLIT, LoopResult, Optimizer, optimize
from skillopt.runlog import RunDir
from skillopt.skills import Skill

KEYS = ("k1", "k2")


# --- fakes ---------------------------------------------------------------------

class FakeEvaluator:
    """Any object with evaluate(skills, split) returning SkillEvaluations built from helpers.make_items."""

    def __init__(self, keys=KEYS):
        self.keys = list(keys)
        self.calls = []  # (list of skill ids, split)
        self.limits = []

    def evaluate(self, skills, split="train", limit=None):
        skills = list(skills)
        self.calls.append(([s.id for s in skills], split))
        self.limits.append(limit)
        out = []
        for s in skills:
            results = {k: make_result(s.id, 4, 2, benchmark=k, split=split) for k in self.keys}
            out.append(SkillEvaluation(skill=s, split=split, results=results))
        return out


def _skills(prefix, n):
    return [make_skill(f"{prefix}-{i}") for i in range(n)]


class Scripted(Optimizer):
    """initial_prompts() returns `initial`; the k-th update() returns script[k] (last entry repeated)."""

    def __init__(self, llm=None, initial=None, script=(), on_update=None, **params):
        super().__init__(llm, **params)
        self.initial = initial if initial is not None else _skills("init", 2)
        self.script = list(script)
        self.on_update = on_update
        self.initial_calls = 0
        self.updates = []  # (results ids, [[ids] per history round], len(history))
        self.raw_updates = []  # (results, history) as received

    def initial_prompts(self):
        self.initial_calls += 1
        return self.initial

    def update(self, results, history):
        self.raw_updates.append((results, history))
        self.updates.append(([r.skill_id for r in results],
                             [[e.skill_id for e in rnd] for rnd in history], len(history)))
        if self.on_update is not None:
            self.on_update(results, history)
        k = len(self.updates) - 1
        if not self.script:
            return []
        return self.script[min(k, len(self.script) - 1)]


class Forever(Scripted):
    """update() always returns a fresh skill."""

    def update(self, results, history):
        super().update(results, history)
        return [make_skill(f"gen-r{len(history) + 1}")]


def _round_ids(res):
    return [[e.skill_id for e in rnd] for rnd in res.history]


# --- AC11 ------------------------------------------------------------------------

def test_stops_on_empty_update():
    opt = Scripted(script=[_skills("a", 1), _skills("b", 2), []])
    ev = FakeEvaluator()
    res = optimize(opt, ev, max_iterations=10)
    assert isinstance(res, LoopResult)
    assert res.stop_reason == "empty_update"
    assert len(res.history) == 3
    assert len(opt.updates) == 3
    assert _round_ids(res) == [["init-0", "init-1"], ["a-0"], ["b-0", "b-1"]]
    assert opt.initial_calls == 1
    assert len(ev.calls) == 3


def test_empty_update_on_first_call():
    opt = Scripted(script=[[]])
    res = optimize(opt, FakeEvaluator(), max_iterations=5)
    assert res.stop_reason == "empty_update"
    assert len(res.history) == 1
    assert len(opt.updates) == 1


def test_history_entries_are_skill_evaluations():
    opt = Scripted(script=[_skills("a", 1), []])
    res = optimize(opt, FakeEvaluator(), max_iterations=3)
    for rnd in res.history:
        assert isinstance(rnd, list)
        for e in rnd:
            assert isinstance(e, SkillEvaluation)
            assert list(e.results) == list(KEYS)


# --- AC12 ------------------------------------------------------------------------

def test_max_iterations_caps_update_calls():
    opt = Forever()
    ev = FakeEvaluator()
    res = optimize(opt, ev, max_iterations=2)
    assert len(opt.updates) == 2
    assert len(res.history) == 3
    assert res.stop_reason == "max_iterations"
    assert len(ev.calls) == 3
    assert _round_ids(res) == [["init-0", "init-1"], ["gen-r1"], ["gen-r2"]]


def test_max_iterations_zero_only_initial():
    opt = Forever()
    ev = FakeEvaluator()
    res = optimize(opt, ev, max_iterations=0)
    assert opt.updates == []
    assert len(res.history) == 1
    assert res.stop_reason == "max_iterations"
    assert ev.calls == [(["init-0", "init-1"], "train")]


def test_max_iterations_one():
    opt = Forever()
    res = optimize(opt, FakeEvaluator(), max_iterations=1)
    assert len(opt.updates) == 1
    assert len(res.history) == 2
    assert res.stop_reason == "max_iterations"


# --- AC13 ------------------------------------------------------------------------

def test_update_receives_results_and_earlier_history():
    opt = Scripted(initial=_skills("init", 3), script=[_skills("a", 2), _skills("b", 1), []])
    res = optimize(opt, FakeEvaluator(), max_iterations=10)
    assert opt.updates[0] == (["init-0", "init-1", "init-2"], [], 0)
    assert opt.updates[1] == (["a-0", "a-1"], [["init-0", "init-1", "init-2"]], 1)
    assert opt.updates[2] == (["b-0"], [["init-0", "init-1", "init-2"], ["a-0", "a-1"]], 2)
    for i, (results, _history) in enumerate(opt.raw_updates):
        assert [e.skill_id for e in results] == [e.skill_id for e in res.history[i]]
        assert all(isinstance(e, SkillEvaluation) for e in results)


def test_update_results_order_matches_prompts():
    order = [make_skill("zz"), make_skill("aa"), make_skill("mm")]
    opt = Scripted(initial=order, script=[[]])
    optimize(opt, FakeEvaluator(), max_iterations=3)
    results, _ = opt.raw_updates[0]
    assert [e.skill_id for e in results] == ["zz", "aa", "mm"]
    assert [e.skill for e in results] == order


def test_mutating_history_does_not_change_loop_result():
    def vandal(results, history):
        history.clear()
        history.append(["garbage"])

    opt = Scripted(script=[_skills("a", 1), _skills("b", 1), []], on_update=vandal)
    res = optimize(opt, FakeEvaluator(), max_iterations=10)
    assert _round_ids(res) == [["init-0", "init-1"], ["a-0"], ["b-0"]]
    # every call still saw the correct earlier rounds (recorded before mutating)
    assert [u[2] for u in opt.updates] == [0, 1, 2]
    assert opt.updates[2][1] == [["init-0", "init-1"], ["a-0"]]


# --- AC14 ------------------------------------------------------------------------

def test_on_round_called_per_round_in_order():
    seen = []
    opt = Scripted(script=[_skills("a", 1), _skills("b", 2), []])
    res = optimize(opt, FakeEvaluator(), max_iterations=10,
                   on_round=lambda i, evs: seen.append((i, [e.skill_id for e in evs])))
    assert seen == [(0, ["init-0", "init-1"]), (1, ["a-0"]), (2, ["b-0", "b-1"])]
    assert [s[1] for s in seen] == _round_ids(res)


def test_on_round_with_max_iterations():
    seen = []
    optimize(Forever(), FakeEvaluator(), max_iterations=2, on_round=lambda i, evs: seen.append(i))
    assert seen == [0, 1, 2]


def test_on_round_called_before_update():
    events = []
    opt = Scripted(script=[_skills("a", 1), []],
                   on_update=lambda results, history: events.append(("update", len(history))))
    optimize(opt, FakeEvaluator(), max_iterations=5, on_round=lambda i, evs: events.append(("round", i)))
    assert events == [("round", 0), ("update", 0), ("round", 1), ("update", 1)]


# --- AC15 ------------------------------------------------------------------------

def test_train_split_constant():
    assert TRAIN_SPLIT == "train"


def test_search_stage_removed():
    import skillopt.optimizer as opt_pkg
    import skillopt.optimizer.loop as loop_mod

    assert not hasattr(opt_pkg, "SEARCH_STAGE")
    assert not hasattr(loop_mod, "SEARCH_STAGE")
    with pytest.raises(ImportError):
        from skillopt.optimizer import SEARCH_STAGE  # noqa: F401


def test_loop_always_evaluates_train_split():
    ev = FakeEvaluator()
    res = optimize(Scripted(script=[_skills("a", 1), _skills("b", 2), []]), ev, max_iterations=5)
    assert [c[1] for c in ev.calls] == ["train", "train", "train"]
    assert ev.limits == [None, None, None]
    assert all(e.split == "train" for rnd in res.history for e in rnd)
    assert all(r.split == "train" for rnd in res.history for e in rnd for r in e.results.values())


def test_stage_kwarg_removed():
    ev = FakeEvaluator()
    with pytest.raises(TypeError):
        optimize(Scripted(script=[_skills("a", 1), []]), ev, max_iterations=3, stage="val")


def test_split_kwarg_not_accepted():
    ev = FakeEvaluator()
    with pytest.raises(TypeError):
        optimize(Scripted(script=[_skills("a", 1), []]), ev, max_iterations=3, split="val")


# --- AC16 ------------------------------------------------------------------------

def test_initial_empty_raises_value_error():
    with pytest.raises(ValueError):
        optimize(Scripted(initial=[]), FakeEvaluator(), max_iterations=2)


@pytest.mark.parametrize("bad", [["not a skill"], [None], [{"id": "x", "instruction": "y"}]])
def test_initial_non_skill_type_error(bad):
    with pytest.raises(TypeError):
        optimize(Scripted(initial=bad), FakeEvaluator(), max_iterations=2)


def test_initial_not_a_list_type_error():
    with pytest.raises(TypeError):
        optimize(Scripted(initial=make_skill("solo")), FakeEvaluator(), max_iterations=2)


def test_update_non_skill_type_error():
    opt = Scripted(script=[[make_skill("ok"), "nope"]])
    with pytest.raises(TypeError):
        optimize(opt, FakeEvaluator(), max_iterations=3)


def test_initial_duplicate_ids_value_error():
    with pytest.raises(ValueError):
        optimize(Scripted(initial=[make_skill("dup"), make_skill("dup", "Other.")]), FakeEvaluator(),
                 max_iterations=2)


def test_update_duplicate_ids_value_error():
    opt = Scripted(script=[[make_skill("d"), make_skill("d", "Different.")]])
    with pytest.raises(ValueError):
        optimize(opt, FakeEvaluator(), max_iterations=3)


@pytest.mark.parametrize("bad", [-1, True, False, "2", None, 1.5])
def test_invalid_max_iterations_before_initial_prompts(bad):
    opt = Scripted()
    ev = FakeEvaluator()
    with pytest.raises(ValueError):
        optimize(opt, ev, max_iterations=bad)
    assert opt.initial_calls == 0
    assert ev.calls == []


def test_same_id_in_different_rounds_allowed():
    opt = Scripted(initial=[make_skill("same")],
                   script=[[make_skill("same", "Rewritten.")], [make_skill("same")], []])
    res = optimize(opt, FakeEvaluator(), max_iterations=10)
    assert _round_ids(res) == [["same"], ["same"], ["same"]]
    assert res.history[1][0].skill.instruction == "Rewritten."
    assert res.stop_reason == "empty_update"


def test_tuples_accepted():
    opt = Scripted(initial=tuple(_skills("init", 2)), script=[tuple(_skills("a", 1)), ()])
    res = optimize(opt, FakeEvaluator(), max_iterations=10)
    assert _round_ids(res) == [["init-0", "init-1"], ["a-0"]]
    assert res.stop_reason == "empty_update"


# --- integration with the real Evaluator + mock LLM -----------------------------

class _ListBench(Benchmark):
    name = "runnerlistbench"
    answer_format = "Answer with a number."

    def __init__(self, golds=(), **kw):
        self._examples = [Example(id=f"ex{i}", question=f"Runner question {i}?", answer=str(g))
                          for i, g in enumerate(golds)]

    def load(self, split):
        if split not in self.splits:
            raise ValueError(split)
        return list(self._examples)

    def extract_answer(self, text):
        nums = re.findall(r"-?\d+", text or "")
        return nums[-1] if nums else None

    def score(self, pred, gold):
        return pred is not None and pred == gold


def test_optimize_with_real_evaluator():
    specs = [BenchmarkSpec("alpha", _ListBench(["3", "4"])),
             BenchmarkSpec("beta", _ListBench(["3", "3", "5"]))]
    target = make_mock(["<think>a b c</think>The answer is 3."])
    ev = Evaluator(target, specs)
    opt = Scripted(llm=make_mock(["x"], role="optimizer"),
                   initial=[make_skill("s0"), make_skill("s1")],
                   script=[[make_skill("s2")], []])
    res = optimize(opt, ev, max_iterations=5)
    assert res.stop_reason == "empty_update"
    assert _round_ids(res) == [["s0", "s1"], ["s2"]]
    for rnd in res.history:
        for e in rnd:
            assert e.split == "train"
            assert list(e.results) == ["alpha", "beta"]
            assert e.results["alpha"].split == "train"
            assert e.results["beta"].split == "train"
            assert e.results["alpha"].accuracy == pytest.approx(0.5)
            assert e.results["beta"].accuracy == pytest.approx(2 / 3)
    assert target.usage.snapshot().calls == 3 * 5


# --- RunDir.round_dir / save_round ---------------------------------------------

def _eval_for(skill, keys=("gsm8k", "bbh", "empty"), split="train"):
    results = {}
    for k in keys:
        n = 0 if k == "empty" else 3
        results[k] = make_result(skill.id, n, min(n, 2), benchmark=k, split=split)
    return SkillEvaluation(skill=skill, split=split, results=results)


def test_round_dir_naming(tmp_path):
    rd = RunDir(tmp_path, "r", {})
    assert rd.round_dir(0) == rd.path / "rounds" / "000"
    assert rd.round_dir(7) == rd.path / "rounds" / "007"
    assert rd.round_dir(123) == rd.path / "rounds" / "123"


def test_save_round_writes_files(tmp_path):
    rd = RunDir(tmp_path, "r", {})
    s1 = make_skill("sk-a", "Alpha.")
    s2 = make_skill("sk-b", "Beta.")
    evs = [_eval_for(s1), _eval_for(s2)]
    out = rd.save_round(0, evs)
    d = rd.path / "rounds" / "000"
    assert out == d
    for sk, e in zip((s1, s2), evs):
        assert read_json(d / "skills" / f"{sk.id}.json") == sk.to_dict()
        assert Skill.from_dict(read_json(d / "skills" / f"{sk.id}.json")) == sk
        for key, r in e.results.items():
            jl = d / "results" / key / f"{sk.id}.jsonl"
            sm = d / "results" / key / f"{sk.id}.summary.json"
            assert jl.is_file(), jl
            assert read_json(sm) == r.summary()
            assert read_jsonl(jl) == [it.to_dict() for it in r.items]
    summary = read_json(d / "summary.json")
    assert summary == [e.summary() for e in evs]
    for entry in summary:
        assert set(entry) == {"skill_id", "split", "benchmarks"}
        assert entry["split"] == "train"
    assert [s["skill_id"] for s in summary] == ["sk-a", "sk-b"]
    assert not (rd.path / "skills").exists()
    assert not (rd.path / "results").exists()


def test_save_round_same_id_separate_dirs(tmp_path):
    rd = RunDir(tmp_path, "r", {})
    rd.save_round(0, [_eval_for(make_skill("same", "First."))])
    rd.save_round(1, [_eval_for(make_skill("same", "Second."))])
    assert read_json(rd.round_dir(0) / "skills" / "same.json")["instruction"] == "First."
    assert read_json(rd.round_dir(1) / "skills" / "same.json")["instruction"] == "Second."
