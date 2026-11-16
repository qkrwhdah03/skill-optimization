"""SkillOrigin, skill_text, skill size in tokens, runlog num_tokens, OPRO origin, cli report wiring (loop 2026-11-16-001)."""
import logging
import re
from pathlib import Path

import pytest

from helpers import CONFIGS, PROJECT_ROOT, element_by_id, find_elements, make_evaluation, make_mock, \
    make_result, make_skill, read_json, uniq
from skillopt.baselines import get_baseline
from skillopt.benchmarks import Benchmark, Example
from skillopt.cli import main
from skillopt.evaluator import BenchmarkSpec, Evaluator, SkillEvaluation
from skillopt.optimizer import MyOptimizer, OPROOptimizer, Optimizer, register_optimizer
from skillopt.runlog import RunDir
from skillopt.skills import FewShotExample, Skill, SkillOrigin, render_prompt, skill_text

CONFIG = CONFIGS / "config.yaml"
OPRO = "optimizer.name=opro"
REFRESH_META = '<meta http-equiv="refresh" content="5">'


# --- logger hygiene (main() configures the skillopt logger) -----------------------

def _snapshot(logger):
    return logger.level, list(logger.handlers), logger.propagate


def _restore(logger, snap):
    level, handlers, propagate = snap
    logger.setLevel(level)
    for h in list(logger.handlers):
        if h not in handlers:
            logger.removeHandler(h)
    for h in handlers:
        if h not in logger.handlers:
            logger.addHandler(h)
    logger.propagate = propagate


@pytest.fixture(autouse=True)
def _restore_skillopt_logger():
    logger = logging.getLogger("skillopt")
    snap = _snapshot(logger)
    yield
    _restore(logger, snap)


# --- AC1-4: SkillOrigin ---------------------------------------------------------------

def test_origin_to_dict():
    o = SkillOrigin(parents=["a", "b"], operation="mutate", note="x")
    assert o.to_dict() == {"parents": ["a", "b"], "operation": "mutate", "note": "x"}


def test_origin_roundtrip():
    o = SkillOrigin(parents=["a", "b-1.v2"], operation="crossover", note="merge two")
    assert SkillOrigin.from_dict(o.to_dict()) == o


def test_origin_default_note():
    assert SkillOrigin(parents=[], operation="initial").note == ""
    o = SkillOrigin.from_dict({"parents": [], "operation": "initial"})
    assert o.note == ""
    assert o.parents == []
    assert o.operation == "initial"


def test_origin_tuple_parents_stored_as_list():
    o = SkillOrigin(parents=("a", "b"), operation="mutate")
    assert o.parents == ["a", "b"]
    assert isinstance(o.parents, list)
    assert o.to_dict()["parents"] == ["a", "b"]


@pytest.mark.parametrize("bad", ["a", None, {"a"}, 3, {"a": 1}])
def test_origin_parents_not_list(bad):
    with pytest.raises(TypeError):
        SkillOrigin(parents=bad, operation="mutate")


@pytest.mark.parametrize("bad", [[1], [None], ["a", 2], [["a"]]])
def test_origin_parent_not_str(bad):
    with pytest.raises(TypeError):
        SkillOrigin(parents=bad, operation="mutate")


@pytest.mark.parametrize("bad", [["bad id"], [""], ["a/b"], ["ok", "q?"]])
def test_origin_parent_invalid_id(bad):
    with pytest.raises(ValueError):
        SkillOrigin(parents=bad, operation="mutate")


@pytest.mark.parametrize("bad", [None, 1, ["mutate"]])
def test_origin_operation_not_str(bad):
    with pytest.raises(TypeError):
        SkillOrigin(parents=[], operation=bad)


@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_origin_operation_blank(bad):
    with pytest.raises(ValueError):
        SkillOrigin(parents=[], operation=bad)


@pytest.mark.parametrize("bad", [None, 1, ["x"]])
def test_origin_note_not_str(bad):
    with pytest.raises(TypeError):
        SkillOrigin(parents=[], operation="mutate", note=bad)


# --- AC5-7: Skill.origin ----------------------------------------------------------------

def test_skill_origin_default_none():
    assert Skill(id="s", instruction="x").origin is None


@pytest.mark.parametrize("bad", [{"parents": []}, "initial", ["a"]])
def test_skill_origin_wrong_type(bad):
    with pytest.raises(TypeError):
        Skill(id="s", instruction="x", origin=bad)


def test_skill_to_dict_has_origin_key():
    d = Skill(id="s", instruction="x").to_dict()
    assert "origin" in d
    assert d["origin"] is None
    o = SkillOrigin(parents=["p"], operation="rewrite", note="n")
    d2 = Skill(id="s", instruction="x", origin=o).to_dict()
    assert d2["origin"] == {"parents": ["p"], "operation": "rewrite", "note": "n"}


def test_skill_with_origin_roundtrip():
    s = Skill(id="c1", instruction="Solve: {question}",
              few_shot=[FewShotExample(question="1+1?", answer="2")], parent_id="p",
              metadata={"k": [1]}, origin=SkillOrigin(parents=["p", "q"], operation="crossover", note="why"))
    s2 = Skill.from_dict(s.to_dict())
    assert s2 == s
    assert isinstance(s2.origin, SkillOrigin)
    assert s2.to_dict() == s.to_dict()


def test_skill_from_dict_missing_or_none_origin():
    assert Skill.from_dict({"id": "s", "instruction": "x"}).origin is None
    assert Skill.from_dict({"id": "s", "instruction": "x", "origin": None}).origin is None


def test_skill_from_dict_ignores_unknown_keys():
    s = Skill.from_dict({"id": "s", "instruction": "x", "num_tokens": 12, "whatever": True,
                         "origin": {"parents": [], "operation": "initial"}})
    assert s.id == "s"
    assert s.origin == SkillOrigin(parents=[], operation="initial", note="")


# --- AC8-9: skill_text ----------------------------------------------------------------------

def test_skill_text_with_few_shot():
    s = Skill(id="s", instruction="  Think.  ",
              few_shot=[FewShotExample(question="q1", answer="a1"), FewShotExample(question="q2", answer="a2")])
    assert skill_text(s) == "Think.\n\nQ: q1\nA: a1\n\nQ: q2\nA: a2"


def test_skill_text_empty():
    assert skill_text(Skill(id="s", instruction="")) == ""


def test_skill_text_instruction_only():
    assert skill_text(Skill(id="s", instruction="\n  Be brief.\n ")) == "Be brief."


def test_skill_text_keeps_placeholder():
    s = Skill(id="s", instruction="Solve: {question} carefully.")
    assert skill_text(s) == "Solve: {question} carefully."


def test_skill_text_few_shot_only():
    s = Skill(id="s", instruction="", few_shot=[FewShotExample(question="q1", answer="a1")])
    assert skill_text(s) == "Q: q1\nA: a1"


def test_skill_text_is_prefix_of_render_prompt():
    s = Skill(id="s", instruction=" Think. ",
              few_shot=[FewShotExample(question="q1", answer="a1")])
    assert render_prompt(s, "Z?").startswith(skill_text(s))
    assert render_prompt(s, "Z?") == "Think.\n\nQ: q1\nA: a1\n\nQ: Z?\nA:"


def test_render_prompt_unchanged():
    s = Skill(id="s", instruction="Solve: {question}")
    assert render_prompt(s, "1+1?", "Answer with a number.") == "Solve: 1+1?\n\nAnswer with a number.\nA:"


# --- AC10-14: skill size in the Evaluator ---------------------------------------------------

class TinyBench(Benchmark):
    name = "tinybench"
    answer_format = "Answer with a number."

    def __init__(self, n=3):
        self._examples = [Example(id=f"ex{i}", question=f"What is {i}?", answer=str(i)) for i in range(n)]

    def load(self, split):
        return list(self._examples)

    def extract_answer(self, text):
        nums = re.findall(r"-?\d+", text or "")
        return nums[-1] if nums else None

    def score(self, pred, gold):
        return pred is not None and pred == gold


def _evaluator(responses=("The answer is 1.",)):
    mock = make_mock(list(responses))
    specs = [BenchmarkSpec(key="k1", benchmark=TinyBench(3)), BenchmarkSpec(key="k2", benchmark=TinyBench(2))]
    return mock, Evaluator(mock, specs, batch_size=4)


def _fs_skill(sid="sz", instruction="Think step by step, then answer."):
    return make_skill(sid, instruction, few_shot=[FewShotExample(question="1+1?", answer="1+1=2. The answer is 2.")])


def test_skill_evaluation_default_skill_tokens():
    ev = SkillEvaluation(skill=make_skill("a"), split="train", results={"k": make_result("a", 2, 1)})
    assert ev.skill_tokens is None


def test_skill_evaluation_summary_keys_unchanged():
    ev = SkillEvaluation(skill=make_skill("a"), split="train", results={"k": make_result("a", 2, 1)},
                         skill_tokens=9)
    assert set(ev.summary()) == {"skill_id", "split", "benchmarks"}


def test_evaluator_skill_tokens():
    mock, ev = _evaluator()
    s = _fs_skill()
    expected = mock.count_tokens(skill_text(s))
    assert expected > 0
    assert ev.skill_tokens(s) == expected
    assert ev.skill_tokens(Skill(id="e", instruction="")) == mock.count_tokens("") == 0


def test_evaluate_skill_sets_skill_tokens():
    mock, ev = _evaluator()
    s = _fs_skill()
    res = ev.evaluate_skill(s)
    assert res.skill_tokens == mock.count_tokens(skill_text(s))
    assert set(res.summary()) == {"skill_id", "split", "benchmarks"}


def test_evaluate_sets_skill_tokens_and_no_extra_usage():
    mock, ev = _evaluator()
    skills = [_fs_skill("s1", "Alpha."), make_skill("s2", "Beta gamma delta.")]
    out = ev.evaluate(skills)
    assert [e.skill_tokens for e in out] == [mock.count_tokens(skill_text(s)) for s in skills]
    # 2 skills x (3 + 2) items; computing skill size must not call the target model
    assert mock.usage.snapshot().calls == 10
    assert len(mock.calls) == 10


def test_skill_tokens_error_returns_none(monkeypatch):
    mock, ev = _evaluator()

    def boom(text):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(mock, "count_tokens", boom)
    assert ev.skill_tokens(_fs_skill()) is None


def test_skill_tokens_error_warns_once(monkeypatch, caplog):
    mock, ev = _evaluator()
    skills = [_fs_skill("s1", "Alpha unique words."), make_skill("s2", "Beta unique words.")]
    texts = {skill_text(s) for s in skills}
    orig = mock.count_tokens

    def flaky(text):
        if text in texts:
            raise RuntimeError("no tokenizer")
        return orig(text)

    monkeypatch.setattr(mock, "count_tokens", flaky)
    with caplog.at_level(logging.WARNING, logger="skillopt.evaluator"):
        out = ev.evaluate(skills)
        single = ev.evaluate_skill(skills[0])
    assert [e.skill_tokens for e in out] == [None, None]
    assert single.skill_tokens is None
    assert all(e.results for e in out)
    warnings = [r for r in caplog.records if r.name == "skillopt.evaluator" and r.levelno >= logging.WARNING]
    assert len(warnings) == 1


# --- AC15-16: runlog ------------------------------------------------------------------------------

def test_save_round_num_tokens_and_origin(tmp_path):
    run = RunDir(tmp_path, "t", {"seed": 0})
    s1 = make_skill("a", origin=SkillOrigin(parents=[], operation="initial"))
    s2 = make_skill("b")
    ev1 = SkillEvaluation(skill=s1, split="train", results={"k": make_result("a", 2, 1, benchmark="k")},
                          skill_tokens=5)
    ev2 = SkillEvaluation(skill=s2, split="train", results={"k": make_result("b", 2, 1, benchmark="k")})
    d = run.save_round(0, [ev1, ev2])
    j1 = read_json(d / "skills" / "a.json")
    j2 = read_json(d / "skills" / "b.json")
    assert j1["num_tokens"] == 5
    assert "num_tokens" in j2 and j2["num_tokens"] is None
    assert j1["origin"] == {"parents": [], "operation": "initial", "note": ""}
    assert "origin" in j2 and j2["origin"] is None
    assert Skill.from_dict(j1) == s1
    assert set(read_json(d / "summary.json")[0]) == {"skill_id", "split", "benchmarks"}


def test_save_skill_num_tokens(tmp_path):
    run = RunDir(tmp_path, "t", {"seed": 0})
    p = run.save_skill(make_skill("a"), num_tokens=7)
    assert read_json(p)["num_tokens"] == 7
    run2 = RunDir(tmp_path, "t2", {"seed": 0})
    p2 = run2.save_skill(make_skill("b"))
    j = read_json(p2)
    assert "num_tokens" in j and j["num_tokens"] is None
    assert "origin" in j


# --- AC17-19: optimizers ---------------------------------------------------------------------------

def _opt(responses, **kw):
    llm = make_mock(list(responses), role="optimizer")
    return llm, OPROOptimizer(llm, **kw)


def _ev(sid, acc, instruction, tokens=20):
    skill = make_skill(sid, instruction)
    return make_evaluation(sid, {"gsm8k": (acc, tokens), "bbh": (acc, tokens)}, skill=skill)


def test_opro_initial_prompts_origin():
    _, o = _opt(["<skill>x</skill>"])
    init = o.initial_prompts()
    assert [s.id for s in init] == ["direct", "cot", "cod"]
    for s in init:
        base = get_baseline(s.id)
        assert isinstance(s.origin, SkillOrigin)
        assert s.origin.operation == "initial"
        assert s.origin.parents == []
        assert isinstance(s.origin.note, str)
        assert (s.id, s.instruction, s.few_shot, s.metadata, s.parent_id) == \
            (base.id, base.instruction, base.few_shot, base.metadata, base.parent_id)


def _check_opro_origins(out, known, top_k, expected_parent):
    assert out
    for s in out:
        assert isinstance(s.origin, SkillOrigin), s.id
        assert s.origin.operation == "opro"
        assert s.parent_id == expected_parent
        assert s.origin.parents[0] == s.parent_id
        assert len(set(s.origin.parents)) == len(s.origin.parents)
        assert set(s.origin.parents) <= known
        assert len(s.origin.parents) <= max(top_k, 1)
        if top_k == 0:
            assert s.origin.parents == [s.parent_id]
        assert isinstance(s.origin.note, str) and s.origin.note.strip()
        assert s.parent_id in s.origin.note


@pytest.mark.parametrize("top_k", [0, 1, 2, 5])
def test_opro_update_origin_first_round(top_k):
    _, o = _opt(["<skill>A</skill>", "<skill>B</skill>"], top_k=top_k)
    r0 = [_ev("c1", 0.25, "Answer directly ALPHA."), _ev("c2", 0.75, "Think step by step BETA."),
          _ev("c0", 0.5, "Middle GAMMA.")]
    out = o.update(r0, [])
    _check_opro_origins(out, {"c0", "c1", "c2"}, top_k, "c2")


@pytest.mark.parametrize("top_k", [0, 1, 3, 5])
def test_opro_update_origin_with_history(top_k):
    _, o = _opt(["<skill>A</skill>", "<skill>B</skill>"], top_k=top_k)
    r0 = [_ev("c1", 0.25, "Answer directly ALPHA."), _ev("c2", 0.75, "Think step by step BETA.")]
    r1 = [_ev("c3", 0.5, "Middle GAMMA."), _ev("c4", 1.0, "Fast DELTA.")]
    out = o.update(r1, [r0])
    _check_opro_origins(out, {"c1", "c2", "c3", "c4"}, top_k, "c4")


def test_opro_parent_tie_broken_by_tokens():
    _, o = _opt(["<skill>A</skill>"], num_candidates=1)
    r0 = [_ev("long", 0.75, "Long ALPHA.", tokens=50), _ev("short", 0.75, "Short BETA.", tokens=10)]
    out = o.update(r0, [])
    assert out[0].parent_id == "short"
    assert out[0].origin.parents[0] == "short"


def test_my_optimizer_still_not_implemented():
    o = MyOptimizer(make_mock(role="optimizer"))
    with pytest.raises(NotImplementedError):
        o.initial_prompts()
    with pytest.raises(NotImplementedError):
        o.update([], [])


# --- AC47-52: cli wiring (mock e2e) ---------------------------------------------------------------

def _base(root):
    return ["--config", str(CONFIG), "--mock", "--set", f"run.root={root.as_posix()}"]


def _run(root, *sets, extra=()):
    argv = _base(root)
    for s in sets:
        argv += ["--set", s]
    return main(argv + list(extra))


def _rounds(run):
    return sorted(p.name for p in (run / "rounds").iterdir() if p.is_dir() and p.name.isdigit())


@pytest.fixture(scope="module")
def opro_run(tmp_path_factory):
    logger = logging.getLogger("skillopt")
    snap = _snapshot(logger)
    try:
        root = tmp_path_factory.mktemp("origin_opro")
        return _run(root, OPRO)
    finally:
        _restore(logger, snap)


def test_cli_report_written(opro_run):
    run = opro_run
    html = (run / "report.html").read_text(encoding="utf-8")
    assert 'http-equiv="refresh"' not in html
    assert not (run / "report.html.tmp").exists()
    rounds = _rounds(run)
    assert rounds == ["000", "001", "002"]
    for rnd in rounds:
        for entry in read_json(run / "rounds" / rnd / "summary.json"):
            assert element_by_id(html, f"r{rnd}-{entry['skill_id']}") is not None, (rnd, entry["skill_id"])
    assert element_by_id(html, "r000-direct") is not None
    assert element_by_id(html, "r001-opro-i1-c0") is not None


def test_cli_report_parent_link(opro_run):
    run = opro_run
    html = (run / "report.html").read_text(encoding="utf-8")
    parent = read_json(run / "rounds" / "001" / "skills" / "opro-i1-c0.json")["parent_id"]
    assert parent in {"direct", "cot", "cod"}
    art = element_by_id(html, "r001-opro-i1-c0").html
    assert f'href="#r000-{parent}"' in art


def test_cli_skill_json_num_tokens_and_origin(opro_run):
    run = opro_run
    counter = make_mock()
    for rnd in _rounds(run):
        for p in sorted((run / "rounds" / rnd / "skills").glob("*.json")):
            d = read_json(p)
            assert isinstance(d["num_tokens"], int) and not isinstance(d["num_tokens"], bool), p
            assert d["num_tokens"] == counter.count_tokens(skill_text(Skill.from_dict(d))), p
            assert isinstance(d["origin"], dict), p
            assert d["origin"]["operation"] == ("initial" if rnd == "000" else "opro"), p


def test_cli_report_called_every_round(tmp_path, monkeypatch):
    import skillopt.report as report_mod

    real = report_mod.write_report
    calls = []

    def spy(run_dir, *a, **kw):
        finished_before = (Path(run_dir) / "run.json").exists()
        out = real(run_dir, *a, **kw)
        text = Path(out).read_text(encoding="utf-8")
        calls.append((finished_before, REFRESH_META in text))
        return out

    monkeypatch.setattr("skillopt.report.write_report", spy)
    run = _run(tmp_path, OPRO)
    assert read_json(run / "run.json")["rounds"] == 3
    assert len(calls) == 4
    assert all(c == (False, True) for c in calls[:-1]), calls
    assert calls[-1] == (True, False)


def test_cli_report_failure_does_not_stop_run(tmp_path, monkeypatch, caplog):
    def broken(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr("skillopt.report.write_report", broken)
    run = _run(tmp_path, OPRO, "max_iterations=1")
    assert (run / "run.json").is_file()
    assert read_json(run / "run.json")["rounds"] == 2
    warnings = [r for r in caplog.records
                if r.name == "skillopt.cli" and r.levelno == logging.WARNING and "report not written" in r.getMessage()]
    assert warnings


def test_eval_only_num_tokens_no_report(tmp_path):
    run = main(_base(tmp_path) + ["--eval-only", "--limit", "1"])
    counter = make_mock()
    files = sorted((run / "skills").glob("*.json"))
    assert files
    for p in files:
        d = read_json(p)
        assert "num_tokens" in d, p
        assert d["num_tokens"] == counter.count_tokens(skill_text(Skill.from_dict(d))), p
    assert not list(run.rglob("report.html"))


# --- AC20: an optimizer that sets no origin --------------------------------------------------------

NO_ORIGIN = uniq("noorigin")


@register_optimizer(NO_ORIGIN)
class NoOriginOptimizer(Optimizer):
    def initial_prompts(self):
        return [Skill(id="base", instruction="Think step by step.")]

    def update(self, results, history):
        r = len(history) + 1
        return [Skill(id=f"child-{r}", instruction=f"Think briefly step by step {r}.", parent_id="base")]


def test_optimizer_without_origin(tmp_path):
    from skillopt.report import load_run

    run = _run(tmp_path, f"optimizer.name={NO_ORIGIN}")
    assert (run / "run.json").is_file()
    assert _rounds(run) == ["000", "001", "002"]
    for rnd, sid in (("000", "base"), ("001", "child-1"), ("002", "child-2")):
        d = read_json(run / "rounds" / rnd / "skills" / f"{sid}.json")
        assert d["origin"] is None
        assert isinstance(d["num_tokens"], int)
    html = (run / "report.html").read_text(encoding="utf-8")
    assert 'http-equiv="refresh"' not in html
    art = element_by_id(html, "r001-child-1")
    assert art is not None
    assert 'href="#r000-base"' in art.html
    data = load_run(run)
    by_round = {r["index"]: {s["id"]: s for s in r["skills"]} for r in data["rounds"]}
    assert by_round[2]["child-2"]["parents"] == [{"id": "base", "round": 0}]
    assert by_round[2]["child-2"]["operation"] is None
    assert by_round[0]["base"]["children"] == [{"id": "child-1", "round": 1}, {"id": "child-2", "round": 2}]


# --- AC53: README -----------------------------------------------------------------------------------

@pytest.mark.parametrize("s", ["report.html", "skillopt.report", "SkillOrigin"])
def test_readme_mentions(s):
    assert s in (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
