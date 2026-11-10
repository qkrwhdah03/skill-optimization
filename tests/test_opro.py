"""OPRO worked example: OPROOptimizer(initial_prompts, update(results, history)) (AC17-18)."""
import pytest

from helpers import make_evaluation, make_mock, make_skill
from skillopt.optimizer import OPROOptimizer, Optimizer, build_optimizer
from skillopt.skills import FewShotExample, Skill

KEYS = ("gsm8k", "bbh")


def _opt(responses, **kw):
    llm = make_mock(list(responses), role="optimizer")
    return llm, OPROOptimizer(llm, **kw)


def _ev(sid, acc_g, acc_b, instruction, tokens=20, few_shot=None):
    skill = make_skill(sid, instruction, few_shot)
    return make_evaluation(sid, {"gsm8k": (acc_g, tokens), "bbh": (acc_b, tokens)}, skill=skill)


def _round0():
    return [_ev("c1", 0.25, 0.25, "Answer directly ALPHA."),
            _ev("c2", 0.75, 0.75, "Think step by step BETA.")]


def _acc_in(prompt, acc):
    forms = {f"{acc:.3f}", f"{acc:.2f}", str(acc), f"{acc * 100:.1f}%", f"{acc * 100:.0f}%"}
    return any(f in prompt for f in forms)


# --- AC17 ------------------------------------------------------------------------

def test_is_optimizer_subclass_and_registered():
    llm = make_mock(["<skill>x</skill>"], role="optimizer")
    o = build_optimizer({"name": "opro"}, llm)
    assert isinstance(o, OPROOptimizer)
    assert isinstance(o, Optimizer)
    assert o.llm is llm


def test_build_passes_params():
    llm = make_mock(["<skill>A</skill>", "<skill>B</skill>", "<skill>C</skill>"], role="optimizer")
    o = build_optimizer({"name": "opro", "num_candidates": 3}, llm)
    out = o.update(_round0(), [])
    assert [s.instruction for s in out] == ["A", "B", "C"]


def test_initial_prompts_default():
    _, o = _opt(["<skill>x</skill>"])
    init = o.initial_prompts()
    assert isinstance(init, list)
    assert all(isinstance(s, Skill) for s in init)
    assert [s.id for s in init] == ["direct", "cot", "cod"]


def test_initial_prompts_custom():
    _, o = _opt(["<skill>x</skill>"], initial=["cot"])
    assert [s.id for s in o.initial_prompts()] == ["cot"]


def test_initial_prompts_no_llm_calls():
    llm, o = _opt(["<skill>x</skill>"])
    o.initial_prompts()
    assert llm.calls == []


# --- AC18 ------------------------------------------------------------------------

def test_update_first_iteration_ids_and_instructions():
    _, o = _opt(["<skill>A</skill>", "<skill>B</skill>"])
    out = o.update(_round0(), [])
    assert all(isinstance(s, Skill) for s in out)
    assert [s.id for s in out] == ["opro-i1-c0", "opro-i1-c1"]
    assert [s.instruction for s in out] == ["A", "B"]


def test_update_default_num_candidates_is_two():
    llm, o = _opt(["<skill>A</skill>", "<skill>B</skill>", "<skill>C</skill>"])
    out = o.update(_round0(), [])
    assert len(out) == 2
    assert len(llm.calls) == 2


def test_update_iteration_from_history_length():
    _, o = _opt(["<skill>A2</skill>", "<skill>B2</skill>"])
    history = [_round0()]
    results = [_ev("opro-i1-c0", 0.5, 0.5, "Round one A."), _ev("opro-i1-c1", 0.5, 0.25, "Round one B.")]
    out = o.update(results, history)
    assert out
    assert all(s.id.startswith("opro-i2-") for s in out)
    assert [s.id for s in out] == ["opro-i2-c0", "opro-i2-c1"]


def test_update_iteration_three():
    _, o = _opt(["<skill>New.</skill>"], num_candidates=1)
    out = o.update([_ev("x", 0.5, 0.5, "X.")], [_round0(), [_ev("y", 0.5, 0.5, "Y.")]])
    assert [s.id for s in out] == ["opro-i3-c0"]


def test_update_skips_instruction_seen_in_results():
    _, o = _opt(["<skill>Answer directly ALPHA.</skill>", "<skill>Fresh one.</skill>"])
    out = o.update(_round0(), [])
    assert [s.instruction for s in out] == ["Fresh one."]


def test_update_skips_instruction_seen_in_history():
    _, o = _opt(["<skill>Think step by step BETA.</skill>", "<skill>Fresh two.</skill>"])
    results = [_ev("r1", 0.5, 0.5, "Current one.")]
    out = o.update(results, [_round0()])
    assert [s.instruction for s in out] == ["Fresh two."]


def test_update_seen_comparison_is_stripped():
    _, o = _opt(["<skill>  Answer directly ALPHA.  </skill>", "<skill>Other.</skill>"])
    out = o.update(_round0(), [])
    assert [s.instruction for s in out] == ["Other."]


def test_update_drops_duplicate_proposals_within_call():
    _, o = _opt(["<skill>Same.</skill>", "<skill>Same.</skill>"])
    out = o.update(_round0(), [])
    assert [s.instruction for s in out] == ["Same."]


def test_update_all_duplicates_returns_empty():
    _, o = _opt(["<skill>Answer directly ALPHA.</skill>", "<skill>Think step by step BETA.</skill>"])
    assert o.update(_round0(), []) == []


def test_update_id_collision_with_results_gets_suffix():
    _, o = _opt(["<skill>Brand new.</skill>"], num_candidates=1)
    results = _round0() + [_ev("opro-i1-c0", 0.5, 0.5, "Old opro.")]
    out = o.update(results, [])
    assert [s.id for s in out] == ["opro-i1-c0-2"]


def test_update_id_collision_with_history_gets_suffix():
    _, o = _opt(["<skill>Brand new.</skill>"], num_candidates=1)
    history = [[_ev("opro-i2-c0", 0.5, 0.5, "Old.")]]
    out = o.update([_ev("opro-i2-c0-2", 0.5, 0.5, "Older.")], history)
    assert [s.id for s in out] == ["opro-i2-c0-3"]


def test_update_ids_unique_within_returned_list():
    _, o = _opt(["<skill>P.</skill>", "<skill>Q.</skill>", "<skill>R.</skill>"], num_candidates=3)
    out = o.update(_round0(), [])
    ids = [s.id for s in out]
    assert len(ids) == len(set(ids)) == 3


def test_meta_prompt_contains_instructions_and_accuracy():
    llm, o = _opt(["<skill>New.</skill>"], num_candidates=1)
    o.update(_round0(), [])
    assert llm.calls
    prompt = llm.calls[0]
    for instr in ("Answer directly ALPHA.", "Think step by step BETA."):
        assert instr in prompt, instr
    for key in KEYS:
        assert key in prompt, key
    for acc in (0.25, 0.75):
        assert _acc_in(prompt, acc), acc


def test_meta_prompt_per_benchmark_accuracy_values():
    llm, o = _opt(["<skill>New.</skill>"], num_candidates=1)
    results = [_ev("pa", 1.0, 0.0, "Per A."), _ev("pb", 0.25, 0.5, "Per B.")]
    o.update(results, [])
    prompt = llm.calls[0]
    for v in (0.25, 0.5):
        assert _acc_in(prompt, v), v


def test_meta_prompt_includes_history_instructions():
    llm, o = _opt(["<skill>New.</skill>"], num_candidates=1)
    o.update([_ev("cur", 0.5, 0.5, "Current GAMMA.")], [_round0()])
    prompt = llm.calls[0]
    for instr in ("Answer directly ALPHA.", "Think step by step BETA.", "Current GAMMA."):
        assert instr in prompt, instr


def test_update_does_not_mutate_inputs():
    _, o = _opt(["<skill>A</skill>", "<skill>B</skill>"])
    results = _round0()
    history = [[_ev("h0", 0.5, 0.5, "Hist.")]]
    ids_r = [e.skill_id for e in results]
    ids_h = [[e.skill_id for e in rnd] for rnd in history]
    o.update(results, history)
    assert [e.skill_id for e in results] == ids_r
    assert [[e.skill_id for e in rnd] for rnd in history] == ids_h
    assert len(history) == 1


def test_update_new_skills_do_not_collide_with_known_ids():
    _, o = _opt(["<skill>A</skill>", "<skill>B</skill>"])
    results = _round0()
    out = o.update(results, [])
    assert not ({s.id for s in out} & {e.skill_id for e in results})


def test_update_with_few_shot_parent_runs():
    fs = [FewShotExample(question="Demo Q?", answer="Demo A. The answer is 1.")]
    _, o = _opt(["<skill>Short.</skill>"], num_candidates=1)
    results = [_ev("best", 1.0, 1.0, "Best.", few_shot=fs), _ev("worse", 0.0, 0.0, "Worse.")]
    out = o.update(results, [])
    assert [s.instruction for s in out] == ["Short."]
