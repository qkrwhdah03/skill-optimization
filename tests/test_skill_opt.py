"""Tests for SkillOptOptimizer (``skillopt``): registry, validation gate, minibatches, edit budget, meta-update, E2E."""
from __future__ import annotations

import logging

import pytest

from helpers import GSM8K_DIR, make_evaluation, make_mock, make_skill, uniq
from skillopt.benchmarks import build_benchmark
from skillopt.evaluator import BenchmarkSpec, EvalResult, Evaluator, ItemResult, SkillEvaluation
from skillopt.llm import BaseLLM, RawGeneration, TokenUsage, register_llm
from skillopt.optimizer import OPTIMIZER_REGISTRY, Optimizer, build_optimizer, discover_optimizers, optimize
from skillopt.optimizer.algorithms.skill_opt import (
    SkillOptOptimizer,
    _cosine_edit_budget,
    _replay_gate,
)
from skillopt.skills import FewShotExample, Skill, SkillOrigin

SKILLOPT_LOGGER = "skillopt.optimizer.algorithms.skill_opt"


class _ScriptedLLM(BaseLLM):
    uses_gpu = False

    def __init__(self, config, role="default"):
        super().__init__(config, role=role)
        self.script: list[tuple[str, bool]] = []
        self.calls: list[str] = []

    def _generate(self, prompt, gen):
        self.calls.append(prompt)
        text, truncated = self.script.pop(0)
        return RawGeneration(text=text, truncated=truncated)

    def count_tokens(self, text):
        return len(text.split())


_SCRIPTED = uniq("skillopt_scripted")
register_llm(_SCRIPTED)(_ScriptedLLM)


def _make_scripted(script: list[tuple[str, bool]]) -> _ScriptedLLM:
    llm = _ScriptedLLM({"backend": _SCRIPTED, "model": "scripted"}, role="optimizer")
    llm.script = list(script)
    return llm


def _opt(responses, **kw):
    llm = make_mock(list(responses), role="optimizer")
    return llm, SkillOptOptimizer(llm, **kw)


def _ev(
    sid: str,
    metrics: dict,
    instruction: str | None = None,
    few_shot=None,
    parent_id: str | None = None,
    origin: SkillOrigin | None = None,
    metadata: dict | None = None,
) -> SkillEvaluation:
    sk = make_skill(
        sid,
        instruction or f"Instruction for {sid}.",
        few_shot=few_shot,
        parent_id=parent_id,
        origin=origin,
        metadata=metadata or {},
    )
    return make_evaluation(sid, metrics, skill=sk)


def _round0():
    return [
        _ev("direct", {"gsm8k": (0.25, 20), "bbh": (0.25, 20)}, "Answer directly ALPHA."),
        _ev("cot", {"gsm8k": (0.75, 100), "bbh": (0.75, 100)}, "Think step by step BETA."),
        _ev("cod", {"gsm8k": (0.75, 50), "bbh": (0.75, 50)}, "Draft briefly GAMMA."),
    ]


# ------------------------------------------------------------------ 1. Registry & validation


def test_registered_and_discovered():
    found = discover_optimizers()
    assert "skillopt.optimizer.algorithms.skill_opt" in found
    assert "skillopt" in OPTIMIZER_REGISTRY

    llm = make_mock(["<skill>Compact.</skill>"], role="optimizer")
    opt = build_optimizer({"name": "skillopt", "num_candidates": 1, "epsilon": 0.03}, llm)
    assert isinstance(opt, SkillOptOptimizer)
    assert isinstance(opt, Optimizer)
    assert opt.llm is llm
    assert opt.params["num_candidates"] == 1
    assert opt.params["epsilon"] == 0.03


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"initial": []},
        {"initial": "cot"},
        {"initial": ["unknown_baseline"]},
        {"initial": ["cot", "cot"]},
        {"anchor_id": "   "},
        {"epsilon": -0.1},
        {"token_weight": -1.0},
        {"min_improvement": -0.01},
        {"num_candidates": -1},
        {"max_edits": 0},
        {"min_edits": 0},
        {"max_edits": 2, "min_edits": 3},
        {"edit_schedule_rounds": 0},
        {"num_failures": -1},
        {"num_successes": -1},
        {"max_trace_chars": -1},
        {"max_skill_chars": 0},
        {"max_rejected_buffer": -1},
        {"epoch_rounds": 0},
        {"max_retries": -1},
    ],
)
def test_constructor_validation_rejects_invalid_params(bad_kwargs):
    llm = make_mock(["<skill>x</skill>"], role="optimizer")
    with pytest.raises(ValueError):
        SkillOptOptimizer(llm, **bad_kwargs)


# ------------------------------------------------------------------ 2. Initial prompts


def test_initial_prompts_defaults_custom_origin_and_no_llm_calls():
    llm, opt = _opt(["<skill>x</skill>"])
    init = opt.initial_prompts()
    assert [s.id for s in init] == ["direct", "cot", "cod"]
    assert all(isinstance(s, Skill) for s in init)
    assert all(s.origin is not None and s.origin.operation == "initial" for s in init)
    assert all(s.origin is not None and s.origin.parents == [] for s in init)
    assert "anchor" in (init[1].origin.note or "")
    assert llm.calls == []

    llm2, opt2 = _opt(["<skill>x</skill>"], initial=["cot", "cod"])
    assert [s.id for s in opt2.initial_prompts()] == ["cot", "cod"]
    assert llm2.calls == []


# ------------------------------------------------------------------ 3. Cosine edit budget


def test_cosine_edit_budget_decay_and_edge_cases():
    budgets = [_cosine_edit_budget(it, max_edits=4, min_edits=2, schedule_rounds=4) for it in range(1, 7)]
    assert budgets[0] == 4
    assert budgets[-1] == 2
    assert all(a >= b for a, b in zip(budgets, budgets[1:]))
    assert all(2 <= b <= 4 for b in budgets)

    assert _cosine_edit_budget(1, max_edits=4, min_edits=2, schedule_rounds=1) == 2
    assert _cosine_edit_budget(2, max_edits=3, min_edits=3, schedule_rounds=5) == 3


# ------------------------------------------------------------------ 4. Validation gate & rejected buffer


def test_validation_gate_promotes_feasible_token_savings_and_higher_accuracy():
    r0 = _round0()
    better_tokens = _ev(
        "skillopt-i1-c0",
        {"gsm8k": (0.75, 30), "bbh": (0.75, 30)},
        "Even tighter draft DELTA.",
        parent_id="cod",
        metadata={"applied_edits": "- [DELETE] redundant step words"},
    )
    worse_cand = _ev(
        "skillopt-i1-c1",
        {"gsm8k": (0.75, 80), "bbh": (0.75, 80)},
        "Longer draft EPSILON.",
        parent_id="cod",
        metadata={"applied_edits": "- [ADD] verbose explanations"},
    )
    anchor, inc, _, acc, rej = _replay_gate(
        [r0, [better_tokens, worse_cand]],
        anchor_id="cot",
        epsilon=0.02,
        token_weight=0.25,
        min_improvement=1e-6,
        max_rejected_buffer=6,
    )
    assert anchor is not None and anchor.ev.skill_id == "cot"
    assert inc is not None and inc.ev.skill_id == "skillopt-i1-c0"
    assert [a.skill_id for a in acc] == ["skillopt-i1-c0"]
    assert [r.skill_id for r in rej] == ["skillopt-i1-c1"]

    # Higher accuracy candidate also promoted
    higher_acc = _ev(
        "skillopt-i2-c0",
        {"gsm8k": (1.0, 35), "bbh": (1.0, 35)},
        "Higher accuracy skill ZETA.",
        parent_id="skillopt-i1-c0",
        metadata={"applied_edits": "- [ADD] keep final constraints"},
    )
    _, inc2, _, acc2, _ = _replay_gate(
        [r0, [better_tokens, worse_cand], [higher_acc]],
        anchor_id="cot",
        epsilon=0.02,
        token_weight=0.25,
        min_improvement=1e-6,
        max_rejected_buffer=6,
    )
    assert inc2 is not None and inc2.ev.skill_id == "skillopt-i2-c0"
    assert [a.skill_id for a in acc2] == ["skillopt-i1-c0", "skillopt-i2-c0"]


def test_validation_gate_rejects_regression_and_feeds_negative_memory():
    r0 = _round0()
    # Violates anchor - epsilon (0.75 - 0.02 = 0.73), even though super short
    regressed = _ev(
        "skillopt-i1-c0",
        {"gsm8k": (0.25, 10), "bbh": (0.25, 10)},
        "Too short regressed skill.",
        parent_id="cod",
        metadata={"applied_edits": "- [DELETE] intermediate state tracking"},
    )
    # Preserves acc=0.75 but increases token cost (80 > 50)
    costlier = _ev(
        "skillopt-i1-c1",
        {"gsm8k": (0.75, 80), "bbh": (0.75, 80)},
        "Costlier skill.",
        parent_id="cod",
        metadata={"applied_edits": "- [ADD] double-check every answer"},
    )

    llm, opt = _opt(
        ["<edits>- [REPLACE] tighter steps</edits>", "<skill>Next candidate skill.</skill>"],
        num_candidates=1,
        two_stage=True,
        epoch_rounds=5,
    )
    out = opt.update([regressed, costlier], [r0])
    assert len(out) == 1
    assert out[0].parent_id == "cod"
    assert out[0].metadata["incumbent_id"] == "cod"
    assert out[0].metadata["gate_rejected_count"] == 2

    prompt = llm.calls[0]
    assert "Rejected-edit negative memory" in prompt
    assert "skillopt-i1-c0" in prompt
    assert "skillopt-i1-c1" in prompt
    assert "- [DELETE] intermediate state tracking" in prompt
    assert "- [ADD] double-check every answer" in prompt


# ------------------------------------------------------------------ 5. Reflection minibatches


def test_reflection_minibatch_separates_failures_verbose_and_compact_successes():
    items = [
        ItemResult(
            example_id="ex_fail",
            prediction="99",
            gold="42",
            correct=False,
            usage=TokenUsage(input_tokens=10, output_tokens=60, reasoning_tokens=50),
            latency=0.1,
            response_text="The answer is 99.",
            reasoning_text="wrong intermediate step",
            question="FAILQ: What is 6 * 7?",
        ),
        ItemResult(
            example_id="ex_verbose",
            prediction="10",
            gold="10",
            correct=True,
            usage=TokenUsage(input_tokens=10, output_tokens=220, reasoning_tokens=200),
            latency=0.2,
            response_text="The answer is 10.",
            reasoning_text="long rambling verification trace",
            question="VERBOSE_Q: What is 5 + 5?",
        ),
        ItemResult(
            example_id="ex_compact",
            prediction="3",
            gold="3",
            correct=True,
            usage=TokenUsage(input_tokens=10, output_tokens=12, reasoning_tokens=6),
            latency=0.05,
            response_text="The answer is 3.",
            reasoning_text="1+2=3",
            question="COMPACT_Q: What is 1 + 2?",
        ),
    ]
    sk = make_skill("cot", "Think step by step.")
    ev = SkillEvaluation(
        skill=sk,
        split="train",
        results={"gsm8k": EvalResult(skill_id="cot", benchmark="gsm8k", split="train", items=items)},
    )

    llm, opt = _opt(
        ["<edits>- [DELETE] rambling</edits>", "<skill>Compact reasoning skill.</skill>"],
        num_candidates=1,
        num_failures=2,
        num_successes=3,
        two_stage=True,
    )
    out = opt.update([ev], [])
    assert [s.instruction for s in out] == ["Compact reasoning skill."]

    reflect_prompt = llm.calls[0]
    assert "Failure cases" in reflect_prompt
    assert "FAILQ: What is 6 * 7?" in reflect_prompt
    assert "Verbose success cases" in reflect_prompt
    assert "VERBOSE_Q: What is 5 + 5?" in reflect_prompt
    assert "Compact success cases" in reflect_prompt
    assert "COMPACT_Q: What is 1 + 2?" in reflect_prompt


# ------------------------------------------------------------------ 6. Two-stage & single-stage & parsing


def test_two_stage_vs_single_stage_and_omitted_edits_fallback():
    r0 = _round0()
    fs = [FewShotExample(question="Demo Q?", answer="Demo A.")]
    r0_fs = [
        _ev("direct", {"gsm8k": (0.25, 20)}, "Direct."),
        _ev("cot", {"gsm8k": (0.75, 50)}, "CoT.", few_shot=fs),
    ]

    # Two-stage: 1 reflect call + 2 apply calls = 3 LLM calls
    llm_two, opt_two = _opt(
        [
            "<reflection>Cut prose.</reflection><edits>- [DELETE] filler\n- [ADD] math eq</edits>",
            "<edits>- [DELETE] filler</edits><skill>Stage2 Skill A.</skill>",
            "<skill>Stage2 Skill B without edits tag.</skill>",
        ],
        num_candidates=2,
        two_stage=True,
    )
    out_two = opt_two.update(r0_fs, [])
    assert len(llm_two.calls) == 3
    assert [s.instruction for s in out_two] == ["Stage2 Skill A.", "Stage2 Skill B without edits tag."]
    assert out_two[0].metadata["applied_edits"] == "- [DELETE] filler"
    assert out_two[1].metadata["applied_edits"] == ""
    assert out_two[0].few_shot == fs
    assert out_two[0].origin is not None and out_two[0].origin.operation == "skillopt_edit"

    # Single-stage: 2 direct calls (no Stage 1 reflection call)
    llm_one, opt_one = _opt(
        [
            "<edits>- [REPLACE] short</edits><skill>Single Stage A.</skill>",
            "<skill>Single Stage B.</skill>",
        ],
        num_candidates=2,
        two_stage=False,
    )
    out_one = opt_one.update(r0, [])
    assert len(llm_one.calls) == 2
    assert [s.instruction for s in out_one] == ["Single Stage A.", "Single Stage B."]


# ------------------------------------------------------------------ 7. Epoch slow / meta update


def test_epoch_meta_update_triggers_on_boundary_and_tags_operation():
    r0 = _round0()
    r1 = [
        _ev(
            "skillopt-i1-c0",
            {"gsm8k": (0.75, 30), "bbh": (0.75, 30)},
            "Round 1 winner.",
            parent_id="cod",
            metadata={"applied_edits": "- [DELETE] filler prose"},
        ),
        _ev(
            "skillopt-i1-c1",
            {"gsm8k": (0.50, 25), "bbh": (0.50, 25)},
            "Round 1 loser.",
            parent_id="cod",
            metadata={"applied_edits": "- [DELETE] core step"},
        ),
    ]

    llm, opt = _opt(
        [
            "<meta>Keep compact eqs; avoid deleting core state steps.</meta>",
            "<edits>- [ADD] state marker</edits>",
            "<edits>- [ADD] eqs</edits><skill>Epoch 2 edit candidate.</skill>",
            "<edits>- [REPLACE] consolidate</edits><skill>Epoch 2 meta candidate.</skill>",
        ],
        num_candidates=2,
        epoch_rounds=2,
        two_stage=True,
    )
    out = opt.update(r1, [r0])
    # 1 meta call + 1 reflect call + 2 candidate calls = 4 calls
    assert len(llm.calls) == 4
    assert "slow meta-skill reflection" in llm.calls[0]
    assert "Keep compact eqs; avoid deleting core state steps." in llm.calls[1]
    assert "Executive Meta-Strategy" in llm.calls[1]
    assert [s.origin.operation for s in out if s.origin is not None] == ["skillopt_edit", "skillopt_meta"]
    assert out[1].metadata["meta_epoch"] == 1


# ------------------------------------------------------------------ 8. Retries, warnings, collisions, non-mutation, E2E


def test_retries_on_empty_duplicate_truncated_too_long_and_warning(caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    r0 = _round0()
    llm = _make_scripted(
        [
            ("<skill>   </skill>", False),          # c0 attempt 1: empty
            ("<skill>Draft briefly GAMMA.</skill>", False),  # c0 attempt 2: duplicate
            ("<skill>Fresh skill X.</skill>", False),        # c0 attempt 3: ok
            ("<think>unfinished", True),                     # c1 attempt 1: truncated
            ("<skill>" + ("a" * 80) + "</skill>", False),    # c1 attempt 2: too_long
            ("<skill>Fresh skill X.</skill>", False),        # c1 attempt 3: duplicate
        ]
    )
    opt = SkillOptOptimizer(llm, num_candidates=2, two_stage=False, max_retries=2, max_skill_chars=40)
    out = opt.update(r0, [])
    assert [s.instruction for s in out] == ["Fresh skill X."]
    assert [s.id for s in out] == ["skillopt-i1-c0"]
    assert len(llm.calls) == 6

    warns = [r for r in caplog.records if r.name == SKILLOPT_LOGGER and r.levelno == logging.WARNING]
    assert len(warns) == 1
    msg = warns[0].getMessage()
    assert "1/2" in msg
    assert "empty=1" in msg
    assert "duplicate=2" in msg
    assert "truncated=1" in msg
    assert "too_long=1" in msg


def test_zero_candidates_returns_empty_without_llm_calls_or_warning(caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    llm, opt = _opt(["<skill>x</skill>"], num_candidates=0)
    assert opt.update(_round0(), []) == []
    assert llm.calls == []
    assert [r for r in caplog.records if r.name == SKILLOPT_LOGGER and r.levelno == logging.WARNING] == []


def test_id_collisions_and_non_mutation():
    r0 = _round0() + [_ev("skillopt-i1-c0", {"gsm8k": (0.5, 40)}, "Old candidate.")]
    history = [[_ev("h0", {"gsm8k": (0.4, 30)}, "Hist.")]]
    results = [_ev("skillopt-i2-c0", {"gsm8k": (0.75, 45)}, "Latest.")]

    # Test collision with existing skillopt-i2-c0 when history has length 1 -> iteration 2
    _, opt = _opt(["<skill>Brand new.</skill>"], num_candidates=1, two_stage=False)
    ids_r = [e.skill_id for e in results]
    ids_h = [[e.skill_id for e in rnd] for rnd in history]
    out = opt.update(results, history)
    assert [s.id for s in out] == ["skillopt-i2-c0-2"]
    assert [e.skill_id for e in results] == ids_r
    assert [[e.skill_id for e in rnd] for rnd in history] == ids_h

    # Collision on iteration 1 as well
    _, opt1 = _opt(["<skill>Another new.</skill>"], num_candidates=1, two_stage=False)
    assert [s.id for s in opt1.update(r0, [])] == ["skillopt-i1-c0-2"]


def test_end_to_end_optimize_loop_with_mock():
    target = make_mock(role="target")
    opt_llm = make_mock(
        [
            "<edits>- [DELETE] filler</edits>",
            "<skill>Step equations only.</skill>",
            "<skill>State transitions only.</skill>",
            "<meta>Keep compact symbols.</meta>",
            "<edits>- [REPLACE] tighter</edits>",
            "<skill>Pure compact symbolic trace.</skill>",
            "<skill>Minimal equations + answer.</skill>",
        ],
        role="optimizer",
    )
    bench = build_benchmark({"name": "gsm8k", "source": "local", "data_dir": str(GSM8K_DIR), "train": 4, "val": 2})
    evaluator = Evaluator(target, [BenchmarkSpec(key="gsm8k", benchmark=bench)], batch_size=4)
    optimizer = SkillOptOptimizer(opt_llm, num_candidates=2, epoch_rounds=2, two_stage=True)

    loop_res = optimize(optimizer, evaluator, max_iterations=2)
    assert loop_res.stop_reason == "max_iterations"
    assert len(loop_res.history) == 3
    assert [ev.skill_id for ev in loop_res.history[0]] == ["direct", "cot", "cod"]
    assert [ev.skill_id for ev in loop_res.history[1]] == ["skillopt-i1-c0", "skillopt-i1-c1"]
    assert [ev.skill_id for ev in loop_res.history[2]] == ["skillopt-i2-c0", "skillopt-i2-c1"]
