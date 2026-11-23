"""Tests for SkillOptOptimizer (``skillopt``): constrained reasoning-length adaptation of SkillOpt."""
from __future__ import annotations

import json
import logging

import pytest

from helpers import GSM8K_DIR, make_evaluation, make_mock, make_skill, uniq
from skillopt.benchmarks import build_benchmark
from skillopt.cli import main as cli_main
from skillopt.evaluator import BenchmarkSpec, EvalResult, Evaluator, ItemResult, SkillEvaluation
from skillopt.llm import BaseLLM, RawGeneration, TokenUsage, register_llm
from skillopt.optimizer import OPTIMIZER_REGISTRY, Optimizer, build_optimizer, discover_optimizers, optimize
from skillopt.optimizer.algorithms.skill_opt import (
    SLOW_UPDATE_END,
    SLOW_UPDATE_START,
    SkillOptOptimizer,
    _PatchEdit,
    _apply_patches,
    _cosine_edit_budget,
    _edit_budget,
    _extract_json_object,
    _join_protected,
    _partition_example_ids,
    _replay_gate,
    _split_protected,
)
from skillopt.skills import FewShotExample, Skill, SkillOrigin

SKILLOPT_LOGGER = "skillopt.optimizer.algorithms.skill_opt"


class _ScriptedLLM(BaseLLM):
    uses_gpu = False

    def __init__(self, config, role="default"):
        super().__init__(config, role=role)
        self.script: list[tuple[str, bool]] = []
        self.calls: list[str] = []
        self.batch_sizes: list[int] = []

    def _generate(self, prompt, gen):
        self.calls.append(prompt)
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        text, truncated = self.script.pop(0)
        return RawGeneration(text=text, truncated=truncated)

    def _generate_batch(self, prompts, gen):
        self.batch_sizes.append(len(prompts))
        return [self._generate(p, gen) for p in prompts]

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


def _item(
    ex_id: str,
    *,
    correct: bool,
    output_tokens: int,
    reasoning_tokens: int = 0,
    input_tokens: int = 10,
    question: str | None = None,
    truncated: bool = False,
) -> ItemResult:
    return ItemResult(
        example_id=ex_id,
        prediction="1" if correct else "2",
        gold="1",
        correct=correct,
        usage=TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        latency=0.05,
        response_text="The answer is 1." if correct else "The answer is 2.",
        reasoning_text=("step " * max(1, reasoning_tokens // 5)) if reasoning_tokens else "",
        question=question or f"Question {ex_id}",
        truncated=truncated,
    )


def _ev_items(
    sid: str,
    by_bench: dict[str, list[ItemResult]],
    instruction: str,
    parent_id: str | None = None,
    origin: SkillOrigin | None = None,
    metadata: dict | None = None,
) -> SkillEvaluation:
    sk = make_skill(
        sid,
        instruction,
        parent_id=parent_id,
        origin=origin,
        metadata=metadata or {},
    )
    results = {
        bench: EvalResult(skill_id=sid, benchmark=bench, split="train", items=list(items))
        for bench, items in by_bench.items()
    }
    return SkillEvaluation(skill=sk, split="train", results=results)


def _round0():
    return [
        _ev("direct", {"gsm8k": (0.25, 20), "bbh": (0.25, 20)}, "Answer directly ALPHA."),
        _ev("cot", {"gsm8k": (0.75, 100), "bbh": (0.75, 100)}, "Think step by step BETA."),
        _ev("cod", {"gsm8k": (0.75, 50), "bbh": (0.75, 50)}, "Draft briefly GAMMA."),
    ]


# ------------------------------------------------------------------ 1. Registry & constructor validation


def test_registered_and_discovered():
    found = discover_optimizers()
    assert "skillopt.optimizer.algorithms.skill_opt" in found
    assert "skillopt" in OPTIMIZER_REGISTRY

    llm = make_mock(["<skill>Compact.</skill>"], role="optimizer")
    opt = build_optimizer(
        {"name": "skillopt", "num_candidates": 1, "epsilon": 0.03, "cost_metric": "reasoning"}, llm
    )
    assert isinstance(opt, SkillOptOptimizer)
    assert isinstance(opt, Optimizer)
    assert opt.llm is llm
    assert opt.params["num_candidates"] == 1
    assert opt.params["epsilon"] == 0.03
    assert opt.params["cost_metric"] == "reasoning"


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        {"initial": []},
        {"initial": "cot"},
        {"initial": ["unknown_baseline"]},
        {"initial": ["cot", "cot"]},
        {"anchor_id": "   "},
        {"epsilon": -0.1},
        {"per_benchmark_epsilon": -0.05},
        {"cost_metric": "invalid_metric"},
        {"acc_bonus": -0.2},
        {"min_improvement": -0.01},
        {"selection_ratio": -0.1},
        {"selection_ratio": 1.0},
        {"split_seed": True},
        {"num_candidates": -1},
        {"rollout_batch_size": 0},
        {"reflection_batch_size": 0},
        {"max_failure_batches": -1},
        {"max_success_batches": -1},
        {"max_edits": 0},
        {"min_edits": 0},
        {"max_edits": 2, "min_edits": 3},
        {"edit_schedule_rounds": 0},
        {"lr_schedule": "exponential"},
        {"edit_mode": "yolo"},
        {"max_trace_chars": -1},
        {"max_skill_chars": 0},
        {"max_rejected_buffer": -1},
        {"epoch_rounds": 0},
        {"slow_update_samples": 0},
        {"max_retries": -1},
    ],
)
def test_constructor_validation_rejects_invalid_params(bad_kwargs):
    llm = make_mock(["<skill>x</skill>"], role="optimizer")
    with pytest.raises(ValueError):
        SkillOptOptimizer(llm, **bad_kwargs)


# ------------------------------------------------------------------ 2. Initial prompts & LR schedules


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


def test_edit_budget_schedules():
    cos_budgets = [_cosine_edit_budget(it, max_edits=4, min_edits=2, schedule_rounds=4) for it in range(1, 7)]
    assert cos_budgets[0] == 4
    assert cos_budgets[-1] == 2
    assert all(a >= b for a, b in zip(cos_budgets, cos_budgets[1:]))
    assert all(2 <= b <= 4 for b in cos_budgets)
    assert _cosine_edit_budget(1, max_edits=4, min_edits=2, schedule_rounds=1) == 2

    lin_budgets = [_edit_budget(it, max_edits=5, min_edits=1, schedule_rounds=5, lr_schedule="linear") for it in range(1, 7)]
    assert lin_budgets == [5, 4, 3, 2, 1, 1]

    const_budgets = [_edit_budget(it, max_edits=4, min_edits=2, schedule_rounds=4, lr_schedule="constant") for it in range(1, 6)]
    assert const_budgets == [4, 4, 4, 4, 4]


# ------------------------------------------------------------------ 3. Protected section & JSON patch engine


def test_protected_section_split_join_and_patch_operations():
    raw_instr = _join_protected(
        "Rule 1: Use minimal equations.\nRule 2: Skip prose restatement.",
        "Always keep final answer in \\boxed{}.",
    )
    body, prot = _split_protected(raw_instr)
    assert "Rule 1: Use minimal equations." in body
    assert prot == "Always keep final answer in \\boxed{}."

    edits = [
        _PatchEdit(op="replace", target="Use minimal   equations.", content="Write short eqs only.", source_type="efficiency"),
        _PatchEdit(op="insert_after", target="Rule 2: Skip prose restatement.", content="Rule 3: Stop once solved.", source_type="efficiency"),
        _PatchEdit(op="append", content="Rule 4: No self-checks.", source_type="efficiency"),
        _PatchEdit(op="delete", target="Always keep final answer in \\boxed{}.", source_type="failure"),
        _PatchEdit(op="replace", target=f"{SLOW_UPDATE_START}", content="hijack", source_type="failure"),
        _PatchEdit(op="delete", target="nonexistent text", source_type="failure"),
    ]
    patched, report, n_applied = _apply_patches(raw_instr, edits, max_skill_chars=1200)
    assert n_applied == 3
    new_body, new_prot = _split_protected(patched)
    assert "Write short eqs only." in new_body
    assert "Rule 3: Stop once solved." in new_body
    assert "Rule 4: No self-checks." in new_body
    # Protected section remains byte-for-byte intact
    assert new_prot == "Always keep final answer in \\boxed{}."
    assert SLOW_UPDATE_START in patched and SLOW_UPDATE_END in patched

    statuses = [(e.op, e.status, e.reason) for e in report]
    assert statuses == [
        ("replace", "applied", ""),
        ("insert_after", "applied", ""),
        ("append", "applied", ""),
        ("delete", "skipped", "protected_section"),
        ("replace", "skipped", "protected_section"),
        ("delete", "skipped", "target_not_found"),
    ]


def test_patch_engine_enforces_max_chars_and_non_empty_guard():
    base = "Keep reasoning short."
    too_big = [_PatchEdit(op="append", content="x" * 200)]
    out, rep, n_app = _apply_patches(base, too_big, max_skill_chars=50)
    assert n_app == 0
    assert out == base
    assert rep[0].status == "skipped" and rep[0].reason == "exceeds_max_chars"

    wipe_all = [_PatchEdit(op="delete", target="Keep reasoning short.")]
    out2, rep2, n_app2 = _apply_patches(base, wipe_all, max_skill_chars=200)
    assert n_app2 == 0
    assert out2 == base
    assert rep2[0].status == "skipped" and rep2[0].reason == "would_empty_skill"

    fenced = 'Here is the patch:\n```json\n{"edits": [{"op": "append", "content": "No filler.",}],}\n```'
    parsed = _extract_json_object(fenced)
    assert parsed == {"edits": [{"op": "append", "content": "No filler."}]}


# ------------------------------------------------------------------ 4. Constrained reasoning-length validation gate


def test_gate_accepts_feasible_length_reduction_and_rejects_costlier_higher_accuracy():
    r0 = _round0()
    # cod has acc=0.75, tokens=50 (norm_cost=0.50); cot anchor has acc=0.75, tokens=100.
    # Candidate 1: preserves acc=0.75 (feasible) and cuts tokens 50 -> 30 (norm_cost=0.30) => ACCEPTED.
    shorter_feasible = _ev(
        "skillopt-i1-c0",
        {"gsm8k": (0.75, 30), "bbh": (0.75, 30)},
        "Ultra-tight eqs.",
        parent_id="cod",
        metadata={"applied_edits": "[delete] filler"},
    )
    # Candidate 2: higher accuracy (1.00 > 0.75) BUT MORE TOKENS (80 > 50) => REJECTED under acc_bonus=0.0!
    costlier_more_accurate = _ev(
        "skillopt-i1-c1",
        {"gsm8k": (1.00, 80), "bbh": (1.00, 80)},
        "Longer accurate skill.",
        parent_id="cod",
        metadata={"applied_edits": "[append] verify twice"},
    )
    gate = _replay_gate(
        [r0, [shorter_feasible, costlier_more_accurate]],
        anchor_id="cot",
        epsilon=0.02,
        acc_bonus=0.0,
        selection_ratio=0.0,
    )
    assert gate.anchor is not None and gate.anchor.ev.skill_id == "cot"
    assert gate.incumbent is not None and gate.incumbent.ev.skill_id == "skillopt-i1-c0"
    assert [a.skill_id for a in gate.accepted] == ["skillopt-i1-c0"]
    assert [r.skill_id for r in gate.all_rejected] == ["skillopt-i1-c1"]

    # When evaluated alone against cod (50 tokens), costlier_more_accurate (80 tokens) MUST still be rejected!
    gate_costlier = _replay_gate(
        [r0, [costlier_more_accurate]],
        anchor_id="cot",
        epsilon=0.02,
        acc_bonus=0.0,
        selection_ratio=0.0,
    )
    assert gate_costlier.incumbent is not None and gate_costlier.incumbent.ev.skill_id == "cod"
    assert gate_costlier.accepted == []
    assert gate_costlier.all_rejected[0].reason == "increased token cost"


def test_gate_rejects_infeasible_shortening_and_visible_only_gaming():
    r0 = [
        _ev_items(
            "cot",
            {"gsm8k": [_item(f"ex{i}", correct=(i < 4), output_tokens=120, reasoning_tokens=100) for i in range(4)]},
            "Think step by step.",
        )
    ]
    # Infeasible: super short (15 tokens), but acc=0.25 < 1.00 - 0.02 = 0.98
    infeasible_short = _ev_items(
        "skillopt-i1-c0",
        {"gsm8k": [_item(f"ex{i}", correct=(i == 0), output_tokens=15, reasoning_tokens=10) for i in range(4)]},
        "Too short.",
        parent_id="cot",
    )
    # Visible-only gaming: acc=1.00, total output dropped 120 -> 102, BUT reasoning_tokens stayed at 100!
    visible_gaming = _ev_items(
        "skillopt-i1-c1",
        {"gsm8k": [_item(f"ex{i}", correct=True, output_tokens=102, reasoning_tokens=100) for i in range(4)]},
        "Hide final prose only.",
        parent_id="cot",
    )
    # Genuine thinking reduction: acc=1.00, output dropped 120 -> 60 and reasoning dropped 100 -> 50
    genuine_short = _ev_items(
        "skillopt-i2-c0",
        {"gsm8k": [_item(f"ex{i}", correct=True, output_tokens=60, reasoning_tokens=50) for i in range(4)]},
        "Compact <think> eqs.",
        parent_id="cot",
    )

    gate = _replay_gate(
        [r0, [infeasible_short, visible_gaming], [genuine_short]],
        anchor_id="cot",
        epsilon=0.02,
        cost_metric="output",
        require_reasoning_reduction=True,
        selection_ratio=0.0,
        epoch_rounds=5,
    )
    assert gate.incumbent is not None and gate.incumbent.ev.skill_id == "skillopt-i2-c0"
    rej_map = {r.skill_id: r.reason for r in gate.all_rejected}
    assert rej_map["skillopt-i1-c0"] == "violated accuracy constraint"
    assert "visible-only shortening" in rej_map["skillopt-i1-c1"]


def test_gate_supports_per_benchmark_epsilon_and_reasoning_cost_metric():
    r0 = [
        _ev_items(
            "cot",
            {
                "gsm8k": [_item(f"g{i}", correct=True, output_tokens=100, reasoning_tokens=80) for i in range(4)],
                "bbh": [_item(f"b{i}", correct=True, output_tokens=100, reasoning_tokens=80) for i in range(4)],
            },
            "CoT anchor.",
        )
    ]
    # Mean acc = (1.0 + 0.75)/2 = 0.875 >= 1.0 - 0.15 = 0.85,
    # BUT bbh dropped by -0.25 > per_benchmark_epsilon=0.10 -> INFEASIBLE!
    skewed_drop = _ev_items(
        "skillopt-i1-c0",
        {
            "gsm8k": [_item(f"g{i}", correct=True, output_tokens=60, reasoning_tokens=40) for i in range(4)],
            "bbh": [_item(f"b{i}", correct=(i < 3), output_tokens=60, reasoning_tokens=40) for i in range(4)],
        },
        "Skewed skill.",
        parent_id="cot",
    )
    gate = _replay_gate(
        [r0, [skewed_drop]],
        anchor_id="cot",
        epsilon=0.15,
        per_benchmark_epsilon=0.10,
        cost_metric="reasoning",
        selection_ratio=0.0,
    )
    assert gate.incumbent is not None and gate.incumbent.ev.skill_id == "cot"
    assert gate.all_rejected[0].reason == "violated accuracy constraint"


# ------------------------------------------------------------------ 5. Disjoint internal D_tr / D_sel split


def test_disjoint_train_selection_split_prevents_selection_leakage():
    items = [
        _item(
            f"ex_{i:02d}",
            correct=(i % 3 != 0),
            output_tokens=40 + 20 * i,
            reasoning_tokens=30 + 15 * i,
            question=f"UNIQUE_QUESTION_TOKEN_{i:02d}",
        )
        for i in range(12)
    ]
    ev = _ev_items("cot", {"gsm8k": items}, "Think step by step.")
    train_ids, sel_ids = _partition_example_ids([[ev]], selection_ratio=0.25, split_seed=42)
    assert len(sel_ids["gsm8k"]) == 3
    assert len(train_ids["gsm8k"]) == 9
    assert sel_ids["gsm8k"].isdisjoint(train_ids["gsm8k"])
    assert sel_ids["gsm8k"] | train_ids["gsm8k"] == {f"ex_{i:02d}" for i in range(12)}

    # Ensure deterministic under split_seed
    train_ids_2, sel_ids_2 = _partition_example_ids([[ev]], selection_ratio=0.25, split_seed=42)
    assert train_ids == train_ids_2 and sel_ids == sel_ids_2

    # Reflection prompts must NEVER expose D_sel questions
    llm = _make_scripted(
        [
            (json.dumps({"patch": {"edits": [{"op": "append", "content": "Rule: use eqs."}]}}), False),
            (json.dumps({"patch": {"edits": [{"op": "append", "content": "Rule: skip prose."}]}}), False),
            (json.dumps({"edits": [{"op": "append", "content": "Rule: skip prose."}, {"op": "append", "content": "Rule: use eqs."}]}), False),
            (json.dumps({"selected_indices": [0]}), False),
        ]
    )
    opt = SkillOptOptimizer(
        llm,
        num_candidates=1,
        selection_ratio=0.25,
        split_seed=42,
        reflection_batch_size=12,
        max_failure_batches=1,
        max_success_batches=1,
        edit_mode="patch",
    )
    out = opt.update([ev], [])
    assert len(out) == 1
    assert "Rule: skip prose." in out[0].instruction
    # First 2 calls are the Stage 1 failure and efficiency analysts
    stage1_prompts = "\n".join(llm.calls[:2])
    for sel_id in sel_ids["gsm8k"]:
        idx_num = sel_id.split("_")[1]
        assert f"UNIQUE_QUESTION_TOKEN_{idx_num}" not in stage1_prompts
    # At least one D_tr item appears
    assert any(f"UNIQUE_QUESTION_TOKEN_{tr_id.split('_')[1]}" in stage1_prompts for tr_id in train_ids["gsm8k"])


# ------------------------------------------------------------------ 6. Multi-stage pipeline: reflect -> merge -> rank -> patch


def test_full_multistage_pipeline_with_parallel_minibatches_and_priority_merge():
    items = [
        _item("f1", correct=False, output_tokens=90, reasoning_tokens=80, question="FAIL_Q1", truncated=True),
        _item("f2", correct=False, output_tokens=70, reasoning_tokens=60, question="FAIL_Q2"),
        _item("f3", correct=False, output_tokens=65, reasoning_tokens=55, question="FAIL_Q3"),
        _item("v1", correct=True, output_tokens=220, reasoning_tokens=200, question="VERBOSE_Q1"),
        _item("v2", correct=True, output_tokens=180, reasoning_tokens=160, question="VERBOSE_Q2"),
        _item("c1", correct=True, output_tokens=20, reasoning_tokens=15, question="COMPACT_Q1"),
    ]
    ev = _ev_items("cot", {"gsm8k": items}, "Line A: Think carefully.\nLine B: Explain every step.")

    llm = _make_scripted(
        [
            # Stage 1 batched calls: 2 failure minibatches + 2 efficiency minibatches (reflection_batch_size=1 -> fail=2, eff=2)
            (json.dumps({"patch": {"edits": [{"op": "append", "content": "Preserve numeric constraints."}]}}), False),
            (json.dumps({"patch": {"edits": [{"op": "append", "content": "Preserve numeric constraints."}]}}), False),
            (json.dumps({"patch": {"edits": [{"op": "replace", "target": "Explain every step.", "content": "Write minimal equations only."}]}}), False),
            (json.dumps({"patch": {"edits": [{"op": "append", "content": "Stop once derived."}]}}), False),
            # Stage 2: merge_failure (2 batches), merge_efficiency (2 batches), merge_final (both streams)
            (json.dumps({"edits": [{"op": "append", "content": "Preserve numeric constraints.", "support_count": 2, "source_type": "failure"}]}), False),
            (json.dumps({"edits": [
                {"op": "replace", "target": "Explain every step.", "content": "Write minimal equations only.", "support_count": 1, "source_type": "efficiency"},
                {"op": "append", "content": "Stop once derived.", "support_count": 1, "source_type": "efficiency"},
            ]}), False),
            (json.dumps({"edits": [
                {"op": "replace", "target": "Explain every step.", "content": "Write minimal equations only.", "support_count": 1, "source_type": "efficiency"},
                {"op": "append", "content": "Stop once derived.", "support_count": 1, "source_type": "efficiency"},
                {"op": "append", "content": "Preserve numeric constraints.", "support_count": 2, "source_type": "failure"},
            ]}), False),
            # Stage 3: ranking
            (json.dumps({"selected_indices": [0, 1, 2]}), False),
        ]
    )
    opt = SkillOptOptimizer(
        llm,
        num_candidates=2,
        selection_ratio=0.0,
        reflection_batch_size=2,
        max_failure_batches=2,
        max_success_batches=2,
        max_edits=2,
        min_edits=2,
        edit_mode="patch",
    )
    out = opt.update([ev], [])
    # Verify Stage 1 ran in a single generate_batch call across failure + efficiency analysts
    assert llm.batch_sizes[0] >= 2
    assert len(out) == 2
    assert "Write minimal equations only." in out[0].instruction
    assert "Stop once derived." in out[0].instruction
    assert out[0].metadata["edit_apply_report"][0]["status"] == "applied"
    # Second candidate uses alternative subset (index 2: Preserve numeric constraints.)
    assert out[0].instruction != out[1].instruction
    assert "Preserve numeric constraints." in out[1].instruction
    # Verify efficiency analyst saw VERBOSE vs COMPACT contrast
    eff_prompt = next(p for p in llm.calls if "reasoning-efficiency analyst" in p)
    assert "VERBOSE" in eff_prompt and "COMPACT" in eff_prompt
    # Since incumbent is feasible (acc=0.60 == anchor 0.60), merge_final prioritizes efficiency
    merge_final_prompt = next(p for p in llm.calls if "FINAL hierarchical patch merge" in p)
    assert "EFFICIENCY EDITS TAKE PRIORITY" in merge_final_prompt


def test_rewrite_mode_preserves_protected_slow_update_block():
    protected_instr = _join_protected("Original body.", "Never drop sign rules.")
    ev = _ev_items(
        "cot",
        {"gsm8k": [_item("x1", correct=True, output_tokens=80, reasoning_tokens=60)]},
        protected_instr,
    )
    llm = _make_scripted(
        [
            # Stage 1 efficiency analyst (1 batch)
            (json.dumps({"patch": {"edits": [{"op": "replace", "target": "Original body.", "content": "Short eqs."}]}}), False),
            # Stage 4 rewrite output (attempts to ignore markers -> _join_protected still preserves protected block)
            ("<skill>Ultra-short symbolic body.</skill>", False),
        ]
    )
    opt = SkillOptOptimizer(
        llm,
        num_candidates=1,
        selection_ratio=0.0,
        edit_mode="rewrite",
        enable_slow_update=False,
    )
    out = opt.update([ev], [])
    assert len(out) == 1
    body, prot = _split_protected(out[0].instruction)
    assert body == "Ultra-short symbolic body."
    assert prot == "Never drop sign rules."


# ------------------------------------------------------------------ 7. Epoch slow update vs optimizer-side meta skill


def test_epoch_slow_update_and_meta_skill_separation_and_buffer_reset():
    # Build 6 D_tr items so longitudinal comparison compares prev_ev -> cur_ev item-by-item
    r0_items = [
        _item(f"ex{i}", correct=(i < 5), output_tokens=100, reasoning_tokens=80, question=f"LongQ {i}")
        for i in range(6)
    ]
    r1_win_items = [
        _item("ex0", correct=True, output_tokens=45, reasoning_tokens=30, question="LongQ 0"),   # efficiency win
        _item("ex1", correct=True, output_tokens=45, reasoning_tokens=30, question="LongQ 1"),   # efficiency win
        _item("ex2", correct=True, output_tokens=45, reasoning_tokens=30, question="LongQ 2"),   # efficiency win
        _item("ex3", correct=True, output_tokens=45, reasoning_tokens=30, question="LongQ 3"),   # efficiency win
        _item("ex4", correct=False, output_tokens=40, reasoning_tokens=30, question="LongQ 4"),  # acc regression (5/6->4/6? wait keep feasible)
        _item("ex5", correct=True, output_tokens=45, reasoning_tokens=30, question="LongQ 5"),   # fixed failure! (acc=5/6=r0)
    ]
    r1_lose_items = [
        _item(f"ex{i}", correct=(i < 2), output_tokens=20, reasoning_tokens=10, question=f"LongQ {i}")
        for i in range(6)
    ]

    r0 = [_ev_items("cot", {"gsm8k": r0_items}, "Think step by step.")]
    r1 = [
        _ev_items("skillopt-i1-c0", {"gsm8k": r1_win_items}, "Use compact equations.", parent_id="cot", metadata={"applied_edits": "[replace] eqs"}),
        _ev_items("skillopt-i1-c1", {"gsm8k": r1_lose_items}, "Skip all steps.", parent_id="cot", metadata={"applied_edits": "[delete] steps"}),
    ]

    llm = _make_scripted(
        [
            # Epoch 1 boundary (iteration=2, num_candidates=2, epoch_rounds=2):
            # 1) meta_skill update
            (json.dumps({"reasoning": "eqs worked", "meta_skill_content": "META_COACH_ONLY: Prefer eqs; never delete step state."}), False),
            # 2) Stage 1 analysts (1 fail batch + 1 eff batch in single generate_batch call)
            (json.dumps({"patch": {"edits": [{"op": "append", "content": "Check edge case signs."}]}}), False),
            (json.dumps({"patch": {"edits": [{"op": "append", "content": "Drop prose transitions."}]}}), False),
            # 3) Stage 2 merge_final
            (json.dumps({"edits": [
                {"op": "append", "content": "Drop prose transitions.", "support_count": 1, "source_type": "efficiency"},
                {"op": "append", "content": "Check edge case signs.", "support_count": 1, "source_type": "failure"},
            ]}), False),
            # 4) Stage 3 ranking
            (json.dumps({"selected_indices": [0, 1]}), False),
            # 5) Epoch slow_update candidate
            (json.dumps({"reasoning": "longitudinal check", "slow_update_content": "SLOW_GUIDANCE: Track variable signs concisely."}), False),
        ]
    )

    opt = SkillOptOptimizer(
        llm,
        num_candidates=2,
        epoch_rounds=2,
        selection_ratio=0.0,
        reflection_batch_size=8,
        enable_slow_update=True,
        enable_meta_skill=True,
        reset_rejected_on_epoch=True,
    )
    out = opt.update(r1, [r0])
    assert len(out) == 2
    fast_skill, slow_skill = out[0], out[1]
    assert fast_skill.origin is not None and fast_skill.origin.operation == "skillopt_edit"
    assert slow_skill.origin is not None and slow_skill.origin.operation == "skillopt_slow"
    assert slow_skill.metadata["slow_update"] is True

    # Optimizer-side meta skill MUST appear in optimizer prompts, NEVER in deployed Skill.instruction!
    assert "META_COACH_ONLY" in opt._meta_skill
    assert any("META_COACH_ONLY" in p for p in llm.calls[1:])
    assert "META_COACH_ONLY" not in fast_skill.instruction
    assert "META_COACH_ONLY" not in slow_skill.instruction

    # Protected slow guidance MUST live inside SLOW_UPDATE markers on slow_skill
    slow_body, slow_prot = _split_protected(slow_skill.instruction)
    assert slow_body == "Use compact equations."
    assert slow_prot == "SLOW_GUIDANCE: Track variable signs concisely."

    # Longitudinal comparison must have reported both regression and improvement without extra rollouts
    slow_prompt = llm.calls[-1]
    assert "Accuracy regressions: 1" in slow_prompt
    assert "Improvements: 5" in slow_prompt

    # Evaluate R2 (Epoch 1 boundary outputs) and confirm epoch_rejected resets for Epoch 2 (iteration=3)
    r2 = [
        _ev_items("skillopt-i2-c0", {"gsm8k": r1_win_items}, fast_skill.instruction, parent_id="skillopt-i1-c0"),
        _ev_items("skillopt-i2-c1", {"gsm8k": r1_win_items}, slow_skill.instruction, parent_id="skillopt-i1-c0"),
    ]
    gate_ep2 = _replay_gate(
        [r0, r1, r2],
        anchor_id="cot",
        epsilon=0.02,
        selection_ratio=0.0,
        epoch_rounds=2,
        num_candidates=2,
        enable_slow_update=True,
        reset_rejected_on_epoch=True,
    )
    # R1 rejected skillopt-i1-c1 and R2 rejected ties, so all_rejected is non-empty,
    # while epoch_rejected was cleanly reset at the Epoch 1 -> Epoch 2 boundary!
    assert len(gate_ep2.all_rejected) >= 1
    assert gate_ep2.epoch_rejected == []


# ------------------------------------------------------------------ 8. Retries, hash deduplication, ID collisions, non-mutation


def test_retries_on_empty_duplicate_truncated_too_long_and_warning(caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    r0 = _round0()
    llm = _make_scripted(
        [
            ("<skill>   </skill>", False),                  # c0 attempt 1: empty
            ("<skill>Draft briefly GAMMA.</skill>", False), # c0 attempt 2: duplicate
            ("<skill>Fresh skill X.</skill>", False),       # c0 attempt 3: ok
            ("<think>unfinished", True),                    # c1 attempt 1: truncated
            ("<skill>" + ("a" * 80) + "</skill>", False),   # c1 attempt 2: too_long
            ("<skill>Fresh skill X.</skill>", False),       # c1 attempt 3: duplicate
        ]
    )
    opt = SkillOptOptimizer(
        llm,
        num_candidates=2,
        selection_ratio=0.0,
        max_failure_batches=0,
        max_success_batches=0,
        edit_mode="rewrite",
        max_retries=2,
        max_skill_chars=40,
    )
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

    _, opt = _opt(
        ["<skill>Brand new.</skill>"],
        num_candidates=1,
        selection_ratio=0.0,
        max_failure_batches=0,
        max_success_batches=0,
        edit_mode="rewrite",
        enable_slow_update=False,
    )
    ids_r = [e.skill_id for e in results]
    ids_h = [[e.skill_id for e in rnd] for rnd in history]
    out = opt.update(results, history)
    assert [s.id for s in out] == ["skillopt-i2-c0-2"]
    assert [e.skill_id for e in results] == ids_r
    assert [[e.skill_id for e in rnd] for rnd in history] == ids_h

    _, opt1 = _opt(
        ["<skill>Another new.</skill>"],
        num_candidates=1,
        selection_ratio=0.0,
        max_failure_batches=0,
        max_success_batches=0,
        edit_mode="rewrite",
        enable_slow_update=False,
    )
    assert [s.id for s in opt1.update(r0, [])] == ["skillopt-i1-c0-2"]


# ------------------------------------------------------------------ 9. E2E optimize loop & CLI --mock


def test_end_to_end_optimize_loop_with_mock():
    target = make_mock(role="target")
    opt_llm = make_mock(
        [
            # Iteration 1: 1 failure analyst (0/6 correct on D_tr) + slot 1 rewrite fallback
            json.dumps({"patch": {"edits": [{"op": "append", "content": "Use short equations only."}]}}),
            "<skill>Fallback diverse draft.</skill>",
            # Iteration 2 (epoch boundary; i1-c0 got 1/6 correct -> both failure + efficiency analysts run):
            json.dumps({"meta_skill_content": "Prefer symbolic rules."}),
            json.dumps({"patch": {"edits": [{"op": "append", "content": "Keep steps short."}]}}),
            json.dumps({"patch": {"edits": [{"op": "append", "content": "No self-checks."}]}}),
            json.dumps({"edits": [
                {"op": "append", "content": "No self-checks.", "source_type": "efficiency"},
                {"op": "append", "content": "Keep steps short.", "source_type": "failure"},
            ]}),
            json.dumps({"selected_indices": [0, 1]}),
            json.dumps({"slow_update_content": "Verify final number format."}),
        ],
        role="optimizer",
    )
    bench = build_benchmark({"name": "gsm8k", "source": "local", "data_dir": str(GSM8K_DIR), "train": 8, "val": 2})
    evaluator = Evaluator(target, [BenchmarkSpec(key="gsm8k", benchmark=bench)], batch_size=8)
    optimizer = SkillOptOptimizer(opt_llm, num_candidates=2, epoch_rounds=2, selection_ratio=0.25)

    loop_res = optimize(optimizer, evaluator, max_iterations=2)
    assert loop_res.stop_reason == "max_iterations"
    assert len(loop_res.history) == 3
    assert [ev.skill_id for ev in loop_res.history[0]] == ["direct", "cot", "cod"]
    assert [ev.skill_id for ev in loop_res.history[1]] == ["skillopt-i1-c0", "skillopt-i1-c1"]
    assert [ev.skill_id for ev in loop_res.history[2]] == ["skillopt-i2-c0", "skillopt-i2-c1"]
    assert loop_res.history[2][1].skill.origin.operation == "skillopt_slow"
    assert SLOW_UPDATE_START in loop_res.history[2][1].skill.instruction


def test_cli_mock_end_to_end_with_skillopt(tmp_path):
    run_dir = cli_main(
        [
            "--mock",
            "--set",
            "optimizer.name=skillopt",
            "--set",
            "max_iterations=2",
            "--set",
            f"run.root={tmp_path.as_posix()}",
        ]
    )
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "report.html").is_file()
    assert (run_dir / "rounds" / "002" / "summary.json").is_file()
