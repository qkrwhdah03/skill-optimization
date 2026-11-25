"""Tests for the single-LLM discrete prompt-optimization suite (``gepa``, ``textgrad``, ``mipro``)."""
from __future__ import annotations

import pytest

from helpers import PROJECT_ROOT, make_evaluation, make_items, make_mock
from skillopt.evaluator import BenchmarkSpec, EvalResult, Evaluator, SkillEvaluation
from skillopt.llm import BaseLLM, RawGeneration, TokenUsage
from skillopt.optimizer import OPTIMIZER_REGISTRY, build_optimizer, discover_optimizers, optimize
from skillopt.skills import FewShotExample, Skill


class _ScriptedLLM(BaseLLM):
    """Deterministic test LLM that records prompts and serves scripted outputs."""

    uses_gpu = False

    def __init__(self, outputs: list[tuple[str, bool]]) -> None:
        super().__init__({"backend": "mock", "model": "scripted"}, role="optimizer")
        self._outputs = list(outputs)
        self.prompts: list[str] = []

    def _generate(self, prompt: str, gen: dict) -> RawGeneration:
        self.prompts.append(prompt)
        if not self._outputs:
            return RawGeneration(text="<skill>Compact fallback step rule.</skill>", output_tokens=8)
        text, truncated = self._outputs.pop(0)
        return RawGeneration(text=text, output_tokens=max(1, len(text.split())), truncated=truncated)

    def count_tokens(self, text: str) -> int:
        return len((text or "").split())


def _sample_rounds(skill_metrics: list[tuple[str, str, dict[str, tuple[float, int]]]]) -> list[SkillEvaluation]:
    evals: list[SkillEvaluation] = []
    for sid, instr, metrics in skill_metrics:
        evals.append(make_evaluation(sid, metrics, n=6, instruction=instr))
    return evals


# ------------------------------------------------------------------ discovery & common contracts


def test_discover_optimizers_includes_new_methods_and_skips_private():
    mods = discover_optimizers()
    assert "skillopt.optimizer.algorithms._common" not in mods
    for name in ("opro", "skillopt", "gepa", "textgrad", "mipro"):
        assert name in OPTIMIZER_REGISTRY, name


@pytest.mark.parametrize("opt_name", ["gepa", "textgrad", "mipro"])
def test_validation_and_initial_prompts_common(opt_name):
    llm = make_mock(["<skill>Compact.</skill>"], role="optimizer")
    opt = build_optimizer({"name": opt_name}, llm)
    init = opt.initial_prompts()
    assert [s.id for s in init] == ["direct", "cot", "cod"]
    assert all(s.origin is not None and s.origin.operation == "initial" for s in init)

    opt_zero = build_optimizer({"name": opt_name, "num_candidates": 0}, llm)
    round0 = _sample_rounds([("cot", "Let's think step by step.", {"gsm8k": (0.8, 400)})])
    assert opt_zero.update(round0, []) == []

    with pytest.raises(ValueError):
        build_optimizer({"name": opt_name, "cost_metric": "invalid"}, llm)
    with pytest.raises(ValueError):
        build_optimizer({"name": opt_name, "initial": []}, llm)
    with pytest.raises(ValueError):
        build_optimizer({"name": opt_name, "initial": ["unknown_baseline"]}, llm)
    with pytest.raises(ValueError):
        build_optimizer({"name": opt_name, "anchor_id": ""}, llm)


@pytest.mark.parametrize("opt_name", ["gepa", "textgrad", "mipro"])
def test_handles_all_empty_benchmarks_safely(opt_name):
    llm = make_mock(["<skill>New compact prompt.</skill>"], role="optimizer")
    opt = build_optimizer({"name": opt_name}, llm)
    empty_ev = SkillEvaluation(
        skill=Skill(id="cot", instruction="Let's think step by step."),
        split="train",
        results={"gsm8k": EvalResult(skill_id="cot", benchmark="gsm8k", split="train", items=[])},
    )
    res = opt.update([empty_ev], [])
    assert isinstance(res, list)


# ------------------------------------------------------------------ GEPA tests


def test_gepa_pareto_specialists_mutate_and_merge():
    llm = _ScriptedLLM([
        ("<skill>Hybrid: solve with short math equations and concise logic tags.</skill>", False),
    ])
    opt = build_optimizer({"name": "gepa", "num_candidates": 1, "merge_prob": 1.0, "epsilon": 0.05}, llm)
    # Two feasible specialists: A dominates gsm8k accuracy; B dominates bbh cost
    round0 = _sample_rounds([
        ("cot", "Let's think step by step.", {"gsm8k": (0.83, 600), "bbh": (0.83, 800)}),
        ("specA", "Solve math accurately.", {"gsm8k": (1.00, 400), "bbh": (0.67, 700)}),
        ("specB", "Solve logic compactly.", {"gsm8k": (0.67, 500), "bbh": (0.83, 300)}),
    ])
    merged = opt.update(round0, [])
    assert len(merged) == 1
    assert merged[0].origin is not None
    assert merged[0].origin.operation == "gepa_merge"
    assert len(merged[0].origin.parents) == 2

    # Mutation mode + retry over invalid/duplicate outputs
    llm_mut = _ScriptedLLM([
        ("", True),  # truncated -> rejected
        ("Let's think step by step.", False),  # duplicate -> rejected
        ("<skill>Use 4-word steps, keep key signs.</skill>", False),
    ])
    opt_mut = build_optimizer(
        {"name": "gepa", "num_candidates": 1, "merge_prob": 0.0, "max_retries": 2},
        llm_mut,
    )
    mutated = opt_mut.update(round0, [])
    assert len(mutated) == 1
    assert mutated[0].instruction == "Use 4-word steps, keep key signs."
    assert mutated[0].origin is not None
    assert mutated[0].origin.operation == "gepa_mutate"
    assert len(mutated[0].origin.parents) == 1


# ------------------------------------------------------------------ TextGrad tests


def test_textgrad_extracts_clusters_and_steps_beam():
    from skillopt.optimizer.algorithms.textgrad import cluster_texts_jaccard

    groups = cluster_texts_jaccard(
        [
            "Verbose restatement of problem facts inflates thinking tokens unnecessarily",
            "Problem facts restatement causes verbose thinking token inflation",
            "Arithmetic sign inversion during final subtraction step",
        ],
        threshold=0.25,
    )
    assert len(groups) == 2
    assert len(groups[0]) == 2

    # Scripted responses: 2 error/bloat trace gradients + 1 cluster abstraction + 1 TGD step
    llm = _ScriptedLLM([
        ("Verbose problem restatement wastes thinking tokens before solving.", False),
        ("Problem restatement inflates thinking tokens without adding accuracy.", False),
        ("Avoid restating problem facts; start directly with equations.", False),
        ("<skill>Start directly with equations; skip restating problem facts.</skill>", False),
    ])
    opt = build_optimizer(
        {
            "name": "textgrad",
            "beam_width": 1,
            "num_candidates": 1,
            "num_error_grads": 1,
            "num_bloat_grads": 1,
            "num_clusters": 2,
        },
        llm,
    )
    round0 = _sample_rounds([
        ("cot", "Let's think step by step.", {"gsm8k": (0.67, 500)}),
        ("cod", "Use brief draft steps.", {"gsm8k": (0.83, 250)}),
    ])
    out = opt.update(round0, [])
    assert len(out) == 1
    assert out[0].origin is not None
    assert out[0].origin.operation == "textgrad_step"
    assert out[0].origin.parents == ["cod"]
    assert out[0].instruction == "Start directly with equations; skip restating problem facts."


# ------------------------------------------------------------------ MIPROv2 tests


def test_mipro_bootstraps_demos_and_explores_unseen_pairs():
    llm = _ScriptedLLM([
        ("<skill>Write compact symbolic steps and output only the final answer.</skill>", False),
        ("<skill>Compute key deltas cleanly in shorthand before answering.</skill>", False),
        ("<skill>List critical constraints briefly, skip verification prose.</skill>", False),
    ])
    opt = build_optimizer(
        {
            "name": "mipro",
            "num_candidates": 2,
            "num_instructions": 5,
            "num_demo_sets": 3,
            "max_demos": 2,
            "max_demo_chars": 120,
        },
        llm,
    )

    items = make_items("cod", 4, 4, output_tokens=25, reasoning_tokens=15)
    # Inject verbose <think> in one item response to verify _strip_think_for_demo cleans it
    items[0].response_text = "<think>long bloated internal reasoning</think>Answer: 1"
    ev_cot = SkillEvaluation(
        skill=Skill(id="cot", instruction="Let's think step by step."),
        split="train",
        results={"gsm8k": EvalResult(skill_id="cot", benchmark="gsm8k", split="train", items=items)},
    )
    ev_cod = SkillEvaluation(
        skill=Skill(id="cod", instruction="Use 5 words per step."),
        split="train",
        results={
            "gsm8k": EvalResult(
                skill_id="cod",
                benchmark="gsm8k",
                split="train",
                items=make_items("cod", 4, 4, output_tokens=12, reasoning_tokens=6),
            )
        },
    )

    proposals = opt.update([ev_cot, ev_cod], [])
    assert len(proposals) == 2
    for sk in proposals:
        assert sk.origin is not None
        assert sk.origin.operation == "mipro_search"
        assert "instruction_idx" in sk.metadata and "demo_set_idx" in sk.metadata
        if sk.few_shot:
            assert all(isinstance(fs, FewShotExample) for fs in sk.few_shot)
            assert all("<think>" not in fs.answer.lower() for fs in sk.few_shot)

    # Ensure at least one proposal selects a non-empty bootstrapped demo set across UCB candidates
    assert any(len(sk.few_shot) > 0 for sk in proposals)

    # Round 2 must never repeat already evaluated (instruction, few_shot) pairs
    round1 = [
        SkillEvaluation(
            skill=sk,
            split="train",
            results={
                "gsm8k": EvalResult(
                    skill_id=sk.id,
                    benchmark="gsm8k",
                    split="train",
                    items=make_items(sk.id, 4, 4, output_tokens=10, reasoning_tokens=5),
                )
            },
        )
        for sk in proposals
    ]
    proposals2 = opt.update(round1, [[ev_cot, ev_cod]])
    assert len(proposals2) >= 1
    seen = {(s.instruction, tuple((d.question, d.answer) for d in s.few_shot)) for s in (ev_cot.skill, ev_cod.skill, *proposals)}
    for sk in proposals2:
        assert (sk.instruction, tuple((d.question, d.answer) for d in sk.few_shot)) not in seen


# ------------------------------------------------------------------ End-to-end optimize() & CLI --mock


@pytest.mark.parametrize("opt_name", ["gepa", "textgrad", "mipro"])
def test_e2e_optimize_loop_and_cli_mock(opt_name, tmp_path):
    from skillopt.benchmarks import build_benchmark
    from skillopt.cli import main as cli_main

    target_llm = make_mock(["The answer is 1."] * 64, role="target")
    opt_llm = make_mock(
        [f"<skill>{opt_name} concise rule {i}</skill>" for i in range(40)],
        role="optimizer",
    )
    bench = build_benchmark({"name": "gsm8k", "source": "local", "data_dir": "data/sample/gsm8k", "train": 4, "val": 2})
    evaluator = Evaluator(target_llm, [BenchmarkSpec(key="gsm8k", benchmark=bench)], batch_size=8)
    opt = build_optimizer({"name": opt_name, "num_candidates": 2}, opt_llm)
    res = optimize(opt, evaluator, max_iterations=2)
    assert len(res.history) >= 2

    run_dir = cli_main([
        "--mock",
        "--set", f"optimizer.name={opt_name}",
        "--set", "max_iterations=1",
        "--set", f"run.root={tmp_path.as_posix()}",
    ])
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "report.html").is_file()
