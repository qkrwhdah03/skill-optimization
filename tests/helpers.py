"""Shared test helpers (plain functions; imported as `from helpers import ...`)."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = PROJECT_ROOT / "configs"
GSM8K_DIR = PROJECT_ROOT / "data" / "sample" / "gsm8k"
BBH_DIR = PROJECT_ROOT / "data" / "sample" / "bbh"


def uniq(prefix: str = "t") -> str:
    """Unique registry-safe name."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mock_config(responses=None, **extra) -> dict:
    cfg = {"backend": "mock", "model": "mock-model"}
    if responses is not None:
        cfg["responses"] = list(responses)
    cfg.update(extra)
    return cfg


def make_mock(responses=None, role: str = "target", **extra):
    from skillopt.llm import build_llm

    return build_llm(mock_config(responses, **extra), role=role)


def make_items(skill_id: str, n: int, n_correct: int, output_tokens: int = 10,
               input_tokens: int = 0, reasoning_tokens: int = 0) -> list:
    """First n_correct items correct, rest incorrect. Failure questions start with FAILQ-<skill_id>."""
    from skillopt.evaluator import ItemResult
    from skillopt.llm import TokenUsage

    items = []
    for i in range(n):
        correct = i < n_correct
        q = f"Q-{skill_id}-{i}" if correct else f"FAILQ-{skill_id}-{i}"
        items.append(ItemResult(
            example_id=f"ex{i}",
            prediction="1" if correct else "2",
            gold="1",
            correct=correct,
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens,
                             reasoning_tokens=reasoning_tokens),
            latency=0.01,
            response_text="The answer is 1." if correct else "The answer is 2.",
            reasoning_text="",
            question=q,
        ))
    return items


def make_result(skill_id: str, n: int, n_correct: int, output_tokens: int = 10,
                input_tokens: int = 0, benchmark: str = "testbench", split: str = "train",
                question_tag: str | None = None):
    """EvalResult with n items, the first n_correct correct.

    Failure questions are ``FAILQ-<question_tag or skill_id>-<i>``.
    """
    from skillopt.evaluator import EvalResult

    tag = question_tag if question_tag is not None else skill_id
    items = make_items(tag, n, n_correct, output_tokens, input_tokens)
    return EvalResult(skill_id=skill_id, benchmark=benchmark, split=split, items=items)


def make_skill(skill_id: str, instruction: str | None = None, few_shot=None, **kw):
    from skillopt.skills import Skill

    return Skill(id=skill_id, instruction=instruction or f"Instruction for {skill_id}.",
                 few_shot=list(few_shot or []), **kw)


def make_evaluation(skill_id: str, metrics: dict, n: int = 4,
                    split: str = "train", input_tokens: int = 0, instruction: str | None = None,
                    skill=None):
    """SkillEvaluation from ``{key: (acc, tokens)}`` or ``{key: (acc, tokens, n)}``.

    Questions are tagged ``<skill_id>@<key>`` so failures are distinguishable per benchmark
    (e.g. ``FAILQ-c3@gsm8k-2``). ``skill`` (a Skill) overrides ``skill_id`` / ``instruction``.
    """
    from skillopt.evaluator import SkillEvaluation

    if skill is None:
        skill = make_skill(skill_id, instruction)
    sid = skill.id
    results = {}
    for key, spec in metrics.items():
        acc, tokens = spec[0], spec[1]
        nn = spec[2] if len(spec) > 2 else n
        results[key] = make_result(sid, nn, int(round(acc * nn)), output_tokens=tokens,
                                   input_tokens=input_tokens, benchmark=key, split=split,
                                   question_tag=f"{sid}@{key}")
    return SkillEvaluation(skill=skill, split=split, results=results)
