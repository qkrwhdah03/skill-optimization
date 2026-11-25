"""Shared private utilities for single-LLM discrete prompt optimizers.

Prefixed with ``_`` so :func:`skillopt.optimizer.base.discover_optimizers`
does not treat this module as an optimizer plugin.
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass

from skillopt.baselines import BASELINE_NAMES, get_baseline
from skillopt.evaluator import EvalResult, ItemResult, SkillEvaluation
from skillopt.skills import FewShotExample, Skill, SkillOrigin

VALID_COST_METRICS: tuple[str, ...] = ("output", "reasoning", "total")

TOL: float = 1e-9

_SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
_INSTR_WRAP_RE = re.compile(r"^\s*<instruction>(.*)</instruction>\s*$", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_INVALID_PLACEHOLDERS = {"...", "<your skill>", "<skill>", "none", "n/a"}


def check_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return value


def check_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an int > 0, got {value!r}")
    return value


def check_non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a float >= 0, got {value!r}")
    return float(value)


def check_choice(value: object, valid: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")
    return value


def validate_initial_baselines(initial: object, anchor_id: object) -> tuple[list[str], str]:
    """Validate Round-0 baseline names and anchor identifier."""
    if isinstance(initial, str) or not isinstance(initial, (list, tuple)) or not initial:
        raise ValueError(f"initial must be a non-empty list/tuple of baseline names, got {initial!r}")
    init_list = [str(name) for name in initial]
    seen: set[str] = set()
    for name in init_list:
        if name not in BASELINE_NAMES:
            raise ValueError(f"unknown baseline '{name}'; available: {', '.join(BASELINE_NAMES)}")
        if name in seen:
            raise ValueError(f"duplicate initial baseline '{name}'")
        seen.add(name)
    if not isinstance(anchor_id, str) or not anchor_id.strip():
        raise ValueError(f"anchor_id must be a non-empty string, got {anchor_id!r}")
    return init_list, anchor_id.strip()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def stable_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def skill_signature(instruction: str, few_shot: list[FewShotExample] | tuple[FewShotExample, ...] = ()) -> str:
    """Order-preserving deduplication hash over instruction + few-shot examples."""
    parts = [normalize_text(instruction)]
    for fs in few_shot:
        parts.append(normalize_text(fs.question))
        parts.append(normalize_text(fs.answer))
    return hashlib.sha256("\n|\n".join(parts).encode("utf-8")).hexdigest()


def compact_preview(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def clean_instruction(raw: str) -> str:
    s = (raw or "").strip()
    m_fence = _FENCE_RE.match(s)
    if m_fence:
        s = m_fence.group(1).strip()
    m_wrap = _INSTR_WRAP_RE.match(s)
    if m_wrap:
        s = m_wrap.group(1).strip()
    return s


def extract_skill_candidate(
    text: str | None,
    truncated: bool,
    seen_hashes: set[str],
    max_skill_chars: int,
) -> tuple[str, str | None]:
    """Parse ``<skill>...</skill>`` and return ``(instruction, rejection_reason_or_None)``."""
    blocks = _SKILL_RE.findall(text or "")
    if blocks:
        instruction = clean_instruction(blocks[-1])
    elif truncated:
        return "", "truncated"
    else:
        instruction = clean_instruction(text or "")

    norm = normalize_text(instruction)
    if not norm or norm in _INVALID_PLACEHOLDERS:
        return "", "empty"
    if len(instruction) > max_skill_chars:
        return instruction, "too_long"
    if stable_hash(instruction) in seen_hashes:
        return instruction, "duplicate"
    return instruction, None


def initial_baseline_skills(initial: list[str], anchor_id: str) -> list[Skill]:
    return [
        dataclasses.replace(
            get_baseline(name),
            origin=SkillOrigin(
                parents=[],
                operation="initial",
                note=f"anchor baseline '{name}'" if name == anchor_id else f"baseline '{name}'",
            ),
        )
        for name in initial
    ]


def next_skill_id(prefix: str, iteration: int, slot: int, used_ids: set[str]) -> str:
    idx = slot
    while True:
        cand = f"{prefix}-i{iteration}-c{idx}"
        if cand not in used_ids:
            used_ids.add(cand)
            return cand
        idx += 1


def item_cost(item: ItemResult, cost_metric: str) -> float:
    if cost_metric == "output":
        return float(item.usage.output_tokens)
    if cost_metric == "reasoning":
        return float(item.usage.reasoning_tokens)
    if cost_metric == "total":
        total = item.usage.total_tokens
        return float(total if total is not None else item.usage.input_tokens + item.usage.output_tokens)
    raise ValueError(f"unknown cost_metric '{cost_metric}'")


def eval_result_cost(res: EvalResult, cost_metric: str) -> float:
    if cost_metric == "output":
        return float(res.avg_output_tokens)
    if cost_metric == "reasoning":
        return float(res.avg_reasoning_tokens)
    if cost_metric == "total":
        return float(res.avg_total_tokens)
    raise ValueError(f"unknown cost_metric '{cost_metric}'")


@dataclass(frozen=True)
class ScoredSkill:
    """Anchor-normalized metrics and feasibility status for one evaluated skill."""

    round_idx: int
    ev: SkillEvaluation
    mean_acc: float
    norm_cost: float
    norm_reasoning: float | None
    mean_cost_raw: float
    mean_output_tokens: float
    mean_reasoning_tokens: float
    feasible: bool
    rank_key: tuple[int, float, float, float]
    bench_acc: dict[str, float]
    bench_cost: dict[str, float]
    bench_trunc: dict[str, float]


def _pick_anchor(rounds: list[list[SkillEvaluation]], anchor_id: str, cost_metric: str) -> SkillEvaluation | None:
    for rnd in rounds:
        for ev in rnd:
            if ev.skill_id == anchor_id and any(r.n > 0 for r in ev.results.values()):
                return ev
    for rnd in rounds:
        for ev in rnd:
            if ev.skill_id == anchor_id:
                return ev
    pool = [ev for rnd in rounds for ev in rnd if any(r.n > 0 for r in ev.results.values())]
    if not pool:
        return None
    return min(
        pool,
        key=lambda ev: (
            -(sum(r.accuracy for r in ev.results.values() if r.n > 0) / len([r for r in ev.results.values() if r.n > 0])),
            sum(eval_result_cost(r, cost_metric) for r in ev.results.values() if r.n > 0),
            ev.skill_id,
        ),
    )


def score_evaluations(
    rounds: list[list[SkillEvaluation]],
    anchor_id: str,
    epsilon: float,
    cost_metric: str = "output",
) -> tuple[ScoredSkill | None, list[ScoredSkill], list[str]]:
    """Score every evaluation across ``rounds`` under anchor-normalized cost and accuracy constraint."""
    anchor_ev = _pick_anchor(rounds, anchor_id, cost_metric)
    if anchor_ev is None:
        return None, [], []

    anchor_active = [(b, r) for b, r in anchor_ev.results.items() if r.n > 0]
    anchor_cost: dict[str, float] = {b: max(eval_result_cost(r, cost_metric), 1.0) for b, r in anchor_active}
    anchor_think: dict[str, float] = {b: float(r.avg_reasoning_tokens) for b, r in anchor_active}
    anchor_acc = (sum(r.accuracy for _, r in anchor_active) / len(anchor_active)) if anchor_active else 0.0

    # Fallback cost denominator for any benchmark missing from anchor_ev
    bench_samples: dict[str, list[float]] = {}
    bench_order: list[str] = []
    for rnd in rounds:
        for ev in rnd:
            for b, r in ev.results.items():
                if r.n > 0:
                    if b not in bench_samples:
                        bench_order.append(b)
                    bench_samples.setdefault(b, []).append(max(eval_result_cost(r, cost_metric), 1.0))
    fallback_cost = {b: max(sum(vals) / len(vals), 1.0) for b, vals in bench_samples.items()}

    all_scored: list[ScoredSkill] = []
    anchor_scored: ScoredSkill | None = None

    for r_idx, rnd in enumerate(rounds):
        for ev in rnd:
            active = [(b, r) for b, r in ev.results.items() if r.n > 0]
            if not active:
                sc = ScoredSkill(
                    round_idx=r_idx,
                    ev=ev,
                    mean_acc=0.0,
                    norm_cost=1.0,
                    norm_reasoning=None,
                    mean_cost_raw=0.0,
                    mean_output_tokens=0.0,
                    mean_reasoning_tokens=0.0,
                    feasible=False,
                    rank_key=(0, 0.0, 0.0, -1.0),
                    bench_acc={},
                    bench_cost={},
                    bench_trunc={},
                )
                all_scored.append(sc)
                if ev is anchor_ev:
                    anchor_scored = sc
                continue

            b_acc = {b: float(r.accuracy) for b, r in active}
            b_cost = {b: eval_result_cost(r, cost_metric) for b, r in active}
            b_trunc = {b: float(r.truncation_rate) for b, r in active}
            k = len(active)
            mean_acc = sum(b_acc.values()) / k
            mean_cost_raw = sum(b_cost.values()) / k
            mean_out = sum(float(r.avg_output_tokens) for _, r in active) / k
            mean_think = sum(float(r.avg_reasoning_tokens) for _, r in active) / k

            ratios = [b_cost[b] / max(anchor_cost.get(b, fallback_cost.get(b, 1.0)), 1.0) for b, _ in active]
            norm_cost = sum(ratios) / len(ratios)
            think_ratios = [
                float(r.avg_reasoning_tokens) / anchor_think[b]
                for b, r in active
                if anchor_think.get(b, 0.0) > 0.0
            ]
            norm_reasoning = (sum(think_ratios) / len(think_ratios)) if think_ratios else None

            feasible = mean_acc >= anchor_acc - epsilon - TOL
            rank_key = (
                1 if feasible else 0,
                -norm_cost if feasible else mean_acc,
                mean_acc,
                -norm_cost,
            )
            sc = ScoredSkill(
                round_idx=r_idx,
                ev=ev,
                mean_acc=mean_acc,
                norm_cost=norm_cost,
                norm_reasoning=norm_reasoning,
                mean_cost_raw=mean_cost_raw,
                mean_output_tokens=mean_out,
                mean_reasoning_tokens=mean_think,
                feasible=feasible,
                rank_key=rank_key,
                bench_acc=b_acc,
                bench_cost=b_cost,
                bench_trunc=b_trunc,
            )
            all_scored.append(sc)
            if ev is anchor_ev:
                anchor_scored = sc

    return anchor_scored, all_scored, bench_order


def format_benchmark_table(scored: ScoredSkill, anchor: ScoredSkill | None) -> str:
    lines: list[str] = []
    for b in scored.bench_acc:
        acc = scored.bench_acc[b]
        cost = scored.bench_cost[b]
        trunc = scored.bench_trunc[b]
        ref = ""
        if anchor is not None and b in anchor.bench_acc:
            d_acc = (acc - anchor.bench_acc[b]) * 100.0
            ratio = cost / max(anchor.bench_cost[b], 1.0)
            ref = f" (d_acc={d_acc:+.1f}pp, cost={ratio:.2f}x)"
        lines.append(f"  - {b}: acc={acc:.3f}, avg_tok={cost:.1f}, trunc={trunc * 100:.1f}%{ref}")
    return "\n".join(lines)


def format_trace_item(bench: str, item: ItemResult, max_trace_chars: int = 450) -> str:
    status = "TRUNCATED" if item.truncated else ("OK" if item.correct else "FAIL")
    q_prev = compact_preview(item.question, max_trace_chars)
    body = (item.reasoning_text or "").strip() or (item.response_text or "").strip()
    trace_prev = compact_preview(body, max_trace_chars)
    return (
        f"[{bench} | {status} | out={item.usage.output_tokens} think={item.usage.reasoning_tokens}]\n"
        f"Q: {q_prev}\n"
        f"pred={compact_preview(str(item.prediction), 80)} | gold={compact_preview(str(item.gold), 80)}\n"
        f"trace: {trace_prev}"
    )
