"""SkillOpt optimizer (``skillopt``): bounded reflective skill evolution with validation gating.

Adapts *SkillOpt: Executive Strategy for Self-Evolving Agent Skills* (Yang et al., NeurIPS 2026)
to reasoning-skill optimization for a frozen target LLM:

    min_s E[T(M, x, s)]   s.t.   A(M, s) >= A_anchor - epsilon

Core modules:
1. Rollout + reflection minibatches separating failure fixes from verbose/compact successes.
2. Two-stage reflect -> merge/rank/apply under a cosine-annealed atomic edit budget ``L_t``
   (``ADD`` / ``DELETE`` / ``REPLACE``), or optional single-stage mode (``two_stage=False``).
3. Validation gate + rejected-edit buffer: replays history deterministically, promotes a
   candidate to incumbent only on strict selection-score improvement, and stores rejected edits
   as negative memory.
4. Epoch-wise slow / meta-skill update: periodically consolidates durable rules and failure traps
   into an executive strategy guiding future edits.
5. Lineage tracking via ``SkillOrigin`` (``initial``, ``skillopt_edit``, ``skillopt_meta``).
"""
from __future__ import annotations

import dataclasses
import logging
import math
import re
from dataclasses import dataclass

from skillopt.baselines import BASELINE_NAMES, get_baseline
from skillopt.evaluator import EvalResult, ItemResult, SkillEvaluation
from skillopt.llm import BaseLLM
from skillopt.optimizer.base import Optimizer, register_optimizer
from skillopt.skills import FewShotExample, Skill, SkillOrigin

logger = logging.getLogger(__name__)

_SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
_EDITS_RE = re.compile(r"<edits>(.*?)</edits>", re.DOTALL | re.IGNORECASE)
_META_RE = re.compile(r"<meta>(.*?)</meta>", re.DOTALL | re.IGNORECASE)
_REFLECTION_RE = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL | re.IGNORECASE)
_INSTR_WRAP_RE = re.compile(r"^\s*<instruction>(.*)</instruction>\s*$", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)

_TOL = 1e-9
_INVALID_PLACEHOLDERS = {"...", "<your skill>", "<skill>", "none", "n/a"}


# ------------------------------------------------------------------ validation & text helpers


def _check_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return value


def _check_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an int > 0, got {value!r}")
    return value


def _check_non_negative_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
        raise ValueError(f"{name} must be a float >= 0, got {value!r}")
    return float(value)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _compact_preview(text: str, limit: int = 180) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


def _clean_instruction(raw: str) -> str:
    s = (raw or "").strip()
    m_fence = _FENCE_RE.match(s)
    if m_fence:
        s = m_fence.group(1).strip()
    m_wrap = _INSTR_WRAP_RE.match(s)
    if m_wrap:
        s = m_wrap.group(1).strip()
    return s


def _cosine_edit_budget(iteration: int, max_edits: int, min_edits: int, schedule_rounds: int) -> int:
    """Cosine-annealed atomic edit budget ``L_t`` decaying from ``max_edits`` down to ``min_edits``."""
    if schedule_rounds <= 1 or max_edits <= min_edits:
        return min_edits
    t = min(max(iteration - 1, 0), schedule_rounds - 1) / (schedule_rounds - 1)
    val = round(min_edits + 0.5 * (max_edits - min_edits) * (1.0 + math.cos(math.pi * t)))
    return max(min_edits, min(max_edits, int(val)))


# ------------------------------------------------------------------ scoring & validation gate


@dataclass(frozen=True)
class _ScoredEval:
    round_idx: int
    ev: SkillEvaluation
    mean_acc: float
    norm_cost: float
    mean_output_tokens: float
    mean_reasoning_tokens: float
    acc_delta: float
    feasible: bool
    score: float
    bench_acc: dict[str, float]
    bench_cost: dict[str, float]


@dataclass(frozen=True)
class _RejectedRecord:
    round_idx: int
    skill_id: str
    parent_id: str
    d_acc: float
    d_norm_cost: float
    reason: str
    summary: str


@dataclass(frozen=True)
class _AcceptedRecord:
    round_idx: int
    skill_id: str
    parent_id: str
    d_acc: float
    d_norm_cost: float
    summary: str


def _eval_summary(ev: SkillEvaluation) -> tuple[float, float, float]:
    active: list[EvalResult] = [r for r in ev.results.values() if r.n > 0]
    if not active:
        return 0.0, 0.0, 0.0
    k = len(active)
    mean_acc = sum(r.accuracy for r in active) / k
    mean_out = sum(r.avg_output_tokens for r in active) / k
    mean_think = sum(r.avg_reasoning_tokens for r in active) / k
    return mean_acc, mean_out, mean_think


def _find_anchor(rounds: list[list[SkillEvaluation]], anchor_id: str) -> SkillEvaluation | None:
    for rnd in rounds:
        for ev in rnd:
            if ev.skill_id == anchor_id and any(r.n > 0 for r in ev.results.values()):
                return ev
    for rnd in rounds:
        for ev in rnd:
            if ev.skill_id == anchor_id:
                return ev
    pool = list(rounds[0]) if rounds and rounds[0] else [ev for rnd in rounds for ev in rnd]
    if not pool:
        return None
    return min(pool, key=lambda ev: (-_eval_summary(ev)[0], _eval_summary(ev)[1], ev.skill_id))


def _score_eval(
    round_idx: int,
    ev: SkillEvaluation,
    anchor_mean_acc: float,
    anchor_cost: dict[str, float],
    fallback_cost: dict[str, float],
    epsilon: float,
    token_weight: float,
) -> _ScoredEval:
    active = [(b, r) for b, r in ev.results.items() if r.n > 0]
    if not active:
        return _ScoredEval(
            round_idx=round_idx,
            ev=ev,
            mean_acc=0.0,
            norm_cost=1.0,
            mean_output_tokens=0.0,
            mean_reasoning_tokens=0.0,
            acc_delta=-anchor_mean_acc,
            feasible=False,
            score=-1.0,
            bench_acc={},
            bench_cost={},
        )
    mean_acc, mean_out, mean_think = _eval_summary(ev)
    bench_acc = {b: float(r.accuracy) for b, r in active}
    bench_cost = {b: float(r.avg_output_tokens) for b, r in active}
    ratios = [
        bench_cost[b] / max(anchor_cost.get(b, fallback_cost.get(b, 1.0)), 1.0)
        for b, _ in active
    ]
    norm_cost = sum(ratios) / len(ratios)
    acc_delta = mean_acc - anchor_mean_acc
    feasible = acc_delta >= -epsilon - _TOL
    if feasible:
        score = 10.0 + token_weight * (1.0 - norm_cost) + mean_acc
    else:
        score = mean_acc - 0.05 * token_weight * norm_cost
    return _ScoredEval(
        round_idx=round_idx,
        ev=ev,
        mean_acc=mean_acc,
        norm_cost=norm_cost,
        mean_output_tokens=mean_out,
        mean_reasoning_tokens=mean_think,
        acc_delta=acc_delta,
        feasible=feasible,
        score=score,
        bench_acc=bench_acc,
        bench_cost=bench_cost,
    )


def _edit_summary_for(ev: SkillEvaluation) -> str:
    sk = ev.skill
    meta = sk.metadata if isinstance(sk.metadata, dict) else {}
    applied = meta.get("applied_edits")
    if isinstance(applied, str) and applied.strip():
        return _compact_preview(applied, 160)
    if sk.origin is not None and sk.origin.note.strip():
        return _compact_preview(sk.origin.note, 160)
    return _compact_preview(sk.instruction, 160)


def _replay_gate(
    rounds: list[list[SkillEvaluation]],
    anchor_id: str,
    epsilon: float,
    token_weight: float,
    min_improvement: float,
    max_rejected_buffer: int,
) -> tuple[_ScoredEval | None, _ScoredEval | None, list[_ScoredEval], list[_AcceptedRecord], list[_RejectedRecord]]:
    """Replay all rounds deterministically to compute incumbent, accepted edits, and rejected buffer."""
    anchor_ev = _find_anchor(rounds, anchor_id)
    if anchor_ev is None:
        return None, None, [], [], []

    anchor_mean_acc, _, _ = _eval_summary(anchor_ev)
    anchor_cost: dict[str, float] = {
        b: max(float(r.avg_output_tokens), 1.0)
        for b, r in anchor_ev.results.items()
        if r.n > 0
    }
    fallback_cost: dict[str, float] = {}
    for rnd in rounds:
        for ev in rnd:
            for b, r in ev.results.items():
                if r.n > 0:
                    fallback_cost[b] = max(fallback_cost.get(b, 1.0), max(float(r.avg_output_tokens), 1.0))

    scored_rounds: list[list[_ScoredEval]] = []
    all_scored: list[_ScoredEval] = []
    anchor_scored: _ScoredEval | None = None
    for r_idx, rnd in enumerate(rounds):
        scored_rnd: list[_ScoredEval] = []
        for ev in rnd:
            sc = _score_eval(
                r_idx, ev, anchor_mean_acc, anchor_cost, fallback_cost, epsilon, token_weight
            )
            scored_rnd.append(sc)
            all_scored.append(sc)
            if ev is anchor_ev:
                anchor_scored = sc
        scored_rounds.append(scored_rnd)

    incumbent: _ScoredEval | None = None
    first_non_empty = next((sr for sr in scored_rounds if sr), [])
    if first_non_empty:
        incumbent = max(first_non_empty, key=lambda s: (s.score, s.mean_acc, -s.norm_cost))

    accepted: list[_AcceptedRecord] = []
    rejected: list[_RejectedRecord] = []
    for r_idx in range(1, len(scored_rounds)):
        cands = scored_rounds[r_idx]
        if not cands:
            continue
        if incumbent is None:
            incumbent = max(cands, key=lambda s: (s.score, s.mean_acc, -s.norm_cost))
            continue

        prev_inc = incumbent
        best = max(cands, key=lambda s: (s.score, s.mean_acc, -s.norm_cost))
        promoted = best if best.score > prev_inc.score + min_improvement else None
        if promoted is not None:
            incumbent = promoted
            accepted.append(
                _AcceptedRecord(
                    round_idx=r_idx,
                    skill_id=promoted.ev.skill_id,
                    parent_id=promoted.ev.skill.parent_id or prev_inc.ev.skill_id,
                    d_acc=promoted.mean_acc - prev_inc.mean_acc,
                    d_norm_cost=promoted.norm_cost - prev_inc.norm_cost,
                    summary=_edit_summary_for(promoted.ev),
                )
            )

        for cand in cands:
            if cand is promoted:
                continue
            d_acc = cand.mean_acc - prev_inc.mean_acc
            d_cost = cand.norm_cost - prev_inc.norm_cost
            if not cand.feasible and prev_inc.feasible:
                reason = "violated accuracy gate"
            elif d_acc < -_TOL:
                reason = "accuracy regressed"
            elif d_cost > _TOL:
                reason = "higher token cost"
            else:
                reason = "no strict gate gain"
            rejected.append(
                _RejectedRecord(
                    round_idx=r_idx,
                    skill_id=cand.ev.skill_id,
                    parent_id=cand.ev.skill.parent_id or prev_inc.ev.skill_id,
                    d_acc=d_acc,
                    d_norm_cost=d_cost,
                    reason=reason,
                    summary=_edit_summary_for(cand.ev),
                )
            )

    if max_rejected_buffer == 0:
        rejected = []
    elif len(rejected) > max_rejected_buffer:
        rejected = rejected[-max_rejected_buffer:]

    return anchor_scored, incumbent, all_scored, accepted, rejected


# ------------------------------------------------------------------ rollout minibatches


def _item_trace(it: ItemResult, max_chars: int) -> str:
    think = (it.reasoning_text or "").strip()
    vis = (it.response_text or "").strip()
    if think and vis:
        combined = f"<think>{think}</think> {vis}"
    else:
        combined = think or vis
    if max_chars <= 0:
        return ""
    if len(combined) > max_chars:
        return combined[:max_chars] + "..."
    return combined


def _collect_failures(ev: SkillEvaluation, limit: int, max_trace_chars: int) -> list[str]:
    if limit <= 0:
        return []
    buckets: list[tuple[str, list[ItemResult]]] = []
    for bench, res in ev.results.items():
        if res.n <= 0:
            continue
        wrong = [it for it in res.items if not it.correct]
        if wrong:
            wrong.sort(
                key=lambda it: (
                    0 if it.truncated else 1,
                    0 if it.prediction is None else 1,
                    it.example_id,
                )
            )
            buckets.append((bench, wrong))
    if not buckets:
        return []

    lines: list[str] = []
    idx = 0
    while len(lines) < limit:
        progressed = False
        for bench, wrong in buckets:
            if idx < len(wrong):
                it = wrong[idx]
                trunc = " [TRUNCATED]" if it.truncated else ""
                lines.append(
                    f"[{bench} #{len(lines) + 1}]{trunc} Q: {_compact_preview(it.question, 220)}\n"
                    f"  Prediction: {it.prediction!r} | Gold: {it.gold!r}\n"
                    f"  Trace: {_item_trace(it, max_trace_chars)}"
                )
                progressed = True
                if len(lines) >= limit:
                    break
        if not progressed:
            break
        idx += 1
    return lines


def _collect_successes(
    ev: SkillEvaluation,
    num_successes: int,
    max_trace_chars: int,
) -> tuple[list[str], list[str]]:
    """Return ``(verbose_success_lines, compact_success_lines)`` from ``ev``."""
    if num_successes <= 0:
        return [], []
    correct_items: list[tuple[str, ItemResult]] = []
    for bench, res in ev.results.items():
        if res.n <= 0:
            continue
        for it in res.items:
            if it.correct:
                correct_items.append((bench, it))
    if not correct_items:
        return [], []

    by_lowest = sorted(
        correct_items,
        key=lambda pair: (pair[1].usage.output_tokens, pair[1].usage.reasoning_tokens, pair[0], pair[1].example_id),
    )
    compact_pair = by_lowest[0]
    compact_lines = [
        f"[{compact_pair[0]}] out={compact_pair[1].usage.output_tokens} "
        f"(think={compact_pair[1].usage.reasoning_tokens}) | Q: {_compact_preview(compact_pair[1].question, 200)}\n"
        f"  Trace: {_item_trace(compact_pair[1], max(120, max_trace_chars // 2))}"
    ]

    verbose_cap = max(1, num_successes - 1)
    by_highest = sorted(
        correct_items,
        key=lambda pair: (-pair[1].usage.output_tokens, -pair[1].usage.reasoning_tokens, pair[0], pair[1].example_id),
    )
    verbose_lines: list[str] = []
    for bench, it in by_highest:
        if len(correct_items) > 1 and bench == compact_pair[0] and it is compact_pair[1]:
            continue
        verbose_lines.append(
            f"[{bench}] out={it.usage.output_tokens} (think={it.usage.reasoning_tokens}) "
            f"| Q: {_compact_preview(it.question, 200)}\n"
            f"  Trace: {_item_trace(it, max_trace_chars)}"
        )
        if len(verbose_lines) >= verbose_cap:
            break
    return verbose_lines, compact_lines


def _build_reflection_context(
    incumbent: _ScoredEval,
    anchor: _ScoredEval | None,
    rejected: list[_RejectedRecord],
    meta_strategy: str,
    num_failures: int,
    num_successes: int,
    max_trace_chars: int,
    epsilon: float,
) -> str:
    per_bench = ", ".join(
        f"{b}: acc={incumbent.bench_acc[b]:.3f}, out={incumbent.bench_cost[b]:.1f}"
        for b in incumbent.bench_acc
    )
    anchor_line = (
        f"Anchor '{anchor.ev.skill_id}': mean_acc={anchor.mean_acc:.3f}, "
        f"mean_out={anchor.mean_output_tokens:.1f}; accuracy floor = {anchor.mean_acc - epsilon:.3f}."
        if anchor is not None
        else "Anchor: unavailable."
    )
    sections: list[str] = [
        "Objective: minimize total output tokens (including internal thinking tokens) while keeping "
        f"accuracy >= anchor - {epsilon:.2f}.",
        anchor_line,
        (
            f"Current incumbent skill '{incumbent.ev.skill_id}' "
            f"(mean_acc={incumbent.mean_acc:.3f}, delta={incumbent.acc_delta:+.3f}, "
            f"norm_cost={incumbent.norm_cost:.2f}x, out={incumbent.mean_output_tokens:.1f}, "
            f"think={incumbent.mean_reasoning_tokens:.1f}, score={incumbent.score:.3f}):\n"
            f"<instruction>\n{incumbent.ev.skill.instruction.strip()}\n</instruction>"
            + (f"\nPer benchmark: {per_bench}" if per_bench else "")
        ),
    ]

    if meta_strategy.strip():
        sections.append(f"Executive Meta-Strategy (slow epoch guidance):\n{meta_strategy.strip()}")

    if rejected:
        rej_lines = [
            f"- round {r.round_idx} {r.skill_id} (from {r.parent_id}): "
            f"acc {r.d_acc:+.3f}, norm_cost {r.d_norm_cost:+.2f}x [{r.reason}] — {r.summary}"
            for r in rejected
        ]
        sections.append(
            "Rejected-edit negative memory (do NOT repeat these failed directions):\n" + "\n".join(rej_lines)
        )

    failures = _collect_failures(incumbent.ev, num_failures, max_trace_chars)
    if failures:
        sections.append("Failure cases (fix missing constraints or formatting errors):\n" + "\n\n".join(failures))

    verbose_ok, compact_ok = _collect_successes(incumbent.ev, num_successes, max_trace_chars)
    if verbose_ok:
        sections.append("Verbose success cases (trim redundant reasoning/prose waste):\n" + "\n\n".join(verbose_ok))
    if compact_ok:
        sections.append("Compact success cases (preserve these winning behaviors):\n" + "\n\n".join(compact_ok))

    return "\n\n".join(sections)


# ------------------------------------------------------------------ candidate parsing


def _parse_candidate(
    text: str | None,
    truncated: bool,
    seen_norm: set[str],
    max_skill_chars: int,
) -> tuple[str, str, str | None]:
    """Return ``(instruction, edits_summary, rejection_reason_or_None)``."""
    raw = text or ""
    edit_blocks = _EDITS_RE.findall(raw)
    edits_summary = _compact_preview(edit_blocks[-1], 240) if edit_blocks else ""

    skill_blocks = _SKILL_RE.findall(raw)
    if skill_blocks:
        instruction = _clean_instruction(skill_blocks[-1])
    elif truncated:
        return "", edits_summary, "truncated"
    else:
        stripped = _REFLECTION_RE.sub("", _EDITS_RE.sub("", raw))
        instruction = _clean_instruction(stripped)

    norm = _normalize_text(instruction)
    if not norm or norm in _INVALID_PLACEHOLDERS:
        return "", edits_summary, "empty"
    if len(instruction) > max_skill_chars:
        return "", edits_summary, "too_long"
    if norm in seen_norm:
        return "", edits_summary, "duplicate"
    return instruction, edits_summary, None


# ------------------------------------------------------------------ optimizer


@register_optimizer("skillopt")
class SkillOptOptimizer(Optimizer):
    """SkillOpt reflective bounded-edit optimizer with validation gating and epoch meta-updates."""

    def __init__(
        self,
        llm: BaseLLM,
        initial: list[str] | tuple[str, ...] = ("direct", "cot", "cod"),
        anchor_id: str = "cot",
        epsilon: float = 0.02,
        token_weight: float = 0.25,
        min_improvement: float = 1e-6,
        num_candidates: int = 2,
        max_edits: int = 4,
        min_edits: int = 2,
        edit_schedule_rounds: int = 4,
        num_failures: int = 4,
        num_successes: int = 3,
        max_trace_chars: int = 500,
        max_skill_chars: int = 1200,
        max_rejected_buffer: int = 6,
        epoch_rounds: int = 2,
        two_stage: bool = True,
        max_retries: int = 2,
        **gen_kwargs,
    ) -> None:
        if isinstance(initial, str) or not isinstance(initial, (list, tuple)) or not initial:
            raise ValueError(f"initial must be a non-empty list/tuple of baseline names, got {initial!r}")
        init_list = [str(name) for name in initial]
        seen_init: set[str] = set()
        for name in init_list:
            if name not in BASELINE_NAMES:
                raise ValueError(f"unknown baseline '{name}'; available: {', '.join(BASELINE_NAMES)}")
            if name in seen_init:
                raise ValueError(f"duplicate initial baseline '{name}'")
            seen_init.add(name)

        if not isinstance(anchor_id, str) or not anchor_id.strip():
            raise ValueError(f"anchor_id must be a non-empty string, got {anchor_id!r}")

        epsilon_f = _check_non_negative_float(epsilon, "epsilon")
        token_weight_f = _check_non_negative_float(token_weight, "token_weight")
        min_imp_f = _check_non_negative_float(min_improvement, "min_improvement")
        num_cand_i = _check_non_negative_int(num_candidates, "num_candidates")
        max_edits_i = _check_positive_int(max_edits, "max_edits")
        min_edits_i = _check_positive_int(min_edits, "min_edits")
        if min_edits_i > max_edits_i:
            raise ValueError(f"min_edits ({min_edits_i}) must be <= max_edits ({max_edits_i})")
        edit_sched_i = _check_positive_int(edit_schedule_rounds, "edit_schedule_rounds")
        num_fail_i = _check_non_negative_int(num_failures, "num_failures")
        num_succ_i = _check_non_negative_int(num_successes, "num_successes")
        max_trace_i = _check_non_negative_int(max_trace_chars, "max_trace_chars")
        max_skill_i = _check_positive_int(max_skill_chars, "max_skill_chars")
        max_rej_i = _check_non_negative_int(max_rejected_buffer, "max_rejected_buffer")
        epoch_rounds_i = _check_positive_int(epoch_rounds, "epoch_rounds")
        two_stage_b = bool(two_stage)
        max_retries_i = _check_non_negative_int(max_retries, "max_retries")

        super().__init__(
            llm,
            initial=init_list,
            anchor_id=anchor_id,
            epsilon=epsilon_f,
            token_weight=token_weight_f,
            min_improvement=min_imp_f,
            num_candidates=num_cand_i,
            max_edits=max_edits_i,
            min_edits=min_edits_i,
            edit_schedule_rounds=edit_sched_i,
            num_failures=num_fail_i,
            num_successes=num_succ_i,
            max_trace_chars=max_trace_i,
            max_skill_chars=max_skill_i,
            max_rejected_buffer=max_rej_i,
            epoch_rounds=epoch_rounds_i,
            two_stage=two_stage_b,
            max_retries=max_retries_i,
            **gen_kwargs,
        )
        self.initial = init_list
        self.anchor_id = anchor_id
        self.epsilon = epsilon_f
        self.token_weight = token_weight_f
        self.min_improvement = min_imp_f
        self.num_candidates = num_cand_i
        self.max_edits = max_edits_i
        self.min_edits = min_edits_i
        self.edit_schedule_rounds = edit_sched_i
        self.num_failures = num_fail_i
        self.num_successes = num_succ_i
        self.max_trace_chars = max_trace_i
        self.max_skill_chars = max_skill_i
        self.max_rejected_buffer = max_rej_i
        self.epoch_rounds = epoch_rounds_i
        self.two_stage = two_stage_b
        self.max_retries = max_retries_i
        self.gen_kwargs = dict(gen_kwargs)
        self._meta_strategy: str = ""

    def initial_prompts(self) -> list[Skill]:
        return [
            dataclasses.replace(
                get_baseline(name),
                origin=SkillOrigin(
                    parents=[],
                    operation="initial",
                    note=f"anchor baseline '{name}'" if name == self.anchor_id else f"baseline '{name}'",
                ),
            )
            for name in self.initial
        ]

    # ------------------------------------------------------------------ prompting

    def _run_meta_update(
        self,
        iteration: int,
        incumbent: _ScoredEval,
        accepted: list[_AcceptedRecord],
        rejected: list[_RejectedRecord],
    ) -> str:
        acc_lines = (
            [
                f"- round {a.round_idx} {a.skill_id} (acc {a.d_acc:+.3f}, norm_cost {a.d_norm_cost:+.2f}x): {a.summary}"
                for a in accepted[-4:]
            ]
            if accepted
            else ["- none yet"]
        )
        rej_lines = (
            [
                f"- round {r.round_idx} {r.skill_id} [{r.reason}] (acc {r.d_acc:+.3f}, "
                f"norm_cost {r.d_norm_cost:+.2f}x): {r.summary}"
                for r in rejected
            ]
            if rejected
            else ["- none"]
        )
        prompt = (
            f"You are performing an Epoch-{iteration // self.epoch_rounds} slow meta-skill reflection for SkillOpt.\n\n"
            f"Current incumbent '{incumbent.ev.skill_id}' (mean_acc={incumbent.mean_acc:.3f}, "
            f"delta={incumbent.acc_delta:+.3f}, norm_cost={incumbent.norm_cost:.2f}x):\n"
            f"<instruction>\n{incumbent.ev.skill.instruction.strip()}\n</instruction>\n\n"
            "Accepted improvements so far (durable rules to preserve):\n"
            + "\n".join(acc_lines)
            + "\n\nRejected directions (traps to avoid):\n"
            + "\n".join(rej_lines)
            + "\n\nSynthesize a concise executive meta-strategy (<= 5 short bullets) covering durable rules to keep, "
            "remaining token-waste targets, and mistakes to avoid. Wrap your strategy in <meta>...</meta>."
        )
        resp = self.llm.generate(prompt, **self.gen_kwargs)
        raw = resp.text or ""
        blocks = _META_RE.findall(raw)
        if blocks:
            return _compact_preview(blocks[-1], 600)
        if resp.truncated:
            return self._meta_strategy
        return _compact_preview(raw, 600) or self._meta_strategy

    def _reflect_edits(self, reflection_ctx: str, budget: int) -> str:
        prompt = (
            "You are Stage 1 (Reflector) of SkillOpt, proposing targeted atomic edits to improve a reasoning skill.\n\n"
            f"{reflection_ctx}\n\n"
            f"Analyze the failure cases and verbose vs. compact success traces. Propose concrete atomic edits "
            f"(target <= {max(budget + 2, budget)} proposals before ranking) formatted as:\n"
            "- [ADD] <new concise rule>\n"
            "- [DELETE] <redundant or harmful rule>\n"
            "- [REPLACE] <old rule> -> <tighter rule>\n"
            "Wrap your brief diagnosis in <reflection>...</reflection> and candidate edits in <edits>...</edits>."
        )
        resp = self.llm.generate(prompt, **self.gen_kwargs)
        raw = resp.text or ""
        blocks = _EDITS_RE.findall(raw)
        if blocks:
            return blocks[-1].strip()
        return raw.strip()

    def _build_apply_prompt(
        self,
        incumbent: _ScoredEval,
        reflection_ctx: str,
        proposed_edits: str | None,
        budget: int,
        slot_idx: int,
        is_meta_slot: bool,
    ) -> str:
        if proposed_edits is not None:
            header = (
                "You are Stage 2 (Executive Editor) of SkillOpt. Merge overlapping proposals, rank them, and apply "
                f"at most L_t = {budget} atomic ADD / DELETE / REPLACE edit(s) to the incumbent instruction."
            )
            body = (
                f"{reflection_ctx}\n\n"
                f"Candidate atomic edits from Stage 1 reflection:\n<edits>\n{proposed_edits or '(none)'}\n</edits>"
            )
        else:
            header = (
                "You are SkillOpt, an executive skill optimizer. Reflect on the rollouts below and apply at most "
                f"L_t = {budget} atomic ADD / DELETE / REPLACE edit(s) to the incumbent instruction."
            )
            body = reflection_ctx

        directives = [
            f"Strict edit budget: apply 1 to {budget} atomic edit(s) only; do NOT rewrite from scratch.",
            f"Keep the final skill concise (under {min(self.max_skill_chars, 800)} chars), domain-general, "
            "and preserve any task-specific final answer format.",
        ]
        if is_meta_slot:
            directives.append(
                "META-EPOCH CONSOLIDATION: Align the incumbent skill with the Executive Meta-Strategy while "
                f"strictly obeying edit budget L_t <= {budget}."
            )
        elif slot_idx > 0:
            directives.append(
                f"Candidate diversity (slot {slot_idx + 1}): choose a distinct edit subset or wording from earlier "
                "candidates in this round."
            )

        return (
            f"{header}\n\n{body}\n\n"
            + "\n".join(f"- {d}" for d in directives)
            + "\n\nOutput format:\n"
            "<edits>\n- [ADD/DELETE/REPLACE] summary of applied atomic edits\n</edits>\n"
            "<skill>\nThe complete updated instruction only.\n</skill>"
        )

    # ------------------------------------------------------------------ update loop

    def update(self, results: list[SkillEvaluation], history: list[list[SkillEvaluation]]) -> list[Skill]:
        rounds = list(history) + [list(results)]
        iteration = len(history) + 1

        if self.num_candidates == 0:
            return []

        anchor, incumbent, all_scored, accepted, rejected = _replay_gate(
            rounds,
            self.anchor_id,
            self.epsilon,
            self.token_weight,
            self.min_improvement,
            self.max_rejected_buffer,
        )
        if incumbent is None:
            return []

        meta_triggered = False
        if iteration % self.epoch_rounds == 0 and (accepted or rejected):
            self._meta_strategy = self._run_meta_update(iteration, incumbent, accepted, rejected)
            meta_triggered = True

        budget = _cosine_edit_budget(iteration, self.max_edits, self.min_edits, self.edit_schedule_rounds)
        reflection_ctx = _build_reflection_context(
            incumbent,
            anchor,
            rejected,
            self._meta_strategy,
            self.num_failures,
            self.num_successes,
            self.max_trace_chars,
            self.epsilon,
        )

        proposed_edits = self._reflect_edits(reflection_ctx, budget) if self.two_stage else None

        seen_norm: set[str] = {_normalize_text(sc.ev.skill.instruction) for sc in all_scored}
        used_ids: set[str] = {sc.ev.skill_id for sc in all_scored}
        proposals: list[Skill] = []
        rejected_counts = {"empty": 0, "duplicate": 0, "truncated": 0, "too_long": 0}
        attempts = 0

        for slot_idx in range(self.num_candidates):
            is_meta_slot = meta_triggered and (slot_idx == self.num_candidates - 1)
            base_prompt = self._build_apply_prompt(
                incumbent, reflection_ctx, proposed_edits, budget, slot_idx, is_meta_slot
            )
            last_reason: str | None = None

            for attempt in range(1 + self.max_retries):
                attempts += 1
                prompt = base_prompt
                if attempt > 0 and last_reason is not None:
                    prompt += (
                        f"\n\nPrevious attempt was rejected ({last_reason}). "
                        f"Output a distinct, concise skill (<= {self.max_skill_chars} chars) inside <skill>...</skill>."
                    )
                resp = self.llm.generate(prompt, **self.gen_kwargs)
                instruction, edits_summary, reason = _parse_candidate(
                    resp.text, resp.truncated, seen_norm, self.max_skill_chars
                )
                if reason is not None:
                    rejected_counts[reason] += 1
                    last_reason = reason
                    logger.debug(
                        f"update round {iteration}: candidate {slot_idx} attempt {attempt + 1} rejected ({reason})"
                    )
                    continue

                seen_norm.add(_normalize_text(instruction))
                base_id = f"skillopt-i{iteration}-c{slot_idx}"
                sid, suffix = base_id, 1
                while sid in used_ids:
                    suffix += 1
                    sid = f"{base_id}-{suffix}"
                used_ids.add(sid)

                few_shot = [
                    FewShotExample(question=fs.question, answer=fs.answer)
                    for fs in incumbent.ev.skill.few_shot
                ]
                operation = "skillopt_meta" if is_meta_slot else "skillopt_edit"
                note = (
                    f"L_t={budget} edits on {incumbent.ev.skill_id} "
                    f"(acc={incumbent.mean_acc:.3f}, norm_cost={incumbent.norm_cost:.2f}x)"
                )
                if edits_summary:
                    note += f": {edits_summary}"

                proposals.append(
                    Skill(
                        id=sid,
                        instruction=instruction,
                        few_shot=few_shot,
                        parent_id=incumbent.ev.skill_id,
                        metadata={
                            "optimizer": "skillopt",
                            "iteration": iteration,
                            "edit_budget": budget,
                            "applied_edits": edits_summary,
                            "incumbent_id": incumbent.ev.skill_id,
                            "incumbent_score": round(incumbent.score, 6),
                            "gate_rejected_count": len(rejected),
                            "meta_epoch": iteration // self.epoch_rounds,
                            "optimizer_model": self.llm.model,
                        },
                        origin=SkillOrigin(parents=[incumbent.ev.skill_id], operation=operation, note=note),
                    )
                )
                break

        if len(proposals) < self.num_candidates:
            msg = (
                f"update round {iteration}: {len(proposals)}/{self.num_candidates} new prompts after {attempts} "
                f"generations (rejected: empty={rejected_counts['empty']}, "
                f"duplicate={rejected_counts['duplicate']}, truncated={rejected_counts['truncated']}, "
                f"too_long={rejected_counts['too_long']})"
            )
            if not proposals:
                msg += "; returning [] stops the loop"
            logger.warning(msg)

        return proposals
