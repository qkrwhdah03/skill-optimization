"""SkillOpt optimizer (``skillopt``): constrained reasoning-length skill evolution.

Adapts *SkillOpt: Executive Strategy for Self-Evolving Agent Skills* (Yang et al.,
arXiv:2605.23904v2) from accuracy maximization to our project's custom objective of
minimizing generated reasoning/output token length under a hard accuracy constraint
relative to the anchor baseline (default ``cot``):

    min_s  E[T(M, x, s)]    s.t.    A(M, s) >= A_anchor - epsilon

Key components aligned with SkillOpt (Algorithm 1 & Appendix A):
1. Internal disjoint train / selection split (``D_tr`` / ``D_sel``) derived
   deterministically from the framework's ``train`` split: reflection sees only
   ``D_tr`` rollouts while candidate gating sees only ``D_sel``.
2. Strict lexicographic validation gate with anti-gaming guard: accepts a candidate
   only when it satisfies the accuracy constraint and strictly reduces normalized
   token cost (rejecting ties and visible-only shortening that leaves
   ``reasoning_tokens`` unreduced when thinking tokens are present).
3. Stratified rollout batching and parallel dual-stream minibatch reflection:
   ``analyst_failure`` (diagnosing accuracy errors / truncations) and
   ``analyst_efficiency`` (diagnosing ``<think>`` bloat vs. compact successes).
4. Hierarchical JSON patch merging (``merge_failure``, ``merge_efficiency``,
   ``merge_final`` with constraint-aware priority), edit ranking, and bounded
   application under a scheduled edit budget ``L_t`` (``append``, ``insert_after``,
   ``replace``, ``delete``), plus bounded rewrite fallback when needed.
5. Protected epoch-wise slow update (``<!-- SLOW_UPDATE_START --> ... <!-- SLOW_UPDATE_END -->``,
   validated through ``D_sel``) separated from optimizer-side meta-skill memory
   (``self._meta_skill``, never sent to the target model), plus an epoch-local
   rejected-edit buffer ``B`` and hash cache ``C``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import random
import re
from dataclasses import dataclass

from skillopt.baselines import BASELINE_NAMES, get_baseline
from skillopt.evaluator import EvalResult, ItemResult, SkillEvaluation
from skillopt.llm import BaseLLM
from skillopt.optimizer.base import Optimizer, register_optimizer
from skillopt.skills import FewShotExample, Skill, SkillOrigin

logger = logging.getLogger(__name__)

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

_VALID_COST_METRICS: tuple[str, ...] = ("output", "reasoning", "total")
_VALID_EDIT_MODES: tuple[str, ...] = ("patch", "rewrite")
_VALID_LR_SCHEDULES: tuple[str, ...] = ("cosine", "linear", "constant")
_VALID_PATCH_OPS: tuple[str, ...] = ("append", "insert_after", "replace", "delete")
_OP_ALIASES: dict[str, str] = {"add": "append", "remove": "delete"}

_SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)
_SLOW_RE = re.compile(r"<slow>(.*?)</slow>", re.DOTALL | re.IGNORECASE)
_META_RE = re.compile(r"<meta>(.*?)</meta>", re.DOTALL | re.IGNORECASE)
_EDITS_RE = re.compile(r"<edits>(.*?)</edits>", re.DOTALL | re.IGNORECASE)
_REFLECTION_RE = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL | re.IGNORECASE)
_INSTR_WRAP_RE = re.compile(r"^\s*<instruction>(.*)</instruction>\s*$", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^\s*```[A-Za-z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_MARKER_LINE_RE = re.compile(
    r"(?im)^\s*(?:" + re.escape(SLOW_UPDATE_START) + r"|" + re.escape(SLOW_UPDATE_END) + r")\s*$"
)

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


def _check_choice(value: object, valid: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")
    return value


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _stable_hash(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def _stable_int_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


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


# ------------------------------------------------------------------ protected section & patch engine


def _strip_protected_markers(text: str) -> str:
    cleaned = (text or "").replace(SLOW_UPDATE_START, "").replace(SLOW_UPDATE_END, "")
    cleaned = _MARKER_LINE_RE.sub("", cleaned)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _split_protected(instruction: str) -> tuple[str, str]:
    """Split ``instruction`` into ``(editable_body, protected_slow_content)``."""
    raw = instruction or ""
    start_idx = raw.find(SLOW_UPDATE_START)
    end_idx = raw.find(SLOW_UPDATE_END)
    if start_idx >= 0 and end_idx > start_idx:
        before = raw[:start_idx].strip()
        protected = raw[start_idx + len(SLOW_UPDATE_START) : end_idx].strip()
        after = raw[end_idx + len(SLOW_UPDATE_END) :].strip()
        editable = "\n\n".join(part for part in (before, after) if part)
        return editable, protected
    return _strip_protected_markers(raw), ""


def _join_protected(editable_body: str, protected_content: str) -> str:
    """Recombine ``editable_body`` and optional ``protected_content`` with markers."""
    body = _strip_protected_markers(editable_body)
    prot = _strip_protected_markers(protected_content)
    if prot:
        if body:
            return f"{body}\n\n{SLOW_UPDATE_START}\n{prot}\n{SLOW_UPDATE_END}"
        return f"{SLOW_UPDATE_START}\n{prot}\n{SLOW_UPDATE_END}"
    return body


@dataclass(frozen=True)
class _PatchEdit:
    op: str
    target: str = ""
    content: str = ""
    support_count: int = 1
    source_type: str = "failure"


@dataclass(frozen=True)
class _EditApplyEntry:
    op: str
    target: str
    status: str
    reason: str = ""
    source_type: str = ""

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "target": self.target,
            "status": self.status,
            "reason": self.reason,
            "source_type": self.source_type,
        }


def _extract_json_object(raw: str) -> dict | None:
    """Extract the first valid JSON object from ``raw``, tolerating fences and prose."""
    text = (raw or "").strip()
    if not text:
        return None
    m_fence = _FENCE_RE.match(text)
    if m_fence:
        text = m_fence.group(1).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass

    n = len(text)
    idx = 0
    while idx < n:
        start = text.find("{", idx)
        if start < 0:
            break
        depth = 0
        in_str = False
        escape = False
        for pos in range(start, n):
            ch = text[pos]
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : pos + 1]
                    for candidate in (snippet, _TRAILING_COMMA_RE.sub(r"\1", snippet)):
                        try:
                            parsed = json.loads(candidate)
                            if isinstance(parsed, dict):
                                return parsed
                        except ValueError:
                            continue
                    break
        idx = start + 1
    return None


def _parse_edits_from_json(data: dict | None, default_source: str) -> list[_PatchEdit]:
    if not isinstance(data, dict):
        return []
    patch_obj = data.get("patch")
    raw_edits = patch_obj.get("edits") if isinstance(patch_obj, dict) else data.get("edits")
    if not isinstance(raw_edits, list):
        return []

    parsed: list[_PatchEdit] = []
    for entry in raw_edits:
        if not isinstance(entry, dict):
            continue
        op_raw = str(entry.get("op", "") or "").strip().lower()
        op = _OP_ALIASES.get(op_raw, op_raw)
        if op not in _VALID_PATCH_OPS:
            continue
        target = str(entry.get("target", "") or "").strip()
        content = str(entry.get("content", "") or "").strip()
        if op == "append" and not content:
            continue
        if op in ("insert_after", "replace") and (not target or not content):
            continue
        if op == "delete" and not target:
            continue

        sup_raw = entry.get("support_count", 1)
        if isinstance(sup_raw, (int, float)) and not isinstance(sup_raw, bool):
            support_count = max(1, int(sup_raw))
        else:
            support_count = 1

        src_raw = str(entry.get("source_type", default_source) or default_source).strip().lower()
        if src_raw == "success":
            source_type = "efficiency"
        elif src_raw in ("failure", "efficiency"):
            source_type = src_raw
        else:
            source_type = default_source

        parsed.append(
            _PatchEdit(
                op=op,
                target=target if op != "append" else "",
                content=content if op != "delete" else "",
                support_count=support_count,
                source_type=source_type,
            )
        )
    return parsed


def _find_target_span(body: str, target: str) -> tuple[int, int] | None:
    if not body or not target:
        return None
    idx = body.find(target)
    if idx >= 0:
        return idx, idx + len(target)
    tokens = target.split()
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(re.escape(tok) for tok in tokens))
    matches = list(pattern.finditer(body))
    if len(matches) == 1:
        return matches[0].span()
    return None


def _apply_patches(
    skill_text: str,
    edits: list[_PatchEdit],
    max_skill_chars: int,
) -> tuple[str, list[_EditApplyEntry], int]:
    """Programmatically apply atomic edits to the editable body, protecting slow guidance."""
    body, protected = _split_protected(skill_text)
    report: list[_EditApplyEntry] = []
    applied = 0

    for edit in edits:
        if (
            SLOW_UPDATE_START in edit.target
            or SLOW_UPDATE_END in edit.target
            or SLOW_UPDATE_START in edit.content
            or SLOW_UPDATE_END in edit.content
        ):
            report.append(
                _EditApplyEntry(
                    op=edit.op,
                    target=edit.target,
                    status="skipped",
                    reason="protected_section",
                    source_type=edit.source_type,
                )
            )
            continue

        if edit.op == "append":
            new_body = f"{body}\n{edit.content}" if body else edit.content
        else:
            span = _find_target_span(body, edit.target)
            if span is None:
                in_protected = bool(protected and _find_target_span(protected, edit.target) is not None)
                report.append(
                    _EditApplyEntry(
                        op=edit.op,
                        target=edit.target,
                        status="skipped",
                        reason="protected_section" if in_protected else "target_not_found",
                        source_type=edit.source_type,
                    )
                )
                continue
            start, end = span
            if edit.op == "insert_after":
                prefix = body[:end]
                suffix = body[end:]
                sep = "" if suffix.startswith("\n") or not suffix else "\n"
                new_body = f"{prefix}\n{edit.content}{sep}{suffix}"
            elif edit.op == "replace":
                new_body = f"{body[:start]}{edit.content}{body[end:]}"
            else:  # delete
                raw_del = f"{body[:start]}{body[end:]}"
                new_body = re.sub(r"\n{3,}", "\n\n", raw_del).strip()

        if new_body == body:
            report.append(
                _EditApplyEntry(
                    op=edit.op,
                    target=edit.target,
                    status="skipped",
                    reason="no_op",
                    source_type=edit.source_type,
                )
            )
            continue

        if not _strip_protected_markers(new_body):
            report.append(
                _EditApplyEntry(
                    op=edit.op,
                    target=edit.target,
                    status="skipped",
                    reason="would_empty_skill",
                    source_type=edit.source_type,
                )
            )
            continue

        candidate = _join_protected(new_body, protected)
        if len(candidate) > max_skill_chars:
            report.append(
                _EditApplyEntry(
                    op=edit.op,
                    target=edit.target,
                    status="skipped",
                    reason="exceeds_max_chars",
                    source_type=edit.source_type,
                )
            )
            continue

        body = _strip_protected_markers(new_body)
        applied += 1
        report.append(
            _EditApplyEntry(
                op=edit.op,
                target=edit.target,
                status="applied",
                reason="",
                source_type=edit.source_type,
            )
        )

    return _join_protected(body, protected), report, applied


def _summarize_edits(edits: list[_PatchEdit], limit: int = 200) -> str:
    if not edits:
        return ""
    parts: list[str] = []
    for e in edits:
        if e.op == "append":
            parts.append(f"[append] {_compact_preview(e.content, 60)}")
        elif e.op == "insert_after":
            parts.append(f"[insert_after '{_compact_preview(e.target, 30)}'] {_compact_preview(e.content, 50)}")
        elif e.op == "replace":
            parts.append(
                f"[replace '{_compact_preview(e.target, 30)}' -> '{_compact_preview(e.content, 40)}']"
            )
        else:
            parts.append(f"[delete '{_compact_preview(e.target, 50)}']")
    return _compact_preview("; ".join(parts), limit)


def _cosine_edit_budget(iteration: int, max_edits: int, min_edits: int, schedule_rounds: int) -> int:
    """Cosine-annealed atomic edit budget ``L_t`` decaying from ``max_edits`` down to ``min_edits``."""
    if schedule_rounds <= 1 or max_edits <= min_edits:
        return min_edits
    t = min(max(iteration - 1, 0), schedule_rounds - 1) / (schedule_rounds - 1)
    val = round(min_edits + 0.5 * (max_edits - min_edits) * (1.0 + math.cos(math.pi * t)))
    return max(min_edits, min(max_edits, int(val)))


def _edit_budget(
    iteration: int,
    max_edits: int,
    min_edits: int,
    schedule_rounds: int,
    lr_schedule: str = "cosine",
) -> int:
    if lr_schedule == "constant":
        return max_edits
    if lr_schedule == "cosine":
        return _cosine_edit_budget(iteration, max_edits, min_edits, schedule_rounds)
    if schedule_rounds <= 1 or max_edits <= min_edits:
        return min_edits
    t = min(max(iteration - 1, 0), schedule_rounds - 1) / (schedule_rounds - 1)
    val = round(max_edits - (max_edits - min_edits) * t)
    return max(min_edits, min(max_edits, int(val)))


# ------------------------------------------------------------------ disjoint split & objective scoring


@dataclass(frozen=True)
class _ScoredEval:
    round_idx: int
    ev: SkillEvaluation
    mean_acc: float
    norm_cost: float
    norm_reasoning: float | None
    mean_output_tokens: float
    mean_reasoning_tokens: float
    acc_delta: float
    worst_bench_acc_delta: float
    feasible: bool
    feasible_utility: float
    score: float
    bench_acc: dict[str, float]
    bench_cost: dict[str, float]
    bench_reasoning: dict[str, float]


@dataclass(frozen=True)
class _RejectedRecord:
    round_idx: int
    skill_id: str
    parent_id: str
    d_acc: float
    d_norm_cost: float
    d_norm_reasoning: float | None
    reason: str
    summary: str
    pattern_note: str = ""


@dataclass(frozen=True)
class _AcceptedRecord:
    round_idx: int
    skill_id: str
    parent_id: str
    d_acc: float
    d_norm_cost: float
    d_norm_reasoning: float | None
    summary: str


@dataclass
class _GateState:
    anchor: _ScoredEval | None
    incumbent: _ScoredEval | None
    best: _ScoredEval | None
    all_scored: list[_ScoredEval]
    hash_cache: dict[str, _ScoredEval]
    accepted: list[_AcceptedRecord]
    all_rejected: list[_RejectedRecord]
    epoch_rejected: list[_RejectedRecord]
    epoch_snapshots: list[_ScoredEval]
    prev_round_incumbent: _ScoredEval | None
    train_ids: dict[str, set[str]]
    sel_ids: dict[str, set[str]]
    anchor_cost: dict[str, float]
    fallback_cost: dict[str, float]


def _partition_example_ids(
    rounds: list[list[SkillEvaluation]],
    selection_ratio: float,
    split_seed: int,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Deterministically partition each benchmark's ``train`` IDs into disjoint ``D_tr`` and ``D_sel``."""
    ids_by_bench: dict[str, set[str]] = {}
    for rnd in rounds:
        for ev in rnd:
            for b, res in ev.results.items():
                if res.n <= 0:
                    continue
                bucket = ids_by_bench.setdefault(b, set())
                for it in res.items:
                    bucket.add(it.example_id)

    train_ids: dict[str, set[str]] = {}
    sel_ids: dict[str, set[str]] = {}
    for b, all_ids in ids_by_bench.items():
        ordered = sorted(all_ids)
        if selection_ratio <= 0.0 or len(ordered) < 2:
            train_ids[b] = set(ordered)
            sel_ids[b] = set(ordered)
            continue
        rng = random.Random(split_seed ^ _stable_int_hash(b))
        shuffled = list(ordered)
        rng.shuffle(shuffled)
        n_sel = max(1, min(len(shuffled) - 1, int(round(len(shuffled) * selection_ratio))))
        sel_ids[b] = set(shuffled[:n_sel])
        train_ids[b] = set(shuffled[n_sel:])
    return train_ids, sel_ids


def _item_cost(it: ItemResult, cost_metric: str) -> float:
    if cost_metric == "output":
        return float(it.usage.output_tokens)
    if cost_metric == "reasoning":
        return float(it.usage.reasoning_tokens)
    if cost_metric == "total":
        return float(it.usage.total_tokens if it.usage.total_tokens is not None else it.usage.input_tokens + it.usage.output_tokens)
    raise ValueError(f"unknown cost_metric '{cost_metric}'")


def _subset_bench_stats(
    res: EvalResult,
    allowed_ids: set[str] | None,
    cost_metric: str,
) -> tuple[float, float, float, float, float]:
    """Return ``(acc, avg_cost, avg_out, avg_reasoning, trunc_rate)`` on ``allowed_ids``."""
    if not res.items:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if allowed_ids is not None:
        subset = [it for it in res.items if it.example_id in allowed_ids]
        items = subset if subset else res.items
    else:
        items = res.items
    k = len(items)
    acc = sum(1.0 if it.correct else 0.0 for it in items) / k
    cost = sum(_item_cost(it, cost_metric) for it in items) / k
    out = sum(float(it.usage.output_tokens) for it in items) / k
    think = sum(float(it.usage.reasoning_tokens) for it in items) / k
    trunc = sum(1.0 if it.truncated else 0.0 for it in items) / k
    return acc, cost, out, think, trunc


def _ev_raw_means(
    ev: SkillEvaluation,
    allowed_ids: dict[str, set[str]] | None = None,
    cost_metric: str = "output",
) -> tuple[float, float, float, float]:
    active = [(b, r) for b, r in ev.results.items() if r.n > 0]
    if not active:
        return 0.0, 0.0, 0.0, 0.0
    accs, costs, outs, thinks = [], [], [], []
    for b, r in active:
        a, c, o, t, _ = _subset_bench_stats(r, allowed_ids.get(b) if allowed_ids else None, cost_metric)
        accs.append(a)
        costs.append(c)
        outs.append(o)
        thinks.append(t)
    k = len(active)
    return sum(accs) / k, sum(costs) / k, sum(outs) / k, sum(thinks) / k


def _find_anchor(
    rounds: list[list[SkillEvaluation]],
    anchor_id: str,
    sel_ids: dict[str, set[str]] | None = None,
    cost_metric: str = "output",
) -> SkillEvaluation | None:
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
    return min(
        pool,
        key=lambda ev: (
            -_ev_raw_means(ev, sel_ids, cost_metric)[0],
            _ev_raw_means(ev, sel_ids, cost_metric)[1],
            ev.skill_id,
        ),
    )


def _score_eval(
    round_idx: int,
    ev: SkillEvaluation,
    sel_ids: dict[str, set[str]],
    cost_metric: str,
    anchor_mean_acc: float,
    anchor_bench_acc: dict[str, float],
    anchor_cost: dict[str, float],
    anchor_reasoning: dict[str, float],
    fallback_cost: dict[str, float],
    epsilon: float,
    per_benchmark_epsilon: float | None,
    acc_bonus: float,
) -> _ScoredEval:
    active = [(b, r) for b, r in ev.results.items() if r.n > 0]
    if not active:
        return _ScoredEval(
            round_idx=round_idx,
            ev=ev,
            mean_acc=0.0,
            norm_cost=1.0,
            norm_reasoning=None,
            mean_output_tokens=0.0,
            mean_reasoning_tokens=0.0,
            acc_delta=-anchor_mean_acc,
            worst_bench_acc_delta=-anchor_mean_acc,
            feasible=False,
            feasible_utility=-1.0,
            score=-1.0,
            bench_acc={},
            bench_cost={},
            bench_reasoning={},
        )

    bench_acc: dict[str, float] = {}
    bench_cost: dict[str, float] = {}
    bench_reasoning: dict[str, float] = {}
    outs: list[float] = []
    thinks: list[float] = []
    for b, r in active:
        a, c, o, t, _ = _subset_bench_stats(r, sel_ids.get(b), cost_metric)
        bench_acc[b] = float(a)
        bench_cost[b] = float(c)
        bench_reasoning[b] = float(t)
        outs.append(float(o))
        thinks.append(float(t))

    k = len(active)
    mean_acc = sum(bench_acc.values()) / k
    mean_out = sum(outs) / k
    mean_think = sum(thinks) / k

    ratios = [
        bench_cost[b] / max(anchor_cost.get(b, fallback_cost.get(b, 1.0)), 1.0)
        for b, _ in active
    ]
    norm_cost = sum(ratios) / len(ratios)

    think_ratios = [
        bench_reasoning[b] / anchor_reasoning[b]
        for b, _ in active
        if anchor_reasoning.get(b, 0.0) > 0.0
    ]
    norm_reasoning = (sum(think_ratios) / len(think_ratios)) if think_ratios else None

    acc_delta = mean_acc - anchor_mean_acc
    bench_deltas = [bench_acc[b] - anchor_bench_acc[b] for b in bench_acc if b in anchor_bench_acc]
    worst_delta = min(bench_deltas) if bench_deltas else acc_delta

    feasible = acc_delta >= -epsilon - _TOL
    if per_benchmark_epsilon is not None:
        feasible = feasible and (worst_delta >= -per_benchmark_epsilon - _TOL)

    feasible_utility = (1.0 - norm_cost) + acc_bonus * acc_delta
    score = (10.0 + feasible_utility) if feasible else (mean_acc - 0.05 * norm_cost)
    return _ScoredEval(
        round_idx=round_idx,
        ev=ev,
        mean_acc=mean_acc,
        norm_cost=norm_cost,
        norm_reasoning=norm_reasoning,
        mean_output_tokens=mean_out,
        mean_reasoning_tokens=mean_think,
        acc_delta=acc_delta,
        worst_bench_acc_delta=worst_delta,
        feasible=feasible,
        feasible_utility=feasible_utility,
        score=score,
        bench_acc=bench_acc,
        bench_cost=bench_cost,
        bench_reasoning=bench_reasoning,
    )


def _candidate_rank_key(s: _ScoredEval) -> tuple[int, float, float, float]:
    return (
        1 if s.feasible else 0,
        s.feasible_utility if s.feasible else s.mean_acc,
        s.mean_acc,
        -s.norm_cost,
    )


def _check_gate_improvement(
    cand: _ScoredEval,
    inc: _ScoredEval,
    min_improvement: float,
    acc_bonus: float,
    cost_metric: str,
    require_reasoning_reduction: bool,
) -> tuple[bool, str]:
    """Strict lexicographic validation gate for ``min E[T] s.t. A >= A_anchor - epsilon``."""
    if cand.feasible and not inc.feasible:
        return True, "achieved accuracy constraint"
    if not cand.feasible and inc.feasible:
        return False, "violated accuracy constraint"

    if not cand.feasible and not inc.feasible:
        if cand.mean_acc > inc.mean_acc + min_improvement:
            return True, "improved accuracy toward constraint"
        if abs(cand.mean_acc - inc.mean_acc) <= _TOL and cand.norm_cost < inc.norm_cost - min_improvement:
            return True, "reduced cost at equal accuracy"
        return False, "accuracy below constraint without gain"

    # Both feasible: enforce anti-gaming guard before cost utility check
    if (
        require_reasoning_reduction
        and cost_metric == "output"
        and inc.norm_reasoning is not None
        and cand.norm_reasoning is not None
        and cand.norm_cost < inc.norm_cost - _TOL
        and cand.norm_reasoning >= inc.norm_reasoning - _TOL
        and (acc_bonus == 0.0 or cand.mean_acc <= inc.mean_acc + min_improvement)
    ):
        return False, "visible-only shortening (reasoning tokens did not decrease)"

    if cand.feasible_utility > inc.feasible_utility + min_improvement:
        return True, "reduced normalized token cost"

    if abs(cand.norm_cost - inc.norm_cost) <= _TOL and cand.mean_acc > inc.mean_acc + min_improvement:
        return True, "increased accuracy headroom at equal cost"

    if cand.norm_cost > inc.norm_cost + _TOL:
        return False, "increased token cost"
    if cand.mean_acc < inc.mean_acc - _TOL:
        return False, "accuracy regressed without sufficient token savings"
    return False, "no strict improvement (tie rejected)"


def _edit_summary_for(ev: SkillEvaluation) -> str:
    sk = ev.skill
    meta = sk.metadata if isinstance(sk.metadata, dict) else {}
    applied = meta.get("applied_edits")
    if isinstance(applied, str) and applied.strip():
        return _compact_preview(applied, 160)
    if sk.origin is not None and sk.origin.note.strip():
        return _compact_preview(sk.origin.note, 160)
    return _compact_preview(sk.instruction, 160)


def _candidate_pattern_note(ev: SkillEvaluation, train_ids: dict[str, set[str]]) -> str:
    n_items = 0
    n_fail = 0
    n_trunc = 0
    for b, res in ev.results.items():
        if res.n <= 0:
            continue
        allowed = train_ids.get(b)
        subset = [it for it in res.items if allowed is None or it.example_id in allowed] or res.items
        n_items += len(subset)
        n_fail += sum(1 for it in subset if not it.correct)
        n_trunc += sum(1 for it in subset if it.truncated)
    if n_items <= 0:
        return ""
    return f"train_fail={n_fail}/{n_items}, trunc={n_trunc}"


def _is_boundary_for_iter(
    iteration: int,
    epoch_rounds: int,
    num_candidates: int,
    enable_slow_update: bool,
) -> bool:
    dedicated_slow = (num_candidates == 1 and enable_slow_update)
    cycle = (epoch_rounds + 1) if dedicated_slow else epoch_rounds
    return iteration >= 2 and (iteration % cycle == 0)


def _replay_gate(
    rounds: list[list[SkillEvaluation]],
    anchor_id: str,
    epsilon: float,
    min_improvement: float = 1e-6,
    max_rejected_buffer: int = 8,
    *,
    per_benchmark_epsilon: float | None = None,
    cost_metric: str = "output",
    acc_bonus: float = 0.0,
    require_reasoning_reduction: bool = True,
    selection_ratio: float = 0.25,
    split_seed: int = 42,
    epoch_rounds: int = 2,
    num_candidates: int = 2,
    enable_slow_update: bool = True,
    reset_rejected_on_epoch: bool = True,
) -> _GateState:
    """Replay all rounds deterministically on ``D_sel`` to reconstruct state, buffer, and snapshots."""
    train_ids, sel_ids = _partition_example_ids(rounds, selection_ratio, split_seed)
    anchor_ev = _find_anchor(rounds, anchor_id, sel_ids, cost_metric)
    if anchor_ev is None:
        return _GateState(
            anchor=None,
            incumbent=None,
            best=None,
            all_scored=[],
            hash_cache={},
            accepted=[],
            all_rejected=[],
            epoch_rejected=[],
            epoch_snapshots=[],
            prev_round_incumbent=None,
            train_ids=train_ids,
            sel_ids=sel_ids,
            anchor_cost={},
            fallback_cost={},
        )

    anchor_bench_acc: dict[str, float] = {}
    anchor_cost: dict[str, float] = {}
    anchor_reasoning: dict[str, float] = {}
    for b, r in anchor_ev.results.items():
        if r.n <= 0:
            continue
        a, c, _, t, _ = _subset_bench_stats(r, sel_ids.get(b), cost_metric)
        anchor_bench_acc[b] = float(a)
        anchor_cost[b] = max(float(c), 1.0)
        if t > 0.0:
            anchor_reasoning[b] = float(t)
    anchor_mean_acc = (
        sum(anchor_bench_acc.values()) / len(anchor_bench_acc) if anchor_bench_acc else 0.0
    )

    fallback_cost: dict[str, float] = {}
    for rnd in rounds:
        for ev in rnd:
            for b, r in ev.results.items():
                if r.n > 0:
                    _, c, _, _, _ = _subset_bench_stats(r, sel_ids.get(b), cost_metric)
                    fallback_cost[b] = max(fallback_cost.get(b, 1.0), max(float(c), 1.0))

    scored_rounds: list[list[_ScoredEval]] = []
    all_scored: list[_ScoredEval] = []
    hash_cache: dict[str, _ScoredEval] = {}
    anchor_scored: _ScoredEval | None = None
    for r_idx, rnd in enumerate(rounds):
        scored_rnd: list[_ScoredEval] = []
        for ev in rnd:
            sc = _score_eval(
                r_idx,
                ev,
                sel_ids,
                cost_metric,
                anchor_mean_acc,
                anchor_bench_acc,
                anchor_cost,
                anchor_reasoning,
                fallback_cost,
                epsilon,
                per_benchmark_epsilon,
                acc_bonus,
            )
            scored_rnd.append(sc)
            all_scored.append(sc)
            h = _stable_hash(ev.skill.instruction)
            if h not in hash_cache:
                hash_cache[h] = sc
            if ev is anchor_ev:
                anchor_scored = sc
        scored_rounds.append(scored_rnd)

    incumbent: _ScoredEval | None = None
    first_non_empty = next((sr for sr in scored_rounds if sr), [])
    if first_non_empty:
        incumbent = max(first_non_empty, key=_candidate_rank_key)
    best = incumbent
    epoch_snapshots: list[_ScoredEval] = [incumbent] if incumbent is not None else []
    prev_round_incumbent: _ScoredEval | None = incumbent

    accepted: list[_AcceptedRecord] = []
    all_rejected: list[_RejectedRecord] = []
    epoch_rejected: list[_RejectedRecord] = []

    for r_idx in range(1, len(scored_rounds)):
        cands = scored_rounds[r_idx]
        if not cands:
            continue
        if incumbent is None:
            incumbent = max(cands, key=_candidate_rank_key)
            best = incumbent
            epoch_snapshots = [incumbent]
            prev_round_incumbent = incumbent
            continue

        prev_inc = incumbent
        prev_round_incumbent = prev_inc

        best_cand = max(cands, key=_candidate_rank_key)
        accept_flag, accept_reason = _check_gate_improvement(
            best_cand,
            prev_inc,
            min_improvement,
            acc_bonus,
            cost_metric,
            require_reasoning_reduction,
        )
        promoted = best_cand if accept_flag else None
        if promoted is not None:
            incumbent = promoted
            if best is None or _check_gate_improvement(
                promoted, best, min_improvement, acc_bonus, cost_metric, require_reasoning_reduction
            )[0]:
                best = promoted
            d_think = (
                (promoted.norm_reasoning - prev_inc.norm_reasoning)
                if promoted.norm_reasoning is not None and prev_inc.norm_reasoning is not None
                else None
            )
            accepted.append(
                _AcceptedRecord(
                    round_idx=r_idx,
                    skill_id=promoted.ev.skill_id,
                    parent_id=promoted.ev.skill.parent_id or prev_inc.ev.skill_id,
                    d_acc=promoted.mean_acc - prev_inc.mean_acc,
                    d_norm_cost=promoted.norm_cost - prev_inc.norm_cost,
                    d_norm_reasoning=d_think,
                    summary=_edit_summary_for(promoted.ev),
                )
            )

        for cand in cands:
            if cand is promoted:
                continue
            if cand is best_cand:
                rej_reason = accept_reason
            else:
                _, rej_reason = _check_gate_improvement(
                    cand,
                    prev_inc,
                    min_improvement,
                    acc_bonus,
                    cost_metric,
                    require_reasoning_reduction,
                )
            d_think = (
                (cand.norm_reasoning - prev_inc.norm_reasoning)
                if cand.norm_reasoning is not None and prev_inc.norm_reasoning is not None
                else None
            )
            rec = _RejectedRecord(
                round_idx=r_idx,
                skill_id=cand.ev.skill_id,
                parent_id=cand.ev.skill.parent_id or prev_inc.ev.skill_id,
                d_acc=cand.mean_acc - prev_inc.mean_acc,
                d_norm_cost=cand.norm_cost - prev_inc.norm_cost,
                d_norm_reasoning=d_think,
                reason=rej_reason,
                summary=_edit_summary_for(cand.ev),
                pattern_note=_candidate_pattern_note(cand.ev, train_ids),
            )
            all_rejected.append(rec)
            epoch_rejected.append(rec)

        if _is_boundary_for_iter(r_idx, epoch_rounds, num_candidates, enable_slow_update):
            epoch_snapshots.append(incumbent)
            if reset_rejected_on_epoch:
                epoch_rejected = []

    if max_rejected_buffer == 0:
        all_rejected = []
        epoch_rejected = []
    else:
        if len(all_rejected) > max_rejected_buffer:
            all_rejected = all_rejected[-max_rejected_buffer:]
        if len(epoch_rejected) > max_rejected_buffer:
            epoch_rejected = epoch_rejected[-max_rejected_buffer:]

    return _GateState(
        anchor=anchor_scored,
        incumbent=incumbent,
        best=best,
        all_scored=all_scored,
        hash_cache=hash_cache,
        accepted=accepted,
        all_rejected=all_rejected,
        epoch_rejected=epoch_rejected,
        epoch_snapshots=epoch_snapshots,
        prev_round_incumbent=prev_round_incumbent,
        train_ids=train_ids,
        sel_ids=sel_ids,
        anchor_cost=anchor_cost,
        fallback_cost=fallback_cost,
    )


# ------------------------------------------------------------------ rollout batching & minibatches


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


def _sample_rollout_batch(
    incumbent_ev: SkillEvaluation,
    train_ids: dict[str, set[str]],
    rollout_batch_size: int,
    epoch_idx: int,
    step_in_epoch: int,
    split_seed: int,
) -> list[tuple[str, ItemResult]]:
    """Sample a stratified rollout batch strictly from ``D_tr`` (never ``D_sel``)."""
    buckets: list[list[tuple[str, ItemResult]]] = []
    for bench, res in incumbent_ev.results.items():
        if res.n <= 0:
            continue
        allowed = train_ids.get(bench)
        items = [it for it in res.items if allowed is None or it.example_id in allowed]
        if not items:
            continue
        rng = random.Random(split_seed + 1009 * epoch_idx + _stable_int_hash(bench))
        ordered = sorted(items, key=lambda it: it.example_id)
        rng.shuffle(ordered)
        buckets.append([(bench, it) for it in ordered])

    if not buckets:
        return []

    interleaved: list[tuple[str, ItemResult]] = []
    max_len = max(len(b) for b in buckets)
    for idx in range(max_len):
        for bucket in buckets:
            if idx < len(bucket):
                interleaved.append(bucket[idx])

    if rollout_batch_size <= 0 or len(interleaved) <= rollout_batch_size:
        return interleaved

    total = len(interleaved)
    start = (step_in_epoch * rollout_batch_size) % total
    return [interleaved[(start + offset) % total] for offset in range(rollout_batch_size)]


def _build_failure_minibatches(
    batch: list[tuple[str, ItemResult]],
    reflection_batch_size: int,
    max_batches: int,
) -> list[list[tuple[str, ItemResult]]]:
    if reflection_batch_size <= 0 or max_batches <= 0:
        return []
    wrong = [pair for pair in batch if not pair[1].correct]
    if not wrong:
        return []
    wrong.sort(
        key=lambda pair: (
            0 if pair[1].truncated else 1,
            0 if pair[1].prediction is None else 1,
            -pair[1].usage.output_tokens,
            pair[0],
            pair[1].example_id,
        )
    )
    batches: list[list[tuple[str, ItemResult]]] = []
    for start in range(0, len(wrong), reflection_batch_size):
        batches.append(wrong[start : start + reflection_batch_size])
        if len(batches) >= max_batches:
            break
    return batches


def _build_efficiency_minibatches(
    batch: list[tuple[str, ItemResult]],
    anchor_cost: dict[str, float],
    fallback_cost: dict[str, float],
    cost_metric: str,
    reflection_batch_size: int,
    max_batches: int,
) -> list[list[tuple[str, ItemResult, str]]]:
    """Build minibatches contrasting verbose successes against compact successes."""
    if reflection_batch_size <= 0 or max_batches <= 0:
        return []
    correct = [pair for pair in batch if pair[1].correct]
    if not correct:
        return []

    def _ratio(pair: tuple[str, ItemResult]) -> float:
        bench, it = pair
        denom = max(anchor_cost.get(bench, fallback_cost.get(bench, 1.0)), 1.0)
        return _item_cost(it, cost_metric) / denom

    by_compact = sorted(correct, key=lambda p: (_ratio(p), _item_cost(p[1], cost_metric), p[0], p[1].example_id))
    by_verbose = sorted(correct, key=lambda p: (-_ratio(p), -_item_cost(p[1], cost_metric), p[0], p[1].example_id))

    if len(correct) == 1:
        b, it = correct[0]
        return [[(b, it, f"SUCCESS {_ratio((b, it)):.2f}x anchor")]]

    compact_exemplar = by_compact[0]
    verbose_pool = [
        pair for pair in by_verbose if not (pair[0] == compact_exemplar[0] and pair[1] is compact_exemplar[1])
    ]
    if not verbose_pool:
        verbose_pool = by_verbose

    verbose_per_batch = max(1, reflection_batch_size - 1)
    batches: list[list[tuple[str, ItemResult, str]]] = []
    for start in range(0, len(verbose_pool), verbose_per_batch):
        chunk = verbose_pool[start : start + verbose_per_batch]
        mb: list[tuple[str, ItemResult, str]] = [
            (b, it, f"VERBOSE {_ratio((b, it)):.2f}x anchor") for b, it in chunk
        ]
        cb, cit = compact_exemplar
        mb.append((cb, cit, f"COMPACT {_ratio(compact_exemplar):.2f}x anchor"))
        batches.append(mb)
        if len(batches) >= max_batches:
            break
    return batches


def _format_failure_minibatch(
    batch: list[tuple[str, ItemResult]],
    max_trace_chars: int,
) -> str:
    blocks: list[str] = []
    for idx, (bench, it) in enumerate(batch, 1):
        trunc = " [TRUNCATED]" if it.truncated else ""
        blocks.append(
            f"[Failure #{idx} | {bench} | out={it.usage.output_tokens} think={it.usage.reasoning_tokens}]{trunc}\n"
            f"Question: {_compact_preview(it.question, 220)}\n"
            f"Prediction: {it.prediction!r} | Gold: {it.gold!r}\n"
            f"Trace: {_item_trace(it, max_trace_chars)}"
        )
    return "\n\n".join(blocks)


def _format_efficiency_minibatch(
    batch: list[tuple[str, ItemResult, str]],
    max_trace_chars: int,
) -> str:
    blocks: list[str] = []
    for idx, (bench, it, tag) in enumerate(batch, 1):
        char_cap = max(140, max_trace_chars // 2) if tag.startswith("COMPACT") else max_trace_chars
        blocks.append(
            f"[#{idx} {tag} | {bench} | out={it.usage.output_tokens} think={it.usage.reasoning_tokens}]\n"
            f"Question: {_compact_preview(it.question, 200)}\n"
            f"Trace: {_item_trace(it, char_cap)}"
        )
    return "\n\n".join(blocks)


def _format_shared_state(
    incumbent: _ScoredEval,
    anchor: _ScoredEval | None,
    rejected: list[_RejectedRecord],
    meta_skill: str,
    epsilon: float,
    cost_metric: str,
) -> str:
    per_bench = ", ".join(
        f"{b}: acc={incumbent.bench_acc[b]:.3f}, {cost_metric}={incumbent.bench_cost[b]:.1f}"
        for b in incumbent.bench_acc
    )
    anchor_line = (
        f"Anchor '{anchor.ev.skill_id}': sel_acc={anchor.mean_acc:.3f}, "
        f"accuracy floor = {anchor.mean_acc - epsilon:.3f} (epsilon={epsilon:.3f})."
        if anchor is not None
        else f"Anchor unavailable (epsilon={epsilon:.3f})."
    )
    think_str = f"{incumbent.norm_reasoning:.2f}x" if incumbent.norm_reasoning is not None else "n/a"
    editable_body, protected_body = _split_protected(incumbent.ev.skill.instruction)
    protected_note = (
        f"\n{SLOW_UPDATE_START}\n{protected_body}\n{SLOW_UPDATE_END}\n"
        "(READ-ONLY protected slow-update block: NEVER target, replace, or delete content inside these markers.)"
        if protected_body
        else ""
    )

    sections = [
        f"Objective: minimize {cost_metric} token length (including internal <think> tokens) while keeping "
        f"accuracy >= anchor - {epsilon:.3f}. Reducing visible output without reducing thinking tokens does NOT count.",
        anchor_line,
        (
            f"Current incumbent '{incumbent.ev.skill_id}' (sel_acc={incumbent.mean_acc:.3f}, "
            f"delta={incumbent.acc_delta:+.3f}, feasible={incumbent.feasible}, "
            f"norm_cost={incumbent.norm_cost:.2f}x, norm_think={think_str}, "
            f"out={incumbent.mean_output_tokens:.1f}, think={incumbent.mean_reasoning_tokens:.1f}):\n"
            f"<editable_skill>\n{editable_body}\n</editable_skill>{protected_note}"
            + (f"\nPer-benchmark selection stats: {per_bench}" if per_bench else "")
        ),
    ]
    if meta_skill.strip():
        sections.append(f"Optimizer Meta-Skill (optimizer-side guidance only):\n{meta_skill.strip()}")
    if rejected:
        rej_lines = []
        for r in rejected:
            pat = f" | {r.pattern_note}" if r.pattern_note else ""
            rej_lines.append(
                f"- round {r.round_idx} {r.skill_id} (from {r.parent_id}): "
                f"d_acc={r.d_acc:+.3f}, d_cost={r.d_norm_cost:+.2f}x [{r.reason}]{pat} — {r.summary}"
            )
        sections.append("Epoch rejected-edit buffer (avoid repeating failed directions):\n" + "\n".join(rej_lines))
    return "\n\n".join(sections)


# ------------------------------------------------------------------ longitudinal comparison


def _build_longitudinal_comparison(
    prev_ev: SkillEvaluation,
    cur_ev: SkillEvaluation,
    train_ids: dict[str, set[str]],
    anchor_cost: dict[str, float],
    fallback_cost: dict[str, float],
    cost_metric: str,
    max_samples: int,
    max_trace_chars: int,
    split_seed: int,
    epoch_idx: int,
) -> str:
    """Compare ``prev_ev`` and ``cur_ev`` item-by-item on ``D_tr`` without extra target rollouts."""
    paired: list[tuple[str, ItemResult, ItemResult]] = []
    for bench, cur_res in cur_ev.results.items():
        if cur_res.n <= 0:
            continue
        prev_res = prev_ev.results.get(bench)
        if prev_res is None or prev_res.n <= 0:
            continue
        allowed = train_ids.get(bench)
        prev_by_id = {it.example_id: it for it in prev_res.items if allowed is None or it.example_id in allowed}
        cur_by_id = {it.example_id: it for it in cur_res.items if allowed is None or it.example_id in allowed}
        for ex_id in sorted(set(prev_by_id) & set(cur_by_id)):
            paired.append((bench, prev_by_id[ex_id], cur_by_id[ex_id]))

    if not paired:
        return "No overlapping D_tr items available between previous and current epoch skills."

    rng = random.Random(split_seed + 4099 * epoch_idx)
    rng.shuffle(paired)
    sampled = paired[: max(1, max_samples)]

    acc_regressions: list[tuple[str, ItemResult, ItemResult]] = []
    bloat_regressions: list[tuple[str, ItemResult, ItemResult]] = []
    persistent_failures: list[tuple[str, ItemResult, ItemResult]] = []
    persistent_verbose: list[tuple[str, ItemResult, ItemResult]] = []
    improvements: list[tuple[str, ItemResult, ItemResult]] = []
    stable_compact: list[tuple[str, ItemResult, ItemResult]] = []

    same_skill = prev_ev.skill_id == cur_ev.skill_id
    for bench, p_it, c_it in sampled:
        p_cost = _item_cost(p_it, cost_metric)
        c_cost = _item_cost(c_it, cost_metric)
        anch = max(anchor_cost.get(bench, fallback_cost.get(bench, 1.0)), 1.0)
        if p_it.correct and not c_it.correct:
            acc_regressions.append((bench, p_it, c_it))
        elif not p_it.correct and not c_it.correct:
            persistent_failures.append((bench, p_it, c_it))
        elif not p_it.correct and c_it.correct:
            improvements.append((bench, p_it, c_it))
        elif not same_skill and c_cost > max(p_cost * 1.15, p_cost + 5.0):
            bloat_regressions.append((bench, p_it, c_it))
        elif not same_skill and c_cost < min(p_cost * 0.85, p_cost - 5.0):
            improvements.append((bench, p_it, c_it))
        elif c_cost > 0.9 * anch:
            persistent_verbose.append((bench, p_it, c_it))
        else:
            stable_compact.append((bench, p_it, c_it))

    lines = [
        f"Longitudinal comparison across {len(sampled)} D_tr tasks "
        f"('{prev_ev.skill_id}' -> '{cur_ev.skill_id}'"
        + (" [NO step edits passed validation this epoch]" if same_skill else "")
        + "):",
        f"- Accuracy regressions: {len(acc_regressions)} | Bloat regressions: {len(bloat_regressions)} "
        f"| Persistent failures: {len(persistent_failures)} | Persistent verbose successes: {len(persistent_verbose)} "
        f"| Improvements: {len(improvements)} | Stable compact successes: {len(stable_compact)}",
    ]

    def _fmt_bucket(title: str, rows: list[tuple[str, ItemResult, ItemResult]], limit: int) -> None:
        if not rows:
            return
        lines.append(f"\n{title}:")
        for bench, p_it, c_it in rows[:limit]:
            lines.append(
                f"  [{bench}] Q: {_compact_preview(c_it.question, 160)}\n"
                f"    Prev: ok={p_it.correct} out={p_it.usage.output_tokens} think={p_it.usage.reasoning_tokens} | "
                f"Cur: ok={c_it.correct} out={c_it.usage.output_tokens} think={c_it.usage.reasoning_tokens}\n"
                f"    Cur trace: {_item_trace(c_it, max(140, max_trace_chars // 2))}"
            )

    _fmt_bucket("Accuracy regressions (fix immediately)", acc_regressions, 3)
    _fmt_bucket("Bloat regressions (reasoning grew longer)", bloat_regressions, 3)
    _fmt_bucket("Persistent failures", persistent_failures, 3)
    _fmt_bucket("Persistent verbose successes (still above 0.9x anchor)", persistent_verbose, 3)
    _fmt_bucket("Efficiency / accuracy improvements (preserve)", improvements, 2)
    return "\n".join(lines)


# ------------------------------------------------------------------ candidate parsing


def _parse_rewrite_candidate(
    text: str | None,
    truncated: bool,
    protected: str,
    seen_hashes: set[str],
    max_skill_chars: int,
) -> tuple[str, str, str | None]:
    """Parse fallback or rewrite-mode candidate into ``(full_instruction, summary, reject_reason)``."""
    raw = text or ""
    edits_summary = ""
    data = _extract_json_object(raw)
    if isinstance(data, dict) and isinstance(data.get("skill"), str):
        body = _clean_instruction(data["skill"])
        if isinstance(data.get("reasoning"), str):
            edits_summary = _compact_preview(data["reasoning"], 180)
    else:
        edit_blocks = _EDITS_RE.findall(raw)
        edits_summary = _compact_preview(edit_blocks[-1], 180) if edit_blocks else ""
        skill_blocks = _SKILL_RE.findall(raw)
        if skill_blocks:
            body = _clean_instruction(skill_blocks[-1])
        elif truncated:
            return "", edits_summary, "truncated"
        else:
            stripped = _REFLECTION_RE.sub("", _EDITS_RE.sub("", raw))
            body = _clean_instruction(stripped)

    body = _strip_protected_markers(body)
    norm_body = _normalize_text(body)
    if not norm_body or norm_body in _INVALID_PLACEHOLDERS:
        return "", edits_summary, "empty"

    full = _join_protected(body, protected)
    if len(full) > max_skill_chars:
        return "", edits_summary, "too_long"
    if _stable_hash(full) in seen_hashes:
        return "", edits_summary, "duplicate"
    return full, edits_summary, None


# ------------------------------------------------------------------ optimizer


@register_optimizer("skillopt")
class SkillOptOptimizer(Optimizer):
    """SkillOpt constrained reasoning-length optimizer with patch updates, validation gating, and slow/meta epochs."""

    def __init__(
        self,
        llm: BaseLLM,
        initial: list[str] | tuple[str, ...] = ("direct", "cot", "cod"),
        anchor_id: str = "cot",
        epsilon: float = 0.02,
        per_benchmark_epsilon: float | None = None,
        cost_metric: str = "output",
        acc_bonus: float = 0.0,
        require_reasoning_reduction: bool = True,
        min_improvement: float = 1e-6,
        selection_ratio: float = 0.25,
        split_seed: int = 42,
        num_candidates: int = 2,
        rollout_batch_size: int = 40,
        reflection_batch_size: int = 8,
        max_failure_batches: int = 2,
        max_success_batches: int = 2,
        max_edits: int = 4,
        min_edits: int = 2,
        edit_schedule_rounds: int = 4,
        lr_schedule: str = "cosine",
        edit_mode: str = "patch",
        max_trace_chars: int = 500,
        max_skill_chars: int = 2400,
        max_rejected_buffer: int = 8,
        epoch_rounds: int = 2,
        slow_update_samples: int = 20,
        enable_slow_update: bool = True,
        enable_meta_skill: bool = True,
        reset_rejected_on_epoch: bool = True,
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
        per_bench_eps_f = (
            None
            if per_benchmark_epsilon is None
            else _check_non_negative_float(per_benchmark_epsilon, "per_benchmark_epsilon")
        )
        cost_metric_s = _check_choice(cost_metric, _VALID_COST_METRICS, "cost_metric")
        acc_bonus_f = _check_non_negative_float(acc_bonus, "acc_bonus")
        req_think_b = bool(require_reasoning_reduction)
        min_imp_f = _check_non_negative_float(min_improvement, "min_improvement")

        sel_ratio_f = _check_non_negative_float(selection_ratio, "selection_ratio")
        if sel_ratio_f >= 1.0:
            raise ValueError(f"selection_ratio must be in [0.0, 1.0), got {selection_ratio!r}")
        if isinstance(split_seed, bool) or not isinstance(split_seed, int):
            raise ValueError(f"split_seed must be an int, got {split_seed!r}")

        num_cand_i = _check_non_negative_int(num_candidates, "num_candidates")
        rollout_bs_i = _check_positive_int(rollout_batch_size, "rollout_batch_size")
        refl_bs_i = _check_positive_int(reflection_batch_size, "reflection_batch_size")
        max_fail_b_i = _check_non_negative_int(max_failure_batches, "max_failure_batches")
        max_succ_b_i = _check_non_negative_int(max_success_batches, "max_success_batches")
        max_edits_i = _check_positive_int(max_edits, "max_edits")
        min_edits_i = _check_positive_int(min_edits, "min_edits")
        if min_edits_i > max_edits_i:
            raise ValueError(f"min_edits ({min_edits_i}) must be <= max_edits ({max_edits_i})")
        edit_sched_i = _check_positive_int(edit_schedule_rounds, "edit_schedule_rounds")
        lr_sched_s = _check_choice(lr_schedule, _VALID_LR_SCHEDULES, "lr_schedule")
        edit_mode_s = _check_choice(edit_mode, _VALID_EDIT_MODES, "edit_mode")
        max_trace_i = _check_non_negative_int(max_trace_chars, "max_trace_chars")
        max_skill_i = _check_positive_int(max_skill_chars, "max_skill_chars")
        max_rej_i = _check_non_negative_int(max_rejected_buffer, "max_rejected_buffer")
        epoch_rounds_i = _check_positive_int(epoch_rounds, "epoch_rounds")
        slow_samples_i = _check_positive_int(slow_update_samples, "slow_update_samples")
        enable_slow_b = bool(enable_slow_update)
        enable_meta_b = bool(enable_meta_skill)
        reset_rej_b = bool(reset_rejected_on_epoch)
        max_retries_i = _check_non_negative_int(max_retries, "max_retries")

        super().__init__(
            llm,
            initial=init_list,
            anchor_id=anchor_id,
            epsilon=epsilon_f,
            per_benchmark_epsilon=per_bench_eps_f,
            cost_metric=cost_metric_s,
            acc_bonus=acc_bonus_f,
            require_reasoning_reduction=req_think_b,
            min_improvement=min_imp_f,
            selection_ratio=sel_ratio_f,
            split_seed=split_seed,
            num_candidates=num_cand_i,
            rollout_batch_size=rollout_bs_i,
            reflection_batch_size=refl_bs_i,
            max_failure_batches=max_fail_b_i,
            max_success_batches=max_succ_b_i,
            max_edits=max_edits_i,
            min_edits=min_edits_i,
            edit_schedule_rounds=edit_sched_i,
            lr_schedule=lr_sched_s,
            edit_mode=edit_mode_s,
            max_trace_chars=max_trace_i,
            max_skill_chars=max_skill_i,
            max_rejected_buffer=max_rej_i,
            epoch_rounds=epoch_rounds_i,
            slow_update_samples=slow_samples_i,
            enable_slow_update=enable_slow_b,
            enable_meta_skill=enable_meta_b,
            reset_rejected_on_epoch=reset_rej_b,
            max_retries=max_retries_i,
            **gen_kwargs,
        )
        self.initial = init_list
        self.anchor_id = anchor_id
        self.epsilon = epsilon_f
        self.per_benchmark_epsilon = per_bench_eps_f
        self.cost_metric = cost_metric_s
        self.acc_bonus = acc_bonus_f
        self.require_reasoning_reduction = req_think_b
        self.min_improvement = min_imp_f
        self.selection_ratio = sel_ratio_f
        self.split_seed = split_seed
        self.num_candidates = num_cand_i
        self.rollout_batch_size = rollout_bs_i
        self.reflection_batch_size = refl_bs_i
        self.max_failure_batches = max_fail_b_i
        self.max_success_batches = max_succ_b_i
        self.max_edits = max_edits_i
        self.min_edits = min_edits_i
        self.edit_schedule_rounds = edit_sched_i
        self.lr_schedule = lr_sched_s
        self.edit_mode = edit_mode_s
        self.max_trace_chars = max_trace_i
        self.max_skill_chars = max_skill_i
        self.max_rejected_buffer = max_rej_i
        self.epoch_rounds = epoch_rounds_i
        self.slow_update_samples = slow_samples_i
        self.enable_slow_update = enable_slow_b
        self.enable_meta_skill = enable_meta_b
        self.reset_rejected_on_epoch = reset_rej_b
        self.max_retries = max_retries_i
        self.gen_kwargs = dict(gen_kwargs)
        self._meta_skill: str = ""

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

    # ------------------------------------------------------------------ Stage 1: Analyst prompts

    def _build_failure_analyst_prompt(
        self,
        shared_state: str,
        minibatch_text: str,
        batch_size: int,
        budget: int,
    ) -> str:
        return (
            "You are an expert failure-analysis agent for reasoning-skill optimization.\n\n"
            f"{shared_state}\n\n"
            f"Failed trajectories on D_tr (batch_size={batch_size}):\n{minibatch_text}\n\n"
            "Identify COMMON failure patterns (accuracy drops, over-compression information loss, formatting errors, "
            "or max_tokens truncations) and propose generalizable atomic patches to fix them without adding bloat.\n"
            f"Budget: propose AT MOST {budget} edits.\n"
            "Respond ONLY with a valid JSON object (no markdown fences, no extra text):\n"
            "{\n"
            f'  "batch_size": {batch_size},\n'
            '  "failure_summary": [{"failure_type": "<type>", "count": 1, "description": "<one-line>"}],\n'
            '  "patch": {\n'
            '    "reasoning": "<why these edits fix common failures without inflating reasoning tokens>",\n'
            '    "edits": [\n'
            '      {"op": "append|insert_after|replace|delete", "target": "<exact text if needed>", "content": "<text>"}\n'
            "    ]\n"
            "  }\n"
            "}\n"
            f"IMPORTANT: Do NOT target, modify, or delete anything between {SLOW_UPDATE_START} and {SLOW_UPDATE_END}."
        )

    def _build_efficiency_analyst_prompt(
        self,
        shared_state: str,
        minibatch_text: str,
        batch_size: int,
        budget: int,
    ) -> str:
        return (
            "You are an expert reasoning-efficiency analyst for SkillOpt.\n\n"
            f"{shared_state}\n\n"
            f"Successful trajectories on D_tr contrasting VERBOSE vs. COMPACT reasoning (batch_size={batch_size}):\n"
            f"{minibatch_text}\n\n"
            "Diagnose systematic sources of token waste inside <think> reasoning (question restatement, verbose prose "
            "transitions, redundant re-verification loops) and extract compact symbolic/structural behaviors from "
            "COMPACT successes that preserve accuracy.\n"
            f"Budget: propose AT MOST {budget} atomic edits (not already in the skill).\n"
            "Respond ONLY with a valid JSON object (no markdown fences, no extra text):\n"
            "{\n"
            f'  "batch_size": {batch_size},\n'
            '  "efficiency_patterns": ["<waste or compact pattern 1>", "<pattern 2>"],\n'
            '  "patch": {\n'
            '    "reasoning": "<how these edits cut thinking/output tokens while preserving accuracy>",\n'
            '    "edits": [\n'
            '      {"op": "append|insert_after|replace|delete", "target": "<exact text if needed>", "content": "<text>"}\n'
            "    ]\n"
            "  }\n"
            "}\n"
            f"IMPORTANT: Do NOT target, modify, or delete anything between {SLOW_UPDATE_START} and {SLOW_UPDATE_END}."
        )

    def _run_stage1_analysts(
        self,
        shared_state: str,
        failure_batches: list[list[tuple[str, ItemResult]]],
        efficiency_batches: list[list[tuple[str, ItemResult, str]]],
        budget: int,
    ) -> tuple[list[list[_PatchEdit]], list[list[_PatchEdit]]]:
        prompts: list[str] = []
        tags: list[str] = []
        for fb in failure_batches:
            prompts.append(
                self._build_failure_analyst_prompt(
                    shared_state, _format_failure_minibatch(fb, self.max_trace_chars), len(fb), budget
                )
            )
            tags.append("failure")
        for eb in efficiency_batches:
            prompts.append(
                self._build_efficiency_analyst_prompt(
                    shared_state, _format_efficiency_minibatch(eb, self.max_trace_chars), len(eb), budget
                )
            )
            tags.append("efficiency")

        if not prompts:
            return [], []

        responses = self.llm.generate_batch(prompts, **self.gen_kwargs)
        fail_patches: list[list[_PatchEdit]] = []
        eff_patches: list[list[_PatchEdit]] = []
        for tag, resp in zip(tags, responses):
            data = _extract_json_object(resp.text or "")
            edits = _parse_edits_from_json(data, tag)
            if edits:
                if tag == "failure":
                    fail_patches.append(edits)
                else:
                    eff_patches.append(edits)
        return fail_patches, eff_patches

    # ------------------------------------------------------------------ Stage 2: Hierarchical Merge

    @staticmethod
    def _dedup_merge_edits(patch_lists: list[list[_PatchEdit]]) -> list[_PatchEdit]:
        merged: dict[tuple[str, str, str], _PatchEdit] = {}
        order: list[tuple[str, str, str]] = []
        for patches in patch_lists:
            for e in patches:
                key = (e.op, _normalize_text(e.target), _normalize_text(e.content))
                if key not in merged:
                    merged[key] = e
                    order.append(key)
                else:
                    prev = merged[key]
                    merged[key] = _PatchEdit(
                        op=prev.op,
                        target=prev.target,
                        content=prev.content,
                        support_count=prev.support_count + e.support_count,
                        source_type=prev.source_type,
                    )
        return [merged[k] for k in order]

    def _merge_single_stream(
        self,
        shared_state: str,
        patch_lists: list[list[_PatchEdit]],
        source_type: str,
    ) -> list[_PatchEdit]:
        if not patch_lists:
            return []
        if len(patch_lists) == 1:
            return list(patch_lists[0])

        serialized = json.dumps(
            [
                [
                    {"op": e.op, "target": e.target, "content": e.content, "source_type": e.source_type}
                    for e in plist
                ]
                for plist in patch_lists
            ],
            ensure_ascii=False,
        )
        focus = (
            "FAILURE-analysis patches (fix systematic accuracy errors & truncations)"
            if source_type == "failure"
            else "EFFICIENCY-analysis patches (cut <think>/output bloat while preserving accuracy)"
        )
        prompt = (
            f"You are a skill-edit coordinator merging {len(patch_lists)} independent {focus}.\n\n"
            f"{shared_state}\n\n"
            f"Patch lists:\n{serialized}\n\n"
            "Deduplicate similar edits, resolve conflicts, favor prevalent patterns across patches, keep edits "
            f"independent, and NEVER target {SLOW_UPDATE_START}..{SLOW_UPDATE_END}.\n"
            "Respond ONLY with a valid JSON object:\n"
            '{"reasoning": "<summary>", "edits": [{"op": "append|insert_after|replace|delete", '
            f'"target": "...", "content": "...", "support_count": 1, "source_type": "{source_type}"' + "}]}"
        )
        resp = self.llm.generate(prompt, **self.gen_kwargs)
        merged = _parse_edits_from_json(_extract_json_object(resp.text or ""), source_type)
        return merged if merged else self._dedup_merge_edits(patch_lists)

    def _merge_final(
        self,
        shared_state: str,
        failure_edits: list[_PatchEdit],
        efficiency_edits: list[_PatchEdit],
        incumbent: _ScoredEval,
    ) -> list[_PatchEdit]:
        if not failure_edits:
            return list(efficiency_edits)
        if not efficiency_edits:
            return list(failure_edits)

        near_floor = (not incumbent.feasible) or (incumbent.acc_delta < -self.epsilon + 0.01)
        priority_rule = (
            "FAILURE EDITS TAKE PRIORITY: the incumbent is near or below the accuracy floor; preserve failure repairs first."
            if near_floor
            else "EFFICIENCY EDITS TAKE PRIORITY: the accuracy constraint is comfortably satisfied; prioritize "
            "reasoning-token reduction while keeping non-conflicting failure guards."
        )
        payload = json.dumps(
            {
                "failure_edits": [dataclasses.asdict(e) for e in failure_edits],
                "efficiency_edits": [dataclasses.asdict(e) for e in efficiency_edits],
            },
            ensure_ascii=False,
        )
        prompt = (
            "You are performing the FINAL hierarchical patch merge for SkillOpt.\n\n"
            f"{shared_state}\n\n"
            f"Priority policy: {priority_rule}\n\n"
            f"Pre-merged edit groups:\n{payload}\n\n"
            "Merge into one coherent, non-redundant edit list, carrying forward support_count and source_type "
            f"and NEVER targeting {SLOW_UPDATE_START}..{SLOW_UPDATE_END}.\n"
            "Respond ONLY with a valid JSON object:\n"
            '{"reasoning": "<priority decisions>", "edits": [{"op": "append|insert_after|replace|delete", '
            '"target": "...", "content": "...", "support_count": 1, "source_type": "failure|efficiency"}]}'
        )
        default_src = "failure" if near_floor else "efficiency"
        resp = self.llm.generate(prompt, **self.gen_kwargs)
        merged = _parse_edits_from_json(_extract_json_object(resp.text or ""), default_src)
        if merged:
            return merged
        ordered = [failure_edits, efficiency_edits] if near_floor else [efficiency_edits, failure_edits]
        return self._dedup_merge_edits(ordered)

    # ------------------------------------------------------------------ Stage 3: Ranking

    def _rank_edits(
        self,
        shared_state: str,
        edits: list[_PatchEdit],
        budget: int,
        incumbent: _ScoredEval,
    ) -> list[_PatchEdit]:
        if len(edits) <= 1:
            return list(edits)

        near_floor = (not incumbent.feasible) or (incumbent.acc_delta < -self.epsilon + 0.01)
        ranked_input = [
            {
                "index": idx,
                "op": e.op,
                "target": e.target,
                "content": e.content,
                "support_count": e.support_count,
                "source_type": e.source_type,
            }
            for idx, e in enumerate(edits)
        ]
        criteria = (
            "1. Accuracy-floor safety: edits repairing systematic failures or truncations rank highest.\n"
            "2. Token compression: edits trimming <think>/output prose bloat across tasks."
            if near_floor
            else "1. Systematic token compression: edits cutting <think>/output reasoning tokens across tasks.\n"
            "2. Accuracy-constraint safety: avoid edits that risk dropping below the accuracy floor."
        )
        prompt = (
            "You are the edit-ranking optimizer for SkillOpt.\n\n"
            f"{shared_state}\n\n"
            f"Candidate edit pool (0-based indices):\n{json.dumps(ranked_input, ensure_ascii=False)}\n\n"
            "Ranking criteria (in order):\n"
            f"{criteria}\n"
            "3. Complementarity (fill gaps without duplicating existing rules).\n"
            "4. Generality and actionability.\n"
            f"Select AT MOST budget L_t = {budget} top edit indices in priority order (you may list extra fallback indices).\n"
            "Respond ONLY with a valid JSON object:\n"
            '{"reasoning": "<brief justification>", "selected_indices": [0]}'
        )
        resp = self.llm.generate(prompt, **self.gen_kwargs)
        data = _extract_json_object(resp.text or "")
        if isinstance(data, dict) and isinstance(data.get("selected_indices"), list):
            chosen: list[int] = []
            seen_idx: set[int] = set()
            for raw_i in data["selected_indices"]:
                if isinstance(raw_i, int) and not isinstance(raw_i, bool) and 0 <= raw_i < len(edits):
                    if raw_i not in seen_idx:
                        seen_idx.add(raw_i)
                        chosen.append(raw_i)
            if chosen:
                remaining = [i for i in range(len(edits)) if i not in seen_idx]
                return [edits[i] for i in chosen + remaining]

        preferred_src = "failure" if near_floor else "efficiency"
        indexed = list(enumerate(edits))
        indexed.sort(key=lambda pair: (0 if pair[1].source_type == preferred_src else 1, -pair[1].support_count, pair[0]))
        return [e for _, e in indexed]

    @staticmethod
    def _candidate_edit_subsets(ranked: list[_PatchEdit], budget: int, n_slots: int) -> list[list[_PatchEdit]]:
        if not ranked or n_slots <= 0:
            return []

        def _sig(edits: list[_PatchEdit]) -> tuple[tuple[str, str, str], ...]:
            return tuple((e.op, e.target, e.content) for e in edits)

        subsets: list[list[_PatchEdit]] = []
        seen_sigs: set[tuple[tuple[str, str, str], ...]] = set()

        def _add(cand: list[_PatchEdit]) -> None:
            if not cand or len(subsets) >= n_slots:
                return
            sig = _sig(cand)
            if sig not in seen_sigs:
                seen_sigs.add(sig)
                subsets.append(cand)

        top = ranked[:budget]
        _add(top)
        if len(ranked) > budget:
            _add((top[:-1] + [ranked[budget]]) if budget > 1 else [ranked[budget]])
        eff_first = [e for e in ranked if e.source_type == "efficiency"] + [
            e for e in ranked if e.source_type != "efficiency"
        ]
        _add(eff_first[:budget])
        if len(top) > 1:
            _add(top[1:])
            _add([top[0]])
        return subsets

    # ------------------------------------------------------------------ Stage 4: Rewrite / Fallback

    def _generate_rewrite_candidate(
        self,
        incumbent: _ScoredEval,
        shared_state: str,
        suggested_edits: list[_PatchEdit],
        budget: int,
        slot_idx: int,
        seen_hashes: set[str],
        rejected_counts: dict[str, int] | None = None,
    ) -> tuple[str, str, str | None]:
        _, protected = _split_protected(incumbent.ev.skill.instruction)
        sugg_block = (
            "\n\nRanked suggestions to incorporate:\n"
            + json.dumps([dataclasses.asdict(e) for e in suggested_edits[:budget]], ensure_ascii=False)
            if suggested_edits
            else ""
        )
        diversity = (
            f"\n- Candidate diversity (slot {slot_idx + 1}): try a distinct concise phrasing or edit subset."
            if slot_idx > 0
            else ""
        )
        base_prompt = (
            "You are SkillOpt updating the editable reasoning skill under our constrained token-length objective.\n\n"
            f"{shared_state}{sugg_block}\n\n"
            f"- Apply at most L_t = {budget} bounded changes to the editable skill; preserve winning accuracy rules.\n"
            f"- Keep the editable skill concise (total under {self.max_skill_chars} chars) and preserve final answer format.\n"
            f"- NEVER include or overwrite {SLOW_UPDATE_START} / {SLOW_UPDATE_END} markers.{diversity}\n\n"
            "Output ONLY the updated editable skill wrapped in <skill>...</skill> (or JSON {\"skill\": \"...\"})."
        )
        last_reason: str | None = None
        for attempt in range(1 + self.max_retries):
            prompt = base_prompt
            if attempt > 0 and last_reason is not None:
                prompt += (
                    f"\n\nPrevious attempt rejected ({last_reason}). "
                    f"Provide a distinct, non-empty editable skill under {self.max_skill_chars} chars inside <skill>...</skill>."
                )
            resp = self.llm.generate(prompt, **self.gen_kwargs)
            full, summary, reason = _parse_rewrite_candidate(
                resp.text, resp.truncated, protected, seen_hashes, self.max_skill_chars
            )
            if reason is None:
                return full, summary, None
            if rejected_counts is not None and reason in rejected_counts:
                rejected_counts[reason] += 1
            last_reason = reason
            logger.debug(f"rewrite candidate slot {slot_idx} attempt {attempt + 1} rejected ({reason})")
        return "", "", last_reason

    # ------------------------------------------------------------------ Epoch Slow Update & Optimizer Meta Skill

    def _run_slow_update_candidate(
        self,
        epoch_num: int,
        incumbent: _ScoredEval,
        longitudinal_text: str,
        shared_state: str,
        seen_hashes: set[str],
    ) -> tuple[str, str]:
        body, prev_protected = _split_protected(incumbent.ev.skill.instruction)
        prompt = (
            f"You are the Epoch-{epoch_num} strategic slow-update advisor for SkillOpt.\n\n"
            f"{shared_state}\n\n"
            f"{longitudinal_text}\n\n"
            f"Previous protected slow-update guidance:\n{prev_protected or '(none yet)'}\n\n"
            "Write updated strategic guidance that OVERWRITES only the protected slow-update block. "
            "Prioritize: (1) preventing accuracy regressions below the anchor floor, "
            "(2) fixing persistent failures without prose bloat, (3) reinforcing compact reasoning habits. "
            "Do NOT duplicate the main editable skill.\n"
            "Respond ONLY with a valid JSON object:\n"
            '{"reasoning": "<reflection on longitudinal changes>", "slow_update_content": "<compact actionable rules>"}'
        )
        resp = self.llm.generate(prompt, **self.gen_kwargs)
        raw = resp.text or ""
        data = _extract_json_object(raw)
        slow_content = ""
        reasoning_note = ""
        if isinstance(data, dict) and isinstance(data.get("slow_update_content"), str):
            slow_content = _strip_protected_markers(data["slow_update_content"])
            reasoning_note = _compact_preview(str(data.get("reasoning", "")), 140)
        else:
            m_slow = _SLOW_RE.findall(raw)
            if m_slow:
                slow_content = _strip_protected_markers(m_slow[-1])

        if not slow_content:
            return "", ""

        candidate = _join_protected(body, slow_content)
        if len(candidate) > self.max_skill_chars:
            headroom = max(40, self.max_skill_chars - len(_join_protected(body, "x")) + 1)
            slow_content = _compact_preview(slow_content, headroom)
            candidate = _join_protected(body, slow_content)
        if len(candidate) > self.max_skill_chars or _stable_hash(candidate) in seen_hashes:
            return "", ""
        return candidate, reasoning_note or _compact_preview(slow_content, 140)

    def _run_meta_skill_update(
        self,
        epoch_num: int,
        incumbent: _ScoredEval,
        longitudinal_text: str,
        accepted: list[_AcceptedRecord],
        rejected: list[_RejectedRecord],
    ) -> str:
        acc_lines = (
            [
                f"- round {a.round_idx} {a.skill_id} (acc {a.d_acc:+.3f}, cost {a.d_norm_cost:+.2f}x): {a.summary}"
                for a in accepted[-5:]
            ]
            if accepted
            else ["- none"]
        )
        rej_lines = (
            [
                f"- round {r.round_idx} {r.skill_id} [{r.reason}] (acc {r.d_acc:+.3f}, "
                f"cost {r.d_norm_cost:+.2f}x): {r.summary}"
                for r in rejected
            ]
            if rejected
            else ["- none"]
        )
        prompt = (
            f"You are an optimizer coach updating the Epoch-{epoch_num} OPTIMIZER-SIDE meta skill.\n"
            "Do NOT write task instructions for the target LLM; write concise coaching for future optimizer calls.\n\n"
            f"Current incumbent '{incumbent.ev.skill_id}' (sel_acc={incumbent.mean_acc:.3f}, "
            f"feasible={incumbent.feasible}, norm_cost={incumbent.norm_cost:.2f}x)\n\n"
            f"{longitudinal_text}\n\n"
            "Accepted improvements:\n" + "\n".join(acc_lines) + "\n\n"
            "Rejected directions:\n" + "\n".join(rej_lines) + "\n\n"
            f"Previous optimizer meta skill:\n{self._meta_skill or '(none)'}\n\n"
            "Synthesize compact coaching (<= 5 bullets) on which edit patterns cut reasoning tokens safely, "
            "what causes accuracy-constraint violations or visible-only gaming, and how to rank edits.\n"
            "Respond ONLY with a valid JSON object:\n"
            '{"reasoning": "<brief reflection>", "meta_skill_content": "<compact optimizer coaching>"}'
        )
        resp = self.llm.generate(prompt, **self.gen_kwargs)
        raw = resp.text or ""
        data = _extract_json_object(raw)
        if isinstance(data, dict) and isinstance(data.get("meta_skill_content"), str):
            return _compact_preview(data["meta_skill_content"], 700)
        blocks = _META_RE.findall(raw)
        if blocks:
            return _compact_preview(blocks[-1], 700)
        return self._meta_skill

    # ------------------------------------------------------------------ update loop

    def _make_skill(
        self,
        sid: str,
        instruction: str,
        incumbent: _ScoredEval,
        iteration: int,
        budget: int,
        operation: str,
        note: str,
        applied_edits_summary: str,
        apply_report: list[_EditApplyEntry],
        gate_rejected_count: int,
        epoch_num: int,
        slow_update: bool = False,
    ) -> Skill:
        few_shot = [
            FewShotExample(question=fs.question, answer=fs.answer)
            for fs in incumbent.ev.skill.few_shot
        ]
        return Skill(
            id=sid,
            instruction=instruction,
            few_shot=few_shot,
            parent_id=incumbent.ev.skill_id,
            metadata={
                "optimizer": "skillopt",
                "iteration": iteration,
                "edit_budget": budget,
                "lr_schedule": self.lr_schedule,
                "edit_mode": self.edit_mode,
                "cost_metric": self.cost_metric,
                "applied_edits": applied_edits_summary,
                "edit_apply_report": [entry.to_dict() for entry in apply_report],
                "incumbent_id": incumbent.ev.skill_id,
                "incumbent_feasible": incumbent.feasible,
                "incumbent_norm_cost": round(incumbent.norm_cost, 6),
                "incumbent_score": round(incumbent.score, 6),
                "gate_rejected_count": gate_rejected_count,
                "meta_epoch": epoch_num,
                "slow_update": slow_update,
                "optimizer_model": self.llm.model,
            },
            origin=SkillOrigin(parents=[incumbent.ev.skill_id], operation=operation, note=note),
        )

    @staticmethod
    def _next_skill_id(iteration: int, slot_idx: int, used_ids: set[str]) -> str:
        base_id = f"skillopt-i{iteration}-c{slot_idx}"
        sid, suffix = base_id, 1
        while sid in used_ids:
            suffix += 1
            sid = f"{base_id}-{suffix}"
        used_ids.add(sid)
        return sid

    def update(self, results: list[SkillEvaluation], history: list[list[SkillEvaluation]]) -> list[Skill]:
        rounds = list(history) + [list(results)]
        iteration = len(history) + 1

        if self.num_candidates == 0:
            return []

        gate = _replay_gate(
            rounds,
            self.anchor_id,
            self.epsilon,
            self.min_improvement,
            self.max_rejected_buffer,
            per_benchmark_epsilon=self.per_benchmark_epsilon,
            cost_metric=self.cost_metric,
            acc_bonus=self.acc_bonus,
            require_reasoning_reduction=self.require_reasoning_reduction,
            selection_ratio=self.selection_ratio,
            split_seed=self.split_seed,
            epoch_rounds=self.epoch_rounds,
            num_candidates=self.num_candidates,
            enable_slow_update=self.enable_slow_update,
            reset_rejected_on_epoch=self.reset_rejected_on_epoch,
        )
        incumbent = gate.incumbent
        if incumbent is None:
            return []

        dedicated_slow_round = (self.num_candidates == 1 and self.enable_slow_update)
        cycle = (self.epoch_rounds + 1) if dedicated_slow_round else self.epoch_rounds
        epoch_num = ((iteration - 1) // cycle) + 1 if cycle > 1 else max(1, iteration - 1)
        step_in_epoch = (iteration - 1) % cycle
        is_epoch_boundary = _is_boundary_for_iter(
            iteration, self.epoch_rounds, self.num_candidates, self.enable_slow_update
        )

        # Epoch-boundary longitudinal comparison & optimizer-side meta skill update
        longitudinal_text = ""
        if is_epoch_boundary:
            if cycle == 1:
                prev_epoch_eval = gate.prev_round_incumbent or incumbent
            else:
                prev_epoch_eval = gate.epoch_snapshots[-1] if gate.epoch_snapshots else incumbent
            longitudinal_text = _build_longitudinal_comparison(
                prev_epoch_eval.ev,
                incumbent.ev,
                gate.train_ids,
                gate.anchor_cost,
                gate.fallback_cost,
                self.cost_metric,
                self.slow_update_samples,
                self.max_trace_chars,
                self.split_seed,
                epoch_num,
            )
            if self.enable_meta_skill:
                self._meta_skill = self._run_meta_skill_update(
                    epoch_num,
                    incumbent,
                    longitudinal_text,
                    gate.accepted,
                    gate.epoch_rejected,
                )

        budget = _edit_budget(
            iteration, self.max_edits, self.min_edits, self.edit_schedule_rounds, self.lr_schedule
        )
        shared_state = _format_shared_state(
            incumbent,
            gate.anchor,
            gate.epoch_rejected,
            self._meta_skill,
            self.epsilon,
            self.cost_metric,
        )
        seen_hashes: set[str] = set(gate.hash_cache.keys())
        used_ids: set[str] = {sc.ev.skill_id for sc in gate.all_scored}

        proposals: list[Skill] = []
        rejected_counts = {"empty": 0, "duplicate": 0, "truncated": 0, "too_long": 0}
        slow_slot_reserved = is_epoch_boundary and self.enable_slow_update
        fast_slots = self.num_candidates - 1 if slow_slot_reserved else self.num_candidates

        # Fast step reflection -> hierarchical merge -> ranking -> patch/rewrite
        ranked_edits: list[_PatchEdit] = []
        if fast_slots > 0 or not slow_slot_reserved:
            rollout_batch = _sample_rollout_batch(
                incumbent.ev,
                gate.train_ids,
                self.rollout_batch_size,
                epoch_num,
                step_in_epoch,
                self.split_seed,
            )
            failure_mbs = _build_failure_minibatches(
                rollout_batch, self.reflection_batch_size, self.max_failure_batches
            )
            efficiency_mbs = _build_efficiency_minibatches(
                rollout_batch,
                gate.anchor_cost,
                gate.fallback_cost,
                self.cost_metric,
                self.reflection_batch_size,
                self.max_success_batches,
            )
            fail_patches, eff_patches = self._run_stage1_analysts(
                shared_state, failure_mbs, efficiency_mbs, budget
            )
            merged_fail = self._merge_single_stream(shared_state, fail_patches, "failure")
            merged_eff = self._merge_single_stream(shared_state, eff_patches, "efficiency")
            merged_all = self._merge_final(shared_state, merged_fail, merged_eff, incumbent)
            ranked_edits = self._rank_edits(shared_state, merged_all, budget, incumbent)

        if self.edit_mode == "patch" and ranked_edits and fast_slots > 0:
            for subset in self._candidate_edit_subsets(ranked_edits, budget, fast_slots):
                patched, report, n_applied = _apply_patches(
                    incumbent.ev.skill.instruction, subset, self.max_skill_chars
                )
                if n_applied <= 0 or _stable_hash(patched) in seen_hashes:
                    continue
                seen_hashes.add(_stable_hash(patched))
                slot_idx = len(proposals)
                sid = self._next_skill_id(iteration, slot_idx, used_ids)
                applied_edits = [e for e, rep in zip(subset, report) if rep.status == "applied"]
                summary = _summarize_edits(applied_edits)
                note = (
                    f"L_t={budget} patch ({n_applied} edits) on {incumbent.ev.skill_id} "
                    f"(acc={incumbent.mean_acc:.3f}, cost={incumbent.norm_cost:.2f}x): {summary}"
                )
                proposals.append(
                    self._make_skill(
                        sid=sid,
                        instruction=patched,
                        incumbent=incumbent,
                        iteration=iteration,
                        budget=budget,
                        operation="skillopt_edit",
                        note=note,
                        applied_edits_summary=summary,
                        apply_report=report,
                        gate_rejected_count=len(gate.epoch_rejected),
                        epoch_num=epoch_num,
                    )
                )

        # Bounded rewrite mode or fallback for any remaining fast slots
        while len(proposals) < fast_slots:
            slot_idx = len(proposals)
            cand_text, rew_summary, _ = self._generate_rewrite_candidate(
                incumbent, shared_state, ranked_edits, budget, slot_idx, seen_hashes, rejected_counts
            )
            if not cand_text:
                break
            seen_hashes.add(_stable_hash(cand_text))
            sid = self._next_skill_id(iteration, slot_idx, used_ids)
            summary = rew_summary or _summarize_edits(ranked_edits[:budget])
            note = (
                f"L_t={budget} rewrite on {incumbent.ev.skill_id} "
                f"(acc={incumbent.mean_acc:.3f}, cost={incumbent.norm_cost:.2f}x)"
                + (f": {summary}" if summary else "")
            )
            proposals.append(
                self._make_skill(
                    sid=sid,
                    instruction=cand_text,
                    incumbent=incumbent,
                    iteration=iteration,
                    budget=budget,
                    operation="skillopt_edit",
                    note=note,
                    applied_edits_summary=summary,
                    apply_report=[],
                    gate_rejected_count=len(gate.epoch_rejected),
                    epoch_num=epoch_num,
                )
            )

        # Protected epoch slow-update slot (goes through D_sel validation next round)
        if slow_slot_reserved:
            slow_text, slow_summary = self._run_slow_update_candidate(
                epoch_num, incumbent, longitudinal_text, shared_state, seen_hashes
            )
            if slow_text:
                seen_hashes.add(_stable_hash(slow_text))
                slot_idx = len(proposals)
                sid = self._next_skill_id(iteration, slot_idx, used_ids)
                proposals.append(
                    self._make_skill(
                        sid=sid,
                        instruction=slow_text,
                        incumbent=incumbent,
                        iteration=iteration,
                        budget=budget,
                        operation="skillopt_slow",
                        note=f"epoch-{epoch_num} slow update on {incumbent.ev.skill_id}: {slow_summary}",
                        applied_edits_summary=f"[slow] {slow_summary}",
                        apply_report=[],
                        gate_rejected_count=len(gate.epoch_rejected),
                        epoch_num=epoch_num,
                        slow_update=True,
                    )
                )
            elif len(proposals) < self.num_candidates:
                # Fallback to a bounded fast edit if slow update produced nothing valid
                cand_text, rew_summary, _ = self._generate_rewrite_candidate(
                    incumbent, shared_state, ranked_edits, budget, len(proposals), seen_hashes, rejected_counts
                )
                if cand_text:
                    seen_hashes.add(_stable_hash(cand_text))
                    slot_idx = len(proposals)
                    sid = self._next_skill_id(iteration, slot_idx, used_ids)
                    proposals.append(
                        self._make_skill(
                            sid=sid,
                            instruction=cand_text,
                            incumbent=incumbent,
                            iteration=iteration,
                            budget=budget,
                            operation="skillopt_edit",
                            note=f"L_t={budget} fallback edit on {incumbent.ev.skill_id}",
                            applied_edits_summary=rew_summary,
                            apply_report=[],
                            gate_rejected_count=len(gate.epoch_rejected),
                            epoch_num=epoch_num,
                        )
                    )

        if len(proposals) < self.num_candidates:
            rej_detail = (
                f" (rejected: empty={rejected_counts['empty']}, duplicate={rejected_counts['duplicate']}, "
                f"truncated={rejected_counts['truncated']}, too_long={rejected_counts['too_long']})"
                if any(rejected_counts.values())
                else ""
            )
            msg = (
                f"update round {iteration}: produced {len(proposals)}/{self.num_candidates} "
                f"distinct valid skill candidates{rej_detail}"
            )
            if not proposals:
                msg += "; returning [] stops the loop"
            logger.warning(msg)

        return proposals
