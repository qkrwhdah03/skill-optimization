"""GEPA-style Pareto Reflective Prompt Optimizer (``gepa``).

Adapts *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*
(Agrawal et al., ICLR 2026 Oral) together with discrete evolutionary crossover
(*EvoPrompt*, Guo et al., ICLR 2024) for our single-LLM, single-skill objective:

    min_s E[T(M, x, s)]   s.t.   A(M, s) >= A_anchor - epsilon

Core mechanisms:
1. Per-benchmark + global Pareto frontier filtering: retains non-dominated skills
   across global ``(mean_acc, -norm_cost)`` and every benchmark's ``(acc, -cost)``.
2. Dominance-frequency parent sampling across feasible frontier specialists.
3. Single-parent reflective mutation (``gepa_mutate``) paired with two-parent
   semantic crossover (``gepa_merge``), fusing accuracy-strong and token-efficient parents.
"""
from __future__ import annotations

import random

from skillopt.evaluator import ItemResult, SkillEvaluation
from skillopt.llm import BaseLLM
from skillopt.optimizer.base import Optimizer, register_optimizer
from skillopt.skills import Skill, SkillOrigin

from ._common import (
    TOL,
    VALID_COST_METRICS,
    ScoredSkill,
    check_choice,
    check_non_negative_float,
    check_non_negative_int,
    check_positive_int,
    compact_preview,
    extract_skill_candidate,
    format_benchmark_table,
    format_trace_item,
    initial_baseline_skills,
    item_cost,
    next_skill_id,
    score_evaluations,
    stable_hash,
    validate_initial_baselines,
)


def _is_dominated(a_cand: float, c_cand: float, a_other: float, c_other: float) -> bool:
    """Return True if ``other`` weakly beats ``cand`` on both accuracy and cost, and strictly on >=1."""
    if a_other + TOL < a_cand or c_other - TOL > c_cand:
        return False
    return (a_other > a_cand + TOL) or (c_other < c_cand - TOL)


def _pareto_front(items: list[ScoredSkill], acc_fn, cost_fn) -> list[ScoredSkill]:
    front: list[ScoredSkill] = []
    for cand in items:
        ac, cc = acc_fn(cand), cost_fn(cand)
        if not any(_is_dominated(ac, cc, acc_fn(other), cost_fn(other)) for other in items if other is not cand):
            front.append(cand)
    return front


@register_optimizer("gepa")
class GEPAOptimizer(Optimizer):
    """Pareto-frontier reflective evolution + crossover optimizer (``gepa``)."""

    def __init__(
        self,
        llm: BaseLLM,
        initial: list[str] | tuple[str, ...] = ("direct", "cot", "cod"),
        anchor_id: str = "cot",
        epsilon: float = 0.02,
        cost_metric: str = "output",
        num_candidates: int = 2,
        merge_prob: float = 0.5,
        max_feedback_items: int = 4,
        max_trace_chars: int = 450,
        max_skill_chars: int = 2400,
        seed: int = 0,
        max_retries: int = 2,
        **gen_kwargs,
    ) -> None:
        init_list, anchor_s = validate_initial_baselines(initial, anchor_id)
        eps_f = check_non_negative_float(epsilon, "epsilon")
        metric_s = check_choice(cost_metric, VALID_COST_METRICS, "cost_metric")
        num_cand_i = check_non_negative_int(num_candidates, "num_candidates")
        merge_p = check_non_negative_float(merge_prob, "merge_prob")
        if merge_p > 1.0:
            raise ValueError(f"merge_prob must be in [0.0, 1.0], got {merge_prob!r}")
        max_fb_i = check_positive_int(max_feedback_items, "max_feedback_items")
        max_trace_i = check_non_negative_int(max_trace_chars, "max_trace_chars")
        max_skill_i = check_positive_int(max_skill_chars, "max_skill_chars")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"seed must be an int, got {seed!r}")
        max_retries_i = check_non_negative_int(max_retries, "max_retries")

        super().__init__(
            llm,
            initial=init_list,
            anchor_id=anchor_s,
            epsilon=eps_f,
            cost_metric=metric_s,
            num_candidates=num_cand_i,
            merge_prob=merge_p,
            max_feedback_items=max_fb_i,
            max_trace_chars=max_trace_i,
            max_skill_chars=max_skill_i,
            seed=seed,
            max_retries=max_retries_i,
            **gen_kwargs,
        )
        self.initial = init_list
        self.anchor_id = anchor_s
        self.epsilon = eps_f
        self.cost_metric = metric_s
        self.num_candidates = num_cand_i
        self.merge_prob = merge_p
        self.max_feedback_items = max_fb_i
        self.max_trace_chars = max_trace_i
        self.max_skill_chars = max_skill_i
        self.seed = seed
        self.max_retries = max_retries_i
        self.gen_kwargs = dict(gen_kwargs)

    def initial_prompts(self) -> list[Skill]:
        return initial_baseline_skills(self.initial, self.anchor_id)

    def _pareto_pool(
        self, scored: list[ScoredSkill], benchmarks: list[str]
    ) -> tuple[list[ScoredSkill], dict[str, int]]:
        latest: dict[str, ScoredSkill] = {}
        for sc in scored:
            if sc.bench_acc:
                latest[sc.ev.skill_id] = sc
        active = list(latest.values())
        if not active:
            return [], {}

        feasible = [sc for sc in active if sc.feasible]
        pool_source = feasible if feasible else active

        freq: dict[str, int] = {}
        members: dict[str, ScoredSkill] = {}

        for sc in _pareto_front(pool_source, lambda s: s.mean_acc, lambda s: s.norm_cost):
            sid = sc.ev.skill_id
            freq[sid] = freq.get(sid, 0) + 1
            members[sid] = sc

        for b in benchmarks:
            b_items = [sc for sc in pool_source if b in sc.bench_acc]
            if not b_items:
                continue
            for sc in _pareto_front(b_items, lambda s, bk=b: s.bench_acc[bk], lambda s, bk=b: s.bench_cost[bk]):
                sid = sc.ev.skill_id
                freq[sid] = freq.get(sid, 0) + 1
                members[sid] = sc

        if not members:
            best = max(pool_source, key=lambda s: s.rank_key)
            return [best], {best.ev.skill_id: 1}

        pool = sorted(members.values(), key=lambda s: (-freq[s.ev.skill_id], s.rank_key), reverse=False)
        return pool, freq

    def _collect_feedback(self, parent: ScoredSkill, anchor: ScoredSkill | None) -> list[str]:
        """Gather reflective failure, truncation, and verbose-trace feedback across worst benchmarks."""
        ranked_benches = sorted(
            parent.bench_acc.keys(),
            key=lambda b: (
                parent.bench_acc[b] - (anchor.bench_acc.get(b, parent.bench_acc[b]) if anchor else 0.0),
                -(parent.bench_cost[b] / max(anchor.bench_cost.get(b, parent.bench_cost[b]), 1.0) if anchor else 1.0),
            ),
        )
        traces: list[tuple[str, ItemResult, int]] = []
        for b in ranked_benches:
            res = parent.ev.results.get(b)
            if not res or not res.items:
                continue
            bad = [it for it in res.items if not it.correct or it.truncated]
            if bad:
                bad.sort(key=lambda it: (0 if it.truncated else 1, -item_cost(it, self.cost_metric)))
                traces.append((b, bad[0], 0))
            good = [it for it in res.items if it.correct and not it.truncated]
            if good:
                good.sort(key=lambda it: -item_cost(it, self.cost_metric))
                traces.append((b, good[0], 1))

        traces.sort(key=lambda t: (t[2], -item_cost(t[1], self.cost_metric)))
        selected = traces[: self.max_feedback_items]
        return [format_trace_item(b, it, self.max_trace_chars) for b, it, _ in selected]

    def _generate_valid(self, prompt: str, seen_hashes: set[str]) -> str | None:
        for _ in range(1 + self.max_retries):
            resp = self.llm.generate(prompt, **self.gen_kwargs)
            cand, reason = extract_skill_candidate(
                resp.text, resp.truncated, seen_hashes, self.max_skill_chars
            )
            if reason is None:
                seen_hashes.add(stable_hash(cand))
                return cand
        return None

    def _mutate(
        self,
        parent: ScoredSkill,
        anchor: ScoredSkill | None,
        iteration: int,
        slot: int,
        used_ids: set[str],
        seen_hashes: set[str],
    ) -> Skill | None:
        feedback = self._collect_feedback(parent, anchor)
        fb_block = "\n\n".join(feedback) if feedback else "(no item traces available)"
        prompt = (
            "You are optimizing a discrete reasoning skill (system prompt) for a frozen reasoning LLM.\n"
            f"Objective: MINIMIZE generated tokens ({self.cost_metric}) while keeping macro accuracy >= "
            f"anchor - {self.epsilon:.3f}"
            + (f" (anchor={anchor.mean_acc:.3f}).\n\n" if anchor else ".\n\n")
            + f"Parent skill '{parent.ev.skill_id}' (acc={parent.mean_acc:.3f}, cost={parent.norm_cost:.2f}x anchor):\n"
            f"```\n{parent.ev.skill.instruction}\n```\n\n"
            f"Per-benchmark scores:\n{format_benchmark_table(parent, anchor)}\n\n"
            f"Diagnostic rollout traces (errors / truncations / verbose reasoning):\n{fb_block}\n\n"
            "Reflect on why accuracy drops or <think> tokens bloat, then write ONE improved, concise instruction "
            f"(<= {self.max_skill_chars} chars) that fixes the errors while eliminating redundant thinking.\n"
            "Return ONLY the new instruction inside <skill>...</skill> tags."
        )
        cand = self._generate_valid(prompt, seen_hashes)
        if cand is None:
            return None
        pid = parent.ev.skill.id
        return Skill(
            id=next_skill_id("gepa", iteration, slot, used_ids),
            instruction=cand,
            few_shot=list(parent.ev.skill.few_shot),
            parent_id=pid,
            metadata={"operator": "mutate", "parent_acc": round(parent.mean_acc, 4), "parent_cost": round(parent.norm_cost, 4)},
            origin=SkillOrigin(
                parents=[pid],
                operation="gepa_mutate",
                note=f"Pareto reflective mutation on {pid} (acc={parent.mean_acc:.3f}, cost={parent.norm_cost:.2f}x)",
            ),
        )

    def _merge(
        self,
        p1: ScoredSkill,
        p2: ScoredSkill,
        anchor: ScoredSkill | None,
        iteration: int,
        slot: int,
        used_ids: set[str],
        seen_hashes: set[str],
    ) -> Skill | None:
        prompt = (
            "You are performing evolutionary crossover (System-Aware Merge) between two Pareto-optimal skills.\n"
            f"Objective: MINIMIZE generated tokens ({self.cost_metric}) while keeping accuracy >= anchor - {self.epsilon:.3f}.\n\n"
            f"Parent A (accuracy specialist: '{p1.ev.skill_id}', acc={p1.mean_acc:.3f}, cost={p1.norm_cost:.2f}x):\n"
            f"```\n{p1.ev.skill.instruction}\n```\n"
            f"{format_benchmark_table(p1, anchor)}\n\n"
            f"Parent B (efficiency specialist: '{p2.ev.skill_id}', acc={p2.mean_acc:.3f}, cost={p2.norm_cost:.2f}x):\n"
            f"```\n{p2.ev.skill.instruction}\n```\n"
            f"{format_benchmark_table(p2, anchor)}\n\n"
            "Combine Parent A's accuracy safeguards with Parent B's token-compression constraints into ONE compact, "
            f"non-redundant skill (<= {self.max_skill_chars} chars). Do not concatenate blindly.\n"
            "Return ONLY the merged instruction inside <skill>...</skill> tags."
        )
        cand = self._generate_valid(prompt, seen_hashes)
        if cand is None:
            return None
        id1, id2 = p1.ev.skill.id, p2.ev.skill.id
        return Skill(
            id=next_skill_id("gepa", iteration, slot, used_ids),
            instruction=cand,
            few_shot=[],
            parent_id=id1,
            metadata={"operator": "merge", "parents": [id1, id2]},
            origin=SkillOrigin(
                parents=[id1, id2],
                operation="gepa_merge",
                note=f"Pareto crossover of {id1} (acc={p1.mean_acc:.3f}) + {id2} (cost={p2.norm_cost:.2f}x)",
            ),
        )

    def update(self, results: list[SkillEvaluation], history: list[list[SkillEvaluation]]) -> list[Skill]:
        if self.num_candidates == 0:
            return []
        rounds = list(history) + [list(results)]
        iteration = len(history) + 1
        anchor, scored, benchmarks = score_evaluations(rounds, self.anchor_id, self.epsilon, self.cost_metric)
        if not scored:
            return []

        pool, freq = self._pareto_pool(scored, benchmarks)
        if not pool:
            return []

        seen_hashes: set[str] = {stable_hash(sc.ev.skill.instruction) for sc in scored}
        used_ids: set[str] = {sc.ev.skill_id for sc in scored}
        rng = random.Random(self.seed + iteration * 97)
        weights = [max(1, freq.get(sc.ev.skill_id, 1)) for sc in pool]

        proposals: list[Skill] = []
        for slot in range(self.num_candidates):
            do_merge = len(pool) >= 2 and rng.random() < self.merge_prob
            skill: Skill | None = None
            if do_merge:
                by_acc = sorted(pool, key=lambda s: (-s.mean_acc, s.norm_cost))
                by_cost = sorted(pool, key=lambda s: (s.norm_cost, -s.mean_acc))
                p1 = by_acc[0]
                p2 = next((s for s in by_cost if s.ev.skill_id != p1.ev.skill_id), pool[1])
                skill = self._merge(p1, p2, anchor, iteration, slot, used_ids, seen_hashes)
            if skill is None:
                parent = rng.choices(pool, weights=weights, k=1)[0]
                skill = self._mutate(parent, anchor, iteration, slot, used_ids, seen_hashes)
            if skill is not None:
                proposals.append(skill)

        return proposals
