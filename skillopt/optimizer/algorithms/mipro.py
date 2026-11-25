"""MIPROv2-style Joint Instruction + Few-Shot Demo Optimizer (``mipro``).

Adapts MIPROv2 (*Opsahl-Ong et al., 2024*) to our constrained reasoning-length
optimization objective:

    min_s E[T(M, x, s)]   s.t.   A(M, s) >= A_anchor - epsilon

Unlike instruction-only optimizers (``opro``, ``skillopt``, ``gepa``, ``textgrad``),
``mipro`` jointly searches over **both**:
1. Concise **few-shot demonstrations** bootstrapped from shortest correct traces
   (stripping ``<think>`` bloat to exemplify compact reasoning), and
2. Grounded **instruction candidates**,

selecting unobserved ``(instruction_idx, demo_set_idx)`` pairs via an additive
empirical-Bayes UCB surrogate over constrained rewards.
"""
from __future__ import annotations

import math
import re

from skillopt.evaluator import ItemResult, SkillEvaluation
from skillopt.llm import BaseLLM
from skillopt.optimizer.base import Optimizer, register_optimizer
from skillopt.skills import FewShotExample, Skill, SkillOrigin

from ._common import (
    VALID_COST_METRICS,
    ScoredSkill,
    check_choice,
    check_non_negative_float,
    check_non_negative_int,
    check_positive_int,
    compact_preview,
    extract_skill_candidate,
    format_benchmark_table,
    initial_baseline_skills,
    next_skill_id,
    score_evaluations,
    skill_signature,
    stable_hash,
    validate_initial_baselines,
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_for_demo(text: str, max_chars: int) -> str:
    cleaned = _THINK_BLOCK_RE.sub("", text or "").strip()
    return compact_preview(cleaned, max_chars)


@register_optimizer("mipro")
class MIPROOptimizer(Optimizer):
    """Joint instruction + few-shot demo surrogate optimizer (``mipro``)."""

    def __init__(
        self,
        llm: BaseLLM,
        initial: list[str] | tuple[str, ...] = ("direct", "cot", "cod"),
        anchor_id: str = "cot",
        epsilon: float = 0.02,
        cost_metric: str = "output",
        num_candidates: int = 2,
        num_instructions: int = 6,
        num_demo_sets: int = 4,
        max_demos: int = 2,
        max_demo_chars: int = 300,
        ucb_beta: float = 0.35,
        max_skill_chars: int = 2400,
        seed: int = 0,
        max_retries: int = 2,
        **gen_kwargs,
    ) -> None:
        init_list, anchor_s = validate_initial_baselines(initial, anchor_id)
        eps_f = check_non_negative_float(epsilon, "epsilon")
        metric_s = check_choice(cost_metric, VALID_COST_METRICS, "cost_metric")
        num_cand_i = check_non_negative_int(num_candidates, "num_candidates")
        num_instr_i = check_positive_int(num_instructions, "num_instructions")
        num_demo_i = check_positive_int(num_demo_sets, "num_demo_sets")
        max_demos_i = check_positive_int(max_demos, "max_demos")
        max_demo_chars_i = check_positive_int(max_demo_chars, "max_demo_chars")
        ucb_f = check_non_negative_float(ucb_beta, "ucb_beta")
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
            num_instructions=num_instr_i,
            num_demo_sets=num_demo_i,
            max_demos=max_demos_i,
            max_demo_chars=max_demo_chars_i,
            ucb_beta=ucb_f,
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
        self.num_instructions = num_instr_i
        self.num_demo_sets = num_demo_i
        self.max_demos = max_demos_i
        self.max_demo_chars = max_demo_chars_i
        self.ucb_beta = ucb_f
        self.max_skill_chars = max_skill_i
        self.seed = seed
        self.max_retries = max_retries_i
        self.gen_kwargs = dict(gen_kwargs)

    def initial_prompts(self) -> list[Skill]:
        return initial_baseline_skills(self.initial, self.anchor_id)

    def _bootstrap_demo_sets(self, rounds: list[list[SkillEvaluation]]) -> list[list[FewShotExample]]:
        """Build zero-shot ``[]`` plus compact correct few-shot demo sets across benchmarks."""
        by_bench: dict[str, list[tuple[int, str, ItemResult]]] = {}
        for rnd in rounds:
            for ev in rnd:
                for b, res in ev.results.items():
                    for it in res.items:
                        if it.correct and not it.truncated and (it.question or "").strip():
                            ans = _strip_think_for_demo(it.response_text or str(it.gold), self.max_demo_chars)
                            if ans:
                                by_bench.setdefault(b, []).append((it.usage.output_tokens, ans, it))

        per_bench_best: dict[str, list[FewShotExample]] = {}
        all_pool: list[tuple[int, str, FewShotExample]] = []
        for b in sorted(by_bench):
            seen_q: set[str] = set()
            items = sorted(by_bench[b], key=lambda t: (t[0], t[2].example_id))
            picks: list[FewShotExample] = []
            for tok, ans, it in items:
                q = compact_preview(it.question, self.max_demo_chars)
                if q in seen_q:
                    continue
                seen_q.add(q)
                fs = FewShotExample(question=q, answer=ans)
                picks.append(fs)
                all_pool.append((tok, b, fs))
                if len(picks) >= self.max_demos:
                    break
            if picks:
                per_bench_best[b] = picks

        demo_sets: list[list[FewShotExample]] = [[]]  # idx 0 = zero-shot baseline
        if not all_pool:
            return demo_sets

        all_pool.sort(key=lambda t: (t[0], t[1]))
        # Set 1: cross-benchmark ultra-compact multi-shot (1 from distinct shortest benchmarks)
        cross: list[FewShotExample] = []
        used_benches: set[str] = set()
        for _, b, fs in all_pool:
            if b not in used_benches:
                cross.append(fs)
                used_benches.add(b)
            if len(cross) >= self.max_demos:
                break
        if cross:
            demo_sets.append(cross)

        # Remaining sets: single compact demos or per-benchmark specialists
        seen_sigs = {skill_signature("", ds) for ds in demo_sets}
        for _, _, fs in all_pool:
            if len(demo_sets) >= self.num_demo_sets:
                break
            cand = [fs]
            sig = skill_signature("", cand)
            if sig not in seen_sigs:
                demo_sets.append(cand)
                seen_sigs.add(sig)

        for b in sorted(per_bench_best):
            if len(demo_sets) >= self.num_demo_sets:
                break
            cand = per_bench_best[b][: self.max_demos]
            sig = skill_signature("", cand)
            if sig not in seen_sigs:
                demo_sets.append(cand)
                seen_sigs.add(sig)

        return demo_sets[: self.num_demo_sets]

    def _propose_instructions(
        self,
        scored: list[ScoredSkill],
        anchor: ScoredSkill | None,
        demo_sets: list[list[FewShotExample]],
    ) -> list[str]:
        pool: list[str] = []
        seen_instr: set[str] = set()
        for sc in scored:
            instr = (sc.ev.skill.instruction or "").strip()
            h = stable_hash(instr)
            if instr and h not in seen_instr:
                pool.append(instr)
                seen_instr.add(h)

        best = max(scored, key=lambda s: s.rank_key)
        demo_preview = "\n".join(
            f"Demo {idx + 1}: Q: {compact_preview(fs.question, 100)} -> A: {compact_preview(fs.answer, 100)}"
            for ds in demo_sets[1:2]
            for idx, fs in enumerate(ds)
        )
        summary_block = format_benchmark_table(best, anchor)

        failures = 0
        max_total_attempts = (self.num_instructions - len(pool)) * (1 + self.max_retries)
        while len(pool) < self.num_instructions and failures <= max(max_total_attempts, self.max_retries):
            style_hint = (
                "Emphasize ultra-short symbolic or telegraphic reasoning steps."
                if len(pool) % 2 == 0
                else "Emphasize direct verification and avoiding redundant re-derivation inside <think>."
            )
            prompt = (
                "You are generating a diverse candidate instruction for MIPROv2 joint instruction+demo search.\n"
                f"Objective: MINIMIZE generated tokens ({self.cost_metric}) while preserving accuracy >= "
                f"anchor - {self.epsilon:.3f}.\n\n"
                f"Current best candidate '{best.ev.skill_id}' (acc={best.mean_acc:.3f}, cost={best.norm_cost:.2f}x):\n"
                f"```\n{best.ev.skill.instruction}\n```\n"
                f"Per-benchmark scores:\n{summary_block}\n\n"
                + (f"Example compact few-shot demos:\n{demo_preview}\n\n" if demo_preview else "")
                + (
                    f"Already explored instructions (write something DISTINCT):\n"
                    + "\n".join(f"- {compact_preview(p, 120)}" for p in pool)
                    + "\n\n"
                )
                + f"Direction: {style_hint}\n"
                + f"Write ONE concise skill instruction (<= {self.max_skill_chars} chars) inside <skill>...</skill> tags."
            )
            resp = self.llm.generate(prompt, **self.gen_kwargs)
            cand, reason = extract_skill_candidate(
                resp.text, resp.truncated, seen_instr, self.max_skill_chars
            )
            if reason is None:
                pool.append(cand)
                seen_instr.add(stable_hash(cand))
            else:
                failures += 1

        return pool

    def _reward(self, sc: ScoredSkill) -> float:
        return (1.0 - sc.norm_cost) if sc.feasible else (-1.0 + sc.mean_acc)

    def _select_ucb_pairs(
        self,
        scored: list[ScoredSkill],
        instructions: list[str],
        demo_sets: list[list[FewShotExample]],
        seen_sigs: set[str],
    ) -> list[tuple[int, int, float]]:
        """Rank unobserved ``(instr_idx, demo_idx)`` pairs by additive empirical-Bayes UCB."""
        instr_map = {stable_hash(txt): idx for idx, txt in enumerate(instructions)}
        demo_map = {skill_signature("", ds): idx for idx, ds in enumerate(demo_sets)}

        instr_sum = [0.0] * len(instructions)
        instr_cnt = [0] * len(instructions)
        demo_sum = [0.0] * len(demo_sets)
        demo_cnt = [0] * len(demo_sets)
        rewards: list[float] = []

        for sc in scored:
            if not sc.bench_acc:
                continue
            r = self._reward(sc)
            rewards.append(r)
            meta = sc.ev.skill.metadata if isinstance(sc.ev.skill.metadata, dict) else {}
            i_idx = meta.get("instruction_idx")
            if not isinstance(i_idx, int) or not (0 <= i_idx < len(instructions)):
                i_idx = instr_map.get(stable_hash(sc.ev.skill.instruction))
            d_idx = meta.get("demo_set_idx")
            if not isinstance(d_idx, int) or not (0 <= d_idx < len(demo_sets)):
                d_idx = demo_map.get(skill_signature("", sc.ev.skill.few_shot))

            if isinstance(i_idx, int) and 0 <= i_idx < len(instructions):
                instr_sum[i_idx] += r
                instr_cnt[i_idx] += 1
            if isinstance(d_idx, int) and 0 <= d_idx < len(demo_sets):
                demo_sum[d_idx] += r
                demo_cnt[d_idx] += 1

        n_obs = max(1, len(rewards))
        mu_prior = sum(rewards) / n_obs if rewards else 0.0
        log_n = math.log(n_obs + 1.0)

        candidates: list[tuple[int, int, float]] = []
        for i_idx, instr in enumerate(instructions):
            mu_i = (instr_sum[i_idx] + mu_prior) / (instr_cnt[i_idx] + 1.0)
            bonus_i = self.ucb_beta * math.sqrt(log_n / (instr_cnt[i_idx] + 1.0))
            for d_idx, ds in enumerate(demo_sets):
                if skill_signature(instr, ds) in seen_sigs:
                    continue
                mu_j = (demo_sum[d_idx] + mu_prior) / (demo_cnt[d_idx] + 1.0)
                bonus_j = self.ucb_beta * math.sqrt(log_n / (demo_cnt[d_idx] + 1.0))
                score = 0.5 * (mu_i + mu_j) + bonus_i + bonus_j
                candidates.append((i_idx, d_idx, score))

        candidates.sort(key=lambda t: (-t[2], t[0], t[1]))
        return candidates[: self.num_candidates]

    def update(self, results: list[SkillEvaluation], history: list[list[SkillEvaluation]]) -> list[Skill]:
        if self.num_candidates == 0:
            return []
        rounds = list(history) + [list(results)]
        iteration = len(history) + 1
        anchor, scored, _ = score_evaluations(rounds, self.anchor_id, self.epsilon, self.cost_metric)
        if not scored:
            return []

        demo_sets = self._bootstrap_demo_sets(rounds)
        instructions = self._propose_instructions(scored, anchor, demo_sets)
        if not instructions or not demo_sets:
            return []

        seen_sigs: set[str] = {
            skill_signature(sc.ev.skill.instruction, sc.ev.skill.few_shot) for sc in scored
        }
        used_ids: set[str] = {sc.ev.skill_id for sc in scored}
        picks = self._select_ucb_pairs(scored, instructions, demo_sets, seen_sigs)
        best = max(scored, key=lambda s: s.rank_key)

        proposals: list[Skill] = []
        for slot, (i_idx, d_idx, ucb) in enumerate(picks):
            instr = instructions[i_idx]
            demos = [FewShotExample(question=fs.question, answer=fs.answer) for fs in demo_sets[d_idx]]
            sig = skill_signature(instr, demos)
            seen_sigs.add(sig)
            proposals.append(
                Skill(
                    id=next_skill_id("mipro", iteration, slot, used_ids),
                    instruction=instr,
                    few_shot=demos,
                    parent_id=best.ev.skill.id,
                    metadata={
                        "instruction_idx": i_idx,
                        "demo_set_idx": d_idx,
                        "ucb": round(ucb, 4),
                    },
                    origin=SkillOrigin(
                        parents=[best.ev.skill.id],
                        operation="mipro_search",
                        note=f"UCB pair (instr={i_idx}, demos={d_idx}, k={len(demos)}, ucb={ucb:.3f})",
                    ),
                )
            )
        return proposals
