"""TextGrad + Semantic Gradient Clustering Prompt Optimizer (``textgrad``).

Unifies three complementary prompt-gradient ideas for single-LLM, single-skill optimization:
1. *TextGrad* (Yuksekgonul et al., Nature 2025) & *ProTeGi* (Pryzant et al., EMNLP 2024):
   generates per-example natural-language critiques ("textual gradients") on failing
   and overly verbose reasoning traces, then applies ``TGD.step()`` updates across a beam.
2. *AgentGrad Step 2* (Chu et al., MLV): clusters similar sample-level textual gradients
   by lexical similarity and abstracts shared underlying rules before applying the update,
   avoiding prompt bloat and per-sample overfitting.

Target objective: ``min_s E[T(M, x, s)]  s.t.  A(M, s) >= A_anchor - epsilon``.
"""
from __future__ import annotations

import re

from skillopt.evaluator import ItemResult, SkillEvaluation
from skillopt.llm import BaseLLM
from skillopt.optimizer.base import Optimizer, register_optimizer
from skillopt.skills import Skill, SkillOrigin

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
    format_trace_item,
    initial_baseline_skills,
    item_cost,
    next_skill_id,
    score_evaluations,
    stable_hash,
    validate_initial_baselines,
)

_STOPWORDS = frozenset(
    {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
        "was", "were", "be", "it", "that", "this", "as", "at", "by", "from", "not", "too",
    }
)


def _token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2 and t not in _STOPWORDS}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def cluster_texts_jaccard(texts: list[str], threshold: float = 0.25) -> list[list[str]]:
    """Greedily group critiques into semantically overlapping clusters via token Jaccard similarity."""
    clusters: list[tuple[set[str], list[str]]] = []
    for txt in texts:
        toks = _token_set(txt)
        placed = False
        for c_toks, bucket in clusters:
            if _jaccard(toks, c_toks) >= threshold:
                bucket.append(txt)
                c_toks.update(toks)
                placed = True
                break
        if not placed:
            clusters.append((set(toks), [txt]))
    clusters.sort(key=lambda pair: -len(pair[1]))
    return [bucket for _, bucket in clusters]


@register_optimizer("textgrad")
class TextGradOptimizer(Optimizer):
    """Beam-search textual gradient descent with gradient clustering (``textgrad``)."""

    def __init__(
        self,
        llm: BaseLLM,
        initial: list[str] | tuple[str, ...] = ("direct", "cot", "cod"),
        anchor_id: str = "cot",
        epsilon: float = 0.02,
        cost_metric: str = "output",
        beam_width: int = 2,
        num_candidates: int = 2,
        num_error_grads: int = 3,
        num_bloat_grads: int = 3,
        num_clusters: int = 3,
        max_trace_chars: int = 450,
        max_skill_chars: int = 2400,
        max_retries: int = 2,
        **gen_kwargs,
    ) -> None:
        init_list, anchor_s = validate_initial_baselines(initial, anchor_id)
        eps_f = check_non_negative_float(epsilon, "epsilon")
        metric_s = check_choice(cost_metric, VALID_COST_METRICS, "cost_metric")
        beam_i = check_positive_int(beam_width, "beam_width")
        num_cand_i = check_non_negative_int(num_candidates, "num_candidates")
        err_i = check_non_negative_int(num_error_grads, "num_error_grads")
        bloat_i = check_non_negative_int(num_bloat_grads, "num_bloat_grads")
        clust_i = check_positive_int(num_clusters, "num_clusters")
        max_trace_i = check_non_negative_int(max_trace_chars, "max_trace_chars")
        max_skill_i = check_positive_int(max_skill_chars, "max_skill_chars")
        max_retries_i = check_non_negative_int(max_retries, "max_retries")

        super().__init__(
            llm,
            initial=init_list,
            anchor_id=anchor_s,
            epsilon=eps_f,
            cost_metric=metric_s,
            beam_width=beam_i,
            num_candidates=num_cand_i,
            num_error_grads=err_i,
            num_bloat_grads=bloat_i,
            num_clusters=clust_i,
            max_trace_chars=max_trace_i,
            max_skill_chars=max_skill_i,
            max_retries=max_retries_i,
            **gen_kwargs,
        )
        self.initial = init_list
        self.anchor_id = anchor_s
        self.epsilon = eps_f
        self.cost_metric = metric_s
        self.beam_width = beam_i
        self.num_candidates = num_cand_i
        self.num_error_grads = err_i
        self.num_bloat_grads = bloat_i
        self.num_clusters = clust_i
        self.max_trace_chars = max_trace_i
        self.max_skill_chars = max_skill_i
        self.max_retries = max_retries_i
        self.gen_kwargs = dict(gen_kwargs)

    def initial_prompts(self) -> list[Skill]:
        return initial_baseline_skills(self.initial, self.anchor_id)

    def _sample_gradient_traces(self, parent: ScoredSkill) -> list[tuple[str, str, ItemResult]]:
        """Collect error/truncation traces and verbose-correct traces across benchmarks."""
        errors: list[tuple[str, ItemResult]] = []
        bloated: list[tuple[str, ItemResult]] = []
        for b, res in parent.ev.results.items():
            for it in res.items:
                if not it.correct or it.truncated:
                    errors.append((b, it))
                else:
                    bloated.append((b, it))

        errors.sort(key=lambda x: (0 if x[1].truncated else 1, -item_cost(x[1], self.cost_metric)))
        bloated.sort(key=lambda x: -item_cost(x[1], self.cost_metric))

        selected: list[tuple[str, str, ItemResult]] = []
        # Round-robin across benchmarks when picking error traces
        seen_b: set[str] = set()
        for b, it in errors:
            if len(selected) >= self.num_error_grads:
                break
            if b not in seen_b:
                selected.append(("error", b, it))
                seen_b.add(b)
        for b, it in errors:
            if len(selected) >= self.num_error_grads:
                break
            if not any(it is existing for _, _, existing in selected):
                selected.append(("error", b, it))

        bloat_picks = 0
        seen_b_bloat: set[str] = set()
        for b, it in bloated:
            if bloat_picks >= self.num_bloat_grads:
                break
            if b not in seen_b_bloat:
                selected.append(("bloat", b, it))
                seen_b_bloat.add(b)
                bloat_picks += 1
        for b, it in bloated:
            if bloat_picks >= self.num_bloat_grads:
                break
            if not any(it is existing for _, _, existing in selected):
                selected.append(("bloat", b, it))
                bloat_picks += 1

        return selected

    def _extract_textual_gradients(self, parent: ScoredSkill) -> list[str]:
        samples = self._sample_gradient_traces(parent)
        if not samples:
            return []
        prompts: list[str] = []
        for kind, bench, it in samples:
            goal = (
                "Explain concisely (1-2 sentences) why the reasoning failed or hit max_tokens and what general "
                "rule in the system skill would prevent it."
                if kind == "error"
                else "Explain concisely (1-2 sentences) what thinking habits caused excessive token bloat here "
                "and how the system skill should instruct the model to skip them."
            )
            prompts.append(
                f"Current skill:\n```\n{parent.ev.skill.instruction}\n```\n\n"
                f"Example trace:\n{format_trace_item(bench, it, self.max_trace_chars)}\n\n"
                f"Textual gradient critique: {goal}"
            )
        responses = self.llm.generate_batch(prompts, **self.gen_kwargs)
        gradients: list[str] = []
        for resp in responses:
            txt = compact_preview(resp.text or "", 260)
            if txt:
                gradients.append(txt)
        return gradients

    def _cluster_and_abstract_gradients(self, gradients: list[str]) -> list[str]:
        if not gradients:
            return []
        clusters = cluster_texts_jaccard(gradients)[: self.num_clusters]
        abstracted: list[str] = []
        for cluster in clusters:
            if len(cluster) == 1:
                abstracted.append(cluster[0])
                continue
            joined = "\n".join(f"- {g}" for g in cluster)
            prompt = (
                "Synthesize the following sample-level prompt critiques into ONE concise, generalizable rule "
                "for improving a reasoning skill (<= 40 words, no dataset-specific trivia):\n"
                f"{joined}\n\nSynthesized gradient rule:"
            )
            resp = self.llm.generate(prompt, **self.gen_kwargs)
            rule = compact_preview(resp.text or "", 240)
            abstracted.append(rule if rule else cluster[0])
        return abstracted

    def _apply_tgd_step(
        self,
        parent: ScoredSkill,
        anchor: ScoredSkill | None,
        gradients: list[str],
        iteration: int,
        slot: int,
        used_ids: set[str],
        seen_hashes: set[str],
    ) -> Skill | None:
        grad_block = "\n".join(f"{i + 1}. {g}" for i, g in enumerate(gradients)) or "1. Reduce verbose thinking while preserving accuracy."
        prompt = (
            "Perform one Textual Gradient Descent step (TGD.step) on the current reasoning skill.\n"
            f"Objective: MINIMIZE generated tokens ({self.cost_metric}) while maintaining accuracy >= "
            f"anchor - {self.epsilon:.3f}.\n\n"
            f"Current skill '{parent.ev.skill_id}' (acc={parent.mean_acc:.3f}, cost={parent.norm_cost:.2f}x):\n"
            f"```\n{parent.ev.skill.instruction}\n```\n\n"
            f"Per-benchmark scores:\n{format_benchmark_table(parent, anchor)}\n\n"
            f"Abstracted textual gradients:\n{grad_block}\n\n"
            f"Apply these gradients to write an updated, compact skill (<= {self.max_skill_chars} chars).\n"
            "Return ONLY the updated skill inside <skill>...</skill> tags."
        )
        for _ in range(1 + self.max_retries):
            resp = self.llm.generate(prompt, **self.gen_kwargs)
            cand, reason = extract_skill_candidate(
                resp.text, resp.truncated, seen_hashes, self.max_skill_chars
            )
            if reason is None:
                seen_hashes.add(stable_hash(cand))
                pid = parent.ev.skill.id
                return Skill(
                    id=next_skill_id("tgd", iteration, slot, used_ids),
                    instruction=cand,
                    few_shot=list(parent.ev.skill.few_shot),
                    parent_id=pid,
                    metadata={"num_gradients": len(gradients), "parent_cost": round(parent.norm_cost, 4)},
                    origin=SkillOrigin(
                        parents=[pid],
                        operation="textgrad_step",
                        note=f"TGD step on {pid} with {len(gradients)} abstracted gradient cluster(s)",
                    ),
                )
        return None

    def update(self, results: list[SkillEvaluation], history: list[list[SkillEvaluation]]) -> list[Skill]:
        if self.num_candidates == 0:
            return []
        rounds = list(history) + [list(results)]
        iteration = len(history) + 1
        anchor, scored, _ = score_evaluations(rounds, self.anchor_id, self.epsilon, self.cost_metric)
        active = [sc for sc in scored if sc.bench_acc]
        if not active:
            return []

        latest: dict[str, ScoredSkill] = {sc.ev.skill_id: sc for sc in active}
        beam = sorted(latest.values(), key=lambda s: s.rank_key, reverse=True)[: self.beam_width]

        seen_hashes: set[str] = {stable_hash(sc.ev.skill.instruction) for sc in scored}
        used_ids: set[str] = {sc.ev.skill_id for sc in scored}

        gradients_cache: dict[str, list[str]] = {}
        proposals: list[Skill] = []
        for slot in range(self.num_candidates):
            parent = beam[slot % len(beam)]
            pid = parent.ev.skill_id
            if pid not in gradients_cache:
                raw_grads = self._extract_textual_gradients(parent)
                gradients_cache[pid] = self._cluster_and_abstract_gradients(raw_grads)
            skill = self._apply_tgd_step(
                parent, anchor, gradients_cache[pid], iteration, slot, used_ids, seen_hashes
            )
            if skill is not None:
                proposals.append(skill)
        return proposals
