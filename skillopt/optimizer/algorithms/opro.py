"""OPRO-style optimizer (a worked example of the ``Optimizer`` interface).

Round 0 evaluates the baselines named in ``initial``. Each ``update`` shows the optimizer LLM the scored past
instructions (per benchmark) and failures of the best one, and asks for better instructions.

Ranking is THIS algorithm's own choice (not the framework's): mean accuracy desc, then mean avg_output_tokens asc,
both averaged over benchmarks with ``n > 0``.

Each of the ``num_candidates`` slots gets up to ``1 + max_retries`` generations; an attempt is rejected when it is
empty, a duplicate of a known instruction, or truncated (hit max_tokens) without a complete ``<skill>`` block.
Rejections are logged at DEBUG; if fewer than ``num_candidates`` prompts are produced, one WARNING summarizes why
(an empty result stops the loop).
"""
from __future__ import annotations

import logging
import re

from skillopt.baselines import get_baseline
from skillopt.evaluator import EvalResult, SkillEvaluation
from skillopt.llm import BaseLLM
from skillopt.optimizer.base import Optimizer, register_optimizer
from skillopt.skills import FewShotExample, Skill

_SKILL_RE = re.compile(r"<skill>(.*?)</skill>", re.DOTALL | re.IGNORECASE)

logger = logging.getLogger(__name__)


def _means(ev: SkillEvaluation) -> tuple[float, float]:
    """Mean accuracy and mean avg output tokens over benchmarks with ``n > 0`` (0.0 if none)."""
    used: list[EvalResult] = [r for r in ev.results.values() if r.n > 0]
    if not used:
        return 0.0, 0.0
    acc = sum(r.accuracy for r in used) / len(used)
    tok = sum(r.avg_output_tokens for r in used) / len(used)
    return acc, tok


def _rank(evals: list[SkillEvaluation]) -> list[tuple[SkillEvaluation, float, float]]:
    scored = [(e, *_means(e)) for e in evals]
    return sorted(scored, key=lambda t: (-t[1], t[2]))


@register_optimizer("opro")
class OPROOptimizer(Optimizer):
    def __init__(
        self,
        llm: BaseLLM,
        initial: list[str] | tuple[str, ...] = ("direct", "cot", "cod"),
        num_candidates: int = 2,
        top_k: int = 5,
        num_failures: int = 3,
        max_trace_chars: int = 600,
        max_retries: int = 2,
        **gen_kwargs,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError(f"max_retries must be an int >= 0, got {max_retries!r}")
        super().__init__(
            llm,
            initial=list(initial),
            num_candidates=num_candidates,
            top_k=top_k,
            num_failures=num_failures,
            max_trace_chars=max_trace_chars,
            max_retries=max_retries,
            **gen_kwargs,
        )
        self.initial = list(initial)
        self.num_candidates = int(num_candidates)
        self.top_k = int(top_k)
        self.num_failures = int(num_failures)
        self.max_trace_chars = int(max_trace_chars)
        self.max_retries = max_retries
        self.gen_kwargs = dict(gen_kwargs)

    def initial_prompts(self) -> list[Skill]:
        return [get_baseline(name) for name in self.initial]

    def build_meta_prompt(self, evals: list[SkillEvaluation]) -> str:
        top = _rank(evals)[: self.top_k]
        parts: list[str] = [
            "You are optimizing a reasoning instruction (a \"skill\") that is given to a language model "
            "before it solves a problem. The goal is to keep accuracy high while using as few output "
            "tokens (including thinking tokens) as possible."
        ]
        if top:
            lines = ["Previous instructions with their scores, from worst to best:"]
            for ev, mean_acc, mean_tok in reversed(top):
                block = [f"<instruction>\n{ev.skill.instruction.strip()}\n</instruction>"]
                for key, r in ev.results.items():
                    block.append(f"{key}: accuracy: {r.accuracy:.3f}, avg output tokens: {r.avg_output_tokens:.1f}")
                block.append(f"mean accuracy: {mean_acc:.3f}, mean avg output tokens: {mean_tok:.1f}")
                lines.append("\n".join(block))
            parts.append("\n\n".join(lines))

            best = top[0][0]
            if self.num_failures > 0:
                flines: list[str] = []
                for key, r in best.results.items():
                    for i, f in enumerate(r.failures(self.num_failures), 1):
                        trace = (f.reasoning_text or f.response_text or "")[: self.max_trace_chars]
                        flines.append(
                            f"[{key} #{i}] Question: {f.question}\n"
                            f"Model prediction: {f.prediction}\n"
                            f"Correct answer: {f.gold}\n"
                            f"Reasoning (truncated): {trace}"
                        )
                if flines:
                    parts.append("\n\n".join([f"Examples the best instruction ({best.skill_id}) got wrong:"] + flines))
        parts.append(
            "Write a NEW instruction that is different from the ones above, keeps (or improves) accuracy, "
            "and makes the model use fewer tokens. Output only the instruction wrapped in "
            "<skill>...</skill>."
        )
        return "\n\n".join(parts)

    @staticmethod
    def _parse(text: str) -> str:
        matches = _SKILL_RE.findall(text or "")
        if matches:
            return matches[-1].strip()
        return (text or "").strip()

    def update(self, results: list[SkillEvaluation], history: list[list[SkillEvaluation]]) -> list[Skill]:
        iteration = len(history) + 1
        evals = [ev for round_ in history for ev in round_] + list(results)
        ranked = _rank(evals)
        best = ranked[0][0].skill if ranked else None
        meta_prompt = self.build_meta_prompt(evals)

        seen = {ev.skill.instruction.strip() for ev in evals}
        used_ids = {ev.skill_id for ev in evals}
        proposals: list[Skill] = []
        rejected = {"empty": 0, "duplicate": 0, "truncated": 0}
        attempts = 0
        for k in range(self.num_candidates):
            for attempt in range(1 + self.max_retries):
                attempts += 1
                resp = self.llm.generate(meta_prompt, **self.gen_kwargs)
                blocks = _SKILL_RE.findall(resp.text or "")
                reason = None
                instruction = ""
                if blocks:
                    instruction = blocks[-1].strip()
                elif resp.truncated:
                    reason = "truncated"
                else:
                    instruction = self._parse(resp.text)
                if reason is None:
                    if not instruction:
                        reason = "empty"
                    elif instruction in seen:
                        reason = "duplicate"
                if reason is not None:
                    rejected[reason] += 1
                    logger.debug(f"update round {iteration}: candidate {k} attempt {attempt + 1} rejected ({reason})")
                    continue
                seen.add(instruction)
                base_id = f"opro-i{iteration}-c{k}"
                sid, n = base_id, 1
                while sid in used_ids:
                    n += 1
                    sid = f"{base_id}-{n}"
                used_ids.add(sid)
                few_shot = (
                    [FewShotExample(question=fs.question, answer=fs.answer) for fs in best.few_shot] if best else []
                )
                proposals.append(
                    Skill(
                        id=sid,
                        instruction=instruction,
                        few_shot=few_shot,
                        parent_id=best.id if best else None,
                        metadata={"optimizer": "opro", "iteration": iteration, "optimizer_model": self.llm.model},
                    )
                )
                break
        if len(proposals) < self.num_candidates:
            msg = (
                f"update round {iteration}: {len(proposals)}/{self.num_candidates} new prompts after {attempts} "
                f"generations (rejected: empty={rejected['empty']}, duplicate={rejected['duplicate']}, "
                f"truncated={rejected['truncated']})"
            )
            if not proposals:
                msg += "; returning [] stops the loop"
            logger.warning(msg)
        return proposals
