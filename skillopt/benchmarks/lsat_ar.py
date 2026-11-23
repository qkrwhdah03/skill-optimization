"""AGIEval LSAT Analytical Reasoning (constraint-satisfaction logic games) benchmark."""
from __future__ import annotations

import re

from .base import Example, read_jsonl, register_benchmark
from .mcq import extract_mcq_answer, format_mcq_options, mcq_answer_format, score_mcq
from .pooled import PooledBenchmark

_TRAILING_PROMPT_RE = re.compile(
    r"\s*A:\s*Among\s+A\s+through\s+E,\s*the\s+answer\s+is\s*:?\s*$",
    re.IGNORECASE,
)


def _gold_letter(raw_gold: object) -> str:
    val = raw_gold[0] if isinstance(raw_gold, (list, tuple)) else raw_gold
    if isinstance(val, int) and not isinstance(val, bool):
        return chr(ord("a") + val)
    s = str(val).strip()
    if s.isdigit():
        return chr(ord("a") + int(s))
    return s.strip("()").lower()


@register_benchmark("lsat_ar")
class LSATAR(PooledBenchmark):
    name = "lsat_ar"
    answer_format = mcq_answer_format(5)
    default_data_dir = "data/sample/lsat_ar"

    def _load_records(self) -> list[dict]:
        if self.source == "hf":
            import datasets  # lazy heavy import

            ds = datasets.load_dataset("hails/agieval-lsat-ar", split="test")
            return [{"query": r["query"], "choices": list(r["choices"]), "gold": list(r["gold"])} for r in ds]
        assert self.data_dir is not None
        return read_jsonl(self.data_dir / "data.jsonl")

    def _example(self, idx: int, rec: dict) -> Example:
        query = _TRAILING_PROMPT_RE.sub("", str(rec["query"])).strip()
        choices = list(rec.get("choices") or ())
        if "(A)" not in query and choices:
            query = f"{query}\n\nOptions:\n{format_mcq_options(choices)}"
        num_choices = len(choices) if choices else 5
        return Example(
            id=f"{self.name}-{idx}",
            question=query,
            answer=_gold_letter(rec["gold"]),
            meta={"num_choices": num_choices},
        )

    def extract_answer(self, text: str) -> str | None:
        return extract_mcq_answer(text, "abcde")

    def score(self, pred: str | None, gold: str) -> bool:
        return score_mcq(pred, gold)
