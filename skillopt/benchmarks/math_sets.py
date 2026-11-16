"""Competition-math benchmarks (MATH-500, AIME 2024/2025, AMC 2023): single pool, ``\\boxed{}`` answers."""
from __future__ import annotations

from typing import ClassVar

from .base import Example, read_jsonl, register_benchmark
from .math_answer import MATH_ANSWER_FORMAT, extract_math_answer, math_answers_equal
from .pooled import PooledBenchmark


def _field(row, key: str):
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return None


class MathBenchmark(PooledBenchmark):
    """Math set with one pool; subclasses only set the class attributes below."""

    answer_format = MATH_ANSWER_FORMAT
    hf_path: ClassVar[str]
    hf_config: ClassVar[str | None] = None
    hf_split: ClassVar[str] = "test"
    question_field: ClassVar[str] = "problem"
    answer_field: ClassVar[str] = "answer"
    meta_fields: ClassVar[tuple[str, ...]] = ("id",)

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        train: int = 0,
        val: int = 0,
        seed: int = 0,
    ) -> None:
        super().__init__(source=source, data_dir=data_dir, train=train, val=val, seed=seed)

    # ------------------------------------------------------------------ data
    def _row(self, r) -> dict:
        rec = {self.question_field: r[self.question_field], self.answer_field: r[self.answer_field]}
        for key in self.meta_fields:
            value = _field(r, key)
            if value is not None:
                rec[key] = value
        return rec

    def _load_records(self) -> list[dict]:
        if self.source == "hf":
            import datasets  # lazy heavy import

            if self.hf_config is None:
                ds = datasets.load_dataset(self.hf_path, split=self.hf_split)
            else:
                ds = datasets.load_dataset(self.hf_path, self.hf_config, split=self.hf_split)
            return [self._row(r) for r in ds]
        assert self.data_dir is not None
        return [self._row(r) for r in read_jsonl(self.data_dir / "data.jsonl")]

    def _example(self, idx: int, rec: dict) -> Example:
        return Example(
            id=f"{self.name}-{idx}",
            question=str(rec[self.question_field]),
            answer=str(rec[self.answer_field]).strip(),
            meta={k: rec[k] for k in self.meta_fields if k in rec},
        )

    # ------------------------------------------------------------------ scoring
    def extract_answer(self, text: str) -> str | None:
        return extract_math_answer(text)

    def score(self, pred: str | None, gold: str) -> bool:
        return math_answers_equal(pred, gold)


@register_benchmark("math500")
class MATH500(MathBenchmark):
    name = "math500"
    hf_path = "HuggingFaceH4/MATH-500"
    hf_split = "test"
    default_data_dir = "data/sample/math500"
    meta_fields = ("subject", "level", "unique_id")


@register_benchmark("aime2024")
class AIME2024(MathBenchmark):
    name = "aime2024"
    hf_path = "HuggingFaceH4/aime_2024"
    hf_split = "train"
    default_data_dir = "data/sample/aime2024"


@register_benchmark("aime2025")
class AIME2025(MathBenchmark):
    name = "aime2025"
    hf_path = "math-ai/aime25"
    hf_split = "test"
    default_data_dir = "data/sample/aime2025"


@register_benchmark("amc2023")
class AMC2023(MathBenchmark):
    name = "amc2023"
    hf_path = "math-ai/amc23"
    hf_split = "test"
    question_field = "question"
    default_data_dir = "data/sample/amc2023"
