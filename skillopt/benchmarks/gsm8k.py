"""GSM8K grade-school math benchmark."""
from __future__ import annotations

import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .base import Benchmark, Example, _resolve, read_jsonl, register_benchmark

_NUM = r"-?[\d,]*\.?\d+"
_ANSWER_IS_RE = re.compile(r"answer is[:\s]*\$?(" + _NUM + r")", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_HASH_RE = re.compile(r"####\s*\$?(" + _NUM + r")")
_NUM_RE = re.compile(_NUM)
_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")


def normalize_number(s: str | None) -> str | None:
    if s is None:
        return None
    t = str(s).strip().replace("$", "").replace(",", "").strip()
    while t.endswith("."):
        t = t[:-1].rstrip()
    try:
        d = Decimal(t)
    except InvalidOperation:
        return t.strip().lower()
    if not d.is_finite():
        return t.lower()
    if d == d.to_integral_value():
        return str(int(d))
    out = format(d.normalize(), "f")
    return out


@register_benchmark("gsm8k")
class GSM8K(Benchmark):
    name = "gsm8k"
    answer_format = 'End your response with "The answer is N." where N is the final number.'
    splits = ("train", "val", "test")

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        train: int = 200,
        val: int = 200,
        seed: int = 0,
    ) -> None:
        if source not in ("local", "hf"):
            raise ValueError(f"unknown gsm8k source '{source}' (expected 'local' or 'hf')")
        self.source = source
        self.data_dir: Path | None = _resolve(data_dir) if data_dir is not None else None
        if source == "local" and self.data_dir is None:
            self.data_dir = _resolve("data/sample/gsm8k")
        self.train = int(train)
        self.val = int(val)
        self.seed = int(seed)
        self._raw: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ data
    def _records(self, which: str) -> list[dict]:
        if which not in self._raw:
            if self.source == "hf":
                import datasets  # lazy heavy import

                ds = datasets.load_dataset("openai/gsm8k", "main", split=which)
                self._raw[which] = [{"question": r["question"], "answer": r["answer"]} for r in ds]
            else:
                assert self.data_dir is not None
                self._raw[which] = read_jsonl(self.data_dir / f"{which}.jsonl")
        return self._raw[which]

    @staticmethod
    def _gold(answer: str) -> str:
        tail = answer.split("####")[-1] if "####" in answer else answer
        return normalize_number(tail) or ""

    def _example(self, prefix: str, idx: int, rec: dict) -> Example:
        return Example(
            id=f"{prefix}-{idx}",
            question=rec["question"],
            answer=self._gold(rec["answer"]),
            meta={"solution": rec["answer"]},
        )

    def load(self, split: str) -> list[Example]:
        if split in ("train", "val"):
            records = self._records("train")
            order = list(range(len(records)))
            random.Random(self.seed).shuffle(order)
            train_idx = order[: self.train]
            val_idx = order[self.train : self.train + self.val]
            chosen = train_idx if split == "train" else val_idx
            return [self._example("train", i, records[i]) for i in chosen]
        if split == "test":
            return [self._example("test", i, r) for i, r in enumerate(self._records("test"))]
        raise ValueError(f"unknown gsm8k split '{split}' (expected train, val or test)")

    # ------------------------------------------------------------------ scoring
    def extract_answer(self, text: str) -> str | None:
        if not text:
            return None
        text = _TAG_RE.sub(" ", text)  # "The answer is <answer>42</answer>." -> "42"
        m = _ANSWER_IS_RE.findall(text)
        if m:
            return normalize_number(m[-1])
        boxed = _BOXED_RE.findall(text)
        for content in reversed(boxed):
            nums = _NUM_RE.findall(content)
            if nums:
                return normalize_number(nums[-1])
        m = _HASH_RE.findall(text)
        if m:
            return normalize_number(m[-1])
        nums = _NUM_RE.findall(text)
        if nums:
            return normalize_number(nums[-1])
        return None

    def score(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        return normalize_number(pred) == normalize_number(gold)
