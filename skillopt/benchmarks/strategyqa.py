"""StrategyQA (multi-hop implicit commonsense reasoning) benchmark."""
from __future__ import annotations

import random
import re
from pathlib import Path

from .base import Benchmark, Example, _resolve, read_jsonl, register_benchmark
from .pooled import configured_splits

STRATEGYQA_ANSWER_FORMAT: str = 'End your response with "The answer is Yes." or "The answer is No."'

_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_PREFIX_RE = re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_YES_NO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_PUNCT_STRIP = " \t\r\n.,:;!?\"'()[]{}*`"


def normalize_yes_no(s: object) -> str | None:
    """Normalize a boolean or string verdict to ``'yes'`` or ``'no'``; else ``None``."""
    if s is None:
        return None
    if isinstance(s, bool):
        return "yes" if s else "no"
    t = _TAG_RE.sub(" ", str(s)).strip(_PUNCT_STRIP).lower()
    if t in ("yes", "true"):
        return "yes"
    if t in ("no", "false"):
        return "no"
    return None


def extract_yes_no_answer(text: str | None) -> str | None:
    """Extract ``'yes'`` or ``'no'`` from model output."""
    if text is None or not text.strip():
        return None
    cleaned = _TAG_RE.sub(" ", text).replace("*", "").replace("`", "")

    prefix_matches = list(_PREFIX_RE.finditer(cleaned))
    if prefix_matches:
        rest = cleaned[prefix_matches[-1].end() :]
        for raw_line in rest.splitlines():
            line = raw_line.strip()
            if line:
                m = _YES_NO_RE.search(line)
                return m.group(1).lower() if m else None
        return None

    for inner in reversed(_BOXED_RE.findall(cleaned)):
        norm = normalize_yes_no(inner)
        if norm is not None:
            return norm

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return None
    hits = _YES_NO_RE.findall(lines[-1])
    return hits[-1].lower() if hits else None


@register_benchmark("strategyqa")
class StrategyQA(Benchmark):
    name = "strategyqa"
    answer_format = STRATEGYQA_ANSWER_FORMAT

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        train: int = 0,
        val: int = 0,
        seed: int = 0,
    ) -> None:
        if source not in ("local", "hf"):
            raise ValueError(f"unknown strategyqa source '{source}' (expected 'local' or 'hf')")
        self.source = source
        self.data_dir: Path | None = _resolve(data_dir) if data_dir is not None else None
        if source == "local" and self.data_dir is None:
            self.data_dir = _resolve("data/sample/strategyqa")
        self.train = int(train)
        self.val = int(val)
        if self.train < 0 or self.val < 0:
            raise ValueError(f"strategyqa: train and val must be >= 0 (got train={train}, val={val})")
        self.seed = int(seed)
        self.splits = configured_splits(self.train, self.val)
        self._raw: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ data
    def _records(self, which: str) -> list[dict]:
        if which not in self._raw:
            if self.source == "hf":
                import datasets  # lazy heavy import

                ds = datasets.load_dataset("ChilleD/StrategyQA", split=which)
                self._raw[which] = [
                    {"qid": r["qid"], "question": r["question"], "answer": r["answer"]} for r in ds
                ]
            else:
                assert self.data_dir is not None
                self._raw[which] = read_jsonl(self.data_dir / f"{which}.jsonl")
        return self._raw[which]

    @staticmethod
    def _example(prefix: str, idx: int, rec: dict) -> Example:
        meta = {"qid": rec["qid"]} if "qid" in rec else {}
        return Example(
            id=f"{prefix}-{idx}",
            question=str(rec["question"]).strip(),
            answer=normalize_yes_no(rec["answer"]) or "",
            meta=meta,
        )

    def load(self, split: str) -> list[Example]:
        if split in ("train", "val"):
            records = self._records("train")
            order = list(range(len(records)))
            random.Random(self.seed).shuffle(order)
            chosen = order[: self.train] if split == "train" else order[self.train : self.train + self.val]
            return [self._example("train", i, records[i]) for i in chosen]
        if split == "test":
            return [self._example("test", i, r) for i, r in enumerate(self._records("test"))]
        raise ValueError(f"unknown strategyqa split '{split}' (expected train, val or test)")

    # ------------------------------------------------------------------ scoring
    def extract_answer(self, text: str) -> str | None:
        return extract_yes_no_answer(text)

    def score(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        return normalize_yes_no(pred) == normalize_yes_no(gold)
