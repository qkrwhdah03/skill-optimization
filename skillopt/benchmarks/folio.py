"""FOLIO (First-Order Logic Reasoning in Natural Language) benchmark."""
from __future__ import annotations

import random
import re
from pathlib import Path

from .base import Benchmark, Example, _resolve, read_jsonl, register_benchmark
from .pooled import configured_splits

FOLIO_LABELS: tuple[str, ...] = ("true", "false", "uncertain")
FOLIO_ANSWER_FORMAT: str = (
    'End your response with "The answer is True.", "The answer is False.", or "The answer is Uncertain."'
)

_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_PREFIX_RE = re.compile(r"(?:the\s+)?(?:final\s+)?answer\s*(?:is\s*:?|:)\s*", re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_LABEL_RE = re.compile(r"\b(true|false|uncertain)\b", re.IGNORECASE)
_PUNCT_STRIP = " \t\r\n.,:;!?\"'()[]{}*`"


def normalize_folio_label(s: str | None) -> str | None:
    """Normalize a FOLIO verdict to ``'true'``, ``'false'``, or ``'uncertain'``; else ``None``."""
    if s is None:
        return None
    t = _TAG_RE.sub(" ", str(s)).strip(_PUNCT_STRIP).lower()
    if t in FOLIO_LABELS:
        return t
    return None


def extract_folio_answer(text: str | None) -> str | None:
    """Extract ``'true'``, ``'false'``, or ``'uncertain'`` from model output."""
    if text is None or not text.strip():
        return None
    cleaned = _TAG_RE.sub(" ", text).replace("*", "").replace("`", "")

    prefix_matches = list(_PREFIX_RE.finditer(cleaned))
    if prefix_matches:
        rest = cleaned[prefix_matches[-1].end() :]
        for raw_line in rest.splitlines():
            line = raw_line.strip()
            if line:
                m = _LABEL_RE.search(line)
                return m.group(1).lower() if m else None
        return None

    for inner in reversed(_BOXED_RE.findall(cleaned)):
        norm = normalize_folio_label(inner)
        if norm is not None:
            return norm

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return None
    hits = _LABEL_RE.findall(lines[-1])
    return hits[-1].lower() if hits else None


@register_benchmark("folio")
class FOLIO(Benchmark):
    name = "folio"
    answer_format = FOLIO_ANSWER_FORMAT

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        train: int = 0,
        val: int = 0,
        seed: int = 0,
    ) -> None:
        if source not in ("local", "hf"):
            raise ValueError(f"unknown folio source '{source}' (expected 'local' or 'hf')")
        self.source = source
        self.data_dir: Path | None = _resolve(data_dir) if data_dir is not None else None
        if source == "local" and self.data_dir is None:
            self.data_dir = _resolve("data/sample/folio")
        self.train = int(train)
        self.val = int(val)
        if self.train < 0 or self.val < 0:
            raise ValueError(f"folio: train and val must be >= 0 (got train={train}, val={val})")
        self.seed = int(seed)
        self.splits = configured_splits(self.train, self.val)
        self._raw: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ data
    def _records(self, which: str) -> list[dict]:
        if which not in self._raw:
            if self.source == "hf":
                import datasets  # lazy heavy import

                hf_split = "validation" if which == "test" else "train"
                ds = datasets.load_dataset("tasksource/folio", split=hf_split)
                self._raw[which] = [
                    {
                        "premises": r["premises"],
                        "conclusion": r["conclusion"],
                        "label": r["label"],
                        "story_id": r["story_id"],
                        "example_id": r["example_id"],
                    }
                    for r in ds
                ]
            else:
                assert self.data_dir is not None
                self._raw[which] = read_jsonl(self.data_dir / f"{which}.jsonl")
        return self._raw[which]

    @staticmethod
    def _example(prefix: str, idx: int, rec: dict) -> Example:
        question = (
            f"Premises:\n{str(rec['premises']).strip()}\n\n"
            f"Conclusion:\n{str(rec['conclusion']).strip()}\n\n"
            "Based solely on the premises above, is the conclusion True, False, or Uncertain?"
        )
        answer = normalize_folio_label(str(rec["label"])) or ""
        meta = {k: rec[k] for k in ("story_id", "example_id") if k in rec}
        return Example(id=f"{prefix}-{idx}", question=question, answer=answer, meta=meta)

    def load(self, split: str) -> list[Example]:
        if split in ("train", "val"):
            records = self._records("train")
            order = list(range(len(records)))
            random.Random(self.seed).shuffle(order)
            chosen = order[: self.train] if split == "train" else order[self.train : self.train + self.val]
            return [self._example("train", i, records[i]) for i in chosen]
        if split == "test":
            return [self._example("test", i, r) for i, r in enumerate(self._records("test"))]
        raise ValueError(f"unknown folio split '{split}' (expected train, val or test)")

    # ------------------------------------------------------------------ scoring
    def extract_answer(self, text: str) -> str | None:
        return extract_folio_answer(text)

    def score(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        return normalize_folio_label(pred) == normalize_folio_label(gold)
