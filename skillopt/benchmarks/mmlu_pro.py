"""MMLU-Pro (10-option cross-discipline expert reasoning) benchmark."""
from __future__ import annotations

from pathlib import Path

from .base import Benchmark, Example, _resolve, read_jsonl, register_benchmark
from .mcq import MCQ_LETTERS, extract_mcq_answer, format_mcq_options, mcq_answer_format, score_mcq
from .pooled import configured_splits, split_indices

MMLU_PRO_CATEGORIES: tuple[str, ...] = (
    "biology",
    "business",
    "chemistry",
    "computer_science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "other",
    "philosophy",
    "physics",
    "psychology",
)

MMLU_PRO_GROUPS: dict[str, list[str]] = {
    "non_math": [c for c in MMLU_PRO_CATEGORIES if c != "math"],
    "stem": ["biology", "chemistry", "computer_science", "engineering", "physics"],
    "humanities_social": ["business", "economics", "health", "history", "law", "philosophy", "psychology"],
}


def _normalize_category(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


@register_benchmark("mmlu_pro")
class MMLUPro(Benchmark):
    name = "mmlu_pro"
    answer_format = mcq_answer_format(10)

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        categories: str | list[str] = "non_math",
        heldout: list[str] | None = None,
        train: int = 0,
        val: int = 0,
        seed: int = 0,
    ) -> None:
        if source not in ("local", "hf"):
            raise ValueError(f"unknown mmlu_pro source '{source}' (expected 'local' or 'hf')")
        self.source = source
        self.data_dir: Path | None = _resolve(data_dir) if data_dir is not None else None
        if source == "local" and self.data_dir is None:
            self.data_dir = _resolve("data/sample/mmlu_pro")
        self.train = int(train)
        self.val = int(val)
        if self.train < 0 or self.val < 0:
            raise ValueError(f"mmlu_pro: train and val must be >= 0 (got train={train}, val={val})")
        self.seed = int(seed)

        available = self._available_categories()
        if categories == "all":
            selected = list(available)
        else:
            raw_cats = [categories] if isinstance(categories, str) else list(categories)
            selected = self._expand(raw_cats, available)
        self.heldout: list[str] = self._expand(list(heldout or []), available)
        held = set(self.heldout)
        self.categories: list[str] = [c for c in selected if c not in held]
        self.splits = configured_splits(self.train, self.val, ("heldout",) if self.heldout else ())
        self._raw: dict[str, list[dict]] | None = None

    # ------------------------------------------------------------------ categories & data
    def _available_categories(self) -> list[str]:
        if self.source == "hf":
            return list(MMLU_PRO_CATEGORIES)
        assert self.data_dir is not None
        if not self.data_dir.is_dir():
            return []
        return sorted(p.stem for p in self.data_dir.glob("*.jsonl"))

    def _expand(self, names: list[str], available: list[str]) -> list[str]:
        known = set(available) | set(MMLU_PRO_CATEGORIES)
        avail_set = set(available)
        out: list[str] = []
        for raw_name in names:
            norm = _normalize_category(raw_name)
            if norm in MMLU_PRO_GROUPS:
                members = list(MMLU_PRO_GROUPS[norm])
                if self.source == "local" and avail_set:
                    members = [m for m in members if m in avail_set]
            elif norm in known:
                members = [norm]
            else:
                raise ValueError(
                    f"unknown MMLU-Pro category or group '{raw_name}'; "
                    f"groups: {sorted(MMLU_PRO_GROUPS)}"
                )
            for m in members:
                if m not in out:
                    out.append(m)
        return out

    def _load_all(self) -> dict[str, list[dict]]:
        if self._raw is None:
            raw: dict[str, list[dict]] = {}
            if self.source == "hf":
                import datasets  # lazy heavy import

                ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split="test")
                for r in ds:
                    cat = _normalize_category(r["category"])
                    raw.setdefault(cat, []).append(
                        {
                            "question": r["question"],
                            "options": list(r["options"]),
                            "answer": r["answer"],
                            "answer_index": r["answer_index"],
                            "category": cat,
                        }
                    )
            else:
                assert self.data_dir is not None
                for cat in dict.fromkeys(self.categories + self.heldout):
                    raw[cat] = read_jsonl(self.data_dir / f"{cat}.jsonl")
            self._raw = raw
        return self._raw

    def _records(self, cat: str) -> list[dict]:
        return self._load_all().get(cat, [])

    @staticmethod
    def _example(cat: str, idx: int, rec: dict) -> Example:
        options = [str(o) for o in rec["options"]]
        question = f"{str(rec['question']).strip()}\n\nOptions:\n{format_mcq_options(options)}"
        raw_ans = rec.get("answer")
        if raw_ans is not None and str(raw_ans).strip():
            answer = str(raw_ans).strip().lower()
        else:
            answer = chr(ord("a") + int(rec["answer_index"]))
        return Example(
            id=f"{cat}-{idx}",
            question=question,
            answer=answer,
            meta={"category": cat, "num_choices": len(options)},
        )

    def load(self, split: str) -> list[Example]:
        if split == "heldout":
            out: list[Example] = []
            for cat in self.heldout:
                recs = self._records(cat)
                out.extend(self._example(cat, i, recs[i]) for i in range(len(recs)))
            return out
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown mmlu_pro split '{split}' (expected train, val, test or heldout)")
        out = []
        for cat in self.categories:
            recs = self._records(cat)
            chosen = split_indices(len(recs), self.train, self.val, self.seed)[split]
            out.extend(self._example(cat, i, recs[i]) for i in chosen)
        return out

    # ------------------------------------------------------------------ scoring
    def answer_format_for(self, example: Example) -> str:
        return mcq_answer_format(int(example.meta.get("num_choices", 10)))

    def extract_answer(self, text: str) -> str | None:
        return extract_mcq_answer(text, "abcdefghij")

    def extract_answer_for(self, text: str, example: Example) -> str | None:
        n = max(2, min(len(MCQ_LETTERS), int(example.meta.get("num_choices", 10))))
        return extract_mcq_answer(text, MCQ_LETTERS[:n].lower())

    def score(self, pred: str | None, gold: str) -> bool:
        return score_mcq(pred, gold)
