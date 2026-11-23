"""MuSR (Testing Soft Reasoning in Natural Language Narratives) benchmark."""
from __future__ import annotations

import ast
from pathlib import Path

from .base import Benchmark, Example, _resolve, read_jsonl, register_benchmark
from .mcq import MCQ_LETTERS, extract_mcq_answer, format_mcq_options, mcq_answer_format, score_mcq
from .pooled import configured_splits, split_indices

MUSR_TASKS: tuple[str, ...] = (
    "murder_mysteries",
    "object_placements",
    "team_allocation",
)


def _parse_choices(raw: object) -> list[str]:
    if isinstance(raw, str):
        parsed = ast.literal_eval(raw)
        return [str(x) for x in parsed]
    return [str(x) for x in raw]  # type: ignore[union-attr]


@register_benchmark("musr")
class MuSR(Benchmark):
    name = "musr"
    answer_format = mcq_answer_format(5)

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        tasks: str | list[str] = "all",
        heldout: list[str] | None = None,
        train: int = 0,
        val: int = 0,
        seed: int = 0,
    ) -> None:
        if source not in ("local", "hf"):
            raise ValueError(f"unknown musr source '{source}' (expected 'local' or 'hf')")
        self.source = source
        self.data_dir: Path | None = _resolve(data_dir) if data_dir is not None else None
        if source == "local" and self.data_dir is None:
            self.data_dir = _resolve("data/sample/musr")
        self.train = int(train)
        self.val = int(val)
        if self.train < 0 or self.val < 0:
            raise ValueError(f"musr: train and val must be >= 0 (got train={train}, val={val})")
        self.seed = int(seed)

        available = self._available_tasks()
        if tasks == "all":
            selected = list(available)
        else:
            selected = self._check([tasks] if isinstance(tasks, str) else list(tasks), available)
        self.heldout: list[str] = self._check(list(heldout or []), available)
        held = set(self.heldout)
        self.tasks: list[str] = [t for t in selected if t not in held]
        self.splits = configured_splits(self.train, self.val, ("heldout",) if self.heldout else ())
        self._raw: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ tasks & data
    def _available_tasks(self) -> list[str]:
        if self.source == "hf":
            return list(MUSR_TASKS)
        assert self.data_dir is not None
        if not self.data_dir.is_dir():
            return []
        return sorted(p.stem for p in self.data_dir.glob("*.jsonl"))

    @staticmethod
    def _check(names: list[str], available: list[str]) -> list[str]:
        known = set(available) | set(MUSR_TASKS)
        out: list[str] = []
        for name in names:
            if name not in known:
                raise ValueError(f"unknown MuSR task '{name}'; valid tasks: {', '.join(sorted(known))}")
            if name not in out:
                out.append(name)
        return out

    def _records(self, task: str) -> list[dict]:
        if task not in self._raw:
            if self.source == "hf":
                import datasets  # lazy heavy import

                ds = datasets.load_dataset("TAUR-Lab/MuSR", split=task)
                self._raw[task] = [
                    {
                        "narrative": r["narrative"],
                        "question": r["question"],
                        "choices": r["choices"],
                        "answer_index": r["answer_index"],
                    }
                    for r in ds
                ]
            else:
                assert self.data_dir is not None
                self._raw[task] = read_jsonl(self.data_dir / f"{task}.jsonl")
        return self._raw[task]

    @staticmethod
    def _example(task: str, idx: int, rec: dict) -> Example:
        choices = _parse_choices(rec["choices"])
        question = (
            f"{str(rec['narrative']).strip()}\n\n"
            f"Question: {str(rec['question']).strip()}\n"
            f"Options:\n{format_mcq_options(choices)}"
        )
        answer = chr(ord("a") + int(rec["answer_index"]))
        return Example(
            id=f"{task}-{idx}",
            question=question,
            answer=answer,
            meta={"task": task, "num_choices": len(choices)},
        )

    def load(self, split: str) -> list[Example]:
        if split == "heldout":
            out: list[Example] = []
            for task in self.heldout:
                recs = self._records(task)
                out.extend(self._example(task, i, recs[i]) for i in range(len(recs)))
            return out
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown musr split '{split}' (expected train, val, test or heldout)")
        out = []
        for task in self.tasks:
            recs = self._records(task)
            chosen = split_indices(len(recs), self.train, self.val, self.seed)[split]
            out.extend(self._example(task, i, recs[i]) for i in chosen)
        return out

    # ------------------------------------------------------------------ scoring
    def answer_format_for(self, example: Example) -> str:
        return mcq_answer_format(int(example.meta.get("num_choices", 5)))

    def extract_answer(self, text: str) -> str | None:
        return extract_mcq_answer(text, "abcde")

    def extract_answer_for(self, text: str, example: Example) -> str | None:
        n = max(2, min(len(MCQ_LETTERS), int(example.meta.get("num_choices", 5))))
        return extract_mcq_answer(text, MCQ_LETTERS[:n].lower())

    def score(self, pred: str | None, gold: str) -> bool:
        return score_mcq(pred, gold)
