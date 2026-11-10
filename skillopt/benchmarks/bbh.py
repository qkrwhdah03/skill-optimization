"""BIG-Bench Hard (BBH) benchmark with held-out tasks."""
from __future__ import annotations

import random
import re
from pathlib import Path

from .base import Benchmark, Example, _resolve, read_jsonl, register_benchmark

BBH_TASKS: tuple[str, ...] = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)

# Split used in the BBH paper (Suzgun et al., 2022).
BBH_TASK_GROUPS: dict[str, list[str]] = {
    "algorithmic": [
        "boolean_expressions",
        "dyck_languages",
        "geometric_shapes",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "multistep_arithmetic_two",
        "navigate",
        "object_counting",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects",
        "web_of_lies",
        "word_sorting",
    ],
    "nlu": [
        "causal_judgement",
        "date_understanding",
        "disambiguation_qa",
        "formal_fallacies",
        "hyperbaton",
        "movie_recommendation",
        "penguins_in_a_table",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "snarks",
        "sports_understanding",
    ],
}

_ANSWER_IS_RE = re.compile(r"answer is", re.IGNORECASE)
_OPTION_RE = re.compile(r"^\(([a-z])\)$")
_WRAP_CHARS = "\"'`*"


def normalize_answer(s: str | None) -> str | None:
    if s is None:
        return None
    t = str(s).strip().lower()
    prev = None
    while t != prev:
        prev = t
        t = t.strip()
        while t.endswith("."):
            t = t[:-1].rstrip()
        t = t.strip(_WRAP_CHARS).strip()
    t = re.sub(r"\s+", " ", t)
    m = _OPTION_RE.match(t)
    if m:
        t = m.group(1)
    return t


@register_benchmark("bbh")
class BBH(Benchmark):
    name = "bbh"
    answer_format = 'End your response with "So the answer is <answer>."'
    splits = ("train", "val", "test", "heldout")

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        tasks: str | list[str] = "all",
        heldout: list[str] | None = None,
        task_groups: dict[str, list[str]] | None = None,
        train: int = 10,
        val: int = 10,
        seed: int = 0,
    ) -> None:
        if source not in ("local", "hf"):
            raise ValueError(f"unknown bbh source '{source}' (expected 'local' or 'hf')")
        self.source = source
        self.data_dir: Path | None = _resolve(data_dir) if data_dir is not None else None
        if source == "local" and self.data_dir is None:
            self.data_dir = _resolve("data/sample/bbh")
        self.task_groups = dict(task_groups) if task_groups is not None else dict(BBH_TASK_GROUPS)
        self.train = int(train)
        self.val = int(val)
        self.seed = int(seed)

        available = self._available_tasks()
        if tasks == "all":
            selected = list(available)
        else:
            selected = self._expand([tasks] if isinstance(tasks, str) else list(tasks), available)
        self.heldout: list[str] = self._expand(list(heldout or []), available)
        held = set(self.heldout)
        self.tasks: list[str] = [t for t in selected if t not in held]
        self._raw: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------ tasks
    def _available_tasks(self) -> list[str]:
        if self.source == "hf":
            return list(BBH_TASKS)
        assert self.data_dir is not None
        if not self.data_dir.is_dir():
            return []
        return sorted(p.stem for p in self.data_dir.glob("*.jsonl"))

    def _expand(self, names: list[str], available: list[str]) -> list[str]:
        known = set(available) | set(BBH_TASKS)
        out: list[str] = []
        for name in names:
            if name in self.task_groups:
                members = list(self.task_groups[name])
            elif name in known:
                members = [name]
            else:
                raise ValueError(
                    f"unknown BBH task or group '{name}'; groups: {sorted(self.task_groups)}"
                )
            for m in members:
                if m not in out:
                    out.append(m)
        return out

    def _records(self, task: str) -> list[dict]:
        if task not in self._raw:
            if self.source == "hf":
                import datasets  # lazy heavy import

                ds = datasets.load_dataset("lukaemon/bbh", task, split="test")
                self._raw[task] = [{"input": r["input"], "target": r["target"]} for r in ds]
            else:
                assert self.data_dir is not None
                self._raw[task] = read_jsonl(self.data_dir / f"{task}.jsonl")
        return self._raw[task]

    def _examples(self, task: str, indices: list[int]) -> list[Example]:
        recs = self._records(task)
        return [
            Example(
                id=f"{task}-{i}",
                question=recs[i]["input"],
                answer=normalize_answer(str(recs[i]["target"])) or "",
                meta={"task": task},
            )
            for i in indices
        ]

    def load(self, split: str) -> list[Example]:
        if split == "heldout":
            out: list[Example] = []
            for task in self.heldout:
                out.extend(self._examples(task, list(range(len(self._records(task))))))
            return out
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown bbh split '{split}' (expected train, val, test or heldout)")
        out = []
        for task in self.tasks:
            order = list(range(len(self._records(task))))
            random.Random(self.seed).shuffle(order)
            if split == "train":
                idx = order[: self.train]
            elif split == "val":
                idx = order[self.train : self.train + self.val]
            else:
                idx = order[self.train + self.val :]
            out.extend(self._examples(task, idx))
        return out

    # ------------------------------------------------------------------ scoring
    def extract_answer(self, text: str) -> str | None:
        if text is None or not text.strip():
            return None
        matches = list(_ANSWER_IS_RE.finditer(text))
        if matches:
            rest = text[matches[-1].end():]
            segment = rest.split("\n", 1)[0].strip().lstrip(":").strip()
        else:
            lines = [ln for ln in text.splitlines() if ln.strip()]
            segment = lines[-1] if lines else ""
        norm = normalize_answer(segment)
        return norm if norm else None

    def score(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        return normalize_answer(pred) == normalize_answer(gold)
