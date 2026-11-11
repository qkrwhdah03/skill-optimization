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

_MULTIPLE_CHOICE = "multiple_choice"
_BINARY = "binary"
_NUMERIC = "numeric"
_DYCK = "dyck"
_WORD_SORTING = "word_sorting"
_FREE_FORM = "free_form"

BBH_TASK_KINDS: dict[str, str] = {
    "boolean_expressions": _BINARY,
    "causal_judgement": _BINARY,
    "date_understanding": _MULTIPLE_CHOICE,
    "disambiguation_qa": _MULTIPLE_CHOICE,
    "dyck_languages": _DYCK,
    "formal_fallacies": _BINARY,
    "geometric_shapes": _MULTIPLE_CHOICE,
    "hyperbaton": _MULTIPLE_CHOICE,
    "logical_deduction_five_objects": _MULTIPLE_CHOICE,
    "logical_deduction_seven_objects": _MULTIPLE_CHOICE,
    "logical_deduction_three_objects": _MULTIPLE_CHOICE,
    "movie_recommendation": _MULTIPLE_CHOICE,
    "multistep_arithmetic_two": _NUMERIC,
    "navigate": _BINARY,
    "object_counting": _NUMERIC,
    "penguins_in_a_table": _MULTIPLE_CHOICE,
    "reasoning_about_colored_objects": _MULTIPLE_CHOICE,
    "ruin_names": _MULTIPLE_CHOICE,
    "salient_translation_error_detection": _MULTIPLE_CHOICE,
    "snarks": _MULTIPLE_CHOICE,
    "sports_understanding": _BINARY,
    "temporal_sequences": _MULTIPLE_CHOICE,
    "tracking_shuffled_objects_five_objects": _MULTIPLE_CHOICE,
    "tracking_shuffled_objects_seven_objects": _MULTIPLE_CHOICE,
    "tracking_shuffled_objects_three_objects": _MULTIPLE_CHOICE,
    "web_of_lies": _BINARY,
    "word_sorting": _WORD_SORTING,
}

# exact gold word forms of each binary task
BBH_BINARY_CHOICES: dict[str, tuple[str, str]] = {
    "boolean_expressions": ("True", "False"),
    "causal_judgement": ("Yes", "No"),
    "navigate": ("Yes", "No"),
    "web_of_lies": ("Yes", "No"),
    "sports_understanding": ("yes", "no"),
    "formal_fallacies": ("valid", "invalid"),
}

_FORMATS: dict[str, str] = {
    _MULTIPLE_CHOICE: 'End your response with "So the answer is (X)." where X is the letter of the correct option.',
    _NUMERIC: 'End your response with "So the answer is N." where N is the final number.',
    _DYCK: (
        'End your response with "So the answer is" followed by only the closing brackets needed to complete '
        "the sequence, separated by spaces."
    ),
    _WORD_SORTING: 'End your response with "So the answer is" followed by the sorted words separated by single spaces.',
    _FREE_FORM: 'End your response with "So the answer is" followed by your final answer.',
}

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_TAG_RE = re.compile(r"</?[A-Za-z][^<>]*>")
_PAREN_LETTER_RE = re.compile(r"\(([A-Za-z])\)")
_LEAD_LETTER_RE = re.compile(r"^([A-Z])(?![A-Za-z0-9])")
_WORD_RE = re.compile(r"[A-Za-z]+")
_DIGIT_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_DYCK_CHARS = set("()[]{}<>")
_STRIP_CHARS = "_\"'\u201c\u201d\u2018\u2019 \t\r\n\f\v"


def _task_kind(task: str | None) -> str:
    if task is None:
        return _FREE_FORM
    return BBH_TASK_KINDS.get(task, _FREE_FORM)


def bbh_answer_format(task: str | None) -> str:
    """Answer-format line for one BBH task (generic free-form line for unknown / None tasks)."""
    kind = _task_kind(task)
    if kind == _BINARY:
        a, b = BBH_BINARY_CHOICES[task]  # type: ignore[index]
        return f'End your response with "So the answer is {a}." or "So the answer is {b}."'
    return _FORMATS[kind]


def _segment(text: str) -> str:
    matches = list(_ANSWER_IS_RE.finditer(text))
    if matches:
        rest = text[matches[-1].end():].lstrip()
        if rest.startswith(":"):
            rest = rest[1:]
        for line in rest.splitlines():
            if line.strip():
                return line.strip()
        return ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def _clean_free_form(segment: str, commas_to_spaces: bool) -> str:
    t = segment.replace("*", "").replace("`", "")
    prev = None
    while t != prev:
        prev = t
        t = t.strip(_STRIP_CHARS)
        while t.endswith("."):
            t = t[:-1]
    if commas_to_spaces:
        t = t.replace(",", " ")
    return re.sub(r"\s+", " ", t).strip()


def _extract_choice_letter(segment: str) -> str | None:
    m = _PAREN_LETTER_RE.search(segment)
    if m:
        return m.group(1).lower()
    t = segment.replace("*", "").replace("`", "").strip(_STRIP_CHARS)
    m = _LEAD_LETTER_RE.match(t)
    if m:
        return m.group(1).lower()
    return None


def extract_bbh_answer(text: str | None, task: str | None = None) -> str | None:
    """Extract a normalized BBH answer from ``text`` using the answer kind of ``task`` (free-form if unknown)."""
    if text is None or not text.strip():
        return None
    segment = _segment(text)
    segment = _BOXED_RE.sub(lambda m: m.group(1), segment)
    segment = _TAG_RE.sub(" ", segment)
    kind = _task_kind(task)

    if kind == _MULTIPLE_CHOICE:
        value: str | None = _extract_choice_letter(segment)
    elif kind == _DYCK:
        value = " ".join(ch for ch in segment if ch in _DYCK_CHARS)
    else:
        cleaned = _clean_free_form(segment, commas_to_spaces=(kind == _WORD_SORTING))
        value = cleaned
        if kind == _BINARY:
            m = _WORD_RE.search(cleaned)
            if m:
                word = m.group(0).lower()
                for choice in BBH_BINARY_CHOICES.get(task or "", ()):
                    if word == choice.lower():
                        value = choice.lower()
                        break
        elif kind == _NUMERIC:
            m = _NUMBER_RE.search(_DIGIT_COMMA_RE.sub("", cleaned))
            if m:
                value = m.group(0)

    if value is None:
        return None
    norm = normalize_answer(value)
    return norm if norm else None


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
    answer_format = _FORMATS[_FREE_FORM]
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
    def answer_format_for(self, example: Example) -> str:
        return bbh_answer_format(example.meta.get("task"))

    def extract_answer(self, text: str) -> str | None:
        return extract_bbh_answer(text, None)

    def extract_answer_for(self, text: str, example: Example) -> str | None:
        return extract_bbh_answer(text, example.meta.get("task"))

    def score(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        return normalize_answer(pred) == normalize_answer(gold)
