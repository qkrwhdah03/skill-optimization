"""BIG-Bench Extra Hard (BBEH): 23 hard reasoning tasks, scored with the official fuzzy matcher."""
from __future__ import annotations

from pathlib import Path

from .base import Benchmark, Example, _resolve, read_jsonl, register_benchmark
from .pooled import configured_splits, split_indices

BBEH_TASKS: tuple[str, ...] = (
    "boardgame_qa",
    "boolean_expressions",
    "buggy_tables",
    "causal_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "geometric_shapes",
    "hyperbaton",
    "linguini",
    "movie_recommendation",
    "multistep_arithmetic",
    "nycc",
    "object_counting",
    "object_properties",
    "sarc_triples",
    "shuffled_objects",
    "spatial_reasoning",
    "sportqa",
    "temporal_sequence",
    "time_arithmetic",
    "web_of_lies",
    "word_sorting",
    "zebra_puzzles",
)

BBEH_ANSWER_FORMAT: str = (
    'End your response with "The answer is:" followed by only your final answer in the form '
    "the question asks for (for multiple choice, only the option letter), with no formatting."
)

# Scoring below is ported from google-deepmind/bbeh bbeh/evaluate.py (Apache-2.0).
_ANSWER_PREFIXES = ("The answer is:", "The final answer is ", "The final answer is: ", "The answer is ")


def _strip_latex(response: str) -> str:
    if response.startswith("$") and response.endswith("$"):
        response = response[1:-1]
    if "boxed{" in response and response.endswith("}"):
        response = response[0:-1].split("boxed{")[1]
    if "text{" in response and response.endswith("}"):
        response = response[0:-1].split("text{")[1]
    if "texttt{" in response and response.endswith("}"):
        response = response[0:-1].split("texttt{")[1]
    return response


def _official_extract(sample: str) -> str:
    answer = sample
    for prefix in _ANSWER_PREFIXES:
        # deliberate difference from the official code: case-insensitive prefix match
        pos = answer.lower().rfind(prefix.lower())
        if pos != -1:
            answer = answer[pos + len(prefix):].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    return _strip_latex(answer)


def extract_bbeh_answer(text: str | None) -> str | None:
    """Official ``preprocess_sample`` (with case-insensitive answer prefixes); None if nothing is left."""
    if text is None or not text.strip():
        return None
    prediction = _official_extract(text.strip()).lower()
    prediction = prediction.replace(", ", ",").replace("**", "")
    prediction = prediction.split("\n")[0]
    prediction = prediction[0:-1] if prediction.endswith(".") else prediction
    prediction = prediction.strip()
    return prediction or None


def normalize_bbeh_reference(s: str) -> str:
    """Official ``preprocess_reference``."""
    reference = str(s).strip().lower()
    return reference.replace(", ", ",")


def bbeh_fuzzy_match(prediction: str, reference: str) -> bool:
    """Official ``fuzzy_match``."""
    if prediction == reference:
        return True
    # (a) vs a
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    # numbers
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    # quote issues
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    # bracket issues
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    # question mark issues
    if prediction.endswith("?") and prediction[:-1] == reference:
        return True
    return False


@register_benchmark("bbeh")
class BBEH(Benchmark):
    name = "bbeh"
    answer_format = BBEH_ANSWER_FORMAT

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        tasks: str | list[str] = "all",
        heldout: list[str] | None = None,
        mini: bool = False,
        train: int = 0,
        val: int = 0,
        seed: int = 0,
    ) -> None:
        if source not in ("local", "hf"):
            raise ValueError(f"unknown bbeh source '{source}' (expected 'local' or 'hf')")
        self.source = source
        self.data_dir: Path | None = _resolve(data_dir) if data_dir is not None else None
        if source == "local" and self.data_dir is None:
            self.data_dir = _resolve("data/sample/bbeh")
        self.mini = bool(mini)
        self.train = int(train)
        self.val = int(val)
        if self.train < 0 or self.val < 0:
            raise ValueError(f"bbeh: train and val must be >= 0 (got train={train}, val={val})")
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
        self._raw: dict[str, list[dict]] | None = None

    # ------------------------------------------------------------------ tasks
    def _available_tasks(self) -> list[str]:
        if self.source == "hf":
            return list(BBEH_TASKS)
        assert self.data_dir is not None
        if not self.data_dir.is_dir():
            return []
        return sorted(p.stem for p in self.data_dir.glob("*.jsonl"))

    @staticmethod
    def _check(names: list[str], available: list[str]) -> list[str]:
        known = set(available) | set(BBEH_TASKS)
        out: list[str] = []
        for name in names:
            if name not in known:
                raise ValueError(f"unknown BBEH task '{name}'; valid tasks: {', '.join(sorted(known))}")
            if name not in out:
                out.append(name)
        return out

    def _load_all(self) -> dict[str, list[dict]]:
        if self._raw is None:
            raw: dict[str, list[dict]] = {}
            if self.source == "hf":
                import datasets  # lazy heavy import

                ds = datasets.load_dataset("BBEH/bbeh", split="train")
                for r in ds:
                    task = str(r["task"]).replace(" ", "_")
                    try:
                        mini = r["mini"]
                    except (KeyError, IndexError, TypeError):
                        mini = 0
                    raw.setdefault(task, []).append({"input": r["input"], "target": r["target"], "mini": mini})
            else:
                assert self.data_dir is not None
                for task in dict.fromkeys(self.tasks + self.heldout):
                    raw[task] = read_jsonl(self.data_dir / f"{task}.jsonl")
            self._raw = raw
        return self._raw

    def _records(self, task: str) -> list[dict]:
        return self._load_all().get(task, [])

    def _pool(self, task: str) -> list[int]:
        """Row indices (source order) of ``task`` after the optional mini filter."""
        recs = self._records(task)
        if self.mini:
            return [i for i, r in enumerate(recs) if r.get("mini")]
        return list(range(len(recs)))

    def _examples(self, task: str, indices: list[int]) -> list[Example]:
        recs = self._records(task)
        return [
            Example(
                id=f"{task}-{i}",
                question=str(recs[i]["input"]),
                answer=normalize_bbeh_reference(str(recs[i]["target"])),
                meta={"task": task},
            )
            for i in indices
        ]

    def load(self, split: str) -> list[Example]:
        if split == "heldout":
            out: list[Example] = []
            for task in self.heldout:
                out.extend(self._examples(task, self._pool(task)))
            return out
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown bbeh split '{split}' (expected train, val, test or heldout)")
        out = []
        for task in self.tasks:
            pool = self._pool(task)
            chosen = split_indices(len(pool), self.train, self.val, self.seed)[split]
            out.extend(self._examples(task, [pool[j] for j in chosen]))
        return out

    # ------------------------------------------------------------------ scoring
    def extract_answer(self, text: str) -> str | None:
        return extract_bbeh_answer(text)

    def extract_answer_for(self, text: str, example: Example) -> str | None:
        return extract_bbeh_answer(text)

    def score(self, pred: str | None, gold: str) -> bool:
        if pred is None:
            return False
        return bbeh_fuzzy_match(pred, gold)
