"""Base for benchmarks that ship a single pool of items (no official train split)."""
from __future__ import annotations

import random
from abc import abstractmethod
from pathlib import Path
from typing import ClassVar

from .base import Benchmark, Example, _resolve


def split_indices(n: int, train: int, val: int, seed: int) -> dict[str, list[int]]:
    """Deterministic train/val/test split of ``range(n)``; each split in ascending index order."""
    order = list(range(n))
    random.Random(seed).shuffle(order)
    return {
        "train": sorted(order[:train]),
        "val": sorted(order[train : train + val]),
        "test": sorted(order[train + val :]),
    }


def configured_splits(train: int, val: int, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    """Splits that can be non-empty: train/val only when their size > 0, test always."""
    out: list[str] = []
    if train > 0:
        out.append("train")
    if val > 0:
        out.append("val")
    out.append("test")
    out.extend(extra)
    return tuple(out)


class PooledBenchmark(Benchmark):
    """Single-pool benchmark: train/val are drawn from the pool with the seed, test is the rest.

    Subclasses provide ``_load_records`` (HF or local rows) and ``_example`` (one row -> Example).
    ``splits`` is set per instance: "train"/"val" only when configured > 0, so test-only sets never
    enter the search loop.
    """

    default_data_dir: ClassVar[str]

    def __init__(
        self,
        source: str = "local",
        data_dir: str | None = None,
        train: int = 0,
        val: int = 0,
        seed: int = 0,
    ) -> None:
        if source not in ("local", "hf"):
            raise ValueError(f"unknown {self.name} source '{source}' (expected 'local' or 'hf')")
        self.source = source
        self.data_dir: Path | None = _resolve(data_dir) if data_dir is not None else None
        if source == "local" and self.data_dir is None:
            self.data_dir = _resolve(self.default_data_dir)
        self.train = int(train)
        self.val = int(val)
        if self.train < 0 or self.val < 0:
            raise ValueError(f"{self.name}: train and val must be >= 0 (got train={train}, val={val})")
        self.seed = int(seed)
        self.splits = configured_splits(self.train, self.val)
        self._raw: list[dict] | None = None

    # ------------------------------------------------------------------ data
    @abstractmethod
    def _load_records(self) -> list[dict]:
        """All pool rows in source order."""

    @abstractmethod
    def _example(self, idx: int, rec: dict) -> Example:
        """Build the Example for pool row ``idx`` (id should be ``f"{self.name}-{idx}"``)."""

    def _records(self) -> list[dict]:
        if self._raw is None:
            self._raw = self._load_records()
        return self._raw

    def load(self, split: str) -> list[Example]:
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown {self.name} split '{split}' (expected train, val or test)")
        records = self._records()
        idx = split_indices(len(records), self.train, self.val, self.seed)[split]
        return [self._example(i, records[i]) for i in idx]
