"""Benchmarks: data splits, prompt building, answer extraction and scoring."""
from .base import Benchmark, Example, build_benchmark, get_benchmark_class, register_benchmark
from .gsm8k import GSM8K
from .bbh import BBH, BBH_TASK_GROUPS, BBH_TASKS
from .pooled import PooledBenchmark
from .math_sets import AIME2024, AIME2025, AMC2023, MATH500, MathBenchmark
from .bbeh import BBEH, BBEH_TASKS
from .musr import MuSR, MUSR_TASKS
from .folio import FOLIO, FOLIO_LABELS
from .lsat_ar import LSATAR
from .strategyqa import StrategyQA
from .mmlu_pro import MMLUPro, MMLU_PRO_CATEGORIES, MMLU_PRO_GROUPS

__all__ = [
    "Benchmark",
    "Example",
    "build_benchmark",
    "get_benchmark_class",
    "register_benchmark",
    "GSM8K",
    "BBH",
    "BBH_TASKS",
    "BBH_TASK_GROUPS",
    "PooledBenchmark",
    "MathBenchmark",
    "MATH500",
    "AIME2024",
    "AIME2025",
    "AMC2023",
    "BBEH",
    "BBEH_TASKS",
    "MuSR",
    "MUSR_TASKS",
    "FOLIO",
    "FOLIO_LABELS",
    "LSATAR",
    "StrategyQA",
    "MMLUPro",
    "MMLU_PRO_CATEGORIES",
    "MMLU_PRO_GROUPS",
]
