"""Reproducible run directories and seeding."""
from __future__ import annotations

import json
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import resolve_path
from .evaluator import EvalResult, SkillEvaluation
from .llm import UsageSnapshot
from .skills import Skill


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # optional

        np.random.seed(seed)
    except Exception:  # not installed or broken install
        pass
    try:
        import torch  # optional

        torch.manual_seed(seed)
    except Exception:  # not installed or broken install
        pass


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


class RunDir:
    def __init__(self, root: str | Path, name: str, config: dict) -> None:
        base = resolve_path(root)
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = base / f"{stamp}_{name}"
        n = 1
        while candidate.exists():
            n += 1
            candidate = base / f"{stamp}_{name}_{n}"
        candidate.mkdir(parents=True)
        self.path: Path = candidate
        with open(self.path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        _write_json(
            self.path / "meta.json",
            {
                "seed": config.get("seed"),
                "created_at": datetime.now().isoformat(),
                "argv": list(sys.argv),
                "python": platform.python_version(),
            },
        )

    def save_skill(self, skill: Skill) -> Path:
        return _write_json(self.path / "skills" / f"{skill.id}.json", skill.to_dict())

    def save_eval(self, result: EvalResult, split: str) -> Path:
        """Write ``results/<split>/<benchmark>/<skill_id>.jsonl`` + ``.summary.json``; returns the summary path."""
        d = self.path / "results" / split / result.benchmark
        d.mkdir(parents=True, exist_ok=True)
        with open(d / f"{result.skill_id}.jsonl", "w", encoding="utf-8") as f:
            for item in result.items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
        return _write_json(d / f"{result.skill_id}.summary.json", result.summary())

    def save_evaluation(self, evaluation: SkillEvaluation) -> list[Path]:
        return [self.save_eval(r, evaluation.split) for r in evaluation.results.values()]

    def save_usage(self, role: str, snapshot: UsageSnapshot) -> Path:
        return _write_json(self.path / "usage" / f"{role}.json", snapshot.to_dict())

    def save_json(self, name: str, obj: Any) -> Path:
        return _write_json(self.path / name, obj)

    def round_dir(self, index: int) -> Path:
        """``<run>/rounds/<index:03d>`` (round 0 = initial prompts)."""
        return self.path / "rounds" / f"{index:03d}"

    def save_round(self, index: int, evaluations: list[SkillEvaluation]) -> Path:
        """Write ``skills/``, ``results/<benchmark>/`` and ``summary.json`` of one round; returns the round dir."""
        d = self.round_dir(index)
        d.mkdir(parents=True, exist_ok=True)
        for ev in evaluations:
            _write_json(d / "skills" / f"{ev.skill_id}.json", ev.skill.to_dict())
            for result in ev.results.values():
                rd = d / "results" / result.benchmark
                rd.mkdir(parents=True, exist_ok=True)
                with open(rd / f"{result.skill_id}.jsonl", "w", encoding="utf-8") as f:
                    for item in result.items:
                        f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
                _write_json(rd / f"{result.skill_id}.summary.json", result.summary())
        _write_json(d / "summary.json", [ev.summary() for ev in evaluations])
        return d
