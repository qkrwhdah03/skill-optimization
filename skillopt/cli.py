"""Single command-line entry point (``run.py``): the optimize loop, or ``--eval-only``.

Wiring and saving only. The loop is ``initial_prompts()`` -> Evaluator -> ``update(results, history)`` -> ... on
split ``train`` until ``update`` returns ``[]`` or ``max_iterations`` updates; the algorithm is the configured
``optimizer:`` class, measurements come from ``skillopt.evaluator``. Every round is saved under ``rounds/<NNN>/``.
``--eval-only --split <split>`` evaluates skills on any other split (val, test, heldout, ...).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .baselines import BASELINE_NAMES, get_baseline
from .config import load_config, read_raw_config, resolve_path
from .evaluator import EvalResult, Evaluator, SkillEvaluation, build_benchmark_specs
from .llm import build_role_llms
from .llm.base import UsageSnapshot
from .logs import LEVELS, setup_logging
from .optimizer import build_optimizer, optimize
from .runlog import RunDir, set_seed
from .skills import Skill

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: str = "configs/config.yaml"

MOCK_OPTIMIZER_RESPONSES: tuple[str, ...] = (
    "<skill>Solve it step by step, keeping each step short.</skill>",
    "<skill>Write only the essential computations, then the answer.</skill>",
    "<skill>Compute directly and state the final answer.</skill>",
)

# Per-benchmark overrides for --mock: bundled sample data, tiny splits.
_MOCK_BENCHMARK_OVERRIDES: dict[str, list[tuple[str, str]]] = {
    "gsm8k": [("source", "local"), ("data_dir", "data/sample/gsm8k"), ("train", "4"), ("val", "2")],
    "bbh": [
        ("source", "local"),
        ("data_dir", "data/sample/bbh"),
        ("tasks", "all"),
        ("train", "2"),
        ("val", "2"),
    ],
}


def mock_overrides(config: dict) -> list[str]:
    """``--set`` strings that turn the real config into a no-GPU mock run (derived from the RAW config)."""
    out = [
        "llm.target.preset=mock",
        "llm.optimizer.preset=mock",
        f"llm.optimizer.responses={json.dumps(list(MOCK_OPTIMIZER_RESPONSES))}",
        "evaluator.batch_size=16",
        "max_iterations=2",
        "run.name=mock",
    ]
    benchmarks = (config or {}).get("benchmarks")
    if isinstance(benchmarks, dict):  # anything else: build_benchmark_specs raises the clear error later
        for bkey, entry in benchmarks.items():
            name = entry.get("name", bkey) if isinstance(entry, dict) else bkey
            for opt, value in _MOCK_BENCHMARK_OVERRIDES.get(name, []):
                out.append(f"benchmarks.{bkey}.{opt}={value}")
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Optimize reasoning skills for a fixed target LLM (or only evaluate skills with --eval-only).",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help=f"path to the YAML config (default {DEFAULT_CONFIG})")
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set max_iterations=5 (repeatable)",
    )
    p.add_argument("--mock", action="store_true", help="no GPU: mock model presets + bundled sample data")
    p.add_argument("--eval-only", action="store_true", help="only evaluate skills (no optimization)")
    p.add_argument("--skills", nargs="+", default=None, help="[--eval-only] baseline names or skill json paths")
    p.add_argument("--split", default=None, help="[--eval-only] split: train, val, test or heldout")
    p.add_argument("--limit", type=int, default=None, help="[--eval-only] max items per benchmark")
    log = p.add_mutually_exclusive_group()
    log.add_argument(
        "--log-level",
        dest="log_level",
        type=str.upper,
        choices=LEVELS,
        default="INFO",
        metavar="LEVEL",
        help=f"terminal log level: {', '.join(LEVELS)} (default INFO; logs go to stderr)",
    )
    log.add_argument("-v", "--verbose", dest="log_level", action="store_const", const="DEBUG", help="= --log-level DEBUG")
    log.add_argument("-q", "--quiet", dest="log_level", action="store_const", const="WARNING", help="= --log-level WARNING")
    return p


def _fmt(r: EvalResult) -> str:
    return (
        f"{r.skill_id:<24} {r.benchmark:<12} acc={r.accuracy:.3f}  out_tok={r.avg_output_tokens:8.1f}  "
        f"think_tok={r.avg_reasoning_tokens:8.1f}  total_tok={r.avg_total_tokens:8.1f}  n={r.n}"
    )


def _log_evaluation(ev: SkillEvaluation, prefix: str | None = None) -> None:
    for r in ev.results.values():
        logger.info(f"{ev.split if prefix is None else prefix} | {_fmt(r)}")


def _fmt_usage(u: UsageSnapshot) -> str:
    return (
        f"calls={u.calls} input={u.input_tokens} output={u.output_tokens} reasoning={u.reasoning_tokens} "
        f"total={u.total_tokens} time={u.latency:.1f}s"
    )


def _benchmark_configs(cfg: dict) -> dict:
    if cfg.get("benchmarks") is None:
        if "benchmark" in cfg:
            raise ValueError(
                "config key 'benchmark:' is no longer supported; use the mapping form instead: "
                "benchmarks: {<name>: {...}}"
            )
        raise ValueError("config is missing 'benchmarks:' (a mapping keyed by benchmark name: {<name>: {...}})")
    return cfg["benchmarks"]


def _run_dir(cfg: dict, suffix: str = "") -> RunDir:
    run_cfg = cfg.get("run", {}) or {}
    return RunDir(run_cfg.get("root", "runs"), f"{run_cfg.get('name', 'run')}{suffix}", cfg)


def _loop_settings(cfg: dict) -> tuple[dict, int]:
    """Validate ``optimizer:`` and ``max_iterations`` (before any LLM is built)."""
    if "optimize" in cfg:
        raise ValueError(
            "config section 'optimize:' is no longer supported; use top-level 'optimizer: {name: ...}' "
            "and 'max_iterations: N' instead"
        )
    opt_cfg = cfg.get("optimizer")
    if not isinstance(opt_cfg, dict) or not isinstance(opt_cfg.get("name"), str):
        raise ValueError(f"config 'optimizer' must be a mapping with a 'name' key, got {opt_cfg!r}")
    max_iterations = cfg.get("max_iterations")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 0:
        raise ValueError(f"config 'max_iterations' must be an int >= 0, got {max_iterations!r}")
    return opt_cfg, max_iterations


def run_loop(cfg: dict) -> Path:
    """Run the optimize loop on split ``train``, saving every round; returns the run dir."""
    bench_cfgs = _benchmark_configs(cfg)
    opt_cfg, max_iterations = _loop_settings(cfg)
    seed = int(cfg.get("seed", 0))
    specs = build_benchmark_specs(bench_cfgs, seed)  # config errors before any LLM is built
    set_seed(seed)
    run = _run_dir(cfg)

    hw = cfg.get("hardware", {}) or {}
    llms = build_role_llms(cfg["llm"], hw.get("vram_gb"))
    target, optimizer_llm = llms["target"], llms["optimizer"]
    evaluator = Evaluator(target, specs, **(cfg.get("evaluator") or {}))
    optimizer = build_optimizer(opt_cfg, optimizer_llm)

    def on_round(index: int, evaluations: list[SkillEvaluation]) -> None:
        rd = run.save_round(index, evaluations)
        tgt_usage, opt_usage = target.usage.snapshot(), optimizer_llm.usage.snapshot()
        run.save_json(
            str((rd / "usage.json").relative_to(run.path)),
            {"target": tgt_usage.to_dict(), "optimizer": opt_usage.to_dict()},
        )
        run.save_usage("target", tgt_usage)
        run.save_usage("optimizer", opt_usage)
        for ev in evaluations:
            _log_evaluation(ev, f"round {index}")

    logger.info(f"run dir: {run.path}")
    result = optimize(optimizer, evaluator, max_iterations, on_round=on_round)

    run.save_json(
        "run.json",
        {"rounds": len(result.history), "stop_reason": result.stop_reason, "max_iterations": max_iterations},
    )
    tgt_usage, opt_usage = target.usage.snapshot(), optimizer_llm.usage.snapshot()
    run.save_usage("target", tgt_usage)
    run.save_usage("optimizer", opt_usage)
    logger.info(f"finished: stop_reason={result.stop_reason} rounds={len(result.history)}")
    logger.info(f"usage target: {_fmt_usage(tgt_usage)}")
    logger.info(f"usage optimizer: {_fmt_usage(opt_usage)}")
    logger.info(f"run dir: {run.path}")
    return run.path


def _resolve_skill(spec: str) -> Skill:
    if spec in BASELINE_NAMES:
        return get_baseline(spec)
    path = resolve_path(spec)
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return Skill.from_dict(json.load(f))
    raise ValueError(f"'{spec}' is neither a baseline ({', '.join(BASELINE_NAMES)}) nor a skill json file")


def run_eval_only(cfg: dict) -> Path:
    """Evaluate ``cfg["eval"]["skills"]`` on split ``eval.split`` (default test) of every benchmark providing it."""
    bench_cfgs = _benchmark_configs(cfg)
    ev_cfg = cfg.get("eval") or {}
    if "stage" in ev_cfg:
        raise ValueError(
            "config key 'eval.stage' is no longer supported; use 'eval.split' (train, val, test or heldout)"
        )
    skills = [_resolve_skill(str(s)) for s in (ev_cfg.get("skills") or list(BASELINE_NAMES))]
    split = str(ev_cfg.get("split") or "test")
    limit = ev_cfg.get("limit")

    seed = int(cfg.get("seed", 0))
    specs = build_benchmark_specs(bench_cfgs, seed)  # config errors before any LLM is built
    set_seed(seed)
    run = _run_dir(cfg, "_eval")

    hw = cfg.get("hardware", {}) or {}
    target = build_role_llms({"target": cfg["llm"]["target"]}, hw.get("vram_gb"))["target"]
    evaluator = Evaluator(target, specs, **(cfg.get("evaluator") or {}))

    logger.info(f"run dir: {run.path}")
    summaries = []
    for i, skill in enumerate(skills, 1):
        logger.info(f"skill {i}/{len(skills)}: {skill.id}")
        evaluation = evaluator.evaluate_skill(skill, split, limit)
        run.save_skill(skill)
        run.save_evaluation(evaluation)
        summaries.append(evaluation.summary())
        _log_evaluation(evaluation)
    run.save_json("summary.json", summaries)
    tgt_usage = target.usage.snapshot()
    run.save_usage("target", tgt_usage)
    logger.info(f"finished: {len(skills)} skills evaluated on split {split}")
    logger.info(f"usage target: {_fmt_usage(tgt_usage)}")
    logger.info(f"run dir: {run.path}")
    return run.path


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    if not args.eval_only and (args.skills is not None or args.split is not None or args.limit is not None):
        parser.error("--skills/--split/--limit require --eval-only")

    overrides = list(args.overrides)
    if args.mock:
        overrides = mock_overrides(read_raw_config(args.config)) + overrides
    cfg = load_config(args.config, overrides)

    if args.eval_only:
        ev_cfg = cfg.get("eval") or {}
        if args.skills is not None:
            ev_cfg["skills"] = list(args.skills)
        if args.split is not None:
            ev_cfg["split"] = args.split
        if args.limit is not None:
            ev_cfg["limit"] = args.limit
        cfg["eval"] = ev_cfg
        return run_eval_only(cfg)
    return run_loop(cfg)
