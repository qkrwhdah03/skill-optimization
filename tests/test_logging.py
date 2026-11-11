"""Progress logging via stdlib `logging` (skillopt/logs.py, CLI flags, per-module log records)."""
import contextlib
import io
import json
import logging
import math
import os
import re
import subprocess
import sys

import pytest
import yaml

from helpers import CONFIGS, PROJECT_ROOT, make_mock, read_json
from skillopt.benchmarks import Benchmark, Example
from skillopt.cli import build_parser, main
from skillopt.evaluator import BenchmarkSpec, Evaluator
from skillopt.llm import engine_key, get_or_create_engine
from skillopt.logs import (
    DATE_FORMAT,
    LEVELS,
    LOG_FORMAT,
    LOGGER_NAME,
    LogFormatter,
    preview,
    resolve_level,
    setup_logging,
)
from skillopt.skills import Skill

CONFIG = CONFIGS / "config.yaml"
OPRO = "optimizer.name=opro"
TIME_RE = r"\d+\.\ds"

PROGRESS_RE = re.compile(r"^eval (\S+)\s+(\S+)\s+(\S+)\s+(\d+)/(\d+)$")
DONE_RE = re.compile(
    r"^eval (\S+)\s+(\S+)\s+(\S+)\s+done n=(\d+) acc=(\S+) out_tok=(\S+) think_tok=(\S+) time=" + TIME_RE + r"$"
)


# --- shared helpers / fixtures ----------------------------------------------

def _snapshot(logger):
    return logger.level, list(logger.handlers), logger.propagate


def _restore(logger, snap):
    level, handlers, propagate = snap
    logger.setLevel(level)
    for h in list(logger.handlers):
        if h not in handlers:
            logger.removeHandler(h)
    for h in handlers:
        if h not in logger.handlers:
            logger.addHandler(h)
    logger.propagate = propagate


@pytest.fixture(autouse=True)
def _restore_skillopt_logger():
    logger = logging.getLogger("skillopt")
    snap = _snapshot(logger)
    yield
    _restore(logger, snap)


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.NOTSET)
        self.records = []

    def emit(self, record):
        self.records.append(record)


class _Captured:
    def __init__(self, run, records, out, err, level_after):
        self.run = run
        self.records = records
        self.out = out
        self.err = err
        self.level_after = level_after

    @property
    def messages(self):
        return [r.getMessage() for r in self.records]

    def by_logger(self, name):
        return [r.getMessage() for r in self.records if r.name == name]


def _argv(root, *sets, extra=()):
    argv = ["--config", str(CONFIG), "--mock", "--set", f"run.root={root.as_posix()}"]
    for s in sets:
        argv += ["--set", s]
    return argv + list(extra)


def _captured_main(argv):
    """Run `main` capturing stdout/stderr and every record reaching the skillopt logger.

    Logger state is NOT restored here, so consecutive calls exercise repeated `main` in one process.
    """
    logger = logging.getLogger("skillopt")
    handler = _ListHandler()
    logger.addHandler(handler)
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run = main(argv)
        level_after = logger.level
    finally:
        logger.removeHandler(handler)
    return _Captured(run, handler.records, out.getvalue(), err.getvalue(), level_after)


@pytest.fixture(scope="module")
def loop_run(tmp_path_factory):
    logger = logging.getLogger("skillopt")
    snap = _snapshot(logger)
    try:
        root = tmp_path_factory.mktemp("log_loop")
        return _captured_main(_argv(root, OPRO, "max_iterations=1"))
    finally:
        _restore(logger, snap)


@pytest.fixture(scope="module")
def eval_run(tmp_path_factory):
    logger = logging.getLogger("skillopt")
    snap = _snapshot(logger)
    try:
        root = tmp_path_factory.mktemp("log_eval")
        return _captured_main(_argv(root, extra=["--eval-only", "--limit", "1"]))
    finally:
        _restore(logger, snap)


@pytest.fixture(scope="module")
def level_runs(tmp_path_factory):
    """Same mock loop with -q then -v (two roots), back to back in one process."""
    logger = logging.getLogger("skillopt")
    snap = _snapshot(logger)
    try:
        q = _captured_main(_argv(tmp_path_factory.mktemp("log_q"), OPRO, "max_iterations=1", extra=["-q"]))
        v = _captured_main(_argv(tmp_path_factory.mktemp("log_v"), OPRO, "max_iterations=1", extra=["-v"]))
        return q, v
    finally:
        _restore(logger, snap)


def _usage_line(role, path):
    u = read_json(path)
    return (
        f"usage {role}: calls={u['calls']} input={u['input_tokens']} output={u['output_tokens']} "
        f"reasoning={u['reasoning_tokens']} total={u['total_tokens']} time={u['latency']:.1f}s"
    )


def _skill_files(run, rnd):
    return sorted((run / "rounds" / rnd / "skills").glob("*.json"))


def _index(messages, predicate, start=0):
    for i in range(start, len(messages)):
        if predicate(messages[i]):
            return i
    raise AssertionError(f"no matching message after index {start}")


# --- AC1: resolve_level -------------------------------------------------------

def test_constants():
    assert LOGGER_NAME == "skillopt"
    assert LOG_FORMAT == "%(asctime)s %(levelname)-7s %(component)-16s | %(message)s"
    assert DATE_FORMAT == "%Y-%m-%d %H:%M:%S"
    assert tuple(LEVELS) == ("DEBUG", "INFO", "WARNING", "ERROR")


@pytest.mark.parametrize("name,expected", [
    ("info", logging.INFO),
    ("INFO", logging.INFO),
    ("Debug", logging.DEBUG),
    ("warning", logging.WARNING),
    ("ERROR", logging.ERROR),
])
def test_resolve_level_names(name, expected):
    assert resolve_level(name) == expected


@pytest.mark.parametrize("value", [logging.DEBUG, logging.WARNING, 5, 42])
def test_resolve_level_int_passthrough(value):
    assert resolve_level(value) == value


@pytest.mark.parametrize("bad", ["verbose", "", "CRITICAL_ISH"])
def test_resolve_level_invalid_string(bad):
    with pytest.raises(ValueError):
        resolve_level(bad)


def test_resolve_level_none():
    with pytest.raises((ValueError, TypeError)):
        resolve_level(None)


# --- AC2: preview -------------------------------------------------------------

def test_preview_short_unchanged():
    assert preview("Solve it step by step.") == "Solve it step by step."


def test_preview_collapses_whitespace():
    assert preview("a\n\n  b\tc") == "a b c"
    assert preview("  padded \n") == "padded"


def test_preview_long_truncated():
    text = "x" * 200
    out = preview(text)
    assert len(out) == 83
    assert out == "x" * 80 + "..."


def test_preview_exact_width_no_ellipsis():
    text = "y" * 80
    assert preview(text) == text


def test_preview_truncates_after_collapsing():
    text = ("word \n" * 40).strip()  # collapses to "word word ..." (199 chars)
    out = preview(text, width=20)
    assert out == "word word word word " + "..."


def test_preview_custom_width():
    assert preview("abcdefghij", width=4) == "abcd..."
    assert preview("abcd", width=4) == "abcd"
    assert preview("abcdefghij", width=1) == "a..."


def test_preview_empty():
    assert preview("") == ""
    assert preview("   \n\t ") == ""


@pytest.mark.parametrize("width", [0, -1])
def test_preview_invalid_width(width):
    with pytest.raises(ValueError):
        preview("abc", width=width)


# --- AC3: LogFormatter ----------------------------------------------------------

def _record(name, msg="hello", level=logging.INFO):
    return logging.LogRecord(name, level, __file__, 1, msg, None, None)


def test_formatter_line_shape():
    line = LogFormatter().format(_record("skillopt.evaluator"))
    assert re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} INFO\s+evaluator\s+\| hello$", line), line


@pytest.mark.parametrize("name,component", [
    ("skillopt.evaluator", "evaluator"),
    ("skillopt.optimizer.loop", "optimizer.loop"),
    ("skillopt", "skillopt"),
    ("vllm", "vllm"),
])
def test_formatter_component(name, component):
    rec = _record(name)
    line = LogFormatter().format(rec)
    assert rec.component == component
    assert re.search(r" INFO\s+" + re.escape(component) + r"\s+\| hello$", line), line


def test_formatter_alignment():
    fmt = LogFormatter()
    lines = [
        fmt.format(_record("skillopt.evaluator", level=logging.INFO)),
        fmt.format(_record("skillopt.optimizer.loop", level=logging.WARNING)),
        fmt.format(_record("skillopt", level=logging.DEBUG)),
        fmt.format(_record("vllm", level=logging.ERROR)),
        fmt.format(_record("skillopt.llm.resources", level=logging.INFO)),  # component exactly 13 chars
        fmt.format(_record("skillopt." + "c" * 16, level=logging.INFO)),  # component exactly 16 chars
    ]
    cols = {line.index("|") for line in lines}
    assert len(cols) == 1, lines


# --- AC4: setup_logging ---------------------------------------------------------

def test_setup_logging_returns_skillopt_logger():
    assert setup_logging("INFO") is logging.getLogger("skillopt")


@pytest.mark.parametrize("level,expected", [("WARNING", 30), ("debug", 10), ("INFO", 20), (logging.ERROR, 40)])
def test_setup_logging_sets_level(level, expected):
    logger = setup_logging(level)
    assert logger.level == expected


def test_setup_logging_default_is_info():
    assert setup_logging().level == logging.INFO


def test_setup_logging_invalid_level():
    with pytest.raises(ValueError):
        setup_logging("loud")


@pytest.fixture
def clean_skillopt_logger():
    """Start from a logger without handlers (earlier `main` runs in other files may have configured it);
    the autouse fixture restores the original state afterwards."""
    logger = logging.getLogger("skillopt")
    for h in list(logger.handlers):
        logger.removeHandler(h)
    return logger


def test_setup_logging_no_handler_stacking(clean_skillopt_logger):
    logger = clean_skillopt_logger
    before = len(logger.handlers)
    setup_logging("INFO")
    setup_logging("DEBUG")
    setup_logging("WARNING")
    assert len(logger.handlers) == before + 1


def test_setup_logging_keeps_foreign_handler(clean_skillopt_logger):
    logger = clean_skillopt_logger
    foreign = logging.NullHandler()
    logger.addHandler(foreign)
    before = len(logger.handlers)
    setup_logging("INFO")
    setup_logging("INFO")
    assert foreign in logger.handlers
    assert len(logger.handlers) == before + 1


def test_setup_logging_handler_uses_log_formatter(clean_skillopt_logger):
    logger = clean_skillopt_logger
    before = list(logger.handlers)
    setup_logging("INFO")
    added = [h for h in logger.handlers if h not in before]
    assert len(added) == 1
    assert isinstance(added[0].formatter, LogFormatter)


def test_setup_logging_propagates_and_leaves_root_alone():
    root = logging.getLogger()
    root_handlers, root_level = list(root.handlers), root.level
    logger = setup_logging("DEBUG")
    assert logger.propagate is True
    assert root.handlers == root_handlers
    assert root.level == root_level


def _assert_clean_ping(err):
    assert "| ping" in err
    assert err.count("| ping") == 1
    assert "--- Logging error ---" not in err
    assert "I/O operation on closed file" not in err


def test_setup_logging_writes_to_stderr(capsys):
    setup_logging("INFO")
    logging.getLogger("skillopt.x").info("ping")
    captured = capsys.readouterr()
    assert captured.out == ""
    _assert_clean_ping(captured.err)
    assert re.search(r"INFO\s+x\s+\| ping", captured.err), captured.err


def test_setup_logging_writes_to_stderr_again(capsys):
    # a second test: the handler must use this test's sys.stderr, not a closed one from earlier
    setup_logging("INFO")
    setup_logging("INFO")
    logging.getLogger("skillopt.x").info("ping")
    _assert_clean_ping(capsys.readouterr().err)


def test_setup_logging_follows_current_stderr(monkeypatch, capsys):
    setup_logging("INFO")
    first = io.StringIO()
    monkeypatch.setattr(sys, "stderr", first)
    logging.getLogger("skillopt.x").info("ping")
    second = io.StringIO()
    monkeypatch.setattr(sys, "stderr", second)
    first.close()
    logging.getLogger("skillopt.x").info("ping")
    _assert_clean_ping(second.getvalue())


def test_setup_logging_level_filters(capsys):
    setup_logging("WARNING")
    logging.getLogger("skillopt.x").info("quiet-info")
    logging.getLogger("skillopt.x").warning("loud-warning")
    err = capsys.readouterr().err
    assert "quiet-info" not in err
    assert "| loud-warning" in err


# --- AC5: import does not configure logging ----------------------------------------

def test_import_does_not_configure_logging():
    code = (
        "import logging\n"
        "import skillopt, skillopt.cli, skillopt.logs\n"
        "lg = logging.getLogger('skillopt')\n"
        "print('HANDLERS', len(lg.handlers), 'LEVEL', lg.level)\n"
        "raise SystemExit(0 if (lg.handlers == [] and lg.level == logging.NOTSET) else 1)\n"
    )
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --- AC6: parser -------------------------------------------------------------------

@pytest.mark.parametrize("flags,expected", [
    ([], "INFO"),
    (["-v"], "DEBUG"),
    (["--verbose"], "DEBUG"),
    (["-q"], "WARNING"),
    (["--quiet"], "WARNING"),
    (["--log-level", "error"], "ERROR"),
    (["--log-level", "debug"], "DEBUG"),
    (["--log-level", "Warning"], "WARNING"),
    (["--log-level", "INFO"], "INFO"),
])
def test_parser_log_level(flags, expected):
    assert build_parser().parse_args(flags).log_level == expected


@pytest.mark.parametrize("flags", [
    ["-v", "-q"],
    ["-q", "--log-level", "INFO"],
    ["--verbose", "--log-level", "DEBUG"],
    ["--log-level", "loud"],
])
def test_parser_log_level_errors(flags, capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(flags)
    assert exc.value.code == 2


# --- AC7: mock loop log records ---------------------------------------------------------

def test_loop_llm_lines(loop_run):
    msgs = loop_run.by_logger("skillopt.llm.registry")
    assert "llm target: backend=mock model=mock-qwen3 dtype=-" in msgs
    assert "llm optimizer: backend=mock model=mock-qwen3 dtype=-" in msgs


def test_loop_run_dir_first_and_last(loop_run):
    cli = loop_run.by_logger("skillopt.cli")
    assert cli[0] == f"run dir: {loop_run.run}"
    assert cli[-1] == f"run dir: {loop_run.run}"
    assert sum(1 for m in loop_run.messages if m.startswith("run dir:")) == 2


def test_loop_optimizer_sequence(loop_run):
    run = loop_run.run
    msgs = loop_run.by_logger("skillopt.optimizer.loop")
    k = len(_skill_files(run, "001"))
    assert k > 0
    i0 = _index(msgs, lambda m: m == "initial prompts: 3")
    i1 = _index(msgs, lambda m: m == "round 0/1: evaluating 3 prompts on split train", i0)
    i2 = _index(msgs, lambda m: m == "round 0: optimizer generating new prompts ...", i1)
    i3 = _index(msgs, lambda m: re.fullmatch(rf"round 0: optimizer returned {k} prompts in {TIME_RE}", m), i2)
    i4 = _index(msgs, lambda m: m == f"round 1/1: evaluating {k} prompts on split train", i3)
    _index(msgs, lambda m: m == "stop: max_iterations after 2 rounds", i4)


def test_loop_initial_prompt_lines(loop_run):
    run = loop_run.run
    msgs = loop_run.by_logger("skillopt.optimizer.loop")
    start = msgs.index("initial prompts: 3")
    end = msgs.index("round 0/1: evaluating 3 prompts on split train")
    block = msgs[start + 1:end]
    expected = []
    for sid in ("direct", "cot", "cod"):
        instr = read_json(run / "rounds" / "000" / "skills" / f"{sid}.json")["instruction"]
        expected.append(f"  new prompt {sid}: {preview(instr)}")
    assert block == expected


def test_loop_new_prompt_lines_per_skill(loop_run):
    run = loop_run.run
    msgs = loop_run.by_logger("skillopt.optimizer.loop")
    k = len(_skill_files(run, "001"))
    ret = _index(msgs, lambda m: re.fullmatch(rf"round 0: optimizer returned {k} prompts in {TIME_RE}", m))
    nxt = msgs.index(f"round 1/1: evaluating {k} prompts on split train")
    block = msgs[ret + 1:nxt]
    expected = set()
    for path in _skill_files(run, "001"):
        skill = read_json(path)
        expected.add(f"  new prompt {skill['id']}: {preview(skill['instruction'])}")
    assert len(block) == k
    assert set(block) == expected


def test_loop_no_debug_at_info(loop_run):
    assert not any("full instruction:" in m for m in loop_run.messages)
    assert all(r.levelno >= logging.INFO for r in loop_run.records)


def test_loop_finished_and_usage(loop_run):
    run = loop_run.run
    cli = loop_run.by_logger("skillopt.cli")
    tail = cli[-4:]
    assert tail[0] == "finished: stop_reason=max_iterations rounds=2"
    assert tail[1] == _usage_line("target", run / "usage" / "target.json")
    assert tail[2] == _usage_line("optimizer", run / "usage" / "optimizer.json")
    assert tail[3] == f"run dir: {run}"


def test_loop_round_result_lines(loop_run):
    run = loop_run.run
    cli = loop_run.by_logger("skillopt.cli")
    for i, rnd in ((0, "000"), (1, "001")):
        lines = [m for m in cli if m.startswith(f"round {i} | ")]
        summary = read_json(run / "rounds" / rnd / "summary.json")
        n_results = sum(len(e["benchmarks"]) for e in summary)
        assert len(lines) == n_results, lines
        for entry in summary:
            for key, s in entry["benchmarks"].items():
                pat = (rf"round {i} \| {re.escape(entry['skill_id'])}\s+{re.escape(key)}\s+"
                       rf"acc={s['accuracy']:.3f}\s+out_tok=\s*\S+\s+think_tok=\s*\S+\s+total_tok=\s*\S+\s+n={s['n']}")
                assert any(re.fullmatch(pat, m) for m in lines), (pat, lines)


def test_loop_evaluator_progress(loop_run):
    run = loop_run.run
    msgs = loop_run.by_logger("skillopt.evaluator")
    assert msgs
    for rnd in ("000", "001"):
        for entry in read_json(run / "rounds" / rnd / "summary.json"):
            for key, s in entry["benchmarks"].items():
                sid = entry["skill_id"]
                progress = [m for m in msgs if (mm := PROGRESS_RE.match(m))
                            and mm.group(1) == sid and mm.group(2) == key and mm.group(3) == "train"]
                done = [m for m in msgs if (mm := DONE_RE.match(m))
                        and mm.group(1) == sid and mm.group(2) == key and mm.group(3) == "train"]
                assert len(progress) == math.ceil(s["n"] / 16), (sid, key, progress)
                assert PROGRESS_RE.match(progress[-1]).group(4) == str(s["n"])
                assert len(done) == 1, (sid, key, done)
                d = DONE_RE.match(done[0])
                assert d.group(4) == str(s["n"])
                assert d.group(5) == f"{s['accuracy']:.3f}"
                assert d.group(6) == f"{s['avg_output_tokens']:.1f}"
                assert d.group(7) == f"{s['avg_reasoning_tokens']:.1f}"


# --- AC8: Evaluator progress (direct) ------------------------------------------------------

class _TinyBench(Benchmark):
    name = "tinybench_logging"
    answer_format = "Answer with a number."

    def __init__(self, n):
        self._examples = [Example(id=f"ex{i}", question=f"Question {i}?", answer=str(i % 3)) for i in range(n)]

    def load(self, split):
        return list(self._examples)

    def extract_answer(self, text):
        nums = re.findall(r"-?\d+", text or "")
        return nums[-1] if nums else None

    def score(self, pred, gold):
        return pred is not None and pred == gold


def _tiny_evaluator(n=5, batch_size=2):
    mock = make_mock(["<think>a b c</think>The answer is 1.", "The answer is 2."])
    return Evaluator(mock, [BenchmarkSpec(key="tb", benchmark=_TinyBench(n))], batch_size=batch_size)


def _skill():
    return Skill(id="sk1", instruction="Think briefly.", few_shot=[])


def _eval_records(caplog):
    return [r for r in caplog.records if r.name == "skillopt.evaluator"]


def _check_done(msg, res):
    d = DONE_RE.match(msg)
    assert d, msg
    assert d.group(1) == res.skill_id
    assert d.group(2) == res.benchmark
    assert d.group(3) == res.split
    assert d.group(4) == str(res.n)
    assert d.group(5) == f"{res.accuracy:.3f}"
    assert d.group(6) == f"{res.avg_output_tokens:.1f}"
    assert d.group(7) == f"{res.avg_reasoning_tokens:.1f}"


def test_evaluator_multi_batch_progress(caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    res = _tiny_evaluator(n=5, batch_size=2).evaluate_split(_skill(), "tb", "train")
    recs = _eval_records(caplog)
    msgs = [r.getMessage() for r in recs]
    progress = [PROGRESS_RE.match(m) for m in msgs if PROGRESS_RE.match(m)]
    assert [(int(p.group(4)), int(p.group(5))) for p in progress] == [(2, 5), (4, 5), (5, 5)]
    assert all(p.group(1) == "sk1" and p.group(2) == "tb" and p.group(3) == "train" for p in progress)
    done = [m for m in msgs if DONE_RE.match(m)]
    assert len(done) == 1
    assert msgs[-1] == done[0]
    _check_done(done[0], res)
    assert all(r.levelno == logging.INFO for r in recs)


def test_evaluator_progress_exact_text(caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    _tiny_evaluator(n=3, batch_size=2).evaluate_split(_skill(), "tb", "val")
    msgs = [r.getMessage() for r in _eval_records(caplog)]
    assert msgs[0] == f"eval {'sk1':<24} {'tb':<10} {'val':<8} {2:>5}/3"
    assert msgs[1] == f"eval {'sk1':<24} {'tb':<10} {'val':<8} {3:>5}/3"


def test_evaluator_limit_total(caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    res = _tiny_evaluator(n=5, batch_size=2).evaluate_split(_skill(), "tb", "train", limit=3)
    msgs = [r.getMessage() for r in _eval_records(caplog)]
    progress = [PROGRESS_RE.match(m) for m in msgs if PROGRESS_RE.match(m)]
    assert [(int(p.group(4)), int(p.group(5))) for p in progress] == [(2, 3), (3, 3)]
    done = [m for m in msgs if DONE_RE.match(m)]
    assert len(done) == 1
    _check_done(done[0], res)
    assert res.n == 3


def test_evaluator_single_batch(caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    _tiny_evaluator(n=4, batch_size=16).evaluate_split(_skill(), "tb", "train")
    msgs = [r.getMessage() for r in _eval_records(caplog)]
    progress = [m for m in msgs if PROGRESS_RE.match(m)]
    assert len(progress) == 1
    assert progress[0].endswith(" 4/4")


def test_evaluator_zero_examples(caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    res = _tiny_evaluator(n=5, batch_size=2).evaluate_split(_skill(), "tb", "train", limit=0)
    msgs = [r.getMessage() for r in _eval_records(caplog)]
    assert not [m for m in msgs if PROGRESS_RE.match(m)]
    done = [m for m in msgs if DONE_RE.match(m)]
    assert len(done) == 1
    assert " done n=0 " in done[0]
    _check_done(done[0], res)


def test_evaluator_logging_does_not_change_results(caplog):
    quiet = _tiny_evaluator(n=5, batch_size=2).evaluate_split(_skill(), "tb", "train")
    caplog.set_level(logging.DEBUG, logger="skillopt")
    loud = _tiny_evaluator(n=5, batch_size=2).evaluate_split(_skill(), "tb", "train")
    strip = lambda d: {k: v for k, v in d.items() if k not in ("latency", "avg_latency")}
    assert strip(quiet.summary()) == strip(loud.summary())


# --- AC9: eval-only -------------------------------------------------------------------------

def test_eval_only_skill_lines(eval_run):
    cli = eval_run.by_logger("skillopt.cli")
    skill_lines = [m for m in cli if m.startswith("skill ")]
    assert skill_lines == ["skill 1/3: direct", "skill 2/3: cot", "skill 3/3: cod"]


def test_eval_only_run_dir_and_tail(eval_run):
    run = eval_run.run
    cli = eval_run.by_logger("skillopt.cli")
    assert cli[0] == f"run dir: {run}"
    assert cli[-3:] == [
        "finished: 3 skills evaluated on split test",
        _usage_line("target", run / "usage" / "target.json"),
        f"run dir: {run}",
    ]
    assert not any(m.startswith("usage optimizer:") for m in eval_run.messages)
    assert sum(1 for m in eval_run.messages if m.startswith("run dir:")) == 2


def test_eval_only_result_lines(eval_run):
    run = eval_run.run
    cli = eval_run.by_logger("skillopt.cli")
    summaries = read_json(run / "summary.json")
    lines = [m for m in cli if m.startswith("test | ")]
    assert len(lines) == sum(len(e["benchmarks"]) for e in summaries)
    # result lines follow their skill line
    for i, entry in enumerate(summaries, start=1):
        pos = cli.index(f"skill {i}/3: {entry['skill_id']}")
        for key in entry["benchmarks"]:
            pat = rf"test \| {re.escape(entry['skill_id'])}\s+{re.escape(key)}\s+acc=.*n=1"
            later = [m for m in cli[pos + 1:] if re.fullmatch(pat, m)]
            assert later, (pat, cli)


def test_eval_only_evaluator_lines(eval_run):
    run = eval_run.run
    msgs = eval_run.by_logger("skillopt.evaluator")
    summaries = read_json(run / "summary.json")
    assert [e["skill_id"] for e in summaries] == ["direct", "cot", "cod"]
    for entry in summaries:
        assert entry["benchmarks"], entry
        for key, s in entry["benchmarks"].items():
            sid = entry["skill_id"]
            progress = [mm for m in msgs if (mm := PROGRESS_RE.match(m))
                        and (mm.group(1), mm.group(2), mm.group(3)) == (sid, key, "test")]
            done = [mm for m in msgs if (mm := DONE_RE.match(m))
                    and (mm.group(1), mm.group(2), mm.group(3)) == (sid, key, "test")]
            assert [(p.group(4), p.group(5)) for p in progress] == [("1", "1")]
            assert len(done) == 1
            assert done[0].group(4) == "1"
            assert done[0].group(5) == f"{s['accuracy']:.3f}"
            assert done[0].group(6) == f"{s['avg_output_tokens']:.1f}"
            assert done[0].group(7) == f"{s['avg_reasoning_tokens']:.1f}"


def test_eval_only_llm_line(eval_run):
    msgs = eval_run.by_logger("skillopt.llm.registry")
    assert msgs == ["llm target: backend=mock model=mock-qwen3 dtype=-"]


# --- AC10: no stdout -----------------------------------------------------------------------------

def test_no_stdout_loop(loop_run):
    assert loop_run.out == ""
    assert "run dir:" in loop_run.err


def test_no_stdout_eval_only(eval_run):
    assert eval_run.out == ""
    assert "run dir:" in eval_run.err


def test_stderr_lines_formatted(loop_run):
    lines = [ln for ln in loop_run.err.splitlines() if ln.strip()]
    assert lines
    fmt = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (DEBUG|INFO|WARNING|ERROR)\s+\S+\s+\| ")
    formatted = [ln for ln in lines if fmt.match(ln)]
    assert any(re.search(r"INFO\s+cli\s+\| run dir: ", ln) for ln in formatted)
    assert any(re.search(r"INFO\s+evaluator\s+\| eval ", ln) for ln in formatted)
    assert any(re.search(r"INFO\s+optimizer\.loop\s+\| round 0/1: ", ln) for ln in formatted)


def test_stderr_no_duplicate_lines(loop_run):
    # each emission appears exactly once
    assert loop_run.err.count("| run dir: ") == 2
    assert loop_run.err.count("| initial prompts: 3") == 1
    assert loop_run.err.count("| stop: max_iterations after 2 rounds") == 1


def test_main_sets_info_level(loop_run):
    assert loop_run.level_after == logging.INFO


# --- AC11: -q / -v ------------------------------------------------------------------------------

def test_quiet_hides_info(level_runs):
    q, _ = level_runs
    assert q.out == ""
    for needle in ("run dir:", "| eval ", "| round ", "| initial prompts", "| llm target"):
        assert needle not in q.err, needle
    assert q.level_after == logging.WARNING
    assert not [r for r in q.records if r.levelno < logging.WARNING]


def test_verbose_shows_debug(level_runs):
    _, v = level_runs
    assert v.out == ""
    assert v.level_after == logging.DEBUG
    assert re.search(r"DEBUG\s+optimizer\.loop\s+\| prompt direct full instruction: ", v.err), v.err[:2000]
    instr = read_json(v.run / "rounds" / "000" / "skills" / "direct.json")["instruction"]
    assert f"prompt direct full instruction: {instr}" in [r.getMessage() for r in v.records]
    for path in _skill_files(v.run, "001"):
        sid = read_json(path)["id"]
        assert f"prompt {sid} full instruction: " in v.err


def test_verbose_after_quiet_no_duplicates(level_runs):
    # repeated main in one process (-q then -v): no stacked handlers
    _, v = level_runs
    assert v.err.count("| run dir: ") == 2
    assert v.err.count("| initial prompts: 3") == 1


# --- AC12: saved outputs identical across levels ---------------------------------------------------

def _strip_latency(obj):
    if isinstance(obj, dict):
        return {k: _strip_latency(v) for k, v in obj.items() if k not in ("latency", "avg_latency")}
    if isinstance(obj, list):
        return [_strip_latency(v) for v in obj]
    return obj


def _load_comparable(path):
    if path.suffix == ".json":
        return _strip_latency(read_json(path))
    if path.suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            return [_strip_latency(json.loads(line)) for line in f if line.strip()]
    return path.read_bytes()


def _rel_files(run):
    return sorted(p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file())


def test_outputs_identical_across_levels(level_runs):
    q, v = level_runs
    assert q.run != v.run
    files = _rel_files(q.run)
    assert files == _rel_files(v.run)
    assert any(f.startswith("rounds/001/") for f in files)
    checked = 0
    for rel in files:
        if rel.startswith("rounds/") or rel == "run.json" or rel.startswith("usage/"):
            assert _load_comparable(q.run / rel) == _load_comparable(v.run / rel), rel
            checked += 1
    assert checked > 0
    cfgs = []
    for run in (q.run, v.run):
        with open(run / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.get("run", {}).pop("root", None)
        cfgs.append(cfg)
    assert cfgs[0] == cfgs[1]


# --- AC13: get_or_create_engine ------------------------------------------------------------------

RES_LOGGER = "skillopt.llm.resources"
VLLM_KEY_CFG = {"backend": "vllm", "model": "m", "engine": {"dtype": "float16"}}


def _res_msgs(caplog, level=None):
    return [r.getMessage() for r in caplog.records
            if r.name == RES_LOGGER and (level is None or r.levelno == level)]


def test_engine_load_logged(clean_engines, caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    calls = []
    engine = object()

    def factory():
        calls.append(1)
        return engine

    assert get_or_create_engine(engine_key(VLLM_KEY_CFG), factory) is engine
    msgs = _res_msgs(caplog, logging.INFO)
    assert msgs[0] == "loading engine vllm:m dtype=float16 ..."
    assert re.fullmatch(rf"engine vllm:m dtype=float16 loaded in {TIME_RE}", msgs[1]), msgs
    assert len(msgs) == 2
    assert calls == [1]


def test_engine_loading_logged_before_factory(clean_engines, caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    seen = []

    def factory():
        seen.append(list(_res_msgs(caplog)))
        return object()

    get_or_create_engine(engine_key(VLLM_KEY_CFG), factory)
    assert seen == [["loading engine vllm:m dtype=float16 ..."]]


def test_engine_cache_hit(clean_engines, caplog):
    caplog.set_level(logging.DEBUG, logger="skillopt")
    calls = []

    def factory():
        calls.append(1)
        return object()

    key = engine_key(VLLM_KEY_CFG)
    first = get_or_create_engine(key, factory)
    caplog.clear()
    second = get_or_create_engine(key, factory)
    assert second is first
    assert calls == [1]
    assert not [m for m in _res_msgs(caplog, logging.INFO) if m.startswith("loading engine")]
    assert "reusing engine vllm:m dtype=float16" in _res_msgs(caplog, logging.DEBUG)


def test_engine_cache_hit_silent_at_info(clean_engines, caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    key = engine_key(VLLM_KEY_CFG)
    get_or_create_engine(key, object)
    caplog.clear()
    get_or_create_engine(key, object)
    assert _res_msgs(caplog) == []


def test_engine_key_without_dtype(clean_engines, caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    get_or_create_engine(engine_key({"backend": "hf", "model": "m2"}), object)
    msgs = _res_msgs(caplog, logging.INFO)
    assert msgs[0] == "loading engine hf:m2 dtype=- ..."
    assert re.fullmatch(rf"engine hf:m2 dtype=- loaded in {TIME_RE}", msgs[1]), msgs


def test_engine_nonstandard_key(clean_engines, caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    engine = object()
    assert get_or_create_engine(("x",), lambda: engine) is engine
    msgs = _res_msgs(caplog, logging.INFO)
    assert msgs[0] == f"loading engine {('x',)!r} ..."
    assert re.fullmatch(re.escape(f"engine {('x',)!r} loaded in ") + TIME_RE, msgs[1]), msgs


def test_engine_factory_error(clean_engines, caplog):
    caplog.set_level(logging.INFO, logger="skillopt")
    key = engine_key(VLLM_KEY_CFG)
    boom = RuntimeError("boom")

    def bad():
        raise boom

    with pytest.raises(RuntimeError) as exc:
        get_or_create_engine(key, bad)
    assert exc.value is boom
    assert not [m for m in _res_msgs(caplog) if " loaded in " in m]

    caplog.clear()
    engine = object()
    assert get_or_create_engine(key, lambda: engine) is engine
    msgs = _res_msgs(caplog, logging.INFO)
    assert msgs[0] == "loading engine vllm:m dtype=float16 ..."
    assert re.fullmatch(rf"engine vllm:m dtype=float16 loaded in {TIME_RE}", msgs[1]), msgs
