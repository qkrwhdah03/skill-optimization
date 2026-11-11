"""Terminal progress logging: the only place that configures handlers for the ``skillopt`` logger tree.

Library modules only emit records via ``logging.getLogger(__name__)``; ``cli.main`` calls ``setup_logging``.
Third-party loggers (vllm, transformers, ...) and the root logger are never touched.
"""
from __future__ import annotations

import logging
import sys

LOGGER_NAME: str = "skillopt"
LOG_FORMAT: str = "%(asctime)s %(levelname)-7s %(component)-16s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR")

_PREFIX = LOGGER_NAME + "."


def resolve_level(level: int | str) -> int:
    """int -> as-is; case-insensitive name in ``LEVELS`` -> ``logging.<LEVEL>``; anything else -> ValueError."""
    if isinstance(level, bool):
        raise ValueError(f"invalid log level {level!r}; expected one of {', '.join(LEVELS)}")
    if isinstance(level, int):
        return level
    if isinstance(level, str) and level.upper() in LEVELS:
        return getattr(logging, level.upper())
    raise ValueError(f"invalid log level {level!r}; expected one of {', '.join(LEVELS)}")


class LogFormatter(logging.Formatter):
    """``LOG_FORMAT`` with ``component`` = logger name without the leading ``skillopt.``."""

    def __init__(self) -> None:
        super().__init__(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        record.component = name[len(_PREFIX):] if name.startswith(_PREFIX) else name
        return super().format(record)


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at emit time (never binds the stream)."""

    def __init__(self) -> None:
        super().__init__(sys.stderr)

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:  # ignore: always the current sys.stderr
        pass


def setup_logging(level: int | str = "INFO") -> logging.Logger:
    """Configure the ``skillopt`` logger with exactly one stderr handler owned by this module."""
    resolved = resolve_level(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolved)
    owned = [h for h in logger.handlers if isinstance(h, _StderrHandler)]
    if owned:
        handler = owned[0]
        for extra in owned[1:]:
            logger.removeHandler(extra)
    else:
        handler = _StderrHandler()
        logger.addHandler(handler)
    handler.setFormatter(LogFormatter())
    handler.setLevel(logging.NOTSET)
    logger.propagate = True
    return logger


def preview(text: str, width: int = 80) -> str:
    """Whitespace-collapsed, stripped ``text``; cut to ``width`` characters + ``...`` when longer."""
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width!r}")
    flat = " ".join(str(text).split())
    if len(flat) <= width:
        return flat
    return flat[:width] + "..."


__all__ = ["LOGGER_NAME", "LOG_FORMAT", "DATE_FORMAT", "LEVELS", "resolve_level", "LogFormatter", "setup_logging", "preview"]
