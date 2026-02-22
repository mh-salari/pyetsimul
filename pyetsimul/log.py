"""Centralized log and print control for PyEtSimul.

Controls all terminal output through a single LogLevel flag.
Optionally writes all output (including warnings.warn) to a log file.

Usage:
    from pyetsimul.log import set_log_level, LogLevel

    set_log_level(LogLevel.SILENT)     # only errors
    set_log_level(LogLevel.WARNING)    # warnings + errors
    set_log_level(LogLevel.INFO)       # everything (default)

    set_log_file("simulation.log")     # also write to file
    close_log_file()                   # stop writing to file
"""

import pathlib
import warnings
from enum import IntEnum
from typing import Any

from tabulate import tabulate as _tabulate


class LogLevel(IntEnum):
    """Controls which messages are shown in the terminal."""

    SILENT = 0
    WARNING = 1
    INFO = 2


_level: LogLevel = LogLevel.INFO
_log_file = None
_original_showwarning = warnings.showwarning


def set_log_level(level: LogLevel) -> None:
    """Set the global log level for all PyEtSimul output."""
    global _level  # noqa: PLW0603
    _level = level


def get_log_level() -> LogLevel:
    """Return the current log level."""
    return _level


def set_log_file(path: str) -> None:
    """Start writing all output (including warnings) to a file.

    The file is opened in write mode, overwriting any existing content.
    Console output continues as normal based on the current log level.
    """
    global _log_file  # noqa: PLW0603
    close_log_file()
    _log_file = pathlib.Path(path).open("w", encoding="utf-8")  # noqa: SIM115
    warnings.showwarning = _showwarning_with_file


def close_log_file() -> None:
    """Stop writing to the log file and close it."""
    global _log_file  # noqa: PLW0603
    if _log_file is not None:
        _log_file.close()
        _log_file = None
    warnings.showwarning = _original_showwarning


def _write_to_file(text: str) -> None:
    """Write a line to the log file if one is open."""
    if _log_file is not None:
        _log_file.write(text + "\n")
        _log_file.flush()


def _showwarning_with_file(
    message: Warning,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: Any = None,
    line: str | None = None,
) -> None:
    """Custom warning handler that also writes to our log file."""
    _original_showwarning(message, category, filename, lineno, file, line)
    _write_to_file(f"[{category.__name__}] {filename}:{lineno}: {message}")


def info(*args: Any, **kwargs: Any) -> None:
    """Print info-level messages (progress, results, tables).

    Shown when log level is INFO. Silenced at WARNING or SILENT.
    """
    if _level >= LogLevel.INFO:
        print(*args, **kwargs)
    text = " ".join(str(a) for a in args)
    _write_to_file(text)


def warning(*args: Any, **kwargs: Any) -> None:
    """Print warning-level messages (validation issues, unusual values).

    Shown when log level is WARNING or INFO. Silenced at SILENT.
    """
    if _level >= LogLevel.WARNING:
        print(*args, **kwargs)
    text = " ".join(str(a) for a in args)
    _write_to_file(f"[WARNING] {text}")


def error(*args: Any, **kwargs: Any) -> None:
    """Print error-level messages. Always shown regardless of log level."""
    print(*args, **kwargs)
    text = " ".join(str(a) for a in args)
    _write_to_file(f"[ERROR] {text}")


def table(data: Any, headers: Any, **kwargs: Any) -> None:
    """Print a tabulate table at info level.

    Shown when log level is INFO. Silenced at WARNING or SILENT.
    """
    formatted = _tabulate(data, headers=headers, **kwargs)
    if _level >= LogLevel.INFO:
        print(formatted)
    _write_to_file(formatted)
