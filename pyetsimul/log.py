"""Centralized log and print control for PyEtSimul.

Controls all terminal output through a single LogLevel flag.
Optionally writes all output (including warnings.warn) to a log file.

Usage:
    from pyetsimul.log import set_log_level, LogLevel

    set_log_level(LogLevel.SILENT)     # only errors
    set_log_level(LogLevel.WARNING)    # warnings + errors
    set_log_level(LogLevel.INFO)       # progress and results as well (default)
    set_log_level(LogLevel.DEBUG)      # also one line for every glint that is not found

    set_log_file("simulation.log")     # also write to file
    close_log_file()                   # stop writing to file
"""

import multiprocessing
import pathlib
import warnings
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any

from tabulate import tabulate as _tabulate


class LogLevel(IntEnum):
    """Controls which messages are shown in the terminal."""

    SILENT = 0
    WARNING = 1
    INFO = 2
    DEBUG = 3


_level: LogLevel = LogLevel.INFO
_log_file_path: pathlib.Path | None = None
_original_showwarning = warnings.showwarning


def set_log_level(level: LogLevel) -> None:
    """Set the global log level for all PyEtSimul output."""
    global _level  # noqa: PLW0603
    _level = level


def get_log_level() -> LogLevel:
    """Return the current log level."""
    return _level


def set_log_file(path: str, mode: str = "a") -> None:
    """Start writing all output (including warnings) to a file.

    mode: "a" to append (default), "w" to overwrite.
    """
    # A spawned worker process imports the main module again. A call made there is skipped, so that it
    # cannot empty the log; a worker gets its log file from configure_worker.
    if multiprocessing.current_process().name != "MainProcess":
        return
    if mode == "w":
        pathlib.Path(path).write_text("", encoding="utf-8")
    _attach_log_file(pathlib.Path(path))


def get_log_file() -> pathlib.Path | None:
    """Return the path of the log file, or None when no log file is set."""
    return _log_file_path


def configure_worker(level: LogLevel, log_file: pathlib.Path | None) -> None:
    """Give a worker process the log level and the log file of the process that starts it.

    A spawned worker inherits neither. The worker appends to the log file and never empties it.
    """
    set_log_level(level)
    if log_file is not None:
        _attach_log_file(log_file)


def _attach_log_file(path: pathlib.Path) -> None:
    """Send the output and the warnings of this process to the log file as well."""
    global _log_file_path  # noqa: PLW0603
    _log_file_path = path
    warnings.showwarning = _showwarning_with_file


def close_log_file() -> None:
    """Stop writing to the log file."""
    global _log_file_path  # noqa: PLW0603
    _log_file_path = None
    warnings.showwarning = _original_showwarning


def _write_to_file(text: str, level: str = "INFO") -> None:
    """Append a timestamped, separated entry to the log file if one is set."""
    if _log_file_path is None:
        return
    stripped = text.strip("\n")
    # Skip lines that are only dashes (visual separators)
    if stripped and all(c == "-" for c in stripped):
        return
    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
    # One write per entry, so that the entries of processes that share the file do not interleave.
    with _log_file_path.open("a", encoding="utf-8") as f:
        f.write(f"--- [{timestamp}] [{level}] ---\n{stripped}\n")


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
    _write_to_file(f"[{category.__name__}] {filename}:{lineno}: {message}", level="WARNING")


def info(*args: Any, **kwargs: Any) -> None:
    """Print info-level messages (progress, results, tables).

    Shown when log level is INFO. Silenced at WARNING or SILENT.
    """
    if _level >= LogLevel.INFO:
        print(*args, **kwargs)
    _write_to_file(" ".join(str(a) for a in args))


def debug(*args: Any, **kwargs: Any) -> None:
    """Print debug-level messages (one line per event that repeats in a simulation, such as a missing glint).

    Shown only when log level is DEBUG, so a large run does not print one line per image by default.
    """
    if _level >= LogLevel.DEBUG:
        print(*args, **kwargs)
        _write_to_file(" ".join(str(a) for a in args), level="DEBUG")


def warning(*args: Any, **kwargs: Any) -> None:
    """Print warning-level messages (validation issues, unusual values).

    Shown when log level is WARNING or INFO. Silenced at SILENT.
    """
    if _level >= LogLevel.WARNING:
        print(*args, **kwargs)
    _write_to_file(" ".join(str(a) for a in args), level="WARNING")


def error(*args: Any, **kwargs: Any) -> None:
    """Print error-level messages. Always shown regardless of log level."""
    print(*args, **kwargs)
    _write_to_file(" ".join(str(a) for a in args), level="ERROR")


def table(data: Any, headers: Any, **kwargs: Any) -> None:
    """Print a tabulate table at info level.

    Shown when log level is INFO. Silenced at WARNING or SILENT.
    """
    formatted = _tabulate(data, headers=headers, **kwargs)
    if _level >= LogLevel.INFO:
        print(formatted)
    _write_to_file(formatted)
