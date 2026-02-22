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

import multiprocessing
import pathlib
import warnings
from datetime import datetime
from enum import IntEnum
from typing import Any

from tabulate import tabulate as _tabulate


class LogLevel(IntEnum):
    """Controls which messages are shown in the terminal."""

    SILENT = 0
    WARNING = 1
    INFO = 2


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
    # Spawned worker processes re-import the main module — skip to avoid wiping the log
    if multiprocessing.current_process().name != "MainProcess":
        return
    global _log_file_path  # noqa: PLW0603
    _log_file_path = pathlib.Path(path)
    if mode == "w":
        _log_file_path.write_text("", encoding="utf-8")
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
    timestamp = datetime.now(tz=datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    with _log_file_path.open("a", encoding="utf-8") as f:
        f.write(f"--- [{timestamp}] [{level}] ---\n")
        f.write(stripped + "\n")


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
