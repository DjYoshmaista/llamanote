"""
Custom Log Formatters for LlamaNote

Provides specialized formatters for different output targets:
- Detailed formatter for file logs
- Console formatter for terminal output
- Structured formatter for systemd journal
"""

import logging
from typing import Optional


class DetailedFormatter(logging.Formatter):
    """
    Detailed formatter for file logs.

    Format: [TIMESTAMP] [LEVEL] logger_name::line.{line}::{function}() - Message

    Example:
        [2025-11-12T14:32:18.123456] [INFO] llamanote.src.core.pipeline::line.245::process_file() - Starting pipeline
    """

    def __init__(self):
        super().__init__(
            fmt="[%(asctime)s] [%(levelname)s] %(name)s::line.%(lineno)d::%(funcName)s() - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )

    def formatTime(self, record, datefmt=None):
        """Add microseconds to timestamp."""
        import datetime
        ct = datetime.datetime.fromtimestamp(record.created)
        if datefmt:
            s = ct.strftime(datefmt)
            # Append microseconds
            s = f"{s}.{int(record.msecs * 1000):06d}"
        else:
            s = ct.isoformat()
        return s


class ConsoleFormatter(logging.Formatter):
    """
    Console formatter for terminal output.

    Format: [LEVEL] logger_name::function() - Message

    Example:
        [INFO] llamanote.src.core.pipeline::process_file() - Starting pipeline

    Supports ANSI color codes if terminal supports them.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def __init__(self, use_colors: bool = True):
        """
        Initialize console formatter.

        Args:
            use_colors: Enable ANSI color codes (default: True)
        """
        super().__init__(
            fmt="[%(levelname)s] %(name)s::%(funcName)s() - %(message)s"
        )
        self.use_colors = use_colors and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        """Check if terminal supports ANSI color codes."""
        import sys
        import os

        # Check if output is a TTY
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False

        # Check platform (Windows needs special handling)
        if sys.platform == 'win32':
            # Try to enable ANSI support on Windows
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except:
                return False

        # Unix/Linux/Mac support ANSI by default
        return True

    def format(self, record):
        """Format log record with optional colors."""
        # Save original levelname
        levelname = record.levelname

        if self.use_colors:
            # Add color to level name
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        # Format the message
        result = super().format(record)

        # Restore original levelname
        record.levelname = levelname

        return result


class StructuredFormatter(logging.Formatter):
    """
    Structured formatter for systemd journal.

    This formatter is primarily used internally by SystemdJournalHandler
    to format structured fields.
    """

    def __init__(self):
        super().__init__(
            fmt="%(message)s"
        )

    def format(self, record):
        """Format message for structured logging."""
        # Basic message formatting
        message = super().format(record)

        # Add exception info if present
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        return message


class CompactFormatter(logging.Formatter):
    """
    Compact formatter for high-volume logging scenarios.

    Format: [HH:MM:SS] [L] module::func - msg

    Example:
        [14:32:18] [I] pipeline::process - Started
    """

    LEVEL_ABBREV = {
        'DEBUG': 'D',
        'INFO': 'I',
        'WARNING': 'W',
        'ERROR': 'E',
        'CRITICAL': 'C'
    }

    def __init__(self):
        # Extract last part of logger name (e.g., "pipeline" from "llamanote.src.core.pipeline")
        super().__init__(
            fmt="[%(asctime)s] [%(levelname)s] %(module)s::%(funcName)s - %(message)s",
            datefmt="%H:%M:%S"
        )

    def format(self, record):
        """Format log record in compact form."""
        # Abbreviate level name
        record.levelname = self.LEVEL_ABBREV.get(record.levelname, record.levelname[0])
        return super().format(record)


def get_formatter(formatter_type: str = "detailed", **kwargs) -> logging.Formatter:
    """
    Factory function to get a formatter by type.

    Args:
        formatter_type: Type of formatter ("detailed", "console", "structured", "compact")
        **kwargs: Additional arguments passed to formatter constructor

    Returns:
        Configured logging.Formatter instance

    Raises:
        ValueError: If formatter_type is unknown
    """
    formatters = {
        "detailed": DetailedFormatter,
        "console": ConsoleFormatter,
        "structured": StructuredFormatter,
        "compact": CompactFormatter
    }

    formatter_class = formatters.get(formatter_type.lower())
    if not formatter_class:
        raise ValueError(
            f"Unknown formatter type: {formatter_type}. "
            f"Available: {', '.join(formatters.keys())}"
        )

    return formatter_class(**kwargs)
