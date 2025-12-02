"""
Custom Logging Handlers for LlamaNote

Provides specialized handlers for:
- systemd/journalctl integration
- Per-module rotating file handlers
- Async/buffered logging
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Try to import systemd journal support
try:
    from systemd import journal
    SYSTEMD_AVAILABLE = True
except ImportError:
    SYSTEMD_AVAILABLE = False
    journal = None


class SystemdJournalHandler(logging.Handler):
    """
    Handler that sends logs to systemd's journal.

    Adds structured fields for better querying:
    - LOGGER_NAME: Full hierarchical logger name
    - FILE_PATH: Source file path
    - LINE_NUMBER: Line number
    - FUNCTION_NAME: Function name
    - SYSLOG_IDENTIFIER: Always "llamanote"
    """

    # Map Python logging levels to syslog priority levels
    LEVEL_MAP = {
        logging.CRITICAL: journal.LOG_CRIT if SYSTEMD_AVAILABLE else 2,
        logging.ERROR: journal.LOG_ERR if SYSTEMD_AVAILABLE else 3,
        logging.WARNING: journal.LOG_WARNING if SYSTEMD_AVAILABLE else 4,
        logging.INFO: journal.LOG_INFO if SYSTEMD_AVAILABLE else 6,
        logging.DEBUG: journal.LOG_DEBUG if SYSTEMD_AVAILABLE else 7,
    }

    def __init__(self, identifier: str = "llamanote"):
        """
        Initialize the journal handler.

        Args:
            identifier: Syslog identifier for filtering (default: "llamanote")
        """
        super().__init__()
        self.identifier = identifier

        if not SYSTEMD_AVAILABLE:
            # Fallback: log warning and use NullHandler behavior
            print(
                "WARNING: systemd-python not available. "
                "Journal logging disabled. Install with: pip install systemd-python",
                file=sys.stderr
            )

    def emit(self, record: logging.LogRecord):
        """Send log record to systemd journal."""
        if not SYSTEMD_AVAILABLE:
            return  # Silently skip if systemd not available

        try:
            # Build structured fields
            fields = {
                "MESSAGE": self.format(record),
                "PRIORITY": self.LEVEL_MAP.get(record.levelno, journal.LOG_INFO),
                "LOGGER_NAME": record.name,
                "FILE_PATH": record.pathname,
                "LINE_NUMBER": record.lineno,
                "FUNCTION_NAME": record.funcName,
                "SYSLOG_IDENTIFIER": self.identifier,
            }

            # Add exception info if present
            if record.exc_info:
                fields["EXCEPTION_INFO"] = self.formatter.formatException(record.exc_info)

            # Send to journal
            journal.send(**fields)

        except Exception as e:
            # Fallback to stderr if journal send fails
            print(f"ERROR: Failed to send log to systemd journal: {e}", file=sys.stderr)
            self.handleError(record)


class SmartRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Enhanced RotatingFileHandler with per-module file management.

    Features:
    - Automatic directory creation
    - Better error handling
    - UTF-8 encoding by default
    - Per-module log files based on logger name
    """

    def __init__(
        self,
        base_dir: Path,
        logger_name: str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 3,
        encoding: str = "utf-8"
    ):
        """
        Initialize rotating file handler for a specific logger.

        Args:
            base_dir: Base directory for log files (e.g., "logs/")
            logger_name: Full hierarchical logger name (e.g., "llamanote.src.core.pipeline")
            max_bytes: Maximum size per log file (default: 10MB)
            backup_count: Number of backup files to keep (default: 3)
            encoding: File encoding (default: utf-8)
        """
        self.base_dir = Path(base_dir)
        self.logger_name = logger_name

        # Create base directory if it doesn't exist
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Failed to create log directory {self.base_dir}: {e}", file=sys.stderr)
            raise

        # Generate log file path: logs/llamanote.src.core.pipeline.log
        log_filename = self.base_dir / f"{logger_name}.log"

        super().__init__(
            filename=str(log_filename),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding
        )

    def emit(self, record: logging.LogRecord):
        """Emit a log record with enhanced error handling."""
        try:
            super().emit(record)
        except Exception as e:
            # Fallback to stderr if file write fails
            print(
                f"ERROR: Failed to write log to {self.baseFilename}: {e}",
                file=sys.stderr
            )
            # Try to log the original message to stderr
            try:
                print(f"[{record.levelname}] {record.name}: {record.getMessage()}", file=sys.stderr)
            except:
                pass


class BufferedHandler(logging.handlers.MemoryHandler):
    """
    Buffered logging handler for high-performance scenarios.

    Collects log records in memory and flushes them in batches
    to reduce I/O overhead.
    """

    def __init__(
        self,
        capacity: int = 100,
        flush_level: int = logging.ERROR,
        target: Optional[logging.Handler] = None
    ):
        """
        Initialize buffered handler.

        Args:
            capacity: Number of records to buffer before auto-flush
            flush_level: Log level that triggers immediate flush (default: ERROR)
            target: Target handler to flush to
        """
        super().__init__(
            capacity=capacity,
            flushLevel=flush_level,
            target=target
        )

    def shouldFlush(self, record: logging.LogRecord) -> bool:
        """Determine if buffer should be flushed."""
        # Flush on capacity or high-priority logs
        return (
            len(self.buffer) >= self.capacity
            or record.levelno >= self.flushLevel
        )


def create_file_handler(
    base_dir: Path,
    logger_name: str,
    level: int = logging.DEBUG,
    formatter: Optional[logging.Formatter] = None
) -> SmartRotatingFileHandler:
    """
    Factory function to create a configured file handler.

    Args:
        base_dir: Base log directory
        logger_name: Logger name for file naming
        level: Minimum log level (default: DEBUG)
        formatter: Log formatter (optional)

    Returns:
        Configured SmartRotatingFileHandler
    """
    handler = SmartRotatingFileHandler(
        base_dir=base_dir,
        logger_name=logger_name,
        max_bytes=10 * 1024 * 1024,  # 10MB
        backup_count=3
    )
    handler.setLevel(level)

    if formatter:
        handler.setFormatter(formatter)

    return handler


def create_console_handler(
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None,
    stream = None
) -> logging.StreamHandler:
    """
    Factory function to create a configured console handler.

    Args:
        level: Minimum log level (default: INFO)
        formatter: Log formatter (optional)
        stream: Output stream (default: sys.stdout)

    Returns:
        Configured StreamHandler
    """
    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setLevel(level)

    if formatter:
        handler.setFormatter(formatter)

    return handler


def create_journal_handler(
    level: int = logging.ERROR,
    formatter: Optional[logging.Formatter] = None,
    identifier: str = "llamanote"
) -> SystemdJournalHandler:
    """
    Factory function to create a configured systemd journal handler.

    Args:
        level: Minimum log level (default: ERROR)
        formatter: Log formatter (optional)
        identifier: Syslog identifier (default: "llamanote")

    Returns:
        Configured SystemdJournalHandler
    """
    handler = SystemdJournalHandler(identifier=identifier)
    handler.setLevel(level)

    if formatter:
        handler.setFormatter(formatter)

    return handler
