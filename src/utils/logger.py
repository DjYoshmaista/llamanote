# llamanote/utils/logger.py
"""
Logging Configuration and Utilities Module
Provides centralized logging setup, context logging, console output, and progress tracking.
"""

import logging
import logging.config
import sys
import time
import json
import traceback
from pathlib import Path
from typing import Optional, Any, Dict, List
from datetime import datetime

# Import logging config function from settings
from ..config.settings import get_logging_config, DEFAULT_LOG_DIR, SAVE_ERROR_CONTEXT

# Global flag to track if logging has been configured
_logging_configured = False

def setup_logging(log_level: int = logging.INFO, log_dir: Optional[Path] = None):
    """Configures logging for the application."""
    global _logging_configured
    if _logging_configured:
        # Update level if already configured
        logging.getLogger("llamanote").setLevel(log_level)
        return

    effective_log_dir = Path(log_dir or DEFAULT_LOG_DIR).resolve()
    try:
        effective_log_dir.mkdir(parents=True, exist_ok=True)
        logging_config = get_logging_config(effective_log_dir)

        # Apply the desired log level to the main 'llamanote' logger handlers
        for handler_name in logging_config.get("loggers", {}).get("llamanote", {}).get("handlers", []):
            if handler_name in logging_config.get("handlers", {}):
                 # Set console handler level directly, keep file handler at DEBUG
                 if handler_name == "console":
                     logging_config["handlers"][handler_name]["level"] = logging.getLevelName(log_level)
                 # else: keep file/error handlers at their configured levels (DEBUG/ERROR)

        # Apply config
        logging.config.dictConfig(logging_config)
        # Ensure the root logger's level doesn't suppress messages lower than INFO for console
        logging.getLogger().handlers[0].setLevel(logging.getLevelName(log_level)) # Assuming console is root's first handler

        _logging_configured = True
        logger = logging.getLogger("llamanote.logger_setup") # Use a specific logger
        logger.info(f"Logging configured. Level: {logging.getLevelName(log_level)}. Log directory: {effective_log_dir}")
    except Exception as e:
        print(f"FATAL: Failed to configure logging: {e}", file=sys.stderr)
        # Fallback to basic config
        logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
        logging.error(f"Logging setup failed, using basic config.", exc_info=True)


class ContextLogger:
    """Enhanced logger with context tracking."""
    def __init__(self, name: str):
        # Ensure base logger name includes 'llamanote' prefix
        if not name.startswith("llamanote."):
            name = f"llamanote.{name}"
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}

    def set_context(self, **kwargs):
        """Set context variables for subsequent log messages."""
        self.context.update(kwargs)

    def clear_context(self):
        """Clear context variables."""
        self.context.clear()

    def _format_message(self, message: str) -> str:
        """Adds context to the log message if present."""
        if self.context:
            context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{message} [Context: {context_str}]"
        return message

    def debug(self, message: str, **kwargs): self.logger.debug(self._format_message(message), **kwargs)
    def info(self, message: str, **kwargs): self.logger.info(self._format_message(message), **kwargs)
    def warning(self, message: str, **kwargs): self.logger.warning(self._format_message(message), **kwargs)

    def error(self, message: str, exc_info=False, save_context: bool = True, **kwargs):
        """Logs an error, optionally saving context."""
        formatted_message = self._format_message(message)
        self.logger.error(formatted_message, exc_info=exc_info, **kwargs)
        if save_context and SAVE_ERROR_CONTEXT:
            self._save_error_context(message, include_traceback=exc_info)

    def critical(self, message: str, exc_info=True, save_context: bool = True, **kwargs):
        """Logs a critical error, saving context."""
        formatted_message = self._format_message(message)
        self.logger.critical(formatted_message, exc_info=exc_info, **kwargs)
        if save_context and SAVE_ERROR_CONTEXT:
            self._save_error_context(message, include_traceback=exc_info)

    def _save_error_context(self, message: str, include_traceback: bool):
        """Saves current context and optional traceback to a JSON file."""
        try:
            # Need to get log_dir reliably. Assume it's configured.
            # This is slightly tricky as the logger doesn't own the log_dir path.
            # We might need ConfigManager here, or pass log_dir during setup.
            # For simplicity, let's assume DEFAULT_LOG_DIR is accessible and created.
            error_log_dir = Path(DEFAULT_LOG_DIR).resolve() # Use default path for error context
            error_log_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            error_file = error_log_dir / f"error_context_{timestamp}.json"

            error_data = {
                "timestamp": timestamp,
                "message": message,
                "logger_name": self.logger.name,
                "context": self.context,
                "traceback": traceback.format_exc() if include_traceback else None,
            }

            with open(error_file, 'w') as f:
                # Use default=str for potentially non-serializable context items
                json.dump(error_data, f, indent=2, default=str)

            self.logger.debug(f"Error context saved to {error_file}")

        except Exception as e:
            # Use the base logger to report failure to save context
            logging.getLogger("llamanote.logger_setup").error(f"Failed to save error context: {e}")


def get_logger_conf(name: str) -> ContextLogger:
    """Factory function to get a ContextLogger instance."""
    # Ensure logging is configured before returning a logger
    if not _logging_configured:
        setup_logging() # Setup with default level if not already done
    return ContextLogger(name)


class ConsoleOutput:
    """Utilities for formatted console output using ANSI codes."""
    # ANSI color codes
    HEADER = '\033[95m'; OKBLUE = '\033[94m'; OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'; WARNING = '\033[93m'; FAIL = '\033[91m'
    ENDC = '\033[0m'; BOLD = '\033[1m'; UNDERLINE = '\033[4m'

    @staticmethod
    def _is_color_supported() -> bool:
        """Check if the terminal supports ANSI color codes."""
        # Simple check for TTY and platform
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty() and sys.platform != 'win32'

    @classmethod
    def _colorize(cls, text: str, *codes: str) -> str:
        """Applies ANSI codes if supported."""
        if cls._is_color_supported():
            return "".join(codes) + text + cls.ENDC
        return text

    @classmethod
    def header(cls, text: str, width: int = 80):
        separator = "=" * width
        print(f"\n{cls._colorize(separator, cls.BOLD, cls.HEADER)}")
        print(f"{cls._colorize(text.center(width), cls.BOLD, cls.HEADER)}")
        print(f"{cls._colorize(separator, cls.BOLD, cls.HEADER)}")

    @classmethod
    def section(cls, text: str, width: int = 60):
        separator = "-" * width
        print(f"\n{cls._colorize(separator, cls.BOLD, cls.OKCYAN)}")
        print(f"{cls._colorize(text, cls.BOLD, cls.OKCYAN)}")
        print(f"{cls._colorize(separator, cls.BOLD, cls.OKCYAN)}")

    @classmethod
    def subsection(cls, text: str):
        print(f"\n{cls._colorize(text, cls.BOLD)}")

    @classmethod
    def progress_bar(cls, current: int, total: int, width: int = 40, prefix: str = "Progress"):
        if total <= 0: return
        percentage = min(1.0, current / total)
        filled = int(width * percentage)
        bar = "█" * filled + "░" * (width - filled)
        percent_str = f"{percentage:.1%}"
        count_str = f"({current}/{total})" if total > 0 else f"({current})"
        # Use carriage return \r to overwrite the line
        print(f"\r{cls._colorize(prefix, cls.OKBLUE)}: |{bar}| {percent_str} {count_str}{cls.ENDC if cls._is_color_supported() else ''}", end="", flush=True)
        if current >= total:
            print() # New line when complete

    @classmethod
    def success(cls, message: str): print(f"{cls._colorize('✅ ' + message, cls.OKGREEN)}")
    @classmethod
    def warning(cls, message: str): print(f"{cls._colorize('⚠️  ' + message, cls.WARNING)}")
    @classmethod
    def error(cls, message: str): print(f"{cls._colorize('❌ ' + message, cls.FAIL)}")
    @classmethod
    def info(cls, message: str): print(f"{cls._colorize('ℹ️  ' + message, cls.OKCYAN)}")


class LoggingProgress:
    """Context manager for logging progress of long operations."""
    def __init__(self, logger: ContextLogger, operation: str, total: Optional[int] = None, log_interval: int = 1):
        self.logger = logger
        self.operation = operation
        self.total = total
        self.current = 0
        self.start_time = None
        self.log_interval = max(1, log_interval) # Log at least every item if interval=0
        self.last_log_time = 0

    def __enter__(self):
        self.start_time = time.time()
        total_str = f" (total: {self.total})" if self.total is not None else ""
        self.logger.info(f"Starting {self.operation}{total_str}")
        self.last_log_time = self.start_time
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        count_str = f" ({self.current}/{self.total} items)" if self.total is not None else f" ({self.current} items)"

        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {elapsed:.3f}s{count_str}")
        else:
            self.logger.error(f"Failed {self.operation} after {elapsed:.3f}s at item {self.current+1}: {exc_val}", exc_info=False) # Keep exc_info False here

    def update(self, increment: int = 1, message: Optional[str] = None):
        """Update progress counter and log periodically."""
        self.current += increment
        now = time.time()

        # Log based on interval (either item count or time)
        should_log = False
        if self.total is not None:
             # Log every N items or if it's the last item
             if self.current % self.log_interval == 0 or self.current == self.total:
                 should_log = True
        elif now - self.last_log_time > 5: # Log every 5 seconds if no total
             should_log = True

        if should_log:
            progress_msg = f"{self.operation}: "
            if self.total is not None:
                percentage = (self.current / self.total) * 100
                progress_msg += f"{self.current}/{self.total} ({percentage:.1f}%)"
            else:
                progress_msg += f"{self.current} items processed"

            if message:
                progress_msg += f" - {message}"

            self.logger.debug(progress_msg)
            self.last_log_time = now

            # Also update console progress bar if total is known
            if self.total is not None:
                 ConsoleOutput.progress_bar(self.current, self.total, prefix=self.operation)
