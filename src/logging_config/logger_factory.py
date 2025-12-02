"""
Hierarchical Logger Factory for LlamaNote

Automatically creates loggers with:
- Hierarchical naming based on file path
- Appropriate handlers based on configuration
- Per-module log level configuration
"""

import logging
import inspect
import sys
from pathlib import Path
from typing import Optional, Dict

from .config import get_config_manager, LogLevel, ModuleConfig
from .handlers import (
    create_file_handler,
    create_console_handler,
    create_journal_handler
)
from .formatter import get_formatter


# Global registry of created loggers
_logger_registry: Dict[str, logging.Logger] = {}

# Root logger name
ROOT_LOGGER_NAME = "llamanote"


def _detect_module_name(calling_frame=None) -> str:
    """
    Detect the module name from the calling context.

    Args:
        calling_frame: Optional frame to inspect (default: caller's caller)

    Returns:
        Full hierarchical module name (e.g., "llamanote.src.core.pipeline")
    """
    if calling_frame is None:
        # Get the frame of the caller's caller (skip this function and get_logger)
        calling_frame = inspect.currentframe().f_back.f_back

    # Get the file path from the frame
    frame_info = inspect.getframeinfo(calling_frame)
    file_path = Path(frame_info.filename).resolve()

    # Get project root (assuming this file is in src/logging_config/)
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
    except:
        # Fallback to current working directory
        project_root = Path.cwd()

    # Try to get relative path from project root
    try:
        rel_path = file_path.relative_to(project_root)
    except ValueError:
        # File is outside project root, use absolute path
        rel_path = file_path

    # Convert path to module name
    # Remove .py extension
    parts = list(rel_path.parts)
    if parts[-1].endswith('.py'):
        parts[-1] = parts[-1][:-3]

    # Handle special cases
    if parts[-1] == '__init__':
        parts = parts[:-1]  # Remove __init__, use package name

    # Replace main.py or llamanote.py with "main"
    if parts[-1] in ['main', 'llamanote']:
        parts[-1] = 'main'

    # Build hierarchical name
    module_parts = [ROOT_LOGGER_NAME] + [p for p in parts if p != '.' and p != '..']
    module_name = '.'.join(module_parts)

    return module_name


def _normalize_module_name(name: str) -> str:
    """
    Normalize a module name to hierarchical format.

    Args:
        name: Module name (e.g., "__name__" variable, "src.core.pipeline", etc.)

    Returns:
        Normalized hierarchical name (e.g., "llamanote.src.core.pipeline")
    """
    # If already starts with root, return as-is
    if name.startswith(ROOT_LOGGER_NAME + "."):
        return name

    # If it's __main__, convert to llamanote.main
    if name == "__main__":
        return f"{ROOT_LOGGER_NAME}.main"

    # If it starts with src. or another common pattern
    if name.startswith("src."):
        return f"{ROOT_LOGGER_NAME}.{name}"

    # If it's a simple name without dots, assume it's a top-level module
    if "." not in name:
        return f"{ROOT_LOGGER_NAME}.{name}"

    # Otherwise, prepend root logger name
    return f"{ROOT_LOGGER_NAME}.{name}"


def create_logger(module_name: str) -> logging.Logger:
    """
    Create a logger with hierarchical naming and appropriate handlers.

    Args:
        module_name: Full hierarchical module name

    Returns:
        Configured logging.Logger instance
    """
    # Normalize module name
    module_name = _normalize_module_name(module_name)

    # Check if logger already exists
    if module_name in _logger_registry:
        return _logger_registry[module_name]

    # Get configuration
    config_manager = get_config_manager()
    logging_config = config_manager.config
    module_config = config_manager.get_module_config(module_name)

    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)  # Set to DEBUG, handlers will filter

    # Prevent propagation to root logger (we manage our own handlers)
    logger.propagate = False

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = get_formatter("detailed")
    console_formatter = get_formatter("console")
    structured_formatter = get_formatter("structured")

    # Add file handler (if enabled)
    if logging_config.file_enabled:
        try:
            file_handler = create_file_handler(
                base_dir=logging_config.log_dir,
                logger_name=module_name,
                level=module_config.file_level.to_logging_level(),
                formatter=detailed_formatter
            )
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"WARNING: Failed to create file handler for {module_name}: {e}", file=sys.stderr)

    # Add console handler (if enabled)
    if logging_config.console_enabled:
        try:
            console_handler = create_console_handler(
                level=module_config.console_level.to_logging_level(),
                formatter=console_formatter
            )
            logger.addHandler(console_handler)
        except Exception as e:
            print(f"WARNING: Failed to create console handler for {module_name}: {e}", file=sys.stderr)

    # Add journal handler (if enabled)
    if logging_config.journal_enabled:
        try:
            journal_handler = create_journal_handler(
                level=module_config.journal_level.to_logging_level(),
                formatter=structured_formatter
            )
            logger.addHandler(journal_handler)
        except Exception as e:
            # Non-fatal, systemd might not be available
            pass

    # Register logger
    _logger_registry[module_name] = logger

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with hierarchical naming.

    This is the main entry point for acquiring loggers in application code.

    Args:
        name: Module name (usually __name__), or None for auto-detection

    Returns:
        Configured logging.Logger instance

    Examples:
        # Recommended: use __name__
        logger = get_logger(__name__)

        # Auto-detection (slower, inspects call stack)
        logger = get_logger()

        # Explicit name
        logger = get_logger("llamanote.src.core.pipeline")
    """
    if name is None:
        # Auto-detect module name from caller's context
        module_name = _detect_module_name()
    else:
        module_name = name

    return create_logger(module_name)


def reconfigure_logger(module_name: str):
    """
    Reconfigure an existing logger with updated settings.

    Useful for runtime configuration changes.

    Args:
        module_name: Module name to reconfigure
    """
    # Normalize name
    module_name = _normalize_module_name(module_name)

    # Remove from registry to force recreation
    if module_name in _logger_registry:
        del _logger_registry[module_name]

    # Recreate with current configuration
    return create_logger(module_name)


def reconfigure_all_loggers():
    """Reconfigure all loggers with updated settings."""
    logger_names = list(_logger_registry.keys())
    for logger_name in logger_names:
        reconfigure_logger(logger_name)


def get_all_loggers() -> Dict[str, logging.Logger]:
    """Get all registered loggers."""
    return _logger_registry.copy()


def shutdown_logging():
    """
    Shutdown all loggers and handlers gracefully.

    Call this before application exit to ensure all logs are flushed.
    """
    # Flush all handlers
    for logger in _logger_registry.values():
        for handler in logger.handlers:
            try:
                handler.flush()
                handler.close()
            except:
                pass

    # Clear registry
    _logger_registry.clear()

    # Shutdown logging module
    logging.shutdown()
