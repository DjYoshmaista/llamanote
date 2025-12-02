"""
LlamaNote Hierarchical Logging System

A self-contained, reusable logging module with:
- Hierarchical logger naming based on file/module structure
- Per-module log level configuration
- Rotating file handlers with automatic cleanup
- systemd/journalctl integration
- Interactive configuration menu
- Persistent configuration storage

## Quick Start

```python
from src.logging_config import get_logger

# Get a logger for the current module
logger = get_logger(__name__)

# Use it
logger.debug("Detailed debugging info")
logger.info("General information")
logger.warning("Something unexpected")
logger.error("An error occurred", exc_info=True)
logger.critical("Critical failure", exc_info=True)
```

## Advanced Usage

```python
from src.logging_config import (
    get_logger,
    get_config_manager,
    LogLevel,
    reconfigure_logger
)

# Get logger
logger = get_logger(__name__)

# Get configuration manager
config_mgr = get_config_manager()

# Change log level for a specific module
config_mgr.set_module_log_level(
    "llamanote.src.core.pipeline",
    console_level=LogLevel.DEBUG
)

# Reconfigure logger to apply changes
reconfigure_logger("llamanote.src.core.pipeline")

# Clear all log files
config_mgr.clear_log_files()
```

## Module Structure

- `logger_factory.py`: Hierarchical logger creation
- `config.py`: Configuration management
- `handlers.py`: Custom handlers (systemd, rotating file)
- `formatter.py`: Custom log formatters
- `menu.py`: Interactive configuration menu

## Configuration

Configuration is stored in `.logging_config.json` in the project root.

Example configuration:
```json
{
  "version": "1.0",
  "global_level": "INFO",
  "log_dir": "logs",
  "rotation_max_bytes": 10485760,
  "rotation_backup_count": 3,
  "console_enabled": true,
  "file_enabled": true,
  "journal_enabled": true,
  "module_configs": {
    "llamanote.src.core.pipeline": {
      "console_level": "WARNING",
      "file_level": "DEBUG",
      "journal_level": "ERROR"
    }
  }
}
```

## See Also

- `documentation/LOGGING_RULES.md`: Complete specification and rules
- `CLAUDE.md`: Integration with project workflow
"""

__version__ = "1.0.0"
__author__ = "LlamaNote Project"

# Public API
from .logger_factory import (
    get_logger,
    create_logger,
    reconfigure_logger,
    reconfigure_all_loggers,
    get_all_loggers,
    shutdown_logging,
    ROOT_LOGGER_NAME
)

from .config import (
    get_config_manager,
    ConfigManager,
    LoggingConfig,
    ModuleConfig,
    LogLevel
)

from .formatter import get_formatter

from .handlers import (
    SystemdJournalHandler,
    SmartRotatingFileHandler,
    create_file_handler,
    create_console_handler,
    create_journal_handler,
    SYSTEMD_AVAILABLE
)

__all__ = [
    # Logger factory
    "get_logger",
    "create_logger",
    "reconfigure_logger",
    "reconfigure_all_loggers",
    "get_all_loggers",
    "shutdown_logging",
    "ROOT_LOGGER_NAME",

    # Configuration
    "get_config_manager",
    "ConfigManager",
    "LoggingConfig",
    "ModuleConfig",
    "LogLevel",

    # Formatters
    "get_formatter",

    # Handlers
    "SystemdJournalHandler",
    "SmartRotatingFileHandler",
    "create_file_handler",
    "create_console_handler",
    "create_journal_handler",
    "SYSTEMD_AVAILABLE",
]


# Initialize logging on module import
def _initialize():
    """Initialize the logging system on module import."""
    # Create log directory
    config_mgr = get_config_manager()
    try:
        config_mgr.config.log_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        import sys
        print(f"WARNING: Failed to create log directory: {e}", file=sys.stderr)


# Run initialization
_initialize()
