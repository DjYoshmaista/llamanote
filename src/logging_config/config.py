"""
Logging Configuration Management for LlamaNote

Handles:
- Per-module log level configuration
- Handler enable/disable
- Configuration persistence
- Runtime configuration updates
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict, field
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    @classmethod
    def from_string(cls, level_str: str) -> 'LogLevel':
        """Convert string to LogLevel enum."""
        try:
            return cls[level_str.upper()]
        except KeyError:
            raise ValueError(f"Invalid log level: {level_str}")

    def to_logging_level(self) -> int:
        """Convert to logging module level."""
        return self.value


@dataclass
class HandlerConfig:
    """Configuration for a logging handler."""
    enabled: bool = True
    level: LogLevel = LogLevel.DEBUG


@dataclass
class ModuleConfig:
    """Configuration for a specific module's logging."""
    console_level: LogLevel = LogLevel.INFO
    file_level: LogLevel = LogLevel.DEBUG
    journal_level: LogLevel = LogLevel.ERROR
    handlers: Dict[str, HandlerConfig] = field(default_factory=lambda: {
        "console": HandlerConfig(enabled=True, level=LogLevel.INFO),
        "file": HandlerConfig(enabled=True, level=LogLevel.DEBUG),
        "journal": HandlerConfig(enabled=True, level=LogLevel.ERROR)
    })


@dataclass
class LoggingConfig:
    """Complete logging configuration for LlamaNote."""
    version: str = "1.0"
    global_level: LogLevel = LogLevel.INFO
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    rotation_max_bytes: int = 10 * 1024 * 1024  # 10MB
    rotation_backup_count: int = 3

    # Default handler states
    console_enabled: bool = True
    file_enabled: bool = True
    journal_enabled: bool = True

    # Per-module overrides
    module_configs: Dict[str, ModuleConfig] = field(default_factory=dict)

    def get_module_config(self, module_name: str) -> ModuleConfig:
        """Get configuration for a specific module, or default."""
        return self.module_configs.get(module_name, self._get_default_module_config(module_name))

    def set_module_config(self, module_name: str, config: ModuleConfig):
        """Set configuration for a specific module."""
        self.module_configs[module_name] = config

    def _get_default_module_config(self, module_name: str) -> ModuleConfig:
        """
        Get default configuration based on module path.

        Implements the per-module rules from LOGGING_RULES.md
        """
        # Main entry points: all to console and file
        if module_name in ["llamanote.main", "llamanote.src.cli", "llamanote.src.menu"]:
            return ModuleConfig(
                console_level=LogLevel.DEBUG,
                file_level=LogLevel.DEBUG,
                journal_level=LogLevel.ERROR
            )

        # Core modules: WARN+ to console
        elif module_name.startswith("llamanote.src.core"):
            return ModuleConfig(
                console_level=LogLevel.WARNING,
                file_level=LogLevel.DEBUG,
                journal_level=LogLevel.WARNING
            )

        # Model modules: WARN+ to console, ERROR to journal
        elif module_name.startswith("llamanote.src.models"):
            return ModuleConfig(
                console_level=LogLevel.WARNING,
                file_level=LogLevel.DEBUG,
                journal_level=LogLevel.ERROR
            )

        # Processing modules: INFO+ to console
        elif module_name.startswith("llamanote.src.processing"):
            return ModuleConfig(
                console_level=LogLevel.INFO,
                file_level=LogLevel.DEBUG,
                journal_level=LogLevel.ERROR
            )

        # Utils modules: WARN+ to console
        elif module_name.startswith("llamanote.src.utils"):
            return ModuleConfig(
                console_level=LogLevel.WARNING,
                file_level=LogLevel.DEBUG,
                journal_level=LogLevel.ERROR
            )

        # All other modules: ERROR+ to console
        else:
            return ModuleConfig(
                console_level=LogLevel.ERROR,
                file_level=LogLevel.DEBUG,
                journal_level=LogLevel.ERROR
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "version": self.version,
            "global_level": self.global_level.name,
            "log_dir": str(self.log_dir),
            "rotation_max_bytes": self.rotation_max_bytes,
            "rotation_backup_count": self.rotation_backup_count,
            "console_enabled": self.console_enabled,
            "file_enabled": self.file_enabled,
            "journal_enabled": self.journal_enabled,
            "module_configs": {
                module_name: {
                    "console_level": config.console_level.name,
                    "file_level": config.file_level.name,
                    "journal_level": config.journal_level.name
                }
                for module_name, config in self.module_configs.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoggingConfig':
        """Create configuration from dictionary."""
        config = cls(
            version=data.get("version", "1.0"),
            global_level=LogLevel.from_string(data.get("global_level", "INFO")),
            log_dir=Path(data.get("log_dir", "logs")),
            rotation_max_bytes=data.get("rotation_max_bytes", 10 * 1024 * 1024),
            rotation_backup_count=data.get("rotation_backup_count", 3),
            console_enabled=data.get("console_enabled", True),
            file_enabled=data.get("file_enabled", True),
            journal_enabled=data.get("journal_enabled", True)
        )

        # Load module configs
        for module_name, module_data in data.get("module_configs", {}).items():
            module_config = ModuleConfig(
                console_level=LogLevel.from_string(module_data.get("console_level", "INFO")),
                file_level=LogLevel.from_string(module_data.get("file_level", "DEBUG")),
                journal_level=LogLevel.from_string(module_data.get("journal_level", "ERROR"))
            )
            config.module_configs[module_name] = module_config

        return config


class ConfigManager:
    """Manages logging configuration loading, saving, and runtime updates."""

    DEFAULT_CONFIG_FILE = ".logging_config.json"

    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_file: Path to configuration file (default: .logging_config.json in project root)
        """
        self.config_file = Path(config_file or self.DEFAULT_CONFIG_FILE)
        self.config = self.load_config()

    def load_config(self) -> LoggingConfig:
        """
        Load configuration from file, or return default.

        Returns:
            LoggingConfig instance
        """
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                return LoggingConfig.from_dict(data)
            except Exception as e:
                print(f"WARNING: Failed to load logging config from {self.config_file}: {e}")
                print("Using default configuration.")
                return LoggingConfig()
        else:
            return LoggingConfig()

    def save_config(self, config: Optional[LoggingConfig] = None):
        """
        Save configuration to file.

        Args:
            config: Configuration to save (default: self.config)
        """
        if config:
            self.config = config

        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config.to_dict(), f, indent=2)
        except Exception as e:
            print(f"ERROR: Failed to save logging config to {self.config_file}: {e}")

    def get_module_config(self, module_name: str) -> ModuleConfig:
        """Get configuration for a specific module."""
        return self.config.get_module_config(module_name)

    def set_module_log_level(
        self,
        module_name: str,
        console_level: Optional[LogLevel] = None,
        file_level: Optional[LogLevel] = None,
        journal_level: Optional[LogLevel] = None
    ):
        """
        Set log levels for a specific module.

        Args:
            module_name: Full module name (e.g., "llamanote.src.core.pipeline")
            console_level: Console handler level (optional)
            file_level: File handler level (optional)
            journal_level: Journal handler level (optional)
        """
        module_config = self.config.get_module_config(module_name)

        if console_level:
            module_config.console_level = console_level
        if file_level:
            module_config.file_level = file_level
        if journal_level:
            module_config.journal_level = journal_level

        self.config.set_module_config(module_name, module_config)
        self.save_config()

    def set_global_level(self, level: LogLevel):
        """Set global log level."""
        self.config.global_level = level
        self.save_config()

    def enable_handler(self, handler_type: str):
        """Enable a handler type globally."""
        if handler_type == "console":
            self.config.console_enabled = True
        elif handler_type == "file":
            self.config.file_enabled = True
        elif handler_type == "journal":
            self.config.journal_enabled = True
        self.save_config()

    def disable_handler(self, handler_type: str):
        """Disable a handler type globally."""
        if handler_type == "console":
            self.config.console_enabled = False
        elif handler_type == "file":
            self.config.file_enabled = False
        elif handler_type == "journal":
            self.config.journal_enabled = False
        self.save_config()

    def list_log_files(self) -> List[Path]:
        """List all log files in the log directory."""
        log_dir = self.config.log_dir
        if not log_dir.exists():
            return []

        return sorted(log_dir.glob("*.log*"))

    def get_log_file_size(self, log_file: Path) -> int:
        """Get size of a log file in bytes."""
        try:
            return log_file.stat().st_size
        except:
            return 0

    def clear_log_files(self, module_name: Optional[str] = None):
        """
        Clear log files.

        Args:
            module_name: If specified, clear only logs for this module.
                        If None, clear all logs.
        """
        log_dir = self.config.log_dir
        if not log_dir.exists():
            return

        if module_name:
            # Clear specific module logs
            pattern = f"{module_name}.log*"
            files = log_dir.glob(pattern)
        else:
            # Clear all logs
            files = log_dir.glob("*.log*")

        for log_file in files:
            try:
                log_file.unlink()
            except Exception as e:
                print(f"WARNING: Failed to delete {log_file}: {e}")

    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = LoggingConfig()
        self.save_config()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager
