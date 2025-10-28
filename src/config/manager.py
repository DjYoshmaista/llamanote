# llamanote/config/manager.py
"""
Configuration Manager Module for LlamaNote Enhanced
Manages loading/saving various configuration types (presets, pipelines, etc.)
and delegates cloud API key management.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import shutil

from src.utils.logger import get_logger_conf
from src.settings import BASE_DIR, DEFAULT_MODEL_KEY # Use central settings
from src.cloud_keys import CloudKeyManager # Import the dedicated key manager

logger = get_logger_conf(__name__)

# --- Helper ---
def _ensure_directory(dir_path: Path):
    """Ensures a directory exists."""
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Could not create or access directory '{dir_path}': {e}")
        raise # Critical if config dirs can't be made

# --- Base CRUD Class (Now defined here for simplicity or could be in its own file) ---
class ConfigCRUD:
    """Base class for managing JSON configuration files in a specific directory."""
    def __init__(self, dir_path: Path, config_type: str):
        self.dir_path = dir_path
        self.config_type = config_type
        _ensure_directory(self.dir_path)

    def _get_config_path(self, name: str) -> Path:
        # Basic sanitization, replace spaces, remove unsafe chars
        safe_name = name.replace(" ", "_")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
        if not safe_name: safe_name = "unnamed_config" # Fallback
        return self.dir_path / f"{safe_name}.json"

    def save(self, name: str, config: Dict[str, Any]) -> bool:
        """Saves a config dictionary to a JSON file."""
        config_file = self._get_config_path(name)
        try:
            config["_metadata"] = {"saved_name": name, "created": datetime.now().isoformat(), "type": self.config_type}
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, sort_keys=True)
            logger.info(f"Saved {self.config_type} config: {name} to {config_file.name}")
            return True
        except TypeError as e:
             logger.error(f"Failed to serialize {self.config_type} config '{name}' to JSON: {e}. Check for non-serializable types.")
             return False
        except Exception as e:
            logger.error(f"Failed to save {self.config_type} config '{name}': {e}", exc_info=True)
            return False

    def load(self, name: str) -> Optional[Dict[str, Any]]:
        """Loads a config dictionary from a JSON file."""
        config_file = self._get_config_path(name)
        if not config_file.exists():
             # Try finding based on original name if sanitization changed it
             possible_original_file = self.dir_path / f"{name}.json"
             if possible_original_file.exists():
                 config_file = possible_original_file
             else:
                 logger.debug(f"{self.config_type.capitalize()} config '{name}' not found at {config_file}")
                 return None
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            config.pop("_metadata", None) # Remove internal metadata
            logger.info(f"Loaded {self.config_type} config: {name} from {config_file.name}")
            return config
        except json.JSONDecodeError as e:
             logger.error(f"Failed to parse {self.config_type} config '{name}' ({config_file.name}): {e}")
             return None
        except Exception as e:
            logger.error(f"Failed to load {self.config_type} config '{name}' ({config_file.name}): {e}", exc_info=True)
            return None

    def list(self) -> List[str]:
        """Lists available config names (attempts to read saved_name from metadata)."""
        names = []
        for f in self.dir_path.glob("*.json"):
             try:
                 with open(f, 'r') as fh:
                      data = json.load(fh)
                 # Use saved name if available, otherwise fallback to stem
                 original_name = data.get("_metadata", {}).get("saved_name", f.stem)
                 names.append(original_name)
             except Exception:
                  names.append(f.stem) # Fallback to filename stem on error
        return sorted(list(set(names))) # Use set to remove potential duplicates if sanitization clashed

    def delete(self, name: str) -> bool:
        """Deletes a config file."""
        config_file = self._get_config_path(name)
        # Also check original name path in case sanitization happened
        possible_original_file = self.dir_path / f"{name}.json"

        deleted = False
        for file_to_delete in [config_file, possible_original_file]:
             if file_to_delete.exists():
                 try:
                     file_to_delete.unlink()
                     logger.info(f"Deleted {self.config_type} config file: {file_to_delete.name} (requested name: {name})")
                     deleted = True
                     # Don't break, might need to delete both if names clash weirdly
                 except Exception as e:
                     logger.error(f"Failed to delete {self.config_type} config file '{file_to_delete.name}': {e}")
                     # Continue trying other path if exists

        if not deleted:
             logger.warning(f"{self.config_type.capitalize()} config '{name}' not found for deletion.")

        return deleted


# --- Main ConfigManager ---
class ConfigManager:
    """Manages application configurations and settings."""
    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        Initializes the ConfigManager.

        Args:
            base_dir: Optional path to the root configuration directory.
                      Defaults to ~/.config/llamanote.
        """
        self.base_dir = Path(base_dir or Path.home() / ".config" / "llamanote").resolve()
        _ensure_directory(self.base_dir)

        # Config Directories Map
        self.dir_map = {
            "config": self.base_dir / "config",          # General app settings
            "filter": self.base_dir / "filters",        # Filter presets
            "preset": self.base_dir / "presets",        # Hyperparameter presets
            "pipeline": self.base_dir / "pipelines",    # Saved pipeline configurations
            "audio": self.base_dir / "audio_configs",   # Saved audio configurations
            "model": self.base_dir / "model_prefs",     # Model preferences (future use)
            "cloud": self.base_dir / "cloud",          # Cloud API keys storage
            "backup": self.base_dir / "backups",       # Backup location
            # Custom paths config lives within 'config' directory
        }
        # Ensure base directories exist
        for dir_type in self.dir_map: _ensure_directory(self.get_dir(dir_type))


        # CRUD Handlers
        self.configs = ConfigCRUD(self.get_dir("config"), "config")
        self.filters = ConfigCRUD(self.get_dir("filter"), "filter")
        self.presets = ConfigCRUD(self.get_dir("preset"), "preset") # For hyperparams
        self.pipelines = ConfigCRUD(self.get_dir("pipeline"), "pipeline")
        self.audio_configs = ConfigCRUD(self.get_dir("audio"), "audio")
        self.model_prefs = ConfigCRUD(self.get_dir("model"), "model")

        # Cloud Keys Manager
        self.cloud_keys = CloudKeyManager(self.get_dir("cloud") / CloudKeyManager.DEFAULT_FILENAME)

        # Custom Paths Handling (uses self.configs CRUD handler)
        self.custom_paths = self._load_custom_paths()

        logger.info(f"Initialized ConfigManager. Base directory: {self.base_dir}")

    # --- Directory Access ---
    def get_dir(self, dir_type: str) -> Path:
        """Gets the path for a specific configuration directory type."""
        # Custom paths override defaults
        path_str = self.custom_paths.get(dir_type)
        if path_str:
            dir_path = Path(path_str).resolve() # Resolve custom paths immediately
        else:
            # Use the predefined map, fallback to a subdirectory in base_dir
            dir_path = self.dir_map.get(dir_type, self.base_dir / dir_type)

        _ensure_directory(dir_path) # Ensure it exists whenever accessed
        return dir_path

    # --- Custom Path Methods ---
    def _load_custom_paths(self) -> Dict[str, str]:
        """Loads custom path configurations."""
        paths_data = self.configs.load("paths")
        # Validate paths (optional: check if they are directories?)
        valid_paths = {}
        if paths_data:
             for k, v in paths_data.items():
                  if isinstance(k, str) and isinstance(v, str):
                       valid_paths[k] = v # Store as string
                  else:
                       logger.warning(f"Ignoring invalid entry in paths.json: {k}={v}")
        return valid_paths

    def set_custom_path(self, name: str, path: Union[str, Path]):
        """Sets and saves a custom path."""
        try:
             # Basic validation: ensure name is known or log warning?
             path_str = str(Path(path).resolve()) # Store resolved absolute path string
             self.custom_paths[name] = path_str
             self.configs.save("paths", self.custom_paths)
             logger.info(f"Set custom path '{name}' to '{path_str}'")
        except Exception as e:
             logger.error(f"Failed to set custom path '{name}' to '{path}': {e}", exc_info=True)


    # --- Cloud Key Passthrough Methods ---
    def load_cloud_keys(self) -> Dict[str, str]:
        return self.cloud_keys.load_active_keys()

    def save_cloud_key(self, provider: str, key: Optional[str]) -> bool:
        return self.cloud_keys.save_key(provider, key)

    def get_configured_providers(self) -> List[str]:
        return self.cloud_keys.get_configured_providers()

    # --- Generic Config Passthrough Methods ---
    def save_config(self, name: str, config: Dict[str, Any], dir_type: str = "config") -> bool:
        handler = getattr(self, f"{dir_type}s", self.configs) # e.g., self.pipelines
        return handler.save(name, config)

    def load_config(self, name: str, dir_type: str = "config") -> Optional[Dict[str, Any]]:
        handler = getattr(self, f"{dir_type}s", self.configs)
        return handler.load(name)

    def list_configs(self, dir_type: str = "config") -> List[str]:
        handler = getattr(self, f"{dir_type}s", self.configs)
        return handler.list()

    def delete_config(self, name: str, dir_type: str = "config") -> bool:
        handler = getattr(self, f"{dir_type}s", self.configs)
        return handler.delete(name)

    # --- Default Config ---
    def get_default_config(self) -> Dict[str, Any]:
        """Gets default config, falling back to hardcoded."""
        loaded = self.configs.load("default")
        if loaded: return loaded
        logger.warning("Default config file not found/loaded. Using hardcoded defaults.")
        # Minimal hardcoded defaults
        return {"mode": "podcast", "model_provider": "local_hf", "model_specifier": DEFAULT_MODEL_KEY}

    def set_default_config(self, config: Dict[str, Any]) -> bool:
        return self.configs.save("default", config)

    # --- Backup/Restore (Simplified using helper) ---
    def _iterate_backup_dirs(self, operation_fn):
        """Helper to iterate over relevant directories for backup/restore."""
        # Define which dir_types are included in backup/restore
        backup_dir_types = ["config", "filter", "preset", "pipeline", "audio", "model", "cloud"]
        for dir_type in backup_dir_types:
            dir_path = self.get_dir(dir_type) # Use get_dir to handle custom paths
            if dir_path.exists() and dir_path.is_dir():
                operation_fn(dir_path, dir_type) # Pass path and type name
            else:
                logger.debug(f"Skipping non-existent directory during backup/restore: {dir_path}")

    def backup_configs(self, backup_name: Optional[str] = None) -> Optional[Path]:
        """Backs up configuration directories."""
        try:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = backup_name or f"backup_{ts}"
            backup_root = self.get_dir("backup") / backup_name # Use get_dir for backup location
            _ensure_directory(backup_root)
            logger.info(f"Starting backup to {backup_root}...")

            def copy_dir(src_dir, dir_type):
                dst_dir = backup_root / dir_type
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                logger.debug(f"Backed up {dir_type} from {src_dir} to {dst_dir}")

            self._iterate_backup_dirs(copy_dir)
            logger.info(f"Configuration backup completed: {backup_root}")
            return backup_root
        except Exception as e:
            logger.error(f"Failed to create backup '{backup_name}': {e}", exc_info=True)
            return None

    def restore_backup(self, backup_name: str) -> bool:
        """Restores configuration from a backup."""
        backup_root = self.get_dir("backup") / backup_name
        if not backup_root.is_dir():
            logger.error(f"Backup directory not found: {backup_root}")
            return False
        try:
            logger.warning(f"Starting restore from backup: {backup_name}. This will overwrite current configs.")
            # Simple confirmation - enhance if needed
            confirm = input("Are you sure you want to restore? (yes/no): ").strip().lower()
            if confirm != 'yes':
                 logger.info("Restore cancelled by user.")
                 return False

            def restore_dir(src_dir_in_backup, dir_type):
                # Need src_dir relative to backup_root
                src_dir_for_copy = backup_root / dir_type
                dst_dir = self.get_dir(dir_type) # Get the live config directory
                # Optional: Clear destination dir first? Risky. Overwrite is safer.
                # if dst_dir.exists(): shutil.rmtree(dst_dir)
                shutil.copytree(src_dir_for_copy, dst_dir, dirs_exist_ok=True)
                logger.debug(f"Restored {dir_type} from {src_dir_for_copy} to {dst_dir}")

            self._iterate_backup_dirs(restore_dir)

            # Reload custom paths after potential restore
            self.custom_paths = self._load_custom_paths()

            logger.info(f"Successfully restored configuration from backup: {backup_name}")
            logger.warning("Application restart might be needed for all changes to take effect.")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup '{backup_name}': {e}", exc_info=True)
            return False

    def list_backups(self) -> List[str]:
        """Lists available backups."""
        backup_base = self.get_dir("backup")
        if not backup_base.exists(): return []
        return sorted([d.name for d in backup_base.iterdir() if d.is_dir()], reverse=True)
