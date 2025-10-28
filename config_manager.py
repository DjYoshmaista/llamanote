"""
Configuration Manager Module
Manages configuration files, filters, and settings, including cloud API keys
"""

import os
import json
import yaml # Consider adding pyyaml to requirements if using YAML
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import shutil

from loggerConf import get_logger_conf

logger = get_logger_conf(__name__)

# List of known cloud providers (case-insensitive keys used internally)
KNOWN_CLOUD_PROVIDERS = [
    "openai", "anthropic", "google", "cohere", "huggingface_token",
    "deepseek", "groq", "openrouter", "qwen" # Added more
]

class ConfigManager:
    """Manages application configurations and settings"""

    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize configuration manager"""
        self.base_dir = base_dir or Path.home() / ".config" / "llamanote"

        # Setup directory structure
        self.config_dir = self.base_dir / "config"
        self.filter_dir = self.base_dir / "filters" # Renamed for clarity
        self.preset_dir = self.base_dir / "presets" # Renamed for clarity
        self.pipeline_dir = self.base_dir / "pipelines" # Renamed for clarity
        self.audio_dir = self.base_dir / "audio_configs" # Renamed for clarity
        self.model_dir = self.base_dir / "model_prefs" # Renamed for clarity
        self.cloud_config_dir = self.base_dir / "cloud" # New directory for cloud keys

        # Create directories
        for dir_path in [self.config_dir, self.filter_dir, self.preset_dir,
                         self.pipeline_dir, self.audio_dir, self.model_dir,
                         self.cloud_config_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Default config file
        self.default_config_file = self.config_dir / "default.json"

        # Custom paths configuration
        self.paths_config_file = self.config_dir / "paths.json"
        self.custom_paths = self._load_custom_paths()

        # Cloud API Keys file
        self.cloud_keys_file = self.cloud_config_dir / "api_keys.json"
        self._ensure_api_key_file_exists() # Ensure file exists with placeholders/warning

        logger.info(f"Initialized ConfigManager at {self.base_dir}")
        logger.debug(f"API Keys file: {self.cloud_keys_file}")


    def _ensure_api_key_file_exists(self):
        """Creates the API key file with placeholders and a warning if it doesn't exist."""
        if not self.cloud_keys_file.exists():
            warning_data = {
                "_WARNING": "Store API keys securely. This file stores keys in plain text. Consider environment variables or a dedicated secrets manager for production.",
            }
            # Add known providers as None initially
            for provider in KNOWN_CLOUD_PROVIDERS:
                 warning_data[provider] = None

            try:
                with open(self.cloud_keys_file, 'w') as f:
                    json.dump(warning_data, f, indent=2, sort_keys=True)
                logger.info(f"Created template API key file at {self.cloud_keys_file}")
                # Optional: Set stricter permissions
                try:
                     os.chmod(self.cloud_keys_file, 0o600) # Read/Write for owner only
                     logger.debug(f"Set permissions for API key file to 600.")
                except OSError as e:
                     logger.warning(f"Could not set permissions for API key file: {e}")

            except Exception as e:
                logger.error(f"Failed to create API key file: {e}")

    def _load_custom_paths(self) -> Dict[str, Path]:
        """Load custom path configurations"""
        if self.paths_config_file.exists():
            try:
                with open(self.paths_config_file, 'r') as f:
                    paths = json.load(f)
                # Validate paths before returning
                valid_paths = {}
                for k, v in paths.items():
                    try:
                        valid_paths[k] = Path(v).resolve() # Resolve to absolute path
                    except Exception as path_e:
                        logger.warning(f"Invalid path '{v}' found for '{k}' in paths config: {path_e}")
                return valid_paths
            except Exception as e:
                logger.error(f"Failed to load custom paths file {self.paths_config_file}: {e}")
        return {}

    def set_custom_path(self, name: str, path: Path):
        """Set a custom path"""
        try:
             resolved_path = Path(path).resolve() # Ensure path exists and is absolute? Or just store string? Store string is safer.
             self.custom_paths[name] = str(path) # Store as string
             self._save_custom_paths()
        except Exception as e:
             logger.error(f"Failed to set custom path '{name}' to '{path}': {e}")


    def _save_custom_paths(self):
        """Save custom path configurations"""
        try:
            # Paths stored as strings
            with open(self.paths_config_file, 'w') as f:
                json.dump(self.custom_paths, f, indent=2, sort_keys=True)
        except Exception as e:
            logger.error(f"Failed to save custom paths to {self.paths_config_file}: {e}")

    def get_dir(self, dir_type: str) -> Path:
        """Get directory path (custom or default), ensuring it exists."""
        path_str = self.custom_paths.get(dir_type)
        if path_str:
            dir_path = Path(path_str)
        else:
            # Map internal names to directory names
            dir_map = {
                "config": self.config_dir,
                "filter": self.filter_dir,
                "preset": self.preset_dir, # For hyperparameter presets
                "pipeline": self.pipeline_dir, # For pipeline configs
                "audio": self.audio_dir, # For audio configs
                "model": self.model_dir, # For model preferences
                "cloud": self.cloud_config_dir # Added cloud config dir
            }
            dir_path = dir_map.get(dir_type, self.base_dir / dir_type)

        # Ensure the directory exists before returning
        try:
             dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             logger.error(f"Could not create or access directory '{dir_path}': {e}")
             # Fallback to base directory? Or raise error? Raising might be safer.
             raise

        return dir_path

    # --- Cloud API Key Management ---
    def load_cloud_keys(self) -> Dict[str, str]:
        """Load API keys from the secure file, returning only non-empty keys."""
        if not self.cloud_keys_file.exists():
            self._ensure_api_key_file_exists() # Create if missing
            return {}
        try:
            # Ensure permissions are appropriate before reading
            try:
                if (self.cloud_keys_file.stat().st_mode & 0o077): # Check if readable/writable by group/others
                    logger.warning(f"API key file {self.cloud_keys_file} has insecure permissions. Setting to 600.")
                    os.chmod(self.cloud_keys_file, 0o600)
            except OSError as e:
                 logger.warning(f"Could not check/set permissions for API key file: {e}")

            with open(self.cloud_keys_file, 'r') as f:
                keys_data = json.load(f)

            # Filter out warning and keys with None or empty string values
            active_keys = {}
            for k, v in keys_data.items():
                if k != "_WARNING" and isinstance(v, str) and v.strip():
                    active_keys[k.lower()] = v.strip() # Store keys lowercased internally

            logger.debug(f"Loaded {len(active_keys)} active API keys.")
            return active_keys

        except json.JSONDecodeError as e:
             logger.error(f"Failed to parse API keys file {self.cloud_keys_file}. It might be corrupted. Error: {e}")
             return {}
        except Exception as e:
            logger.error(f"Failed to load API keys from {self.cloud_keys_file}: {e}")
            return {}

    def save_cloud_key(self, provider: str, key: Optional[str]) -> bool:
        """Save or update an API key for a provider (case-insensitive)."""
        provider_lower = provider.lower()
        if provider_lower == "_warning": # Prevent overwriting warning key
             logger.error("Cannot save key with reserved name '_warning'.")
             return False

        full_keys_data = {}
        if self.cloud_keys_file.exists():
            try:
                with open(self.cloud_keys_file, 'r') as f:
                    full_keys_data = json.load(f)
                    if not isinstance(full_keys_data, dict): # Handle corrupted file
                         logger.warning(f"API keys file {self.cloud_keys_file} is corrupted. Overwriting.")
                         full_keys_data = {}
            except json.JSONDecodeError:
                logger.warning(f"API keys file {self.cloud_keys_file} is corrupted. Overwriting.")
                full_keys_data = {}
            except Exception as e:
                logger.error(f"Error reading existing API keys file: {e}. Attempting to overwrite.")
                full_keys_data = {}

        # Ensure warning key exists
        if "_WARNING" not in full_keys_data:
            full_keys_data["_WARNING"] = "Store API keys securely. This file stores keys in plain text. Consider environment variables or a dedicated secrets manager for production."

        # Update or add the specific provider key (using lowercase internally)
        if key and key.strip():
             full_keys_data[provider_lower] = key.strip()
             action = "updated"
        else:
             # Remove key if value is None or empty
             if provider_lower in full_keys_data:
                  del full_keys_data[provider_lower]
                  action = "cleared"
             else:
                  action = "ignored (already empty)" # No action needed

        # Ensure all known providers have at least a None entry if not set
        for p in KNOWN_CLOUD_PROVIDERS:
             if p not in full_keys_data:
                  full_keys_data[p] = None


        try:
            with open(self.cloud_keys_file, 'w') as f:
                json.dump(full_keys_data, f, indent=2, sort_keys=True)
            # Set permissions after writing
            try:
                 os.chmod(self.cloud_keys_file, 0o600)
            except OSError as e:
                 logger.warning(f"Could not set permissions for API key file after saving: {e}")

            logger.info(f"API key for '{provider}' {action}.")
            return True
        except Exception as e:
            logger.error(f"Failed to save API key for {provider} to {self.cloud_keys_file}: {e}")
            return False

    def get_configured_providers(self) -> List[str]:
         """Returns a sorted list of providers for which an API key is actively set."""
         active_keys = self.load_cloud_keys()
         return sorted(list(active_keys.keys()))

    # --- Generic Config Methods (Unchanged, but use get_dir) ---
# --- Generic Config Methods (Refactored for DRY) ---
    def save_config(self, name: str, config: Dict[str, Any], dir_type: str = "config") -> bool:
        """
        Generic config save method.
        
        Args:
            name: Config name (filename without extension)
            config: Config dictionary to save
            dir_type: Config directory type (config/filter/preset/pipeline/audio/model/cloud)
        """
        try:
            config_dir = self.get_dir(dir_type)
            config_file = config_dir / f"{name}.json"
            config["_metadata"] = {"created": datetime.now().isoformat(), "type": dir_type}
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved {dir_type} configuration: {name} to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {dir_type} configuration '{name}': {e}")
            return False

    def load_config(self, name: str, dir_type: str = "config") -> Optional[Dict[str, Any]]:
        """Generic config load method."""
        try:
            config_dir = self.get_dir(dir_type)
            config_file = config_dir / f"{name}.json"
            if not config_file.exists():
                return None
            with open(config_file, 'r') as f:
                config = json.load(f)
            config.pop("_metadata", None)
            logger.info(f"Loaded {dir_type} configuration: {name} from {config_file}")
            return config
        except Exception as e:
            logger.error(f"Failed to load {dir_type} configuration '{name}': {e}")
            return None

    def list_configs(self, dir_type: str = "config") -> List[str]:
        """List all configs of a given type."""
        config_dir = self.get_dir(dir_type)
        configs = [f.stem for f in config_dir.glob("*.json")]
        return sorted(configs)

    def delete_config(self, name: str, dir_type: str = "config") -> bool:
        """Generic config delete method."""
        try:
            config_dir = self.get_dir(dir_type)
            config_file = config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
                logger.info(f"Deleted {dir_type} configuration: {name}")
                return True
            logger.warning(f"{dir_type.capitalize()} configuration '{name}' not found")
            return False
        except Exception as e:
            logger.error(f"Failed to delete {dir_type} configuration '{name}': {e}")
            return False

    # --- Specialized convenience methods (wrappers) ---
    def save_pipeline_config(self, name: str, config: Dict[str, Any]) -> bool:
        return self.save_config(name, config, "pipeline")
    
    def load_pipeline_config(self, name: str) -> Optional[Dict[str, Any]]:
        return self.load_config(name, "pipeline")
    
    def save_audio_config(self, name: str, config: Dict[str, Any]) -> bool:
        return self.save_config(name, config, "audio")
    
    def load_audio_config(self, name: str) -> Optional[Dict[str, Any]]:
        return self.load_config(name, "audio")
    
    def save_filter_preset(self, name: str, filters: Dict[str, Any]) -> bool:
        return self.save_config(name, filters, "filter")
    
    def load_filter_preset(self, name: str) -> Optional[Dict[str, Any]]:
        return self.load_config(name, "filter")
    
    def save_hyperparameter_preset(self, name: str, params: Dict[str, Any]) -> bool:
        return self.save_config(name, params, "preset")
    
    def load_hyperparameter_preset(self, name: str) -> Optional[Dict[str, Any]]:
        return self.load_config(name, "preset")
    
    def save_model_preference(self, name: str, model_info: Dict[str, Any]) -> bool:
        return self.save_config(name, model_info, "model")
    
    def load_model_preference(self, name: str) -> Optional[Dict[str, Any]]:
        return self.load_config(name, "model")

    # --- Default Config (Unchanged) ---
    def get_default_config(self) -> Dict[str, Any]:
        loaded_config = self.load_config("default", "config")
        if loaded_config: return loaded_config
        # Return hardcoded defaults if file doesn't exist or fails to load
        logger.warning("Default config file not found or failed to load. Using hardcoded defaults.")
        return {
            "mode": "podcast", "model_provider": "local", "model_specifier": DEFAULT_MODEL,
            "memory_profile": "medium_vram", "chunk_size": 1000, "output_format": "markdown",
            "audio_model_provider": "local", "audio_model_specifier": "microsoft/speecht5_tts",
            "sample_rate": 16000, "run_audio_generation": False
        }
    def set_default_config(self, config: Dict[str, Any]) -> bool:
        return self.save_config("default", config, "config")

    # --- Backup/Restore (Unchanged, but use get_dir) ---
    def backup_configs(self, backup_name: Optional[str] = None) -> Optional[Path]:
        try:
            backup_name = backup_name or f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir = self.base_dir / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Starting backup to {backup_dir}...")
            # Include cloud config dir in backup
            for dir_name in ["config", "filter", "preset", "pipeline", "audio", "model", "cloud"]:
                src = self.get_dir(dir_name)
                if src.exists() and src.is_dir():
                    dst = backup_dir / dir_name
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    logger.debug(f"Copied {src} to {dst}")
                else:
                    logger.debug(f"Skipping non-existent directory: {src}")
            logger.info(f"Configuration backup completed: {backup_dir}")
            return backup_dir
        except Exception as e:
            logger.error(f"Failed to create backup '{backup_name}': {e}", exc_info=True)
            return None

    def restore_backup(self, backup_name: str) -> bool:
        try:
            backup_dir = self.base_dir / "backups" / backup_name
            if not backup_dir.exists() or not backup_dir.is_dir():
                logger.error(f"Backup directory not found: {backup_dir}")
                return False
            logger.info(f"Starting restore from backup: {backup_name}")
            # Restore all directories found in the backup
            for item in backup_dir.iterdir():
                if item.is_dir():
                     dir_name = item.name # e.g., 'config', 'cloud'
                     dst = self.get_dir(dir_name) # Get the target directory
                     # Clear destination before copying? Or just overwrite? Overwrite is simpler.
                     shutil.copytree(item, dst, dirs_exist_ok=True)
                     logger.debug(f"Restored {item} to {dst}")

            logger.info(f"Successfully restored configuration from backup: {backup_name}")
            # Reload keys after restore
            # Note: This might require re-initializing dependent components (like MenuSystem state)
            # Or the application needs to be restarted after restore.
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup '{backup_name}': {e}", exc_info=True)
            return False

    def list_backups(self) -> List[str]:
        """List available backups"""
        backup_base = self.base_dir / "backups"
        if not backup_base.exists(): return []
        backups = [d.name for d in backup_base.iterdir() if d.is_dir()]
        return sorted(backups, reverse=True) # Show newest first
