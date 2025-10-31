# llamanote/config/cloud_keys.py
"""
Cloud API Key Management for LlamaNote Enhanced
Handles loading, saving, and managing API keys securely.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, Any

from ..utils.logger import get_logger_conf
from .settings import KNOWN_CLOUD_PROVIDERS # Import known provider list

logger = get_logger_conf(__name__)

class CloudKeyManager:
    """Manages cloud provider API keys securely."""
    DEFAULT_FILENAME = "api_keys.json"
    WARNING_KEY = "_WARNING"
    DEFAULT_WARNING = "Store API keys securely. This file may store keys in plain text."

    def __init__(self, keys_file_path: Path):
        """
        Initializes the key manager.

        Args:
            keys_file_path: The absolute path to the JSON file storing API keys.
        """
        self.keys_file = keys_file_path
        self._ensure_api_key_file_exists()

    def _ensure_api_key_file_exists(self):
        """Creates the API key file with placeholders and warning if it doesn't exist."""
        if not self.keys_file.exists():
            try:
                self.keys_file.parent.mkdir(parents=True, exist_ok=True)
                warning_data = {
                    self.WARNING_KEY: self.DEFAULT_WARNING,
                    **{provider: None for provider in KNOWN_CLOUD_PROVIDERS}
                }
                self._write_keys(warning_data) # Use helper to write and set permissions
                logger.info(f"Created template API key file: {self.keys_file}")
            except Exception as e:
                logger.error(f"Failed to create API key file at {self.keys_file}: {e}")

    def _read_keys(self) -> Dict[str, Any]:
        """Reads the entire key file, handling potential errors."""
        if not self.keys_file.exists():
            logger.warning(f"API keys file not found: {self.keys_file}. Cannot load keys.")
            return {}
        try:
            # Check permissions before reading (optional but recommended)
            if sys.platform != 'win32': # os.chmod modes behave differently on Windows
                current_mode = self.keys_file.stat().st_mode
                # Check if group or other has read/write/execute permissions
                if current_mode & 0o077:
                    logger.warning(f"API key file {self.keys_file} has insecure permissions ({oct(current_mode)}). Attempting to set to 600.")
                    try:
                        os.chmod(self.keys_file, 0o600)
                    except OSError as e:
                        logger.warning(f"Could not set secure permissions for API key file: {e}")

            with open(self.keys_file, 'r', encoding='utf-8') as f:
                keys_data = json.load(f)
            if not isinstance(keys_data, dict):
                 logger.error(f"API keys file {self.keys_file} is corrupted (not a JSON object).")
                 return {}
            return keys_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API keys file {self.keys_file}: {e}. File might be corrupted.")
            return {}
        except Exception as e:
            logger.error(f"Failed to load API keys from {self.keys_file}: {e}", exc_info=True)
            return {}

    def _write_keys(self, keys_data: Dict[str, Any]) -> bool:
        """Writes the entire key file with secure permissions."""
        try:
            temp_file = self.keys_file.with_suffix(".tmp")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(keys_data, f, indent=2, sort_keys=True)

            # Set permissions before renaming to final file
            if sys.platform != 'win32':
                os.chmod(temp_file, 0o600)

            # Atomic rename (replace existing file)
            os.replace(temp_file, self.keys_file)

            logger.debug(f"Successfully wrote API keys to {self.keys_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to write API keys to {self.keys_file}: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_file.exists():
                try: temp_file.unlink()
                except OSError: pass
            return False

    def load_active_keys(self) -> Dict[str, str]:
        """Loads only the currently set, non-empty API keys (provider names lowercased)."""
        all_keys = self._read_keys()
        active_keys = {}
        for k, v in all_keys.items():
            # Exclude warning key and filter for non-empty strings
            if k != self.WARNING_KEY and isinstance(v, str) and v.strip():
                active_keys[k.lower()] = v.strip() # Store provider name lowercased
        logger.debug(f"Loaded {len(active_keys)} active API keys.")
        return active_keys

    def save_key(self, provider: str, key: Optional[str]) -> bool:
        """
        Saves, updates, or clears an API key for a specific provider.
        Provider names are handled case-insensitively.
        """
        provider_lower = provider.lower()
        if provider_lower == self.WARNING_KEY.lower():
             logger.error(f"Cannot save key with reserved name '{self.WARNING_KEY}'.")
             return False

        keys_data = self._read_keys()
        if not isinstance(keys_data, dict): keys_data = {} # Recover from corruption

        # Ensure warning key and placeholders for all known providers exist
        keys_data.setdefault(self.WARNING_KEY, self.DEFAULT_WARNING)
        for p in KNOWN_CLOUD_PROVIDERS:
            keys_data.setdefault(p.lower(), None) # Use lowercase for internal consistency

        # Update or clear the specific provider key
        current_value = keys_data.get(provider_lower)
        new_value = key.strip() if key and isinstance(key, str) and key.strip() else None

        if current_value == new_value:
            action = "ignored (no change)"
        elif new_value is not None:
            keys_data[provider_lower] = new_value
            action = "updated"
        else: # new_value is None, clear the key
            keys_data[provider_lower] = None # Set to None, don't delete key
            action = "cleared"

        # Write the updated data back to the file
        if self._write_keys(keys_data):
            logger.info(f"API key for '{provider}' {action}.")
            return True
        return False

    def get_key(self, provider: str) -> Optional[str]:
        """Retrieves the API key for a specific provider (case-insensitive)."""
        provider_lower = provider.lower()
        # Load active keys fresh each time to ensure consistency
        active_keys = self.load_active_keys()
        return active_keys.get(provider_lower)

    def get_configured_providers(self) -> List[str]:
        """Returns a sorted list of provider names (lowercase) for which an API key is actively set."""
        active_keys = self.load_active_keys()
        return sorted(list(active_keys.keys()))
