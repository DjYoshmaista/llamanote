# llamanote/utils/helpers.py
"""
Miscellaneous Helper Functions for LlamaNote Enhanced
Includes resource cleanup, path generation, token estimation, etc.
"""

import gc
import threading
import re
import os
from pathlib import Path
from typing import List, Any, Optional, Union
from datetime import datetime

from .logger import get_logger_conf

# Attempt optional imports
try:
    import torch
    TORCH_AVAILABLE_FOR_HELPERS = True
except ImportError:
    TORCH_AVAILABLE_FOR_HELPERS = False
    torch = None

logger = get_logger_conf(__name__)

# --- Resource Management ---

def cleanup_resources(objects_to_delete: List[Any], clear_cuda: bool = True):
    """
    Attempts to release resources held by objects and run garbage collection.

    Args:
        objects_to_delete: A list of object references to delete.
        clear_cuda: Whether to attempt clearing the CUDA cache.
    """
    if not isinstance(objects_to_delete, list):
        objects_to_delete = [objects_to_delete]

    deleted_count = 0
    for obj in objects_to_delete:
        if obj is not None:
            try:
                # Try common cleanup methods if they exist
                if hasattr(obj, 'unload'): obj.unload()
                elif hasattr(obj, 'close'): obj.close()
                elif hasattr(obj, 'release'): obj.release()
                # Remove reference (actual deletion depends on GC)
                del obj
                deleted_count += 1
            except Exception as e:
                logger.debug(f"Error during object cleanup: {e}")

    logger.debug(f"Cleared references for {deleted_count} object(s).")

    # Force garbage collection
    collected = gc.collect()
    logger.debug(f"Garbage collector ran, collected {collected} objects.")

    # Clear CUDA cache if requested and possible
    if clear_cuda and TORCH_AVAILABLE_FOR_HELPERS and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared.")
        except Exception as e:
            logger.warning(f"Could not clear CUDA cache: {e}")


# --- Path Generation ---

class PathGenerator:
    """Helper class for generating consistent file paths."""
    def __init__(self, base_output_dir: Path, timestamp_files: bool = True, create_backups: bool = True):
        self.base_output_dir = base_output_dir
        self.timestamp_files = timestamp_files
        self.create_backups = create_backups
        self.backup_dir = base_output_dir / "backups"

    def _ensure_dir(self, dir_path: Path):
        """Ensures a directory exists."""
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create directory '{dir_path}': {e}")
            # Depending on severity, could raise error here

    def generate_output_path(self,
                             input_path: Path,
                             suffix: str = "_processed",
                             extension: str = ".md",
                             sub_dir: Optional[str] = None,
                             base_filename_override: Optional[str] = None) -> Path:
        """Generates a unique output path in the designated output directory."""
        output_dir = self.base_output_dir
        if sub_dir:
            output_dir = output_dir / sub_dir
        self._ensure_dir(output_dir)

        base_name = base_filename_override if base_filename_override else input_path.stem
        # Sanitize base_name further if needed
        base_name = re.sub(r'[\\/*?:"<>|]', '_', base_name)

        if self.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{base_name}{suffix}_{timestamp}{extension}"
            output_path = output_dir / output_name
        else:
            # Handle potential collisions if not timestamping
            output_name = f"{base_name}{suffix}{extension}"
            output_path = output_dir / output_name
            counter = 1
            while output_path.exists():
                output_name = f"{base_name}{suffix}_{counter}{extension}"
                output_path = output_dir / output_name
                counter += 1
                if counter > 100: # Safety break
                     logger.warning(f"Could not find unique filename for {base_name} after 100 attempts.")
                     # Use timestamp as fallback to prevent infinite loop
                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                     output_name = f"{base_name}{suffix}_{timestamp}{extension}"
                     output_path = output_dir / output_name
                     break

        return output_path

    def handle_existing_file(self, file_path: Path) -> Path:
        """Creates a backup or generates a new name if file exists."""
        if not file_path.exists():
            return file_path

        if self.create_backups:
            self._create_backup(file_path)
            return file_path # Allow overwrite after backup
        else:
            # Generate unique name (similar logic to non-timestamped generation)
            original_stem = file_path.stem
            original_suffix = file_path.suffix
            parent_dir = file_path.parent
            counter = 1
            while file_path.exists():
                new_stem = f"{original_stem}_{counter}"
                file_path = parent_dir / f"{new_stem}{original_suffix}"
                counter += 1
                if counter > 100: # Safety break
                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                     file_path = parent_dir / f"{original_stem}_{timestamp}{original_suffix}"
                     logger.warning(f"Could not find unique filename for {original_stem}, using timestamp.")
                     break
            logger.debug(f"Generated unique filename: {file_path.name}")
            return file_path


    def _create_backup(self, file_path: Path):
        """Creates a timestamped backup of a file."""
        if not file_path.exists(): return
        self._ensure_dir(self.backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        try:
            shutil.copy2(file_path, backup_path)
            logger.debug(f"Created backup: {backup_path.name}")
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path.name}: {e}")

# --- Token Estimation ---

# Simple character-based estimation (can be replaced with actual tokenization if needed)
# Assumes roughly 4 characters per token on average for English text.
CHARS_PER_TOKEN_ESTIMATE = 4

def estimate_tokens(text: Optional[str], provider: Optional[str] = None) -> int:
    """
    Estimates the number of tokens in a string.

    Args:
        text: The input text.
        provider: Optional provider name (future use for provider-specific tokenizers).

    Returns:
        Estimated number of tokens.
    """
    if not text:
        return 0

    # Provider-specific logic could go here in the future
    # For now, use a simple character count heuristic
    return (len(text) + CHARS_PER_TOKEN_ESTIMATE - 1) // CHARS_PER_TOKEN_ESTIMATE


# --- Type Conversions ---

def ensure_path(path_like: Union[str, Path]) -> Path:
    """Converts a string or Path to a resolved Path object."""
    return Path(os.path.expanduser(path_like)).resolve()

def ensure_string(path_like: Union[str, Path]) -> str:
    """Converts a string or Path to a string representation."""
    return str(path_like)


# --- Device Management ---

class DeviceManager:
    """Singleton class to manage device detection."""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DeviceManager, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.cuda_available = False
        self.device_count = 0
        self.current_device = "cpu"
        if TORCH_AVAILABLE_FOR_HELPERS:
            try:
                self.cuda_available = torch.cuda.is_available()
                if self.cuda_available:
                    self.device_count = torch.cuda.device_count()
                    self.current_device = "cuda"
            except Exception as e:
                logger.warning(f"Error checking CUDA availability: {e}")
        logger.info(f"DeviceManager initialized: CUDA available={self.cuda_available}, Device count={self.device_count}, Default device={self.current_device}")

    def get_device(self) -> str:
        """Returns 'cuda' if available, otherwise 'cpu'."""
        return self.current_device

    def is_cuda_available(self) -> bool:
        return self.cuda_available

    def get_gpu_count(self) -> int:
        return self.device_count

# Convenience function to access the manager
def get_device_manager() -> DeviceManager:
    return DeviceManager()
