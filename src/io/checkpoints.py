# llamanote/io/checkpoints.py
"""
Checkpoint Management Module
Handles saving and loading of intermediate pipeline stage data.
"""

import pickle
from pathlib import Path
from typing import Optional, Any, List

from ..utils.logger import get_logger_conf
from ..config.settings import DEFAULT_CACHE_DIR, CHECKPOINT_FORMAT

logger = get_logger_conf(__name__)

class CheckpointManager:
    """Manages reading and writing pipeline stage checkpoints."""

    def __init__(self, base_cache_dir: Path, run_id: str):
        """
        Initializes the CheckpointManager for a specific pipeline run.

        Args:
            base_cache_dir: The root cache directory (e.g., /path/to/cache).
            run_id: A unique identifier for the current run (e.g., "filename_timestamp").
        """
        self.checkpoint_dir = base_cache_dir / "checkpoints" / run_id
        self.format = CHECKPOINT_FORMAT.lower().lstrip('.')
        self.logger = get_logger_conf(f"{__name__}.{run_id}")

        if self.format not in ['pickle', 'json']:
            self.logger.warning(f"Unsupported checkpoint format '{self.format}'. Defaulting to 'pickle'.")
            self.format = 'pickle'

    def _ensure_dir(self):
        """Ensures the checkpoint directory for this run exists."""
        try:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create checkpoint directory: {self.checkpoint_dir}. Checkpoints disabled. Error: {e}")

    def _get_checkpoint_path(self, stage_name: str) -> Path:
        """Gets the file path for a specific stage's checkpoint."""
        return self.checkpoint_dir / f"{stage_name}.{self.format}"

    def save(self, stage_name: str, data: Any):
        """
        Saves the data for a given pipeline stage.

        Args:
            stage_name: The name of the stage (e.g., "extract", "preprocess").
            data: The data payload to save.
        """
        self._ensure_dir()
        checkpoint_file = self._get_checkpoint_path(stage_name)
        
        try:
            if self.format == 'pickle':
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(data, f)
            elif self.format == 'json':
                # Note: JSON format will fail if data is not serializable
                import json
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, default=str) # Use default=str as a fallback
                    
            self.logger.debug(f"Saved checkpoint for stage '{stage_name}' to {checkpoint_file.name}")
        except (pickle.PicklingError, TypeError, OSError) as e:
            self.logger.error(f"Failed to save checkpoint for stage '{stage_name}': {e}", exc_info=True)

    def load(self, stage_name: str) -> Optional[Any]:
        """
        Loads the data for a given pipeline stage.

        Args:
            stage_name: The name of the stage to load.

        Returns:
            The loaded data, or None if the checkpoint doesn't exist or fails to load.
        """
        checkpoint_file = self._get_checkpoint_path(stage_name)
        
        if not checkpoint_file.exists():
            self.logger.debug(f"No checkpoint found for stage '{stage_name}' at {checkpoint_file}")
            return None
            
        try:
            if self.format == 'pickle':
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
            elif self.format == 'json':
                import json
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
            self.logger.info(f"Loaded checkpoint for stage '{stage_name}' from {checkpoint_file.name}")
            return data
        except (pickle.UnpicklingError, json.JSONDecodeError, EOFError, TypeError, OSError) as e:
            self.logger.error(f"Failed to load checkpoint for stage '{stage_name}': {e}. File might be corrupt.", exc_info=True)
            # Optionally delete the corrupt file
            # self.delete(stage_name)
            return None

    def delete(self, stage_name: str):
        """Deletes a specific stage checkpoint."""
        checkpoint_file = self._get_checkpoint_path(stage_name)
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                self.logger.debug(f"Deleted checkpoint: {checkpoint_file.name}")
            except OSError as e:
                self.logger.warning(f"Could not delete checkpoint {checkpoint_file.name}: {e}")

    def cleanup(self):
        """Removes the entire checkpoint directory for this run."""
        if self.checkpoint_dir.exists():
            try:
                import shutil
                shutil.rmtree(self.checkpoint_dir)
                self.logger.info(f"Cleaned up checkpoint directory: {self.checkpoint_dir}")
            except OSError as e:
                self.logger.error(f"Failed to clean up checkpoint directory {self.checkpoint_dir}: {e}")
