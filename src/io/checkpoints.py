# llamanote/io/checkpoints.py
"""
Advanced Checkpoint Management Module
Handles saving and loading of intermediate pipeline stage data with
stage-aware compatibility checking and hash-based organization.
"""

import pickle
import hashlib
import json
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime
from dataclasses import asdict

from ..utils.logger import get_logger_conf, ConsoleOutput
from ..config.settings import DEFAULT_CACHE_DIR
from ..core.types import PipelineConfig, AudioConfig

logger = get_logger_conf(__name__)

# Version for checkpoint format compatibility
CHECKPOINT_VERSION = "1.0"

# Define which config fields matter for each stage
STAGE_RELEVANT_FIELDS = {
    "extract": ["input_file"],
    "preprocess": ["input_file"],  # Preprocess depends on extracted text
    "chunk": ["chunking_strategy", "chunk_size", "chunk_overlap"],
    "process": ["model_provider", "model_specifier", "system_prompt", "mode"],
    "filter": ["model_provider"],  # Filter depends on model output format
    "format": ["output_format", "mode"],
    "save": [],  # Save can always be rerun, no compatibility check needed
    "audio": ["audio_provider", "audio_specifier", "audio_config", "generate_audio"]
}


class CheckpointManager:
    """
    Advanced checkpoint manager with hash-based organization and stage-aware compatibility.

    Checkpoint file format (.ckpt):
        Single pickle file containing a dict with:
        {
            "metadata": {...},  # JSON-serializable metadata
            "data": {...}       # Actual pipeline data payload
        }

    Directory structure:
        checkpoints/
        └── <input_hash>/
            └── <config_hash>_<stage>.ckpt
    """

    def __init__(self,
                 base_checkpoint_dir: Optional[Path] = None,
                 resume_mode: str = "auto"):
        """
        Initialize the CheckpointManager.

        Args:
            base_checkpoint_dir: Root checkpoint directory (defaults to project_root/checkpoints)
            resume_mode: "auto" (automatically resume), "interactive" (ask user), "disabled" (no resume)
        """
        if base_checkpoint_dir is None:
            # Use project root / checkpoints
            base_checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"

        self.base_dir = Path(base_checkpoint_dir)
        self.resume_mode = resume_mode
        self.logger = logger

        # Create checkpoint directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _generate_input_hash(self, input_path: Path) -> str:
        """Generate 8-character hash from input file path."""
        # Use file stem (name without extension) for hash
        file_identifier = input_path.stem
        hash_obj = hashlib.sha256(file_identifier.encode('utf-8'))
        return hash_obj.hexdigest()[:8]

    def _generate_config_hash(self, config: PipelineConfig, stage: str, timestamp: Optional[str] = None) -> str:
        """
        Generate 12-character hash from configuration relevant to the given stage.

        Args:
            config: Pipeline configuration
            stage: Stage name to generate hash for
            timestamp: Optional timestamp to include in hash

        Returns:
            12-character hash string
        """
        # Get relevant fields for this stage and all previous stages
        relevant_fields = set()
        stage_order = ["extract", "preprocess", "chunk", "process", "filter", "format", "save", "audio"]

        # Include all fields up to and including current stage
        try:
            stage_idx = stage_order.index(stage)
            for s in stage_order[:stage_idx + 1]:
                relevant_fields.update(STAGE_RELEVANT_FIELDS.get(s, []))
        except ValueError:
            # Unknown stage, include all fields
            for fields in STAGE_RELEVANT_FIELDS.values():
                relevant_fields.update(fields)

        # Build hash input from relevant config values
        hash_input = {}
        config_dict = asdict(config)

        for field in sorted(relevant_fields):  # Sort for consistent ordering
            if field == "input_file":
                continue  # Input file handled separately
            elif field == "audio_config" and config.audio_config:
                # Include audio config details
                hash_input["audio_config"] = {
                    "sample_rate": config.audio_config.sample_rate,
                    "output_format": config.audio_config.output_format,
                    "speed": config.audio_config.speed,
                    "pitch_shift": config.audio_config.pitch_shift,
                    "cloud_voice": getattr(config.audio_config, 'cloud_voice', None)
                }
            elif field in config_dict:
                value = config_dict[field]
                # Convert to string representation for hashing
                if isinstance(value, (Path, object)):
                    hash_input[field] = str(value)
                else:
                    hash_input[field] = value

        # Add timestamp if provided
        if timestamp:
            hash_input["timestamp"] = timestamp

        # Generate hash
        hash_str = json.dumps(hash_input, sort_keys=True, default=str)
        hash_obj = hashlib.sha256(hash_str.encode('utf-8'))
        return hash_obj.hexdigest()[:12]

    def _get_checkpoint_path(self, input_path: Path, config: PipelineConfig, stage: str, timestamp: Optional[str] = None) -> Path:
        """Get the full path for a checkpoint file."""
        input_hash = self._generate_input_hash(input_path)
        config_hash = self._generate_config_hash(config, stage, timestamp)

        checkpoint_dir = self.base_dir / input_hash
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{config_hash}_{stage}.ckpt"
        return checkpoint_dir / filename

    def _create_metadata(self,
                        input_path: Path,
                        config: PipelineConfig,
                        stage: str,
                        data_keys: List[str]) -> Dict[str, Any]:
        """Create metadata dictionary for checkpoint."""
        timestamp = datetime.now().isoformat()

        metadata = {
            "version": CHECKPOINT_VERSION,
            "timestamp": timestamp,
            "input_file": str(input_path),
            "input_stem": input_path.stem,
            "stage": stage,
            "data_keys": data_keys,
            "config": {
                "text_model": f"{config.model_provider}:{config.model_specifier}",
                "mode": str(config.mode),
                "chunking": {
                    "strategy": str(config.chunking_strategy),
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap
                } if stage in ["chunk", "process"] else None,
                "output_format": config.output_format,
            },
            "hash_info": {
                "input_hash": self._generate_input_hash(input_path),
                "config_hash": self._generate_config_hash(config, stage, timestamp)
            }
        }

        # Add audio config if audio stage
        if stage == "audio" and config.generate_audio:
            metadata["config"]["audio"] = {
                "provider": config.audio_provider,
                "model": config.audio_specifier,
                "sample_rate": config.audio_config.sample_rate if config.audio_config else None,
                "output_format": config.audio_config.output_format if config.audio_config else None
            }

        return metadata

    def save(self,
             input_path: Path,
             config: PipelineConfig,
             stage: str,
             data: Dict[str, Any]) -> bool:
        """
        Save a checkpoint for the given stage.

        Args:
            input_path: Path to the input file being processed
            config: Pipeline configuration
            stage: Stage name
            data: Data payload dictionary to save

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            checkpoint_path = self._get_checkpoint_path(input_path, config, stage)

            # Create metadata
            data_keys = list(data.keys())
            metadata = self._create_metadata(input_path, config, stage, data_keys)

            # Create checkpoint object
            checkpoint = {
                "metadata": metadata,
                "data": data
            }

            # Save as single pickle file
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

            self.logger.info(f"Saved checkpoint for stage '{stage}' to {checkpoint_path.name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for stage '{stage}': {e}", exc_info=True)
            return False

    def load(self, checkpoint_path: Path) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Load a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Tuple of (metadata, data) if successful, None otherwise
        """
        if not checkpoint_path.exists():
            self.logger.debug(f"Checkpoint file not found: {checkpoint_path}")
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

            if not isinstance(checkpoint, dict) or "metadata" not in checkpoint or "data" not in checkpoint:
                self.logger.error(f"Invalid checkpoint format in {checkpoint_path.name}")
                return None

            metadata = checkpoint["metadata"]
            data = checkpoint["data"]

            # Validate version
            if metadata.get("version") != CHECKPOINT_VERSION:
                self.logger.warning(f"Checkpoint version mismatch: {metadata.get('version')} vs {CHECKPOINT_VERSION}")

            self.logger.info(f"Loaded checkpoint from {checkpoint_path.name} (stage: {metadata.get('stage')})")
            return metadata, data

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path.name}: {e}", exc_info=True)
            return None

    def _check_stage_compatibility(self,
                                   checkpoint_metadata: Dict[str, Any],
                                   config: PipelineConfig,
                                   resume_from_stage: str) -> Tuple[bool, List[str]]:
        """
        Check if a checkpoint is compatible with the current configuration for resuming.

        Args:
            checkpoint_metadata: Metadata from the checkpoint
            config: Current pipeline configuration
            resume_from_stage: Stage we want to resume from

        Returns:
            Tuple of (is_compatible, list_of_incompatibilities)
        """
        incompatibilities = []
        checkpoint_config = checkpoint_metadata.get("config", {})

        # Check version
        if checkpoint_metadata.get("version") != CHECKPOINT_VERSION:
            incompatibilities.append(f"Version mismatch: checkpoint v{checkpoint_metadata.get('version')} vs current v{CHECKPOINT_VERSION}")

        # Get relevant fields for stages from resume point onwards
        stage_order = ["extract", "preprocess", "chunk", "process", "filter", "format", "save", "audio"]
        try:
            resume_idx = stage_order.index(resume_from_stage)
        except ValueError:
            incompatibilities.append(f"Unknown stage: {resume_from_stage}")
            return False, incompatibilities

        # Collect relevant fields for upcoming stages
        relevant_fields = set()
        for s in stage_order[resume_idx:]:
            relevant_fields.update(STAGE_RELEVANT_FIELDS.get(s, []))

        # Check text model if process stage or later
        if "model_provider" in relevant_fields or "model_specifier" in relevant_fields:
            checkpoint_model = checkpoint_config.get("text_model", "")
            current_model = f"{config.model_provider}:{config.model_specifier}"
            if checkpoint_model != current_model:
                incompatibilities.append(f"Text model mismatch: {checkpoint_model} vs {current_model}")

        # Check chunking settings if chunk stage or later
        if "chunking_strategy" in relevant_fields:
            checkpoint_chunking = checkpoint_config.get("chunking", {})
            if checkpoint_chunking:
                if str(checkpoint_chunking.get("strategy")) != str(config.chunking_strategy):
                    incompatibilities.append(f"Chunking strategy mismatch")
                if checkpoint_chunking.get("chunk_size") != config.chunk_size:
                    incompatibilities.append(f"Chunk size mismatch")

        # Check audio settings if audio stage
        if "audio_provider" in relevant_fields and config.generate_audio:
            checkpoint_audio = checkpoint_config.get("audio", {})
            if checkpoint_audio:
                current_audio = f"{config.audio_provider}:{config.audio_specifier}"
                checkpoint_audio_model = f"{checkpoint_audio.get('provider')}:{checkpoint_audio.get('model')}"
                if checkpoint_audio_model != current_audio:
                    incompatibilities.append(f"Audio model mismatch: {checkpoint_audio_model} vs {current_audio}")

        # Check output format if format stage or later
        if "output_format" in relevant_fields:
            if checkpoint_config.get("output_format") != config.output_format:
                # This is a minor incompatibility, just a warning
                self.logger.debug(f"Output format differs but can proceed")

        return len(incompatibilities) == 0, incompatibilities

    def find_compatible_checkpoints(self,
                                    input_path: Path,
                                    config: PipelineConfig,
                                    stages_to_run: Optional[List[str]] = None) -> List[Tuple[str, Path, Dict[str, Any]]]:
        """
        Find all compatible checkpoints for resuming the pipeline.

        Args:
            input_path: Input file being processed
            config: Current pipeline configuration
            stages_to_run: List of stages to run (for determining compatibility)

        Returns:
            List of tuples: (stage_name, checkpoint_path, metadata)
            Sorted by stage order (latest stage first)
        """
        input_hash = self._generate_input_hash(input_path)
        checkpoint_dir = self.base_dir / input_hash

        if not checkpoint_dir.exists():
            return []

        compatible_checkpoints = []
        stage_order = ["extract", "preprocess", "chunk", "process", "filter", "format", "save", "audio"]

        # Find all checkpoint files
        for ckpt_file in checkpoint_dir.glob("*.ckpt"):
            try:
                # Extract stage from filename
                stage = ckpt_file.stem.split('_', 1)[1]  # Format: {hash}_{stage}

                # Load and check compatibility
                result = self.load(ckpt_file)
                if result is None:
                    continue

                metadata, _ = result

                # Check if stage is in our pipeline
                if stages_to_run and stage not in stages_to_run:
                    continue

                # Check compatibility
                is_compatible, incompatibilities = self._check_stage_compatibility(metadata, config, stage)

                if is_compatible:
                    compatible_checkpoints.append((stage, ckpt_file, metadata))
                else:
                    self.logger.debug(f"Checkpoint {ckpt_file.name} incompatible: {', '.join(incompatibilities)}")

            except Exception as e:
                self.logger.warning(f"Error checking checkpoint {ckpt_file.name}: {e}")
                continue

        # Sort by stage order (latest first)
        def stage_index(item):
            stage_name = item[0]
            try:
                return stage_order.index(stage_name)
            except ValueError:
                return -1

        compatible_checkpoints.sort(key=stage_index, reverse=True)
        return compatible_checkpoints

    def get_resume_checkpoint(self,
                             input_path: Path,
                             config: PipelineConfig,
                             stages_to_run: Optional[List[str]] = None) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Get the best checkpoint to resume from based on resume_mode.

        Args:
            input_path: Input file being processed
            config: Current pipeline configuration
            stages_to_run: List of stages to run

        Returns:
            Tuple of (stage_name, data) if resuming, None otherwise
        """
        if self.resume_mode == "disabled":
            return None

        compatible = self.find_compatible_checkpoints(input_path, config, stages_to_run)

        if not compatible:
            self.logger.debug("No compatible checkpoints found")
            return None

        if self.resume_mode == "auto":
            # Use the latest compatible checkpoint
            stage, ckpt_path, metadata = compatible[0]
            result = self.load(ckpt_path)
            if result:
                _, data = result
                ConsoleOutput.info(f"Resuming from checkpoint: stage '{stage}' ({metadata['timestamp']})")
                return stage, data
            return None

        elif self.resume_mode == "interactive":
            # Present options to user
            return self._interactive_checkpoint_selection(compatible)

        return None

    def _interactive_checkpoint_selection(self,
                                         compatible_checkpoints: List[Tuple[str, Path, Dict[str, Any]]]) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Interactive checkpoint selection (to be implemented with menu system).

        For now, just auto-select the latest.
        """
        # TODO: Integrate with menu system for interactive selection
        if compatible_checkpoints:
            stage, ckpt_path, metadata = compatible_checkpoints[0]
            result = self.load(ckpt_path)
            if result:
                _, data = result
                ConsoleOutput.info(f"Auto-selecting checkpoint: stage '{stage}'")
                return stage, data
        return None

    def cleanup_old_checkpoints(self, input_path: Path, keep_latest: int = 3):
        """
        Clean up old checkpoints for a given input file, keeping only the N most recent.

        Args:
            input_path: Input file
            keep_latest: Number of most recent checkpoints to keep per stage
        """
        input_hash = self._generate_input_hash(input_path)
        checkpoint_dir = self.base_dir / input_hash

        if not checkpoint_dir.exists():
            return

        # Group checkpoints by stage
        checkpoints_by_stage: Dict[str, List[Tuple[Path, float]]] = {}

        for ckpt_file in checkpoint_dir.glob("*.ckpt"):
            try:
                stage = ckpt_file.stem.split('_', 1)[1]
                mtime = ckpt_file.stat().st_mtime

                if stage not in checkpoints_by_stage:
                    checkpoints_by_stage[stage] = []
                checkpoints_by_stage[stage].append((ckpt_file, mtime))
            except Exception:
                continue

        # For each stage, keep only the N most recent
        for stage, ckpt_list in checkpoints_by_stage.items():
            # Sort by modification time (newest first)
            ckpt_list.sort(key=lambda x: x[1], reverse=True)

            # Delete old checkpoints
            for ckpt_file, _ in ckpt_list[keep_latest:]:
                try:
                    ckpt_file.unlink()
                    self.logger.debug(f"Deleted old checkpoint: {ckpt_file.name}")
                except OSError as e:
                    self.logger.warning(f"Could not delete checkpoint {ckpt_file.name}: {e}")

    def delete_all_checkpoints(self, input_path: Optional[Path] = None):
        """
        Delete all checkpoints, or all checkpoints for a specific input file.

        Args:
            input_path: If provided, delete only checkpoints for this file. Otherwise delete all.
        """
        if input_path:
            input_hash = self._generate_input_hash(input_path)
            checkpoint_dir = self.base_dir / input_hash

            if checkpoint_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(checkpoint_dir)
                    self.logger.info(f"Deleted all checkpoints for {input_path.name}")
                except OSError as e:
                    self.logger.error(f"Failed to delete checkpoints: {e}")
        else:
            # Delete entire checkpoint directory
            if self.base_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(self.base_dir)
                    self.base_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info("Deleted all checkpoints")
                except OSError as e:
                    self.logger.error(f"Failed to delete checkpoints: {e}")
