# llamanote/io/checkpoints.py
"""
Advanced Checkpoint Management Module
Handles saving and loading of intermediate pipeline stage data with
stage-aware compatibility checking and hash-based organization.
"""

import pickle
import gzip
import hashlib
import json
from pathlib import Path
from typing import Optional, Any, Dict, List, Tuple
from datetime import datetime
from dataclasses import asdict
import os

from ..utils.logger import get_logger_conf, ConsoleOutput
from ..config.settings import DEFAULT_CACHE_DIR
from ..core.types import PipelineConfig, AudioConfig

logger = get_logger_conf(__name__)

# Version for checkpoint format compatibility
CHECKPOINT_VERSION = "2.0"  # Updated for compression and enhanced metadata

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
                 resume_mode: str = "auto",
                 enable_compression: bool = True):
        """
        Initialize the CheckpointManager.

        Args:
            base_checkpoint_dir: Root checkpoint directory (defaults to project_root/checkpoints)
            resume_mode: "auto" (automatically resume), "interactive" (ask user), "disabled" (no resume)
            enable_compression: Whether to use gzip compression for checkpoints (default: True)
        """
        if base_checkpoint_dir is None:
            # Use project root / checkpoints
            base_checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"

        self.base_dir = Path(base_checkpoint_dir)
        self.resume_mode = resume_mode
        self.enable_compression = enable_compression
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

    def _get_checkpoint_path(self, input_path: Path, config: PipelineConfig, stage: str, timestamp: Optional[str] = None, chunk_index: Optional[int] = None) -> Path:
        """
        Get the full path for a checkpoint file.

        Args:
            input_path: Input file path
            config: Pipeline configuration
            stage: Stage name
            timestamp: Optional timestamp for hash
            chunk_index: Optional chunk index for mid-stage checkpoints

        Returns:
            Path to checkpoint file
        """
        input_hash = self._generate_input_hash(input_path)
        config_hash = self._generate_config_hash(config, stage, timestamp)

        checkpoint_dir = self.base_dir / input_hash
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Include chunk index in filename if provided (for mid-stage checkpoints)
        if chunk_index is not None:
            filename = f"{config_hash}_{stage}_chunk{chunk_index:04d}.ckpt"
        else:
            filename = f"{config_hash}_{stage}.ckpt"
        return checkpoint_dir / filename

    def _create_metadata(self,
                        input_path: Path,
                        config: PipelineConfig,
                        stage: str,
                        data_keys: List[str]) -> Dict[str, Any]:
        """Create comprehensive metadata dictionary for checkpoint."""
        timestamp = datetime.now().isoformat()

        metadata = {
            "version": CHECKPOINT_VERSION,
            "timestamp": timestamp,
            "input_file": str(input_path),
            "input_stem": input_path.stem,
            "input_size_mb": round(input_path.stat().st_size / (1024 * 1024), 2) if input_path.exists() else 0,
            "stage": stage,
            "data_keys": data_keys,

            # Core configuration
            "config": {
                "text_model": {
                    "provider": config.model_provider,
                    "specifier": config.model_specifier,
                    "full_name": f"{config.model_provider}:{config.model_specifier}"
                },
                "mode": str(config.mode),
                "system_prompt": config.system_prompt[:100] + "..." if config.system_prompt and len(config.system_prompt) > 100 else config.system_prompt,

                # Chunking settings (always include for reference)
                "chunking": {
                    "strategy": str(config.chunking_strategy),
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap
                },

                # Processing settings
                "processing": {
                    "remove_thinking": config.remove_thinking,
                    "clean_for_audio": config.clean_for_audio,
                    "preserve_pdf_layout": config.preserve_pdf_layout,
                    "max_retries": config.max_retries,
                    "fallback_on_error": config.fallback_on_error,
                },

                # Output settings
                "output": {
                    "format": config.output_format,
                    "directory": str(config.output_dir) if config.output_dir else None,
                    "timestamp_outputs": config.timestamp_outputs,
                    "include_metadata": config.include_metadata,
                },

                # Checkpoint settings
                "checkpointing": {
                    "enabled": config.enable_checkpoints,
                    "interval": config.checkpoint_interval,
                    "resume_mode": config.checkpoint_resume_mode,
                    "cleanup_keep": config.checkpoint_cleanup_keep,
                },
            },

            # Hyperparameters
            "hyperparameters": {},

            # Memory/Hardware config
            "memory_config": {},

            # Hash information
            "hash_info": {
                "input_hash": self._generate_input_hash(input_path),
                "config_hash": self._generate_config_hash(config, stage, timestamp)
            }
        }

        # Add hyperparameters if available
        try:
            hyperparams = config.get_hyperparameters()
            metadata["hyperparameters"] = {
                "temperature": hyperparams.temperature,
                "top_p": hyperparams.top_p,
                "top_k": hyperparams.top_k,
                "max_new_tokens": hyperparams.max_new_tokens,
                "repetition_penalty": hyperparams.repetition_penalty,
                "do_sample": hyperparams.do_sample,
            }
        except Exception as e:
            logger.debug(f"Could not get hyperparameters: {e}")

        # Add quantization config
        if config.quantization_config:
            metadata["memory_config"]["quantization"] = {
                "method": config.quantization_config.method,
                "compute_dtype": str(config.quantization_config.compute_dtype) if config.quantization_config.compute_dtype else None,
                "use_double_quant": config.quantization_config.use_double_quant,
                "quant_type": config.quantization_config.quant_type,
            }

        # Add layer split config
        if config.layer_split_config:
            metadata["memory_config"]["layer_split"] = {
                "enabled": config.layer_split_config.enabled,
                "gpu_layers": config.layer_split_config.gpu_layers,
                "max_gpu_memory": config.layer_split_config.max_gpu_memory,
                "max_cpu_memory": config.layer_split_config.max_cpu_memory,
                "low_cpu_mem_usage": config.layer_split_config.low_cpu_mem_usage,
                "offload_folder": str(config.layer_split_config.offload_folder) if config.layer_split_config.offload_folder else None,
            }

        # Add audio config if audio stage
        if stage == "audio" and config.generate_audio:
            metadata["config"]["audio"] = {
                "provider": config.audio_provider,
                "model": config.audio_specifier,
                "full_name": f"{config.audio_provider}:{config.audio_specifier}",
                "sample_rate": config.audio_config.sample_rate if config.audio_config else None,
                "output_format": config.audio_config.output_format if config.audio_config else None,
                "chunk_size": config.audio_config.chunk_size if config.audio_config else None,
                "speed": config.audio_config.speed if config.audio_config else None,
                "pitch_shift": config.audio_config.pitch_shift if config.audio_config else None,
                "quantization": config.audio_config.quantization if config.audio_config else None,
            }

        return metadata

    def save(self,
             input_path: Path,
             config: PipelineConfig,
             stage: str,
             data: Dict[str, Any],
             chunk_index: Optional[int] = None) -> bool:
        """
        Save a checkpoint for the given stage.

        Args:
            input_path: Path to the input file being processed
            config: Pipeline configuration
            stage: Stage name
            data: Data payload dictionary to save
            chunk_index: Optional chunk index for mid-stage checkpoints

        Returns:
            True if save succeeded, False otherwise
        """
        try:
            checkpoint_path = self._get_checkpoint_path(input_path, config, stage, chunk_index=chunk_index)

            # Create metadata
            data_keys = list(data.keys())
            metadata = self._create_metadata(input_path, config, stage, data_keys)

            # Add chunk index to metadata if provided
            if chunk_index is not None:
                metadata["chunk_index"] = chunk_index

            # Create checkpoint object
            checkpoint = {
                "metadata": metadata,
                "data": data
            }

            # Save as pickle file (with optional compression)
            # Use atomic write: write to temp file first, then rename
            temp_path = checkpoint_path.with_suffix('.tmp')
            try:
                if self.enable_compression:
                    with gzip.open(temp_path, 'wb', compresslevel=6) as f:
                        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    with open(temp_path, 'wb') as f:
                        pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Atomic rename (replaces existing file if present)
                temp_path.replace(checkpoint_path)

                # Get file size for reporting
                file_size_mb = round(checkpoint_path.stat().st_size / (1024 * 1024), 2)
                chunk_str = f" (chunk {chunk_index})" if chunk_index is not None else ""
                compressed_str = " [compressed]" if self.enable_compression else ""
                self.logger.info(f"Saved checkpoint for stage '{stage}'{chunk_str} to {checkpoint_path.name} ({file_size_mb}MB{compressed_str})")
                return True
            except Exception as write_error:
                # Clean up temp file if it exists
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                raise write_error

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
            # Try loading with gzip first (for compressed checkpoints)
            try:
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                compressed = True
            except (OSError, gzip.BadGzipFile):
                # Not gzipped, try regular pickle
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                compressed = False

            if not isinstance(checkpoint, dict) or "metadata" not in checkpoint or "data" not in checkpoint:
                self.logger.error(f"Invalid checkpoint format in {checkpoint_path.name}")
                return None

            metadata = checkpoint["metadata"]
            data = checkpoint["data"]

            # Validate version (allow backwards compatibility with 1.0)
            checkpoint_version = metadata.get("version", "1.0")
            if checkpoint_version not in [CHECKPOINT_VERSION, "1.0"]:
                self.logger.warning(f"Checkpoint version mismatch: {checkpoint_version} vs {CHECKPOINT_VERSION}")

            compressed_str = " [compressed]" if compressed else ""
            self.logger.info(f"Loaded checkpoint from {checkpoint_path.name} (stage: {metadata.get('stage')}){compressed_str}")
            return metadata, data

        except Exception as e:
            # Log full error to file, but show clean message to user
            self.logger.debug(f"Failed to load checkpoint {checkpoint_path.name}: {e}", exc_info=True)
            self.logger.warning(f"Skipping corrupted checkpoint: {checkpoint_path.name}")
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

        # Find all checkpoint files (including mid-stage checkpoints)
        for ckpt_file in checkpoint_dir.glob("*.ckpt"):
            try:
                # Extract stage from filename
                # Format: {hash}_{stage}.ckpt or {hash}_{stage}_chunk{index}.ckpt
                filename_parts = ckpt_file.stem.split('_')
                if len(filename_parts) >= 2:
                    # Check if it's a chunked checkpoint
                    if len(filename_parts) >= 3 and filename_parts[2].startswith('chunk'):
                        stage = filename_parts[1]
                    else:
                        stage = filename_parts[1]
                else:
                    self.logger.warning(f"Unexpected checkpoint filename format: {ckpt_file.name}")
                    continue

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
                             stages_to_run: Optional[List[str]] = None) -> Optional[Tuple[str, Dict[str, Any], Optional[int]]]:
        """
        Get the best checkpoint to resume from based on resume_mode.

        Args:
            input_path: Input file being processed
            config: Current pipeline configuration
            stages_to_run: List of stages to run

        Returns:
            Tuple of (stage_name, data, chunk_index) if resuming, None otherwise
            chunk_index is None for stage-level checkpoints, or an int for mid-stage checkpoints
        """
        if self.resume_mode == "disabled":
            return None

        compatible = self.find_compatible_checkpoints(input_path, config, stages_to_run)

        if not compatible:
            self.logger.debug("No compatible checkpoints found")
            return None

        if self.resume_mode == "auto":
            # Use the latest compatible checkpoint (including mid-stage ones)
            stage, ckpt_path, metadata = compatible[0]
            result = self.load(ckpt_path)
            if result:
                _, data = result
                chunk_index = metadata.get("chunk_index")
                chunk_str = f" (chunk {chunk_index})" if chunk_index is not None else ""
                ConsoleOutput.info(f"Resuming from checkpoint: stage '{stage}'{chunk_str} ({metadata['timestamp']})")
                return stage, data, chunk_index
            return None

        elif self.resume_mode == "interactive":
            # Present options to user
            return self._interactive_checkpoint_selection(compatible)

        return None

    def _interactive_checkpoint_selection(self,
                                         compatible_checkpoints: List[Tuple[str, Path, Dict[str, Any]]]) -> Optional[Tuple[str, Dict[str, Any], Optional[int]]]:
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
                chunk_index = metadata.get("chunk_index")
                ConsoleOutput.info(f"Auto-selecting checkpoint: stage '{stage}'")
                return stage, data, chunk_index
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

    def generate_checkpoint_report(self, checkpoint_path: Path) -> str:
        """
        Generate a detailed human-readable report for a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Formatted report string
        """
        result = self.load(checkpoint_path)
        if not result:
            return f"Error: Could not load checkpoint {checkpoint_path.name}"

        metadata, data = result

        # Get file size
        file_size_mb = round(checkpoint_path.stat().st_size / (1024 * 1024), 2)

        # Build report
        lines = []
        lines.append("=" * 80)
        lines.append(f"CHECKPOINT REPORT: {checkpoint_path.name}")
        lines.append("=" * 80)
        lines.append("")

        # Basic info
        lines.append("## Basic Information")
        lines.append(f"Version:          {metadata.get('version', 'Unknown')}")
        lines.append(f"Timestamp:        {metadata.get('timestamp', 'Unknown')}")
        lines.append(f"Stage:            {metadata.get('stage', 'Unknown')}")
        lines.append(f"Checkpoint Size:  {file_size_mb} MB")
        lines.append("")

        # Input file info
        lines.append("## Input File")
        lines.append(f"File:             {metadata.get('input_file', 'Unknown')}")
        lines.append(f"File Stem:        {metadata.get('input_stem', 'Unknown')}")
        lines.append(f"Input Size:       {metadata.get('input_size_mb', 0)} MB")
        lines.append("")

        # Model configuration
        config = metadata.get('config', {})
        text_model = config.get('text_model', {})
        if isinstance(text_model, dict):
            lines.append("## Text Model")
            lines.append(f"Provider:         {text_model.get('provider', 'Unknown')}")
            lines.append(f"Model:            {text_model.get('specifier', 'Unknown')}")
            lines.append(f"Full Name:        {text_model.get('full_name', 'Unknown')}")
        else:
            lines.append("## Text Model")
            lines.append(f"Model:            {text_model}")
        lines.append(f"Mode:             {config.get('mode', 'Unknown')}")
        lines.append("")

        # Hyperparameters
        hyperparams = metadata.get('hyperparameters', {})
        if hyperparams:
            lines.append("## Hyperparameters")
            lines.append(f"Temperature:      {hyperparams.get('temperature', 'N/A')}")
            lines.append(f"Top P:            {hyperparams.get('top_p', 'N/A')}")
            lines.append(f"Top K:            {hyperparams.get('top_k', 'N/A')}")
            lines.append(f"Max Tokens:       {hyperparams.get('max_new_tokens', 'N/A')}")
            lines.append(f"Repetition Penalty: {hyperparams.get('repetition_penalty', 'N/A')}")
            lines.append("")

        # Chunking configuration
        chunking = config.get('chunking', {})
        if chunking:
            lines.append("## Chunking Settings")
            lines.append(f"Strategy:         {chunking.get('strategy', 'Unknown')}")
            lines.append(f"Chunk Size:       {chunking.get('chunk_size', 'Unknown')}")
            lines.append(f"Chunk Overlap:    {chunking.get('chunk_overlap', 'Unknown')}")
            lines.append("")

        # Memory configuration
        memory_config = metadata.get('memory_config', {})
        quant = memory_config.get('quantization', {})
        if quant:
            lines.append("## Quantization Settings")
            lines.append(f"Method:           {quant.get('method', 'None')}")
            lines.append(f"Compute Dtype:    {quant.get('compute_dtype', 'N/A')}")
            lines.append(f"Quant Type:       {quant.get('quant_type', 'N/A')}")
            lines.append("")

        layer_split = memory_config.get('layer_split', {})
        if layer_split and layer_split.get('enabled'):
            lines.append("## Layer Split/Offloading")
            lines.append(f"GPU Layers:       {layer_split.get('gpu_layers', 'Auto')}")
            lines.append(f"Max GPU Memory:   {layer_split.get('max_gpu_memory', 'N/A')}")
            lines.append(f"Max CPU Memory:   {layer_split.get('max_cpu_memory', 'N/A')}")
            lines.append(f"Low CPU Mem:      {layer_split.get('low_cpu_mem_usage', False)}")
            lines.append("")

        # Audio configuration (if applicable)
        audio_config = config.get('audio', {})
        if audio_config:
            lines.append("## Audio Settings")
            lines.append(f"Provider:         {audio_config.get('provider', 'Unknown')}")
            lines.append(f"Model:            {audio_config.get('model', 'Unknown')}")
            lines.append(f"Sample Rate:      {audio_config.get('sample_rate', 'N/A')} Hz")
            lines.append(f"Output Format:    {audio_config.get('output_format', 'Unknown')}")
            lines.append(f"Speed:            {audio_config.get('speed', 1.0)}x")
            lines.append(f"Pitch Shift:      {audio_config.get('pitch_shift', 0)} semitones")
            lines.append("")

            # Intermediate audio file (if saved)
            if data.get('intermediate_audio_path'):
                lines.append("## Intermediate Audio")
                lines.append(f"Audio File:       {data.get('intermediate_audio_path')}")
                intermediate_path = Path(data.get('intermediate_audio_path'))
                if intermediate_path.exists():
                    audio_size = round(intermediate_path.stat().st_size / (1024 * 1024), 2)
                    lines.append(f"Audio Size:       {audio_size} MB")
                lines.append("")

        # Checkpoint settings
        ckpt_settings = config.get('checkpointing', {})
        if ckpt_settings:
            lines.append("## Checkpoint Settings")
            lines.append(f"Enabled:          {ckpt_settings.get('enabled', False)}")
            lines.append(f"Interval:         Every {ckpt_settings.get('interval', 10)} chunks")
            lines.append(f"Resume Mode:      {ckpt_settings.get('resume_mode', 'auto')}")
            lines.append("")

        # Data payload info
        lines.append("## Data Payload")
        lines.append(f"Keys:             {', '.join(metadata.get('data_keys', []))}")

        # Chunk information (if available)
        chunk_index = metadata.get('chunk_index')
        if chunk_index is not None:
            lines.append(f"Chunk Index:      {chunk_index}")

            # Get chunk stats from data if available
            if 'chunks' in data:
                lines.append(f"Total Chunks:     {len(data.get('chunks', []))}")
            if 'processed_chunks' in data:
                processed_count = len([c for c in data.get('processed_chunks', []) if c and c != '[Skipped - loaded from checkpoint]'])
                lines.append(f"Processed Chunks: {processed_count}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def list_all_checkpoints(self) -> List[Tuple[Path, Dict[str, Any]]]:
        """
        List all checkpoints with their metadata.

        Returns:
            List of tuples (checkpoint_path, metadata)
        """
        checkpoints = []

        if not self.base_dir.exists():
            return checkpoints

        for ckpt_file in self.base_dir.rglob("*.ckpt"):
            result = self.load(ckpt_file)
            if result:
                metadata, _ = result
                checkpoints.append((ckpt_file, metadata))

        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)

        return checkpoints

    def cleanup_corrupted_checkpoints(self) -> int:
        """
        Find and delete corrupted checkpoint files.

        Returns:
            Number of corrupted checkpoints deleted
        """
        deleted_count = 0

        if not self.base_dir.exists():
            return 0

        for ckpt_file in self.base_dir.rglob("*.ckpt"):
            try:
                result = self.load(ckpt_file)
                if result is None:
                    # Corrupted or invalid checkpoint
                    self.logger.warning(f"Deleting corrupted checkpoint: {ckpt_file.name}")
                    ckpt_file.unlink()
                    deleted_count += 1
            except Exception as e:
                # Failed to load - likely corrupted
                self.logger.warning(f"Deleting corrupted checkpoint {ckpt_file.name}: {e}")
                try:
                    ckpt_file.unlink()
                    deleted_count += 1
                except Exception as del_error:
                    self.logger.error(f"Failed to delete corrupted checkpoint: {del_error}")

        if deleted_count > 0:
            self.logger.info(f"Cleaned up {deleted_count} corrupted checkpoint(s)")

        return deleted_count

    def get_checkpoint_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all checkpoints.

        Returns:
            Dictionary with checkpoint statistics
        """
        checkpoints = self.list_all_checkpoints()

        # Calculate statistics
        total_checkpoints = len(checkpoints)
        total_size_bytes = sum(ckpt[0].stat().st_size for ckpt in checkpoints)
        total_size_mb = round(total_size_bytes / (1024 * 1024), 2)

        # Group by input file
        by_input = {}
        for ckpt_path, metadata in checkpoints:
            input_file = metadata.get('input_stem', 'Unknown')
            if input_file not in by_input:
                by_input[input_file] = []
            by_input[input_file].append((ckpt_path, metadata))

        # Group by stage
        by_stage = {}
        for ckpt_path, metadata in checkpoints:
            stage = metadata.get('stage', 'Unknown')
            if stage not in by_stage:
                by_stage[stage] = 0
            by_stage[stage] += 1

        return {
            "total_checkpoints": total_checkpoints,
            "total_size_mb": total_size_mb,
            "by_input_file": {k: len(v) for k, v in by_input.items()},
            "by_stage": by_stage,
            "checkpoint_dir": str(self.base_dir),
        }
