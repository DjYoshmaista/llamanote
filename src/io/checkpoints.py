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

# Define which config fields can have fuzzy matching (allow minor differences)
FUZZY_MATCH_FIELDS = {
    "system_prompt": 0.9,  # 90% similarity required
    "chunk_size": 0.1,  # Allow 10% variation
    "chunk_overlap": 0.2,  # Allow 20% variation
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

        # Initialize model abbreviation manager for descriptive checkpoint naming
        try:
            from .model_abbreviations import ModelAbbreviationManager
            self.model_abbrev_mgr = ModelAbbreviationManager()
            self.logger.debug("ModelAbbreviationManager initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize ModelAbbreviationManager: {e}")
            self.model_abbrev_mgr = None

        # Initialize checkpoint registry for fast discovery
        try:
            from .checkpoint_registry import CheckpointRegistry
            self.registry = CheckpointRegistry(checkpoint_dir=self.base_dir)
            self.logger.debug("CheckpointRegistry initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize CheckpointRegistry: {e}")
            self.registry = None

        # Create checkpoint directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Track archival milestones for current session (Phase 2 implementation)
        # Key: (input_stem, text_model, audio_model) -> set of saved milestone percentages
        self.archival_milestones: Dict[Tuple[str, str, str], set] = {}

        # Milestone thresholds for archival checkpoints (20%, 40%, 60%, 80%, 100%)
        self.milestone_thresholds = [20, 40, 60, 80, 100]

    # ==================== Audio Compression Methods (Phase 1) ====================

    def _compress_audio_array(self, audio: 'np.ndarray', sample_rate: int) -> bytes:
        """
        Compress audio numpy array to FLAC format (lossless compression).

        FLAC provides ~50-60% space savings while maintaining identical audio quality.

        Args:
            audio: Audio numpy array (float32 or int16)
            sample_rate: Sample rate in Hz

        Returns:
            Compressed audio as bytes (FLAC format)
        """
        try:
            import soundfile as sf
            import io
            import numpy as np

            # Create in-memory buffer
            buffer = io.BytesIO()

            # Write audio to buffer as FLAC
            # FLAC24 provides best compression for float data
            sf.write(buffer, audio, sample_rate, format='FLAC', subtype='PCM_24')

            # Get bytes
            buffer.seek(0)
            compressed_bytes = buffer.read()

            self.logger.debug(f"Compressed audio: {audio.nbytes} bytes → {len(compressed_bytes)} bytes "
                            f"({len(compressed_bytes)/audio.nbytes*100:.1f}% of original)")

            return compressed_bytes

        except Exception as e:
            self.logger.error(f"Failed to compress audio array: {e}", exc_info=True)
            raise

    def _decompress_audio_array(self, compressed_bytes: bytes) -> Tuple['np.ndarray', int]:
        """
        Decompress FLAC audio bytes back to numpy array.

        Args:
            compressed_bytes: FLAC-compressed audio bytes

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        try:
            import soundfile as sf
            import io

            # Create buffer from bytes
            buffer = io.BytesIO(compressed_bytes)

            # Read audio from buffer
            audio, sample_rate = sf.read(buffer)

            self.logger.debug(f"Decompressed audio: {len(compressed_bytes)} bytes → {audio.nbytes} bytes")

            return audio, sample_rate

        except Exception as e:
            self.logger.error(f"Failed to decompress audio array: {e}", exc_info=True)
            raise

    def _compress_audio_arrays_in_checkpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress audio arrays in checkpoint data before saving.

        Args:
            data: Checkpoint data dict (may contain 'audio_arrays')

        Returns:
            Modified data dict with compressed audio (modifies in-place, also returns for chaining)
        """
        if 'audio_arrays' not in data or not data['audio_arrays']:
            return data

        import numpy as np

        audio_arrays = data['audio_arrays']
        compressed_audio_list = []

        # Get sample rate from data if available, fallback to 22050 Hz
        sample_rate = data.get('sample_rate', 22050)

        total_original_bytes = 0
        total_compressed_bytes = 0

        for idx, audio_array in enumerate(audio_arrays):
            if isinstance(audio_array, np.ndarray) and audio_array.size > 0:
                try:
                    compressed_bytes = self._compress_audio_array(audio_array, sample_rate)
                    compressed_audio_list.append({
                        'compressed_bytes': compressed_bytes,
                        'sample_rate': sample_rate,
                        'original_shape': audio_array.shape,
                        'original_dtype': str(audio_array.dtype)
                    })

                    total_original_bytes += audio_array.nbytes
                    total_compressed_bytes += len(compressed_bytes)

                except Exception as e:
                    self.logger.warning(f"Failed to compress audio array {idx}: {e}")
                    # Fallback: keep original array
                    compressed_audio_list.append(audio_array)
            else:
                # Not a numpy array or empty, keep as-is
                compressed_audio_list.append(audio_array)

        # Replace audio_arrays with compressed version
        data['audio_arrays'] = compressed_audio_list
        data['_audio_compressed'] = True  # Mark as compressed for decompression later

        if total_original_bytes > 0:
            compression_ratio = total_compressed_bytes / total_original_bytes
            self.logger.info(f"Compressed {len(audio_arrays)} audio arrays: "
                           f"{total_original_bytes/(1024*1024):.1f} MB → "
                           f"{total_compressed_bytes/(1024*1024):.1f} MB "
                           f"({compression_ratio*100:.1f}% of original)")

        return data

    def _decompress_audio_arrays_in_checkpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decompress audio arrays after loading checkpoint.

        Args:
            data: Loaded checkpoint data (may have compressed audio)

        Returns:
            Modified data dict with decompressed audio arrays (modifies in-place, also returns)
        """
        if '_audio_compressed' not in data or not data.get('_audio_compressed'):
            # Not compressed, return as-is
            return data

        if 'audio_arrays' not in data:
            return data

        import numpy as np

        compressed_audio_list = data['audio_arrays']
        decompressed_audio_list = []

        for idx, item in enumerate(compressed_audio_list):
            if isinstance(item, dict) and 'compressed_bytes' in item:
                try:
                    audio_array, sample_rate = self._decompress_audio_array(item['compressed_bytes'])
                    decompressed_audio_list.append(audio_array)
                except Exception as e:
                    self.logger.error(f"Failed to decompress audio array {idx}: {e}", exc_info=True)
                    # Critical error - cannot recover audio data
                    decompressed_audio_list.append(None)
            else:
                # Not compressed format (fallback for old checkpoints), keep as-is
                decompressed_audio_list.append(item)

        data['audio_arrays'] = decompressed_audio_list
        del data['_audio_compressed']  # Remove compression flag

        self.logger.info(f"Decompressed {len(decompressed_audio_list)} audio arrays")

        return data

    # ==================== Hash Generation Methods ====================

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

    def calculate_pipeline_completion(
        self,
        current_stage: str,
        current_chunk: int,
        total_chunks: int,
        stage_weights: Dict[str, float],
        completed_stages: List[str]
    ) -> int:
        """
        Calculate overall pipeline completion percentage using weighted stages.

        This method computes progress by:
        1. Summing the weights of all completed stages
        2. Adding the partial weight of the current stage based on chunk progress
        3. Dividing by total weight to get percentage

        Args:
            current_stage: Name of the current pipeline stage
            current_chunk: Current chunk index (0-based or 1-based, just needs to be consistent)
            total_chunks: Total number of chunks in current stage
            stage_weights: Dictionary mapping stage names to processing time weights
            completed_stages: List of stage names that have been completed

        Returns:
            Completion percentage as integer (0-100)

        Example:
            >>> calculate_pipeline_completion(
            ...     current_stage="process",
            ...     current_chunk=20,
            ...     total_chunks=55,
            ...     stage_weights=DEFAULT_STAGE_WEIGHTS,
            ...     completed_stages=["extract", "preprocess", "chunk"]
            ... )
            15  # 15% complete
        """
        # Sum weight of completed stages
        completed_weight = sum(stage_weights.get(s, 0.0) for s in completed_stages)

        # Add partial weight of current stage
        current_stage_weight = stage_weights.get(current_stage, 0.0)
        if total_chunks > 0:
            stage_progress = current_chunk / total_chunks
        else:
            stage_progress = 0.0
        current_weight = current_stage_weight * stage_progress

        # Calculate percentage
        total_weight = sum(stage_weights.values())
        if total_weight > 0:
            completion_pct = (completed_weight + current_weight) / total_weight * 100
        else:
            # No weights defined, fallback to 0%
            completion_pct = 0.0

        # Round to nearest integer and clamp to 0-100 range
        return max(0, min(100, round(completion_pct)))

    def _should_create_archival_checkpoint(self, input_path: Path, config: PipelineConfig,
                                          completion_pct: int) -> Optional[int]:
        """
        Determine if current checkpoint should be saved as archival milestone.

        Archival checkpoints are preserved forever for diagnostics and reference.
        Only the FIRST checkpoint to reach each milestone threshold is saved as archival.

        Args:
            input_path: Input file path
            config: Pipeline configuration
            completion_pct: Current completion percentage (0-100)

        Returns:
            Milestone percentage to save (20, 40, 60, 80, 100) or None if not a milestone
        """
        # Create session key (file + models)
        file_stem = input_path.stem
        text_model = config.model_specifier or "unknown"
        audio_model = "unknown"

        # Get audio model name from config
        if hasattr(config, 'audio_config') and config.audio_config:
            if hasattr(config.audio_config, 'model_name'):
                audio_model = config.audio_config.model_name or "unknown"
        elif hasattr(config, 'audio_specifier') and config.audio_specifier:
            audio_model = config.audio_specifier

        session_key = (file_stem, text_model, audio_model)

        # Initialize milestone tracking for this session if needed
        if session_key not in self.archival_milestones:
            self.archival_milestones[session_key] = set()

        saved_milestones = self.archival_milestones[session_key]

        # Check if we've crossed a new milestone threshold
        for milestone in self.milestone_thresholds:
            if completion_pct >= milestone and milestone not in saved_milestones:
                # First checkpoint >= this milestone - save as archival
                saved_milestones.add(milestone)
                self.logger.info(f"✓ Archival milestone reached: {milestone}% (actual: {completion_pct}%)")
                return milestone

        return None

    def _get_checkpoint_path(self, input_path: Path, config: PipelineConfig, stage: str,
                            timestamp: Optional[str] = None, chunk_index: Optional[int] = None,
                            total_chunks: Optional[int] = None, completion_pct: Optional[int] = None,
                            checkpoint_type: str = "normal") -> Path:
        """
        Get the full path for a checkpoint file with descriptive naming.

        Args:
            input_path: Input file path
            config: Pipeline configuration
            stage: Stage name
            timestamp: Optional timestamp for hash
            chunk_index: Optional chunk index for mid-stage checkpoints
            total_chunks: Optional total number of chunks (for progress display)
            completion_pct: Optional overall pipeline completion percentage (0-100)
            checkpoint_type: Type of checkpoint - "normal", "rolling", or "archival"

        Returns:
            Path to checkpoint file

        Filename formats:
            Rolling checkpoint (audio stage, always overwritten):
                {filename}_{ext}-{text_model}-{audio_model}-ROLLING-audio.ckpt
                Example: BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-ROLLING-audio.ckpt

            Archival checkpoint (audio stage, milestone preservation):
                {filename}_{ext}-{text_model}-{audio_model}-ARCHIVE-{pct}pct-audio.ckpt
                Example: BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-ARCHIVE-20pct-audio.ckpt

            New format (with completion_pct):
                {filename}_{ext}-{text_model}-{audio_model}-chk{N}-{pct}pct.ckpt
                Example: BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-chk0020-15pct.ckpt

            Legacy format (for compatibility):
                {filename}_{stage}_{progress}_{timestamp}.ckpt
                Example: paper_process_chunk040of055_20251106_142151.ckpt
        """
        input_hash = self._generate_input_hash(input_path)
        config_hash = self._generate_config_hash(config, stage, timestamp)

        checkpoint_dir = self.base_dir / input_hash
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get input file stem and extension
        file_stem = input_path.stem
        file_ext = input_path.suffix.lstrip('.') if input_path.suffix else 'unknown'

        # Handle rolling and archival checkpoints for audio stage
        if stage == "audio" and checkpoint_type in ["rolling", "archival"] and self.model_abbrev_mgr is not None:
            # Get model abbreviations
            text_model = config.model_specifier or "unknown"
            audio_model_name = "unknown"
            if hasattr(config, 'audio_config') and config.audio_config and hasattr(config.audio_config, 'model_name'):
                audio_model_name = config.audio_config.model_name or "unknown"
            elif hasattr(config, 'audio_specifier') and config.audio_specifier:
                audio_model_name = config.audio_specifier

            text_abbrev = self.model_abbrev_mgr.abbreviate_text_model(text_model)
            audio_abbrev = self.model_abbrev_mgr.abbreviate_audio_model(audio_model_name)

            if checkpoint_type == "rolling":
                # Rolling checkpoint - always same filename (overwriteable)
                filename = f"{file_stem}_{file_ext}-{text_abbrev}-{audio_abbrev}-ROLLING-audio.ckpt"
            else:  # archival
                # Archival checkpoint - includes milestone percentage
                if completion_pct is None:
                    completion_pct = 0
                filename = f"{file_stem}_{file_ext}-{text_abbrev}-{audio_abbrev}-ARCHIVE-{completion_pct}pct-audio.ckpt"

            return checkpoint_dir / filename

        # Determine if we should use new naming format
        use_new_format = (completion_pct is not None and self.model_abbrev_mgr is not None)

        if use_new_format:
            # NEW FORMAT: {file}_{ext}-{text_model}-{audio_model}-chk{N}-{pct}pct.ckpt

            # Get model abbreviations
            text_model = config.model_specifier or "unknown"
            audio_model_name = "unknown"
            if hasattr(config, 'audio_config') and config.audio_config and hasattr(config.audio_config, 'model_name'):
                audio_model_name = config.audio_config.model_name or "unknown"
            elif hasattr(config, 'audio_specifier') and config.audio_specifier:
                audio_model_name = config.audio_specifier

            text_abbrev = self.model_abbrev_mgr.abbreviate_text_model(text_model)
            audio_abbrev = self.model_abbrev_mgr.abbreviate_audio_model(audio_model_name)

            # Build new format filename
            if chunk_index is not None:
                # Mid-stage checkpoint with chunk and completion percentage
                filename = f"{file_stem}_{file_ext}-{text_abbrev}-{audio_abbrev}-chk{chunk_index:04d}-{completion_pct}pct.ckpt"
            else:
                # Stage-level checkpoint (no chunk number, just stage)
                filename = f"{file_stem}_{file_ext}-{text_abbrev}-{audio_abbrev}-{stage}.ckpt"
        else:
            # LEGACY FORMAT: {filename}_{stage}_{progress}_{timestamp}.ckpt

            # Generate timestamp for filename
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Build legacy filename with progress information
            if chunk_index is not None and total_chunks is not None:
                # Show progress: chunk040of055
                progress = f"chunk{chunk_index:04d}of{total_chunks:04d}"
                filename = f"{file_stem}_{stage}_{progress}_{ts}.ckpt"
            elif chunk_index is not None:
                # Only chunk index known
                filename = f"{file_stem}_{stage}_chunk{chunk_index:04d}_{ts}.ckpt"
            else:
                # Stage-level checkpoint (no chunks)
                filename = f"{file_stem}_{stage}_{ts}.ckpt"

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

        # Store FULL configuration as serializable dict for restoration
        metadata["full_config"] = self._serialize_full_config(config)

        return metadata

    def _serialize_full_config(self, config: PipelineConfig) -> Dict[str, Any]:
        """
        Serialize complete PipelineConfig to dictionary for checkpoint restoration.

        Args:
            config: Pipeline configuration to serialize

        Returns:
            Dictionary containing all config values in serializable format
        """
        full_config = {}

        try:
            # Try dataclass conversion first
            if hasattr(config, '__dataclass_fields__'):
                full_config = asdict(config)
            else:
                # Manual conversion for non-dataclass configs
                for key, value in config.__dict__.items():
                    full_config[key] = self._serialize_value(value)

            # Convert specific known types
            if 'mode' in full_config:
                full_config['mode'] = str(full_config['mode'])
            if 'chunking_strategy' in full_config:
                full_config['chunking_strategy'] = str(full_config['chunking_strategy'])
            if 'output_dir' in full_config and full_config['output_dir']:
                full_config['output_dir'] = str(full_config['output_dir'])

        except Exception as e:
            self.logger.warning(f"Could not fully serialize config: {e}")
            # Return partial config as fallback
            full_config = {"_serialization_error": str(e)}

        return full_config

    def _serialize_value(self, value: Any) -> Any:
        """Helper to serialize various types to JSON-compatible format."""
        if isinstance(value, Path):
            return str(value)
        elif hasattr(value, '__dataclass_fields__'):
            return asdict(value)
        elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            # Try to convert object to dict
            try:
                return {k: self._serialize_value(v) for k, v in value.__dict__.items()}
            except:
                return str(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return value

    def _save_single_checkpoint(self, input_path: Path, config: PipelineConfig, stage: str,
                               data: Dict[str, Any], chunk_index: Optional[int] = None,
                               total_chunks: Optional[int] = None, completion_pct: Optional[int] = None,
                               config_uuid: Optional[str] = None, checkpoint_type: str = "normal") -> bool:
        """
        Save a single checkpoint file (helper for dual checkpoint saving).

        Args:
            input_path: Path to the input file being processed
            config: Pipeline configuration
            stage: Stage name
            data: Data payload dictionary to save
            chunk_index: Optional chunk index
            total_chunks: Optional total number of chunks
            completion_pct: Optional completion percentage
            config_uuid: Optional UUID linking to saved configuration
            checkpoint_type: "normal", "rolling", or "archival"

        Returns:
            True if save succeeded
        """
        try:
            # Get checkpoint path based on type
            checkpoint_path = self._get_checkpoint_path(
                input_path, config, stage,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                completion_pct=completion_pct,
                checkpoint_type=checkpoint_type
            )

            # Create metadata
            data_keys = list(data.keys())
            metadata = self._create_metadata(input_path, config, stage, data_keys)

            # Add checkpoint type to metadata
            metadata["checkpoint_type"] = checkpoint_type

            # Add config UUID if provided
            if config_uuid:
                metadata["config_uuid"] = config_uuid

            # Add chunk info
            if chunk_index is not None:
                metadata["chunk_index"] = chunk_index
            if total_chunks is not None:
                metadata["total_chunks"] = total_chunks
            if completion_pct is not None:
                metadata["completion_pct"] = completion_pct

            # Create checkpoint object
            checkpoint = {
                "metadata": metadata,
                "data": data
            }

            # Save with atomic write
            temp_path = checkpoint_path.with_suffix('.tmp')

            if self.enable_compression:
                with gzip.open(temp_path, 'wb', compresslevel=6) as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic rename (replaces existing file if rolling checkpoint)
            temp_path.replace(checkpoint_path)

            # Get file size
            compressed_size = checkpoint_path.stat().st_size
            file_size_mb = round(compressed_size / (1024 * 1024), 2)

            # Log save
            type_str = f" [{checkpoint_type}]" if checkpoint_type != "normal" else ""
            chunk_str = f" (chunk {chunk_index})" if chunk_index is not None else ""
            self.logger.info(f"Saved{type_str} checkpoint for stage '{stage}'{chunk_str} to {checkpoint_path.name} ({file_size_mb}MB)")

            # Create metadata sidecar file
            uncompressed_size = compressed_size if not self.enable_compression else int(compressed_size * 1.7)
            self._create_metadata_file(
                checkpoint_path=checkpoint_path,
                metadata=metadata,
                checkpoint_size_bytes=uncompressed_size,
                compressed_size_bytes=compressed_size,
                checkpoint_hash=None
            )

            # Update registry
            if self.registry is not None:
                self._update_registry_entry(
                    checkpoint_path=checkpoint_path,
                    metadata=metadata,
                    input_path=input_path,
                    compressed_size_mb=file_size_mb
                )

            return True

        except Exception as e:
            self.logger.error(f"Failed to save {checkpoint_type} checkpoint: {e}", exc_info=True)
            return False

    def save(self,
             input_path: Path,
             config: PipelineConfig,
             stage: str,
             data: Dict[str, Any],
             chunk_index: Optional[int] = None,
             total_chunks: Optional[int] = None,
             config_uuid: Optional[str] = None) -> bool:
        """
        Save checkpoint(s) with smart lifecycle management.

        For audio stage:
        - Always saves rolling checkpoint (overwrites previous)
        - Additionally saves archival checkpoint if milestone reached
        - Compresses audio arrays with FLAC before saving (lossless, ~50-60% space savings)

        For other stages:
        - Normal checkpoint behavior

        Args:
            input_path: Path to the input file being processed
            config: Pipeline configuration
            stage: Stage name
            data: Data payload dictionary to save
            chunk_index: Optional chunk index for mid-stage checkpoints
            total_chunks: Optional total number of chunks (for progress display)
            config_uuid: Optional UUID linking to saved configuration

        Returns:
            True if save succeeded, False otherwise
        """
        self.logger.info(f"=== CheckpointManager.save ENTRY: stage={stage}, chunk={chunk_index} ===")
        try:
            # Calculate completion percentage
            completion_pct = None
            if chunk_index is not None and total_chunks is not None:
                try:
                    from ..config.settings import DEFAULT_STAGE_WEIGHTS
                    completed_stages = data.get('completed_stages', [])
                    completion_pct = self.calculate_pipeline_completion(
                        current_stage=stage,
                        current_chunk=chunk_index,
                        total_chunks=total_chunks,
                        stage_weights=DEFAULT_STAGE_WEIGHTS,
                        completed_stages=completed_stages
                    )
                    self.logger.debug(f"Calculated completion: {completion_pct}%")
                except Exception as e:
                    self.logger.debug(f"Could not calculate completion percentage: {e}")
                    completion_pct = None

            # Compress audio arrays if present (Phase 1 - FLAC compression)
            if stage == "audio":
                self.logger.info("Compressing audio arrays with FLAC...")
                data = self._compress_audio_arrays_in_checkpoint(data)

            # Determine checkpoint type(s) to save
            if stage == "audio" and completion_pct is not None:
                # Check if this is an archival milestone
                milestone_pct = self._should_create_archival_checkpoint(input_path, config, completion_pct)

                # Always save rolling checkpoint (overwrites previous)
                self.logger.info("Saving rolling audio checkpoint...")
                success_rolling = self._save_single_checkpoint(
                    input_path, config, stage, data,
                    chunk_index, total_chunks, completion_pct,
                    config_uuid, checkpoint_type="rolling"
                )

                # Additionally save archival checkpoint if milestone
                if milestone_pct is not None:
                    self.logger.info(f"Saving archival checkpoint at {milestone_pct}% milestone...")
                    success_archival = self._save_single_checkpoint(
                        input_path, config, stage, data,
                        chunk_index, total_chunks, milestone_pct,  # Use milestone percentage
                        config_uuid, checkpoint_type="archival"
                    )
                    return success_rolling and success_archival

                return success_rolling
            else:
                # Non-audio stages or no completion percentage: normal checkpoint behavior
                return self._save_single_checkpoint(
                    input_path, config, stage, data,
                    chunk_index, total_chunks, completion_pct,
                    config_uuid, checkpoint_type="normal"
                )

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for stage '{stage}': {e}", exc_info=True)
            return False

    def _create_metadata_file(
        self,
        checkpoint_path: Path,
        metadata: Dict[str, Any],
        checkpoint_size_bytes: int,
        compressed_size_bytes: int,
        checkpoint_hash: Optional[str] = None
    ) -> bool:
        """
        Create a JSON metadata sidecar file for fast checkpoint discovery.

        Args:
            checkpoint_path: Path to the checkpoint .ckpt file
            metadata: Metadata dictionary from checkpoint
            checkpoint_size_bytes: Uncompressed checkpoint size in bytes
            compressed_size_bytes: Compressed checkpoint file size in bytes
            checkpoint_hash: Optional SHA256 hash of checkpoint file

        Returns:
            True if metadata file created successfully, False otherwise
        """
        try:
            # Build enhanced metadata with file information
            enhanced_metadata = metadata.copy()

            # Add checkpoint file information
            enhanced_metadata["checkpoint_file"] = {
                "filename": checkpoint_path.name,
                "size_bytes": checkpoint_size_bytes,
                "size_mb": round(checkpoint_size_bytes / (1024 * 1024), 2),
                "compressed_size_bytes": compressed_size_bytes,
                "compressed_size_mb": round(compressed_size_bytes / (1024 * 1024), 2),
                "compression_ratio": round(compressed_size_bytes / checkpoint_size_bytes, 3) if checkpoint_size_bytes > 0 else 1.0,
            }

            # Add hash if provided
            if checkpoint_hash:
                enhanced_metadata["checkpoint_file"]["hash_sha256"] = checkpoint_hash

            # Generate metadata file path (.ckpt.meta.json)
            metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.meta.json')

            # Atomic write: write to temp file first
            temp_meta_path = metadata_path.with_suffix('.tmp')

            try:
                with open(temp_meta_path, 'w', encoding='utf-8') as f:
                    json.dump(enhanced_metadata, f, indent=2, default=str)

                # Atomic rename
                temp_meta_path.replace(metadata_path)

                self.logger.debug(f"Created metadata file: {metadata_path.name}")
                return True

            except Exception as write_error:
                # Clean up temp file if it exists
                if temp_meta_path.exists():
                    try:
                        temp_meta_path.unlink()
                    except:
                        pass
                raise write_error

        except Exception as e:
            # Metadata file creation failure should not break checkpoint saving
            self.logger.warning(f"Failed to create metadata file for {checkpoint_path.name}: {e}")
            return False

    def _update_registry_entry(
        self,
        checkpoint_path: Path,
        metadata: Dict[str, Any],
        input_path: Path,
        compressed_size_mb: float
    ) -> bool:
        """
        Update the checkpoint registry with a new or updated entry.

        Args:
            checkpoint_path: Path to the checkpoint file
            metadata: Checkpoint metadata
            input_path: Original input file path
            compressed_size_mb: Compressed checkpoint size in MB

        Returns:
            True if registry updated successfully
        """
        try:
            # Extract models from metadata
            text_model = "unknown"
            audio_model = "unknown"

            config = metadata.get("config", {})
            if isinstance(config.get("text_model"), dict):
                text_model = config["text_model"].get("specifier", "unknown")
            elif config.get("text_model"):
                text_model = str(config["text_model"])

            if isinstance(config.get("audio"), dict):
                audio_model = config["audio"].get("model", "unknown")

            # Build registry entry
            registry_entry = {
                "filename": checkpoint_path.name,
                "input_file": str(input_path),
                "input_stem": input_path.stem,
                "file_extension": input_path.suffix.lstrip('.') if input_path.suffix else "unknown",
                "stage": metadata.get("stage", "unknown"),
                "chunk_index": metadata.get("chunk_index"),
                "total_chunks": metadata.get("total_chunks"),
                "completion_pct": metadata.get("completion_pct"),
                "text_model": text_model,
                "audio_model": audio_model,
                "hyperparameter_preset": "default",  # Could extract from metadata if available
                "created_timestamp": metadata.get("timestamp", datetime.now().isoformat()),
                "size_mb": metadata.get("checkpoint_file", {}).get("size_mb", 0),
                "compressed_size_mb": compressed_size_mb,
                "hash": metadata.get("hash_info", {}).get("input_hash", ""),
                "status": "active"
            }

            # Add to registry
            self.registry.add_entry(registry_entry)
            self.logger.debug(f"Updated registry for {checkpoint_path.name}")
            return True

        except Exception as e:
            # Registry updates should not break checkpoint saving
            self.logger.warning(f"Failed to update registry: {e}")
            return False

    def load_metadata_from_file(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load metadata from JSON sidecar file (fast path, no pickle loading).

        Args:
            checkpoint_path: Path to the checkpoint .ckpt file

        Returns:
            Metadata dict if successful, None otherwise
        """
        metadata_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.meta.json')

        if not metadata_path.exists():
            self.logger.debug(f"No metadata file found for {checkpoint_path.name}")
            return None

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.logger.debug(f"Loaded metadata from {metadata_path.name}")
            return metadata
        except Exception as e:
            self.logger.warning(f"Failed to load metadata file {metadata_path.name}: {e}")
            return None

    def load_metadata_only(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load only the metadata from a checkpoint file (more efficient for discovery).

        Tries JSON sidecar file first (fast path), falls back to pickle extraction.

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            Metadata dict if successful, None otherwise
        """
        if not checkpoint_path.exists():
            return None

        # Fast path: Try JSON metadata file first
        metadata_from_json = self.load_metadata_from_file(checkpoint_path)
        if metadata_from_json is not None:
            return metadata_from_json

        # Fallback path: Extract from pickle (slower)
        self.logger.debug(f"No JSON metadata for {checkpoint_path.name}, extracting from pickle")
        try:
            # Try loading with gzip first (for compressed checkpoints)
            try:
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            except (OSError, gzip.BadGzipFile):
                # Not gzipped, try regular pickle
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)

            if isinstance(checkpoint, dict) and "metadata" in checkpoint:
                return checkpoint["metadata"]
            return None

        except Exception as e:
            self.logger.debug(f"Failed to load metadata from {checkpoint_path.name}: {e}")
            return None

    def load(self, checkpoint_path: Path) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Load a checkpoint file (both metadata and data).

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

            # Decompress audio arrays if present (Phase 1 - FLAC decompression)
            if metadata.get("stage") == "audio":
                self.logger.debug("Decompressing audio arrays from FLAC...")
                data = self._decompress_audio_arrays_in_checkpoint(data)

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

        # Check version (allow backwards compatibility with 1.0)
        checkpoint_version = checkpoint_metadata.get("version", "1.0")
        if checkpoint_version not in [CHECKPOINT_VERSION, "1.0"]:
            incompatibilities.append(f"Version mismatch: checkpoint v{checkpoint_version} vs current v{CHECKPOINT_VERSION}")

        # Get relevant fields for the resume stage itself (not future stages)
        # We only care that the checkpoint stage was created with compatible settings
        # Future stages can use different settings
        stage_order = ["extract", "preprocess", "chunk", "process", "filter", "format", "save", "audio"]
        try:
            resume_idx = stage_order.index(resume_from_stage)
        except ValueError:
            incompatibilities.append(f"Unknown stage: {resume_from_stage}")
            return False, incompatibilities

        # Only check relevant fields for the resume stage itself
        relevant_fields = set(STAGE_RELEVANT_FIELDS.get(resume_from_stage, []))

        # Check text model only if the resume stage requires it
        if "model_provider" in relevant_fields or "model_specifier" in relevant_fields:
            checkpoint_model_dict = checkpoint_config.get("text_model", {})
            if isinstance(checkpoint_model_dict, dict):
                checkpoint_model = checkpoint_model_dict.get("full_name", "")
            else:
                checkpoint_model = str(checkpoint_model_dict)
            current_model = f"{config.model_provider}:{config.model_specifier}"
            if checkpoint_model != current_model:
                incompatibilities.append(f"Text model mismatch: {checkpoint_model} vs {current_model}")

        # Check chunking settings if chunk stage or later
        if "chunking_strategy" in relevant_fields:
            checkpoint_chunking = checkpoint_config.get("chunking", {})
            if checkpoint_chunking:
                if str(checkpoint_chunking.get("strategy")) != str(config.chunking_strategy):
                    incompatibilities.append(f"Chunking strategy mismatch")

                # Allow fuzzy matching for chunk_size
                checkpoint_chunk_size = checkpoint_chunking.get("chunk_size")
                if checkpoint_chunk_size and checkpoint_chunk_size != config.chunk_size:
                    # Check if within acceptable variation
                    variation = abs(checkpoint_chunk_size - config.chunk_size) / checkpoint_chunk_size
                    if variation > FUZZY_MATCH_FIELDS.get("chunk_size", 0.1):
                        incompatibilities.append(f"Chunk size mismatch: {checkpoint_chunk_size} vs {config.chunk_size}")
                    else:
                        self.logger.info(f"Chunk size differs slightly ({checkpoint_chunk_size} vs {config.chunk_size}) but within tolerance")

                # Allow fuzzy matching for chunk_overlap
                checkpoint_overlap = checkpoint_chunking.get("chunk_overlap")
                if checkpoint_overlap and checkpoint_overlap != config.chunk_overlap:
                    variation = abs(checkpoint_overlap - config.chunk_overlap) / max(checkpoint_overlap, 1)
                    if variation > FUZZY_MATCH_FIELDS.get("chunk_overlap", 0.2):
                        incompatibilities.append(f"Chunk overlap mismatch: {checkpoint_overlap} vs {config.chunk_overlap}")
                    else:
                        self.logger.info(f"Chunk overlap differs slightly but within tolerance")

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

    def find_rolling_audio_checkpoint(self, input_path: Path, config: PipelineConfig) -> Optional[Path]:
        """
        Find the rolling audio checkpoint for current file/config (Phase 4 - Resume Logic).

        Rolling checkpoints provide the latest audio processing state and are preferred
        for resume operations. They are overwritten on each save, so only one exists per
        file+model+config combination.

        Args:
            input_path: Input file path
            config: Pipeline configuration

        Returns:
            Path to rolling checkpoint if exists, None otherwise
        """
        try:
            # Generate expected rolling checkpoint filename
            checkpoint_path = self._get_checkpoint_path(
                input_path, config, "audio",
                checkpoint_type="rolling"
            )

            if checkpoint_path.exists():
                self.logger.info(f"Found rolling audio checkpoint: {checkpoint_path.name}")
                return checkpoint_path

            self.logger.debug("No rolling audio checkpoint found")
            return None

        except Exception as e:
            self.logger.debug(f"Error finding rolling checkpoint: {e}")
            return None

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

                # Load only metadata for compatibility check (more efficient)
                metadata = self.load_metadata_only(ckpt_file)
                if metadata is None:
                    continue

                # Check if this checkpoint stage could be useful for the stages we want to run
                # A checkpoint is useful if:
                # 1. Its stage is in stages_to_run (we're resuming exactly that stage), OR
                # 2. Its stage comes before any stage in stages_to_run (provides data for later stages)
                if stages_to_run:
                    checkpoint_stage_idx = stage_order.index(stage) if stage in stage_order else -1
                    earliest_run_stage_idx = min([stage_order.index(s) for s in stages_to_run if s in stage_order], default=-1)

                    # Skip checkpoint if it's after all stages we want to run
                    if checkpoint_stage_idx > earliest_run_stage_idx:
                        continue

                # Check compatibility
                is_compatible, incompatibilities = self._check_stage_compatibility(metadata, config, stage)

                if is_compatible:
                    compatible_checkpoints.append((stage, ckpt_file, metadata))
                else:
                    self.logger.info(f"Checkpoint {ckpt_file.name} incompatible: {', '.join(incompatibilities)}")

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

    def find_checkpoints_from_registry(
        self,
        input_path: Path,
        filter_by_stage: Optional[str] = None,
        filter_by_model: Optional[str] = None,
        min_completion: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Find checkpoints using the registry (fast path).

        Args:
            input_path: Input file being processed
            filter_by_stage: Optional stage filter
            filter_by_model: Optional model abbreviation filter
            min_completion: Optional minimum completion percentage

        Returns:
            List of checkpoint info dictionaries from registry
        """
        if self.registry is None:
            self.logger.debug("Registry not available, falling back to filesystem scan")
            return []

        try:
            input_stem = input_path.stem

            # Get all checkpoints for this input
            all_checkpoints = self.registry.find_by_input(input_stem)

            if not all_checkpoints:
                return []

            # Apply filters
            filtered = all_checkpoints

            if filter_by_stage:
                filtered = [c for c in filtered if c.get('stage') == filter_by_stage]

            if filter_by_model:
                filtered = [c for c in filtered if
                           filter_by_model in c.get('text_model', '') or
                           filter_by_model in c.get('audio_model', '')]

            if min_completion is not None:
                filtered = [c for c in filtered if
                           int(c.get('completion_pct', 0) or 0) >= min_completion]

            # Filter to only 'active' status
            filtered = [c for c in filtered if c.get('status') == 'active']

            return filtered

        except Exception as e:
            self.logger.error(f"Registry-based search failed: {e}", exc_info=True)
            return []

    def get_best_checkpoint_from_registry(
        self,
        input_path: Path,
        prefer_latest: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best checkpoint for resume using registry (fast).

        Args:
            input_path: Input file being processed
            prefer_latest: If True, prefer highest completion. If False, prefer stage order.

        Returns:
            Checkpoint info dict or None if no checkpoints found
        """
        checkpoints = self.find_checkpoints_from_registry(input_path)

        if not checkpoints:
            return None

        if prefer_latest:
            # Sort by completion percentage (highest first)
            checkpoints_sorted = sorted(
                checkpoints,
                key=lambda x: int(x.get('completion_pct', 0) or 0),
                reverse=True
            )
        else:
            # Sort by timestamp (latest first)
            checkpoints_sorted = sorted(
                checkpoints,
                key=lambda x: x.get('created_timestamp', ''),
                reverse=True
            )

        return checkpoints_sorted[0] if checkpoints_sorted else None

    def get_checkpoint_selection_menu(
        self,
        input_path: Path
    ) -> List[Dict[str, Any]]:
        """
        Get formatted checkpoint list for interactive selection menu.

        Args:
            input_path: Input file being processed

        Returns:
            List of checkpoint info dicts sorted by completion (highest first)
        """
        checkpoints = self.find_checkpoints_from_registry(input_path)

        if not checkpoints:
            return []

        # Sort by completion percentage (highest first)
        checkpoints_sorted = sorted(
            checkpoints,
            key=lambda x: int(x.get('completion_pct', 0) or 0),
            reverse=True
        )

        return checkpoints_sorted

    def get_resume_checkpoint(self,
                             input_path: Path,
                             config: PipelineConfig,
                             stages_to_run: Optional[List[str]] = None) -> Optional[Tuple[str, Dict[str, Any], Optional[int], Optional[PipelineConfig]]]:
        """
        Get the best checkpoint to resume from based on resume_mode.

        Args:
            input_path: Input file being processed
            config: Current pipeline configuration
            stages_to_run: List of stages to run

        Returns:
            Tuple of (stage_name, data, chunk_index, restored_config) if resuming, None otherwise
            chunk_index is None for stage-level checkpoints, or an int for mid-stage checkpoints
            restored_config is the configuration from checkpoint (or None if not available)
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

                # Attempt to restore configuration from checkpoint
                restored_config = self.restore_config_from_checkpoint(metadata, config)

                return stage, data, chunk_index, restored_config
            return None

        elif self.resume_mode == "interactive":
            # Present options to user
            return self._interactive_checkpoint_selection(compatible)

        return None

    def restore_config_from_checkpoint(self, metadata: Dict[str, Any], current_config: PipelineConfig) -> Optional[PipelineConfig]:
        """
        Restore PipelineConfig from checkpoint metadata.

        Args:
            metadata: Checkpoint metadata containing full_config
            current_config: Current configuration (used as fallback)

        Returns:
            Restored configuration or None if restoration failed
        """
        try:
            full_config = metadata.get("full_config")
            if not full_config or "_serialization_error" in full_config:
                self.logger.warning("No complete config in checkpoint, using current config")
                return None

            # Create new config from stored values
            # Note: This is a simplified restoration - may need model-specific handling
            restored = PipelineConfig(**full_config)
            self.logger.info("Successfully restored configuration from checkpoint")
            return restored

        except Exception as e:
            self.logger.warning(f"Could not restore config from checkpoint: {e}")
            return None

    def calculate_remaining_stages(self, checkpoint_stage: str, all_stages: Optional[List[str]] = None) -> List[str]:
        """
        Calculate which stages remain to be completed after a checkpoint.

        Args:
            checkpoint_stage: The stage where the checkpoint was saved
            all_stages: Complete list of stages (defaults to standard pipeline stages)

        Returns:
            List of stages that still need to be run
        """
        if all_stages is None:
            all_stages = ["extract", "preprocess", "chunk", "process", "filter", "format", "save", "audio"]

        try:
            checkpoint_idx = all_stages.index(checkpoint_stage)
            # Return stages after the checkpoint stage
            remaining = all_stages[checkpoint_idx + 1:]
            self.logger.info(f"Checkpoint at '{checkpoint_stage}', remaining stages: {remaining}")
            return remaining
        except ValueError:
            self.logger.warning(f"Unknown checkpoint stage '{checkpoint_stage}', cannot calculate remaining stages")
            return all_stages

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

                    # Remove from registry
                    if self.registry is not None:
                        self.registry.remove_entry(ckpt_file.name)

                    # Also delete metadata file if it exists
                    meta_file = ckpt_file.with_suffix(ckpt_file.suffix + '.meta.json')
                    if meta_file.exists():
                        meta_file.unlink()

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

                    # Clean up registry entries for this input file
                    if self.registry is not None:
                        input_stem = input_path.stem
                        entries = self.registry.find_by_input(input_stem)
                        for entry in entries:
                            self.registry.remove_entry(entry.get("filename"))

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

                    # Re-initialize registry (creates empty registry file)
                    if self.registry is not None:
                        from .checkpoint_registry import CheckpointRegistry
                        self.registry = CheckpointRegistry(checkpoint_dir=self.base_dir)

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

    def regenerate_metadata_files(
        self,
        checkpoint_paths: Optional[List[Path]] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate .meta.json files for checkpoints that don't have them.

        Args:
            checkpoint_paths: Specific checkpoint paths to regenerate.
                             If None, scans all checkpoints in base_dir.
            progress_callback: Optional callback function(current, total, checkpoint_name)
                              called after processing each checkpoint.

        Returns:
            Dictionary with statistics:
            {
                "total_checked": int,
                "success_count": int,
                "error_count": int,
                "skipped_count": int,  # Already had metadata
                "errors": List[str]
            }
        """
        stats = {
            "total_checked": 0,
            "success_count": 0,
            "error_count": 0,
            "skipped_count": 0,
            "errors": []
        }

        # Determine which checkpoints to process
        if checkpoint_paths is None:
            # Scan all checkpoints in base directory
            if not self.base_dir.exists():
                self.logger.warning("Checkpoint directory does not exist")
                return stats

            checkpoint_paths = list(self.base_dir.rglob("*.ckpt"))

        stats["total_checked"] = len(checkpoint_paths)
        self.logger.info(f"Checking {stats['total_checked']} checkpoint(s) for metadata files")

        for idx, ckpt_path in enumerate(checkpoint_paths, 1):
            try:
                # Check if metadata file already exists
                metadata_path = ckpt_path.with_suffix(ckpt_path.suffix + '.meta.json')
                if metadata_path.exists():
                    self.logger.debug(f"Metadata already exists for {ckpt_path.name}, skipping")
                    stats["skipped_count"] += 1
                    if progress_callback:
                        progress_callback(idx, stats["total_checked"], ckpt_path.name)
                    continue

                # Load checkpoint to extract metadata
                self.logger.debug(f"Extracting metadata from {ckpt_path.name}")
                result = self.load(ckpt_path)

                if result is None:
                    error_msg = f"Failed to load checkpoint: {ckpt_path.name}"
                    self.logger.warning(error_msg)
                    stats["error_count"] += 1
                    stats["errors"].append(error_msg)
                    if progress_callback:
                        progress_callback(idx, stats["total_checked"], ckpt_path.name)
                    continue

                metadata, _ = result

                # Get file sizes
                compressed_size = ckpt_path.stat().st_size
                uncompressed_size = compressed_size if not self.enable_compression else int(compressed_size * 1.7)

                # Create metadata file
                success = self._create_metadata_file(
                    checkpoint_path=ckpt_path,
                    metadata=metadata,
                    checkpoint_size_bytes=uncompressed_size,
                    compressed_size_bytes=compressed_size,
                    checkpoint_hash=None
                )

                if success:
                    stats["success_count"] += 1
                    self.logger.debug(f"Created metadata for {ckpt_path.name}")
                else:
                    error_msg = f"Failed to create metadata for: {ckpt_path.name}"
                    stats["error_count"] += 1
                    stats["errors"].append(error_msg)

                # Call progress callback
                if progress_callback:
                    progress_callback(idx, stats["total_checked"], ckpt_path.name)

            except Exception as e:
                error_msg = f"Error processing {ckpt_path.name}: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                stats["error_count"] += 1
                stats["errors"].append(error_msg)
                if progress_callback:
                    progress_callback(idx, stats["total_checked"], ckpt_path.name)

        # Log summary
        self.logger.info(
            f"Metadata regeneration complete: "
            f"{stats['success_count']} created, "
            f"{stats['skipped_count']} skipped, "
            f"{stats['error_count']} errors"
        )

        return stats

    def find_checkpoints_without_metadata(self) -> List[Path]:
        """
        Find all checkpoints that don't have .meta.json sidecar files.

        Returns:
            List of checkpoint paths missing metadata files
        """
        missing = []

        if not self.base_dir.exists():
            return missing

        for ckpt_path in self.base_dir.rglob("*.ckpt"):
            metadata_path = ckpt_path.with_suffix(ckpt_path.suffix + '.meta.json')
            if not metadata_path.exists():
                missing.append(ckpt_path)

        return missing

    def is_legacy_checkpoint(self, checkpoint_path: Path) -> bool:
        """
        Check if a checkpoint uses the old naming schema.

        Legacy format: {hash}_{stage}_chunk{index}_{timestamp}.ckpt
        New format: {file}_{ext}-{text_model}-{audio_model}-chk{N}-{pct}pct.ckpt

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if checkpoint uses legacy naming
        """
        filename = checkpoint_path.stem

        # Legacy pattern: starts with 8-character hash
        # Example: abc12345_process_chunk0020of0055_20251106_142151
        import re
        legacy_pattern = r'^[a-f0-9]{8}_'

        return bool(re.match(legacy_pattern, filename))

    def find_legacy_checkpoints(self) -> List[Path]:
        """
        Find all checkpoints using the old naming schema.

        Returns:
            List of legacy checkpoint paths
        """
        legacy_checkpoints = []

        if not self.base_dir.exists():
            return legacy_checkpoints

        for ckpt_path in self.base_dir.rglob("*.ckpt"):
            if self.is_legacy_checkpoint(ckpt_path):
                legacy_checkpoints.append(ckpt_path)

        return legacy_checkpoints

    def verify_and_sync_registry(self) -> Dict[str, Any]:
        """
        Verify registry accuracy and sync with filesystem.

        This method should be called on startup to ensure registry is up-to-date.
        It will:
        1. Remove stale entries (checkpoints that no longer exist)
        2. Find new checkpoint files not in registry
        3. Report statistics

        Returns:
            Dictionary with verification results:
            {
                "registry_entries": int,
                "stale_removed": int,
                "new_found": int,
                "missing_metadata": int
            }
        """
        if self.registry is None:
            self.logger.warning("Registry not initialized, skipping verification")
            return {
                "registry_entries": 0,
                "stale_removed": 0,
                "new_found": 0,
                "missing_metadata": 0
            }

        results = {
            "registry_entries": 0,
            "stale_removed": 0,
            "new_found": 0,
            "missing_metadata": 0
        }

        try:
            # Step 1: Clean up stale entries
            stale_removed = self.registry.cleanup_missing_entries(self.base_dir)
            results["stale_removed"] = stale_removed

            # Step 2: Find new checkpoint files not in registry
            all_checkpoints = set()
            if self.base_dir.exists():
                for ckpt_file in self.base_dir.rglob("*.ckpt"):
                    all_checkpoints.add(ckpt_file.name)

            # Get registry filenames
            registry_entries = self.registry.load()
            results["registry_entries"] = len(registry_entries)
            registry_filenames = set(e.get("filename") for e in registry_entries if e.get("filename"))

            # Find new files
            new_files = all_checkpoints - registry_filenames
            results["new_found"] = len(new_files)

            if new_files:
                self.logger.info(f"Found {len(new_files)} checkpoint(s) not in registry")

            # Step 3: Check for missing metadata files
            missing_metadata_files = self.find_checkpoints_without_metadata()
            results["missing_metadata"] = len(missing_metadata_files)

            # Log summary
            if stale_removed > 0 or new_files or missing_metadata_files:
                self.logger.info(
                    f"Registry verification complete: "
                    f"{results['registry_entries']} entries, "
                    f"{stale_removed} stale removed, "
                    f"{len(new_files)} new found, "
                    f"{len(missing_metadata_files)} missing metadata"
                )

            return results

        except Exception as e:
            self.logger.error(f"Failed to verify registry: {e}", exc_info=True)
            return results
