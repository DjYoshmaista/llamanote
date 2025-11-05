# llamanote/io/checkpoint_config.py
"""
Checkpoint Configuration Management

Manages configuration files that are required to resume from checkpoints.
Each checkpoint requires specific settings to be loaded correctly.

System:
1. Generate UUID from checkpoint's required settings
2. Save configuration file with this UUID
3. Reuse configuration files when UUIDs match
4. Automatically load correct configuration when resuming from checkpoint
"""

import json
import hashlib
import uuid as uuid_lib
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict

from ..utils.logger import get_logger_conf
from ..core.types import PipelineConfig, QuantizationConfig, LayerSplitConfig, AudioConfig

logger = get_logger_conf(__name__)


class CheckpointConfigManager:
    """
    Manages configuration files for checkpoint resume.

    Each configuration is identified by a UUID derived from its settings.
    Identical configurations share the same UUID and config file.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the checkpoint config manager.

        Args:
            config_dir: Directory to store configs (defaults to .config/llamanote/checkpoint_configs)
        """
        if config_dir is None:
            from ..config.settings import DEFAULT_CONFIG_DIR
            config_dir = DEFAULT_CONFIG_DIR / "checkpoint_configs"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _extract_resume_requirements(self, pipeline_config: PipelineConfig) -> Dict[str, Any]:
        """
        Extract only the settings required for checkpoint resume.

        These are the settings that must match for a checkpoint to be usable.

        Args:
            pipeline_config: Full pipeline configuration

        Returns:
            Dict of resume-critical settings
        """
        requirements = {
            # Model settings
            'model_provider': pipeline_config.model_provider,
            'model_specifier': pipeline_config.model_specifier,

            # Quantization (affects model loading)
            'quantization': None,
            'layer_split': None,

            # Processing settings
            'mode': pipeline_config.mode.value if hasattr(pipeline_config.mode, 'value') else str(pipeline_config.mode),
            'system_prompt': pipeline_config.system_prompt,
            'chunking_strategy': pipeline_config.chunking_strategy.value if hasattr(pipeline_config.chunking_strategy, 'value') else str(pipeline_config.chunking_strategy),
            'chunk_size': pipeline_config.chunk_size,
            'chunk_overlap': pipeline_config.chunk_overlap,

            # Audio settings (if applicable)
            'generate_audio': pipeline_config.generate_audio,
            'audio_provider': pipeline_config.audio_provider,
            'audio_specifier': pipeline_config.audio_specifier,
        }

        # Include quantization config if present
        if pipeline_config.quantization_config:
            requirements['quantization'] = {
                'method': pipeline_config.quantization_config.method,
                'compute_dtype': str(pipeline_config.quantization_config.compute_dtype) if pipeline_config.quantization_config.compute_dtype else None,
                'use_double_quant': pipeline_config.quantization_config.use_double_quant,
                'quant_type': pipeline_config.quantization_config.quant_type
            }

        # Include layer split config if present
        if pipeline_config.layer_split_config:
            requirements['layer_split'] = {
                'enabled': pipeline_config.layer_split_config.enabled,
                'gpu_layers': pipeline_config.layer_split_config.gpu_layers,
                'max_gpu_memory': pipeline_config.layer_split_config.max_gpu_memory,
                'max_cpu_memory': pipeline_config.layer_split_config.max_cpu_memory,
                'auto_oom_handling': pipeline_config.layer_split_config.auto_oom_handling
            }

        # Include audio config basics if present
        if pipeline_config.audio_config and pipeline_config.generate_audio:
            requirements['audio_config'] = {
                'sample_rate': pipeline_config.audio_config.sample_rate,
                'output_format': pipeline_config.audio_config.output_format,
                'quantization': pipeline_config.audio_config.quantization,
                'enable_cpu_offload': pipeline_config.audio_config.enable_cpu_offload
            }

        return requirements

    def _generate_uuid(self, requirements: Dict[str, Any]) -> str:
        """
        Generate UUID from configuration requirements.

        Args:
            requirements: Dict of resume-critical settings

        Returns:
            UUID string (format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
        """
        # Convert to deterministic JSON string
        requirements_str = json.dumps(requirements, sort_keys=True, default=str)

        # Generate SHA256 hash
        hash_bytes = hashlib.sha256(requirements_str.encode('utf-8')).digest()

        # Create UUID from first 16 bytes of hash (UUID is 128 bits = 16 bytes)
        generated_uuid = uuid_lib.UUID(bytes=hash_bytes[:16])

        return str(generated_uuid)

    def save_config(self, pipeline_config: PipelineConfig) -> str:
        """
        Save configuration required for checkpoint resume.

        If an identical configuration already exists, returns its UUID
        without creating a duplicate.

        Args:
            pipeline_config: Full pipeline configuration

        Returns:
            UUID of the configuration
        """
        # Extract resume requirements
        requirements = self._extract_resume_requirements(pipeline_config)

        # Generate UUID
        config_uuid = self._generate_uuid(requirements)

        # Check if config already exists
        config_file = self.config_dir / f"{config_uuid}.json"

        if config_file.exists():
            self.logger.debug(f"Configuration {config_uuid} already exists, reusing")
            return config_uuid

        # Save new configuration
        config_data = {
            'uuid': config_uuid,
            'requirements': requirements,
            'full_config': asdict(pipeline_config)  # Save full config for reference
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)

        self.logger.info(f"Saved checkpoint configuration: {config_uuid}")
        return config_uuid

    def load_config(self, config_uuid: str) -> Optional[PipelineConfig]:
        """
        Load configuration by UUID.

        Args:
            config_uuid: UUID of the configuration

        Returns:
            PipelineConfig if found, None otherwise
        """
        config_file = self.config_dir / f"{config_uuid}.json"

        if not config_file.exists():
            self.logger.error(f"Configuration {config_uuid} not found")
            return None

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            # Reconstruct PipelineConfig from saved data
            full_config = config_data['full_config']

            # Reconstruct complex types
            from ..core.types import ProcessingMode, ChunkingStrategy

            # Convert mode
            mode = ProcessingMode(full_config.get('mode', 'podcast'))

            # Convert chunking strategy
            chunking_strategy = ChunkingStrategy(full_config.get('chunking_strategy', 'word_boundary'))

            # Reconstruct quantization config
            quant_config = None
            if full_config.get('quantization_config'):
                qc = full_config['quantization_config']
                import torch
                compute_dtype = eval(qc['compute_dtype']) if qc.get('compute_dtype') and qc['compute_dtype'] != 'None' else None

                quant_config = QuantizationConfig(
                    method=qc.get('method', 'none'),
                    compute_dtype=compute_dtype,
                    use_double_quant=qc.get('use_double_quant', True),
                    quant_type=qc.get('quant_type', 'nf4')
                )

            # Reconstruct layer split config
            split_config = None
            if full_config.get('layer_split_config'):
                sc = full_config['layer_split_config']
                split_config = LayerSplitConfig(
                    enabled=sc.get('enabled', True),
                    gpu_layers=sc.get('gpu_layers', -1),
                    max_gpu_memory=sc.get('max_gpu_memory', {0: '4GB'}),
                    max_cpu_memory=sc.get('max_cpu_memory', '28GB'),
                    offload_folder=Path(sc['offload_folder']) if sc.get('offload_folder') else None,
                    offload_state_dict=sc.get('offload_state_dict', True),
                    low_cpu_mem_usage=sc.get('low_cpu_mem_usage', True),
                    auto_oom_handling=sc.get('auto_oom_handling', True)
                )

            # Reconstruct audio config
            audio_config = None
            if full_config.get('audio_config'):
                ac = full_config['audio_config']
                audio_config = AudioConfig(**{k: v for k, v in ac.items() if k in AudioConfig.__dataclass_fields__})

            # Create PipelineConfig
            pipeline_config = PipelineConfig(
                mode=mode,
                model_provider=full_config.get('model_provider', 'local_hf'),
                model_specifier=full_config.get('model_specifier', ''),
                system_prompt=full_config.get('system_prompt'),
                stages=full_config.get('stages'),
                output_format=full_config.get('output_format', 'markdown'),
                output_dir=Path(full_config['output_dir']) if full_config.get('output_dir') else None,
                timestamp_outputs=full_config.get('timestamp_outputs', True),
                include_metadata=full_config.get('include_metadata', True),
                chunking_strategy=chunking_strategy,
                chunk_size=full_config.get('chunk_size', 1000),
                chunk_overlap=full_config.get('chunk_overlap', 50),
                preserve_pdf_layout=full_config.get('preserve_pdf_layout', True),
                clean_for_audio=full_config.get('clean_for_audio', True),
                remove_thinking=full_config.get('remove_thinking', True),
                markdown_style=full_config.get('markdown_style'),
                add_emotions=full_config.get('add_emotions', True),
                hyperparameters=None,  # Will be reconstructed separately
                quantization_config=quant_config,
                layer_split_config=split_config,
                generate_audio=full_config.get('generate_audio', False),
                audio_config=audio_config,
                audio_provider=full_config.get('audio_provider'),
                audio_specifier=full_config.get('audio_specifier')
            )

            self.logger.info(f"Loaded checkpoint configuration: {config_uuid}")
            return pipeline_config

        except Exception as e:
            self.logger.error(f"Failed to load configuration {config_uuid}: {e}", exc_info=True)
            return None

    def get_uuid_for_config(self, pipeline_config: PipelineConfig) -> str:
        """
        Get UUID for a configuration without saving it.

        Useful for checking if a config already exists.

        Args:
            pipeline_config: Pipeline configuration

        Returns:
            UUID string
        """
        requirements = self._extract_resume_requirements(pipeline_config)
        return self._generate_uuid(requirements)

    def list_configs(self) -> list[Dict[str, Any]]:
        """
        List all saved configurations.

        Returns:
            List of config summaries
        """
        configs = []

        for config_file in self.config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)

                configs.append({
                    'uuid': config_data['uuid'],
                    'model': config_data['requirements'].get('model_specifier'),
                    'mode': config_data['requirements'].get('mode'),
                    'quantization': config_data['requirements'].get('quantization', {}).get('method'),
                    'file': config_file.name
                })
            except Exception as e:
                self.logger.warning(f"Failed to read {config_file}: {e}")

        return configs


__all__ = ['CheckpointConfigManager']
