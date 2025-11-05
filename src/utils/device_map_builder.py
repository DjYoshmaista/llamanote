# llamanote/utils/device_map_builder.py
"""
Custom Device Map Builder for Layer Splitting

Creates explicit device maps for Hugging Face Transformers models,
specifying exactly which layers go on GPU vs CPU.

Unlike device_map="auto" which lets Accelerate decide, this module
creates deterministic layer placement that can be saved and reused.
"""

import json
import hashlib
import torch
from pathlib import Path
from typing import Dict, Optional, Any, Tuple, List
from dataclasses import dataclass, asdict
from transformers import AutoConfig

from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


@dataclass
class LayerSplitResult:
    """Result of a successful layer split configuration."""
    device_map: Dict[str, Any]
    layers_on_gpu: int
    layers_on_cpu: int
    total_layers: int
    gpu_memory_used_mb: Optional[float] = None
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'device_map': self.device_map,
            'layers_on_gpu': self.layers_on_gpu,
            'layers_on_cpu': self.layers_on_cpu,
            'total_layers': self.total_layers,
            'gpu_memory_used_mb': self.gpu_memory_used_mb,
            'success': self.success
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerSplitResult':
        """Create from dictionary."""
        return cls(**data)


class DeviceMapBuilder:
    """
    Builds explicit device maps for layer splitting.

    This creates deterministic layer placement that can be:
    1. Tested incrementally to find optimal split
    2. Saved to disk for reuse
    3. Loaded for consistent inference
    """

    def __init__(self,
                 model_id: str,
                 cache_dir: Optional[Path] = None,
                 trust_remote_code: bool = True):
        """
        Initialize the device map builder.

        Args:
            model_id: Hugging Face model ID
            cache_dir: Cache directory for model config
            trust_remote_code: Whether to trust remote code
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.logger = logger

        # Get model config to determine architecture
        self.config = self._load_model_config()
        self.model_type = self.config.model_type if hasattr(self.config, 'model_type') else 'unknown'
        self.num_layers = self._get_num_layers()

        self.logger.info(f"DeviceMapBuilder initialized for {model_id}")
        self.logger.info(f"Model type: {self.model_type}, Total layers: {self.num_layers}")

    def _load_model_config(self) -> Any:
        """Load model configuration."""
        try:
            config = AutoConfig.from_pretrained(
                self.model_id,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                trust_remote_code=self.trust_remote_code
            )
            return config
        except Exception as e:
            self.logger.error(f"Failed to load model config: {e}")
            raise

    def _get_num_layers(self) -> int:
        """Get number of layers in the model."""
        # Try common attribute names
        for attr in ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers']:
            if hasattr(self.config, attr):
                return getattr(self.config, attr)

        self.logger.warning(f"Could not determine number of layers for {self.model_id}")
        return 0

    def _get_layer_names(self) -> Dict[str, str]:
        """
        Get the naming pattern for layers in this model architecture.

        Returns:
            Dict with keys: 'embed', 'layers', 'norm', 'lm_head'
        """
        # Common patterns for different architectures
        patterns = {
            'llama': {
                'embed': 'model.embed_tokens',
                'layers': 'model.layers',
                'norm': 'model.norm',
                'lm_head': 'lm_head'
            },
            'qwen2': {
                'embed': 'model.embed_tokens',
                'layers': 'model.layers',
                'norm': 'model.norm',
                'lm_head': 'lm_head'
            },
            'gemma': {
                'embed': 'model.embed_tokens',
                'layers': 'model.layers',
                'norm': 'model.norm',
                'lm_head': 'lm_head'
            },
            'gpt2': {
                'embed': 'transformer.wte',
                'layers': 'transformer.h',
                'norm': 'transformer.ln_f',
                'lm_head': 'lm_head'
            },
            'gpt_neox': {
                'embed': 'gpt_neox.embed_in',
                'layers': 'gpt_neox.layers',
                'norm': 'gpt_neox.final_layer_norm',
                'lm_head': 'embed_out'
            }
        }

        # Return pattern for this model type
        model_type_lower = self.model_type.lower()
        for key, pattern in patterns.items():
            if key in model_type_lower:
                return pattern

        # Default pattern (works for most models)
        return patterns['llama']

    def build_device_map(self,
                        layers_on_gpu: int,
                        include_embeddings_on_gpu: bool = True,
                        include_lm_head_on_gpu: bool = True) -> Dict[str, Any]:
        """
        Build an explicit device map.

        Args:
            layers_on_gpu: Number of transformer layers to place on GPU
            include_embeddings_on_gpu: Put embeddings on GPU
            include_lm_head_on_gpu: Put language model head on GPU

        Returns:
            Device map dict mapping module names to devices
        """
        if self.num_layers == 0:
            self.logger.warning("Unknown layer count, falling back to auto")
            return "auto"

        layers_on_gpu = max(0, min(layers_on_gpu, self.num_layers))
        layers_on_cpu = self.num_layers - layers_on_gpu

        layer_names = self._get_layer_names()
        device_map = {}

        # Embeddings
        if include_embeddings_on_gpu:
            device_map[layer_names['embed']] = 0
        else:
            device_map[layer_names['embed']] = 'cpu'

        # Transformer layers
        for i in range(self.num_layers):
            layer_name = f"{layer_names['layers']}.{i}"
            if i < layers_on_gpu:
                device_map[layer_name] = 0  # GPU 0
            else:
                device_map[layer_name] = 'cpu'

        # Final norm
        if layers_on_gpu > 0:
            device_map[layer_names['norm']] = 0
        else:
            device_map[layer_names['norm']] = 'cpu'

        # LM head
        if include_lm_head_on_gpu:
            device_map[layer_names['lm_head']] = 0
        else:
            device_map[layer_names['lm_head']] = 'cpu'

        self.logger.info(f"Built device map: {layers_on_gpu} layers on GPU, {layers_on_cpu} on CPU")

        return device_map

    def create_split_result(self,
                           layers_on_gpu: int,
                           gpu_memory_used_mb: Optional[float] = None) -> LayerSplitResult:
        """Create a LayerSplitResult with device map."""
        device_map = self.build_device_map(layers_on_gpu)

        return LayerSplitResult(
            device_map=device_map,
            layers_on_gpu=layers_on_gpu,
            layers_on_cpu=self.num_layers - layers_on_gpu,
            total_layers=self.num_layers,
            gpu_memory_used_mb=gpu_memory_used_mb,
            success=True
        )


class LayerSplitConfigManager:
    """
    Manages saving and loading of successful layer split configurations.

    This allows the system to remember which split configurations worked
    and reuse them for consistent performance.
    """

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize the config manager.

        Args:
            config_dir: Directory to store split configs (defaults to cache/layer_splits)
        """
        if config_dir is None:
            from ..config.settings import DEFAULT_CACHE_DIR
            config_dir = DEFAULT_CACHE_DIR / "layer_splits"

        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def _generate_config_hash(self,
                             model_id: str,
                             quant_method: str,
                             gpu_memory_gb: float) -> str:
        """
        Generate a hash for this configuration.

        Args:
            model_id: Hugging Face model ID
            quant_method: Quantization method (none, 4bit, 8bit, 16bit)
            gpu_memory_gb: Available GPU memory in GB

        Returns:
            12-character hash
        """
        config_str = f"{model_id}:{quant_method}:{gpu_memory_gb:.1f}"
        hash_obj = hashlib.sha256(config_str.encode('utf-8'))
        return hash_obj.hexdigest()[:12]

    def save_config(self,
                   model_id: str,
                   quant_method: str,
                   gpu_memory_gb: float,
                   min_split: LayerSplitResult,
                   max_split: LayerSplitResult) -> Path:
        """
        Save a successful layer split configuration.

        Args:
            model_id: Hugging Face model ID
            quant_method: Quantization method used
            gpu_memory_gb: GPU memory available
            min_split: Minimum layers on CPU (most on GPU)
            max_split: Maximum layers on CPU (least on GPU)

        Returns:
            Path to saved config file
        """
        config_hash = self._generate_config_hash(model_id, quant_method, gpu_memory_gb)
        config_file = self.config_dir / f"split_{config_hash}.json"

        from datetime import datetime

        config_data = {
            'model_id': model_id,
            'quant_method': quant_method,
            'gpu_memory_gb': gpu_memory_gb,
            'config_hash': config_hash,
            'min_split': min_split.to_dict(),
            'max_split': max_split.to_dict(),
            'created_at': datetime.now().isoformat()
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)

        self.logger.info(f"Saved layer split config: {config_file.name}")
        self.logger.info(f"  Min split: {min_split.layers_on_gpu} GPU / {min_split.layers_on_cpu} CPU")
        self.logger.info(f"  Max split: {max_split.layers_on_gpu} GPU / {max_split.layers_on_cpu} CPU")

        return config_file

    def load_config(self,
                   model_id: str,
                   quant_method: str,
                   gpu_memory_gb: float) -> Optional[Tuple[LayerSplitResult, LayerSplitResult]]:
        """
        Load a saved layer split configuration.

        Args:
            model_id: Hugging Face model ID
            quant_method: Quantization method
            gpu_memory_gb: GPU memory available

        Returns:
            Tuple of (min_split, max_split) if found, None otherwise
        """
        config_hash = self._generate_config_hash(model_id, quant_method, gpu_memory_gb)
        config_file = self.config_dir / f"split_{config_hash}.json"

        if not config_file.exists():
            self.logger.debug(f"No saved config found for {config_hash}")
            return None

        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)

            min_split = LayerSplitResult.from_dict(config_data['min_split'])
            max_split = LayerSplitResult.from_dict(config_data['max_split'])

            self.logger.info(f"Loaded layer split config: {config_file.name}")
            self.logger.info(f"  Min split: {min_split.layers_on_gpu} GPU / {min_split.layers_on_cpu} CPU")
            self.logger.info(f"  Max split: {max_split.layers_on_gpu} GPU / {max_split.layers_on_cpu} CPU")

            return min_split, max_split

        except Exception as e:
            self.logger.error(f"Failed to load config {config_file}: {e}")
            return None

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all saved configurations."""
        configs = []

        for config_file in self.config_dir.glob("split_*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                configs.append({
                    'file': config_file.name,
                    'model_id': config_data.get('model_id'),
                    'quant_method': config_data.get('quant_method'),
                    'gpu_memory_gb': config_data.get('gpu_memory_gb'),
                    'min_gpu_layers': config_data.get('min_split', {}).get('layers_on_gpu'),
                    'max_gpu_layers': config_data.get('max_split', {}).get('layers_on_gpu')
                })
            except Exception as e:
                self.logger.warning(f"Failed to read {config_file}: {e}")

        return configs


__all__ = [
    'DeviceMapBuilder',
    'LayerSplitResult',
    'LayerSplitConfigManager'
]
