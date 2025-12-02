# src/io/layer_split_cache.py
"""
Layer Split Cache Manager

Manages cached layer split configurations to avoid redundant discovery.
Stores optimal GPU/CPU layer distributions with validation metadata.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import torch

from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class LayerSplitCacheManager:
    """
    Manages cached layer split configurations.

    Responsibilities:
    - Generate cache keys from model/GPU/quantization
    - Load/save cached split configurations
    - Validate hardware compatibility
    - Manage cache directory structure
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files (default: cache/layer_splits/)
        """
        if cache_dir is None:
            # Default to cache/layer_splits/ in project root
            project_root = Path(__file__).parent.parent.parent
            cache_dir = project_root / "cache" / "layer_splits"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger_conf(f"{__name__}.LayerSplitCacheManager")

        self.logger.debug(f"Initialized LayerSplitCacheManager with cache_dir: {self.cache_dir}")

    def generate_cache_key(
        self,
        model_id: str,
        quantization: str,
        gpu_name: Optional[str] = None
    ) -> str:
        """
        Generate unique cache key from model, quantization, and GPU.

        Args:
            model_id: HuggingFace model identifier
            quantization: Quantization method (4bit, 8bit, 16bit, none)
            gpu_name: GPU name (auto-detected if None)

        Returns:
            16-character hex cache key
        """
        if gpu_name is None:
            gpu_name = self._get_gpu_name()

        # Normalize inputs
        model_id_normalized = model_id.lower().replace("/", "_").replace("-", "_")
        quantization_normalized = quantization.lower()
        gpu_name_normalized = gpu_name.lower().replace(" ", "_")

        # Create hash input
        hash_input = f"{model_id_normalized}_{quantization_normalized}_{gpu_name_normalized}"

        # Generate MD5 hash (first 16 chars)
        cache_key = hashlib.md5(hash_input.encode()).hexdigest()[:16]

        self.logger.debug(f"Generated cache key: {cache_key} for {model_id} ({quantization}, {gpu_name})")

        return cache_key

    def _get_gpu_name(self) -> str:
        """
        Auto-detect GPU name.

        Returns:
            GPU name string or "cpu" if no GPU
        """
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                return gpu_name
            except Exception as e:
                self.logger.warning(f"Failed to get GPU name: {e}")
                return "unknown_gpu"
        else:
            return "cpu"

    def _get_gpu_vram_mb(self) -> float:
        """
        Get total GPU VRAM in MB.

        Returns:
            VRAM size in MB or 0 if no GPU
        """
        if torch.cuda.is_available():
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                return total_vram
            except Exception as e:
                self.logger.warning(f"Failed to get GPU VRAM: {e}")
                return 0.0
        else:
            return 0.0

    def _get_cuda_version(self) -> str:
        """
        Get CUDA version string.

        Returns:
            CUDA version or "N/A"
        """
        if torch.cuda.is_available():
            try:
                return torch.version.cuda or "unknown"
            except Exception:
                return "unknown"
        else:
            return "N/A"

    def load_cached_split(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Load cached layer split configuration.

        Args:
            cache_key: Cache key identifier

        Returns:
            Cached configuration dict or None if not found
        """
        cache_file = self.cache_dir / f"split_{cache_key}.json"

        if not cache_file.exists():
            self.logger.debug(f"No cached split found for key: {cache_key}")
            return None

        try:
            with open(cache_file, 'r') as f:
                cached_config = json.load(f)

            self.logger.info(f"Loaded cached split from {cache_file.name}")
            return cached_config

        except Exception as e:
            self.logger.error(f"Failed to load cached split {cache_file}: {e}", exc_info=True)
            return None

    def save_split_configuration(
        self,
        cache_key: str,
        max_gpu_layers: int,
        min_gpu_layers: int,
        model_id: str,
        quantization: str,
        num_layers: int,
        model_config: Dict[str, Any],
        max_estimated_memory_mb: float,
        min_estimated_memory_mb: float,
        validated_batch_size: int,
        validated_max_tokens: int,
        safety_margin: float = 1.2
    ) -> bool:
        """
        Save layer split configuration to cache.

        Args:
            cache_key: Cache key identifier
            max_gpu_layers: Maximum GPU layers (performance configuration)
            min_gpu_layers: Minimum GPU layers (conservative configuration)
            model_id: HuggingFace model identifier
            quantization: Quantization method
            num_layers: Total number of layers in model
            model_config: Model configuration dict (hidden_size, num_heads, etc.)
            max_estimated_memory_mb: Estimated peak memory for max configuration
            min_estimated_memory_mb: Estimated peak memory for min configuration
            validated_batch_size: Batch size used for validation
            validated_max_tokens: Max tokens used for validation
            safety_margin: Safety margin applied during validation

        Returns:
            True if save successful, False otherwise
        """
        cache_file = self.cache_dir / f"split_{cache_key}.json"

        # Build configuration object
        config = {
            "cache_key": cache_key,
            "created_timestamp": datetime.now().isoformat(),
            "model_info": {
                "model_id": model_id,
                "quantization": quantization,
                "num_layers": num_layers,
                "hidden_size": model_config.get("hidden_size", 0),
                "num_attention_heads": model_config.get("num_attention_heads", 0),
                "num_key_value_heads": model_config.get("num_key_value_heads"),
            },
            "hardware_info": {
                "gpu_name": self._get_gpu_name(),
                "total_vram_mb": self._get_gpu_vram_mb(),
                "cuda_version": self._get_cuda_version()
            },
            "configurations": {
                "max_gpu_layers": {
                    "layers_on_gpu": max_gpu_layers,
                    "estimated_peak_memory_mb": max_estimated_memory_mb,
                    "validated_batch_size": validated_batch_size,
                    "validated_max_tokens": validated_max_tokens,
                    "safety_margin": safety_margin,
                    "last_validated": datetime.now().isoformat()
                },
                "min_gpu_layers": {
                    "layers_on_gpu": min_gpu_layers,
                    "estimated_peak_memory_mb": min_estimated_memory_mb,
                    "validated_batch_size": validated_batch_size,
                    "validated_max_tokens": validated_max_tokens,
                    "safety_margin": safety_margin,
                    "last_validated": datetime.now().isoformat()
                }
            }
        }

        try:
            # Write to temporary file first (atomic write)
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(config, f, indent=2)

            # Rename to final name (atomic operation)
            temp_file.rename(cache_file)

            self.logger.info(f"Saved layer split configuration to {cache_file.name}")
            self.logger.info(f"  Max GPU layers: {max_gpu_layers}/{num_layers} (~{max_estimated_memory_mb:.0f} MB)")
            self.logger.info(f"  Min GPU layers: {min_gpu_layers}/{num_layers} (~{min_estimated_memory_mb:.0f} MB)")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save layer split configuration: {e}", exc_info=True)
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
            return False

    def validate_hardware_match(self, cached_config: Dict[str, Any]) -> bool:
        """
        Validate that cached configuration matches current hardware.

        Args:
            cached_config: Cached configuration dict

        Returns:
            True if hardware matches, False otherwise
        """
        if not cached_config or "hardware_info" not in cached_config:
            return False

        cached_hw = cached_config["hardware_info"]

        # Check GPU name
        current_gpu = self._get_gpu_name()
        cached_gpu = cached_hw.get("gpu_name", "")

        if current_gpu != cached_gpu:
            self.logger.warning(f"GPU mismatch: cached={cached_gpu}, current={current_gpu}")
            return False

        # Check VRAM (allow 5% tolerance for reporting variations)
        current_vram = self._get_gpu_vram_mb()
        cached_vram = cached_hw.get("total_vram_mb", 0.0)

        if abs(current_vram - cached_vram) > (cached_vram * 0.05):
            self.logger.warning(f"VRAM mismatch: cached={cached_vram:.0f}MB, current={current_vram:.0f}MB")
            return False

        # Check CUDA version (major version must match)
        current_cuda = self._get_cuda_version()
        cached_cuda = cached_hw.get("cuda_version", "")

        current_major = current_cuda.split('.')[0] if current_cuda != "N/A" else ""
        cached_major = cached_cuda.split('.')[0] if cached_cuda != "N/A" else ""

        if current_major != cached_major:
            self.logger.warning(f"CUDA major version mismatch: cached={cached_cuda}, current={current_cuda}")
            return False

        self.logger.debug("Hardware validation passed")
        return True

    def list_cached_splits(self) -> List[Dict[str, Any]]:
        """
        List all cached layer split configurations.

        Returns:
            List of cache metadata dicts
        """
        cached_splits = []

        for cache_file in self.cache_dir.glob("split_*.json"):
            try:
                with open(cache_file, 'r') as f:
                    config = json.load(f)

                # Extract summary info
                summary = {
                    "cache_key": config.get("cache_key"),
                    "model_id": config.get("model_info", {}).get("model_id"),
                    "quantization": config.get("model_info", {}).get("quantization"),
                    "gpu_name": config.get("hardware_info", {}).get("gpu_name"),
                    "max_gpu_layers": config.get("configurations", {}).get("max_gpu_layers", {}).get("layers_on_gpu"),
                    "min_gpu_layers": config.get("configurations", {}).get("min_gpu_layers", {}).get("layers_on_gpu"),
                    "created_timestamp": config.get("created_timestamp"),
                    "file_path": str(cache_file)
                }

                cached_splits.append(summary)

            except Exception as e:
                self.logger.warning(f"Failed to read cache file {cache_file}: {e}")
                continue

        return cached_splits

    def delete_cached_split(self, cache_key: str) -> bool:
        """
        Delete a cached layer split configuration.

        Args:
            cache_key: Cache key identifier

        Returns:
            True if deleted, False if not found or error
        """
        cache_file = self.cache_dir / f"split_{cache_key}.json"

        if not cache_file.exists():
            self.logger.warning(f"Cache file not found: {cache_key}")
            return False

        try:
            cache_file.unlink()
            self.logger.info(f"Deleted cached split: {cache_key}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete cached split {cache_key}: {e}", exc_info=True)
            return False

    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the cache.

        Returns:
            Dict with cache statistics
        """
        cache_files = list(self.cache_dir.glob("split_*.json"))

        total_size_bytes = sum(f.stat().st_size for f in cache_files)
        total_size_mb = total_size_bytes / (1024**2)

        return {
            "num_cached_configs": len(cache_files),
            "total_size_mb": total_size_mb,
            "cache_dir": str(self.cache_dir),
            "cache_files": [f.name for f in cache_files]
        }
