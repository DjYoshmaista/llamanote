# llamanote/utils/memory_manager.py
"""
Advanced Memory Management Utilities

Provides intelligent CUDA OOM error handling with progressive layer offloading,
memory monitoring, and cache management for efficient GPU memory usage.
"""

import gc
import re
import torch
from typing import Dict, Optional, Callable, Any, Tuple
from pathlib import Path
from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class CUDAMemoryManager:
    """Manages CUDA memory with intelligent OOM handling."""

    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current GPU memory usage statistics in MB."""
        stats = {}
        try:
            if torch.cuda.is_available():
                stats['allocated_mb'] = torch.cuda.memory_allocated() / (1024**2)
                stats['reserved_mb'] = torch.cuda.memory_reserved() / (1024**2)
                stats['max_allocated_mb'] = torch.cuda.max_memory_allocated() / (1024**2)
                stats['total_mb'] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                stats['free_mb'] = stats['total_mb'] - stats['allocated_mb']
                stats['utilization_pct'] = (stats['allocated_mb'] / stats['total_mb']) * 100
        except Exception as e:
            logger.warning(f"Failed to get GPU memory stats: {e}")
        return stats

    @staticmethod
    def clear_cache(aggressive: bool = False):
        """Clear GPU cache and optionally run garbage collection."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if aggressive:
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()
                logger.debug("GPU cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")

    @staticmethod
    def reset_peak_stats():
        """Reset peak memory statistics."""
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            logger.warning(f"Failed to reset peak stats: {e}")

    @staticmethod
    def is_oom_error(exception: Exception) -> bool:
        """
        Check if an exception is a CUDA Out of Memory error.

        Args:
            exception: The exception to check

        Returns:
            True if this is a CUDA OOM error
        """
        error_msg = str(exception).lower()
        oom_patterns = [
            "cuda out of memory",
            "cudnn error: out of memory",
            "out of memory",
            "cuda error: out of memory",
            "cuda error: device memory allocation failed",
            "oom recovery failed"  # Catch our own OOM recovery failures
        ]
        return any(pattern in error_msg for pattern in oom_patterns)

    @staticmethod
    def parse_memory_string(mem_str: str) -> int:
        """
        Parse memory string (e.g., "4GB", "2048MB") to bytes.

        Args:
            mem_str: Memory string to parse

        Returns:
            Memory size in bytes
        """
        mem_str = mem_str.upper().strip()

        # Match pattern like "4GB" or "2048MB"
        match = re.match(r'(\d+(?:\.\d+)?)\s*(GB|MB|KB|B)?', mem_str)
        if not match:
            raise ValueError(f"Invalid memory string: {mem_str}")

        value = float(match.group(1))
        unit = match.group(2) or 'B'

        multipliers = {
            'B': 1,
            'KB': 1024,
            'MB': 1024**2,
            'GB': 1024**3
        }

        return int(value * multipliers[unit])

    @staticmethod
    def format_bytes(bytes_val: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}TB"


class OOMRecoveryStrategy:
    """
    Implements progressive recovery strategies for CUDA Out of Memory errors.

    This class provides intelligent handling of OOM errors by:
    1. First attempting to offload KV cache and context to CPU
    2. Then progressively offloading model layers until memory is sufficient
    3. Adjusting device maps dynamically based on available memory
    """

    def __init__(self,
                 initial_gpu_memory: str = "4GB",
                 initial_cpu_memory: str = "28GB",
                 min_gpu_memory: str = "1GB",
                 model_num_layers: Optional[int] = None,
                 use_iterative_layer_split: bool = True,
                 offload_folder: Optional[Path] = None,
                 model_id: Optional[str] = None,
                 use_explicit_device_map: bool = True,
                 logger=None):
        """
        Initialize OOM recovery strategy.

        Args:
            initial_gpu_memory: Initial GPU memory allocation
            initial_cpu_memory: CPU memory allocation for offloading
            min_gpu_memory: Minimum GPU memory to maintain
            model_num_layers: Number of layers in the model (for iterative splitting)
            use_iterative_layer_split: Use layer-by-layer splitting instead of percentage-based
            offload_folder: Folder for disk offloading (optional)
            model_id: Model ID for building explicit device maps (optional)
            use_explicit_device_map: Use explicit device maps instead of max_memory (default: True)
            logger: Logger instance
        """
        self.initial_gpu_memory = initial_gpu_memory
        self.initial_cpu_memory = initial_cpu_memory
        self.min_gpu_memory = min_gpu_memory
        self.model_num_layers = model_num_layers
        self.use_iterative_layer_split = use_iterative_layer_split
        self.offload_folder = offload_folder
        self.model_id = model_id
        self.use_explicit_device_map = use_explicit_device_map
        self.logger = logger or get_logger_conf(__name__)

        self.attempt_count = 0
        self.max_attempts = 50  # Increased to allow for layer-by-layer attempts

        # Track what we've tried
        self.tried_cache_offload = False
        self.tried_context_offload = False
        self.tried_disk_offload = False
        self.current_gpu_memory_gb = self._parse_gb(initial_gpu_memory)
        self.min_gpu_memory_gb = self._parse_gb(min_gpu_memory)

        # Layer-based splitting state
        self.layers_offloaded_to_cpu = 0  # Number of layers moved to CPU
        self.layer_split_increment = 1  # Move layers one at a time initially

        # Device map builder for explicit layer placement
        self.device_map_builder = None
        if self.use_explicit_device_map and self.model_id and self.model_num_layers:
            try:
                from .device_map_builder import DeviceMapBuilder
                self.device_map_builder = DeviceMapBuilder(self.model_id)
                self.logger.info(f"Initialized DeviceMapBuilder for explicit layer placement ({self.model_num_layers} layers)")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DeviceMapBuilder: {e}. Falling back to max_memory approach.")
                self.device_map_builder = None

    def _parse_gb(self, mem_str: str) -> float:
        """Parse memory string to GB float."""
        bytes_val = CUDAMemoryManager.parse_memory_string(mem_str)
        return bytes_val / (1024**3)

    def should_retry(self) -> bool:
        """Check if we should attempt another recovery."""
        return self.attempt_count < self.max_attempts and self.current_gpu_memory_gb >= self.min_gpu_memory_gb

    def get_next_strategy(self) -> Dict[str, Any]:
        """
        Get the next recovery strategy to try.

        Returns:
            Dictionary with recovery strategy parameters
        """
        self.attempt_count += 1

        strategy = {
            'attempt': self.attempt_count,
            'action': None,
            'params': {}
        }

        # Strategy 1: Clear cache aggressively
        if self.attempt_count == 1:
            strategy['action'] = 'clear_cache'
            strategy['params'] = {'aggressive': True}
            self.logger.info("Strategy 1: Aggressive cache clearing")
            return strategy

        # Strategy 2: Offload KV cache to CPU
        if self.attempt_count == 2 and not self.tried_cache_offload:
            strategy['action'] = 'offload_kv_cache'
            strategy['params'] = {'device': 'cpu'}
            self.tried_cache_offload = True
            self.logger.info("Strategy 2: Offloading KV cache to CPU")
            return strategy

        # Strategy 3: Offload context/activations to CPU
        if self.attempt_count == 3 and not self.tried_context_offload:
            strategy['action'] = 'offload_context'
            strategy['params'] = {'device': 'cpu'}
            self.tried_context_offload = True
            self.logger.info("Strategy 3: Offloading context/activations to CPU")
            return strategy

        # Strategy 4+: Progressive GPU memory reduction
        # NOTE: Transformers doesn't support direct "X layers on GPU, Y on CPU" specification
        # Instead, we progressively reduce max_gpu_memory which forces accelerate to offload more layers
        if self.use_iterative_layer_split and self.model_num_layers:
            # Calculate memory reduction based on layers we want to offload
            if self.layers_offloaded_to_cpu < self.model_num_layers:
                self.layers_offloaded_to_cpu += self.layer_split_increment

                # Accelerate offloading if we're past 50% of attempts
                if self.attempt_count > 25 and self.layer_split_increment == 1:
                    self.layer_split_increment = 2  # Move 2 layers at a time
                    self.logger.info(f"Accelerating layer offload: moving {self.layer_split_increment} layers per attempt")

                layers_on_gpu = max(0, self.model_num_layers - self.layers_offloaded_to_cpu)

                # Calculate proportional GPU memory allocation
                # Reduce GPU memory proportionally to force more layer offloading
                if layers_on_gpu > 0:
                    # Calculate what percentage of layers should be on GPU
                    gpu_layer_ratio = layers_on_gpu / self.model_num_layers
                    # Reduce initial GPU memory by this ratio (plus some overhead for embeddings/buffers)
                    target_gpu_memory_gb = max(
                        self.current_gpu_memory_gb * gpu_layer_ratio * 0.85,  # 0.85 to force offloading
                        self.min_gpu_memory_gb
                    )
                else:
                    # All layers to CPU - use minimum GPU memory (just for buffers/embeddings)
                    target_gpu_memory_gb = self.min_gpu_memory_gb

                self.current_gpu_memory_gb = target_gpu_memory_gb

                strategy['action'] = 'split_layers'

                # Use explicit device map if available, otherwise fall back to max_memory
                if self.device_map_builder:
                    # Build explicit device map for deterministic layer placement
                    device_map = self.device_map_builder.build_device_map(layers_on_gpu)
                    strategy['params'] = {
                        'device_map': device_map,  # Explicit layer placement
                        'offload_state_dict': True,
                        'low_cpu_mem_usage': True
                    }
                    self.logger.info(f"Using explicit device map: {layers_on_gpu} GPU layers, {self.layers_offloaded_to_cpu} CPU layers")
                else:
                    # Fall back to max_memory approach (less deterministic)
                    strategy['params'] = {
                        'max_memory': {0: f"{self.current_gpu_memory_gb:.2f}GB", 'cpu': self.initial_cpu_memory},
                        'device_map': 'auto',  # Let accelerate decide layer placement based on max_memory
                        'offload_state_dict': True,
                        'low_cpu_mem_usage': True
                    }

                # Add offload_folder if available and not tried yet
                if self.offload_folder and not self.tried_disk_offload:
                    strategy['params']['offload_folder'] = str(self.offload_folder)
                    # Mark as tried after a certain number of attempts
                    if self.attempt_count > 10:
                        self.tried_disk_offload = True

                if layers_on_gpu > 0:
                    self.logger.info(
                        f"Strategy {self.attempt_count}: Targeting {layers_on_gpu}/{self.model_num_layers} layers on GPU "
                        f"by reducing GPU memory to {self.current_gpu_memory_gb:.2f}GB"
                    )
                else:
                    self.logger.info(
                        f"Strategy {self.attempt_count}: Forcing all {self.model_num_layers} layers to CPU "
                        f"(GPU memory: {self.current_gpu_memory_gb:.2f}GB for buffers only)"
                    )

                return strategy
        else:
            # Fallback to percentage-based GPU memory reduction
            reduction_factor = 0.8
            new_gpu_memory_gb = max(
                self.current_gpu_memory_gb * reduction_factor,
                self.min_gpu_memory_gb
            )

            if new_gpu_memory_gb < self.current_gpu_memory_gb:
                self.current_gpu_memory_gb = new_gpu_memory_gb
                strategy['action'] = 'reduce_gpu_memory'
                strategy['params'] = {
                    'max_memory': {0: f"{self.current_gpu_memory_gb:.1f}GB", 'cpu': self.initial_cpu_memory},
                    'device_map': 'auto',
                    'offload_state_dict': True,
                    'low_cpu_mem_usage': True
                }
                # Add offload_folder if available
                if self.offload_folder:
                    strategy['params']['offload_folder'] = str(self.offload_folder)

                self.logger.info(f"Strategy {self.attempt_count}: Reducing GPU memory to {self.current_gpu_memory_gb:.1f}GB")
                return strategy

        # No more strategies available
        strategy['action'] = 'exhausted'
        self.logger.error("All recovery strategies exhausted")
        return strategy


def with_oom_handling(
    load_fn: Callable,
    recovery_strategy: Optional[OOMRecoveryStrategy] = None,
    on_strategy_change: Optional[Callable[[Dict[str, Any]], None]] = None,
    logger=None
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Wrapper function that executes a model loading function with automatic OOM handling.

    Args:
        load_fn: Function to load the model (should accept strategy params as kwargs)
        recovery_strategy: OOM recovery strategy instance
        on_strategy_change: Callback function called when strategy changes
        logger: Logger instance

    Returns:
        Tuple of (loaded_model, final_strategy_params) or raises exception

    Example:
        >>> def load_model(**kwargs):
        ...     return Model.from_pretrained("model_id", **kwargs)
        >>>
        >>> strategy = OOMRecoveryStrategy(initial_gpu_memory="4GB")
        >>> model, params = with_oom_handling(load_model, strategy)
    """
    logger = logger or get_logger_conf(__name__)

    if recovery_strategy is None:
        recovery_strategy = OOMRecoveryStrategy()

    last_exception = None
    final_params = None

    while recovery_strategy.should_retry():
        try:
            # Clear cache before attempt
            CUDAMemoryManager.clear_cache(aggressive=True)

            # Get next strategy
            strategy = recovery_strategy.get_next_strategy()

            if strategy['action'] == 'exhausted':
                break

            # Apply strategy parameters
            if strategy['action'] == 'clear_cache':
                CUDAMemoryManager.clear_cache(aggressive=strategy['params']['aggressive'])
                # Try loading with no changes
                result = load_fn()
                return result, None

            elif strategy['action'] in ['offload_kv_cache', 'offload_context']:
                # These require model-specific implementation
                # Pass as parameters to load_fn
                result = load_fn(**strategy['params'])
                return result, strategy['params']

            elif strategy['action'] in ['reduce_gpu_memory', 'split_layers']:
                # Apply memory reduction or layer splitting
                if on_strategy_change:
                    on_strategy_change(strategy['params'])

                result = load_fn(**strategy['params'])
                final_params = strategy['params']

                if strategy['action'] == 'split_layers':
                    layers_gpu = strategy['params'].get('layers_on_gpu', 0)
                    layers_cpu = strategy['params'].get('layers_on_cpu', 0)
                    logger.info(f"✓ Successfully loaded with layer split: {layers_gpu} GPU / {layers_cpu} CPU")
                else:
                    logger.info(f"✓ Successfully loaded with reduced GPU memory")

                return result, final_params

            else:
                # Unknown strategy, try loading anyway
                result = load_fn()
                return result, None

        except Exception as e:
            last_exception = e

            # Log the actual error for debugging
            logger.debug(f"Error during attempt {recovery_strategy.attempt_count}: {type(e).__name__}: {str(e)}")

            # Check if this is an OOM error
            if not CUDAMemoryManager.is_oom_error(e):
                # Not an OOM error, re-raise immediately
                logger.error(f"Non-OOM error encountered during model loading: {type(e).__name__}: {e}")
                logger.error("This is not a CUDA OOM error - re-raising without recovery attempts")
                raise

            # Log OOM and continue to next strategy
            mem_stats = CUDAMemoryManager.get_memory_stats()
            logger.warning(
                f"OOM Error on attempt {recovery_strategy.attempt_count}: "
                f"GPU {mem_stats.get('allocated_mb', 0):.0f}MB allocated, "
                f"{mem_stats.get('free_mb', 0):.0f}MB free"
            )

            # Clear cache aggressively after OOM
            CUDAMemoryManager.clear_cache(aggressive=True)

    # All strategies exhausted
    logger.error("Failed to recover from OOM error after all strategies")
    if last_exception:
        raise last_exception
    else:
        raise RuntimeError("OOM recovery failed: all strategies exhausted")


def log_memory_summary(stage: str = "current", logger=None):
    """
    Log a formatted summary of current GPU memory usage.

    Args:
        stage: Description of current stage (e.g., "before loading", "after generation")
        logger: Logger instance
    """
    logger = logger or get_logger_conf(__name__)

    stats = CUDAMemoryManager.get_memory_stats()
    if not stats:
        return

    logger.info(f"GPU Memory [{stage}]:")
    logger.info(f"  Allocated: {stats.get('allocated_mb', 0):.1f}MB ({stats.get('utilization_pct', 0):.1f}%)")
    logger.info(f"  Reserved:  {stats.get('reserved_mb', 0):.1f}MB")
    logger.info(f"  Free:      {stats.get('free_mb', 0):.1f}MB")
    logger.info(f"  Peak:      {stats.get('max_allocated_mb', 0):.1f}MB")


__all__ = [
    'CUDAMemoryManager',
    'OOMRecoveryStrategy',
    'with_oom_handling',
    'log_memory_summary'
]
