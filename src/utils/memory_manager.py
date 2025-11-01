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
            "cuda error: device memory allocation failed"
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
                 logger=None):
        """
        Initialize OOM recovery strategy.

        Args:
            initial_gpu_memory: Initial GPU memory allocation
            initial_cpu_memory: CPU memory allocation for offloading
            min_gpu_memory: Minimum GPU memory to maintain
            logger: Logger instance
        """
        self.initial_gpu_memory = initial_gpu_memory
        self.initial_cpu_memory = initial_cpu_memory
        self.min_gpu_memory = min_gpu_memory
        self.logger = logger or get_logger_conf(__name__)

        self.attempt_count = 0
        self.max_attempts = 10

        # Track what we've tried
        self.tried_cache_offload = False
        self.tried_context_offload = False
        self.current_gpu_memory_gb = self._parse_gb(initial_gpu_memory)
        self.min_gpu_memory_gb = self._parse_gb(min_gpu_memory)

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

        # Strategy 4+: Progressive GPU memory reduction (offload more layers)
        # Reduce GPU memory by 20% each time
        reduction_factor = 0.8
        new_gpu_memory_gb = max(
            self.current_gpu_memory_gb * reduction_factor,
            self.min_gpu_memory_gb
        )

        if new_gpu_memory_gb < self.current_gpu_memory_gb:
            self.current_gpu_memory_gb = new_gpu_memory_gb
            strategy['action'] = 'reduce_gpu_memory'
            strategy['params'] = {
                'max_gpu_memory': {0: f"{self.current_gpu_memory_gb:.1f}GB"},
                'max_cpu_memory': self.initial_cpu_memory,
                'device_map': 'auto',
                'offload_state_dict': True,
                'low_cpu_mem_usage': True
            }
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

            elif strategy['action'] == 'reduce_gpu_memory':
                # Apply memory reduction
                if on_strategy_change:
                    on_strategy_change(strategy['params'])

                result = load_fn(**strategy['params'])
                final_params = strategy['params']
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
