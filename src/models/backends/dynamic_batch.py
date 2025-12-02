"""
Dynamic Batch Size Manager

Automatically adjusts batch size based on available GPU memory and sequence lengths.
"""

import torch
from typing import Optional, List, Tuple, Dict
from ...utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class DynamicBatchManager:
    """
    Manages dynamic batch sizing based on GPU memory availability.

    Strategy:
    1. Start with configured batch_size
    2. Monitor OOM errors and GPU memory usage
    3. Adaptively reduce batch size on OOM
    4. Optionally increase batch size when memory is underutilized
    """

    def __init__(
        self,
        initial_batch_size: int = 1,
        min_batch_size: int = 1,
        max_batch_size: int = 8,
        enable_dynamic: bool = True,
        memory_threshold: float = 0.85  # Use up to 85% of available VRAM
    ):
        """
        Initialize dynamic batch manager.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            enable_dynamic: Enable dynamic adjustment
            memory_threshold: Target memory utilization (0.0-1.0)
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.enable_dynamic = enable_dynamic
        self.memory_threshold = memory_threshold

        self.oom_count = 0
        self.success_count = 0
        self.last_successful_batch_size = initial_batch_size

        # Track memory usage patterns
        self.memory_history: List[float] = []
        self.max_memory_history = 10

        # Track memory usage per batch size for better prediction
        # Key: batch_size, Value: List[memory_used_mb]
        self.memory_by_batch_size: Dict[int, List[float]] = {}
        self.max_samples_per_size = 5  # Keep up to 5 samples per batch size

        logger.info(
            f"DynamicBatchManager initialized: "
            f"initial={initial_batch_size}, range=[{min_batch_size}, {max_batch_size}], "
            f"dynamic={'enabled' if enable_dynamic else 'disabled'}"
        )

    def get_batch_size(self, num_items_remaining: Optional[int] = None) -> int:
        """
        Get current batch size for next batch.

        Args:
            num_items_remaining: Optional number of items left to process

        Returns:
            Batch size to use
        """
        batch_size = self.current_batch_size

        # Don't exceed remaining items
        if num_items_remaining is not None and num_items_remaining < batch_size:
            batch_size = max(1, num_items_remaining)

        return batch_size

    def record_success(self, actual_batch_size: int, peak_memory_mb: Optional[float] = None):
        """
        Record successful batch processing.

        Args:
            actual_batch_size: Size of batch that succeeded
            peak_memory_mb: Peak GPU memory used (MB)
        """
        self.success_count += 1
        self.last_successful_batch_size = actual_batch_size

        if peak_memory_mb is not None:
            # Add to general memory history
            self.memory_history.append(peak_memory_mb)
            if len(self.memory_history) > self.max_memory_history:
                self.memory_history.pop(0)

            # Add to batch-size-specific tracking for better predictions
            if actual_batch_size not in self.memory_by_batch_size:
                self.memory_by_batch_size[actual_batch_size] = []
            self.memory_by_batch_size[actual_batch_size].append(peak_memory_mb)

            # Limit samples per batch size to prevent unbounded growth
            if len(self.memory_by_batch_size[actual_batch_size]) > self.max_samples_per_size:
                self.memory_by_batch_size[actual_batch_size].pop(0)

        # Consider increasing batch size if we have headroom
        if self.enable_dynamic and self.should_increase_batch_size():
            self._increase_batch_size()

    def record_oom(self, failed_batch_size: int):
        """
        Record OOM error and reduce batch size.

        Args:
            failed_batch_size: Size of batch that caused OOM
        """
        self.oom_count += 1
        logger.warning(f"OOM detected at batch_size={failed_batch_size}")

        if self.enable_dynamic:
            self._decrease_batch_size()
        else:
            logger.warning("Dynamic batching disabled - keeping batch size fixed")

    def should_increase_batch_size(self) -> bool:
        """Check if batch size should be increased."""
        if not self.enable_dynamic:
            return False

        # Need several successes before trying to increase
        if self.success_count < 5:
            return False

        # Already at max
        if self.current_batch_size >= self.max_batch_size:
            return False

        # Check if we have memory headroom
        if self.memory_history and len(self.memory_history) >= 3:
            avg_memory_util = sum(self.memory_history) / len(self.memory_history)
            available_vram_mb = self._get_available_vram_mb()

            if available_vram_mb > 0:
                current_util = avg_memory_util / available_vram_mb
                # Only increase if we're using less than threshold
                if current_util < self.memory_threshold * 0.8:  # Conservative (68%)
                    return True

        return False

    def calculate_optimal_batch_size(self) -> Optional[int]:
        """
        Calculate optimal batch size based on memory usage patterns.

        Uses linear extrapolation from observed memory usage to predict
        the maximum batch size that fits in available VRAM.

        If we have data from multiple batch sizes, uses weighted regression
        for more accurate predictions.

        Returns:
            Optimal batch size, or None if not enough data
        """
        if not self.memory_history or len(self.memory_history) < 3:
            return None

        # Get available VRAM
        available_vram = self._get_available_vram_mb()
        if available_vram <= 0:
            return None

        # Calculate how much VRAM we can use (with safety margin)
        usable_vram = available_vram * self.memory_threshold

        # Try to use batch-size-specific data for better prediction
        if len(self.memory_by_batch_size) >= 2:
            # We have data from multiple batch sizes - use weighted regression
            memory_per_item = self._estimate_memory_per_item_from_history()
            if memory_per_item is None:
                # Fallback to simple method
                return self._calculate_optimal_simple(usable_vram)

            # Estimate base overhead (model weights, etc.)
            # Use smallest batch size data to estimate overhead
            smallest_batch = min(self.memory_by_batch_size.keys())
            avg_mem_smallest = sum(self.memory_by_batch_size[smallest_batch]) / len(self.memory_by_batch_size[smallest_batch])
            base_overhead = avg_mem_smallest - (memory_per_item * smallest_batch)
            base_overhead = max(0, base_overhead)  # Ensure non-negative

            # Predict optimal batch size
            optimal_size = int((usable_vram - base_overhead) / memory_per_item)

            # Clamp to valid range
            optimal_size = max(self.min_batch_size, min(optimal_size, self.max_batch_size))

            logger.debug(
                f"Optimal batch size calculation (regression): "
                f"mem_per_item={memory_per_item:.1f}MB, base_overhead={base_overhead:.1f}MB, "
                f"available={available_vram:.1f}MB, optimal={optimal_size}"
            )

            return optimal_size
        else:
            # Fallback to simple linear extrapolation
            return self._calculate_optimal_simple(usable_vram)

    def _estimate_memory_per_item_from_history(self) -> Optional[float]:
        """
        Estimate memory per item using weighted average from batch size history.

        Returns:
            Estimated memory per item in MB, or None if insufficient data
        """
        if not self.memory_by_batch_size:
            return None

        # Calculate memory-per-item for each batch size
        estimates = []
        weights = []

        for batch_size, memory_samples in self.memory_by_batch_size.items():
            if not memory_samples or batch_size == 0:
                continue

            avg_memory = sum(memory_samples) / len(memory_samples)
            memory_per_item = avg_memory / batch_size
            estimates.append(memory_per_item)

            # Weight by number of samples and recency (prefer recent data)
            weight = len(memory_samples)
            weights.append(weight)

        if not estimates:
            return None

        # Calculate weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return None

        weighted_avg = sum(est * wt for est, wt in zip(estimates, weights)) / total_weight

        logger.debug(
            f"Estimated memory per item: {weighted_avg:.1f}MB "
            f"(from {len(estimates)} batch sizes, total weight={total_weight})"
        )

        return weighted_avg

    def _calculate_optimal_simple(self, usable_vram: float) -> Optional[int]:
        """
        Simple optimal batch size calculation using current batch data.

        Args:
            usable_vram: Available VRAM to use (MB)

        Returns:
            Optimal batch size, or None if not enough data
        """
        if not self.memory_history:
            return None

        # Get average memory per item in current batch
        avg_memory_used = sum(self.memory_history) / len(self.memory_history)
        if self.last_successful_batch_size == 0:
            return None

        memory_per_item = avg_memory_used / self.last_successful_batch_size

        # We assume ~20% of current memory is base overhead (model weights, etc.)
        base_overhead = avg_memory_used * 0.2
        optimal_size = int((usable_vram - base_overhead) / memory_per_item)

        # Clamp to valid range
        optimal_size = max(self.min_batch_size, min(optimal_size, self.max_batch_size))

        logger.debug(
            f"Optimal batch size calculation (simple): "
            f"avg_mem={avg_memory_used:.1f}MB, mem_per_item={memory_per_item:.1f}MB, "
            f"available={usable_vram:.1f}MB, optimal={optimal_size}"
        )

        return optimal_size

    def _increase_batch_size(self):
        """Increase batch size intelligently based on available memory."""
        old_size = self.current_batch_size

        # Try to calculate optimal batch size
        optimal_size = self.calculate_optimal_batch_size()

        if optimal_size and optimal_size > self.current_batch_size:
            # We have enough data to predict - jump to optimal size
            # But be conservative: take midpoint between current and optimal
            new_size = (self.current_batch_size + optimal_size) // 2
            self.current_batch_size = min(new_size, self.max_batch_size)
            logger.info(
                f"Increasing batch size (predicted optimal={optimal_size}): "
                f"{old_size} → {self.current_batch_size}"
            )
        else:
            # Not enough data - conservative increment by 1
            self.current_batch_size = min(self.current_batch_size + 1, self.max_batch_size)
            logger.info(f"Increasing batch size (conservative): {old_size} → {self.current_batch_size}")

        if self.current_batch_size > old_size:
            self.success_count = 0  # Reset counter

    def _decrease_batch_size(self):
        """Decrease batch size after OOM."""
        old_size = self.current_batch_size
        # Reduce by half (more aggressive)
        self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)

        if self.current_batch_size < old_size:
            logger.warning(f"Reducing batch size due to OOM: {old_size} → {self.current_batch_size}")
            self.success_count = 0  # Reset counter

    def _get_available_vram_mb(self) -> float:
        """Get available VRAM in MB."""
        if not torch.cuda.is_available():
            return 0.0

        try:
            # Get total and allocated memory for device 0
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            available = (total_memory - allocated_memory) / (1024 ** 2)  # Convert to MB
            return available
        except Exception as e:
            logger.debug(f"Could not get VRAM info: {e}")
            return 0.0

    def get_statistics(self) -> dict:
        """Get statistics about batch management."""
        return {
            "current_batch_size": self.current_batch_size,
            "success_count": self.success_count,
            "oom_count": self.oom_count,
            "last_successful": self.last_successful_batch_size,
            "avg_memory_mb": sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0,
            "available_vram_mb": self._get_available_vram_mb()
        }

    def reset(self):
        """Reset statistics (keeps current batch size)."""
        self.success_count = 0
        self.oom_count = 0
        self.memory_history.clear()
        self.memory_by_batch_size.clear()


class DynamicBatchSizer:
    """
    Helper class to estimate optimal batch size based on sequence lengths.

    This uses a simpler heuristic approach based on sequence length.
    """

    @staticmethod
    def estimate_batch_size(
        sequence_lengths: List[int],
        available_memory_mb: float,
        model_memory_mb: float,
        bytes_per_token: int = 4,  # 4 bytes for fp32, 2 for fp16/bf16
        safety_margin: float = 0.8  # Use 80% of available memory
    ) -> int:
        """
        Estimate optimal batch size based on sequence lengths and memory.

        Args:
            sequence_lengths: List of sequence lengths in current batch
            available_memory_mb: Available GPU memory (MB)
            model_memory_mb: Memory used by model (MB)
            bytes_per_token: Bytes per token in KV cache
            safety_margin: Safety factor for memory estimation

        Returns:
            Estimated optimal batch size
        """
        if not sequence_lengths:
            return 1

        # Get max sequence length (worst case)
        max_seq_len = max(sequence_lengths)

        # Rough estimate: KV cache memory = batch_size * seq_len * hidden_size * num_layers * 2 (K+V)
        # Simplified: ~2KB per token per batch item for typical models
        memory_per_token_mb = bytes_per_token / 1024.0  # Convert to MB
        memory_per_item_mb = max_seq_len * memory_per_token_mb

        # Available memory after model
        usable_memory_mb = (available_memory_mb - model_memory_mb) * safety_margin

        if memory_per_item_mb > 0:
            estimated_batch_size = int(usable_memory_mb / memory_per_item_mb)
            return max(1, estimated_batch_size)

        return 1
