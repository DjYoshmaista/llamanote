# src/utils/memory_estimator.py
"""
Memory Estimation Utilities

Provides accurate memory usage projections for models before loading.
Helps users understand memory requirements and avoid OOM errors.
"""

import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from transformers import AutoConfig

from .logger import get_logger_conf

logger = get_logger_conf(__name__)


@dataclass
class MemoryProjection:
    """Container for memory usage estimates."""
    model_size_mb: float
    kv_cache_size_mb: float
    context_size_mb: float
    total_size_mb: float
    available_vram_mb: float
    available_ram_mb: float
    quantization: str
    num_layers: int
    sufficient_vram: bool
    sufficient_ram: bool

    def format_size(self, mb: float) -> str:
        """Format size in MB to human-readable string."""
        if mb < 1024:
            return f"{mb:.2f} MB"
        else:
            return f"{mb / 1024:.2f} GB"

    def get_status_color(self) -> str:
        """Get ANSI color code based on memory status."""
        if self.sufficient_vram:
            return "\033[92m"  # Green
        elif self.sufficient_ram:
            return "\033[93m"  # Yellow
        else:
            return "\033[91m"  # Red

    def get_status_symbol(self) -> str:
        """Get status symbol."""
        if self.sufficient_vram:
            return "✓"
        elif self.sufficient_ram:
            return "⚠"
        else:
            return "✗"


class MemoryEstimator:
    """Estimates memory requirements for model loading and inference."""

    # Bytes per parameter for different quantization levels
    BYTES_PER_PARAM = {
        "none": 4.0,      # float32
        "16bit": 2.0,     # float16/bfloat16
        "8bit": 1.0,      # int8
        "4bit": 0.5,      # int4
    }

    # Overhead multipliers for different components
    OVERHEAD_MULTIPLIERS = {
        "model": 1.1,      # 10% overhead for buffers, gradients, etc.
        "kv_cache": 1.05,  # 5% overhead for cache management
        "context": 1.02,   # 2% overhead for context processing
    }

    @staticmethod
    def get_available_memory() -> Tuple[float, float]:
        """
        Get available VRAM and RAM in MB.

        Returns:
            Tuple of (available_vram_mb, available_ram_mb)
        """
        available_vram = 0.0
        available_ram = 0.0

        # Get VRAM
        if torch.cuda.is_available():
            try:
                available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                # Subtract currently allocated
                allocated = torch.cuda.memory_allocated(0) / (1024**2)
                available_vram -= allocated
            except Exception as e:
                logger.warning(f"Could not get VRAM info: {e}")

        # Get RAM (estimate based on system)
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_ram = mem.available / (1024**2)
        except ImportError:
            # Fallback estimate
            available_ram = 28 * 1024  # 28 GB default estimate
        except Exception as e:
            logger.warning(f"Could not get RAM info: {e}")
            available_ram = 28 * 1024

        return available_vram, available_ram

    @staticmethod
    def estimate_model_size(
        model_id: str,
        quantization: str = "4bit",
        cache_dir: Optional[Path] = None,
        trust_remote_code: bool = True,
        num_parameters: Optional[int] = None
    ) -> float:
        """
        Estimate model size in MB.

        Args:
            model_id: HuggingFace model identifier
            quantization: Quantization method ("none", "4bit", "8bit", "16bit")
            cache_dir: Model cache directory
            trust_remote_code: Whether to trust remote code
            num_parameters: If known, use this instead of loading config

        Returns:
            Estimated size in MB
        """
        try:
            # If num_parameters provided, use it directly
            if num_parameters is not None:
                bytes_per_param = MemoryEstimator.BYTES_PER_PARAM.get(quantization, 4.0)
                base_size_mb = (num_parameters * bytes_per_param) / (1024**2)
                overhead = MemoryEstimator.OVERHEAD_MULTIPLIERS["model"]
                return base_size_mb * overhead

            # Otherwise, load config to get parameter count
            config = AutoConfig.from_pretrained(
                model_id,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=trust_remote_code
            )

            # Try to get parameter count from config
            num_params = None

            # Different models store this differently
            if hasattr(config, 'num_parameters'):
                num_params = config.num_parameters
            elif hasattr(config, 'n_params'):
                num_params = config.n_params
            else:
                # Estimate from architecture
                hidden_size = getattr(config, 'hidden_size', 4096)
                num_layers = getattr(config, 'num_hidden_layers', None) or \
                             getattr(config, 'n_layer', None) or \
                             getattr(config, 'num_layers', 32)
                vocab_size = getattr(config, 'vocab_size', 50000)

                # Rough estimate: embeddings + layers
                # Each transformer layer ≈ 4 * hidden_size^2 + some bias
                embedding_params = vocab_size * hidden_size
                layer_params = num_layers * (12 * hidden_size * hidden_size)  # Approximation
                num_params = embedding_params + layer_params

            # Calculate size
            bytes_per_param = MemoryEstimator.BYTES_PER_PARAM.get(quantization, 4.0)
            base_size_mb = (num_params * bytes_per_param) / (1024**2)
            overhead = MemoryEstimator.OVERHEAD_MULTIPLIERS["model"]

            estimated_size = base_size_mb * overhead
            logger.debug(f"Estimated model size: {estimated_size:.2f} MB ({num_params:,} params, {quantization})")

            return estimated_size

        except Exception as e:
            logger.warning(f"Could not estimate model size accurately: {e}")
            # Fallback: assume ~1.5B params model with given quantization
            fallback_params = 1_500_000_000
            bytes_per_param = MemoryEstimator.BYTES_PER_PARAM.get(quantization, 4.0)
            return (fallback_params * bytes_per_param / (1024**2)) * 1.1

    @staticmethod
    def estimate_kv_cache_size(
        num_layers: int,
        hidden_size: int,
        num_attention_heads: int,
        max_seq_length: int = 4096,
        batch_size: int = 1,
        dtype_bytes: int = 2  # float16/bfloat16
    ) -> float:
        """
        Estimate KV cache size in MB.

        Args:
            num_layers: Number of model layers
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            max_seq_length: Maximum sequence length
            batch_size: Batch size
            dtype_bytes: Bytes per element (2 for fp16, 4 for fp32)

        Returns:
            Estimated KV cache size in MB
        """
        # KV cache stores keys and values for each layer
        # Shape: [batch_size, num_heads, seq_length, head_dim]
        head_dim = hidden_size // num_attention_heads

        # Size per layer: 2 (K+V) * batch * heads * seq_len * head_dim * bytes
        size_per_layer = 2 * batch_size * num_attention_heads * max_seq_length * head_dim * dtype_bytes

        # Total for all layers
        total_size_bytes = size_per_layer * num_layers
        total_size_mb = total_size_bytes / (1024**2)

        # Apply overhead
        overhead = MemoryEstimator.OVERHEAD_MULTIPLIERS["kv_cache"]
        estimated_size = total_size_mb * overhead

        logger.debug(f"Estimated KV cache size: {estimated_size:.2f} MB (seq_len={max_seq_length})")

        return estimated_size

    @staticmethod
    def estimate_context_size(
        max_seq_length: int = 4096,
        hidden_size: int = 4096,
        batch_size: int = 1,
        dtype_bytes: int = 2
    ) -> float:
        """
        Estimate context window size in MB.

        Args:
            max_seq_length: Maximum sequence length
            hidden_size: Hidden dimension size
            batch_size: Batch size
            dtype_bytes: Bytes per element

        Returns:
            Estimated context size in MB
        """
        # Context includes input embeddings and attention masks
        # Embeddings: [batch_size, seq_length, hidden_size]
        embeddings_size = batch_size * max_seq_length * hidden_size * dtype_bytes

        # Attention mask: [batch_size, seq_length, seq_length] (usually bool, 1 byte)
        attention_mask_size = batch_size * max_seq_length * max_seq_length * 1

        total_size_bytes = embeddings_size + attention_mask_size
        total_size_mb = total_size_bytes / (1024**2)

        # Apply overhead
        overhead = MemoryEstimator.OVERHEAD_MULTIPLIERS["context"]
        estimated_size = total_size_mb * overhead

        logger.debug(f"Estimated context size: {estimated_size:.2f} MB")

        return estimated_size

    @staticmethod
    def create_projection(
        model_id: str,
        quantization: str = "4bit",
        max_seq_length: int = 4096,
        cache_dir: Optional[Path] = None,
        trust_remote_code: bool = True
    ) -> MemoryProjection:
        """
        Create complete memory projection for a model.

        Args:
            model_id: HuggingFace model identifier
            quantization: Quantization method
            max_seq_length: Maximum sequence length
            cache_dir: Model cache directory
            trust_remote_code: Whether to trust remote code

        Returns:
            MemoryProjection object with all estimates
        """
        try:
            # Load config to get model architecture details
            config = AutoConfig.from_pretrained(
                model_id,
                cache_dir=str(cache_dir) if cache_dir else None,
                trust_remote_code=trust_remote_code
            )

            hidden_size = getattr(config, 'hidden_size', 4096)
            num_layers = getattr(config, 'num_hidden_layers', None) or \
                        getattr(config, 'n_layer', None) or \
                        getattr(config, 'num_layers', 32)
            num_heads = getattr(config, 'num_attention_heads', None) or \
                       getattr(config, 'n_head', 32)

            # Get dtype bytes based on quantization
            if quantization in ["4bit", "8bit"]:
                dtype_bytes = 2  # Computation still in fp16
            elif quantization == "16bit":
                dtype_bytes = 2
            else:
                dtype_bytes = 4  # fp32

        except Exception as e:
            logger.warning(f"Could not load model config, using defaults: {e}")
            hidden_size = 4096
            num_layers = 28
            num_heads = 32
            dtype_bytes = 2

        # Estimate components
        model_size = MemoryEstimator.estimate_model_size(
            model_id, quantization, cache_dir, trust_remote_code
        )

        kv_cache_size = MemoryEstimator.estimate_kv_cache_size(
            num_layers, hidden_size, num_heads, max_seq_length, dtype_bytes=dtype_bytes
        )

        context_size = MemoryEstimator.estimate_context_size(
            max_seq_length, hidden_size, dtype_bytes=dtype_bytes
        )

        total_size = model_size + kv_cache_size + context_size

        # Get available memory
        available_vram, available_ram = MemoryEstimator.get_available_memory()

        # Determine sufficiency
        sufficient_vram = total_size < available_vram
        sufficient_ram = total_size < available_ram

        return MemoryProjection(
            model_size_mb=model_size,
            kv_cache_size_mb=kv_cache_size,
            context_size_mb=context_size,
            total_size_mb=total_size,
            available_vram_mb=available_vram,
            available_ram_mb=available_ram,
            quantization=quantization,
            num_layers=num_layers,
            sufficient_vram=sufficient_vram,
            sufficient_ram=sufficient_ram
        )

    @staticmethod
    def format_projection(projection: MemoryProjection, show_color: bool = True) -> str:
        """
        Format memory projection for display.

        Args:
            projection: MemoryProjection object
            show_color: Whether to use ANSI color codes

        Returns:
            Formatted string for display
        """
        reset = "\033[0m" if show_color else ""
        bold = "\033[1m" if show_color else ""
        color = projection.get_status_color() if show_color else ""
        symbol = projection.get_status_symbol()

        lines = [
            "",
            "━" * 60,
            f"{bold}📊 Memory Projection{reset}",
            "━" * 60,
            f"Quantization: {projection.quantization}",
            f"Layers: {projection.num_layers}",
            "",
            "Estimated Memory Requirements:",
            f"  Base Model:        {projection.format_size(projection.model_size_mb)}",
            f"  KV-Cache:          {projection.format_size(projection.kv_cache_size_mb)}",
            f"  Context Window:    {projection.format_size(projection.context_size_mb)}",
            "  " + "─" * 40,
            f"  {bold}Total Estimated:   {projection.format_size(projection.total_size_mb)}{reset}",
            "",
            "System Resources:",
            f"  Available VRAM:    {projection.format_size(projection.available_vram_mb)} {color}{symbol if projection.sufficient_vram else '✗'}{reset}",
            f"  Available RAM:     {projection.format_size(projection.available_ram_mb)} {color}{symbol if projection.sufficient_ram else '✗'}{reset}",
            "",
        ]

        # Add status message
        if projection.sufficient_vram:
            lines.append(f"{color}Status: ✓ Sufficient VRAM available{reset}")
            lines.append("Proceeding with GPU load...")
        elif projection.sufficient_ram:
            lines.append(f"{color}Status: ⚠ Insufficient VRAM, will use CPU offloading{reset}")
            lines.append("Proceeding with CPU/GPU split...")
        else:
            lines.append(f"{color}Status: ✗ Insufficient memory{reset}")
            lines.append("Consider: 1) Smaller model, 2) Aggressive quantization, 3) Reduce context length")

        lines.append("━" * 60)
        lines.append("")

        return "\n".join(lines)


def display_memory_projection(
    model_id: str,
    quantization: str = "4bit",
    max_seq_length: int = 4096,
    cache_dir: Optional[Path] = None,
    trust_remote_code: bool = True
) -> MemoryProjection:
    """
    Create and display memory projection.

    Args:
        model_id: HuggingFace model identifier
        quantization: Quantization method
        max_seq_length: Maximum sequence length
        cache_dir: Model cache directory
        trust_remote_code: Whether to trust remote code

    Returns:
        MemoryProjection object
    """
    projection = MemoryEstimator.create_projection(
        model_id, quantization, max_seq_length, cache_dir, trust_remote_code
    )

    formatted = MemoryEstimator.format_projection(projection)
    print(formatted)

    return projection
