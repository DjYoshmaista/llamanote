# src/models/attention_patcher.py
"""
Attention Patcher Module

Provides utilities to monkey-patch transformer models with custom attention mechanisms,
including sliding window attention. This allows transparent integration of memory-efficient
attention without modifying model architectures.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Dict, Any, List
from functools import wraps

from .sliding_window_attention import SlidingWindowAttention, SlidingWindowConfig
from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class AttentionPatcher:
    """
    Patches transformer models to use sliding window attention.

    This patcher monkey-patches the forward methods of attention modules
    to apply sliding window context management automatically.
    """

    def __init__(self, sliding_window_config: Optional[SlidingWindowConfig] = None):
        """
        Initialize attention patcher.

        Args:
            sliding_window_config: Configuration for sliding window (uses defaults if None)
        """
        self.config = sliding_window_config or SlidingWindowConfig()
        self.sliding_window = SlidingWindowAttention(self.config)

        # Track patched modules
        self.patched_modules: Dict[str, nn.Module] = {}
        self.original_forwards: Dict[str, Callable] = {}

        logger.info("AttentionPatcher initialized")

    def patch_model(self, model: nn.Module, model_type: str = "auto") -> bool:
        """
        Patch a transformer model with sliding window attention.

        Args:
            model: The transformer model to patch
            model_type: Type of model ("auto", "llama", "mistral", "qwen", etc.)

        Returns:
            True if patching succeeded, False otherwise
        """
        logger.info(f"Patching model with sliding window attention (type={model_type})")

        try:
            # Auto-detect model type if needed
            if model_type == "auto":
                model_type = self._detect_model_type(model)
                logger.info(f"Auto-detected model type: {model_type}")

            # Find attention modules based on model type
            attention_modules = self._find_attention_modules(model, model_type)

            if not attention_modules:
                logger.warning("No attention modules found to patch")
                return False

            # Patch each attention module
            patched_count = 0
            for name, module in attention_modules:
                if self._patch_attention_module(name, module):
                    patched_count += 1

            logger.info(f"Successfully patched {patched_count}/{len(attention_modules)} attention modules")
            return patched_count > 0

        except Exception as e:
            logger.error(f"Failed to patch model: {e}", exc_info=True)
            return False

    def unpatch_model(self) -> bool:
        """
        Restore original attention modules.

        Returns:
            True if unpatching succeeded
        """
        logger.info("Unpatching model attention modules")

        try:
            for name, module in self.patched_modules.items():
                if name in self.original_forwards:
                    module.forward = self.original_forwards[name]
                    logger.debug(f"Restored original forward for: {name}")

            self.patched_modules.clear()
            self.original_forwards.clear()

            logger.info("Successfully unpatched all modules")
            return True

        except Exception as e:
            logger.error(f"Failed to unpatch model: {e}", exc_info=True)
            return False

    def _detect_model_type(self, model: nn.Module) -> str:
        """
        Auto-detect the model architecture type.

        Args:
            model: The model to detect

        Returns:
            Detected model type string
        """
        model_class_name = model.__class__.__name__.lower()

        if "llama" in model_class_name:
            return "llama"
        elif "mistral" in model_class_name:
            return "mistral"
        elif "qwen" in model_class_name:
            return "qwen"
        elif "gpt" in model_class_name:
            return "gpt"
        elif "phi" in model_class_name:
            return "phi"
        else:
            logger.warning(f"Unknown model type: {model_class_name}, using generic patching")
            return "generic"

    def _find_attention_modules(
        self, model: nn.Module, model_type: str
    ) -> List[tuple[str, nn.Module]]:
        """
        Find attention modules in the model.

        Args:
            model: The model to search
            model_type: Type of model

        Returns:
            List of (name, module) tuples for attention modules
        """
        attention_modules = []

        # Common attention module names across different architectures
        attention_names = [
            "self_attn",     # LLaMA, Mistral
            "attn",          # GPT-2, GPT-Neo
            "attention",     # Generic
            "self_attention", # Some models
        ]

        for name, module in model.named_modules():
            # Check if module name contains attention indicator
            module_name_lower = name.lower()
            if any(attn_name in module_name_lower for attn_name in attention_names):
                # Verify it's an actual attention module (has forward method)
                if hasattr(module, 'forward') and callable(module.forward):
                    attention_modules.append((name, module))
                    logger.debug(f"Found attention module: {name}")

        return attention_modules

    def _patch_attention_module(self, name: str, module: nn.Module) -> bool:
        """
        Patch a single attention module.

        Args:
            name: Module name
            module: The attention module

        Returns:
            True if patching succeeded
        """
        try:
            # Store original forward
            original_forward = module.forward

            # Create patched forward function
            @wraps(original_forward)
            def patched_forward(*args, **kwargs):
                # Extract attention inputs
                # This is a simplified version - full implementation would need
                # to handle specific argument structures for each model type

                # Call sliding window attention wrapper
                return self.sliding_window.forward(
                    *args,
                    original_attention_fn=original_forward,
                    **kwargs
                )

            # Apply patch
            module.forward = patched_forward

            # Track patched module
            self.patched_modules[name] = module
            self.original_forwards[name] = original_forward

            logger.debug(f"Patched attention module: {name}")
            return True

        except Exception as e:
            logger.error(f"Failed to patch module {name}: {e}", exc_info=True)
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics from sliding window attention."""
        stats = self.sliding_window.get_statistics()
        stats['patched_modules'] = len(self.patched_modules)
        return stats

    def format_statistics(self) -> str:
        """Format statistics for display."""
        base_stats = self.sliding_window.format_statistics()
        additional = f"\nPatched Modules: {len(self.patched_modules)}\n"
        return base_stats.replace("━" * 60 + "\n", "━" * 60 + additional, 1)

    def reset_statistics(self):
        """Reset sliding window statistics."""
        self.sliding_window.reset()

    def __repr__(self) -> str:
        """String representation."""
        return (f"AttentionPatcher(patched_modules={len(self.patched_modules)}, "
                f"window_size={self.config.window_size})")


# Convenience function for quick patching
def patch_model_with_sliding_window(
    model: nn.Module,
    window_size: int = 2048,
    keep_prefix: int = 128,
    stride: int = 512,
    model_type: str = "auto"
) -> Optional[AttentionPatcher]:
    """
    Convenience function to patch a model with sliding window attention.

    Args:
        model: The transformer model to patch
        window_size: Size of the sliding window
        keep_prefix: Number of prefix tokens to preserve
        stride: Sliding stride
        model_type: Model architecture type

    Returns:
        AttentionPatcher instance if successful, None otherwise
    """
    config = SlidingWindowConfig(
        window_size=window_size,
        stride=stride,
        keep_prefix_tokens=keep_prefix
    )

    patcher = AttentionPatcher(sliding_window_config=config)

    if patcher.patch_model(model, model_type=model_type):
        logger.info(f"Successfully patched model with sliding window (size={window_size})")
        return patcher
    else:
        logger.error("Failed to patch model with sliding window")
        return None
