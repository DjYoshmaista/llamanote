# src/models/cache/generation_wrapper.py
"""
Cached Generation Wrapper

This module provides a wrapper around a HuggingFace model's `generate` method
to enable advanced caching and sliding window attention.

It orchestrates the interaction between:
- The HuggingFace model
- The model-specific adapter
- The custom LlamaNoteDynamicCache

This allows for a seamless, drop-in replacement for the standard `generate`
method with enhanced performance and memory management.
"""

from typing import Optional, Dict, Any, Type
import torch

from .transformers_cache_impl import LlamaNoteDynamicCache, CacheStrategyConfig
from .model_adapters import apply_adapter, get_adapter, ModelAdapter
from ...utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

# --- Conditional Imports ---
try:
    from transformers import PreTrainedModel, GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    class PreTrainedModel: pass
    class GenerationConfig: pass


class CachedGenerationWrapper:
    """
    Wraps a HuggingFace model to provide cached generation.

    This class applies the necessary adaptations to the model and uses a
    custom cache implementation during generation to achieve better performance
    and memory efficiency.

    Example:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        wrapper = CachedGenerationWrapper(model)

        # Use the wrapper's generate method instead of the model's
        outputs = wrapper.generate(**inputs)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        strategy: str = "balanced",
        enable_hybrid_cache: bool = True,
        enable_sliding_window: bool = True,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not found")

        self.model = model
        self.model_type = model.config.model_type
        self.adapter: Optional[ModelAdapter] = None

        # Initialize cache
        strategy_config = CacheStrategyConfig(strategy=strategy)
        self.cache = LlamaNoteDynamicCache(
            strategy_config=strategy_config,
            enable_hybrid_cache=enable_hybrid_cache,
            enable_sliding_window=enable_sliding_window,
            model_type=self.model_type
        )

        # Apply model adapter
        self._apply_model_adapter()

    def _apply_model_adapter(self):
        """Find and apply the appropriate model adapter."""
        AdapterClass = get_adapter(self.model_type)
        if AdapterClass:
            self.adapter = AdapterClass(self.model, self.model.config)
            if self.adapter.adapt():
                logger.info(f"Successfully applied '{self.adapter.__class__.__name__}' to model")
            else:
                logger.error(f"Failed to apply adapter for model type '{self.model_type}'")
                self.adapter = None
        else:
            logger.warning(f"No adapter found for '{self.model_type}'. Using fallback behavior.")

    def generate(self, *args, **kwargs) -> Any:
        """
        Perform cached generation.

        This method intercepts the call to the model's `generate` method,
        injects the custom cache, and handles any necessary input/output
        transformations.

        Args:
            *args: Positional arguments for the model's `generate` method.
            **kwargs: Keyword arguments for the model's `generate` method.

        Returns:
            The output from the model's `generate` method.
        """
        if self.adapter is None:
            logger.warning("No adapter applied, falling back to standard generation")
            return self.model.generate(*args, **kwargs)

        # --- Prepare for Cached Generation ---
        logger.debug("Starting cached generation")

        # 1. Inject the custom cache into kwargs
        # The `past_key_values` argument is where Transformers expects the cache
        kwargs["past_key_values"] = self.cache

        # 2. Ensure `use_cache` is True
        kwargs["use_cache"] = True

        # 3. Prepare inputs if the adapter requires it
        # (This is not typically needed for Llama-like models)
        # For example, if we were adapting a model that doesn't use `position_ids`
        # in the same way, we might need to add or modify them here.
        # inputs = self.adapter.prepare_inputs(kwargs)

        # 4. Modify generation config for sliding window
        # This is crucial for making the model aware of the cache's behavior.
        # We can either pass a modified GenerationConfig or set attributes directly.
        generation_config = kwargs.get("generation_config", self.model.generation_config)

        if isinstance(generation_config, GenerationConfig):
            # Set sliding window parameters if not already set
            if not hasattr(generation_config, 'sliding_window') or generation_config.sliding_window is None:
                generation_config.sliding_window = self.cache.strategy_config.get_window_size()
                logger.debug(f"Set generation_config.sliding_window to {generation_config.sliding_window}")

            # Transformers uses `attention_mask` to handle the sliding window when
            # `_attn_implementation` is "sdpa". We need to ensure the logic in
            # the patched `_update_causal_mask` is correctly triggered.

        kwargs["generation_config"] = generation_config

        # --- Execute Generation ---
        logger.debug(f"Calling model.generate with custom cache and config")
        try:
            outputs = self.model.generate(*args, **kwargs)
            logger.debug("Cached generation finished")

            # --- Post-Generation ---
            # Log cache statistics
            logger.info(self.cache.format_statistics())

            return outputs

        except Exception as e:
            logger.error(f"Error during cached generation: {e}", exc_info=True)
            # In case of error, log stats and re-raise
            logger.info(self.cache.format_statistics())
            raise

    def reset_cache(self):
        """Reset the internal cache state."""
        self.cache.reset()
        logger.info("Generation wrapper cache has been reset.")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics from the internal cache."""
        return self.cache.get_statistics()

    def __getattr__(self, name: str) -> Any:
        """
        Forward any other attribute access to the underlying model.

        This makes the wrapper transparent, so you can still call other model
        methods like `from_pretrained`, `to`, etc.
        """
        return getattr(self.model, name)