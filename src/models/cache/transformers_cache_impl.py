# src/models/cache/transformers_cache_impl.py
"""
Transformers-Compatible Cache Implementation

Provides a proper implementation of HuggingFace Transformers Cache interface
that integrates hybrid KV-cache and sliding window attention.

This module is standalone and can be used independently in other frameworks.
"""

import torch
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

try:
    from transformers.cache_utils import Cache, DynamicCache, CacheLayerMixin
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create minimal stubs for standalone use
    class Cache:
        def update(self, *args, **kwargs): raise NotImplementedError
        def get_seq_length(self, *args, **kwargs): raise NotImplementedError
        def get_max_length(self): raise NotImplementedError
    DynamicCache = Cache
    class CacheLayerMixin:
        is_compileable = False

from ...utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class LlamaNoteLayer(CacheLayerMixin if TRANSFORMERS_AVAILABLE else object):
    """
    Custom cache layer that wraps key-value tensors.

    This is needed for compatibility with transformers' Cache interface,
    which expects self.layers to be a list of layer objects, not integers.
    """

    # Mark as not compileable (transformers checks this attribute)
    is_compileable = False
    is_sliding = False

    def __init__(self, layer_idx: int):
        if TRANSFORMERS_AVAILABLE:
            super().__init__()
        self.layer_idx = layer_idx
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.is_initialized = False
        self.dtype = None
        self.device = None

    def lazy_initialization(self, key_states: torch.Tensor):
        """Initialize layer with tensor properties."""
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the layer's key-value cache."""
        if not self.is_initialized:
            self.lazy_initialization(key_states)
            self.keys = key_states
            self.values = value_states
        else:
            # Concatenate along sequence dimension
            self.keys = torch.cat([self.keys, key_states], dim=2)
            self.values = torch.cat([self.values, value_states], dim=2)

        return self.keys, self.values

    def get_seq_length(self) -> int:
        """Get the sequence length for this layer."""
        if self.keys is None or not self.is_initialized:
            return 0
        return self.keys.shape[2]

    def get_mask_sizes(self, cache_position: torch.Tensor) -> Tuple[int, int]:
        """
        Return the length and offset of the cache, used to generate the mask.

        Args:
            cache_position: Current cache position tensor

        Returns:
            Tuple of (kv_length, kv_offset)
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int:
        """
        Returns the maximum sequence length of the cache object.

        For dynamic cache layers, there is no maximum length, so return -1.
        """
        return -1

    def __repr__(self) -> str:
        return f"LlamaNoteLayer(idx={self.layer_idx}, seq_len={self.get_seq_length()})"


class MemoryStrategy(Enum):
    """Memory management strategy for cache."""
    AGGRESSIVE = "aggressive"  # Aggressive memory savings, smaller windows
    BALANCED = "balanced"      # Moderate windows, balance quality/memory
    QUALITY = "quality"        # Large windows, prioritize quality


@dataclass
class CacheStrategyConfig:
    """Configuration for cache memory strategy."""
    strategy: MemoryStrategy = MemoryStrategy.BALANCED

    # Window sizes for each strategy
    window_sizes: Dict[str, int] = None
    prefix_sizes: Dict[str, int] = None
    hot_cache_sizes: Dict[str, float] = None  # MB
    cold_cache_sizes: Dict[str, float] = None  # MB

    def __post_init__(self):
        """Set defaults based on strategy and validate strategy type."""
        if isinstance(self.strategy, str):
            try:
                self.strategy = MemoryStrategy(self.strategy.lower())
            except ValueError:
                logger.warning(f"Invalid memory strategy '{self.strategy}'. Defaulting to 'balanced'.")
                self.strategy = MemoryStrategy.BALANCED

        if self.window_sizes is None:
            self.window_sizes = {
                "aggressive": 1024,
                "balanced": 2048,
                "quality": 4096
            }

        if self.prefix_sizes is None:
            self.prefix_sizes = {
                "aggressive": 64,
                "balanced": 128,
                "quality": 256
            }

        if self.hot_cache_sizes is None:
            self.hot_cache_sizes = {
                "aggressive": 256.0,  # MB
                "balanced": 512.0,
                "quality": 1024.0
            }

        if self.cold_cache_sizes is None:
            self.cold_cache_sizes = {
                "aggressive": 1024.0,  # MB
                "balanced": 2048.0,
                "quality": 4096.0
            }

    def get_window_size(self) -> int:
        """Get window size for current strategy."""
        return self.window_sizes[self.strategy.value]

    def get_prefix_size(self) -> int:
        """Get prefix size for current strategy."""
        return self.prefix_sizes[self.strategy.value]

    def get_hot_cache_size(self) -> float:
        """Get hot cache size in MB for current strategy."""
        return self.hot_cache_sizes[self.strategy.value]

    def get_cold_cache_size(self) -> float:
        """Get cold cache size in MB for current strategy."""
        return self.cold_cache_sizes[self.strategy.value]


class LlamaNoteDynamicCache(Cache if TRANSFORMERS_AVAILABLE else object):
    """
    Custom Cache implementation that integrates with HuggingFace Transformers.

    This cache supports:
    - Hybrid hot/cold storage with LRU eviction
    - Sliding window attention with prefix preservation
    - Configurable memory strategies (aggressive/balanced/quality)
    - Model-agnostic interface

    Can be used standalone or integrated into generation pipelines.
    """

    def __init__(
        self,
        strategy_config: Optional[CacheStrategyConfig] = None,
        enable_hybrid_cache: bool = True,
        enable_sliding_window: bool = True,
        model_type: Optional[str] = None
    ):
        """Initialize the cache."""
        self.strategy_config = strategy_config or CacheStrategyConfig()
        self.enable_hybrid = enable_hybrid_cache
        self.enable_sliding = enable_sliding_window
        self.model_type = model_type

        # Storage for key-value pairs per layer
        # Structure: {layer_idx: (key_tensor, value_tensor)}
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Compatibility with transformers Cache interface
        # This list contains LlamaNoteLayer objects (not integers!)
        # Transformers expects layer objects with is_compileable attribute
        self.layers: List[LlamaNoteLayer] = []

        # Track mapping from layer index to layer object
        self._layer_map: Dict[int, LlamaNoteLayer] = {}

        # Track which layers are in hot vs cold storage
        self._hot_layers: set = set()
        self._cold_layers: set = set()

        # Cold storage (CPU)
        self._cold_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # LRU tracking
        self._access_counts: Dict[int, int] = {}
        self._access_order: List[int] = []  # Most recent at end

        # Sliding window state
        self._total_tokens_seen = 0
        self._window_slides = 0
        self._tokens_discarded = 0

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'promotions': 0,  # Cold -> Hot
            'demotions': 0,   # Hot -> Cold
            'evictions': 0,   # Complete removals
            'total_tokens': 0,
            'window_slides': 0,
        }

        logger.info(f"LlamaNoteDynamicCache initialized: strategy={self.strategy_config.strategy.value}, "
                   f"hybrid={enable_hybrid_cache}, sliding={enable_sliding_window}")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states.

        This is the main interface method called by model.generate().

        Args:
            key_states: Key tensor [batch, num_heads, seq_len, head_dim]
            value_states: Value tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_kwargs: Additional cache arguments

        Returns:
            Tuple of (updated_keys, updated_values)
        """
        # Track access
        self._record_access(layer_idx)

        # Get existing cache for this layer
        if layer_idx in self._cache:
            # Append new states to existing cache
            existing_keys, existing_values = self._cache[layer_idx]

            # Concatenate along sequence dimension
            updated_keys = torch.cat([existing_keys, key_states], dim=2)
            updated_values = torch.cat([existing_values, value_states], dim=2)

            self.stats['cache_hits'] += 1
        elif layer_idx in self._cold_cache and self.enable_hybrid:
            # Promote from cold storage
            existing_keys, existing_values = self._cold_cache[layer_idx]

            # Move to hot device and concatenate
            device = key_states.device
            existing_keys = existing_keys.to(device)
            existing_values = existing_values.to(device)

            updated_keys = torch.cat([existing_keys, key_states], dim=2)
            updated_values = torch.cat([existing_values, value_states], dim=2)

            # Remove from cold storage
            del self._cold_cache[layer_idx]
            self._cold_layers.discard(layer_idx)

            self.stats['promotions'] += 1
            logger.debug(f"Promoted layer {layer_idx} from cold to hot cache")
        else:
            # New layer, just use the new states
            updated_keys = key_states
            updated_values = value_states
            self.stats['cache_misses'] += 1

        # Apply sliding window if enabled
        if self.enable_sliding:
            updated_keys, updated_values = self._apply_sliding_window(
                updated_keys, updated_values, layer_idx
            )

        # Store in hot cache
        self._cache[layer_idx] = (updated_keys, updated_values)

        # Update layers list for transformers compatibility
        if layer_idx not in self._layer_map:
            # Create a new layer object
            layer_obj = LlamaNoteLayer(layer_idx)
            self._layer_map[layer_idx] = layer_obj
            self.layers.append(layer_obj)

        # Update the layer object with the new key-value states
        self._layer_map[layer_idx].keys = updated_keys
        self._layer_map[layer_idx].values = updated_values
        self._layer_map[layer_idx].is_initialized = True

        self._hot_layers.add(layer_idx)

        # Check if we need to evict
        if self.enable_hybrid:
            self._manage_cache_size()

        return updated_keys, updated_values

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Get sequence length for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Sequence length
        """
        if layer_idx in self._cache:
            return self._cache[layer_idx][0].shape[2]
        elif layer_idx in self._cold_cache:
            return self._cold_cache[layer_idx][0].shape[2]
        return 0

    def get_max_length(self) -> Optional[int]:
        """Get maximum sequence length across all layers."""
        max_len = 0

        for layer_idx in list(self._cache.keys()) + list(self._cold_cache.keys()):
            seq_len = self.get_seq_length(layer_idx)
            max_len = max(max_len, seq_len)

        return max_len if max_len > 0 else None

    @property
    def is_initialized(self) -> bool:
        """Return whether the cache data is initialized."""
        return len(self.layers) > 0 and all(layer.is_initialized for layer in self.layers)

    def _record_access(self, layer_idx: int):
        """Record access to a layer for LRU tracking."""
        self._access_counts[layer_idx] = self._access_counts.get(layer_idx, 0) + 1

        # Update access order (LRU)
        if layer_idx in self._access_order:
            self._access_order.remove(layer_idx)
        self._access_order.append(layer_idx)

    def _apply_sliding_window(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sliding window to key-value tensors.

        Args:
            keys: Key tensor
            values: Value tensor
            layer_idx: Layer index

        Returns:
            Windowed (keys, values)
        """
        seq_len = keys.shape[2]
        window_size = self.strategy_config.get_window_size()
        prefix_size = self.strategy_config.get_prefix_size()

        # Update statistics
        self._total_tokens_seen = max(self._total_tokens_seen, seq_len)
        self.stats['total_tokens'] = self._total_tokens_seen

        # Check if windowing is needed
        if seq_len <= window_size:
            return keys, values

        # Apply sliding window with prefix preservation
        if prefix_size > 0:
            # Keep prefix + most recent tokens
            actual_prefix = min(prefix_size, seq_len)
            remaining_window = window_size - actual_prefix

            if seq_len > actual_prefix + remaining_window:
                # Need to slide
                prefix_keys = keys[:, :, :actual_prefix, :]
                prefix_values = values[:, :, :actual_prefix, :]

                recent_start = seq_len - remaining_window
                recent_keys = keys[:, :, recent_start:, :]
                recent_values = values[:, :, recent_start:, :]

                windowed_keys = torch.cat([prefix_keys, recent_keys], dim=2)
                windowed_values = torch.cat([prefix_values, recent_values], dim=2)

                # Update statistics
                self._window_slides += 1
                self._tokens_discarded += (recent_start - actual_prefix)
                self.stats['window_slides'] = self._window_slides

                logger.debug(f"Slid window for layer {layer_idx}: "
                           f"kept {actual_prefix} prefix + {remaining_window} recent, "
                           f"discarded {recent_start - actual_prefix} tokens")

                return windowed_keys, windowed_values
        else:
            # No prefix, just keep most recent tokens
            if seq_len > window_size:
                start_pos = seq_len - window_size
                windowed_keys = keys[:, :, start_pos:, :]
                windowed_values = values[:, :, start_pos:, :]

                self._window_slides += 1
                self._tokens_discarded += start_pos
                self.stats['window_slides'] = self._window_slides

                return windowed_keys, windowed_values

        return keys, values

    def _manage_cache_size(self):
        """Manage cache size by evicting LRU entries to cold storage."""
        # Calculate current hot cache size
        hot_size_mb = self._calculate_cache_size(self._cache)
        hot_limit_mb = self.strategy_config.get_hot_cache_size()

        # Check if we need to evict
        if hot_size_mb > hot_limit_mb:
            # Evict LRU entries to cold storage
            num_to_evict = max(1, len(self._cache) // 4)  # Evict 25% at a time

            for _ in range(num_to_evict):
                if not self._access_order:
                    break

                # Get least recently used layer
                lru_layer = self._access_order.pop(0)

                if lru_layer in self._cache:
                    # Move to cold storage
                    keys, values = self._cache[lru_layer]

                    # Move to CPU
                    cold_keys = keys.cpu()
                    cold_values = values.cpu()

                    self._cold_cache[lru_layer] = (cold_keys, cold_values)
                    self._cold_layers.add(lru_layer)

                    # Remove from hot storage
                    del self._cache[lru_layer]
                    self._hot_layers.discard(lru_layer)

                    self.stats['demotions'] += 1

                    logger.debug(f"Demoted layer {lru_layer} to cold cache")

        # Check cold cache size
        cold_size_mb = self._calculate_cache_size(self._cold_cache)
        cold_limit_mb = self.strategy_config.get_cold_cache_size()

        if cold_size_mb > cold_limit_mb:
            # Evict completely (remove oldest)
            num_to_evict = max(1, len(self._cold_cache) // 4)

            # Get oldest layers
            cold_layers = sorted(self._cold_cache.keys())[:num_to_evict]

            for layer_idx in cold_layers:
                del self._cold_cache[layer_idx]
                self._cold_layers.discard(layer_idx)
                self.stats['evictions'] += 1

                logger.debug(f"Evicted layer {layer_idx} from cold cache")

    def _calculate_cache_size(self, cache_dict: Dict) -> float:
        """Calculate cache size in MB."""
        total_bytes = 0

        for keys, values in cache_dict.values():
            total_bytes += keys.element_size() * keys.nelement()
            total_bytes += values.element_size() * values.nelement()

        return total_bytes / (1024 * 1024)

    def reset(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._cold_cache.clear()
        self._hot_layers.clear()
        self._cold_layers.clear()
        self._access_counts.clear()
        self._access_order.clear()
        self.layers.clear()
        self._layer_map.clear()

        logger.info("Cache reset")

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        hot_size = self._calculate_cache_size(self._cache)
        cold_size = self._calculate_cache_size(self._cold_cache)

        stats = {
            **self.stats,
            'hot_cache_size_mb': hot_size,
            'cold_cache_size_mb': cold_size,
            'hot_layers': len(self._hot_layers),
            'cold_layers': len(self._cold_layers),
            'total_layers': len(self._hot_layers) + len(self._cold_layers),
            'tokens_discarded': self._tokens_discarded,
            'strategy': self.strategy_config.strategy.value,
            'window_size': self.strategy_config.get_window_size(),
            'prefix_size': self.strategy_config.get_prefix_size(),
        }

        return stats

    def format_statistics(self) -> str:
        """Format statistics for display."""
        stats = self.get_statistics()

        lines = [
            "",
            "━" * 70,
            "📊 LlamaNote Dynamic Cache Statistics",
            "━" * 70,
            f"Strategy: {stats['strategy'].upper()}",
            f"Window Size: {stats['window_size']} tokens | Prefix: {stats['prefix_size']} tokens",
            "",
            "Cache Tiers:",
            f"  Hot (GPU):  {stats['hot_layers']} layers ({stats['hot_cache_size_mb']:.2f} MB)",
            f"  Cold (CPU): {stats['cold_layers']} layers ({stats['cold_cache_size_mb']:.2f} MB)",
            "",
            "Access Patterns:",
            f"  Cache Hits:   {stats['cache_hits']}",
            f"  Cache Misses: {stats['cache_misses']}",
            f"  Hit Rate:     {stats['cache_hits'] / max(1, stats['cache_hits'] + stats['cache_misses']) * 100:.1f}%",
            "",
            "Memory Management:",
            f"  Promotions (Cold→Hot): {stats['promotions']}",
            f"  Demotions (Hot→Cold):  {stats['demotions']}",
            f"  Evictions (Complete):  {stats['evictions']}",
            "",
            "Sliding Window:",
            f"  Total Tokens Seen: {stats['total_tokens']}",
            f"  Window Slides:     {stats['window_slides']}",
            f"  Tokens Discarded:  {stats['tokens_discarded']}",
            "━" * 70,
            ""
        ]

        return "\n".join(lines)

    def __len__(self) -> int:
        """Return the number of layers in the cache."""
        return len(self.layers)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Support for backwards-compatible `past_key_values` indexing.

        This allows code like `past_key_values[0][0].shape[2]` to get the sequence length.

        Args:
            layer_idx: Layer index

        Returns:
            Tuple of (keys, values) tensors
        """
        if layer_idx < len(self.layers):
            layer = self.layers[layer_idx]
            if layer.keys is not None and layer.values is not None:
                return layer.keys, layer.values
            else:
                # Return empty tensors if not initialized
                return torch.tensor([]), torch.tensor([])
        else:
            raise KeyError(
                f"Cache only has {len(self.layers)} layers, attempted to access layer with index {layer_idx}"
            )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LlamaNoteDynamicCache(strategy={self.strategy_config.strategy.value}, "
            f"hot_layers={len(self._hot_layers)}, cold_layers={len(self._cold_layers)})"
        )