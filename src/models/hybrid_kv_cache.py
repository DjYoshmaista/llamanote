# src/models/hybrid_kv_cache.py
"""
Hybrid KV-Cache with LRU Management

Implements a two-level Key-Value cache for transformer models:
- Hot cache: Fast storage (GPU VRAM) for frequently accessed entries
- Cold cache: Slow storage (CPU RAM or disk) for rarely accessed entries

Uses LRU eviction policy to automatically manage hot/cold data split.
Compatible with HuggingFace Transformers Cache interface.
"""

import torch
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

try:
    from transformers.cache_utils import Cache
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Fallback if transformers not available
    TRANSFORMERS_AVAILABLE = False
    Cache = None

from ..utils.lru_cache_manager import LRUCacheManager
from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


@dataclass
class CacheConfig:
    """Configuration for Hybrid KV-Cache."""
    hot_cache_max_size_mb: float = 512.0  # Max size for hot cache in MB
    cold_cache_max_size_mb: float = 2048.0  # Max size for cold cache in MB
    eviction_batch_size: int = 4  # Number of entries to evict at once
    hot_threshold_percentile: float = 0.75  # Top 25% most accessed are hot
    device_hot: str = "cuda"  # Device for hot cache
    device_cold: str = "cpu"  # Device for cold cache
    enable_cold_cache: bool = True  # Enable cold cache tier


class HybridKVCache:
    """
    Hybrid two-level KV-Cache with automatic LRU-based management.

    This cache maintains two storage tiers:
    1. Hot cache (GPU): Fast access for frequently used entries
    2. Cold cache (CPU/Disk): Slower access for rarely used entries

    The cache automatically promotes cold entries to hot when accessed,
    and demotes hot entries to cold when the hot cache is full.

    Compatible with HuggingFace Transformers Cache interface.
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the hybrid KV-cache.

        Args:
            config: Cache configuration (uses defaults if None)
        """

        self.config = config or CacheConfig()

        # Convert MB to bytes for LRU manager
        hot_size_bytes = int(self.config.hot_cache_max_size_mb * 1024 * 1024)
        cold_size_bytes = int(self.config.cold_cache_max_size_mb * 1024 * 1024)

        # Initialize LRU managers for each tier
        self.hot_lru = LRUCacheManager(max_hot_size_bytes=hot_size_bytes)
        self.cold_lru = LRUCacheManager(max_hot_size_bytes=cold_size_bytes) if self.config.enable_cold_cache else None

        # Storage for actual cache data
        # Structure: {layer_idx: {'keys': Tensor, 'values': Tensor, 'device': str}}
        self.hot_storage: Dict[int, Dict[str, Any]] = {}
        self.cold_storage: Dict[int, Dict[str, Any]] = {} if self.config.enable_cold_cache else None

        # Track which layers are in which tier
        self.hot_layers: set = set()
        self.cold_layers: set = set() if self.config.enable_cold_cache else set()

        # Statistics
        self.promotions: int = 0  # Cold -> Hot moves
        self.demotions: int = 0   # Hot -> Cold moves

        logger.info(f"HybridKVCache initialized: hot={self.config.hot_cache_max_size_mb}MB, "
                   f"cold={self.config.cold_cache_max_size_mb}MB")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states for a layer.

        This is the main interface method compatible with Transformers.

        Args:
            key_states: Key tensor [batch, num_heads, seq_len, head_dim]
            value_states: Value tensor [batch, num_heads, seq_len, head_dim]
            layer_idx: The layer index being cached
            cache_kwargs: Additional cache arguments (unused)

        Returns:
            Tuple of (updated_keys, updated_values)
        """
        # Calculate size of this cache entry
        entry_size = self._calculate_tensor_size(key_states) + self._calculate_tensor_size(value_states)

        # Check if layer already exists in cache
        if layer_idx in self.hot_layers:
            # Update existing hot entry
            self._update_hot_entry(layer_idx, key_states, value_states)
            self.hot_lru.access(layer_idx)
        elif self.cold_layers and layer_idx in self.cold_layers:
            # Promote from cold to hot
            self._promote_to_hot(layer_idx, key_states, value_states)
        else:
            # New entry - add to hot cache
            self._add_to_hot(layer_idx, key_states, value_states, entry_size)

        # Return the cached values (from hot cache after potential promotion)
        return self.hot_storage[layer_idx]['keys'], self.hot_storage[layer_idx]['values']

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Get the sequence length for a layer.

        Args:
            layer_idx: Layer index (default 0)

        Returns:
            Sequence length of cached keys/values
        """
        if layer_idx in self.hot_storage:
            return self.hot_storage[layer_idx]['keys'].shape[2]
        elif self.cold_storage and layer_idx in self.cold_storage:
            return self.cold_storage[layer_idx]['keys'].shape[2]
        return 0

    def get_max_length(self) -> Optional[int]:
        """Get maximum sequence length across all cached layers."""
        max_len = 0
        for layer_idx in list(self.hot_storage.keys()) + (list(self.cold_storage.keys()) if self.cold_storage else []):
            seq_len = self.get_seq_length(layer_idx)
            max_len = max(max_len, seq_len)
        return max_len if max_len > 0 else None

    def _calculate_tensor_size(self, tensor: torch.Tensor) -> int:
        """Calculate size of tensor in bytes."""
        return tensor.element_size() * tensor.nelement()

    def _add_to_hot(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor, size: int):
        """Add a new entry to hot cache, evicting if necessary."""
        # Check if hot cache is full
        if self.hot_lru.is_full():
            self._evict_from_hot()

        # Move tensors to hot device if needed
        if keys.device.type != self.config.device_hot:
            keys = keys.to(self.config.device_hot)
        if values.device.type != self.config.device_hot:
            values = values.to(self.config.device_hot)

        # Store in hot cache
        self.hot_storage[layer_idx] = {
            'keys': keys,
            'values': values,
            'device': self.config.device_hot
        }
        self.hot_layers.add(layer_idx)
        self.hot_lru.add(layer_idx, size)

        logger.debug(f"Added layer {layer_idx} to hot cache (size={size} bytes)")

    def _update_hot_entry(self, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
        """Update an existing hot cache entry by concatenating new states."""
        existing_keys = self.hot_storage[layer_idx]['keys']
        existing_values = self.hot_storage[layer_idx]['values']

        # Concatenate along sequence dimension (dim=2)
        updated_keys = torch.cat([existing_keys, keys], dim=2)
        updated_values = torch.cat([existing_values, values], dim=2)

        self.hot_storage[layer_idx]['keys'] = updated_keys
        self.hot_storage[layer_idx]['values'] = updated_values

        # Update size in LRU
        new_size = self._calculate_tensor_size(updated_keys) + self._calculate_tensor_size(updated_values)
        self.hot_lru.add(layer_idx, new_size)

        logger.debug(f"Updated layer {layer_idx} in hot cache (new size={new_size} bytes)")

    def _evict_from_hot(self):
        """Evict least recently used entries from hot cache."""
        if not self.config.enable_cold_cache:
            # If no cold cache, just remove entries
            evicted = self.hot_lru.evict_lru(count=self.config.eviction_batch_size)
            for layer_idx, _ in evicted:
                if layer_idx in self.hot_storage:
                    del self.hot_storage[layer_idx]
                    self.hot_layers.discard(layer_idx)
            logger.debug(f"Evicted {len(evicted)} entries from hot cache (no cold cache)")
            return

        # Evict to cold cache
        evicted = self.hot_lru.evict_lru(count=self.config.eviction_batch_size)

        for layer_idx, entry in evicted:
            if layer_idx not in self.hot_storage:
                continue

            # Move to cold cache
            hot_data = self.hot_storage[layer_idx]
            keys = hot_data['keys'].to(self.config.device_cold)
            values = hot_data['values'].to(self.config.device_cold)

            self.cold_storage[layer_idx] = {
                'keys': keys,
                'values': values,
                'device': self.config.device_cold
            }
            self.cold_layers.add(layer_idx)
            self.cold_lru.add(layer_idx, entry.size_bytes)

            # Remove from hot
            del self.hot_storage[layer_idx]
            self.hot_layers.discard(layer_idx)

            self.demotions += 1

        logger.debug(f"Demoted {len(evicted)} entries from hot to cold cache")

    def _promote_to_hot(self, layer_idx: int, new_keys: torch.Tensor, new_values: torch.Tensor):
        """Promote an entry from cold cache to hot cache."""
        if not self.config.enable_cold_cache or layer_idx not in self.cold_storage:
            return

        # Get existing cold data
        cold_data = self.cold_storage[layer_idx]
        existing_keys = cold_data['keys']
        existing_values = cold_data['values']

        # Concatenate with new states
        updated_keys = torch.cat([existing_keys, new_keys], dim=2)
        updated_values = torch.cat([existing_values, new_values], dim=2)

        # Move to hot device
        updated_keys = updated_keys.to(self.config.device_hot)
        updated_values = updated_values.to(self.config.device_hot)

        # Calculate size
        size = self._calculate_tensor_size(updated_keys) + self._calculate_tensor_size(updated_values)

        # Remove from cold
        del self.cold_storage[layer_idx]
        self.cold_layers.discard(layer_idx)
        self.cold_lru.remove(layer_idx)

        # Add to hot (will evict if necessary)
        if self.hot_lru.is_full():
            self._evict_from_hot()

        self.hot_storage[layer_idx] = {
            'keys': updated_keys,
            'values': updated_values,
            'device': self.config.device_hot
        }
        self.hot_layers.add(layer_idx)
        self.hot_lru.add(layer_idx, size)

        self.promotions += 1
        logger.debug(f"Promoted layer {layer_idx} from cold to hot cache")

    def clear(self):
        """Clear all cache entries."""
        self.hot_storage.clear()
        self.hot_layers.clear()
        self.hot_lru.clear()

        if self.cold_storage:
            self.cold_storage.clear()
            self.cold_layers.clear()
            self.cold_lru.clear()

        logger.info("Cleared all cache entries")

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hot_stats = self.hot_lru.get_statistics()
        cold_stats = self.cold_lru.get_statistics() if self.cold_lru else None

        stats = {
            'hot_cache': {
                'entries': len(self.hot_layers),
                'size_bytes': hot_stats.hot_cache_size_bytes,
                'hit_rate': hot_stats.hit_rate,
            },
            'promotions': self.promotions,
            'demotions': self.demotions,
        }

        if cold_stats:
            stats['cold_cache'] = {
                'entries': len(self.cold_layers),
                'size_bytes': cold_stats.hot_cache_size_bytes,
                'hit_rate': cold_stats.hit_rate,
            }

        return stats

    def format_statistics(self) -> str:
        """Format cache statistics for display."""
        stats = self.get_statistics()

        lines = [
            "",
            "━" * 60,
            "🔥 Hybrid KV-Cache Statistics",
            "━" * 60,
            f"Hot Cache (GPU): {stats['hot_cache']['entries']} layers, "
            f"{stats['hot_cache']['size_bytes'] / (1024**2):.2f} MB",
            f"  Hit Rate: {stats['hot_cache']['hit_rate'] * 100:.1f}%",
        ]

        if 'cold_cache' in stats:
            lines.extend([
                f"Cold Cache (CPU): {stats['cold_cache']['entries']} layers, "
                f"{stats['cold_cache']['size_bytes'] / (1024**2):.2f} MB",
                f"  Hit Rate: {stats['cold_cache']['hit_rate'] * 100:.1f}%",
            ])

        lines.extend([
            "",
            f"Total Promotions (Cold→Hot): {stats['promotions']}",
            f"Total Demotions (Hot→Cold): {stats['demotions']}",
            "━" * 60,
            ""
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of cache."""
        return (f"HybridKVCache(hot_layers={len(self.hot_layers)}, "
                f"cold_layers={len(self.cold_layers) if self.cold_layers else 0}, "
                f"promotions={self.promotions}, demotions={self.demotions})")
