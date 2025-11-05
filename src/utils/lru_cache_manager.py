# src/utils/lru_cache_manager.py
"""
LRU Cache Manager

Implements Least-Recently-Used eviction policy for managing cache entries.
Used by the Hybrid KV-Cache to determine which entries to evict to slower storage.
"""

from typing import Dict, Any, Optional, List, Tuple, OrderedDict as OrderedDictType
from collections import OrderedDict
from dataclasses import dataclass, field
import time

from .logger import get_logger_conf

logger = get_logger_conf(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""
    key: Any
    size_bytes: int
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    creation_time: float = field(default_factory=time.time)

    def touch(self):
        """Update last access time and increment access count."""
        self.last_access_time = time.time()
        self.access_count += 1


@dataclass
class CacheStatistics:
    """Statistics about cache performance."""
    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    hot_cache_size_bytes: int = 0
    cold_cache_size_bytes: int = 0
    hot_cache_entries: int = 0
    cold_cache_entries: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0

    def format_stats(self) -> str:
        """Format statistics for display."""
        lines = [
            "",
            "━" * 50,
            "📊 Cache Statistics",
            "━" * 50,
            f"Hit Rate: {self.hit_rate * 100:.2f}%",
            f"Total Hits: {self.total_hits}",
            f"Total Misses: {self.total_misses}",
            f"Total Evictions: {self.total_evictions}",
            "",
            f"Hot Cache (GPU/Fast): {self.hot_cache_entries} entries ({self._format_bytes(self.hot_cache_size_bytes)})",
            f"Cold Cache (CPU/Slow): {self.cold_cache_entries} entries ({self._format_bytes(self.cold_cache_size_bytes)})",
            "━" * 50,
            ""
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_bytes(bytes_val: int) -> str:
        """Format bytes to human-readable string."""
        if bytes_val < 1024:
            return f"{bytes_val} B"
        elif bytes_val < 1024**2:
            return f"{bytes_val / 1024:.2f} KB"
        elif bytes_val < 1024**3:
            return f"{bytes_val / (1024**2):.2f} MB"
        else:
            return f"{bytes_val / (1024**3):.2f} GB"


class LRUCacheManager:
    """
    Manages cache entries using Least-Recently-Used eviction policy.

    This manager tracks which cache entries are hot (frequently accessed)
    and which are cold (rarely accessed), enabling intelligent eviction
    decisions for the Hybrid KV-Cache.
    """

    def __init__(self, max_hot_size_bytes: Optional[int] = None, max_entries: Optional[int] = None):
        """
        Initialize the LRU cache manager.

        Args:
            max_hot_size_bytes: Maximum size in bytes for hot cache (None = unlimited)
            max_entries: Maximum number of entries (None = unlimited)
        """
        self.max_hot_size_bytes = max_hot_size_bytes
        self.max_entries = max_entries

        # OrderedDict maintains insertion order and allows efficient LRU
        self._cache: OrderedDictType[Any, CacheEntry] = OrderedDict()

        # Statistics
        self.stats = CacheStatistics()

        logger.debug(f"LRUCacheManager initialized (max_hot_size={max_hot_size_bytes}, max_entries={max_entries})")

    def access(self, key: Any) -> bool:
        """
        Record an access to a cache entry.

        Args:
            key: The cache key being accessed

        Returns:
            True if key exists (cache hit), False if not (cache miss)
        """
        if key in self._cache:
            # Cache hit - move to end (most recently used)
            entry = self._cache.pop(key)
            entry.touch()
            self._cache[key] = entry
            self.stats.total_hits += 1
            return True
        else:
            # Cache miss
            self.stats.total_misses += 1
            return False

    def add(self, key: Any, size_bytes: int):
        """
        Add a new entry to the cache.

        Args:
            key: The cache key
            size_bytes: Size of the cache entry in bytes
        """
        # If key already exists, update it
        if key in self._cache:
            self._cache.pop(key)

        # Create new entry
        entry = CacheEntry(key=key, size_bytes=size_bytes)
        self._cache[key] = entry

        logger.debug(f"Added cache entry: key={key}, size={size_bytes} bytes")

    def remove(self, key: Any) -> Optional[CacheEntry]:
        """
        Remove an entry from the cache.

        Args:
            key: The cache key to remove

        Returns:
            The removed CacheEntry if it existed, None otherwise
        """
        return self._cache.pop(key, None)

    def get_current_size(self) -> int:
        """Get current total cache size in bytes."""
        return sum(entry.size_bytes for entry in self._cache.values())

    def get_entry_count(self) -> int:
        """Get current number of cache entries."""
        return len(self._cache)

    def is_full(self) -> bool:
        """Check if cache is full based on size or entry limits."""
        if self.max_hot_size_bytes and self.get_current_size() >= self.max_hot_size_bytes:
            return True
        if self.max_entries and self.get_entry_count() >= self.max_entries:
            return True
        return False

    def get_lru_entries(self, count: int = 1) -> List[Tuple[Any, CacheEntry]]:
        """
        Get the least recently used entries.

        Args:
            count: Number of LRU entries to return

        Returns:
            List of (key, entry) tuples for the least recently used entries
        """
        entries = list(self._cache.items())
        return entries[:count]

    def evict_lru(self, count: int = 1) -> List[Tuple[Any, CacheEntry]]:
        """
        Evict the least recently used entries.

        Args:
            count: Number of entries to evict

        Returns:
            List of (key, entry) tuples that were evicted
        """
        evicted = []
        for _ in range(min(count, len(self._cache))):
            if not self._cache:
                break
            key, entry = self._cache.popitem(last=False)  # Remove from front (oldest)
            evicted.append((key, entry))
            self.stats.total_evictions += 1
            logger.debug(f"Evicted LRU entry: key={key}, size={entry.size_bytes} bytes")

        return evicted

    def evict_until_size(self, target_size: int) -> List[Tuple[Any, CacheEntry]]:
        """
        Evict entries until cache size is below target.

        Args:
            target_size: Target size in bytes

        Returns:
            List of (key, entry) tuples that were evicted
        """
        evicted = []
        current_size = self.get_current_size()

        while current_size > target_size and self._cache:
            key, entry = self._cache.popitem(last=False)
            evicted.append((key, entry))
            current_size -= entry.size_bytes
            self.stats.total_evictions += 1
            logger.debug(f"Evicted entry to meet size target: key={key}, size={entry.size_bytes} bytes")

        return evicted

    def get_hot_keys(self, threshold_percentile: float = 0.75) -> List[Any]:
        """
        Get keys that are considered "hot" based on access patterns.

        Args:
            threshold_percentile: Percentile threshold for hotness (0.0-1.0)

        Returns:
            List of hot cache keys
        """
        if not self._cache:
            return []

        # Sort by access count (descending)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count,
            reverse=True
        )

        # Get top percentile
        threshold_idx = int(len(sorted_entries) * threshold_percentile)
        hot_entries = sorted_entries[:threshold_idx] if threshold_idx > 0 else sorted_entries[:1]

        return [key for key, _ in hot_entries]

    def get_cold_keys(self, threshold_percentile: float = 0.25) -> List[Any]:
        """
        Get keys that are considered "cold" based on access patterns.

        Args:
            threshold_percentile: Percentile threshold for coldness (0.0-1.0)

        Returns:
            List of cold cache keys
        """
        if not self._cache:
            return []

        # Sort by access count (ascending)
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].access_count
        )

        # Get bottom percentile
        threshold_idx = int(len(sorted_entries) * threshold_percentile)
        cold_entries = sorted_entries[:threshold_idx] if threshold_idx > 0 else sorted_entries[:1]

        return [key for key, _ in cold_entries]

    def clear(self):
        """Clear all cache entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")

    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        self.stats.hot_cache_entries = len(self._cache)
        self.stats.hot_cache_size_bytes = self.get_current_size()
        return self.stats

    def reset_statistics(self):
        """Reset cache statistics."""
        self.stats = CacheStatistics()
        logger.debug("Cache statistics reset")

    def __len__(self) -> int:
        """Return number of cache entries."""
        return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        """Check if key exists in cache."""
        return key in self._cache

    def __repr__(self) -> str:
        """String representation of cache manager."""
        return (f"LRUCacheManager(entries={len(self._cache)}, "
                f"size={self.get_current_size()} bytes, "
                f"hit_rate={self.stats.hit_rate * 100:.1f}%)")
