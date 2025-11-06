# src/models/speaker_embeddings/manager.py
"""
Speaker Embedding Manager

Central orchestrator for speaker embedding generation, caching, and persistence.
This manager provides a unified interface for working with multiple speakers and
embedding generation methods.

Key features:
- Automatic method selection (random vs dataset vs audio)
- Speaker → embedding caching
- Persistent storage across sessions
- Thread-safe operations
- Integration with AudioConfig
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from threading import Lock
from dataclasses import asdict

from .base import EmbeddingConfig, EmbeddingMethod
from .random_generator import RandomEmbeddingGenerator
from .dataset_sampler import DatasetEmbeddingSampler
from ...utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class SpeakerEmbeddingManager:
    """
    Centralized manager for speaker embeddings.

    This manager handles the lifecycle of speaker embeddings:
    - Generation (random, dataset, or audio-based)
    - Caching in memory
    - Persistence to disk
    - Thread-safe access

    Example:
        manager = SpeakerEmbeddingManager(
            method="auto",
            cache_dir=Path("cache/speakers")
        )

        # Get or create embedding for speaker
        embedding = manager.get_embedding("Host")

        # Get embeddings for multiple speakers
        embeddings = manager.get_embeddings_batch(["Host", "Guest", "Narrator"])

        # Save mappings for later use
        manager.save_mappings()
    """

    def __init__(
        self,
        method: str = "auto",
        config: Optional[EmbeddingConfig] = None,
        cache_dir: Optional[Path] = None,
        enable_cache: bool = True,
        enable_persistence: bool = True
    ):
        """
        Initialize speaker embedding manager.

        Args:
            method: Generation method - "auto", "random", "dataset", "audio"
            config: Optional EmbeddingConfig to override defaults
            cache_dir: Directory for persistent cache (default: cache/speakers)
            enable_cache: Enable in-memory caching
            enable_persistence: Enable disk persistence
        """
        self.method = method
        self.config = config or EmbeddingConfig()
        self.enable_cache = enable_cache
        self.enable_persistence = enable_persistence

        # Set cache directory
        if cache_dir is None:
            cache_dir = Path("cache/speakers")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache: speaker_name -> embedding
        self._embedding_cache: Dict[str, np.ndarray] = {}

        # Speaker metadata: speaker_name -> config dict
        self._speaker_metadata: Dict[str, Dict[str, Any]] = {}

        # Thread safety
        self._lock = Lock()

        # Generators (lazy-initialized)
        self._random_generator: Optional[RandomEmbeddingGenerator] = None
        self._dataset_sampler: Optional[DatasetEmbeddingSampler] = None

        # Load persisted mappings if they exist
        if self.enable_persistence:
            self._load_mappings()

        logger.info(
            f"SpeakerEmbeddingManager initialized: method={method}, "
            f"cache_dir={cache_dir}, cache={enable_cache}, persistence={enable_persistence}"
        )

    def get_embedding(
        self,
        speaker_name: str,
        device: str = "cpu",
        method_override: Optional[str] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Get or generate embedding for a speaker.

        If the speaker already has a cached embedding, returns it.
        Otherwise, generates a new embedding based on the configured method.

        Args:
            speaker_name: Name of the speaker
            device: Target device for tensor ("cpu", "cuda", "cuda:0", etc.)
            method_override: Override the default method for this speaker
            **kwargs: Additional parameters for generation (seed, dataset_index, etc.)

        Returns:
            torch.Tensor: Speaker embedding of shape (embedding_dim,)
        """
        with self._lock:
            # Check cache first
            if self.enable_cache and speaker_name in self._embedding_cache:
                logger.debug(f"Using cached embedding for speaker '{speaker_name}'")
                embedding_np = self._embedding_cache[speaker_name]
                return torch.from_numpy(embedding_np).to(device)

            # Generate new embedding
            method = method_override or self.method
            logger.info(f"Generating new embedding for speaker '{speaker_name}' using method '{method}'")

            embedding_np = self._generate_embedding(speaker_name, method, **kwargs)

            # Cache the embedding
            if self.enable_cache:
                self._embedding_cache[speaker_name] = embedding_np
                self._speaker_metadata[speaker_name] = {
                    "method": method,
                    "params": kwargs,
                    "embedding_dim": embedding_np.shape[0]
                }

            # Persist if enabled
            if self.enable_persistence:
                self._save_speaker_embedding(speaker_name, embedding_np)

            return torch.from_numpy(embedding_np).to(device)

    def get_embeddings_batch(
        self,
        speaker_names: List[str],
        device: str = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Get embeddings for multiple speakers.

        Args:
            speaker_names: List of speaker names
            device: Target device for tensors

        Returns:
            Dict mapping speaker_name -> embedding tensor
        """
        embeddings = {}
        for speaker_name in speaker_names:
            embeddings[speaker_name] = self.get_embedding(speaker_name, device=device)

        logger.debug(f"Retrieved {len(embeddings)} speaker embeddings")
        return embeddings

    def _generate_embedding(
        self,
        speaker_name: str,
        method: str,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a new embedding using specified method.

        Args:
            speaker_name: Name of the speaker
            method: Generation method
            **kwargs: Additional parameters

        Returns:
            np.ndarray: Generated embedding
        """
        # Auto method selection
        if method == "auto":
            # Default to random for now
            # Could be enhanced with heuristics (e.g., dataset for specific names)
            method = "random"

        if method in ["random", "gaussian", "uniform"]:
            return self._generate_random(speaker_name, **kwargs)

        elif method == "dataset":
            return self._generate_dataset(speaker_name, **kwargs)

        elif method == "audio":
            raise NotImplementedError("Audio-based extraction will be implemented in Phase 3")

        else:
            raise ValueError(f"Unknown embedding generation method: {method}")

    def _generate_random(self, speaker_name: str, **kwargs) -> np.ndarray:
        """Generate random embedding."""
        if self._random_generator is None:
            config = EmbeddingConfig(**asdict(self.config))
            config.method = EmbeddingMethod.RANDOM
            self._random_generator = RandomEmbeddingGenerator(config)

        return self._random_generator.generate(speaker_name=speaker_name, **kwargs)

    def _generate_dataset(self, speaker_name: str, **kwargs) -> np.ndarray:
        """Generate embedding from dataset."""
        if self._dataset_sampler is None:
            config = EmbeddingConfig(**asdict(self.config))
            config.method = EmbeddingMethod.DATASET
            self._dataset_sampler = DatasetEmbeddingSampler(config)

        return self._dataset_sampler.generate(speaker_name=speaker_name, **kwargs)

    def has_embedding(self, speaker_name: str) -> bool:
        """
        Check if speaker already has an embedding.

        Args:
            speaker_name: Name of the speaker

        Returns:
            bool: True if embedding exists in cache
        """
        with self._lock:
            return speaker_name in self._embedding_cache

    def regenerate_embedding(
        self,
        speaker_name: str,
        device: str = "cpu",
        **kwargs
    ) -> torch.Tensor:
        """
        Force regeneration of embedding for a speaker.

        Useful if the user wants to try a different voice.

        Args:
            speaker_name: Name of the speaker
            device: Target device
            **kwargs: Generation parameters

        Returns:
            torch.Tensor: New embedding
        """
        with self._lock:
            # Remove from cache
            if speaker_name in self._embedding_cache:
                del self._embedding_cache[speaker_name]
            if speaker_name in self._speaker_metadata:
                del self._speaker_metadata[speaker_name]

        # Generate new embedding
        return self.get_embedding(speaker_name, device=device, **kwargs)

    def set_embedding(
        self,
        speaker_name: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Manually set an embedding for a speaker.

        Useful for custom embeddings or audio-extracted embeddings.

        Args:
            speaker_name: Name of the speaker
            embedding: Embedding vector
            metadata: Optional metadata about the embedding
        """
        with self._lock:
            self._embedding_cache[speaker_name] = embedding
            self._speaker_metadata[speaker_name] = metadata or {}

            if self.enable_persistence:
                self._save_speaker_embedding(speaker_name, embedding)

        logger.info(f"Set custom embedding for speaker '{speaker_name}'")

    def remove_speaker(self, speaker_name: str):
        """
        Remove a speaker from cache and disk.

        Args:
            speaker_name: Name of the speaker to remove
        """
        with self._lock:
            if speaker_name in self._embedding_cache:
                del self._embedding_cache[speaker_name]
            if speaker_name in self._speaker_metadata:
                del self._speaker_metadata[speaker_name]

            # Remove from disk
            embedding_path = self.cache_dir / f"{speaker_name}.npy"
            if embedding_path.exists():
                embedding_path.unlink()

        logger.info(f"Removed speaker '{speaker_name}'")

    def clear_cache(self):
        """Clear all cached embeddings from memory."""
        with self._lock:
            self._embedding_cache.clear()
            self._speaker_metadata.clear()

        logger.info("Cleared speaker embedding cache")

    def list_speakers(self) -> List[str]:
        """
        Get list of all cached speakers.

        Returns:
            List of speaker names
        """
        with self._lock:
            return list(self._embedding_cache.keys())

    def get_speaker_info(self, speaker_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a speaker's embedding.

        Args:
            speaker_name: Name of the speaker

        Returns:
            Dict with metadata or None if speaker not found
        """
        with self._lock:
            if speaker_name in self._speaker_metadata:
                return self._speaker_metadata[speaker_name].copy()
            return None

    def _save_speaker_embedding(self, speaker_name: str, embedding: np.ndarray):
        """Save speaker embedding to disk."""
        try:
            embedding_path = self.cache_dir / f"{speaker_name}.npy"
            np.save(embedding_path, embedding)
            logger.debug(f"Saved embedding for speaker '{speaker_name}' to {embedding_path}")
        except Exception as e:
            logger.error(f"Failed to save embedding for speaker '{speaker_name}': {e}")

    def _load_speaker_embedding(self, speaker_name: str) -> Optional[np.ndarray]:
        """Load speaker embedding from disk."""
        try:
            embedding_path = self.cache_dir / f"{speaker_name}.npy"
            if embedding_path.exists():
                embedding = np.load(embedding_path)
                logger.debug(f"Loaded embedding for speaker '{speaker_name}' from {embedding_path}")
                return embedding
        except Exception as e:
            logger.error(f"Failed to load embedding for speaker '{speaker_name}': {e}")
        return None

    def save_mappings(self):
        """Save speaker metadata to disk."""
        if not self.enable_persistence:
            return

        try:
            mappings_path = self.cache_dir / "speaker_mappings.json"
            with open(mappings_path, 'w') as f:
                json.dump(self._speaker_metadata, f, indent=2)
            logger.info(f"Saved speaker mappings to {mappings_path}")
        except Exception as e:
            logger.error(f"Failed to save speaker mappings: {e}")

    def _load_mappings(self):
        """Load speaker metadata from disk."""
        try:
            mappings_path = self.cache_dir / "speaker_mappings.json"
            if mappings_path.exists():
                with open(mappings_path, 'r') as f:
                    self._speaker_metadata = json.load(f)

                # Load embeddings for each speaker
                for speaker_name in self._speaker_metadata.keys():
                    embedding = self._load_speaker_embedding(speaker_name)
                    if embedding is not None:
                        self._embedding_cache[speaker_name] = embedding

                logger.info(
                    f"Loaded {len(self._embedding_cache)} speaker embeddings from cache"
                )
        except Exception as e:
            logger.error(f"Failed to load speaker mappings: {e}")

    def export_embeddings(self, output_path: Path) -> Dict[str, Any]:
        """
        Export all embeddings and metadata to a single file.

        Args:
            output_path: Path to save export file

        Returns:
            Dict with export statistics
        """
        export_data = {
            "method": self.method,
            "config": asdict(self.config),
            "speakers": {}
        }

        with self._lock:
            for speaker_name, embedding in self._embedding_cache.items():
                export_data["speakers"][speaker_name] = {
                    "embedding": embedding.tolist(),
                    "metadata": self._speaker_metadata.get(speaker_name, {})
                }

        try:
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)

            stats = {
                "speakers_exported": len(export_data["speakers"]),
                "output_path": str(output_path),
                "success": True
            }
            logger.info(f"Exported {stats['speakers_exported']} speaker embeddings to {output_path}")
            return stats

        except Exception as e:
            logger.error(f"Failed to export embeddings: {e}")
            return {"success": False, "error": str(e)}

    def import_embeddings(self, input_path: Path) -> Dict[str, Any]:
        """
        Import embeddings from an export file.

        Args:
            input_path: Path to import file

        Returns:
            Dict with import statistics
        """
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)

            imported_count = 0
            with self._lock:
                for speaker_name, data in import_data.get("speakers", {}).items():
                    embedding = np.array(data["embedding"], dtype=np.float32)
                    metadata = data.get("metadata", {})

                    self._embedding_cache[speaker_name] = embedding
                    self._speaker_metadata[speaker_name] = metadata

                    if self.enable_persistence:
                        self._save_speaker_embedding(speaker_name, embedding)

                    imported_count += 1

            if self.enable_persistence:
                self.save_mappings()

            stats = {
                "speakers_imported": imported_count,
                "success": True
            }
            logger.info(f"Imported {imported_count} speaker embeddings from {input_path}")
            return stats

        except Exception as e:
            logger.error(f"Failed to import embeddings: {e}")
            return {"success": False, "error": str(e)}

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the manager state.

        Returns:
            Dict with statistics
        """
        with self._lock:
            stats = {
                "method": self.method,
                "cached_speakers": len(self._embedding_cache),
                "speakers": list(self._embedding_cache.keys()),
                "cache_enabled": self.enable_cache,
                "persistence_enabled": self.enable_persistence,
                "cache_dir": str(self.cache_dir),
                "generators_initialized": {
                    "random": self._random_generator is not None,
                    "dataset": self._dataset_sampler is not None
                }
            }

        return stats

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"SpeakerEmbeddingManager("
            f"method={self.method}, "
            f"cached_speakers={len(self._embedding_cache)}, "
            f"cache={self.enable_cache}, "
            f"persistence={self.enable_persistence})"
        )
