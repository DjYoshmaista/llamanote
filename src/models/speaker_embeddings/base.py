# src/models/speaker_embeddings/base.py
"""
Base classes and interfaces for speaker embedding generation.

This module defines the abstract base class that all embedding generators must implement,
as well as common data structures and enumerations used throughout the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np
import torch


class EmbeddingMethod(Enum):
    """Methods for generating speaker embeddings."""
    RANDOM = "random"           # Generate random embeddings from distributions
    GAUSSIAN = "gaussian"       # Gaussian distribution (alias for random)
    UNIFORM = "uniform"         # Uniform distribution
    DATASET = "dataset"         # Sample from pre-computed dataset
    AUDIO = "audio"            # Extract from audio files (future)
    INTERPOLATE = "interpolate" # Interpolate between existing embeddings (future)


@dataclass
class EmbeddingConfig:
    """
    Configuration for speaker embedding generation.

    This dataclass encapsulates all parameters needed to generate a speaker embedding,
    regardless of the generation method.
    """
    # Common parameters
    method: EmbeddingMethod = EmbeddingMethod.RANDOM
    embedding_dim: int = 512  # Default for SpeechT5 x-vectors
    normalize: bool = True     # L2 normalization (highly recommended)
    seed: Optional[int] = None # For reproducibility

    # Random generation parameters
    distribution: str = "gaussian"  # "gaussian" or "uniform"
    mean: float = 0.0
    std: float = 1.0
    min_val: float = -1.0
    max_val: float = 1.0

    # Dataset sampling parameters
    dataset_name: str = "Matthijs/cmu-arctic-xvectors"
    dataset_index: Optional[int] = None
    gender_filter: Optional[str] = None  # "male", "female", or None

    # Audio extraction parameters (future)
    audio_path: Optional[str] = None
    audio_duration_min: float = 3.0  # Minimum audio duration in seconds
    model_name: str = "speechbrain/spkrec-xvect-voxceleb"

    # Cache settings
    enable_cache: bool = True
    cache_dir: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration."""
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")

        if self.method == EmbeddingMethod.AUDIO and self.audio_path is None:
            raise ValueError("audio_path required when method=AUDIO")

        if self.distribution not in ["gaussian", "uniform"]:
            raise ValueError(f"distribution must be 'gaussian' or 'uniform', got {self.distribution}")


class SpeakerEmbeddingGenerator(ABC):
    """
    Abstract base class for all speaker embedding generators.

    All concrete generators (RandomEmbeddingGenerator, AudioEmbeddingExtractor,
    DatasetEmbeddingSampler) must implement this interface.

    Design principles:
    - Thread-safe: Methods should be safe to call from multiple threads
    - Deterministic: Given the same config and seed, produce the same embedding
    - Normalized: Output embeddings should be L2-normalized by default
    - Device-agnostic: Support both CPU and GPU tensors
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize the generator with configuration.

        Args:
            config: EmbeddingConfig instance. If None, uses default configuration.
        """
        self.config = config or EmbeddingConfig()

    @abstractmethod
    def generate(
        self,
        speaker_name: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a speaker embedding.

        This is the main method that concrete classes must implement. It should
        generate a speaker embedding vector according to the generator's strategy
        and the provided configuration.

        Args:
            speaker_name: Optional identifier for the speaker (for logging/caching)
            **kwargs: Additional method-specific parameters that override config

        Returns:
            np.ndarray: Speaker embedding vector of shape (embedding_dim,)
                       Typically L2-normalized to unit length.

        Raises:
            ValueError: If required parameters are missing or invalid
            RuntimeError: If generation fails due to external factors
        """
        pass

    def generate_torch(
        self,
        speaker_name: Optional[str] = None,
        device: str = "cpu",
        **kwargs
    ) -> torch.Tensor:
        """
        Generate a speaker embedding as PyTorch tensor.

        Convenience method that wraps generate() and returns a torch.Tensor.

        Args:
            speaker_name: Optional identifier for the speaker
            device: Target device ("cpu", "cuda", "cuda:0", etc.)
            **kwargs: Additional method-specific parameters

        Returns:
            torch.Tensor: Speaker embedding tensor of shape (embedding_dim,)
        """
        embedding_np = self.generate(speaker_name=speaker_name, **kwargs)
        return torch.from_numpy(embedding_np).to(device)

    def generate_batch(
        self,
        speaker_names: list[str],
        **kwargs
    ) -> np.ndarray:
        """
        Generate multiple speaker embeddings in batch.

        Default implementation calls generate() for each speaker. Subclasses
        can override this for more efficient batch generation.

        Args:
            speaker_names: List of speaker identifiers
            **kwargs: Additional method-specific parameters

        Returns:
            np.ndarray: Batch of embeddings, shape (batch_size, embedding_dim)
        """
        embeddings = [
            self.generate(speaker_name=name, **kwargs)
            for name in speaker_names
        ]
        return np.stack(embeddings, axis=0)

    def generate_batch_torch(
        self,
        speaker_names: list[str],
        device: str = "cpu",
        **kwargs
    ) -> torch.Tensor:
        """
        Generate batch of embeddings as PyTorch tensor.

        Args:
            speaker_names: List of speaker identifiers
            device: Target device
            **kwargs: Additional method-specific parameters

        Returns:
            torch.Tensor: Batch of embeddings, shape (batch_size, embedding_dim)
        """
        embeddings_np = self.generate_batch(speaker_names=speaker_names, **kwargs)
        return torch.from_numpy(embeddings_np).to(device)

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        L2-normalize an embedding to unit length.

        This is critical for speaker embeddings as most models expect normalized vectors.

        Args:
            embedding: Unnormalized embedding vector

        Returns:
            np.ndarray: Normalized embedding with L2 norm = 1.0
        """
        norm = np.linalg.norm(embedding)
        if norm < 1e-8:
            # Avoid division by zero - return zero vector
            return embedding
        return embedding / norm

    @staticmethod
    def normalize_embedding_torch(embedding: torch.Tensor) -> torch.Tensor:
        """
        L2-normalize a PyTorch tensor embedding.

        Args:
            embedding: Unnormalized embedding tensor

        Returns:
            torch.Tensor: Normalized embedding with L2 norm = 1.0
        """
        norm = torch.norm(embedding)
        if norm < 1e-8:
            return embedding
        return embedding / norm

    def validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that an embedding meets basic requirements.

        Checks:
        - Correct dimensionality
        - Finite values (no NaN or Inf)
        - Normalized (if config.normalize=True)

        Args:
            embedding: Embedding vector to validate

        Returns:
            bool: True if valid, False otherwise
        """
        # Check shape
        if embedding.shape != (self.config.embedding_dim,):
            return False

        # Check for invalid values
        if not np.all(np.isfinite(embedding)):
            return False

        # Check normalization
        if self.config.normalize:
            norm = np.linalg.norm(embedding)
            if not (0.99 <= norm <= 1.01):  # Allow small tolerance
                return False

        return True

    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this generator.

        Returns:
            dict: Dictionary containing generator type, config, and metadata
        """
        return {
            "generator_type": self.__class__.__name__,
            "method": self.config.method.value,
            "embedding_dim": self.config.embedding_dim,
            "normalize": self.config.normalize,
            "config": self.config,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"method={self.config.method.value}, "
            f"dim={self.config.embedding_dim}, "
            f"normalize={self.config.normalize})"
        )
