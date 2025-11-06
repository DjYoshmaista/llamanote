# src/models/speaker_embeddings/random_generator.py
"""
Random speaker embedding generation.

This module generates synthetic speaker embeddings from statistical distributions.
While these embeddings don't correspond to real speakers, they can be used to create
distinct, consistent voices for text-to-speech generation.

Key features:
- Gaussian and uniform distributions
- Deterministic with seeding for reproducibility
- Gender bias heuristics (optional, experimental)
- Fast generation with minimal dependencies

Pros:
- No external dependencies (audio files, datasets)
- Instant generation (microseconds)
- Fully deterministic with seed
- Infinite variety of voices

Cons:
- May produce unnatural-sounding voices
- No control over voice characteristics
- Less diverse than dataset sampling
- Occasional "weird" voices requiring regeneration
"""

import numpy as np
from typing import Optional
from .base import SpeakerEmbeddingGenerator, EmbeddingConfig, EmbeddingMethod


class RandomEmbeddingGenerator(SpeakerEmbeddingGenerator):
    """
    Generate random speaker embeddings from statistical distributions.

    This generator creates synthetic embeddings by sampling from Gaussian or
    uniform distributions. While simple, this approach can produce diverse and
    consistent speaker identities.

    Example:
        # Basic usage with default Gaussian distribution
        generator = RandomEmbeddingGenerator()
        embedding = generator.generate(speaker_name="Host")

        # Uniform distribution with custom range
        config = EmbeddingConfig(
            distribution="uniform",
            min_val=-0.5,
            max_val=0.5,
            seed=42
        )
        generator = RandomEmbeddingGenerator(config)
        embedding = generator.generate()

        # Generate multiple embeddings
        embeddings = generator.generate_batch(["Host", "Guest", "Narrator"])
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize random embedding generator.

        Args:
            config: EmbeddingConfig with random generation parameters
        """
        super().__init__(config)

        # Validate that this is a random generation config
        if self.config.method not in [
            EmbeddingMethod.RANDOM,
            EmbeddingMethod.GAUSSIAN,
            EmbeddingMethod.UNIFORM
        ]:
            # Auto-correct to match distribution
            if self.config.distribution == "gaussian":
                self.config.method = EmbeddingMethod.GAUSSIAN
            else:
                self.config.method = EmbeddingMethod.UNIFORM

        # Cache for speaker-specific seeds
        self._speaker_seeds: dict[str, int] = {}

    def generate(
        self,
        speaker_name: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a random speaker embedding.

        If speaker_name is provided and a seed is set, the same speaker will
        always get the same embedding (deterministic).

        Args:
            speaker_name: Optional speaker identifier for deterministic generation
            **kwargs: Override config parameters:
                - distribution: "gaussian" or "uniform"
                - mean, std: For Gaussian
                - min_val, max_val: For uniform
                - seed: Random seed
                - normalize: Whether to L2-normalize

        Returns:
            np.ndarray: Speaker embedding of shape (embedding_dim,)
        """
        # Extract parameters (kwargs override config)
        distribution = kwargs.get("distribution", self.config.distribution)
        embedding_dim = kwargs.get("embedding_dim", self.config.embedding_dim)
        normalize = kwargs.get("normalize", self.config.normalize)
        seed = kwargs.get("seed", self.config.seed)

        # Determine random seed
        rng_seed = self._get_seed_for_speaker(speaker_name, seed)

        # Create RNG with seed
        rng = np.random.RandomState(rng_seed) if rng_seed is not None else np.random

        # Generate embedding based on distribution
        if distribution == "gaussian":
            mean = kwargs.get("mean", self.config.mean)
            std = kwargs.get("std", self.config.std)
            embedding = self._generate_gaussian(rng, embedding_dim, mean, std)

        elif distribution == "uniform":
            min_val = kwargs.get("min_val", self.config.min_val)
            max_val = kwargs.get("max_val", self.config.max_val)
            embedding = self._generate_uniform(rng, embedding_dim, min_val, max_val)

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Normalize if requested
        if normalize:
            embedding = self.normalize_embedding(embedding)

        return embedding.astype(np.float32)

    def _generate_gaussian(
        self,
        rng: np.random.RandomState,
        dim: int,
        mean: float,
        std: float
    ) -> np.ndarray:
        """
        Generate embedding from Gaussian distribution.

        Args:
            rng: Random number generator
            dim: Embedding dimensionality
            mean: Mean of distribution
            std: Standard deviation

        Returns:
            np.ndarray: Unnormalized embedding
        """
        return rng.normal(loc=mean, scale=std, size=dim)

    def _generate_uniform(
        self,
        rng: np.random.RandomState,
        dim: int,
        min_val: float,
        max_val: float
    ) -> np.ndarray:
        """
        Generate embedding from uniform distribution.

        Args:
            rng: Random number generator
            dim: Embedding dimensionality
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            np.ndarray: Unnormalized embedding
        """
        return rng.uniform(low=min_val, high=max_val, size=dim)

    def _get_seed_for_speaker(
        self,
        speaker_name: Optional[str],
        base_seed: Optional[int]
    ) -> Optional[int]:
        """
        Get deterministic seed for a specific speaker.

        If speaker_name is provided, generates a consistent seed based on the name.
        This ensures the same speaker always gets the same embedding.

        Args:
            speaker_name: Speaker identifier
            base_seed: Base seed from config

        Returns:
            int or None: Seed for RNG, or None for non-deterministic
        """
        if base_seed is None and speaker_name is None:
            return None

        if speaker_name is not None:
            # Check cache first
            if speaker_name in self._speaker_seeds:
                return self._speaker_seeds[speaker_name]

            # Generate seed from speaker name hash
            name_hash = hash(speaker_name) & 0x7FFFFFFF  # Ensure positive 32-bit int

            if base_seed is not None:
                # Combine base seed with name hash
                speaker_seed = (base_seed + name_hash) & 0x7FFFFFFF
            else:
                speaker_seed = name_hash

            # Cache for future use
            self._speaker_seeds[speaker_name] = speaker_seed
            return speaker_seed

        return base_seed

    def generate_with_gender_bias(
        self,
        gender: str,
        speaker_name: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Generate embedding with gender bias (experimental).

        This is a heuristic approach that applies statistical biases to the
        embedding distribution to (potentially) generate more masculine or
        feminine-sounding voices.

        WARNING: This is experimental and based on rough heuristics. There's no
        guarantee it will produce the desired effect, as speaker embeddings don't
        have explicit gender dimensions.

        Args:
            gender: "male", "female", or "neutral"
            speaker_name: Optional speaker identifier
            **kwargs: Additional parameters

        Returns:
            np.ndarray: Speaker embedding with gender bias applied
        """
        # Generate base embedding
        embedding = self.generate(speaker_name=speaker_name, **kwargs)

        if gender == "male":
            # Male voices: Slightly amplify lower dimensions (arbitrary heuristic)
            # This assumes lower dimensions might correlate with pitch
            embedding[:embedding.shape[0]//4] *= 1.2

        elif gender == "female":
            # Female voices: Slightly amplify higher dimensions
            embedding[3*embedding.shape[0]//4:] *= 1.2

        # Re-normalize
        if kwargs.get("normalize", self.config.normalize):
            embedding = self.normalize_embedding(embedding)

        return embedding

    def interpolate(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
        alpha: float = 0.5,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Interpolate between two embeddings.

        Creates a new speaker embedding that's a blend of two existing ones.
        Can be used to create voice variations or transitions.

        Args:
            embedding1: First embedding
            embedding2: Second embedding
            alpha: Interpolation factor (0.0 = embedding1, 1.0 = embedding2)
            normalize: Whether to normalize result

        Returns:
            np.ndarray: Interpolated embedding
        """
        if embedding1.shape != embedding2.shape:
            raise ValueError(
                f"Embedding shapes must match: {embedding1.shape} vs {embedding2.shape}"
            )

        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        # Linear interpolation
        interpolated = (1 - alpha) * embedding1 + alpha * embedding2

        if normalize:
            interpolated = self.normalize_embedding(interpolated)

        return interpolated

    def add_noise(
        self,
        embedding: np.ndarray,
        noise_scale: float = 0.05,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Add small random noise to an embedding.

        Useful for creating slight variations of a speaker's voice.

        Args:
            embedding: Original embedding
            noise_scale: Standard deviation of Gaussian noise
            normalize: Whether to normalize result

        Returns:
            np.ndarray: Noisy embedding
        """
        noise = np.random.randn(*embedding.shape) * noise_scale
        noisy = embedding + noise

        if normalize:
            noisy = self.normalize_embedding(noisy)

        return noisy.astype(np.float32)

    def get_info(self) -> dict:
        """Get generator information."""
        info = super().get_info()
        info.update({
            "distribution": self.config.distribution,
            "mean": self.config.mean,
            "std": self.config.std,
            "min_val": self.config.min_val,
            "max_val": self.config.max_val,
            "cached_speakers": len(self._speaker_seeds),
        })
        return info
