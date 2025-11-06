# src/models/speaker_embeddings/dataset_sampler.py
"""
Dataset-based speaker embedding sampling.

This module samples pre-computed speaker embeddings from datasets like
cmu-arctic-xvectors. These embeddings come from real speakers and generally
produce more natural-sounding voices than random generation.

Key features:
- 7000+ pre-computed speaker embeddings
- Optional gender filtering
- Consistent sampling with speaker name mapping
- Lazy dataset loading for performance

Pros:
- High-quality embeddings from real speakers
- More natural voice characteristics
- Large variety of distinct voices
- No audio processing needed

Cons:
- Requires dataset download (~50MB first time)
- Limited to dataset size (can't generate infinite voices)
- Gender filtering is heuristic-based
- Less control over specific voice characteristics
"""

import numpy as np
from typing import Optional, Dict, Any
from .base import SpeakerEmbeddingGenerator, EmbeddingConfig, EmbeddingMethod


class DatasetEmbeddingSampler(SpeakerEmbeddingGenerator):
    """
    Sample speaker embeddings from pre-computed datasets.

    This sampler loads embeddings from datasets like cmu-arctic-xvectors and
    samples them by index or randomly. It maintains a mapping from speaker names
    to dataset indices for consistency.

    Example:
        # Basic usage - random sampling
        sampler = DatasetEmbeddingSampler()
        embedding = sampler.generate(speaker_name="Host")

        # Sample specific index
        embedding = sampler.generate(speaker_name="Guest", dataset_index=1234)

        # Gender filtering (heuristic)
        config = EmbeddingConfig(method=EmbeddingMethod.DATASET, gender_filter="male")
        sampler = DatasetEmbeddingSampler(config)
        embedding = sampler.generate(speaker_name="Male_Narrator")
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize dataset sampler.

        Args:
            config: EmbeddingConfig with dataset parameters
        """
        super().__init__(config)

        # Lazy-load dataset on first use
        self._dataset = None
        self._dataset_loaded = False
        self._dataset_size = None

        # Cache speaker name -> dataset index mapping
        self._speaker_indices: Dict[str, int] = {}

        # Gender index ranges (heuristic - based on empirical observation)
        # These are rough ranges where male/female voices tend to cluster
        self._gender_ranges = {
            "male": (0, 4000),      # Indices 0-4000 tend to be male
            "female": (4000, 7000), # Indices 4000-7000 tend to be female
        }

    def _load_dataset(self):
        """
        Lazy-load the embeddings dataset.

        Loads cmu-arctic-xvectors from HuggingFace datasets on first use.
        """
        if self._dataset_loaded:
            return

        try:
            from datasets import load_dataset

            # Load dataset
            dataset_name = self.config.dataset_name
            self._dataset = load_dataset(dataset_name, split="validation")
            self._dataset_size = len(self._dataset)
            self._dataset_loaded = True

            # Log info
            from ...utils.logger import get_logger_conf
            logger = get_logger_conf(__name__)
            logger.info(
                f"Loaded speaker embedding dataset '{dataset_name}' "
                f"with {self._dataset_size} embeddings"
            )

        except ImportError:
            raise RuntimeError(
                "datasets library required for dataset sampling. "
                "Install with: pip install datasets"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset {self.config.dataset_name}: {e}")

    def generate(
        self,
        speaker_name: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Sample a speaker embedding from the dataset.

        If speaker_name is provided, consistently returns the same embedding
        for that speaker. If dataset_index is specified, samples that specific index.

        Args:
            speaker_name: Optional speaker identifier for consistent sampling
            **kwargs: Override config parameters:
                - dataset_index: Specific index to sample
                - gender_filter: "male", "female", or None
                - seed: Random seed for sampling

        Returns:
            np.ndarray: Speaker embedding of shape (embedding_dim,)
        """
        # Load dataset if needed
        self._load_dataset()

        # Get parameters
        dataset_index = kwargs.get("dataset_index", self.config.dataset_index)
        gender_filter = kwargs.get("gender_filter", self.config.gender_filter)
        seed = kwargs.get("seed", self.config.seed)
        normalize = kwargs.get("normalize", self.config.normalize)

        # Determine index to sample
        if dataset_index is not None:
            # Use explicit index
            idx = dataset_index
        elif speaker_name is not None:
            # Get consistent index for this speaker
            idx = self._get_index_for_speaker(speaker_name, gender_filter, seed)
        else:
            # Random sampling
            idx = self._sample_random_index(gender_filter, seed)

        # Validate index
        if idx < 0 or idx >= self._dataset_size:
            raise ValueError(
                f"Invalid dataset index {idx}, dataset has {self._dataset_size} embeddings"
            )

        # Sample embedding
        embedding = self._sample_embedding_at_index(idx)

        # Normalize if requested
        if normalize:
            embedding = self.normalize_embedding(embedding)

        return embedding.astype(np.float32)

    def _get_index_for_speaker(
        self,
        speaker_name: str,
        gender_filter: Optional[str],
        seed: Optional[int]
    ) -> int:
        """
        Get consistent dataset index for a specific speaker.

        Args:
            speaker_name: Speaker identifier
            gender_filter: Optional gender filter
            seed: Optional random seed

        Returns:
            int: Dataset index for this speaker
        """
        # Check cache
        cache_key = f"{speaker_name}_{gender_filter}_{seed}"
        if cache_key in self._speaker_indices:
            return self._speaker_indices[cache_key]

        # Generate index from speaker name hash
        name_hash = hash(speaker_name)

        # Apply gender filter
        if gender_filter in self._gender_ranges:
            start, end = self._gender_ranges[gender_filter]
            idx = start + (name_hash % (end - start))
        else:
            idx = name_hash % self._dataset_size

        # Apply seed if provided
        if seed is not None:
            rng = np.random.RandomState(seed)
            idx = (idx + rng.randint(0, self._dataset_size)) % self._dataset_size

        # Cache the index
        self._speaker_indices[cache_key] = idx

        return idx

    def _sample_random_index(
        self,
        gender_filter: Optional[str],
        seed: Optional[int]
    ) -> int:
        """
        Sample a random index from the dataset.

        Args:
            gender_filter: Optional gender filter
            seed: Optional random seed

        Returns:
            int: Random dataset index
        """
        rng = np.random.RandomState(seed) if seed is not None else np.random

        # Apply gender filter
        if gender_filter in self._gender_ranges:
            start, end = self._gender_ranges[gender_filter]
            idx = rng.randint(start, end)
        else:
            idx = rng.randint(0, self._dataset_size)

        return idx

    def _sample_embedding_at_index(self, idx: int) -> np.ndarray:
        """
        Extract embedding at specific dataset index.

        Args:
            idx: Dataset index

        Returns:
            np.ndarray: Embedding vector
        """
        # Get embedding from dataset
        sample = self._dataset[idx]

        # Extract x-vector (key name depends on dataset)
        if "xvector" in sample:
            embedding = np.array(sample["xvector"], dtype=np.float32)
        elif "embedding" in sample:
            embedding = np.array(sample["embedding"], dtype=np.float32)
        else:
            raise KeyError(f"Could not find embedding in dataset sample: {sample.keys()}")

        # Validate dimensionality
        if len(embedding.shape) != 1:
            # Flatten if necessary
            embedding = embedding.flatten()

        return embedding

    def get_random_embeddings(
        self,
        count: int,
        gender_filter: Optional[str] = None,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Sample multiple random embeddings.

        Args:
            count: Number of embeddings to sample
            gender_filter: Optional gender filter
            seed: Optional random seed

        Returns:
            np.ndarray: Array of embeddings, shape (count, embedding_dim)
        """
        self._load_dataset()

        rng = np.random.RandomState(seed) if seed is not None else np.random

        # Determine index range
        if gender_filter in self._gender_ranges:
            start, end = self._gender_ranges[gender_filter]
        else:
            start, end = 0, self._dataset_size

        # Sample indices
        indices = rng.randint(start, end, size=count)

        # Sample embeddings
        embeddings = [self._sample_embedding_at_index(idx) for idx in indices]

        return np.stack(embeddings, axis=0)

    def get_embedding_info(self, idx: int) -> Dict[str, Any]:
        """
        Get metadata about an embedding in the dataset.

        Args:
            idx: Dataset index

        Returns:
            dict: Embedding metadata
        """
        self._load_dataset()

        if idx < 0 or idx >= self._dataset_size:
            raise ValueError(f"Invalid index {idx}, dataset has {self._dataset_size} embeddings")

        sample = self._dataset[idx]

        return {
            "index": idx,
            "dataset": self.config.dataset_name,
            "keys": list(sample.keys()),
            "embedding_shape": np.array(sample.get("xvector", sample.get("embedding"))).shape,
        }

    def list_speaker_mappings(self) -> Dict[str, int]:
        """
        Get all cached speaker name -> index mappings.

        Returns:
            dict: Speaker name to dataset index mapping
        """
        return self._speaker_indices.copy()

    def set_gender_ranges(
        self,
        male_range: tuple[int, int],
        female_range: tuple[int, int]
    ):
        """
        Customize gender index ranges.

        Allows tuning the heuristic gender filtering based on empirical testing.

        Args:
            male_range: (start, end) indices for male voices
            female_range: (start, end) indices for female voices
        """
        self._gender_ranges["male"] = male_range
        self._gender_ranges["female"] = female_range

    def get_dataset_size(self) -> int:
        """
        Get total number of embeddings in dataset.

        Returns:
            int: Dataset size
        """
        self._load_dataset()
        return self._dataset_size

    def get_info(self) -> Dict[str, Any]:
        """Get sampler information."""
        info = super().get_info()
        info.update({
            "dataset_name": self.config.dataset_name,
            "dataset_loaded": self._dataset_loaded,
            "dataset_size": self._dataset_size,
            "cached_speakers": len(self._speaker_indices),
            "gender_ranges": self._gender_ranges,
        })
        return info
