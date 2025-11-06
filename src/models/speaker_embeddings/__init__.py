# src/models/speaker_embeddings/__init__.py
"""
Speaker Embedding Management System

This module provides tools for generating, managing, and applying speaker embeddings
for multi-speaker Text-to-Speech (TTS) generation. It supports multiple methods for
creating speaker embeddings:

1. Random Generation: Create synthetic embeddings from statistical distributions
2. Audio Extraction: Extract embeddings from audio files using pre-trained models
3. Dataset Sampling: Sample embeddings from pre-computed datasets

The module is designed to work with SpeechT5 and other TTS models that accept
speaker embeddings as conditioning vectors.

Architecture:
- base.py: Abstract base classes for all embedding generators
- random_generator.py: Random/synthetic embedding generation
- audio_extractor.py: Audio-based embedding extraction (future)
- dataset_sampler.py: Dataset-based embedding sampling
- manager.py: Unified interface for managing speaker embeddings
- cache.py: Persistence and caching layer (future)

Usage:
    from src.models.speaker_embeddings import (
        RandomEmbeddingGenerator,
        DatasetEmbeddingSampler,
        SpeakerEmbeddingManager
    )

    # Create a manager
    manager = SpeakerEmbeddingManager()

    # Generate random embedding for a new speaker
    embedding = manager.get_or_create_embedding(
        speaker_name="Host",
        method="random"
    )

    # Or sample from dataset
    embedding = manager.get_or_create_embedding(
        speaker_name="Guest",
        method="dataset",
        dataset_index=1234
    )
"""

from .base import (
    SpeakerEmbeddingGenerator,
    EmbeddingMethod,
    EmbeddingConfig
)
from .random_generator import RandomEmbeddingGenerator
from .dataset_sampler import DatasetEmbeddingSampler
from .manager import SpeakerEmbeddingManager
from .audio_extractor import AudioEmbeddingExtractor, AudioQualityAssessment

__all__ = [
    # Base classes
    "SpeakerEmbeddingGenerator",
    "EmbeddingMethod",
    "EmbeddingConfig",

    # Generators
    "RandomEmbeddingGenerator",
    "DatasetEmbeddingSampler",
    "AudioEmbeddingExtractor",
    "AudioQualityAssessment",

    # Manager
    "SpeakerEmbeddingManager",
]

__version__ = "0.1.0"
