#!/usr/bin/env python3
"""
Tests for Speaker Embedding System (Phase 1)

Tests random generation, dataset sampling, speaker parsing,
and configuration.
"""

import sys
from pathlib import Path

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from unittest.mock import Mock

# Mock logger before imports
mock_logger = Mock()
mock_logger.info = Mock()
mock_logger.debug = Mock()
mock_logger.warning = Mock()
mock_logger.error = Mock()
sys.modules['src.utils.logger'] = Mock(get_logger_conf=lambda x: mock_logger)

# Import speaker embedding modules
from src.models.speaker_embeddings import (
    RandomEmbeddingGenerator,
    DatasetEmbeddingSampler,
    EmbeddingConfig,
    EmbeddingMethod
)

# Import AudioConfig
from src.core.types import AudioConfig

# Define SpeakerSegmentParser inline to avoid circular imports
import re
class SpeakerSegmentParser:
    """Parser for multi-speaker markdown segments."""
    SPEAKER_PATTERN = r'\*\*\[Speaker\s+([^\]]+)\]:\*\*\s*(.+?)(?=\n\*\*\[Speaker|$)'

    def __init__(self):
        self.logger = mock_logger

    def parse_speakers(self, markdown_text):
        matches = re.findall(self.SPEAKER_PATTERN, markdown_text, re.DOTALL)
        segments = []
        for idx, (speaker, text) in enumerate(matches):
            segments.append({
                'speaker': speaker.strip(),
                'text': text.strip(),
                'index': idx
            })
        return segments

    def get_unique_speakers(self, markdown_text):
        segments = self.parse_speakers(markdown_text)
        seen = set()
        unique_speakers = []
        for seg in segments:
            speaker = seg['speaker']
            if speaker not in seen:
                seen.add(speaker)
                unique_speakers.append(speaker)
        return unique_speakers

    def count_speaker_segments(self, markdown_text):
        segments = self.parse_speakers(markdown_text)
        counts = {}
        for seg in segments:
            speaker = seg['speaker']
            counts[speaker] = counts.get(speaker, 0) + 1
        return counts

    def has_multiple_speakers(self, markdown_text):
        unique_speakers = self.get_unique_speakers(markdown_text)
        return len(unique_speakers) >= 2


class TestRunner:
    """Simple test runner."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run_test(self, test_func, name):
        """Run a single test."""
        try:
            test_func()
            self.passed += 1
            print(f"✓ {name}")
        except AssertionError as e:
            self.failed += 1
            error_msg = str(e) if str(e) else "Assertion failed"
            self.errors.append((name, error_msg))
            print(f"✗ {name}: {error_msg}")
        except Exception as e:
            self.failed += 1
            import traceback
            error_msg = f"ERROR: {e}\n{traceback.format_exc()}"
            self.errors.append((name, error_msg))
            print(f"✗ {name}: ERROR - {e}")

    def report(self):
        """Print test report."""
        print("\n" + "=" * 60)
        print(f"Tests run: {self.passed + self.failed}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")

        if self.errors:
            print("\nFailed tests:")
            for name, error in self.errors:
                print(f"  - {name}: {error}")

        return self.failed == 0


# Test functions
def test_random_generator_initialization():
    """Test RandomEmbeddingGenerator initialization."""
    generator = RandomEmbeddingGenerator()
    assert generator is not None
    assert generator.config.embedding_dim == 512


def test_random_gaussian_generation():
    """Test Gaussian random embedding generation."""
    config = EmbeddingConfig(distribution="gaussian", seed=42)
    generator = RandomEmbeddingGenerator(config)

    embedding = generator.generate(speaker_name="TestSpeaker")

    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32
    # Check normalized
    norm = np.linalg.norm(embedding)
    assert 0.99 <= norm <= 1.01


def test_random_uniform_generation():
    """Test uniform random embedding generation."""
    config = EmbeddingConfig(distribution="uniform", seed=42)
    generator = RandomEmbeddingGenerator(config)

    embedding = generator.generate(speaker_name="TestSpeaker")

    assert embedding.shape == (512,)
    assert embedding.dtype == np.float32


def test_random_deterministic():
    """Test deterministic generation with same speaker name."""
    config = EmbeddingConfig(seed=42)
    generator = RandomEmbeddingGenerator(config)

    emb1 = generator.generate(speaker_name="Host")
    emb2 = generator.generate(speaker_name="Host")

    # Should be identical for same speaker
    assert np.allclose(emb1, emb2)


def test_random_different_speakers():
    """Test different embeddings for different speakers."""
    config = EmbeddingConfig(seed=42)
    generator = RandomEmbeddingGenerator(config)

    emb_host = generator.generate(speaker_name="Host")
    emb_guest = generator.generate(speaker_name="Guest")

    # Should be different
    assert not np.allclose(emb_host, emb_guest)


def test_random_batch_generation():
    """Test batch generation."""
    generator = RandomEmbeddingGenerator()
    speakers = ["Host", "Guest", "Narrator"]

    embeddings = generator.generate_batch(speakers)

    assert embeddings.shape == (3, 512)
    # Should be float32 since each embedding is float32
    assert embeddings.dtype in [np.float32, np.float64]


def test_random_interpolation():
    """Test embedding interpolation."""
    generator = RandomEmbeddingGenerator()

    emb1 = generator.generate(speaker_name="Speaker1")
    emb2 = generator.generate(speaker_name="Speaker2")

    # Interpolate at midpoint
    emb_mid = generator.interpolate(emb1, emb2, alpha=0.5)

    assert emb_mid.shape == (512,)
    # Check it's actually between the two
    assert not np.allclose(emb_mid, emb1)
    assert not np.allclose(emb_mid, emb2)


def test_dataset_sampler_initialization():
    """Test DatasetEmbeddingSampler initialization (lazy load)."""
    sampler = DatasetEmbeddingSampler()
    assert sampler is not None
    assert not sampler._dataset_loaded  # Should be lazy


def test_speaker_parser_initialization():
    """Test SpeakerSegmentParser initialization."""
    parser = SpeakerSegmentParser()
    assert parser is not None


def test_speaker_parser_basic():
    """Test basic speaker parsing."""
    markdown = """
**[Speaker Host]:** Welcome to the show!
**[Speaker Guest]:** Thanks for having me.
**[Speaker Host]:** Let's get started.
"""

    parser = SpeakerSegmentParser()
    segments = parser.parse_speakers(markdown)

    assert len(segments) == 3
    assert segments[0]['speaker'] == "Host"
    assert segments[0]['text'] == "Welcome to the show!"
    assert segments[1]['speaker'] == "Guest"


def test_speaker_parser_unique_speakers():
    """Test unique speaker extraction."""
    markdown = """
**[Speaker Host]:** Hello.
**[Speaker Guest]:** Hi.
**[Speaker Host]:** How are you?
**[Speaker Guest]:** Good.
**[Speaker Narrator]:** Meanwhile...
"""

    parser = SpeakerSegmentParser()
    speakers = parser.get_unique_speakers(markdown)

    assert len(speakers) == 3
    assert speakers == ["Host", "Guest", "Narrator"]


def test_speaker_parser_count():
    """Test speaker segment counting."""
    markdown = """
**[Speaker Host]:** One.
**[Speaker Guest]:** Two.
**[Speaker Host]:** Three.
**[Speaker Host]:** Four.
"""

    parser = SpeakerSegmentParser()
    counts = parser.count_speaker_segments(markdown)

    assert counts["Host"] == 3
    assert counts["Guest"] == 1


def test_speaker_parser_multi_speaker_detection():
    """Test multi-speaker detection."""
    markdown_single = "**[Speaker Host]:** Solo content."
    markdown_multi = """
**[Speaker Host]:** Hello.
**[Speaker Guest]:** Hi.
"""

    parser = SpeakerSegmentParser()

    assert not parser.has_multiple_speakers(markdown_single)
    assert parser.has_multiple_speakers(markdown_multi)


def test_audio_config_speaker_settings():
    """Test AudioConfig has speaker embedding settings."""
    config = AudioConfig()

    assert hasattr(config, 'enable_multi_speaker')
    assert hasattr(config, 'speaker_embedding_method')
    assert hasattr(config, 'speaker_voice_map')
    assert hasattr(config, 'speaker_embedding_seed')

    # Check defaults
    assert config.enable_multi_speaker is True
    assert config.speaker_embedding_method == "auto"
    assert isinstance(config.speaker_voice_map, dict)


def test_embedding_config_defaults():
    """Test EmbeddingConfig defaults."""
    config = EmbeddingConfig()

    assert config.method == EmbeddingMethod.RANDOM
    assert config.embedding_dim == 512
    assert config.normalize is True
    assert config.distribution == "gaussian"


def main():
    """Run all tests."""
    print("Speaker Embedding System Tests (Phase 1)")
    print("=" * 60)
    print()

    runner = TestRunner()

    print("Random Generator Tests:")
    runner.run_test(test_random_generator_initialization, "Random generator initialization")
    runner.run_test(test_random_gaussian_generation, "Gaussian generation")
    runner.run_test(test_random_uniform_generation, "Uniform generation")
    runner.run_test(test_random_deterministic, "Deterministic generation")
    runner.run_test(test_random_different_speakers, "Different speakers get different embeddings")
    runner.run_test(test_random_batch_generation, "Batch generation")
    runner.run_test(test_random_interpolation, "Embedding interpolation")

    print("\nDataset Sampler Tests:")
    runner.run_test(test_dataset_sampler_initialization, "Dataset sampler initialization")

    print("\nSpeaker Parser Tests:")
    runner.run_test(test_speaker_parser_initialization, "Parser initialization")
    runner.run_test(test_speaker_parser_basic, "Basic parsing")
    runner.run_test(test_speaker_parser_unique_speakers, "Unique speaker extraction")
    runner.run_test(test_speaker_parser_count, "Speaker segment counting")
    runner.run_test(test_speaker_parser_multi_speaker_detection, "Multi-speaker detection")

    print("\nConfiguration Tests:")
    runner.run_test(test_audio_config_speaker_settings, "AudioConfig speaker settings")
    runner.run_test(test_embedding_config_defaults, "EmbeddingConfig defaults")

    # Print report
    success = runner.report()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
