"""
Configuration management for compression settings.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class CompressionConfig:
    """Configuration for compression operations."""

    # General settings
    size_threshold_mb: float = 10.0  # Only compress files larger than this
    delete_original: bool = False  # Whether to delete original after compression
    auto_compress: bool = True  # Automatically compress after generation

    # Audio compression settings
    audio_format: str = "mp3"  # mp3, flac, ogg, opus
    audio_quality: str = "high"  # low, medium, high, lossless
    audio_bitrate_kbps: Optional[int] = None  # Override quality preset
    audio_sample_rate: Optional[int] = None  # Resample to this rate (None = keep original)

    # Text compression settings
    text_compression_level: int = 9  # gzip compression level (1-9)
    text_format: str = "gz"  # gz, bz2, xz

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CompressionConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def load(cls, config_path: Path) -> "CompressionConfig":
        """Load configuration from file."""
        if not config_path.exists():
            return cls()  # Return defaults

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception:
            return cls()  # Return defaults on error

    def save(self, config_path: Path):
        """Save configuration to file."""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_audio_quality_preset(self) -> Dict[str, Any]:
        """
        Get audio compression preset for selected quality.

        Returns:
            Dictionary with format-specific settings
        """
        presets = {
            "mp3": {
                "low": {"bitrate": "128k"},
                "medium": {"bitrate": "192k"},
                "high": {"bitrate": "320k"},
                "lossless": {"bitrate": "320k"}  # MP3 can't be truly lossless
            },
            "flac": {
                "low": {"compression_level": 0},
                "medium": {"compression_level": 5},
                "high": {"compression_level": 8},
                "lossless": {"compression_level": 8}  # FLAC is always lossless
            },
            "ogg": {
                "low": {"quality": 3},
                "medium": {"quality": 6},
                "high": {"quality": 9},
                "lossless": {"quality": 10}
            },
            "opus": {
                "low": {"bitrate": "64k"},
                "medium": {"bitrate": "128k"},
                "high": {"bitrate": "256k"},
                "lossless": {"bitrate": "512k"}  # Opus can't be truly lossless
            }
        }

        format_presets = presets.get(self.audio_format, presets["mp3"])
        return format_presets.get(self.audio_quality, format_presets["high"])
