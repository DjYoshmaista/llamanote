"""
Compression subsystem for audio and text files.

Provides utilities for compressing various file types while maintaining
data integrity and allowing user control over quality settings.
"""

from .compress_audio import AudioCompressor
from .compress_text import TextCompressor
from .compression_config import CompressionConfig

__all__ = [
    "AudioCompressor",
    "TextCompressor",
    "CompressionConfig"
]
