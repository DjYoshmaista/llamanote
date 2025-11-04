"""
Audio file compression utilities.

Supports multiple formats (MP3, FLAC, OGG, OPUS) with quality presets
and user-configurable settings.
"""

import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import soundfile as sf

from ..utils.logger import ContextLogger
from .compression_config import CompressionConfig


class AudioCompressor:
    """
    Compresses audio files using various codecs and quality settings.

    Uses ffmpeg for compression with support for MP3, FLAC, OGG Vorbis, and Opus.
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        logger: Optional[ContextLogger] = None
    ):
        """
        Initialize audio compressor.

        Args:
            config: Compression configuration
            logger: Optional logger instance
        """
        self.config = config or CompressionConfig()
        self.logger = logger or ContextLogger("AudioCompressor")

    def get_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        """
        Get information about an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio metadata
        """
        try:
            info = sf.info(str(audio_path))

            file_size_mb = audio_path.stat().st_size / (1024 * 1024)

            return {
                "file_path": str(audio_path),
                "file_size_mb": round(file_size_mb, 2),
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_seconds": round(info.duration, 2),
                "format": info.format,
                "subtype": info.subtype
            }

        except Exception as e:
            self.logger.error(f"Failed to get audio info: {e}")
            return {}

    def should_compress(self, audio_path: Path) -> bool:
        """
        Check if file should be compressed based on size threshold.

        Args:
            audio_path: Path to audio file

        Returns:
            True if file should be compressed
        """
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        return file_size_mb > self.config.size_threshold_mb

    def compress_audio(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        format_override: Optional[str] = None,
        quality_override: Optional[str] = None
    ) -> Optional[Path]:
        """
        Compress audio file.

        Args:
            input_path: Input audio file path
            output_path: Output path (None = auto-generate)
            format_override: Override configured format
            quality_override: Override configured quality

        Returns:
            Path to compressed file or None on failure
        """
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return None

        # Determine output format and quality
        output_format = format_override or self.config.audio_format
        quality = quality_override or self.config.audio_quality

        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.with_suffix(f".{output_format}")

        self.logger.info(f"Compressing {input_path.name} to {output_format} ({quality} quality)")

        # Get info about input file
        input_info = self.get_audio_info(input_path)

        # Build ffmpeg command
        cmd = self._build_ffmpeg_command(
            input_path,
            output_path,
            output_format,
            quality,
            input_info
        )

        # Execute compression
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if output_path.exists():
                output_info = self.get_audio_info(output_path)
                compression_ratio = (
                    (1 - output_info["file_size_mb"] / input_info["file_size_mb"]) * 100
                    if input_info.get("file_size_mb", 0) > 0 else 0
                )

                self.logger.info(
                    f"Compression complete: {input_info['file_size_mb']:.2f}MB → "
                    f"{output_info['file_size_mb']:.2f}MB "
                    f"({compression_ratio:.1f}% reduction)"
                )

                # Delete original if configured
                if self.config.delete_original and input_path != output_path:
                    input_path.unlink()
                    self.logger.info(f"Deleted original file: {input_path}")

                return output_path

            else:
                self.logger.error("Compression failed: output file not created")
                return None

        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg error: {e.stderr}")
            return None

        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return None

    def _build_ffmpeg_command(
        self,
        input_path: Path,
        output_path: Path,
        output_format: str,
        quality: str,
        input_info: Dict[str, Any]
    ) -> list:
        """
        Build ffmpeg command for compression.

        Args:
            input_path: Input file path
            output_path: Output file path
            output_format: Output format
            quality: Quality preset
            input_info: Input file metadata

        Returns:
            List of command arguments
        """
        # Get quality preset
        temp_config = CompressionConfig(
            audio_format=output_format,
            audio_quality=quality
        )
        preset = temp_config.get_audio_quality_preset()

        cmd = ["ffmpeg", "-i", str(input_path), "-y"]

        # Apply sample rate override if set
        if self.config.audio_sample_rate:
            cmd.extend(["-ar", str(self.config.audio_sample_rate)])

        # Format-specific encoding
        if output_format == "mp3":
            cmd.extend(["-codec:a", "libmp3lame"])
            if self.config.audio_bitrate_kbps:
                cmd.extend(["-b:a", f"{self.config.audio_bitrate_kbps}k"])
            else:
                cmd.extend(["-b:a", preset.get("bitrate", "192k")])

        elif output_format == "flac":
            cmd.extend(["-codec:a", "flac"])
            level = preset.get("compression_level", 5)
            cmd.extend(["-compression_level", str(level)])

        elif output_format == "ogg":
            cmd.extend(["-codec:a", "libvorbis"])
            if self.config.audio_bitrate_kbps:
                cmd.extend(["-b:a", f"{self.config.audio_bitrate_kbps}k"])
            else:
                cmd.extend(["-qscale:a", str(preset.get("quality", 6))])

        elif output_format == "opus":
            cmd.extend(["-codec:a", "libopus"])
            if self.config.audio_bitrate_kbps:
                cmd.extend(["-b:a", f"{self.config.audio_bitrate_kbps}k"])
            else:
                cmd.extend(["-b:a", preset.get("bitrate", "128k")])

        cmd.append(str(output_path))

        return cmd

    def get_compression_preview(
        self,
        input_path: Path,
        output_format: str,
        quality: str
    ) -> Dict[str, Any]:
        """
        Get preview of compression without actually compressing.

        Args:
            input_path: Input file path
            output_format: Target format
            quality: Target quality

        Returns:
            Dictionary with preview information
        """
        input_info = self.get_audio_info(input_path)

        # Estimate compression ratios
        compression_estimates = {
            "mp3": {"low": 0.90, "medium": 0.85, "high": 0.70, "lossless": 0.70},
            "flac": {"low": 0.50, "medium": 0.50, "high": 0.50, "lossless": 0.50},
            "ogg": {"low": 0.92, "medium": 0.88, "high": 0.75, "lossless": 0.65},
            "opus": {"low": 0.95, "medium": 0.90, "high": 0.80, "lossless": 0.70}
        }

        format_estimates = compression_estimates.get(output_format, compression_estimates["mp3"])
        compression_ratio = format_estimates.get(quality, 0.80)

        estimated_size_mb = input_info["file_size_mb"] * compression_ratio
        size_reduction_percent = (1 - compression_ratio) * 100

        # Determine if lossy
        is_lossless = (output_format == "flac") or (quality == "lossless" and output_format in ["ogg"])

        # Get quality info
        temp_config = CompressionConfig(audio_format=output_format, audio_quality=quality)
        preset = temp_config.get_audio_quality_preset()

        preview = {
            "input_info": input_info,
            "output_format": output_format,
            "quality": quality,
            "is_lossless": is_lossless,
            "estimated_size_mb": round(estimated_size_mb, 2),
            "estimated_reduction_percent": round(size_reduction_percent, 1),
            "settings": preset,
            "potential_data_loss": not is_lossless
        }

        return preview

    def check_ffmpeg_available(self) -> bool:
        """
        Check if ffmpeg is available on the system.

        Returns:
            True if ffmpeg is available
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False


if __name__ == "__main__":
    """Standalone script for audio compression."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Compress audio files")
    parser.add_argument("input", type=Path, help="Input audio file")
    parser.add_argument("-o", "--output", type=Path, help="Output file path")
    parser.add_argument("-f", "--format", choices=["mp3", "flac", "ogg", "opus"],
                        default="mp3", help="Output format")
    parser.add_argument("-q", "--quality", choices=["low", "medium", "high", "lossless"],
                        default="high", help="Quality preset")
    parser.add_argument("--delete-original", action="store_true",
                        help="Delete original file after compression")

    args = parser.parse_args()

    # Create config
    config = CompressionConfig(
        audio_format=args.format,
        audio_quality=args.quality,
        delete_original=args.delete_original
    )

    # Create compressor
    compressor = AudioCompressor(config=config)

    # Check ffmpeg
    if not compressor.check_ffmpeg_available():
        print("ERROR: ffmpeg not found. Please install ffmpeg.")
        sys.exit(1)

    # Compress
    output = compressor.compress_audio(args.input, args.output)

    if output:
        print(f"Successfully compressed to: {output}")
        sys.exit(0)
    else:
        print("Compression failed")
        sys.exit(1)
