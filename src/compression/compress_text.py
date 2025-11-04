"""
Text file compression utilities.

Supports gzip, bz2, and xz compression for text files (JSON, MD, LOG)
with 100% data integrity preservation.
"""

import gzip
import bz2
import lzma
from pathlib import Path
from typing import Optional

from ..utils.logger import ContextLogger
from .compression_config import CompressionConfig


class TextCompressor:
    """
    Compresses text files using lossless compression algorithms.

    Supports gzip, bz2, and xz with configurable compression levels.
    """

    def __init__(
        self,
        config: Optional[CompressionConfig] = None,
        logger: Optional[ContextLogger] = None
    ):
        """
        Initialize text compressor.

        Args:
            config: Compression configuration
            logger: Optional logger instance
        """
        self.config = config or CompressionConfig()
        self.logger = logger or ContextLogger("TextCompressor")

    def should_compress(self, file_path: Path) -> bool:
        """
        Check if file should be compressed based on size threshold.

        Args:
            file_path: Path to file

        Returns:
            True if file should be compressed
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        return file_size_mb > self.config.size_threshold_mb

    def compress_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        format_override: Optional[str] = None
    ) -> Optional[Path]:
        """
        Compress text file.

        Args:
            input_path: Input file path
            output_path: Output path (None = auto-generate)
            format_override: Override configured format

        Returns:
            Path to compressed file or None on failure
        """
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return None

        # Determine compression format
        compression_format = format_override or self.config.text_format

        # Generate output path if not provided
        if output_path is None:
            output_path = input_path.with_suffix(input_path.suffix + f".{compression_format}")

        self.logger.info(f"Compressing {input_path.name} using {compression_format}")

        try:
            # Read input file
            with open(input_path, 'rb') as f_in:
                data = f_in.read()

            original_size = len(data)

            # Compress based on format
            if compression_format == "gz":
                compressed_data = gzip.compress(
                    data,
                    compresslevel=self.config.text_compression_level
                )
            elif compression_format == "bz2":
                compressed_data = bz2.compress(
                    data,
                    compresslevel=self.config.text_compression_level
                )
            elif compression_format == "xz":
                compressed_data = lzma.compress(
                    data,
                    preset=self.config.text_compression_level
                )
            else:
                self.logger.error(f"Unsupported compression format: {compression_format}")
                return None

            # Write compressed file
            with open(output_path, 'wb') as f_out:
                f_out.write(compressed_data)

            compressed_size = len(compressed_data)
            compression_ratio = (1 - compressed_size / original_size) * 100

            self.logger.info(
                f"Compression complete: {original_size / (1024*1024):.2f}MB → "
                f"{compressed_size / (1024*1024):.2f}MB "
                f"({compression_ratio:.1f}% reduction)"
            )

            # Delete original if configured
            if self.config.delete_original:
                input_path.unlink()
                self.logger.info(f"Deleted original file: {input_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return None

    def decompress_file(
        self,
        input_path: Path,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Decompress text file.

        Args:
            input_path: Compressed file path
            output_path: Output path (None = auto-detect)

        Returns:
            Path to decompressed file or None on failure
        """
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            return None

        # Detect compression format from extension
        if input_path.suffix == ".gz":
            compression_format = "gz"
        elif input_path.suffix == ".bz2":
            compression_format = "bz2"
        elif input_path.suffix == ".xz":
            compression_format = "xz"
        else:
            self.logger.error(f"Unknown compression format: {input_path.suffix}")
            return None

        # Generate output path if not provided
        if output_path is None:
            # Remove compression extension
            output_path = input_path.with_suffix('')

        self.logger.info(f"Decompressing {input_path.name}")

        try:
            # Read compressed file
            with open(input_path, 'rb') as f_in:
                compressed_data = f_in.read()

            # Decompress based on format
            if compression_format == "gz":
                data = gzip.decompress(compressed_data)
            elif compression_format == "bz2":
                data = bz2.decompress(compressed_data)
            elif compression_format == "xz":
                data = lzma.decompress(compressed_data)
            else:
                self.logger.error(f"Unsupported compression format: {compression_format}")
                return None

            # Write decompressed file
            with open(output_path, 'wb') as f_out:
                f_out.write(data)

            self.logger.info(f"Decompression complete: {output_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return None

    def compress_if_needed(self, file_path: Path) -> Optional[Path]:
        """
        Compress file only if it exceeds size threshold.

        Args:
            file_path: File to potentially compress

        Returns:
            Path to compressed file if compressed, original path if not, None on failure
        """
        if not self.should_compress(file_path):
            self.logger.debug(f"File {file_path.name} below threshold, skipping compression")
            return file_path

        return self.compress_file(file_path)

    def batch_compress(
        self,
        directory: Path,
        pattern: str = "*",
        recursive: bool = True
    ) -> list:
        """
        Batch compress files in a directory.

        Args:
            directory: Directory to search
            pattern: File pattern to match
            recursive: Whether to search recursively

        Returns:
            List of compressed file paths
        """
        if not directory.exists():
            self.logger.error(f"Directory not found: {directory}")
            return []

        compressed_files = []

        # Find files
        if recursive:
            files = directory.rglob(pattern)
        else:
            files = directory.glob(pattern)

        for file_path in files:
            if file_path.is_file() and not file_path.suffix in ['.gz', '.bz2', '.xz']:
                # Skip already compressed files
                result = self.compress_if_needed(file_path)
                if result and result != file_path:
                    compressed_files.append(result)

        self.logger.info(f"Batch compression complete: {len(compressed_files)} files compressed")

        return compressed_files


if __name__ == "__main__":
    """Standalone script for text compression."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Compress text files")
    parser.add_argument("input", type=Path, help="Input file or directory")
    parser.add_argument("-o", "--output", type=Path, help="Output file path")
    parser.add_argument("-f", "--format", choices=["gz", "bz2", "xz"],
                        default="gz", help="Compression format")
    parser.add_argument("-l", "--level", type=int, default=9,
                        help="Compression level (1-9)")
    parser.add_argument("--delete-original", action="store_true",
                        help="Delete original file after compression")
    parser.add_argument("-d", "--decompress", action="store_true",
                        help="Decompress instead of compress")
    parser.add_argument("-b", "--batch", action="store_true",
                        help="Batch compress directory")
    parser.add_argument("--pattern", default="*.{json,md,log,txt}",
                        help="File pattern for batch mode")

    args = parser.parse_args()

    # Create config
    config = CompressionConfig(
        text_format=args.format,
        text_compression_level=args.level,
        delete_original=args.delete_original
    )

    # Create compressor
    compressor = TextCompressor(config=config)

    # Execute operation
    if args.decompress:
        output = compressor.decompress_file(args.input, args.output)
        if output:
            print(f"Successfully decompressed to: {output}")
            sys.exit(0)
        else:
            print("Decompression failed")
            sys.exit(1)

    elif args.batch:
        if not args.input.is_dir():
            print("ERROR: Input must be a directory for batch mode")
            sys.exit(1)

        compressed = compressor.batch_compress(args.input, args.pattern)
        print(f"Successfully compressed {len(compressed)} files")
        sys.exit(0)

    else:
        output = compressor.compress_file(args.input, args.output)
        if output:
            print(f"Successfully compressed to: {output}")
            sys.exit(0)
        else:
            print("Compression failed")
            sys.exit(1)
