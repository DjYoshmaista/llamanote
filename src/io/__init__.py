# llamanote/io/__init__.py
"""
Input/Output Package for LlamaNote Enhanced

This package handles all file system interactions:
- file_handler: Saving/loading final output files, metadata, and backups.
- batch_manager: Discovering and managing lists of input files.
- checkpoints: Saving and loading intermediate pipeline state.
"""

from .file_handler import FileHandler, MetadataFormatter
from .batch_manager import BatchFileManager
from .checkpoints import CheckpointManager

__all__ = [
    "FileHandler",
    "MetadataFormatter",
    "BatchFileManager",
    "CheckpointManager",
]
