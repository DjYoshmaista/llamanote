"""
Directory structure management for hierarchical output organization.

Manages the creation and organization of output directories following the structure:
output/{file}/{session}/{checkpoint}/{chunk}/
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import os

from ..utils.logger import ContextLogger


class DirectoryManager:
    """
    Manages hierarchical directory structure for outputs.

    Structure:
        output/
        ├── {input_file}/
        │   └── {session_id}/
        │       ├── {checkpoint_name}/
        │       │   ├── {chunk_name}/
        │       │   │   ├── audio files
        │       │   │   ├── transcripts
        │       │   │   └── reports
        │       │   ├── checkpoint_report.json
        │       │   └── checkpoint_transcript.md
        │       ├── final_output.wav
        │       ├── session_report.json
        │       ├── session_transcript.md
        │       └── session_errors.log
    """

    def __init__(
        self,
        base_output_dir: Path,
        logger: Optional[ContextLogger] = None
    ):
        """
        Initialize directory manager.

        Args:
            base_output_dir: Base output directory (e.g., 'output/')
            logger: Optional logger instance
        """
        self.base_output_dir = Path(base_output_dir)
        self.logger = logger or ContextLogger("DirectoryManager")

    def get_session_dir(self, input_stem: str, session_id: str) -> Path:
        """
        Get the session directory path.

        Args:
            input_stem: Input file stem (without extension)
            session_id: Unique session identifier

        Returns:
            Path to session directory
        """
        return self.base_output_dir / input_stem / session_id

    def get_checkpoint_dir(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str
    ) -> Path:
        """
        Get the checkpoint directory path.

        Args:
            input_stem: Input file stem
            session_id: Session identifier
            checkpoint_name: Checkpoint name (e.g., 'checkpoint1')

        Returns:
            Path to checkpoint directory
        """
        return self.get_session_dir(input_stem, session_id) / checkpoint_name

    def get_chunk_dir(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str,
        chunk_name: str
    ) -> Path:
        """
        Get the chunk directory path.

        Args:
            input_stem: Input file stem
            session_id: Session identifier
            checkpoint_name: Checkpoint name
            chunk_name: Chunk name (e.g., 'chunk1')

        Returns:
            Path to chunk directory
        """
        return self.get_checkpoint_dir(
            input_stem,
            session_id,
            checkpoint_name
        ) / chunk_name

    def create_session_structure(
        self,
        input_stem: str,
        session_id: str
    ) -> Path:
        """
        Create the base session directory structure.

        Args:
            input_stem: Input file stem
            session_id: Session identifier

        Returns:
            Path to created session directory
        """
        session_dir = self.get_session_dir(input_stem, session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Created session directory: {session_dir}")
        return session_dir

    def create_checkpoint_structure(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str
    ) -> Path:
        """
        Create checkpoint directory structure.

        Args:
            input_stem: Input file stem
            session_id: Session identifier
            checkpoint_name: Checkpoint name

        Returns:
            Path to created checkpoint directory
        """
        checkpoint_dir = self.get_checkpoint_dir(input_stem, session_id, checkpoint_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Created checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir

    def create_chunk_structure(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str,
        chunk_name: str
    ) -> Path:
        """
        Create chunk directory structure.

        Args:
            input_stem: Input file stem
            session_id: Session identifier
            checkpoint_name: Checkpoint name
            chunk_name: Chunk name

        Returns:
            Path to created chunk directory
        """
        chunk_dir = self.get_chunk_dir(
            input_stem,
            session_id,
            checkpoint_name,
            chunk_name
        )
        chunk_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Created chunk directory: {chunk_dir}")
        return chunk_dir

    def get_chunk_audio_path(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str,
        chunk_name: str,
        audio_format: str = "wav"
    ) -> Path:
        """Get path for chunk audio file."""
        chunk_dir = self.get_chunk_dir(input_stem, session_id, checkpoint_name, chunk_name)
        return chunk_dir / f"{chunk_name}.{audio_format}"

    def get_chunk_transcript_path(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str,
        chunk_name: str
    ) -> Path:
        """Get path for chunk transcript file."""
        chunk_dir = self.get_chunk_dir(input_stem, session_id, checkpoint_name, chunk_name)
        return chunk_dir / f"{chunk_name}_transcript.md"

    def get_chunk_original_text_path(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str,
        chunk_name: str
    ) -> Path:
        """Get path for chunk original text file."""
        chunk_dir = self.get_chunk_dir(input_stem, session_id, checkpoint_name, chunk_name)
        return chunk_dir / f"{chunk_name}_original.md"

    def get_chunk_report_path(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str,
        chunk_name: str
    ) -> Path:
        """Get path for chunk report file."""
        chunk_dir = self.get_chunk_dir(input_stem, session_id, checkpoint_name, chunk_name)
        return chunk_dir / f"{chunk_name}_report.json"

    def get_checkpoint_report_path(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str
    ) -> Path:
        """Get path for checkpoint report file."""
        checkpoint_dir = self.get_checkpoint_dir(input_stem, session_id, checkpoint_name)
        return checkpoint_dir / f"{checkpoint_name}_report.json"

    def get_checkpoint_transcript_path(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str
    ) -> Path:
        """Get path for checkpoint transcript file."""
        checkpoint_dir = self.get_checkpoint_dir(input_stem, session_id, checkpoint_name)
        return checkpoint_dir / f"{checkpoint_name}_transcript.md"

    def get_session_final_output_path(
        self,
        input_stem: str,
        session_id: str,
        audio_format: str = "wav"
    ) -> Path:
        """Get path for final session output audio."""
        session_dir = self.get_session_dir(input_stem, session_id)
        return session_dir / f"{input_stem}_final.{audio_format}"

    def get_session_report_path(
        self,
        input_stem: str,
        session_id: str
    ) -> Path:
        """Get path for session report file."""
        session_dir = self.get_session_dir(input_stem, session_id)
        return session_dir / f"{session_id}_report.json"

    def get_session_transcript_path(
        self,
        input_stem: str,
        session_id: str
    ) -> Path:
        """Get path for session transcript file."""
        session_dir = self.get_session_dir(input_stem, session_id)
        return session_dir / f"{session_id}_transcript.md"

    def get_session_errors_log_path(
        self,
        input_stem: str,
        session_id: str
    ) -> Path:
        """Get path for session errors log file."""
        session_dir = self.get_session_dir(input_stem, session_id)
        return session_dir / f"{session_id}_errors.log"

    def create_symlink(self, target: Path, link: Path) -> bool:
        """
        Create a symbolic link.

        Args:
            target: Target file/directory
            link: Symbolic link path

        Returns:
            True if successful
        """
        try:
            # Create parent directory if needed
            link.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing link if present
            if link.exists() or link.is_symlink():
                link.unlink()

            # Create symbolic link
            os.symlink(target.absolute(), link.absolute())
            self.logger.debug(f"Created symlink: {link} → {target}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create symlink {link} → {target}: {e}")
            return False

    def list_sessions(self, input_stem: str) -> List[str]:
        """
        List all sessions for a given input file.

        Args:
            input_stem: Input file stem

        Returns:
            List of session IDs
        """
        input_dir = self.base_output_dir / input_stem
        if not input_dir.exists():
            return []

        return [
            d.name for d in input_dir.iterdir()
            if d.is_dir() and d.name.startswith('session')
        ]

    def list_checkpoints(self, input_stem: str, session_id: str) -> List[str]:
        """
        List all checkpoints for a session.

        Args:
            input_stem: Input file stem
            session_id: Session identifier

        Returns:
            List of checkpoint names
        """
        session_dir = self.get_session_dir(input_stem, session_id)
        if not session_dir.exists():
            return []

        return [
            d.name for d in session_dir.iterdir()
            if d.is_dir() and d.name.startswith('checkpoint')
        ]

    def list_chunks(
        self,
        input_stem: str,
        session_id: str,
        checkpoint_name: str
    ) -> List[str]:
        """
        List all chunks for a checkpoint.

        Args:
            input_stem: Input file stem
            session_id: Session identifier
            checkpoint_name: Checkpoint name

        Returns:
            List of chunk names
        """
        checkpoint_dir = self.get_checkpoint_dir(input_stem, session_id, checkpoint_name)
        if not checkpoint_dir.exists():
            return []

        return [
            d.name for d in checkpoint_dir.iterdir()
            if d.is_dir() and d.name.startswith('chunk')
        ]
