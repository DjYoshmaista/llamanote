"""
Text file database and management system.

Manages source text files, maintains a database of processed files,
and provides file editing capabilities.
"""

import json
import shutil
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..utils.logger import ContextLogger


class TextFileManager:
    """
    Manages text files and maintains a database of processed files.

    Directory structure:
        text_files/
        ├── database.json
        └── {file_stem}/
            ├── original/
            │   └── {original_file}
            ├── metadata.json
            └── sessions/
                ├── session1 → output/{file_stem}/session1/
                └── session2 → output/{file_stem}/session2/
    """

    def __init__(
        self,
        text_files_dir: Path,
        logger: Optional[ContextLogger] = None
    ):
        """
        Initialize text file manager.

        Args:
            text_files_dir: Base directory for text files (e.g., 'text_files/')
            logger: Optional logger instance
        """
        self.text_files_dir = Path(text_files_dir)
        self.database_path = self.text_files_dir / "database.json"
        self.logger = logger or ContextLogger("TextFileManager")
        self._ensure_structure()

    def _ensure_structure(self):
        """Ensure base directory and database exist."""
        self.text_files_dir.mkdir(parents=True, exist_ok=True)

        if not self.database_path.exists():
            self._save_database({"files": {}})

    def _load_database(self) -> Dict[str, Any]:
        """Load the text files database."""
        try:
            with open(self.database_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load text files database: {e}")
            return {"files": {}}

    def _save_database(self, data: Dict[str, Any]):
        """Save the text files database."""
        try:
            with open(self.database_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save text files database: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file hash
        """
        sha256 = hashlib.sha256()

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    def add_file(
        self,
        source_file: Path,
        copy_file: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Add a file to the database and copy it to managed storage.

        Args:
            source_file: Source file path
            copy_file: Whether to copy file to managed storage

        Returns:
            File entry data or None on failure
        """
        source_file = Path(source_file)

        if not source_file.exists():
            self.logger.error(f"Source file not found: {source_file}")
            return None

        # Create file stem directory
        file_stem = source_file.stem
        file_dir = self.text_files_dir / file_stem
        original_dir = file_dir / "original"
        sessions_dir = file_dir / "sessions"

        original_dir.mkdir(parents=True, exist_ok=True)
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Copy file to original directory
        stored_path = original_dir / source_file.name

        if copy_file:
            try:
                shutil.copy2(source_file, stored_path)
                self.logger.info(f"Copied file to: {stored_path}")
            except Exception as e:
                self.logger.error(f"Failed to copy file: {e}")
                return None

        # Compute file hash
        file_hash = self._compute_file_hash(stored_path if copy_file else source_file)

        # Create database entry
        db = self._load_database()

        file_entry = {
            "original_path": str(source_file.absolute()),
            "stored_path": str(stored_path),
            "file_stem": file_stem,
            "file_name": source_file.name,
            "file_hash": file_hash,
            "added_date": datetime.now().isoformat(),
            "last_processed": None,
            "sessions": [],
            "file_size_bytes": stored_path.stat().st_size if stored_path.exists() else 0
        }

        db["files"][file_stem] = file_entry

        # Save metadata in file directory
        metadata_path = file_dir / "metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(file_entry, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save file metadata: {e}")

        self._save_database(db)
        self.logger.info(f"Added file to database: {file_stem}")

        return file_entry

    def get_file(self, file_stem: str) -> Optional[Dict[str, Any]]:
        """
        Get file information from database.

        Args:
            file_stem: File stem (without extension)

        Returns:
            File entry data or None if not found
        """
        db = self._load_database()
        return db["files"].get(file_stem)

    def get_stored_path(self, file_stem: str) -> Optional[Path]:
        """
        Get the stored path for a file.

        Args:
            file_stem: File stem

        Returns:
            Path to stored file or None
        """
        file_entry = self.get_file(file_stem)

        if not file_entry:
            return None

        stored_path = Path(file_entry["stored_path"])

        if not stored_path.exists():
            self.logger.warning(f"Stored file not found: {stored_path}")
            return None

        return stored_path

    def update_file_sessions(
        self,
        file_stem: str,
        session_id: str,
        output_session_dir: Path
    ):
        """
        Add session to file's session list and create symlink.

        Args:
            file_stem: File stem
            session_id: Session identifier
            output_session_dir: Path to output session directory
        """
        db = self._load_database()

        if file_stem not in db["files"]:
            self.logger.error(f"File not found in database: {file_stem}")
            return

        # Add session to list if not already present
        if session_id not in db["files"][file_stem]["sessions"]:
            db["files"][file_stem]["sessions"].append(session_id)
            db["files"][file_stem]["last_processed"] = datetime.now().isoformat()

        # Create symlink in sessions directory
        sessions_dir = self.text_files_dir / file_stem / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        symlink_path = sessions_dir / session_id

        try:
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()

            symlink_path.symlink_to(output_session_dir.absolute())
            self.logger.debug(f"Created session symlink: {symlink_path} → {output_session_dir}")

        except Exception as e:
            self.logger.error(f"Failed to create session symlink: {e}")

        # Update metadata file
        file_dir = self.text_files_dir / file_stem
        metadata_path = file_dir / "metadata.json"

        try:
            with open(metadata_path, 'w') as f:
                json.dump(db["files"][file_stem], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to update file metadata: {e}")

        self._save_database(db)

    def list_files(self) -> List[Dict[str, Any]]:
        """
        List all files in database.

        Returns:
            List of file entries
        """
        db = self._load_database()
        return list(db["files"].values())

    def delete_file(self, file_stem: str, delete_stored: bool = False) -> bool:
        """
        Delete file from database.

        Args:
            file_stem: File stem
            delete_stored: Whether to delete stored files

        Returns:
            True if successful
        """
        db = self._load_database()

        if file_stem not in db["files"]:
            return False

        # Optionally delete stored files
        if delete_stored:
            file_dir = self.text_files_dir / file_stem
            if file_dir.exists():
                try:
                    shutil.rmtree(file_dir)
                    self.logger.info(f"Deleted stored files for: {file_stem}")
                except Exception as e:
                    self.logger.error(f"Failed to delete stored files: {e}")

        # Remove from database
        del db["files"][file_stem]
        self._save_database(db)

        self.logger.info(f"Removed file from database: {file_stem}")
        return True

    def verify_file_integrity(self, file_stem: str) -> bool:
        """
        Verify stored file matches original hash.

        Args:
            file_stem: File stem

        Returns:
            True if integrity check passes
        """
        file_entry = self.get_file(file_stem)

        if not file_entry:
            return False

        stored_path = Path(file_entry["stored_path"])

        if not stored_path.exists():
            self.logger.error(f"Stored file not found: {stored_path}")
            return False

        current_hash = self._compute_file_hash(stored_path)
        stored_hash = file_entry["file_hash"]

        if current_hash != stored_hash:
            self.logger.error(
                f"File integrity check failed for {file_stem}: "
                f"expected {stored_hash}, got {current_hash}"
            )
            return False

        return True

    def get_file_for_processing(self, source_file: Path) -> Path:
        """
        Get file path for processing, adding to database if needed.

        Args:
            source_file: Original source file path

        Returns:
            Path to use for processing (stored copy)
        """
        source_file = Path(source_file)
        file_stem = source_file.stem

        # Check if file already in database
        file_entry = self.get_file(file_stem)

        if file_entry:
            stored_path = Path(file_entry["stored_path"])

            # Verify integrity
            if self.verify_file_integrity(file_stem):
                self.logger.info(f"Using stored file: {stored_path}")
                return stored_path
            else:
                self.logger.warning(f"Stored file corrupted, re-adding: {file_stem}")
                # Re-add file
                self.delete_file(file_stem, delete_stored=True)

        # Add file to database
        file_entry = self.add_file(source_file, copy_file=True)

        if not file_entry:
            # Fall back to original file if copy fails
            self.logger.warning(f"Using original file: {source_file}")
            return source_file

        return Path(file_entry["stored_path"])
