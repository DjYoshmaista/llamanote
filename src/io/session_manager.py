"""
Session tracking and management system.

A session represents a distinct processing attempt. Resuming from a checkpoint
continues the same session.
"""

import json
import uuid
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from ..utils.logger import ContextLogger


class SessionManager:
    """
    Manages processing sessions.

    A session represents a complete processing run from start to finish.
    Resuming from a checkpoint continues the same session.
    """

    def __init__(
        self,
        sessions_db_path: Path,
        logger: Optional[ContextLogger] = None
    ):
        """
        Initialize session manager.

        Args:
            sessions_db_path: Path to sessions database file
            logger: Optional logger instance
        """
        self.sessions_db_path = Path(sessions_db_path)
        self.logger = logger or ContextLogger("SessionManager")
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Ensure sessions database file exists."""
        if not self.sessions_db_path.exists():
            self.sessions_db_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_db({"sessions": {}})

    def _load_db(self) -> Dict[str, Any]:
        """Load sessions database."""
        try:
            with open(self.sessions_db_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load sessions database: {e}")
            return {"sessions": {}}

    def _save_db(self, data: Dict[str, Any]):
        """Save sessions database."""
        try:
            with open(self.sessions_db_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save sessions database: {e}")

    def create_session(
        self,
        input_file: Path,
        config: Dict[str, Any]
    ) -> str:
        """
        Create a new session.

        Args:
            input_file: Input file being processed
            config: Pipeline configuration

        Returns:
            Unique session identifier
        """
        session_id = f"session{uuid.uuid4().hex[:8]}"

        db = self._load_db()

        db["sessions"][session_id] = {
            "session_id": session_id,
            "input_file": str(input_file),
            "input_stem": input_file.stem,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "active",
            "config": config,
            "checkpoints": [],
            "statistics": {
                "total_chunks": 0,
                "total_tokens": 0,
                "total_time_seconds": 0,
                "memory_peak_mb": 0
            }
        }

        self._save_db(db)
        self.logger.info(f"Created new session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session information.

        Args:
            session_id: Session identifier

        Returns:
            Session data or None if not found
        """
        db = self._load_db()
        return db["sessions"].get(session_id)

    def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any]
    ):
        """
        Update session information.

        Args:
            session_id: Session identifier
            updates: Dictionary of updates to apply
        """
        db = self._load_db()

        if session_id not in db["sessions"]:
            self.logger.error(f"Session not found: {session_id}")
            return

        # Update timestamp
        updates["updated_at"] = datetime.now().isoformat()

        # Apply updates
        db["sessions"][session_id].update(updates)

        self._save_db(db)
        self.logger.debug(f"Updated session: {session_id}")

    def add_checkpoint_to_session(
        self,
        session_id: str,
        checkpoint_name: str,
        checkpoint_metadata: Dict[str, Any]
    ):
        """
        Add a checkpoint to session history.

        Args:
            session_id: Session identifier
            checkpoint_name: Checkpoint name
            checkpoint_metadata: Checkpoint metadata
        """
        db = self._load_db()

        if session_id not in db["sessions"]:
            self.logger.error(f"Session not found: {session_id}")
            return

        checkpoint_entry = {
            "name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "metadata": checkpoint_metadata
        }

        db["sessions"][session_id]["checkpoints"].append(checkpoint_entry)
        db["sessions"][session_id]["updated_at"] = datetime.now().isoformat()

        self._save_db(db)
        self.logger.debug(f"Added checkpoint {checkpoint_name} to session {session_id}")

    def update_session_statistics(
        self,
        session_id: str,
        stats: Dict[str, Any]
    ):
        """
        Update session statistics.

        Args:
            session_id: Session identifier
            stats: Statistics to update
        """
        db = self._load_db()

        if session_id not in db["sessions"]:
            self.logger.error(f"Session not found: {session_id}")
            return

        db["sessions"][session_id]["statistics"].update(stats)
        db["sessions"][session_id]["updated_at"] = datetime.now().isoformat()

        self._save_db(db)

    def complete_session(
        self,
        session_id: str,
        final_output_path: Optional[Path] = None
    ):
        """
        Mark session as completed.

        Args:
            session_id: Session identifier
            final_output_path: Path to final output file
        """
        updates = {
            "status": "completed",
            "completed_at": datetime.now().isoformat()
        }

        if final_output_path:
            updates["final_output"] = str(final_output_path)

        self.update_session(session_id, updates)
        self.logger.info(f"Completed session: {session_id}")

    def fail_session(
        self,
        session_id: str,
        error_message: str
    ):
        """
        Mark session as failed.

        Args:
            session_id: Session identifier
            error_message: Error message
        """
        updates = {
            "status": "failed",
            "failed_at": datetime.now().isoformat(),
            "error": error_message
        }

        self.update_session(session_id, updates)
        self.logger.error(f"Failed session {session_id}: {error_message}")

    def list_sessions_for_file(self, input_stem: str) -> list:
        """
        List all sessions for a given input file.

        Args:
            input_stem: Input file stem (without extension)

        Returns:
            List of session data
        """
        db = self._load_db()

        return [
            session for session in db["sessions"].values()
            if session["input_stem"] == input_stem
        ]

    def get_latest_session_for_file(self, input_stem: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent session for a given input file.

        Args:
            input_stem: Input file stem

        Returns:
            Latest session data or None
        """
        sessions = self.list_sessions_for_file(input_stem)

        if not sessions:
            return None

        # Sort by created_at timestamp
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        return sessions[0]

    def get_session_from_checkpoint(
        self,
        checkpoint_metadata: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract session ID from checkpoint metadata.

        Args:
            checkpoint_metadata: Checkpoint metadata

        Returns:
            Session ID if found, None otherwise
        """
        return checkpoint_metadata.get("session_id")

    def list_all_sessions(self) -> list:
        """
        List all sessions in database.

        Returns:
            List of all session data
        """
        db = self._load_db()
        return list(db["sessions"].values())

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session from the database.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        db = self._load_db()

        if session_id in db["sessions"]:
            del db["sessions"][session_id]
            self._save_db(db)
            self.logger.info(f"Deleted session: {session_id}")
            return True

        return False

    def generate_session_id_from_checkpoint(
        self,
        checkpoint_metadata: Dict[str, Any],
        input_stem: str
    ) -> str:
        """
        Generate or retrieve session ID for resuming.

        If checkpoint has session_id, use it (continue session).
        Otherwise, create new session.

        Args:
            checkpoint_metadata: Checkpoint metadata
            input_stem: Input file stem

        Returns:
            Session ID to use
        """
        # Check if checkpoint has existing session ID
        existing_session_id = checkpoint_metadata.get("session_id")

        if existing_session_id:
            session = self.get_session(existing_session_id)
            if session:
                self.logger.info(f"Resuming session: {existing_session_id}")
                return existing_session_id

        # Create new session ID if none exists or old one not found
        session_id = f"session{uuid.uuid4().hex[:8]}"
        self.logger.info(f"Created new session for resume: {session_id}")
        return session_id
