"""
Report generation system for chunks, checkpoints, and sessions.

Generates comprehensive JSON reports containing metadata, statistics,
and processing information at multiple levels of granularity.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import time

from ..utils.logger import ContextLogger


class ReportGenerator:
    """
    Generates comprehensive reports at chunk, checkpoint, and session levels.
    """

    def __init__(self, logger: Optional[ContextLogger] = None):
        """
        Initialize report generator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or ContextLogger("ReportGenerator")

    def _compute_text_hash(self, text: str) -> str:
        """
        Compute SHA256 hash of text content.

        Args:
            text: Text content

        Returns:
            Hex digest of text hash
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def generate_chunk_report(
        self,
        chunk_index: int,
        text_content: str,
        audio_settings: Dict[str, Any],
        generation_start_time: float,
        generation_end_time: float,
        token_count: int,
        errors: List[str] = None,
        transcript_comparison: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a chunk-level report.

        Args:
            chunk_index: Index of chunk
            text_content: Original text content
            audio_settings: Audio generation settings
            generation_start_time: Start timestamp
            generation_end_time: End timestamp
            token_count: Number of tokens in chunk
            errors: List of error messages
            transcript_comparison: Transcript comparison results

        Returns:
            Chunk report dictionary
        """
        generation_time = generation_end_time - generation_start_time

        report = {
            "report_type": "chunk",
            "report_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "chunk_index": chunk_index,
            "text_content": text_content,
            "text_hash": self._compute_text_hash(text_content),
            "text_length_chars": len(text_content),
            "token_count": token_count,
            "audio_settings": audio_settings,
            "timing": {
                "generation_start": datetime.fromtimestamp(generation_start_time).isoformat(),
                "generation_end": datetime.fromtimestamp(generation_end_time).isoformat(),
                "generation_time_seconds": round(generation_time, 2),
                "tokens_per_second": round(token_count / generation_time, 2) if generation_time > 0 else 0
            },
            "errors": errors or [],
            "transcript_comparison": transcript_comparison
        }

        return report

    def save_chunk_report(
        self,
        report: Dict[str, Any],
        output_path: Path
    ) -> bool:
        """
        Save chunk report to file.

        Args:
            report: Chunk report dictionary
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.debug(f"Saved chunk report: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save chunk report: {e}")
            return False

    def generate_checkpoint_report(
        self,
        checkpoint_name: str,
        checkpoint_metadata: Dict[str, Any],
        chunk_reports: List[Dict[str, Any]],
        partial_progress_percent: float
    ) -> Dict[str, Any]:
        """
        Generate a checkpoint-level report.

        Args:
            checkpoint_name: Name of checkpoint
            checkpoint_metadata: Checkpoint metadata
            chunk_reports: List of chunk report summaries
            partial_progress_percent: Progress percentage

        Returns:
            Checkpoint report dictionary
        """
        total_tokens = sum(chunk.get("token_count", 0) for chunk in chunk_reports)
        total_time = sum(
            chunk.get("timing", {}).get("generation_time_seconds", 0)
            for chunk in chunk_reports
        )
        total_errors = sum(len(chunk.get("errors", [])) for chunk in chunk_reports)

        report = {
            "report_type": "checkpoint",
            "report_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "checkpoint_name": checkpoint_name,
            "checkpoint_metadata": checkpoint_metadata,
            "statistics": {
                "chunks_processed": len(chunk_reports),
                "total_tokens": total_tokens,
                "total_generation_time_seconds": round(total_time, 2),
                "average_tokens_per_chunk": round(total_tokens / len(chunk_reports), 2) if chunk_reports else 0,
                "total_errors": total_errors
            },
            "partial_progress_percent": round(partial_progress_percent, 2),
            "chunk_summaries": [
                {
                    "chunk_index": chunk.get("chunk_index"),
                    "token_count": chunk.get("token_count"),
                    "generation_time_seconds": chunk.get("timing", {}).get("generation_time_seconds"),
                    "has_errors": len(chunk.get("errors", [])) > 0,
                    "text_hash": chunk.get("text_hash")
                }
                for chunk in chunk_reports
            ]
        }

        return report

    def save_checkpoint_report(
        self,
        report: Dict[str, Any],
        output_path: Path
    ) -> bool:
        """
        Save checkpoint report to file.

        Args:
            report: Checkpoint report dictionary
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.debug(f"Saved checkpoint report: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint report: {e}")
            return False

    def generate_session_report(
        self,
        session_id: str,
        session_data: Dict[str, Any],
        pipeline_config: Dict[str, Any],
        all_stages: List[str],
        checkpoint_reports: List[Dict[str, Any]],
        final_statistics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a session-level report.

        Args:
            session_id: Session identifier
            session_data: Session metadata
            pipeline_config: Complete pipeline configuration
            all_stages: List of stages executed
            checkpoint_reports: List of checkpoint report summaries
            final_statistics: Final processing statistics

        Returns:
            Session report dictionary
        """
        total_chunks = sum(
            cp.get("statistics", {}).get("chunks_processed", 0)
            for cp in checkpoint_reports
        )
        total_tokens = sum(
            cp.get("statistics", {}).get("total_tokens", 0)
            for cp in checkpoint_reports
        )
        total_time = sum(
            cp.get("statistics", {}).get("total_generation_time_seconds", 0)
            for cp in checkpoint_reports
        )

        report = {
            "report_type": "session",
            "report_version": "1.0",
            "generated_at": datetime.now().isoformat(),
            "session_id": session_id,
            "session_data": session_data,
            "pipeline_configuration": pipeline_config,
            "stages_executed": all_stages,
            "statistics": {
                "total_checkpoints": len(checkpoint_reports),
                "total_chunks": total_chunks,
                "total_tokens": total_tokens,
                "total_generation_time_seconds": round(total_time, 2),
                "total_time_formatted": self._format_duration(total_time),
                "average_tokens_per_second": round(total_tokens / total_time, 2) if total_time > 0 else 0,
                **final_statistics
            },
            "checkpoint_summaries": [
                {
                    "checkpoint_name": cp.get("checkpoint_name"),
                    "chunks_processed": cp.get("statistics", {}).get("chunks_processed"),
                    "total_tokens": cp.get("statistics", {}).get("total_tokens"),
                    "generation_time_seconds": cp.get("statistics", {}).get("total_generation_time_seconds"),
                    "progress_percent": cp.get("partial_progress_percent")
                }
                for cp in checkpoint_reports
            ]
        }

        return report

    def save_session_report(
        self,
        report: Dict[str, Any],
        output_path: Path
    ) -> bool:
        """
        Save session report to file.

        Args:
            report: Session report dictionary
            output_path: Output file path

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            self.logger.info(f"Saved session report: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save session report: {e}")
            return False

    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted string (e.g., "01:23:45")
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def load_report(self, report_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a report from file.

        Args:
            report_path: Path to report file

        Returns:
            Report dictionary or None on failure
        """
        try:
            with open(report_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load report {report_path}: {e}")
            return None

    def aggregate_chunk_reports(
        self,
        chunk_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Load and aggregate all chunk reports in a directory.

        Args:
            chunk_dir: Directory containing chunk reports

        Returns:
            List of chunk reports
        """
        chunk_reports = []

        if not chunk_dir.exists():
            return chunk_reports

        for report_file in chunk_dir.glob("*_report.json"):
            report = self.load_report(report_file)
            if report:
                chunk_reports.append(report)

        # Sort by chunk index
        chunk_reports.sort(key=lambda x: x.get("chunk_index", 0))

        return chunk_reports

    def aggregate_checkpoint_reports(
        self,
        session_dir: Path
    ) -> List[Dict[str, Any]]:
        """
        Load and aggregate all checkpoint reports in a session directory.

        Args:
            session_dir: Session directory

        Returns:
            List of checkpoint reports
        """
        checkpoint_reports = []

        if not session_dir.exists():
            return checkpoint_reports

        for checkpoint_dir in session_dir.iterdir():
            if checkpoint_dir.is_dir() and checkpoint_dir.name.startswith('checkpoint'):
                report_file = checkpoint_dir / f"{checkpoint_dir.name}_report.json"
                if report_file.exists():
                    report = self.load_report(report_file)
                    if report:
                        checkpoint_reports.append(report)

        return checkpoint_reports
