# llamanote/io/file_handler.py
"""
File Handler Module
Manages file input/output operations, path generation, backups, and reporting.
"""

import json
import pickle
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
import time

from ..utils.logger import get_logger_conf
from ..utils.decorators import log_execution_time
from ..utils.helpers import PathGenerator # Import refactored helper
from ..utils.validators import validate_directory_path
from ..config.settings import (
    DEFAULT_OUTPUT_DIR, DEFAULT_CACHE_DIR, OUTPUT_FORMAT_OPTIONS,
    DEFAULT_OUTPUT_FORMAT, TIMESTAMP_OUTPUTS, INCLUDE_METADATA
)
# Import core types
from ..core.types import PipelineResult, PDFMetadata, ProcessedFile

logger = get_logger_conf(__name__)


# --- Metadata Formatting (Refactoring Item 2) ---
class MetadataFormatter:
    """Helper class to format metadata into different output types."""

    @staticmethod
    def format_metadata(metadata: Dict[str, Any], format_type: str) -> Optional[str]:
        """
        Formats metadata for inclusion in output files.

        Args:
            metadata: Key-value metadata dictionary.
            format_type: 'markdown', 'html', or 'json'.

        Returns:
            Formatted string or None if no metadata or format is unsupported.
        """
        if not metadata:
            return None
        
        # Filter out private keys
        clean_metadata = {k: v for k, v in metadata.items() if not k.startswith('_')}
        if not clean_metadata:
             return None
        
        if format_type in ["markdown", "md"]:
            lines = ["---", "# Document Metadata"]
            for key, value in clean_metadata.items():
                lines.append(f"- **{key}**: {value}")
            lines.append("---")
            return "\n".join(lines)
            
        elif format_type == "html":
            lines = ['<div class="metadata">', '<h2>Document Metadata</h2>', '<dl>']
            for key, value in clean_metadata.items():
                lines.append(f"    <dt>{key}:</dt><dd>{value}</dd>")
            lines.append("</dl>")
            lines.append("</div>")
            return "\n".join(lines)
            
        elif format_type == "json":
            # Metadata is typically included as a key in the JSON,
            # not formatted as a string *within* the JSON text content.
            # This method is for formatting *into* the text body.
            # For JSON output, the save_text method should handle it.
            return None # Or maybe a JSON string representation?
            
        return None # Unsupported format for metadata embedding

# --- File Handler Class ---

class FileHandler:
    """Handles file saving, path generation, backups, and reporting."""
    
    def __init__(self,
                 output_dir: Optional[Union[str, Path]] = None,
                 cache_dir: Optional[Union[str, Path]] = None,
                 timestamp_outputs: bool = TIMESTAMP_OUTPUTS,
                 create_backups: bool = True):
        
        try:
            self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
            self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).resolve()
        except Exception as e:
             logger.error(f"Invalid path provided to FileHandler: {e}. Using defaults.", exc_info=True)
             self.output_dir = DEFAULT_OUTPUT_DIR.resolve()
             self.cache_dir = DEFAULT_CACHE_DIR.resolve()

        # Ensure directories exist
        try:
             self.output_dir.mkdir(parents=True, exist_ok=True)
             self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
             logger.error(f"Could not create necessary directories: {e}. File operations may fail.")
             # Depending on severity, we might want to raise this
        
        self.timestamp_outputs = timestamp_outputs
        self.path_generator = PathGenerator(
            base_output_dir=self.output_dir,
            timestamp_files=timestamp_outputs,
            create_backups=create_backups
        )
        
        self.processed_files: List[ProcessedFile] = []
        logger.info(f"FileHandler initialized (Output: {self.output_dir}, Cache: {self.cache_dir})")
        
    def get_output_path(self,
                       input_path: Path,
                       suffix: str = "_processed",
                       extension: str = ".md",
                       sub_dir: Optional[str] = None,
                       base_path_override: Optional[Path] = None) -> Path:
        """
        Generate output path for a processed file using the PathGenerator.

        Args:
            input_path: Original input file path (used for default naming).
            suffix: Suffix to add to filename.
            extension: New file extension (e.g., ".md", ".txt").
            sub_dir: Optional subdirectory within the main output dir.
            base_path_override: If provided, use this as the base (file or dir).
                                If it's a directory, behavior is like output_dir.
                                If it's a file path, it's used as the target.
                                
        Returns:
            Output file path
        """
        
        if base_path_override:
            base_path_override = Path(base_path_override)
            if base_path_override.is_dir():
                 # If user provided a directory via -o, use that as the base_output_dir
                 # This isn't ideal usage of base_path_override, but handles CLI case
                 generator = PathGenerator(base_path_override, self.timestamp_outputs, self.path_generator.create_backups)
                 return generator.generate_output_path(input_path, suffix, extension, sub_dir, None)
            else:
                 # User provided a specific file path, check/backup it
                 output_path = self.path_generator.handle_existing_file(base_path_override)
                 return output_path
        
        # Default behavior: use instance's path_generator
        return self.path_generator.generate_output_path(
            input_path, suffix, extension, sub_dir, None
        )

    @log_execution_time()
    def save_text(self,
                 text: str,
                 output_path: Path, # This should be the *full* path now
                 format: str = DEFAULT_OUTPUT_FORMAT,
                 metadata: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        """
        Save text in specified format using a strategy-like approach.
        
        Args:
            text: Text to save
            output_path: The full, final path to save the file to (already unique)
            format: Output format (markdown, text, json, html)
            metadata: Optional metadata to include
            
        Returns:
            Path to saved file, or None on failure
        """
        format_lower = format.lower().lstrip('.')
        if format_lower not in OUTPUT_FORMAT_OPTIONS:
            logger.warning(f"Unknown format '{format_lower}', defaulting to 'text'")
            format_lower = "text"
            
        output_path = output_path.with_suffix(f".{format_lower}")
        
        content = ""
        try:
            # --- Strategy Pattern for Formatting ---
            if format_lower == "json":
                content = json.dumps({
                    "metadata": metadata or {},
                    "content": text,
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
            else:
                # Handle text-based formats (md, html, txt)
                if INCLUDE_METADATA and metadata:
                    meta_header = MetadataFormatter.format_metadata(metadata, format_lower)
                    if meta_header:
                        content += meta_header + "\n\n"
                
                # Add main content
                if format_lower == "html":
                     # Basic markdown-like conversion for HTML
                     body_text = text.replace('\n\n', '</p><p>').replace('\n', '<br>')
                     content += f"<div class=\"content\">\n<p>{body_text}</p>\n</div>"
                     # Wrap in basic HTML structure
                     content = self._wrap_html(content, metadata)
                else:
                    # For "markdown" and "text", the text is used as-is
                    content += text
            
            # --- File Writing ---
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved {format_lower} file: {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save file to {output_path}: {e}", exc_info=True)
            return None # Return None on failure
            
    def _wrap_html(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Wraps content in a basic HTML structure."""
        title = metadata.get("name", "Processed Document") if metadata else "Processed Document"
        # (Using similar CSS as original file_handler)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: sans-serif; line-height: 1.6; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }}
        .metadata {{ background: #f4f4f4; border: 1px solid #ddd; padding: 1rem; margin-bottom: 2rem; }}
        .metadata h2 {{ margin-top: 0; }}
        .metadata dl {{ display: grid; grid-template-columns: auto 1fr; gap: 0.5rem 1rem; }}
        .metadata dt {{ font-weight: bold; }}
        .content {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
{content}
</body>
</html>"""
        
    def record_processed_file(self, result: PipelineResult):
        """Records a completed PipelineResult."""
        if not result:
            return
            
        processed = ProcessedFile(
            input_path=result.input_file,
            output_path=result.output_file,
            format=result.statistics.get("output_format", "unknown"), # Get format from stats if available
            timestamp=datetime.now().isoformat(),
            processing_time=result.processing_time,
            metadata=result.statistics or {},
            success=result.success,
            error_message=result.error_message
        )
        self.processed_files.append(processed)
        
    def save_processing_report(self, report_path: Optional[Path] = None) -> Optional[Path]:
        """Saves a JSON report of all processed files in this session."""
        if not self.processed_files:
            logger.info("No files processed in this session, skipping report.")
            return None
            
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%M%S")
            report_path = self.output_dir / f"processing_report_{timestamp}.json"
            
        report_data = {
            "report_generated": datetime.now().isoformat(),
            "total_files": len(self.processed_files),
            "successful": sum(1 for f in self.processed_files if f.success),
            "failed": sum(1 for f in self.processed_files if not f.success),
            "total_processing_time": sum(f.processing_time for f in self.processed_files),
            "files": []
        }
        
        # Serialize ProcessedFile objects
        for f in self.processed_files:
            file_data = asdict(f)
            file_data["input_path"] = str(file_data["input_path"])
            file_data["output_path"] = str(file_data["output_path"]) if file_data["output_path"] else None
            report_data["files"].append(file_data)
            
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2)
            logger.info(f"Saved processing report: {report_path.name}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to save processing report: {e}", exc_info=True)
            return None
            
    def cleanup_old_files(self, days: int = 7):
        """Cleans up old cache files and backups."""
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Clean cache (checkpoints, etc.)
        self._cleanup_dir(self.cache_dir, cutoff_time)
        
        # Clean old backups
        backup_dir = self.output_dir / "backups"
        if backup_dir.exists():
            self._cleanup_dir(backup_dir, cutoff_time)
            
    def _cleanup_dir(self, directory: Path, cutoff_time: float):
        """Helper to recursively delete old files in a directory."""
        cleaned_count = 0
        try:
             for item in directory.rglob('*'):
                 if item.is_file() and item.stat().st_mtime < cutoff_time:
                     try:
                         item.unlink()
                         cleaned_count += 1
                     except Exception as e:
                         logger.warning(f"Could not delete old file {item}: {e}")
        except Exception as e:
             logger.error(f"Error during cleanup of {directory}: {e}")
             
        if cleaned_count > 0:
             logger.info(f"Cleaned up {cleaned_count} old files from {directory}")
