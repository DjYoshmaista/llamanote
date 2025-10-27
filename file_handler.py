"""
File Handler Module
Manages file input/output operations and batch processing
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import shutil
from dataclasses import dataclass, asdict

from logging_config import get_logger, log_execution_time
from config import (
    OUTPUT_DIR,
    CACHE_DIR,
    SUPPORTED_FORMATS,
    OUTPUT_FORMAT_OPTIONS,
    DEFAULT_OUTPUT_FORMAT,
    TIMESTAMP_OUTPUTS,
    INCLUDE_METADATA
)

logger = get_logger(__name__)


@dataclass
class ProcessedFile:
    """Container for processed file information"""
    input_path: Path
    output_path: Path
    format: str
    timestamp: str
    processing_time: float
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class FileHandler:
    """Handles file operations for the processing pipeline"""
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 cache_dir: Optional[Path] = None,
                 timestamp_outputs: bool = TIMESTAMP_OUTPUTS,
                 create_backups: bool = True):
        """
        Initialize file handler
        
        Args:
            output_dir: Output directory for processed files
            cache_dir: Cache directory for intermediate files
            timestamp_outputs: Whether to add timestamps to output filenames
            create_backups: Whether to create backups before overwriting
        """
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.cache_dir = Path(cache_dir or CACHE_DIR)
        self.timestamp_outputs = timestamp_outputs
        self.create_backups = create_backups
        
        # Create directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processed files
        self.processed_files: List[ProcessedFile] = []
        
        logger.info(f"Initialized FileHandler (output_dir={self.output_dir})")
        
    def get_output_path(self,
                       input_path: Path,
                       suffix: str = "_processed",
                       extension: str = ".md") -> Path:
        """
        Generate output path for a processed file
        
        Args:
            input_path: Original input file path
            suffix: Suffix to add to filename
            extension: New file extension
            
        Returns:
            Output file path
        """
        input_path = Path(input_path)
        
        # Base filename without extension
        base_name = input_path.stem
        
        # Add timestamp if enabled
        if self.timestamp_outputs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"{base_name}{suffix}_{timestamp}{extension}"
        else:
            output_name = f"{base_name}{suffix}{extension}"
            
        output_path = self.output_dir / output_name
        
        # Handle existing files
        if output_path.exists() and not self.timestamp_outputs:
            if self.create_backups:
                self._create_backup(output_path)
            else:
                # Generate unique name
                counter = 1
                while output_path.exists():
                    output_name = f"{base_name}{suffix}_{counter}{extension}"
                    output_path = self.output_dir / output_name
                    counter += 1
                    
        return output_path
        
    def _create_backup(self, file_path: Path):
        """Create a backup of an existing file"""
        if not file_path.exists():
            return
            
        backup_dir = self.output_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
        
    @log_execution_time()
    def save_text(self,
                 text: str,
                 output_path: Path,
                 format: str = DEFAULT_OUTPUT_FORMAT,
                 metadata: Optional[Dict[str, Any]] = None) -> Path:
        """
        Save text in specified format
        
        Args:
            text: Text to save
            output_path: Output file path
            format: Output format (markdown, text, json, html)
            metadata: Optional metadata to include
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        
        try:
            if format == "markdown" or format == "md":
                content = self._format_as_markdown(text, metadata)
                output_path = output_path.with_suffix('.md')
                
            elif format == "text" or format == "txt":
                content = text
                output_path = output_path.with_suffix('.txt')
                
            elif format == "json":
                content = json.dumps({
                    "text": text,
                    "metadata": metadata or {},
                    "timestamp": datetime.now().isoformat()
                }, indent=2)
                output_path = output_path.with_suffix('.json')
                
            elif format == "html":
                content = self._format_as_html(text, metadata)
                output_path = output_path.with_suffix('.html')
                
            else:
                logger.warning(f"Unknown format '{format}', using text")
                content = text
                output_path = output_path.with_suffix('.txt')
                
            # Write file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info(f"Saved {format} file: {output_path.name}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise
            
    def _format_as_markdown(self, text: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Format text as markdown with optional metadata"""
        content = []
        
        if INCLUDE_METADATA and metadata:
            content.append("---")
            content.append("# Document Metadata")
            for key, value in metadata.items():
                if not key.startswith('_'):  # Skip private keys
                    content.append(f"- **{key}**: {value}")
            content.append("---")
            content.append("")
            
        content.append(text)
        
        return '\n'.join(content)
        
    def _format_as_html(self, text: str, metadata: Optional[Dict[str, Any]]) -> str:
        """Format text as HTML"""
        # Convert markdown to HTML if needed
        html_content = text.replace('\n', '<br>\n')
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Processed Document</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            color: #333;
        }}
        .metadata {{
            background: #f5f5f5;
            padding: 1rem;
            border-radius: 5px;
            margin-bottom: 2rem;
        }}
        .metadata h2 {{
            margin-top: 0;
        }}
        .metadata dl {{
            display: grid;
            grid-template-columns: auto 1fr;
            gap: 0.5rem 1rem;
        }}
        .metadata dt {{
            font-weight: bold;
        }}
        .content {{
            white-space: pre-wrap;
        }}
    </style>
</head>
<body>
"""
        
        if INCLUDE_METADATA and metadata:
            html += """
    <div class="metadata">
        <h2>Document Metadata</h2>
        <dl>
"""
            for key, value in metadata.items():
                if not key.startswith('_'):
                    html += f"            <dt>{key}:</dt><dd>{value}</dd>\n"
            html += """        </dl>
    </div>
"""
        
        html += f"""
    <div class="content">
{html_content}
    </div>
</body>
</html>"""
        
        return html
        
    def save_checkpoint(self, 
                       data: Any,
                       checkpoint_name: str,
                       stage: str) -> Path:
        """
        Save a processing checkpoint
        
        Args:
            data: Data to checkpoint
            checkpoint_name: Name for the checkpoint
            stage: Processing stage name
            
        Returns:
            Path to checkpoint file
        """
        checkpoint_dir = self.cache_dir / "checkpoints" / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_file = checkpoint_dir / f"{stage}_{timestamp}.pkl"
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
            
        logger.debug(f"Saved checkpoint: {checkpoint_file.name}")
        return checkpoint_file
        
    def load_checkpoint(self,
                       checkpoint_name: str,
                       stage: str,
                       latest: bool = True) -> Optional[Any]:
        """
        Load a processing checkpoint
        
        Args:
            checkpoint_name: Name of the checkpoint
            stage: Processing stage name
            latest: Whether to load the latest checkpoint
            
        Returns:
            Checkpoint data or None if not found
        """
        checkpoint_dir = self.cache_dir / "checkpoints" / checkpoint_name
        
        if not checkpoint_dir.exists():
            logger.debug(f"No checkpoint directory: {checkpoint_dir}")
            return None
            
        # Find checkpoint files for this stage
        pattern = f"{stage}_*.pkl"
        checkpoint_files = list(checkpoint_dir.glob(pattern))
        
        if not checkpoint_files:
            logger.debug(f"No checkpoints found for stage: {stage}")
            return None
            
        # Get the latest or first checkpoint
        if latest:
            checkpoint_file = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        else:
            checkpoint_file = checkpoint_files[0]
            
        try:
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logger.debug(f"Loaded checkpoint: {checkpoint_file.name}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
            
    def cleanup_old_files(self, days: int = 7):
        """
        Clean up old files from output and cache directories
        
        Args:
            days: Remove files older than this many days
        """
        import time
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Clean cache directory
        cleaned_count = 0
        for cache_file in self.cache_dir.rglob('*'):
            if cache_file.is_file() and cache_file.stat().st_mtime < cutoff_time:
                try:
                    cache_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete {cache_file}: {e}")
                    
        # Clean old backups
        backup_dir = self.output_dir / "backups"
        if backup_dir.exists():
            for backup_file in backup_dir.glob('*'):
                if backup_file.is_file() and backup_file.stat().st_mtime < cutoff_time:
                    try:
                        backup_file.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Could not delete {backup_file}: {e}")
                        
        logger.info(f"Cleaned up {cleaned_count} old files")
        
    def record_processed_file(self,
                             input_path: Path,
                             output_path: Path,
                             format: str,
                             processing_time: float,
                             metadata: Dict[str, Any],
                             success: bool = True,
                             error_message: Optional[str] = None):
        """Record information about a processed file"""
        processed = ProcessedFile(
            input_path=Path(input_path),
            output_path=Path(output_path),
            format=format,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time,
            metadata=metadata,
            success=success,
            error_message=error_message
        )
        
        self.processed_files.append(processed)
        
    def save_processing_report(self, report_path: Optional[Path] = None) -> Path:
        """
        Save a report of all processed files
        
        Args:
            report_path: Path for the report file
            
        Returns:
            Path to saved report
        """
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = self.output_dir / f"processing_report_{timestamp}.json"
            
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_files": len(self.processed_files),
            "successful": sum(1 for f in self.processed_files if f.success),
            "failed": sum(1 for f in self.processed_files if not f.success),
            "total_processing_time": sum(f.processing_time for f in self.processed_files),
            "files": [asdict(f) for f in self.processed_files]
        }
        
        # Convert Path objects to strings for JSON serialization
        for file_info in report_data["files"]:
            file_info["input_path"] = str(file_info["input_path"])
            file_info["output_path"] = str(file_info["output_path"])
            
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Saved processing report: {report_path}")
        return report_path


class BatchFileManager:
    """Manage batch processing of multiple files"""
    
    def __init__(self, file_handler: Optional[FileHandler] = None):
        """
        Initialize batch file manager
        
        Args:
            file_handler: FileHandler instance to use
        """
        self.file_handler = file_handler or FileHandler()
        self.logger = get_logger(f"{__name__}.BatchManager")
        
    def collect_input_files(self,
                           input_paths: Union[List[Path], Path],
                           recursive: bool = False,
                           pattern: str = "*") -> List[Path]:
        """
        Collect all input files to process
        
        Args:
            input_paths: List of paths or single path (file or directory)
            recursive: Whether to search directories recursively
            pattern: File pattern to match
            
        Returns:
            List of file paths to process
        """
        if isinstance(input_paths, (str, Path)):
            input_paths = [Path(input_paths)]
        else:
            input_paths = [Path(p) for p in input_paths]
            
        files = []
        
        for path in input_paths:
            if path.is_file():
                # Single file
                if self._is_supported_format(path):
                    files.append(path)
                else:
                    self.logger.warning(f"Unsupported format: {path.suffix}")
                    
            elif path.is_dir():
                # Directory
                if recursive:
                    found_files = path.rglob(pattern)
                else:
                    found_files = path.glob(pattern)
                    
                for f in found_files:
                    if f.is_file() and self._is_supported_format(f):
                        files.append(f)
            else:
                self.logger.warning(f"Path not found: {path}")
                
        # Remove duplicates while preserving order
        seen = set()
        unique_files = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique_files.append(f)
                
        self.logger.info(f"Collected {len(unique_files)} files to process")
        return unique_files
        
    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported"""
        return file_path.suffix.lower() in SUPPORTED_FORMATS
        
    def create_batch_manifest(self, 
                             files: List[Path],
                             manifest_path: Optional[Path] = None) -> Path:
        """
        Create a manifest file for batch processing
        
        Args:
            files: List of files to include
            manifest_path: Path for manifest file
            
        Returns:
            Path to manifest file
        """
        if manifest_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            manifest_path = self.file_handler.cache_dir / f"batch_manifest_{timestamp}.json"
            
        manifest_data = {
            "created": datetime.now().isoformat(),
            "total_files": len(files),
            "files": [
                {
                    "path": str(f),
                    "size_mb": f.stat().st_size / (1024 * 1024),
                    "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                }
                for f in files
            ]
        }
        
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
            
        self.logger.info(f"Created batch manifest: {manifest_path}")
        return manifest_path
        
    def load_batch_manifest(self, manifest_path: Path) -> List[Path]:
        """
        Load files from a batch manifest
        
        Args:
            manifest_path: Path to manifest file
            
        Returns:
            List of file paths
        """
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
                
            files = [Path(f["path"]) for f in manifest_data["files"]]
            self.logger.info(f"Loaded {len(files)} files from manifest")
            return files
            
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            return []
