# llamanote/io/batch_manager.py
"""
Batch File Manager Module
Handles discovery and management of files for batch processing.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime

from ..utils.logger import get_logger_conf
from ..utils.validators import validate_file_path
from ..config.settings import SUPPORTED_FORMATS

logger = get_logger_conf(__name__)

class BatchFileManager:
    """Manages finding and listing files for batch operations."""

    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize batch file manager
        
        Args:
            supported_formats: List of supported file extensions (e.g., ['.pdf', '.txt'])
        """
        self.supported_formats = supported_formats or SUPPORTED_FORMATS
        self.logger = get_logger_conf(f"{__name__}.BatchManager")

    def collect_input_files(self,
                           input_paths: Union[List[str], List[Path]],
                           recursive: bool = False) -> List[Path]:
        """
        Collects all valid input files from a list of paths (files or directories).

        Args:
            input_paths: List of string or Path objects.
            recursive: Whether to search directories recursively.

        Returns:
            A list of unique, valid file Paths.
        """
        resolved_paths = [Path(os.path.expanduser(p)).resolve() for p in input_paths]
        
        files_to_process = []
        seen_paths = set()

        for path in resolved_paths:
            if path in seen_paths: # Avoid processing the same path twice
                continue
                
            if path.is_file():
                is_valid, msg = validate_file_path(
                    path,
                    check_existence=True,
                    allowed_extensions=self.supported_formats
                )
                if is_valid:
                    if path not in seen_paths:
                        files_to_process.append(path)
                        seen_paths.add(path)
                else:
                    self.logger.warning(f"Skipping invalid file: {path.name} ({msg})")
                    
            elif path.is_dir():
                self.logger.info(f"Scanning directory: {path} (Recursive={recursive})")
                search_pattern = "**/*" if recursive else "*"
                
                for f in path.glob(search_pattern):
                    if f.is_file():
                         is_valid, msg = validate_file_path(
                             f,
                             check_existence=True, # Already know it exists, but good practice
                             allowed_extensions=self.supported_formats
                         )
                         if is_valid:
                             if f not in seen_paths:
                                 files_to_process.append(f)
                                 seen_paths.add(f)
                         else:
                             # Log only if it's a *potentially* supported type but invalid
                             if f.suffix.lower() in self.supported_formats:
                                 self.logger.warning(f"Skipping invalid file: {f.name} ({msg})")
            else:
                self.logger.warning(f"Input path not found or is not a file/directory: {path}")

        self.logger.info(f"Collected {len(files_to_process)} valid files to process.")
        return sorted(list(files_to_process)) # Return a sorted list

    def create_batch_manifest(self,
                             files: List[Path],
                             manifest_path: Path) -> bool:
        """
        Create a JSON manifest file for a list of files.
        
        Args:
            files: List of file paths to include.
            manifest_path: Path to save the manifest file.
            
        Returns:
            True if manifest was saved successfully, False otherwise.
        """
        try:
            manifest_data = {
                "created_at": datetime.now().isoformat(),
                "file_count": len(files),
                "files": [
                    {
                        "path": str(f.resolve()),
                        "name": f.name,
                        "size_bytes": f.stat().st_size,
                        "modified_time": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    }
                    for f in files if f.exists()
                ]
            }
            
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2)
                
            self.logger.info(f"Created batch manifest: {manifest_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create batch manifest: {e}", exc_info=True)
            return False
        
    def load_batch_manifest(self, manifest_path: Path) -> List[Path]:
        """
        Load a list of file paths from a batch manifest.
        
        Args:
            manifest_path: Path to manifest file.
            
        Returns:
            List of file paths (as Path objects).
        """
        if not manifest_path.is_file():
            self.logger.error(f"Manifest file not found: {manifest_path}")
            return []
            
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest_data = json.load(f)
                
            files = [Path(f["path"]) for f in manifest_data.get("files", [])]
            # Optional: Add check for file existence
            existing_files = [f for f in files if f.exists()]
            if len(existing_files) != len(files):
                 self.logger.warning(f"Loaded {len(existing_files)} files from manifest. {len(files) - len(existing_files)} files were missing.")
            else:
                 self.logger.info(f"Loaded {len(existing_files)} files from manifest.")
            return existing_files
            
        except json.JSONDecodeError as e:
             self.logger.error(f"Failed to parse manifest file (invalid JSON): {manifest_path} - {e}")
             return []
        except KeyError:
             self.logger.error(f"Failed to load manifest: File format is incorrect (missing 'files' key).")
             return []
        except Exception as e:
            self.logger.error(f"Failed to load manifest {manifest_path}: {e}", exc_info=True)
            return []
