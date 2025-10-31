# llamanote/io/batch_manager.py
"""
Batch File Manager Module
Handles discovery and management of files for batch processing.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import os
import sys

from ..utils.logger import get_logger_conf
from ..utils.validators import validate_file_path

# Load SUPPORTED_FORMATS from environment variable
DEFAULT_SUPPORTED_FORMATS = ['.pdf', '.txt', '.md']
SUPPORTED_FORMATS_STR = os.getenv("SUPPORTED_FORMATS", ".pdf,.txt,.md")
SUPPORTED_FORMATS = [ext.strip() for ext in SUPPORTED_FORMATS_STR.split(',') if ext.strip()]
if not SUPPORTED_FORMATS:
    print("Warning: SUPPORTED_FORMATS from .env is empty or invalid. Usin defaults.", file=sys.stderr)
    SUPPORTED_FORMATS = DEFAULT_SUPPORTED_FORMATS

logger = get_logger_conf(__name__)

class BatchFileManager:
    """Manages finding and listing files for batch operations."""

    def __init__(self, supported_formats: Optional[List[str]] = None):
        """
        Initialize batch file manager
        
        Args:
            supported_formats: List of supported file extensions (e.g., ['.pdf', '.txt'])
                               If None, uses value loaded from environment/default values
        """
        # Use provided formats, otherwise use the lobally loaded ones
        self.supported_formats = supported_formats if supported_formats is not None else DEFAULT_SUPPORTED_FORMATS
        self.logger = get_logger_conf(f"{__name__}.BatchManager")
        # Ensure self.supported_formats is always a list
        if not isinstance(self.supported_formats, list):
            self.logger.warning(f"Invalid supported_formats provided:\n`{self.supported_formats}`\n")
            self.supported_formats = DEFAULT_SUPPORTED_FORMATS

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
        resolved_paths = []
        for p in input_paths:
            try:
                resolved_paths.append(Path(os.path.expanduser(p)).resolve())
            except Exception as e:
                self.logger.warning(f"Could not resolve path '{p}': {e}.  Skipping.")

        files_to_process = []
        seen_paths = set()

        for path in resolved_paths:
            if path in seen_paths: # Avoid processing the same path twice
                continue
                
            if path.is_file():
                # Use self.supported_formats
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
                "files": []
            }
            for f in files:
                try:
                    if f.exists() and f.is_file():
                        manifest_data["files"].append({
                            "path": str(f.resolve()),
                            "name": f.name,
                            "size_bytes": f.stat().st_size,
                            "modified_time": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                        })
                    else:
                        self.logger.warning(f"File not found or not a file, skipping from manifest: {f}")
                except OSError as stat_err:
                    self.logger.warning(f"Could not stat file '{f}'. skipping from manifest: {stat_err}")

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

            if "files" not in manifest_data or not isinstance(manifest_data["files"], list):
                raise KeyError("Manifest format is incorrect (missing 'files' list).")

            files: List[Path] = []
            for file_entry in manifest_data.get("files", []):
                if isinstance(file_entry, dict) and "path" in file_entry:
                    try:
                        files.append(Path(file_entry["path"]))
                    except Exception as path_err:
                        self.logger.warning(f"Invalid path in manifest entry, skipping: {file_entry.get('path')}: ({path_err}).")
                else:
                    self.logger.warning(f"Skipping invalid entry in manifest: {file_entry}")

            # Optional check for file existence
            existing_files = [f for f in files if f.exists()]
            if len(existing_files) != len(files):
                missing_count = len(files) - len(existing_files)
                self.logger.warning(f"Loaded {len(existing_files)} files from manifest. {missing_count} files not found.")
            else:
                self.logger.info(f"Loaded {len(existing_files)} files from manifest.")
            return existing_files

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse manifest file (invalid JSON): {manifest_path} - {e}")
            return []
        except KeyError as e:
            self.logger.error(f"Failed to load manifest: {e}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to load manifest {manifest_path}: {e}", exc_info=True)
            return []
