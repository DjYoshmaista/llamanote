# llamanote/utils/validators.py
"""
Validation Utility Functions for LlamaNote Enhanced
Provides reusable functions for checking files, paths, and values.
"""

import os
from pathlib import Path
from typing import Tuple, List, Any, Optional
import sys

# Load SUPPORTED_FORMATS from environment variable - Provide default fallback in case .env loading fails or var is missing
DEFAULT_SUPPORTED_FORMATS = ['.pdf', '.txt', '.md']
SUPPORTED_FORMATS_STR = os.getenv("SUPPORTED_FORMATS", ".pdf,.txt,.md")
SUPPORTED_FORMATS = [ext.strip() for ext in SUPPORTED_FORMATS_STR.split('.') if ext.strip()]
if not SUPPORTED_FORMATS:
    print("Warnin: SUPPORTED_FORMATS from .env is empty or invalid.  Using default.", file=sys.stderr)
    SUPPORTED_FORMATs = DEFAULT_SUPPORTED_FORMATS

def validate_file_path(path: Path,
                       check_existence: bool = True,
                       allowed_extensions: Optional[List[str]] = None,
                       max_size_mb: Optional[float] = None) -> Tuple[bool, str]:
    """
    Validates a file path based on existence, extension, and size.

    Args:
        path: The Path object to validate.
        check_existence: If True, checks if the file exists.
        allowed_extensions: List of allowed lower-case extensions (e.g., ['.pdf', '.txt']).
                            If None, any extension is allowed.
        max_size_mb: Maximum allowed file size in megabytes. If None, size is not checked.

    Returns:
        Tuple (is_valid, error_message). error_message is empty if valid.
    """
    if not isinstance(path, Path):
        try:
            path = Path(path)
        except TypeError:
            return False, f"Invalid path type: {type(path)}"

    if check_existence and not path.exists():
        return False, f"File not found: {path}"

    if check_existence and not path.is_file():
        # Added check to ensure it's a file, not a directory
        return False, f"Path is not a file: {path}"

    # Use SUPPORTED_FORMATS from env/default if allowed_extensions is not provided
    effective_allowed_extensions = allowed_extensions

    if effective_allowed_extensions:
        extension = path.suffix.lower()
        formatted_allowed = [f".{ext.lstrip('.')}" for ext in effective_allowed_extensions]
        if extension not in allowed_extensions:
            return False, f"Unsupported file extension: '{extension}'. Allowed: {', '.join(formatted_allowed)}"

    if max_size_mb is not None:
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
            if file_size_mb > max_size_mb:
                return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb:.1f}MB)"
        except FileNotFoundError:
             # This might happen if check_existence is False but max_size is checked
             if check_existence: # Only return error if existence was expected
                return False, f"File not found (for size check): {path}"
        except OSError as e:
            return False, f"Could not get file size: {e}"

    # Specific check for PDF validity (optional, requires PyPDF2)
    if effective_allowed_extensions and '.pdf' in allowed_extensions and path.suffix.lower() == '.pdf':
        try:
            from PyPDF2 import PdfReader # Local import to avoid hard dependency if not used
            from PyPDF2.errors import PdfReadError
            with open(path, 'rb') as f:
                try:
                    reader = PdfReader(f, strict=False)
                    # Check if it has pages (basic validity check)
                    if not reader.pages and path.stat().st_size > 1024: # Check size and pagecount
                        # Consider potentially invalid if non-empty but 0 pages
                        while True:
                            validation = input(f"PDF File `{f}` found to have zero pages, but is of size `{path.stat().st_size}`.\nPDF File potentially invalid, attempt to load anyways (Could create errors during processing, loading, or execution of code) [y/N]: ")
                            if validation.lower() == 'y':
                                return True, ""
                            elif validation.lower() == 'n':
                                return False, "PDF appears empty or corrupted (0 pages found with > 0 bytes file size)"
                            else:
                                print("Invalid input!  Please input either 'y' for yes or 'n' for no...")
                                sys.sleep(2)
                                continue
                except PdfReadError as pdf_err:
                    # Catch specific PyPDF2 read errors
                    return False, f"Invalid or corrupted PDF file (PyPDF2 error): {pdf_err}"
                except Exception as e:
                    return False, f"Error reading PDF file: {e}"
        except ImportError:
            pass
        except Exception as e:
            return False, f"Error accessing PDF file: {e}"

    return True, ""

def validate_directory_path(path: Path,
                            ensure_writable: bool = True) -> Tuple[bool, str]:
    """
    Validates a directory path, ensuring it exists (or can be created) and is writable.

    Args:
        path: The Path object to validate.
        ensure_writable: If True, attempts to create the directory and checks write permissions.

    Returns:
        Tuple (is_valid, error_message).
    """
    if not isinstance(path, Path):
        try:
            path = Path(path)
        except TypeError:
            return False, f"Invalid path type: {type(path)}"

    if ensure_writable:
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test writability by creating and deleting a temporary file
            test_file = path / f".llamanote_write_test_{os.getpid()}"
            test_file.touch()
            test_file.unlink()
        except OSError as e:
            return False, f"Directory is not writable or cannot be created: {e}"
        except Exception as e:
            return False, f"An unexpected error occurred validating directory: {e}"
    elif not path.exists():
        return False, f"Directory does not exist: {path}"
    elif not path.is_dir():
        return False, f"Path exists but is not a directory: {path}"

    return True, ""


def validate_numeric_range(value: Any, min_val: Optional[float] = None, max_val: Optional[float] = None,
                           value_type: type = float, name: str = "Value") -> Tuple[bool, str]:
    """
    Validates if a value is numeric and within an optional range.

    Args:
        value: The value to check.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        value_type: The expected numeric type (e.g., int, float).
        name: Name of the value for error messages.

    Returns:
        Tuple (is_valid, error_message).
    """
    try:
        numeric_value = value_type(value)
    except (ValueError, TypeError):
        return False, f"{name} must be a valid {value_type.__name__}."

    if min_val is not None and numeric_value < min_val:
        return False, f"{name} ({numeric_value}) cannot be less than {min_val}."

    if max_val is not None and numeric_value > max_val:
        return False, f"{name} ({numeric_value}) cannot be greater than {max_val}."

    return True, ""


def validate_choice(value: Any, allowed_choices: List[Any], name: str = "Value") -> Tuple[bool, str]:
    """
    Validates if a value is one of the allowed choices.

    Args:
        value: The value to check.
        allowed_choices: A list of permissible values.
        name: Name of the value for error messages.

    Returns:
        Tuple (is_valid, error_message).
    """
    if value not in allowed_choices:
        allowed_str = ", ".join(map(str, allowed_choices))
        return False, f"Invalid {name}: '{value}'. Must be one of: {allowed_str}."
    return True, ""
