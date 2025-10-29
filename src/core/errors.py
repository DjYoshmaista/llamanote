# llamanote/core/errors.py
"""
Custom Exception Classes for LlamaNote Enhanced
"""

from typing import Optional

class LlamaNoteError(Exception):
    """Base exception for all LlamaNote errors."""
    pass

class ConfigurationError(LlamaNoteError):
    """Error related to configuration loading or validation."""
    def __init__(self, message: str, config_path: Optional[str] = None):
        self.config_path = config_path
        full_message = f"Configuration Error: {message}"
        if config_path:
            full_message += f" (Path: {config_path})"
        super().__init__(full_message)

class FileProcessingError(LlamaNoteError):
    """Error during file handling or processing (e.g., PDF extraction)."""
    def __init__(self, message: str, file_path: Optional[str] = None):
        self.file_path = file_path
        full_message = f"File Processing Error: {message}"
        if file_path:
            full_message += f" (File: {file_path})"
        super().__init__(full_message)

class PDFExtractionError(FileProcessingError):
    """Specific error during PDF text extraction."""
    pass

class ModelError(LlamaNoteError):
    """Base error related to model operations."""
    def __init__(self, message: str, model_id: Optional[str] = None):
        self.model_id = model_id
        full_message = f"Model Error: {message}"
        if model_id:
            full_message += f" (Model: {model_id})"
        super().__init__(full_message)

class ModelLoadError(ModelError):
    """Error during model loading."""
    pass

class GenerationError(ModelError):
    """Error during text or audio generation."""
    pass

class BackendError(ModelError):
    """Error related to specific backend communication or setup (e.g., API key)."""
    def __init__(self, message: str, backend_provider: Optional[str] = None):
        self.backend_provider = backend_provider
        full_message = f"Backend Error: {message}"
        if backend_provider:
            full_message += f" (Provider: {backend_provider})"
        # Use super().__init__ from ModelError to include model_id if relevant
        # Need to decide how to pass model_id here, maybe add it to constructor?
        super().__init__(full_message) # For now, just pass the message

class PipelineError(LlamaNoteError):
    """Generic error occurring during pipeline execution."""
    def __init__(self, message: str, stage: Optional[str] = None):
        self.stage = stage
        full_message = f"Pipeline Error: {message}"
        if stage:
            full_message += f" (Stage: {stage})"
        super().__init__(full_message)

class MissingDataError(PipelineError):
    """Error indicating required data is missing between pipeline stages."""
    def __init__(self, stage: str, missing_key: str):
        message = f"Missing required data '{missing_key}' for stage '{stage}'. A previous stage might have failed or been skipped."
        super().__init__(message, stage=stage)
        self.missing_key = missing_key

class ValidationError(LlamaNoteError):
    """Error for general input validation failures."""
    pass
