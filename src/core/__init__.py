# llamanote/core/__init__.py
"""
Core Package for LlamaNote Enhanced

This package contains the central processing pipeline,
shared data types, and custom error classes.
"""

from .types import (
    ProcessingMode,
    ChunkingStrategy,
    PipelineConfig,
    PipelineResult,
    GenerationResult,
    AudioConfig,
    AudioResult,
    QuantizationConfig,
    LayerSplitConfig,
    PDFMetadata,
    ExtractionResult,
    TextChunk,
    ChunkingResult,
    FilterResult
)
from .errors import (
    LlamaNoteError,
    ConfigurationError,
    FileProcessingError,
    PDFExtractionError,
    ModelError,
    ModelLoadError,
    GenerationError,
    BackendError,
    PipelineError,
    MissingDataError,
    ValidationError
)
from .pipeline import ProcessingPipeline
from .stages import (
    run_extraction_stage,
    run_preprocess_stage,
    run_chunking_stage,
    run_processing_stage,
    run_filtering_stage,
    run_formatting_stage,
    run_audio_stage
)


__all__ = [
    # types.py
    "ProcessingMode",
    "ChunkingStrategy",
    "PipelineConfig",
    "PipelineResult",
    "GenerationResult",
    "AudioConfig",
    "AudioResult",
    "QuantizationConfig",
    "LayerSplitConfig",
    "PDFMetadata",
    "ExtractionResult",
    "TextChunk",
    "ChunkingResult",
    "FilterResult",
    
    # errors.py
    "LlamaNoteError",
    "ConfigurationError",
    "FileProcessingError",
    "PDFExtractionError",
    "ModelError",
    "ModelLoadError",
    "GenerationError",
    "BackendError",
    "PipelineError",
    "MissingDataError",
    "ValidationError",
    
    # pipeline.py
    "ProcessingPipeline",
    
    # stages.py
    "run_extraction_stage",
    "run_preprocess_stage",
    "run_chunking_stage",
    "run_processing_stage",
    "run_filtering_stage",
    "run_formatting_stage",
    "run_audio_stage",
]
