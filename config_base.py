"""
Base Configuration - No Circular Imports
Contains only basic paths and constants that other modules need
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
OFFLOAD_DIR = BASE_DIR / "offload"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, LOG_DIR, CACHE_DIR, OFFLOAD_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Quantization options
QUANTIZATION_OPTIONS = ["none", "4bit", "8bit", "16bit"]
DEFAULT_QUANTIZATION = "4bit"

# Layer splitting configuration
ENABLE_LAYER_SPLITTING = True
DEFAULT_GPU_LAYERS = -1  # -1 means auto-detect

# Chunk processing settings
CHUNK_SIZE_MIN = 100
CHUNK_SIZE_MAX = 5000
CHUNK_SIZE_DEFAULT = 1000
CHUNK_OVERLAP = 50

# Default model selection
DEFAULT_MODEL = "qwen3-4b"
FALLBACK_MODEL = "gemma-270m"

# File processing settings
MAX_PDF_SIZE_MB = 100
MAX_CHARS_PER_FILE = 10000000
SUPPORTED_FORMATS = [".pdf", ".txt", ".md", ".docx"]
BATCH_PROCESSING_ENABLED = True
MAX_PARALLEL_FILES = 3

# Output settings
OUTPUT_FORMAT_OPTIONS = ["markdown", "text", "json", "html"]
DEFAULT_OUTPUT_FORMAT = "markdown"
DEFAULT_OUTPUT_DIR = "~/.cache/llamanote/outputs/"
INCLUDE_METADATA = True
TIMESTAMP_OUTPUTS = True

# Error handling settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
FALLBACK_ON_ERROR = True
SAVE_ERROR_CONTEXT = True

# Processing pipeline settings
PIPELINE_STAGES = [
    "extract",
    "preprocess",
    "chunk",
    "process",
    "filter",
    "format",
    "save"
]

ENABLE_STAGE_CHECKPOINTS = True
CHECKPOINT_FORMAT = "pickle"

# API/Service settings
ENABLE_API_MODE = False
API_RATE_LIMIT = 10
API_TIMEOUT_SECONDS = 300
