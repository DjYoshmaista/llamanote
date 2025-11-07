# llamanote/config/settings.py
"""
Core Configuration Settings for LlamaNote Enhanced
Consolidates paths, constants, default values, prompts, and styles.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import torch # Keep for dtype definition if needed elsewhere

# === Base Paths ===
try:
    # If running within a package structure
    BASE_DIR = Path(__file__).parent.parent.parent
except NameError:
    # Fallback if __file__ is not defined (e.g., interactive session)
    BASE_DIR = Path.cwd()

# Default directories relative to BASE_DIR
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
DEFAULT_LOG_DIR = BASE_DIR / "logs"
DEFAULT_CACHE_DIR = BASE_DIR / "cache"  # For metadata/info cache (NOT for model files)
DEFAULT_OFFLOAD_DIR = BASE_DIR / "offload"
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "llamanote" # User config

# Ensure default directories exist (create if necessary)
# Note: Directory creation is now primarily handled by ConfigManager/helpers
# for dir_path in [DEFAULT_OUTPUT_DIR, DEFAULT_LOG_DIR, DEFAULT_CACHE_DIR, DEFAULT_OFFLOAD_DIR]:
#     try:
#         dir_path.mkdir(exist_ok=True, parents=True)
#     except OSError:
#          # Log warning if possible (logger might not be configured yet)
#          print(f"Warning: Could not create default directory {dir_path}")
#          pass # Avoid crashing on startup

# === Model Cache Settings ===
# Use HuggingFace's standard cache directory to avoid duplication
# HuggingFace Hub cache location (respects HF_HOME environment variable)
import os
HF_CACHE_HOME = Path(os.getenv('HF_HOME', Path.home() / '.cache' / 'huggingface'))
HF_HUB_CACHE = HF_CACHE_HOME / 'hub'

USER_MODEL_CACHE_DIR = None  # User-defined cache path (overrides default)
# Default to HuggingFace hub cache to consolidate all model downloads
DEFAULT_MODEL_CACHE_DIR = HF_HUB_CACHE
# === Core Constants ===
QUANTIZATION_OPTIONS: List[str] = ["none", "4bit", "8bit", "16bit"] # 16bit often means float16/bfloat16
DEFAULT_QUANTIZATION: str = "4bit"

# Layer splitting configuration defaults
ENABLE_LAYER_SPLITTING: bool = True
DEFAULT_GPU_LAYERS: int = -1  # -1 means auto-detect (GGUF primarily)

# Chunk processing settings
CHUNK_SIZE_MIN: int = 100
CHUNK_SIZE_MAX: int = 50000 # Increased max based on original
CHUNK_SIZE_DEFAULT: int = 1000 # Default changed in original config.py
CHUNK_OVERLAP: int = 50

# Default model selection keys (resolved by registry)
DEFAULT_MODEL_KEY: str = "qwen3-4b" # Changed in original config.py
FALLBACK_MODEL_KEY: str = "gemma3-270m" # Changed from gemma-270m

# File processing settings
MAX_PDF_SIZE_MB: float = 100.0
MAX_CHARS_PER_FILE: int = 10_000_000
SUPPORTED_FORMATS: List[str] = [".pdf", ".txt", ".md"] # Removed .docx - needs specific parser
BATCH_PROCESSING_ENABLED: bool = True
MAX_PARALLEL_FILES: int = 1 # Parallel processing not implemented

# Output settings
OUTPUT_FORMAT_OPTIONS: List[str] = ["markdown", "text", "json", "html"]
DEFAULT_OUTPUT_FORMAT: str = "markdown"
DEFAULT_OUTPUT_DIR_NAME: str = "output" # Relative to base or user config
INCLUDE_METADATA: bool = True
TIMESTAMP_OUTPUTS: bool = True

# Error handling settings
MAX_RETRIES: int = 3
RETRY_DELAY_SECONDS: float = 2.0
FALLBACK_ON_ERROR: bool = True # Fallback to original chunk if LLM fails
SAVE_ERROR_CONTEXT: bool = True # Save detailed context on critical errors

# Processing pipeline settings
DEFAULT_PIPELINE_STAGES: List[str] = [
    "extract",
    "preprocess",
    "chunk",
    "process",
    "filter",
    "format",
    "save",
    "audio"
]
ENABLE_STAGE_CHECKPOINTS: bool = True
CHECKPOINT_FORMAT: str = "pickle" # Currently only pickle supported
CHECKPOINT_RESUME_MODE: str = "auto"  # "auto", "interactive", or "disabled"
CHECKPOINT_CLEANUP_KEEP: int = 3  # Number of checkpoints to keep per stage

# Progress tracking settings
# Stage weights for pipeline progress calculation (based on typical processing time)
# Total weight: ~84 units (Process: 53.6%, Audio: 41.7% of total time)
DEFAULT_STAGE_WEIGHTS: Dict[str, float] = {
    "extract": 0.5,      # Fast - PDF/text extraction
    "preprocess": 0.5,   # Fast - text cleaning
    "chunk": 0.3,        # Very fast - text splitting
    "process": 45.0,     # SLOW - LLM generation (main bottleneck)
    "filter": 2.0,       # Medium - post-processing
    "format": 0.2,       # Fast - formatting output
    "save": 0.5,         # Fast - file I/O
    "audio": 35.0,       # SLOW - Audio generation (TTS)
}

# API/Service settings (Placeholder for future API mode)
ENABLE_API_MODE: bool = False
API_RATE_LIMIT: int = 10
API_TIMEOUT_SECONDS: int = 300

# === Default Prompts ===
DEFAULT_SYSTEM_PROMPT: str = """
You are a world class text pre-processor. Analyze the following raw text extracted from a document.
Clean it up, remove any irrelevant formatting artifacts (like excessive newlines, page numbers, headers/footers if obvious), and reformat it for readability and downstream use (e.g., feeding into a summarizer or audio generator).
Focus ONLY on cleaning and reformatting. DO NOT summarize, add opinions, or change the core meaning.
Remove or translate elements unsuitable for plain text (e.g., complex LaTeX math, broken table structures) intelligently.
Return ONLY the cleaned text, without any introductory phrases, acknowledgments, or markdown formatting unless it was present and meaningful in the original text structure (like lists or headings).

Raw text follows:
"""

# Text preprocessing prompt (for preprocess stage) - used for initial cleanup
PREPROCESS_PROMPT_PODCAST: str = """
Clean up this text from a PDF document. Remove formatting artifacts, fix broken sentences, and convert complex notation to plain language. Return only the cleaned text - no commentary or acknowledgments.

Text:
"""

# Podcast script generation prompt (for process stage) - generates the actual dialogue
PODCAST_GENERATION_PROMPT: str = """
Convert the following content into an engaging podcast dialogue between a Host and a Guest.

Rules:
- Create natural, conversational exchanges
- Host asks questions, Guest provides informed responses
- Explain concepts clearly without repetition
- NO meta-commentary about tasks or preprocessing
- NO thinking process or instructions in output
- Start immediately with dialogue

Format:
**[Speaker Host]:** [dialogue]
**[Speaker Guest]:** [response]

Content:
"""


# === Markdown Formatting Styles ===
@dataclass
class MarkdownStyle:
    """Configuration for a specific markdown formatting style."""
    name: str
    emphasis_markers: Dict[str, str] = field(default_factory=dict)
    emotion_markers: Dict[str, str] = field(default_factory=dict)
    structure_markers: Dict[str, str] = field(default_factory=dict)

MARKDOWN_STYLES: Dict[str, MarkdownStyle] = {
    "podcast": MarkdownStyle(
        name="Podcast Script",
        emphasis_markers={
            "strong": "**{}**", "emphasis": "*{}*", "pause": "... {} ...",
            "slow": "~{}~", "fast": "^{}^"
        },
        emotion_markers={
            "excited": "🎉 {}", "thoughtful": "🤔 {}", "serious": "😐 {}",
            "humorous": "😄 {}", "surprised": "😲 {}", "questioning": "❓ {}"
        },
        structure_markers={
            "section": "\n## {}\n", "subsection": "\n### {}\n",
            "transition": "\n---\n", "speaker_change": "\n**[Speaker {}]:**\n"
        }
    ),
    "narrative": MarkdownStyle(
        name="Narrative Style",
        emphasis_markers={"strong": "**{}**", "emphasis": "*{}*", "whisper": "_{}_", "shout": "***{}***"},
        emotion_markers={"narrator": "📖 {}", "dialogue": "💬 {}", "action": "🎬 {}", "description": "🖼️ {}"},
        structure_markers={"chapter": "\n# {}\n", "scene": "\n## {}\n", "break": "\n* * *\n"}
    ),
    "technical": MarkdownStyle(
        name="Technical Documentation",
        emphasis_markers={"code": "`{}`", "important": "**⚠️ {}**", "note": "📝 *{}*", "tip": "💡 {}"},
        structure_markers={"section": "\n## {}\n", "code_block": "\n```\n{}\n```\n", "list_item": "- {}"}
    ),
     "default": MarkdownStyle( # Added a basic default
        name="Default Markdown",
        emphasis_markers={"strong": "**{}**", "emphasis": "*{}*"},
        structure_markers={"section": "\n## {}\n", "subsection": "\n### {}\n"}
    )
}

# === Response Filtering Patterns ===
# Moved thinking patterns to constants.py as they are static data

# === Cloud Provider Constants ===
KNOWN_CLOUD_PROVIDERS: List[str] = [ # Used by CloudKeyManager
    "openai", "anthropic", "google", "cohere", "huggingface_token",
    "deepseek", "groq", "openrouter", "qwen"
]
SUPPORTED_LLM_PROVIDERS: List[str] = ["local_hf", "local_gguf", "openai", "google", "anthropic"] # Currently implemented backends
SUPPORTED_TTS_PROVIDERS: List[str] = ["local_audio", "openai_audio"] # Currently implemented audio backends

# === Logging Configuration ===
# Keep LOGGING_CONFIG dictionary here for easy access, ensuring LOG_DIR is resolved
def get_logging_config(log_dir: Path) -> Dict[str, Any]:
    """Generates the logging configuration dictionary."""
    log_dir.mkdir(parents=True, exist_ok=True) # Ensure log dir exists
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"},
            "simple": {"format": "%(asctime)s - %(levelname)s - %(message)s", "datefmt": "%H:%M:%S"}
        },
        "handlers": {
            "console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "simple", "stream": "ext://sys.stdout"},
            "file": {"class": "logging.handlers.RotatingFileHandler", "level": "DEBUG", "formatter": "detailed", "filename": str(log_dir / "llamanote.log"), "maxBytes": 10485760, "backupCount": 5},
            "error_file": {"class": "logging.handlers.RotatingFileHandler", "level": "ERROR", "formatter": "detailed", "filename": str(log_dir / "errors.log"), "maxBytes": 10485760, "backupCount": 5}
        },
        "loggers": {
            # Configure specific loggers if needed, e.g., 'llamanote' base logger
             "llamanote": {"level": "DEBUG", "handlers": ["console", "file", "error_file"], "propagate": False},
             # Reduce noise from libraries
             "httpx": {"level": "WARNING", "handlers": ["console", "file"]},
             "httpcore": {"level": "WARNING", "handlers": ["console", "file"]},
             "openai": {"level": "WARNING", "handlers": ["console", "file"]},
             "anthropic": {"level": "WARNING", "handlers": ["console", "file"]},
             "google": {"level": "WARNING", "handlers": ["console", "file"]},
             "huggingface_hub": {"level": "WARNING", "handlers": ["console", "file"]},
             "transformers": {"level": "WARNING", "handlers": ["console", "file"]},
             "torch": {"level": "WARNING", "handlers": ["console", "file"]},
             "accelerate": {"level": "WARNING", "handlers": ["console", "file"]},
        },
        "root": {"level": "INFO", "handlers": ["console"]} # Root only logs INFO+ to console by default
    }
