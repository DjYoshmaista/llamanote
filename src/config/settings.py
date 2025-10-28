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
DEFAULT_CACHE_DIR = BASE_DIR / "cache"
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
    "save"
]
ENABLE_STAGE_CHECKPOINTS: bool = True
CHECKPOINT_FORMAT: str = "pickle" # Currently only pickle supported

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

# Kept original PREPROCESS_PROMPT for potential specific use cases (like podcast mode default)
PREPROCESS_PROMPT_PODCAST: str = """
You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO
ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?

Here's the text:
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

# Note: Actual LOG_DIR path is now determined by ConfigManager, which reads this file.
# The logger setup in utils/logger.py will use the path provided by ConfigManager.
