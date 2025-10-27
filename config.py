"""
Enhanced Configuration File for LlamaNote PDF Processor
Centralized configuration for all modules and settings
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

# Base paths
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
LOG_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
OFFLOAD_DIR = BASE_DIR / "offload"

# Create directories if they don't exist
for dir_path in [OUTPUT_DIR, LOG_DIR, CACHE_DIR, OFFLOAD_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Original preprocessing prompt
PREPROCESS_PROMPT = """
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

# Model configurations
@dataclass
class ModelConfig:
    """Configuration for different model options"""
    name: str
    model_id: str
    supports_thinking: bool = False
    thinking_tokens: List[str] = field(default_factory=list)
    max_context: int = 32768
    optimal_chunk_size: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: Optional[int] = None  # None means dynamic sizing
    quantization_support: List[str] = field(default_factory=lambda: ["4bit", "8bit"])

# Available models
MODELS = {
    "qwen3-4b": ModelConfig(
        name="Qwen3-4B Thinking",
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        supports_thinking=True,
        thinking_tokens=["<think>", "</think>", "<|thinking|>", "<|/thinking|>", "[THINK]", "[/THINK]"],
        max_context=32768,
        optimal_chunk_size=1500,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=None
    ),
    "gemma-270m": ModelConfig(
        name="Gemma 270M",
        model_id="google/gemma-3-270m",
        supports_thinking=False,
        max_context=8192,
        optimal_chunk_size=1000,
        temperature=0.8,
        top_p=0.95,
        max_new_tokens=2048
    ),
    "llama-3.2-1b": ModelConfig(
        name="Llama 3.2 1B",
        model_id="meta-llama/Llama-3.2-1B-Instruct",
        supports_thinking=False,
        max_context=8192,
        optimal_chunk_size=1200,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=2048
    ),
}

# Default model selection
DEFAULT_MODEL = "qwen3-4b"
FALLBACK_MODEL = "gemma-270m"

# Chunk processing settings
CHUNK_SIZE_MIN = 100
CHUNK_SIZE_MAX = 5000
CHUNK_SIZE_DEFAULT = 1000
CHUNK_OVERLAP = 50  # Characters to overlap between chunks for context

# Memory optimization settings
@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    use_quantization: bool = True
    quantization_type: str = "8bit"  # "4bit", "8bit", or "none"
    max_gpu_memory: str = "10GB"
    max_cpu_memory: str = "30GB"
    use_flash_attention: bool = True
    use_gradient_checkpointing: bool = False
    offload_to_disk: bool = True
    batch_size: int = 1
    
MEMORY_PROFILES = {
    "low_vram": MemoryConfig(
        use_quantization=True,
        quantization_type="4bit",
        max_gpu_memory="4GB",
        max_cpu_memory="16GB",
        use_flash_attention=True,
        batch_size=1
    ),
    "medium_vram": MemoryConfig(
        use_quantization=True,
        quantization_type="8bit",
        max_gpu_memory="8GB",
        max_cpu_memory="24GB",
        use_flash_attention=True,
        batch_size=1
    ),
    "high_vram": MemoryConfig(
        use_quantization=False,
        quantization_type="none",
        max_gpu_memory="24GB",
        max_cpu_memory="32GB",
        use_flash_attention=True,
        batch_size=2
    ),
    "cpu_only": MemoryConfig(
        use_quantization=True,
        quantization_type="8bit",
        max_gpu_memory="0GB",
        max_cpu_memory="64GB",
        use_flash_attention=False,
        batch_size=1
    )
}

# Markdown formatting options
@dataclass
class MarkdownStyle:
    """Markdown formatting style configuration"""
    name: str
    emphasis_markers: Dict[str, str]
    emotion_markers: Dict[str, str]
    structure_markers: Dict[str, str]

MARKDOWN_STYLES = {
    "podcast": MarkdownStyle(
        name="Podcast Script",
        emphasis_markers={
            "strong": "**{}**",
            "emphasis": "*{}*",
            "pause": "... {} ...",
            "slow": "~{}~",
            "fast": "^{}^"
        },
        emotion_markers={
            "excited": "🎉 {}",
            "thoughtful": "🤔 {}",
            "serious": "😐 {}",
            "humorous": "😄 {}",
            "surprised": "😲 {}",
            "questioning": "❓ {}"
        },
        structure_markers={
            "section": "\n## {}\n",
            "subsection": "\n### {}\n",
            "transition": "\n---\n",
            "speaker_change": "\n**[Speaker {}]:**\n"
        }
    ),
    "narrative": MarkdownStyle(
        name="Narrative Style",
        emphasis_markers={
            "strong": "**{}**",
            "emphasis": "*{}*",
            "whisper": "_{}_",
            "shout": "***{}***"
        },
        emotion_markers={
            "narrator": "📖 {}",
            "dialogue": "💬 {}",
            "action": "🎬 {}",
            "description": "🖼️ {}"
        },
        structure_markers={
            "chapter": "\n# {}\n",
            "scene": "\n## {}\n",
            "break": "\n* * *\n"
        }
    ),
    "technical": MarkdownStyle(
        name="Technical Documentation",
        emphasis_markers={
            "code": "`{}`",
            "important": "**⚠️ {}**",
            "note": "📝 *{}*",
            "tip": "💡 {}"
        },
        emotion_markers={},
        structure_markers={
            "section": "\n## {}\n",
            "code_block": "\n```\n{}\n```\n",
            "list_item": "- {}"
        }
    )
}

# Response filtering patterns
THINKING_PATTERNS = [
    # Common thinking model patterns
    (r"<think>(.*?)</think>", ""),
    (r"<\|thinking\|>(.*?)<\|/thinking\|>", ""),
    (r"\[THINK\](.*?)\[/THINK\]", ""),
    (r"<thinking>(.*?)</thinking>", ""),
    (r"```thinking(.*?)```", ""),
    # Chain of thought patterns
    (r"Let me think.*?(?=\n\n)", ""),
    (r"Step \d+:.*?(?=\n\n)", ""),
    (r"First,.*?(?=\n\n)", ""),
    # Internal monologue patterns
    (r"\(thinking:.*?\)", ""),
    (r"\[internal:.*?\]", ""),
]

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "simple": {
            "format": "%(asctime)s - %(levelname)s - %(message)s",
            "datefmt": "%H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "llamanote.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "errors.log"),
            "maxBytes": 10485760,
            "backupCount": 5
        }
    },
    "loggers": {
        "llamanote": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error_file"],
            "propagate": False
        },
        "transformers": {
            "level": "WARNING",
            "handlers": ["console", "file"]
        },
        "torch": {
            "level": "WARNING",
            "handlers": ["console", "file"]
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"]
    }
}

# File processing settings
MAX_PDF_SIZE_MB = 100
MAX_CHARS_PER_FILE = 10000000
SUPPORTED_FORMATS = [".pdf", ".txt", ".md", ".docx"]
BATCH_PROCESSING_ENABLED = True
MAX_PARALLEL_FILES = 3

# Output settings
OUTPUT_FORMAT_OPTIONS = ["markdown", "text", "json", "html"]
DEFAULT_OUTPUT_FORMAT = "markdown"
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
CHECKPOINT_FORMAT = "pickle"  # or "json"

# API/Service settings (for future module usage)
ENABLE_API_MODE = False
API_RATE_LIMIT = 10  # requests per minute
API_TIMEOUT_SECONDS = 300
