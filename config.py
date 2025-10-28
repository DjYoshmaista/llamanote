"""
Enhanced Configuration File for LlamaNote PDF Processor
Centralized configuration for all modules and settings
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
# Import base config (no circular dependencies)
from config_base import (
    BASE_DIR, OUTPUT_DIR, LOG_DIR, CACHE_DIR, OFFLOAD_DIR,
    QUANTIZATION_OPTIONS, DEFAULT_QUANTIZATION,
    ENABLE_LAYER_SPLITTING, DEFAULT_GPU_LAYERS,
    CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP,
    DEFAULT_MODEL, FALLBACK_MODEL,
    MAX_PDF_SIZE_MB, MAX_CHARS_PER_FILE, SUPPORTED_FORMATS,
    BATCH_PROCESSING_ENABLED, MAX_PARALLEL_FILES,
    OUTPUT_FORMAT_OPTIONS, DEFAULT_OUTPUT_FORMAT,
    INCLUDE_METADATA, TIMESTAMP_OUTPUTS,
    MAX_RETRIES, RETRY_DELAY_SECONDS, FALLBACK_ON_ERROR, SAVE_ERROR_CONTEXT,
    PIPELINE_STAGES, ENABLE_STAGE_CHECKPOINTS, CHECKPOINT_FORMAT,
    ENABLE_API_MODE, API_RATE_LIMIT, API_TIMEOUT_SECONDS
)

# Type checking imports (not evaluated at runtime)
if TYPE_CHECKING:
    from hyperparameters import HyperparameterConfig
    from model_registry import ModelEntry

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

# Model configurations (legacy - kept for backward compatibility)
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
    max_new_tokens: Optional[int] = None
    quantization_support: List[str] = field(default_factory=lambda: ["4bit", "8bit"])


# Legacy MODELS dict - dynamically populated from registry
def _get_legacy_models() -> Dict[str, ModelConfig]:
    """Get legacy MODELS dict from registry for backward compatibility"""
    try:
        from model_registry import get_registry
        
        registry = get_registry()
        models = {}
        
        # Get predefined models
        for entry in registry.list_models(predefined_only=True):
            # Convert to old ModelConfig format
            key = entry.model_id.split('/')[-1].lower().replace('-', '').replace('.', '')
            if 'qwen' in key:
                key = 'qwen3-4b'
            elif 'gemma' in key:
                key = 'gemma-270m'
            elif 'llama' in key:
                key = 'llama-3.2-1b'
            
            models[key] = ModelConfig(
                name=entry.name,
                model_id=entry.model_id,
                supports_thinking=entry.supports_thinking,
                thinking_tokens=entry.thinking_tokens or [],
                max_context=entry.max_context,
                optimal_chunk_size=entry.optimal_chunk_size,
                temperature=entry.temperature,
                top_p=entry.top_p,
                max_new_tokens=entry.max_new_tokens,
                quantization_support=entry.quantization_support or ["4bit", "8bit"]
            )
        
        return models
    except:
        # Fallback if registry not available
        return {
            "qwen3-4b": ModelConfig(
                name="Qwen3-4B Thinking",
                model_id="Qwen/Qwen2.5-4B-Instruct",
                supports_thinking=True,
                thinking_tokens=["<think>", "</think>"],
                max_context=32768,
                optimal_chunk_size=1500
            )
        }


# Dynamic MODELS dict
MODELS = _get_legacy_models()


# Memory optimization settings
@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    use_quantization: bool = True
    quantization_type: str = "8bit"
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
            "maxBytes": 10485760,
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


# Lazy loading functions to avoid circular imports
def get_default_hyperparams():
    """Lazy load default hyperparameters"""
    from hyperparameters import HyperparameterConfig
    return HyperparameterConfig()


def get_model_hyperparams(model_identifier: str):
    """
    Get hyperparameters for a specific model
    
    Args:
        model_identifier: Model ID or key
        
    Returns:
        HyperparameterConfig
    """
    from hyperparameters import HyperparameterConfig
    from model_registry import get_model_config
    
    # Try to get from registry
    model_entry = get_model_config(model_identifier)
    
    if model_entry:
        return HyperparameterConfig(
            temperature=model_entry.temperature,
            top_p=model_entry.top_p,
            max_new_tokens=model_entry.max_new_tokens or 2048
        )
    
    # Default
    return HyperparameterConfig()


def reload_models():
    """Reload models from registry"""
    global MODELS
    MODELS = _get_legacy_models()
