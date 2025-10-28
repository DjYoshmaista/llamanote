"""
Enhanced Configuration File for LlamaNote PDF Processor
Centralized configuration for all modules and settings
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
import torch # Import needed for MemoryConfig dtype, if used

# Import base config (no circular dependencies)
from config_base import (
    BASE_DIR, OUTPUT_DIR, LOG_DIR, CACHE_DIR, OFFLOAD_DIR,
    QUANTIZATION_OPTIONS, DEFAULT_QUANTIZATION,
    ENABLE_LAYER_SPLITTING, DEFAULT_GPU_LAYERS,
    CHUNK_SIZE_MIN, CHUNK_SIZE_MAX, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP,
    DEFAULT_MODEL, FALLBACK_MODEL, # DEFAULT_MODEL is now just a key string
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
    # Use the centralized types now
    from pipeline_types import QuantizationConfig, LayerSplitConfig
    from hyperparameters import HyperparameterConfig
    from model_registry import ModelEntry

# Original preprocessing prompt (kept as is)
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

MODELS: Dict[str, str] = {} # Initialize as empty

def reload_models():
    """Reload MODELS dict from registry's predefined models"""
    global MODELS
    try:
        from model_registry import get_registry
        registry = get_registry()
        MODELS = {}
        predefined = registry.list_models(predefined_only=True)
        for entry in predefined:
            if entry.short_key:
                MODELS[entry.short_key] = entry.model_id
            else:
                # Fallback key if short_key is missing (should not happen for predefined)
                fallback_key = entry.model_id.split('/')[-1].lower().replace('-', '').replace('.', '')
                MODELS[fallback_key] = entry.model_id
        # Ensure default and fallback keys exist if possible
        if DEFAULT_MODEL not in MODELS:
             entry = registry.get_by_key(DEFAULT_MODEL)
             if entry: MODELS[DEFAULT_MODEL] = entry.model_id
        if FALLBACK_MODEL not in MODELS:
             entry = registry.get_by_key(FALLBACK_MODEL)
             if entry: MODELS[FALLBACK_MODEL] = entry.model_id

        # print(f"Reloaded MODELS dict: {list(MODELS.keys())}") # Debug print
    except Exception as e:
        print(f"Warning: Could not reload MODELS dict from registry: {e}")
        # Provide minimal fallback if registry fails
        MODELS = {
            DEFAULT_MODEL: "Qwen/Qwen3-4B-Instruct-2507",
            FALLBACK_MODEL: "google/gemma-3-270m"
        }

# Initial population
reload_models()


# Memory optimization settings (Legacy - may be replaced by LayerSplitConfig/QuantizationConfig)
@dataclass
class MemoryConfig:
    """(LEGACY) Memory optimization configuration - Prefer direct QuantizationConfig/LayerSplitConfig"""
    use_quantization: bool = True
    quantization_type: str = "4bit" # Should match DEFAULT_QUANTIZATION
    max_gpu_memory: str = "10GB" # Example default
    max_cpu_memory: str = "30GB" # Example default
    use_flash_attention: bool = False # Specific to transformers backend
    use_gradient_checkpointing: bool = True # Usually for training
    offload_to_disk: bool = True # Maps to LayerSplitConfig offload folder
    batch_size: int = 128 # Relevant for batch processing


# Example memory profiles (Can be used to create Quantization/LayerSplit configs)
MEMORY_PROFILES = {
    "low_vram": {
        "quantization": "4bit",
        "max_gpu_memory": "4GB",
        "max_cpu_memory": "16GB",
        "gpu_layers": -1 # Auto layers for GGUF
    },
    "medium_vram": {
        "quantization": DEFAULT_QUANTIZATION,
        "max_gpu_memory": "10GB",
        "max_cpu_memory": "30GB",
        "gpu_layers": DEFAULT_GPU_LAYERS
    },
    "high_vram": {
        "quantization": "none", # Or maybe 8bit
        "max_gpu_memory": "24GB",
        "max_cpu_memory": "64GB",
        "gpu_layers": -1
    },
    "cpu_only": {
        "quantization": "none", # BNB Quantization usually needs GPU
        "max_gpu_memory": "0GB",
        "max_cpu_memory": "64GB",
        "gpu_layers": 0 # Explicitly CPU only for GGUF
    }
}


# Markdown formatting options (Kept as is)
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
        emotion_markers={},
        structure_markers={"section": "\n## {}\n", "code_block": "\n```\n{}\n```\n", "list_item": "- {}"}
    )
}


# Response filtering patterns (Kept as is, filter uses registry info now)
THINKING_PATTERNS = [
    (r"<think>(.*?)</think>", ""),
    (r"<\|thinking\|>(.*?)<\|/thinking\|>", ""),
    (r"\[THINK\](.*?)\[/THINK\]", ""),
    (r"<thinking>(.*?)</thinking>", ""),
    (r"```thinking(.*?)```", ""),
    (r"Let me think.*?(?=\n\n)", ""),
    (r"Step \d+:.*?(?=\n\n)", ""),
    (r"First,.*?(?=\n\n)", ""),
    (r"\(thinking:.*?\)", ""),
    (r"\[internal:.*?\]", ""),
]


# Logging configuration (Kept as is)
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"},
        "simple": {"format": "%(asctime)s - %(levelname)s - %(message)s", "datefmt": "%H:%M:%S"}
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "level": "INFO", "formatter": "simple", "stream": "ext://sys.stdout"},
        "file": {"class": "logging.handlers.RotatingFileHandler", "level": "DEBUG", "formatter": "detailed", "filename": str(LOG_DIR / "llamanote.log"), "maxBytes": 10485760, "backupCount": 5},
        "error_file": {"class": "logging.handlers.RotatingFileHandler", "level": "ERROR", "formatter": "detailed", "filename": str(LOG_DIR / "errors.log"), "maxBytes": 10485760, "backupCount": 5}
    },
    "loggers": {
        "llamanote": {"level": "DEBUG", "handlers": ["console", "file", "error_file"], "propagate": False},
        "transformers": {"level": "WARNING", "handlers": ["console", "file"]},
        "torch": {"level": "WARNING", "handlers": ["console", "file"]}
    },
    "root": {"level": "INFO", "handlers": ["console", "file"]}
}

# --- Lazy Loading Helpers ---
# These now correctly point to the moved/centralized definitions

def get_default_hyperparams():
    """Lazy load default hyperparameters"""
    from hyperparameters import HyperparameterConfig
    return HyperparameterConfig()

def get_model_hyperparams(model_identifier: str):
    """Get hyperparameters based on model entry in registry"""
    from hyperparameters import HyperparameterConfig
    from model_registry import get_model_config

    model_entry = get_model_config(model_identifier) # Use registry function
    if model_entry:
        # Create HyperparameterConfig from ModelEntry defaults
        return HyperparameterConfig(
            temperature=model_entry.temperature,
            top_p=model_entry.top_p,
            max_new_tokens=model_entry.max_new_tokens or 2048 # Default if None
            # Add other relevant mappings if ModelEntry stores more defaults
        )
    return HyperparameterConfig() # Return default if not found
