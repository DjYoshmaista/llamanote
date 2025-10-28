# llamanote/config/__init__.py
"""
Configuration Package for LlamaNote Enhanced

This package contains modules for managing application settings,
paths, memory profiles, hyperparameter presets, and API keys.
"""

from .settings import (
    BASE_DIR, DEFAULT_OUTPUT_DIR as OUTPUT_DIR, DEFAULT_LOG_DIR as LOG_DIR, DEFAULT_CACHE_DIR as CACHE_DIR,
    DEFAULT_OFFLOAD_DIR as OFFLOAD_DIR, DEFAULT_MODEL_KEY, FALLBACK_MODEL_KEY, SUPPORTED_FORMATS,
    DEFAULT_PIPELINE_STAGES, PREPROCESS_PROMPT_PODCAST, DEFAULT_SYSTEM_PROMPT
)
from .manager import ConfigManager
from .profiles import MemoryProfile, get_memory_profile, create_configs_from_memory_profile
from .presets import get_hyperparameter_preset, list_hyperparameter_presets
from .cloud_keys import CloudKeyManager

__all__ = [
    # settings.py exports
    "BASE_DIR",
    "OUTPUT_DIR",
    "LOG_DIR",
    "CACHE_DIR",
    "OFFLOAD_DIR",
    "DEFAULT_MODEL_KEY",
    "FALLBACK_MODEL_KEY",
    "SUPPORTED_FORMATS",
    "DEFAULT_PIPELINE_STAGES",
    "PREPROCESS_PROMPT_PODCAST",
    "DEFAULT_SYSTEM_PROMPT",
    
    # manager.py exports
    "ConfigManager",
    
    # profiles.py exports
    "MemoryProfile",
    "get_memory_profile",
    "create_configs_from_memory_profile",
    
    # presets.py exports
    "get_hyperparameter_preset",
    "list_hyperparameter_presets",
    
    # cloud_keys.py exports
    "CloudKeyManager",
]
