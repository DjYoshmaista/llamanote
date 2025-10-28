# llamanote/models/__init__.py
"""
Models Package for LlamaNote Enhanced

This package contains modules related to AI models:
- registry: Manages a database of known models.
- hub: Interface for interacting with Hugging Face Hub.
- hyperparameters: Defines and manages generation parameters.
- backends: Contains the concrete implementations for different
             LLM and Audio generation backends.
"""

from .registry import get_registry, get_model_entry, ModelEntry
from .hub import ModelHub, ModelHubInfo
from .hyperparameters import HyperparameterConfig, HYPERPARAMETER_DEFINITIONS

__all__ = [
    "get_registry",
    "get_model_entry",
    "ModelEntry",
    "ModelHub",
    "ModelHubInfo",
    "HyperparameterConfig",
    "HYPERPARAMETER_DEFINITIONS",
]
