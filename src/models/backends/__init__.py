# llamanote/models/backends/__init__.py
"""
Backend Factory for LlamaNote

This module provides factory functions to instantiate the correct
LLM (Large Language Model) or Audio (Text-to-Speech) backend
based on the provided configuration.
"""

from typing import Dict, Optional, Any

from ...utils.logger import get_logger_conf, ConsoleOutput
from ...core.errors import ConfigurationError, ModelLoadError
from ...core.types import (
    AudioConfig,
    QuantizationConfig,
    LayerSplitConfig
)
from ..hyperparameters import HyperparameterConfig
from .base import LLMBackend, AudioBackend

# --- Conditional Imports for Backends ---

# LLM Backends
try:
    from .local_hf import LocalHFBackend
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    LocalHFBackend = None
    TRANSFORMERS_AVAILABLE = False

try:
    from .local_gguf import LlamaCppBackend, LLAMACPP_AVAILABLE, GGUFConfig
except ImportError:
    LlamaCppBackend = None
    LLAMACPP_AVAILABLE = False
    GGUFConfig = None # type: ignore

try:
    from .openai_llm import OpenAIBackend, OPENAI_AVAILABLE
except ImportError:
    OpenAIBackend = None
    OPENAI_AVAILABLE = False

try:
    from .google_llm import GoogleAIBackend, GOOGLE_AI_AVAILABLE
except ImportError:
    GoogleAIBackend = None
    GOOGLE_AI_AVAILABLE = False

try:
    from .anthropic_llm import AnthropicBackend, ANTHROPIC_AVAILABLE
except ImportError:
    AnthropicBackend = None
    ANTHROPIC_AVAILABLE = False

# Audio Backends
try:
    from .local_audio import LocalAudioBackend
    TTS_TRANSFORMERS_AVAILABLE = True # Assumes transformers is needed
except ImportError as e:
    LocalAudioBackend = None
    TTS_TRANSFORMERS_AVAILABLE = False
    # Log the actual import error for debugging
    import logging
    logging.getLogger(__name__).warning(f"LocalAudioBackend not available: {e}")
    
try:
    from .openai_audio import OpenAIAudioBackend as OpenAITTSBackend
    TTS_OPENAI_AVAILABLE = True # This relies on 'openai' package
except ImportError:
    OpenAITTSBackend = None
    TTS_OPENAI_AVAILABLE = False
    
# ... Import other backends as they are created ...

logger = get_logger_conf(__name__)

def get_llm_backend(
    provider: str,
    model_specifier: str,
    api_keys: Dict[str, str],
    hyperparameters: HyperparameterConfig,
    model_entry: Optional['ModelEntry'] = None, # Type hint as string if circular
    quantization_config: Optional[QuantizationConfig] = None,
    layer_split_config: Optional[LayerSplitConfig] = None,
    **kwargs
) -> Optional[LLMBackend]:
    """
    Factory function to create the appropriate LLM backend.

    Args:
        provider: The name of the provider (e.g., "local_hf", "openai").
        model_specifier: The model ID, name, or path.
        api_keys: Dictionary of API keys.
        hyperparameters: The generation hyperparameters.
        model_entry: The ModelEntry object (required for local_hf).
        quantization_config: Quantization settings.
        layer_split_config: Layer splitting settings.

    Returns:
        An initialized LLMBackend instance or None if creation fails.
    """
    provider_lower = provider.lower()
    logger.info(f"Attempting to create LLM backend for provider: {provider_lower}, model: {model_specifier}")

    try:
        if provider_lower == "local_hf":
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Hugging Face 'transformers' library not installed.")
            if not model_entry:
                 # This check should ideally happen in the caller (menu/cli)
                 # but we double-check here.
                 logger.error(f"ModelEntry is required for LocalHFBackend but was not provided.")
                 raise ConfigurationError(f"Missing ModelEntry for local HF model {model_specifier}")
            
            return LocalHFBackend(
                model_id=model_specifier,
                model_entry=model_entry,
                quant_config=quantization_config or QuantizationConfig(),
                split_config=layer_split_config or LayerSplitConfig(),
                hyperparams=hyperparameters
            )

        elif provider_lower == "local_gguf":
            if not LLAMACPP_AVAILABLE:
                raise ImportError("llama-cpp-python not found. Cannot use GGUF backend.")
            
            from pathlib import Path
            gguf_path = Path(model_specifier)
            if not gguf_path.is_file():
                raise FileNotFoundError(f"GGUF model file not found: {gguf_path}")

            # Use settings from LayerSplitConfig for GGUF
            n_gpu = layer_split_config.gpu_layers if layer_split_config else -1
            n_ctx = model_entry.max_context if model_entry else (hyperparameters.max_length or 4096)

            gguf_config = GGUFConfig(
                model_path=gguf_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu,
                # Add other GGUFConfig params from LayerSplitConfig if needed (e.g., n_batch)
            )
            return LlamaCppBackend(
                config=gguf_config,
                hyperparameters=hyperparameters
            )

        elif provider_lower == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library not found. Please install 'openai'.")
            api_key = api_keys.get("openai")
            if not api_key:
                raise ConfigurationError("OpenAI API key not found in configuration.")
            return OpenAIBackend(
                api_key=api_key,
                model_specifier=model_specifier,
                hyperparams=hyperparameters
            )

        elif provider_lower == "google":
            if not GOOGLE_AI_AVAILABLE:
                raise ImportError("Google Generative AI library not found. Please install 'google-generativeai'.")
            api_key = api_keys.get("google")
            if not api_key:
                raise ConfigurationError("Google API key not found in configuration.")
            return GoogleAIBackend(
                api_key=api_key,
                model_specifier=model_specifier,
                hyperparams=hyperparameters
            )

        elif provider_lower == "anthropic":
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic library not found. Please install 'anthropic'.")
            api_key = api_keys.get("anthropic")
            if not api_key:
                raise ConfigurationError("Anthropic API key not found in configuration.")
            return AnthropicBackend(
                api_key=api_key,
                model_specifier=model_specifier,
                hyperparams=hyperparameters
            )

        else:
            logger.error(f"Unsupported LLM provider: '{provider_lower}'.")
            raise ConfigurationError(f"Unsupported LLM provider: {provider}")

    except (ImportError, ConfigurationError, FileNotFoundError) as e:
        logger.error(f"Failed to create LLM backend for {provider}: {e}", exc_info=False)
        ConsoleOutput.error(f"Error initializing LLM backend '{provider}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating LLM backend for {provider}: {e}", exc_info=True)
        ConsoleOutput.error(f"Unexpected error initializing LLM backend '{provider}': {e}")
        return None


def get_audio_backend(
    provider: str,
    model_specifier: str,
    api_keys: Dict[str, str],
    config: AudioConfig,
    layer_split_config: Optional[LayerSplitConfig] = None
) -> Optional[AudioBackend]:
    """Factory function to create the appropriate audio backend.

    Args:
        provider: The audio backend provider (e.g., "local_audio", "openai_audio").
        model_specifier: The model ID or path.
        api_keys: Dictionary of API keys.
        config: Audio configuration.
        layer_split_config: Layer splitting settings for local models.

    Returns:
        An initialized AudioBackend instance or None if creation fails.
    """
    provider_lower = provider.lower()
    logger.info(f"Attempting to initialize audio backend for provider: {provider_lower}, model: {model_specifier}")

    try:
        if provider_lower == "local_audio":
            if not TTS_TRANSFORMERS_AVAILABLE:
                raise ImportError("LocalAudioBackend not available. Check that transformers, datasets, torch, and numpy are installed.")
            return LocalAudioBackend(
                config=config,
                model_specifier=model_specifier,
                layer_split_config=layer_split_config or LayerSplitConfig()
            )
        
        elif provider_lower == "openai_audio":
            if not TTS_OPENAI_AVAILABLE:
                raise ImportError("OpenAI TTS backend not available. Please install 'openai'.")
            api_key = api_keys.get("openai") # Uses the 'openai' key
            if not api_key:
                raise ConfigurationError("OpenAI API key not found in configuration.")
            return OpenAITTSBackend(config=config, model_specifier=model_specifier, api_key=api_key)
        
        # --- Add other cloud TTS providers here ---
        # elif provider_lower == "google_tts":
        #     if not GOOGLE_TTS_AVAILABLE: raise ImportError(...)
        #     api_key = api_keys.get("google")
        #     return GoogleTTSBackend(...)

        else:
            logger.error(f"Unsupported audio provider: {provider_lower}")
            raise ConfigurationError(f"Unsupported audio provider: {provider}")

    except (ImportError, ConfigurationError) as e:
         logger.error(f"Failed to create audio backend for {provider}: {e}", exc_info=False)
         ConsoleOutput.error(f"Error initializing audio backend '{provider}': {e}")
         return None
    except Exception as e:
         logger.error(f"Unexpected error creating audio backend for {provider}: {e}", exc_info=True)
         ConsoleOutput.error(f"Unexpected error initializing audio backend '{provider}': {e}")
         return None
