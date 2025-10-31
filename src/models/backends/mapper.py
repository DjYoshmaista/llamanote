# llamanote/models/backends/mapper.py
"""
Hyperparameter Mapping Utility

Translates the generic HyperparameterConfig dataclass into the specific
keyword arguments required by different LLM APIs (OpenAI, Anthropic, etc.)
and local libraries (transformers, llama-cpp-python).
"""

from typing import Dict, Any, Optional, List, Tuple
import torch

from ..hyperparameters import HyperparameterConfig
from ...core.types import AudioConfig
from ...utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

class HyperparamMapper:
    """
    Handles mapping from the internal HyperparameterConfig to vendor-specific
    generation arguments.
    """

    @staticmethod
    def _apply_bounds(value: float, bounds: Tuple[float, float]) -> float:
        """Helper to clamp a value within a min/max bound."""
        return max(bounds[0], min(bounds[1], value))

    @staticmethod
    def get_hf_transformers_config(
        hp: HyperparameterConfig,
        tokenizer: Any, # AutoTokenizer instance
        model_max_context: int
    ) -> Dict[str, Any]:
        """
        Maps HyperparameterConfig to a dict suitable for
        Hugging Face Transformers `model.generate()`.
        """
        
        # Calculate max_new_tokens dynamically if not explicitly set
        # This requires tokenizing the prompt, which happens in the backend.
        # We'll just pass 'max_new_tokens' and let the backend resolve it.
        
        gen_kwargs = {
            "max_new_tokens": hp.max_new_tokens, # Can be None, handled by backend
            "temperature": hp.temperature if hp.do_sample else 0.0, # temp=0.0 can be problematic, 1e-9 often better
            "top_p": hp.top_p if hp.do_sample else 1.0,
            "top_k": hp.top_k if hp.do_sample else 50, # Default to 50 if sampling not explicitly defined?
            "do_sample": hp.do_sample,
            "repetition_penalty": hp.repetition_penalty,
            "length_penalty": hp.length_penalty,
            "no_repeat_ngram_size": hp.no_repeat_ngram_size,
            "num_beams": hp.num_beams,
            "num_beam_groups": hp.num_beam_groups,
            "diversity_penalty": hp.diversity_penalty,
            "early_stopping": hp.early_stopping,
            "num_return_sequences": hp.num_return_sequences,
            "use_cache": hp.use_cache,
        }

        # Handle token IDs
        if hp.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = hp.pad_token_id
        elif tokenizer.pad_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
            
        if hp.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = hp.eos_token_id
        elif tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
            
        if hp.bos_token_id is not None:
            gen_kwargs["bos_token_id"] = hp.bos_token_id
        elif tokenizer.bos_token_id is not None:
            gen_kwargs["bos_token_id"] = tokenizer.bos_token_id
            
        if hp.forced_bos_token_id is not None:
            gen_kwargs["forced_bos_token_id"] = hp.forced_bos_token_id
        if hp.forced_eos_token_id is not None:
            gen_kwargs["forced_eos_token_id"] = hp.forced_eos_token_id

        # Clean out None values which can cause issues
        final_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        # Special handling for sampling vs. greedy
        if not hp.do_sample:
            final_kwargs["temperature"] = 1.0 # HF recommends temp=1, top_k=1 for greedy
            final_kwargs["top_k"] = 1
            final_kwargs.pop("top_p", None) # Remove top_p for greedy
            final_kwargs.pop("typical_p", None) # Remove typical_p for greedy

        return final_kwargs


    @staticmethod
    def get_llama_cpp_config(
        hp: HyperparameterConfig,
        n_ctx: int
    ) -> Dict[str, Any]:
        """
        Maps HyperparameterConfig to a dict suitable for
        llama-cpp-python `model()`.
        """
        gen_kwargs = {
            "max_tokens": hp.max_new_tokens or n_ctx // 2, # Default to half context
            "temperature": hp.temperature if hp.do_sample else 0.0,
            "top_p": hp.top_p if hp.do_sample else 1.0,
            "top_k": hp.top_k if hp.do_sample else 40, # 40 is llama.cpp default if do_sample=True
            "repeat_penalty": hp.repetition_penalty,
            # "stop": stop or [], # Stop tokens handled separately in backend
            # "grammar": grammar, # Grammar handled separately in backend
        }
        
        if not hp.do_sample:
            gen_kwargs["temperature"] = 0.0
            gen_kwargs["top_k"] = 1 # Set to 1 for greedy

        # Clean out None values (except max_tokens which has a default)
        final_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        return final_kwargs


    @staticmethod
    def get_openai_config(hp: HyperparameterConfig) -> Dict[str, Any]:
        """Maps HyperparameterConfig to OpenAI's API parameters."""
        params = {}
        
        # Max tokens
        if hp.max_new_tokens is not None:
            params["max_tokens"] = hp.max_new_tokens
        
        # Temperature
        if not hp.do_sample:
            params["temperature"] = 0.0
        elif hp.temperature is not None:
            params["temperature"] = HyperparamMapper._apply_bounds(hp.temperature, (0.0, 2.0))
        
        # Top-p
        if hp.top_p is not None and hp.do_sample:
            # OpenAI API: "We recommend altering this or temperature but not both."
            # If temp is not default, top_p is often ignored.
            # We'll send it anyway, but it might not be used.
            # Value of 1.0 is default and means "don't use".
            top_p_val = HyperparamMapper._apply_bounds(hp.top_p, (0.001, 1.0))
            if top_p_val < 1.0:
                 params["top_p"] = top_p_val
        
        # Top-k is NOT supported by OpenAI ChatCompletion API
        
        # Repetition Penalty (maps to presence_penalty for OpenAI)
        if hp.repetition_penalty is not None and hp.repetition_penalty != 1.0:
            # This is an approximation. repetition_penalty penalizes based on presence AND frequency.
            # presence_penalty only penalizes presence.
            # We map 1.0 (default) -> 0.0 (default)
            # Map 1.1 -> 0.1, 2.0 -> 1.0 etc.
            presence_penalty = max(0.0, min(2.0, hp.repetition_penalty - 1.0))
            if abs(presence_penalty) > 1e-6:
                params["presence_penalty"] = presence_penalty

        # no_repeat_ngram_size is NOT supported
        
        # num_return_sequences (maps to 'n')
        if hp.num_return_sequences > 1:
            params["n"] = hp.num_return_sequences

        # stop (handled by backend)

        return params


    @staticmethod
    def get_google_config(hp: HyperparameterConfig) -> Dict[str, Any]:
        """Maps HyperparameterConfig to Google's GenerationConfig."""
        params = {}
        
        # Max tokens
        if hp.max_new_tokens is not None:
            params["max_output_tokens"] = hp.max_new_tokens
        
        # Temperature
        if not hp.do_sample:
            params["temperature"] = 0.0
        elif hp.temperature is not None:
            params["temperature"] = HyperparamMapper._apply_bounds(hp.temperature, (0.0, 1.0)) # Google's range is 0.0-1.0
        
        # Top-p
        if hp.top_p is not None and hp.do_sample and hp.top_p < 1.0:
            params["top_p"] = HyperparamMapper._apply_bounds(hp.top_p, (0.001, 0.999)) # Cannot be 1.0

        # Top-k
        if hp.top_k > 0 and hp.do_sample:
            params["top_k"] = hp.top_k
            
        # num_return_sequences (maps to 'candidate_count')
        if hp.num_return_sequences > 1:
            params["candidate_count"] = hp.num_return_sequences
        
        # stop (handled by backend)
        # repetition_penalty is NOT supported
        # no_repeat_ngram_size is NOT supported
        
        return params

    @staticmethod
    def get_anthropic_config(hp: HyperparameterConfig) -> Dict[str, Any]:
        """Maps HyperparameterConfig to Anthropic's API parameters."""
        params = {}
        
        # Max tokens (REQUIRED by Anthropic)
        params["max_tokens"] = hp.max_new_tokens or 4096 # Anthropic requires this, set a default
        
        # Temperature
        if not hp.do_sample:
            params["temperature"] = 0.0
        elif hp.temperature is not None:
            params["temperature"] = HyperparamMapper._apply_bounds(hp.temperature, (0.0, 1.0)) # Anthropic's range
        
        # Top-p
        if hp.top_p is not None and hp.do_sample and hp.top_p < 1.0:
            params["top_p"] = HyperparamMapper._apply_bounds(hp.top_p, (0.001, 1.0))
        
        # Top-k
        if hp.top_k > 0 and hp.do_sample:
            params["top_k"] = hp.top_k

        # repetition_penalty is NOT supported
        # no_repeat_ngram_size is NOT supported
        # num_return_sequences is NOT supported
        
        return params

    @staticmethod
    def get_audio_config(hp: AudioConfig, provider: str) -> Dict[str, Any]:
        """Maps AudioConfig to provider-specific TTS API parameters."""
        if provider == "openai_audio":
            params = {
                "model": hp.model_specifier,
                "voice": hp.cloud_voice or "alloy",
                "response_format": hp.output_format if hp.output_format in ["mp3", "opus", "aac", "flac"] else "mp3",
                "speed": HyperparamMapper._apply_bounds(hp.speed, (0.25, 4.0))
            }
            # Note: pitch and volume_normalize are post-processing, not API params
            return params
        
        if provider == "local_audio":
            # For local, settings are often passed to pipeline/model init,
            # but some generation-time args might be needed.
            params = {}
            # e.g., if using SpeechT5, speaker_embedding might be passed here
            # This is handled in the LocalAudioBackend's generate method itself.
            return params
            
        logger.warning(f"No hyperparameter mapping defined for audio provider: {provider}")
        return {}
