# llamanote/models/backends/base.py
"""
Abstract Base Classes for LLM and Audio Backends
"""

import abc
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Assuming core types are defined
from ...core.types import (
    GenerationResult,
    AudioConfig,
    AudioResult
)
from ..hyperparameters import HyperparameterConfig
from ...core.errors import ModelLoadError
from ...utils.logger import get_logger_conf
from ...utils.decorators import log_execution_time
from ...utils.helpers import estimate_tokens

class LLMBackend(abc.ABC):
    """Abstract base class for all Language Model backends."""

    def __init__(self, provider_id: str, model_specifier: str, hyperparams: Optional[HyperparameterConfig] = None):
        self._provider_id = provider_id
        self.model_specifier = model_specifier
        self.hyperparams = hyperparams or HyperparameterConfig() # Ensure it's never None
        self.logger = get_logger_conf(f"llamanote.models.backends.{self.__class__.__name__}")
        self.model_handle: Any = None # Generic handle for the loaded model or API client
        self.is_loaded: bool = False

    @abc.abstractmethod
    def load(self, **kwargs) -> bool:
        """
        Load the model into memory or initialize the API client.
        Must set self.model_handle if successful.
        Returns True on success, False on failure.
        """
        pass

    @abc.abstractmethod
    def unload(self):
        """
        Unload the model from memory and/or clean up resources.
        Should set self.model_handle to None.
        """
        pass

    @abc.abstractmethod
    def _generate_request(self, 
                          prompt: str, 
                          hyperparams: HyperparameterConfig, 
                          **kwargs) -> GenerationResult:
        """
        Provider-specific implementation for generating text from a raw prompt.
        This method is called by the public 'generate' method.
        """
        pass

    @abc.abstractmethod
    def _chat_request(self, 
                      system_prompt: str, 
                      user_message: str, 
                      hyperparams: HyperparameterConfig, 
                      **kwargs) -> GenerationResult:
        """
        Provider-specific implementation for generating text from chat messages.
        This method is called by the public 'process_chat' method.
        """
        pass

    def _build_error_result(self, error: Exception, duration: float, input_tokens: int) -> GenerationResult:
        """Creates a standardized GenerationResult for errors."""
        error_msg = f"[{self.provider_identifier.upper()} Error: {str(error)}]"
        return GenerationResult(
            raw_output=error_msg,
            filtered_output=error_msg,
            input_tokens=input_tokens,
            output_tokens=0,
            generation_time=duration,
            memory_used=0,
            device_map={"provider": self.provider_identifier, "status": "error"},
            error_message=str(error)
        )

    @log_execution_time(logger_name="llamanote.models.backends")
    def generate(self, 
                 prompt: str, 
                 hyperparams: Optional[HyperparameterConfig] = None, 
                 **kwargs) -> GenerationResult:
        """
        Public method to generate text from a raw prompt.
        Handles loading, error checking, and result standardization.
        """
        start_time = time.time()
        input_tokens = estimate_tokens(prompt, self.provider_identifier) # Use helper
        
        try:
            if not self.is_loaded:
                self.logger.info(f"Backend '{self.model_identifier}' not loaded. Loading now...")
                if not self.load():
                     raise ModelLoadError("Backend failed to load/initialize.", self.model_identifier)
                self.is_loaded = True

            hp_to_use = hyperparams or self.hyperparams
            result = self._generate_request(prompt, hp_to_use, **kwargs)
            
            # Ensure time is set correctly by the decorator
            result.generation_time = time.time() - start_time 
            
            # Log performance
            if result.output_tokens > 0 and result.generation_time > 0.01:
                tps = result.output_tokens / result.generation_time
                self.logger.info(f"Generated {result.output_tokens} tokens in {result.generation_time:.2f}s ({tps:.1f} t/s)")
            
            return result

        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Generation failed after {generation_time:.2f}s: {e}", exc_info=True)
            return self._build_error_result(e, generation_time, input_tokens)

    @log_execution_time(logger_name="llamanote.models.backends")
    def process_with_chat_template(self, 
                                   system_prompt: str, 
                                   user_message: str, 
                                   hyperparams: Optional[HyperparameterConfig] = None, 
                                   **kwargs) -> GenerationResult:
        """
        Public method to generate text using a chat format.
        Handles loading, error checking, and result standardization.
        """
        start_time = time.time()
        input_tokens = estimate_tokens(system_prompt + user_message, self.provider_identifier)
        
        try:
            if not self.is_loaded:
                self.logger.info(f"Backend '{self.model_identifier}' not loaded. Loading now...")
                if not self.load():
                     raise ModelLoadError("Backend failed to load/initialize.", self.model_identifier)
                self.is_loaded = True

            hp_to_use = hyperparams or self.hyperparams
            result = self._chat_request(system_prompt, user_message, hp_to_use, **kwargs)

            # Ensure time is set correctly by the decorator
            result.generation_time = time.time() - start_time

            # Log performance
            if result.output_tokens > 0 and result.generation_time > 0.01:
                tps = result.output_tokens / result.generation_time
                self.logger.info(f"Chat generated {result.output_tokens} tokens in {result.generation_time:.2f}s ({tps:.1f} t/s)")

            return result

        except Exception as e:
            generation_time = time.time() - start_time
            self.logger.error(f"Chat generation failed after {generation_time:.2f}s: {e}", exc_info=True)
            return self._build_error_result(e, generation_time, input_tokens)

    @property
    def provider_identifier(self) -> str:
        """Returns the provider name (e.g., 'local_hf', 'openai')."""
        return self._provider_id

    @property
    def model_identifier(self) -> str:
        """Returns a unique string for the model (e.g., 'local_hf:model-id')."""
        return f"{self.provider_identifier}:{self.model_specifier}"

    def __del__(self):
        """Ensure cleanup on object deletion."""
        try:
            if self.is_loaded:
                self.unload()
        except Exception:
            pass # Suppress errors during garbage collection

# --- Abstract Audio Backend ---

class AudioBackend(abc.ABC):
    """Abstract base class for audio generation backends."""

    def __init__(self, provider_id: str, model_specifier: str, config: AudioConfig):
        self.config = config
        self._provider_id = provider_id
        self.model_specifier = model_specifier # e.g., HF ID or cloud model name
        self.logger = get_logger_conf(f"{self.__class__.__name__}")
        self.model_handle: Any = None # Generic handle for loaded model/client/pipeline
        self.is_loaded: bool = False

    @abc.abstractmethod
    def load(self, **kwargs) -> bool:
        """Load the model/pipeline or initialize the API client."""
        pass

    @abc.abstractmethod
    def unload(self):
        """Unload model/pipeline or clean up resources."""
        pass

    @abc.abstractmethod
    @log_execution_time()
    def generate_audio(self,
                      text: str,
                      output_path: Path,
                      chunk_text: bool = True,
                      **kwargs) -> Optional[AudioResult]:
        """
        Generate audio from text and save to output_path.

        Args:
            text: The text to synthesize.
            output_path: The *base* path (without extension) to save the audio.
                         The final extension will be determined by config.
            chunk_text: Whether to split long text into chunks.

        Returns:
            An AudioResult object or None on failure.
        """
        pass

    def _resolve_output_path(self, output_path: Optional[Path]) -> Path:
        """
        Resolve the output path with the correct extension based on config.

        Args:
            output_path: Optional path (with or without extension)

        Returns:
            Path with the correct extension from config.output_format
        """
        if output_path is None:
            raise ValueError("output_path cannot be None")

        # Ensure proper extension
        target_ext = f".{self.config.output_format.lower()}"
        if output_path.suffix.lower() != target_ext:
            output_path = output_path.with_suffix(target_ext)

        return output_path

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks based on config.chunk_size.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        chunk_size = self.config.chunk_size
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        # Split by sentences or paragraphs when possible
        # For now, simple chunking with word boundary awareness
        words = text.split()
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        # Add remaining words
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def _combine_audio(self, audio_arrays: List, sample_rate: int):
        """
        Combine multiple audio arrays into a single array.

        Args:
            audio_arrays: List of numpy arrays containing audio data
            sample_rate: Sample rate of the audio

        Returns:
            Combined numpy array or None on failure
        """
        try:
            import numpy as np

            if not audio_arrays:
                return None

            if len(audio_arrays) == 1:
                return audio_arrays[0]

            # Add small silence between chunks (100ms)
            silence_samples = int(sample_rate * 0.1)
            silence = np.zeros(silence_samples, dtype=np.float32)

            # Combine with silence between
            combined = []
            for i, audio in enumerate(audio_arrays):
                combined.append(audio)
                if i < len(audio_arrays) - 1:  # Don't add silence after last chunk
                    combined.append(silence)

            return np.concatenate(combined)

        except Exception as e:
            self.logger.error(f"Failed to combine audio arrays: {e}")
            return None

    @property
    def provider_identifier(self) -> str:
        """Returns the provider name (e.g., 'local_audio', 'openai_audio')."""
        return self._provider_id

    @property
    def model_identifier(self) -> str:
        """Returns a unique string for the model (e.g., 'local_audio:model-id')."""
        return f"{self.provider_identifier}:{self.model_specifier}"

    def __del__(self):
        """Ensure cleanup on object deletion."""
        try:
            if self.is_loaded:
                self.unload()
        except Exception:
            pass
