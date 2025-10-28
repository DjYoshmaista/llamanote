# llamanote/models/backends/google_llm.py
"""
Google AI (Gemini) LLM Backend
"""

import time
from typing import Optional, Dict, Any

from .base import LLMBackend
from ...core.types import GenerationResult
from ..hyperparameters import HyperparameterConfig
from ...utils.logger import get_logger_conf, ConsoleOutput
from .mapper import HyperparamMapper # Import the mapper
from ...core.errors import ModelLoadError, GenerationError

# Try importing Google AI library
try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    genai = None
    google_exceptions = None
    GOOGLE_AI_AVAILABLE = False

logger = get_logger_conf(__name__)

class GoogleAIBackend(LLMBackend):
    """Implementation for Google AI (Gemini) API."""

    def __init__(self,
                 api_key: str,
                 model_specifier: str,
                 hyperparams: Optional[HyperparameterConfig] = None):
        
        super().__init__(provider_id="google",
                         model_specifier=model_specifier,
                         hyperparams=hyperparams)
                         
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed. Run 'pip install google-generativeai'")

        self.api_key = api_key
        # self.model_handle (defined in base class) will hold the GenerativeModel object
        self._system_prompt_cache: Optional[str] = None # Cache system prompt to avoid re-init

    def load(self, **kwargs) -> bool:
        """Initialize the Google AI client and model."""
        if self.model_handle:
            return True
            
        try:
            genai.configure(api_key=self.api_key)
            # Initialize the base model reference here, without system prompt yet
            # System prompt will be applied in _chat_request
            self.model_handle = genai.GenerativeModel(self.model_specifier)
            
            # Test the model/key with a simple, cheap call
            self.model_handle.count_tokens("test connection")
            
            logger.info(f"Initialized GoogleAIBackend for model: {self.model_specifier} and key verified.")
            return True
        except (google_exceptions.PermissionDenied, google_exceptions.InvalidArgument) as e:
             logger.error(f"Google API key is invalid or API not enabled: {e}")
             ConsoleOutput.error("The provided Google API key is invalid or the Generative Language API is not enabled.")
             raise ModelLoadError(f"Google API key invalid: {e}", self.model_specifier) from e
        except Exception as e:
            logger.error(f"Failed to initialize Google AI client or verify key: {e}", exc_info=True)
            ConsoleOutput.error(f"Google API connection failed: {e}. Check key and network access.")
            raise ModelLoadError(f"Google API connection failed: {e}", self.model_specifier) from e

    def unload(self):
        """No explicit unloading needed, just clear model reference."""
        if self.model_handle is not None:
             del self.model_handle
             self.model_handle = None
             self._system_prompt_cache = None
             logger.debug("Google AI model reference cleared.")

    def _get_configured_model(self, system_prompt: Optional[str]) -> Any:
        """
        Gets the GenerativeModel instance, re-initializing if system prompt changes.
        """
        if self.model_handle is None:
             raise ModelLoadError("Google AI Model not loaded. Call load() first.", self.model_specifier)

        # Google's API ties system_instruction to the model object instance
        current_system_prompt = system_prompt or ""
        cached_system_prompt = self._system_prompt_cache or ""

        if current_system_prompt != cached_system_prompt:
            logger.info(f"Re-configuring Google AI model with system prompt: {current_system_prompt[:50]}...")
            try:
                 # Create a new model instance with the system instruction
                 self.model_handle = genai.GenerativeModel(
                     self.model_specifier,
                     system_instruction=current_system_prompt if current_system_prompt else None
                 )
                 # Verify it works
                 self.model_handle.count_tokens("test prompt")
                 self._system_prompt_cache = current_system_prompt
            except Exception as e:
                 logger.error(f"Failed to set system prompt for Google AI: {e}. Prepending to user message.", exc_info=True)
                 # Fallback: create model without system prompt
                 self.model_handle = genai.GenerativeModel(self.model_specifier)
                 self._system_prompt_cache = None # Mark as failed
                 # The caller (_chat_request) will handle prepending
                 
        return self.model_handle

    def _generate_request(self,
                          prompt: str,
                          hp_override: HyperparameterConfig,
                          **kwargs) -> GenerationResult:
        """Generate text using Google AI (treating raw prompt as user message)."""
        return self._chat_request(
            system_prompt="",
            user_message=prompt,
            hp_override=hp_override,
            **kwargs
        )

    def _chat_request(self,
                      system_prompt: str,
                      user_message: str,
                      hp_override: HyperparameterConfig,
                      **kwargs) -> GenerationResult:
        """Generate text using Google AI with system and user prompts."""
        
        # Get the model, potentially re-initializing it with the system prompt
        try:
             model_to_use = self._get_configured_model(system_prompt)
             
             # If system prompt failed to set, prepend it
             if self._system_prompt_cache is None and system_prompt:
                  user_message = f"System Instruction: {system_prompt}\n\nUser Request: {user_message}"
                  
        except Exception as e:
             raise GenerationError(f"Failed to configure Google model: {e}", self.model_specifier) from e
             
        generation_config = genai.types.GenerationConfig(
            **HyperparamMapper.get_google_config(hp_override)
        )
        
        # Estimate input tokens
        input_tokens = 0
        try:
            input_tokens = model_to_use.count_tokens(user_message).total_tokens
            if self._system_prompt_cache: # Add system prompt tokens if set
                 input_tokens += model_to_use.count_tokens(self._system_prompt_cache).total_tokens
        except Exception as e:
            logger.warning(f"Could not count input tokens: {e}")
            input_tokens = (len(system_prompt) + len(user_message)) // 4 # Fallback

        try:
            # Use generate_content for flexibility
            response = model_to_use.generate_content(
                user_message,
                generation_config=generation_config
            )

            raw_output = ""
            
            # Handle potential safety blocks or empty responses
            if not response.parts:
                 finish_reason = "Unknown"
                 if hasattr(response, 'candidates') and response.candidates:
                      finish_reason = response.candidates[0].finish_reason.name
                 if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                      finish_reason = f"Prompt Blocked: {response.prompt_feedback.block_reason.name}"
                 
                 logger.warning(f"Google AI response blocked or empty. Finish Reason: {finish_reason}")
                 raw_output = f"[Google API Error: {finish_reason}]"
                 raise GenerationError(raw_output, self.model_specifier)
            
            raw_output = response.text
            
            # Count output tokens
            output_tokens = 0
            try:
                output_tokens = model_to_use.count_tokens(raw_output).total_tokens
            except Exception as e:
                logger.warning(f"Could not count output tokens: {e}")
                output_tokens = len(raw_toutput) // 4 # Fallback
            
            return GenerationResult(
                raw_output=raw_output,
                filtered_output=raw_output, # Filtering done in pipeline
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=0.0, # Set by base decorator
                memory_used=0,
                device_map={"provider": "google"}
            )

        except Exception as e:
            # Catch exceptions from generate_content call
            logger.error(f"Google AI API call failed: {e}", exc_info=True)
            if "API key not valid" in str(e):
                 ConsoleOutput.error("Google API key is invalid.")
            raise GenerationError(f"Google API call failed: {e}", self.model_specifier) from e
