# llamanote/models/backends/openai_llm.py
"""
OpenAI API LLM Backend
"""

import time
from typing import Optional, Dict, Any

from .base import LLMBackend
from ...core.types import GenerationResult
from ..hyperparameters import HyperparameterConfig
from ...utils.logger import get_logger_conf, ConsoleOutput
from .mapper import HyperparamMapper # Import the mapper
from ...core.errors import ModelLoadError, GenerationError

# Try importing OpenAI library
try:
    import openai
    from openai import OpenAI, AuthenticationError
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None
    OpenAI = None # type: ignore
    AuthenticationError = None # type: ignore
    OPENAI_AVAILABLE = False

logger = get_logger_conf(__name__)

class OpenAIBackend(LLMBackend):
    """Implementation for OpenAI API."""

    def __init__(self,
                 api_key: str,
                 model_specifier: str = "gpt-4o",
                 hyperparams: Optional[HyperparameterConfig] = None):
        
        super().__init__(provider_id="openai",
                         model_specifier=model_specifier,
                         hyperparams=hyperparams)
                         
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI library not installed. Run 'pip install openai'")

        self.api_key = api_key
        # self.model_handle (defined in base class) will hold the OpenAI client

    def load(self, **kwargs) -> bool:
        """Initialize the OpenAI client and test the API key."""
        if self.model_handle:
            return True
            
        try:
            self.model_handle = OpenAI(api_key=self.api_key)
            self.model_handle.models.list()  # Test API key by listing models
            logger.info("OpenAI API client initialized and key verified.")
            return True
        except AuthenticationError as e:
             logger.error(f"OpenAI API key is invalid: {e}")
             ConsoleOutput.error("OpenAI API key is invalid. Please check your configuration.")
             raise ModelLoadError(f"OpenAI API key invalid: {e}", self.model_specifier) from e
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client or verify key: {e}", exc_info=True)
            ConsoleOutput.error(f"OpenAI connection failed: {e}. Check key and network access.")
            raise ModelLoadError(f"OpenAI connection failed: {e}", self.model_specifier) from e

    def unload(self):
        """No explicit unloading needed, just clear client reference."""
        if self.model_handle is not None:
             del self.model_handle # Remove reference
             self.model_handle = None
             logger.debug("OpenAI client reference cleared.")

    def _generate_request(self,
                          prompt: str,
                          hp_override: HyperparameterConfig,
                          **kwargs) -> GenerationResult:
        """Generate text using ChatCompletion with a single user prompt."""
        # OpenAI's non-chat 'completion' endpoint is legacy.
        # We will route all 'generate' calls to the 'chat' endpoint
        # with a simple "user" message.
        logger.debug("Using _chat_request for non-chat generation call.")
        return self._chat_request(
            system_prompt="", # No system prompt
            user_message=prompt,
            hp_override=hp_override,
            **kwargs
        )

    def _chat_request(self,
                      system_prompt: str,
                      user_message: str,
                      hp_override: HyperparameterConfig,
                      **kwargs) -> GenerationResult:
        """Generate text using OpenAI API with system and user prompts."""
        if self.model_handle is None:
            raise ModelLoadError("OpenAI client not initialized. Call load() first.", self.model_specifier)
            
        api_params = HyperparamMapper.get_openai_config(hp_override)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.model_handle.chat.completions.create(
                model=self.model_specifier,
                messages=messages,
                **api_params
            )
            
            # Handle potential empty response
            raw_output = ""
            if response.choices and response.choices[0].message:
                 raw_output = response.choices[0].message.content or ""

            # Get token usage
            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0
            
            # Fallback token estimation if usage is missing
            if input_tokens == 0: input_tokens = len(user_message) // 4 + len(system_prompt) // 4
            if output_tokens == 0: output_tokens = len(raw_output) // 4

            return GenerationResult(
                raw_output=raw_output,
                filtered_output=raw_output, # Cloud models generally don't use <think> tags
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=0.0, # Set by base decorator
                memory_used=0,
                device_map={"provider": "openai"}
            )
        
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}", exc_info=True)
            raise GenerationError(f"OpenAI API call failed: {e}", self.model_specifier) from e
