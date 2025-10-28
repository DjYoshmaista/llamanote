# llamanote/models/backends/anthropic_llm.py
"""
Anthropic (Claude) LLM Backend
"""

import time
from typing import Optional, Dict, Any

from .base import LLMBackend
from ...core.types import GenerationResult
from ..hyperparameters import HyperparameterConfig
from ...utils.logger import get_logger_conf, ConsoleOutput
from .mapper import HyperparamMapper # Import the mapper
from ...core.errors import ModelLoadError, GenerationError

# Try importing Anthropic library
try:
    import anthropic
    from anthropic import Anthropic, AuthenticationError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    Anthropic = None # type: ignore
    AuthenticationError = None # type: ignore
    ANTHROPIC_AVAILABLE = False

logger = get_logger_conf(__name__)

class AnthropicBackend(LLMBackend):
    """Implementation for Anthropic (Claude) API."""

    def __init__(self,
                 api_key: str,
                 model_specifier: str,
                 hyperparams: Optional[HyperparameterConfig] = None):
        
        super().__init__(provider_id="anthropic",
                         model_specifier=model_specifier,
                         hyperparams=hyperparams)
                         
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic library not installed. Run 'pip install anthropic'")

        self.api_key = api_key
        # self.model_handle (defined in base class) will hold the Anthropic client

    def load(self, **kwargs) -> bool:
        """Initialize the Anthropic client and test the API key."""
        if self.model_handle:
            return True
            
        try:
            self.model_handle = Anthropic(api_key=self.api_key)
            # Test API key with a simple, cheap call (count tokens)
            self.model_handle.count_tokens("test connection")
            logger.info(f"Anthropic API client initialized for model {self.model_specifier} and key verified.")
            return True
        except AuthenticationError as e:
             logger.error(f"Anthropic API key is invalid: {e}")
             ConsoleOutput.error("Anthropic API key is invalid. Please check your configuration.")
             raise ModelLoadError(f"Anthropic API key invalid: {e}", self.model_specifier) from e
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client or verify key: {e}", exc_info=True)
            ConsoleOutput.error(f"Anthropic connection failed: {e}. Check key and network access.")
            raise ModelLoadError(f"Anthropic connection failed: {e}", self.model_specifier) from e

    def unload(self):
        """No explicit unloading needed, just clear client reference."""
        if self.model_handle is not None:
             del self.model_handle # Remove reference
             self.model_handle = None
             logger.debug("Anthropic client reference cleared.")

    def _generate_request(self,
                          prompt: str,
                          hp_override: HyperparameterConfig,
                          **kwargs) -> GenerationResult:
        """Generate text using Anthropic (treating raw prompt as user message)."""
        # Anthropic's Messages API requires a 'user' role, not just a raw prompt.
        # We also provide a generic system prompt.
        default_system = "You are a helpful assistant. Process the user's request."
        return self._chat_request(
            system_prompt=default_system,
            user_message=prompt,
            hp_override=hp_override,
            **kwargs
        )

    def _chat_request(self,
                      system_prompt: str,
                      user_message: str,
                      hp_override: HyperparameterConfig,
                      **kwargs) -> GenerationResult:
        """Generate text using Anthropic with system and user prompts (Messages API)."""
        if self.model_handle is None:
            raise ModelLoadError("Anthropic client not initialized. Call load() first.", self.model_specifier)
            
        api_params = HyperparamMapper.get_anthropic_config(hp_override)

        messages = [
            {"role": "user", "content": user_message}
        ]

        try:
            response = self.model_handle.messages.create(
                model=self.model_specifier,
                system=system_prompt if system_prompt else None, # Pass system prompt
                messages=messages,
                **api_params
            )

            raw_output = ""
            # Handle response structure (content is a list of blocks)
            if response.content and isinstance(response.content, list):
                # Concatenate text from all text blocks
                text_blocks = [block.text for block in response.content if block.type == 'text']
                raw_output = "\n".join(text_blocks)
            
            if not raw_output and response.stop_reason:
                 self.logger.warning(f"Anthropic generated 0 tokens. Finish Reason: {response.stop_reason}")
                 if response.stop_reason == 'max_tokens':
                     raw_output = "[Anthropic Error: Output truncated, max_tokens reached]"
                 else:
                     raw_output = f"[Anthropic Error: {response.stop_reason}]"

            # Token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            
            return GenerationResult(
                raw_output=raw_output,
                filtered_output=raw_output, # Filtering done in pipeline
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_time=0.0, # Set by base decorator
                memory_used=0,
                device_map={"provider": "anthropic"}
            )
        
        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}", exc_info=True)
            raise GenerationError(f"Anthropic API call failed: {e}", self.model_specifier) from e
