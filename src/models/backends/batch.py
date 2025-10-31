# llamanote/models/backends/batch.py
"""
Batch Processing Module
Handles batch processing of text chunks through LLM backends.
"""

from typing import List, Optional
from ...utils.logger import get_logger_conf, LoggingProgress
from ...core.types import GenerationResult
from ..hyperparameters import HyperparameterConfig
from .base import LLMBackend

logger = get_logger_conf(__name__)


class BatchProcessor:
    """Processes multiple text chunks through an LLM backend."""

    def __init__(self, backend: LLMBackend):
        """
        Initialize batch processor.

        Args:
            backend: The LLM backend to use for processing.
        """
        self.backend = backend
        self.logger = get_logger_conf(f"{__name__}.BatchProcessor")

    def process_batch(
        self,
        texts: List[str],
        system_prompt: str,
        hyperparams: Optional[HyperparameterConfig] = None,
        remove_thinking: bool = True,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Process a batch of text chunks sequentially.

        Args:
            texts: List of text chunks to process.
            system_prompt: System prompt to use for all chunks.
            hyperparams: Hyperparameters for generation.
            remove_thinking: Whether to remove thinking tags from output.
            **kwargs: Additional arguments to pass to the backend.

        Returns:
            List of GenerationResult objects, one per chunk.
        """
        if not texts:
            self.logger.warning("Empty text list provided to batch processor.")
            return []

        self.logger.info(f"Processing batch of {len(texts)} chunks...")
        results = []

        with LoggingProgress(self.logger, "Processing chunks", len(texts)) as progress:
            for i, text in enumerate(texts):
                try:
                    # Use chat-based processing if available
                    if hasattr(self.backend, 'process_with_chat_template'):
                        result = self.backend.process_with_chat_template(
                            system_prompt=system_prompt,
                            user_message=text,
                            hyperparams=hyperparams,
                            **kwargs
                        )
                    else:
                        # Fallback to raw generation
                        # Construct a simple prompt
                        prompt = f"{system_prompt}\n\nUser: {text}\n\nAssistant:"
                        result = self.backend.generate(
                            prompt=prompt,
                            hyperparams=hyperparams,
                            **kwargs
                        )

                    results.append(result)
                    progress.update(1)

                except Exception as e:
                    self.logger.error(f"Error processing chunk {i+1}/{len(texts)}: {e}", exc_info=True)
                    # Create an error result
                    error_result = GenerationResult(
                        raw_output=f"[Error: {str(e)}]",
                        filtered_output=f"[Error: {str(e)}]",
                        input_tokens=0,
                        output_tokens=0,
                        generation_time=0.0,
                        memory_used=0,
                        device_map={},
                        error_message=str(e)
                    )
                    results.append(error_result)
                    progress.update(1)

        self.logger.info(f"Batch processing complete: {len(results)} results")
        return results
