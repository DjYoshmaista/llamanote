# llamanote/models/backends/batch.py
"""
Batch Processing Module
Handles batch processing of text chunks through LLM backends with checkpointing support.
"""

from typing import List, Optional, Callable, Any, Dict
from pathlib import Path
from ...utils.logger import get_logger_conf, LoggingProgress, DualProgressTracker
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
        checkpoint_callback: Optional[Callable[[int, List[GenerationResult], Dict[str, Any]], None]] = None,
        checkpoint_interval: int = 10,
        resume_from_chunk: Optional[int] = None,
        resume_results: Optional[List[GenerationResult]] = None,
        dual_tracker: Optional[DualProgressTracker] = None,
        **kwargs
    ) -> List[GenerationResult]:
        """
        Process a batch of text chunks sequentially with checkpointing support.

        Args:
            texts: List of text chunks to process.
            system_prompt: System prompt to use for all chunks.
            hyperparams: Hyperparameters for generation.
            remove_thinking: Whether to remove thinking tags from output.
            checkpoint_callback: Optional callback function to save checkpoint.
                                 Called with (chunk_index, results_so_far, extra_data)
            checkpoint_interval: Save checkpoint every N chunks
            resume_from_chunk: If provided, resume from this chunk index (skip earlier chunks)
            resume_results: Optional pre-computed results for chunks before resume_from_chunk
            dual_tracker: Optional dual progress tracker for displaying progress
            **kwargs: Additional arguments to pass to the backend.

        Returns:
            List of GenerationResult objects, one per chunk.
        """
        if not texts:
            self.logger.warning("Empty text list provided to batch processor.")
            return []

        self.logger.info(f"Processing batch of {len(texts)} chunks...")

        # Determine starting point
        start_index = resume_from_chunk if resume_from_chunk is not None else 0
        if start_index > 0:
            self.logger.info(f"Resuming from chunk {start_index}/{len(texts)}")

        results = []

        # Initialize dual tracker stage progress BEFORE entering context
        if dual_tracker:
            dual_tracker.set_stage_progress(start_index, len(texts), "Processing chunks")

        with LoggingProgress(self.logger, "Processing chunks", len(texts), dual_tracker=dual_tracker) as progress:
            # If resuming, use actual results from checkpoint or create placeholders
            if start_index > 0:
                if resume_results and len(resume_results) >= start_index:
                    # Use actual results from checkpoint
                    results.extend(resume_results[:start_index])
                    self.logger.info(f"Loaded {start_index} results from checkpoint")
                else:
                    # Create placeholder results (fallback if no resume_results provided)
                    self.logger.warning(f"No resume results provided, creating placeholders for {start_index} chunks")
                    for i in range(start_index):
                        results.append(GenerationResult(
                            raw_output="[Skipped - loaded from checkpoint]",
                            filtered_output="[Skipped - loaded from checkpoint]",
                            input_tokens=0,
                            output_tokens=0,
                            generation_time=0.0,
                            memory_used=0,
                            device_map={},
                            error_message=None
                        ))
                # Update progress to reflect skipped chunks
                for _ in range(start_index):
                    progress.update(1)

            for i in range(start_index, len(texts)):
                text = texts[i]
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

                    # Save checkpoint if callback provided and interval reached
                    if checkpoint_callback and checkpoint_interval > 0:
                        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(texts):
                            self.logger.debug(f"Saving checkpoint at chunk {i + 1}/{len(texts)}")
                            checkpoint_callback(i + 1, results, {"system_prompt": system_prompt})

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
