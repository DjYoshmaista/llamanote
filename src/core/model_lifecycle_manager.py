# src/core/model_lifecycle_manager.py
"""
Model Lifecycle Manager

Manages dynamic loading and unloading of models based on pipeline stage requirements.
Integrates with the Stage Analyzer to optimize memory usage throughout pipeline execution.
"""

import gc
import torch
from typing import Optional, Dict, Any
from pathlib import Path

from .stage_analyzer import ModelLifecyclePlan, ModelType
from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class ModelLifecycleManager:
    """
    Manages model loading and unloading throughout pipeline execution.

    This class coordinates with backends to load models only when needed
    and unload them as soon as they're no longer required, minimizing
    peak memory usage.
    """

    def __init__(self, lifecycle_plan: ModelLifecyclePlan, enable_dynamic_loading: bool = True):
        """
        Initialize the lifecycle manager.

        Args:
            lifecycle_plan: The plan created by StageAnalyzer
            enable_dynamic_loading: If False, disable dynamic loading/unloading
        """
        self.plan = lifecycle_plan
        self.enabled = enable_dynamic_loading

        # Track currently loaded models
        self.text_model_loaded: bool = False
        self.audio_model_loaded: bool = False

        # Cache references to backends (set externally)
        self.text_backend: Optional[Any] = None
        self.audio_backend: Optional[Any] = None

        # Statistics
        self.load_count: int = 0
        self.unload_count: int = 0
        self.memory_freed_mb: float = 0.0

        logger.info(f"ModelLifecycleManager initialized (enabled={self.enabled})")

    def set_text_backend(self, backend: Any):
        """Set the text LLM backend reference."""
        self.text_backend = backend
        logger.debug(f"Text backend registered: {type(backend).__name__}")

    def set_audio_backend(self, backend: Any):
        """Set the audio TTS backend reference."""
        self.audio_backend = backend
        logger.debug(f"Audio backend registered: {type(backend).__name__}")

    def _free_memory(self) -> float:
        """
        Force garbage collection and clear CUDA cache.

        Returns:
            Estimated memory freed in MB
        """
        memory_before = 0.0
        memory_after = 0.0

        if torch.cuda.is_available():
            memory_before = torch.cuda.memory_allocated() / (1024**2)

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_after = torch.cuda.memory_allocated() / (1024**2)

        freed = memory_before - memory_after
        return max(0.0, freed)

    def unload_text_model(self, reason: str = "Lifecycle management"):
        """
        Unload the text LLM model to free memory.

        Args:
            reason: Human-readable reason for unloading
        """
        if not self.enabled:
            logger.debug("Dynamic loading disabled, skipping unload")
            return

        if not self.text_model_loaded:
            logger.debug("Text model not loaded, nothing to unload")
            return

        if self.text_backend is None:
            logger.warning("Text backend not set, cannot unload")
            return

        logger.info(f"⬇️ Unloading text model: {reason}")

        try:
            # Call backend's unload method
            if hasattr(self.text_backend, 'unload'):
                self.text_backend.unload()
            else:
                # Fallback: manually clear model and tokenizer
                if hasattr(self.text_backend, 'model'):
                    self.text_backend.model = None
                if hasattr(self.text_backend, 'tokenizer'):
                    self.text_backend.tokenizer = None

            self.text_model_loaded = False
            self.unload_count += 1

            # Free memory
            freed = self._free_memory()
            self.memory_freed_mb += freed

            logger.info(f"✅ Text model unloaded successfully (freed ~{freed:.1f} MB)")

        except Exception as e:
            logger.error(f"Failed to unload text model: {e}", exc_info=True)

    def unload_audio_model(self, reason: str = "Lifecycle management"):
        """
        Unload the audio TTS model to free memory.

        Args:
            reason: Human-readable reason for unloading
        """
        if not self.enabled:
            logger.debug("Dynamic loading disabled, skipping unload")
            return

        if not self.audio_model_loaded:
            logger.debug("Audio model not loaded, nothing to unload")
            return

        if self.audio_backend is None:
            logger.warning("Audio backend not set, cannot unload")
            return

        logger.info(f"⬇️ Unloading audio model: {reason}")

        try:
            # Call backend's unload method
            if hasattr(self.audio_backend, 'unload'):
                self.audio_backend.unload()
            else:
                # Fallback: manually clear model components
                if hasattr(self.audio_backend, 'model'):
                    self.audio_backend.model = None
                if hasattr(self.audio_backend, 'vocoder'):
                    self.audio_backend.vocoder = None
                if hasattr(self.audio_backend, 'processor'):
                    self.audio_backend.processor = None

            self.audio_model_loaded = False
            self.unload_count += 1

            # Free memory
            freed = self._free_memory()
            self.memory_freed_mb += freed

            logger.info(f"✅ Audio model unloaded successfully (freed ~{freed:.1f} MB)")

        except Exception as e:
            logger.error(f"Failed to unload audio model: {e}", exc_info=True)

    def load_text_model(self, reason: str = "Lifecycle management"):
        """
        Load the text LLM model.

        Args:
            reason: Human-readable reason for loading
        """
        if not self.enabled:
            logger.debug("Dynamic loading disabled, skipping load")
            return

        if self.text_model_loaded:
            logger.debug("Text model already loaded")
            return

        if self.text_backend is None:
            logger.warning("Text backend not set, cannot load")
            return

        logger.info(f"⬆️ Loading text model: {reason}")

        try:
            # Call backend's load method
            if hasattr(self.text_backend, 'load') and not hasattr(self.text_backend, 'model'):
                self.text_backend.load()
            elif hasattr(self.text_backend, 'model') and self.text_backend.model is None:
                # Model was unloaded but backend exists, need to reload
                if hasattr(self.text_backend, 'load'):
                    self.text_backend.load()
                else:
                    logger.warning("Backend has no load() method, cannot reload")
                    return

            self.text_model_loaded = True
            self.load_count += 1

            logger.info(f"✅ Text model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load text model: {e}", exc_info=True)
            raise

    def load_audio_model(self, reason: str = "Lifecycle management"):
        """
        Load the audio TTS model.

        Args:
            reason: Human-readable reason for loading
        """
        if not self.enabled:
            logger.debug("Dynamic loading disabled, skipping load")
            return

        if self.audio_model_loaded:
            logger.debug("Audio model already loaded")
            return

        if self.audio_backend is None:
            logger.warning("Audio backend not set, cannot load")
            return

        logger.info(f"⬆️ Loading audio model: {reason}")

        try:
            # Call backend's load method
            if hasattr(self.audio_backend, 'load') and not hasattr(self.audio_backend, 'model'):
                self.audio_backend.load()
            elif hasattr(self.audio_backend, 'model') and self.audio_backend.model is None:
                # Model was unloaded but backend exists, need to reload
                if hasattr(self.audio_backend, 'load'):
                    self.audio_backend.load()
                else:
                    logger.warning("Backend has no load() method, cannot reload")
                    return

            self.audio_model_loaded = True
            self.load_count += 1

            logger.info(f"✅ Audio model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load audio model: {e}", exc_info=True)
            raise

    def handle_stage_transition(self, current_stage: str, next_stage: Optional[str] = None):
        """
        Handle model loading/unloading when transitioning between stages.

        This is called before entering a new stage to ensure proper models are loaded.

        Args:
            current_stage: The stage we're about to enter
            next_stage: The stage after current (for lookahead optimization)
        """
        if not self.enabled:
            return

        # Get lifecycle events for this stage
        events = self.plan.get_events_before_stage(current_stage)

        for event in events:
            if event.action == "load":
                if event.model_type == ModelType.TEXT_LLM:
                    self.load_text_model(event.reason)
                elif event.model_type == ModelType.AUDIO_TTS:
                    self.load_audio_model(event.reason)

            elif event.action == "unload":
                if event.model_type == ModelType.TEXT_LLM:
                    self.unload_text_model(event.reason)
                elif event.model_type == ModelType.AUDIO_TTS:
                    self.unload_audio_model(event.reason)

    def cleanup(self):
        """
        Cleanup all loaded models at the end of pipeline execution.

        This ensures no models are left in memory after the pipeline completes.
        """
        logger.info("ModelLifecycleManager cleanup starting...")

        if self.text_model_loaded:
            self.unload_text_model("Pipeline complete")

        if self.audio_model_loaded:
            self.unload_audio_model("Pipeline complete")

        # Final memory cleanup
        freed = self._free_memory()
        self.memory_freed_mb += freed

        logger.info(f"ModelLifecycleManager cleanup complete")
        logger.info(f"Total loads: {self.load_count}, Total unloads: {self.unload_count}")
        logger.info(f"Total memory freed: ~{self.memory_freed_mb:.1f} MB")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about lifecycle management."""
        return {
            "enabled": self.enabled,
            "load_count": self.load_count,
            "unload_count": self.unload_count,
            "memory_freed_mb": self.memory_freed_mb,
            "text_model_loaded": self.text_model_loaded,
            "audio_model_loaded": self.audio_model_loaded,
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure cleanup."""
        self.cleanup()
        return False
