# src/core/stage_analyzer.py
"""
Stage Analyzer for Dynamic Model Lifecycle

Analyzes pipeline stages to determine which models are required and when
they can be safely loaded/unloaded to optimize memory usage.
"""

from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple
from enum import Enum

from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class ModelType(Enum):
    """Types of models that can be loaded."""
    TEXT_LLM = "text_llm"
    AUDIO_TTS = "audio_tts"


@dataclass
class ModelLifecycleEvent:
    """Represents a model load or unload event."""
    stage: str
    model_type: ModelType
    action: str  # "load" or "unload"
    reason: str  # Human-readable explanation


@dataclass
class ModelLifecyclePlan:
    """Complete plan for model loading/unloading throughout pipeline execution."""
    events: List[ModelLifecycleEvent] = field(default_factory=list)
    stages_requiring_text: Set[str] = field(default_factory=set)
    stages_requiring_audio: Set[str] = field(default_factory=set)

    def get_events_before_stage(self, stage: str) -> List[ModelLifecycleEvent]:
        """Get all lifecycle events that should occur before a stage."""
        events = []
        for event in self.events:
            if event.stage == stage:
                events.append(event)
        return events

    def should_load_text_model_before(self, stage: str) -> bool:
        """Check if text model should be loaded before this stage."""
        for event in self.get_events_before_stage(stage):
            if event.model_type == ModelType.TEXT_LLM and event.action == "load":
                return True
        return False

    def should_unload_text_model_after(self, stage: str) -> bool:
        """Check if text model should be unloaded after this stage."""
        # Look for unload events in the NEXT stage
        stage_idx = None
        for i, event in enumerate(self.events):
            if event.stage == stage:
                stage_idx = i
                break

        if stage_idx is not None and stage_idx + 1 < len(self.events):
            next_event = self.events[stage_idx + 1]
            if next_event.model_type == ModelType.TEXT_LLM and next_event.action == "unload":
                return True
        return False

    def should_load_audio_model_before(self, stage: str) -> bool:
        """Check if audio model should be loaded before this stage."""
        for event in self.get_events_before_stage(stage):
            if event.model_type == ModelType.AUDIO_TTS and event.action == "load":
                return True
        return False

    def format_plan(self) -> str:
        """Format the lifecycle plan for display."""
        lines = [
            "",
            "━" * 60,
            "📋 Model Lifecycle Plan",
            "━" * 60,
        ]

        current_stage = None
        for event in self.events:
            if event.stage != current_stage:
                current_stage = event.stage
                lines.append(f"\nStage: {current_stage}")

            action_symbol = "⬆️" if event.action == "load" else "⬇️"
            model_name = "Text LLM" if event.model_type == ModelType.TEXT_LLM else "Audio TTS"
            lines.append(f"  {action_symbol} {event.action.upper()} {model_name}: {event.reason}")

        lines.append("━" * 60)
        lines.append("")
        return "\n".join(lines)


class StageAnalyzer:
    """Analyzes pipeline stages to determine model requirements."""

    # Stages that require the text LLM model
    TEXT_MODEL_STAGES = {
        "process",      # Text generation/transformation
        "llm",          # Direct LLM inference
        "generate",     # Generation stage
        "transform",    # Text transformation
    }

    # Stages that require the audio TTS model
    AUDIO_MODEL_STAGES = {
        "audio",        # Audio generation
        "tts",          # Text-to-speech
        "synthesize",   # Audio synthesis
    }

    # Stages that don't require any model
    NO_MODEL_STAGES = {
        "extract",      # PDF/text extraction
        "chunk",        # Text chunking
        "filter",       # Response filtering
        "format",       # Output formatting
        "save",         # Saving results
    }

    @staticmethod
    def requires_text_model(stage: str) -> bool:
        """
        Determine if a pipeline stage requires the text LLM model.

        Args:
            stage: Pipeline stage name

        Returns:
            True if the stage requires a text model
        """
        return stage.lower() in StageAnalyzer.TEXT_MODEL_STAGES

    @staticmethod
    def requires_audio_model(stage: str) -> bool:
        """
        Determine if a pipeline stage requires the audio TTS model.

        Args:
            stage: Pipeline stage name

        Returns:
            True if the stage requires an audio model
        """
        return stage.lower() in StageAnalyzer.AUDIO_MODEL_STAGES

    @staticmethod
    def requires_no_model(stage: str) -> bool:
        """
        Determine if a pipeline stage requires no models at all.

        Args:
            stage: Pipeline stage name

        Returns:
            True if the stage requires no models
        """
        return stage.lower() in StageAnalyzer.NO_MODEL_STAGES

    @staticmethod
    def analyze_stages(stages: List[str]) -> Tuple[Set[str], Set[str]]:
        """
        Analyze a list of stages to determine which models are needed.

        Args:
            stages: List of pipeline stage names

        Returns:
            Tuple of (text_model_stages, audio_model_stages)
        """
        text_stages = set()
        audio_stages = set()

        for stage in stages:
            if StageAnalyzer.requires_text_model(stage):
                text_stages.add(stage)
            if StageAnalyzer.requires_audio_model(stage):
                audio_stages.add(stage)

        return text_stages, audio_stages

    @staticmethod
    def get_model_lifecycle_plan(stages: List[str]) -> ModelLifecyclePlan:
        """
        Create a complete model lifecycle plan for a pipeline run.

        This determines when models should be loaded and unloaded to minimize
        memory usage while ensuring models are available when needed.

        Args:
            stages: Ordered list of pipeline stages to execute

        Returns:
            ModelLifecyclePlan with all load/unload events
        """
        plan = ModelLifecyclePlan()
        events = []

        # Analyze which stages need which models
        text_stages, audio_stages = StageAnalyzer.analyze_stages(stages)
        plan.stages_requiring_text = text_stages
        plan.stages_requiring_audio = audio_stages

        # Track model load state
        text_model_loaded = False
        audio_model_loaded = False

        # Find first and last usage of each model
        first_text_stage = None
        last_text_stage = None
        first_audio_stage = None
        last_audio_stage = None

        for stage in stages:
            if StageAnalyzer.requires_text_model(stage):
                if first_text_stage is None:
                    first_text_stage = stage
                last_text_stage = stage

            if StageAnalyzer.requires_audio_model(stage):
                if first_audio_stage is None:
                    first_audio_stage = stage
                last_audio_stage = stage

        # Generate lifecycle events
        for stage in stages:
            stage_events = []

            # Check if we need to load text model before this stage
            if stage == first_text_stage and not text_model_loaded:
                stage_events.append(ModelLifecycleEvent(
                    stage=stage,
                    model_type=ModelType.TEXT_LLM,
                    action="load",
                    reason=f"First text generation stage ({stage})"
                ))
                text_model_loaded = True

            # Check if we need to load audio model before this stage
            if stage == first_audio_stage and not audio_model_loaded:
                stage_events.append(ModelLifecycleEvent(
                    stage=stage,
                    model_type=ModelType.AUDIO_TTS,
                    action="load",
                    reason=f"First audio generation stage ({stage})"
                ))
                audio_model_loaded = True

            events.extend(stage_events)

        # Add unload events after last usage
        # Unload text model after its last usage (if audio stages follow)
        if last_text_stage and audio_stages:
            # Find the stage immediately after last text usage
            last_text_idx = stages.index(last_text_stage)
            if last_text_idx < len(stages) - 1:
                next_stage = stages[last_text_idx + 1]
                events.append(ModelLifecycleEvent(
                    stage=next_stage,
                    model_type=ModelType.TEXT_LLM,
                    action="unload",
                    reason=f"Text processing complete, freeing memory for audio"
                ))
                text_model_loaded = False

        # Unload audio model after last usage (at pipeline end)
        if last_audio_stage:
            # Find the stage immediately after last audio usage
            last_audio_idx = stages.index(last_audio_stage)
            if last_audio_idx < len(stages) - 1:
                next_stage = stages[last_audio_idx + 1]
                events.append(ModelLifecycleEvent(
                    stage=next_stage,
                    model_type=ModelType.AUDIO_TTS,
                    action="unload",
                    reason=f"Audio generation complete"
                ))
                audio_model_loaded = False

        plan.events = events

        # Log the plan
        logger.info(f"Created model lifecycle plan with {len(events)} events")
        logger.debug(f"Text model stages: {text_stages}")
        logger.debug(f"Audio model stages: {audio_stages}")

        return plan


def create_lifecycle_plan(stages: List[str], verbose: bool = False) -> ModelLifecyclePlan:
    """
    Convenience function to create and optionally display a lifecycle plan.

    Args:
        stages: Pipeline stages to analyze
        verbose: If True, print the formatted plan

    Returns:
        ModelLifecyclePlan object
    """
    plan = StageAnalyzer.get_model_lifecycle_plan(stages)

    if verbose:
        print(plan.format_plan())

    return plan
