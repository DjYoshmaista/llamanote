# llamanote/core/pipeline.py
"""
Processing Pipeline Module
Orchestrates the multi-stage workflow from file input to final output.
Refactored from original processing_pipeline.py
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..utils.logger import get_logger_conf, LoggingProgress, MemoryMonitor, ConsoleOutput, DualProgressTracker
from ..utils.progress_tracking import ProgressManager
from ..utils.decorators import log_execution_time
from ..config.settings import (
    DEFAULT_PIPELINE_STAGES, PREPROCESS_PROMPT_PODCAST, DEFAULT_SYSTEM_PROMPT,
    INCLUDE_METADATA, TIMESTAMP_OUTPUTS, DEFAULT_OUTPUT_DIR,
    MAX_CHARS_PER_FILE, MAX_PDF_SIZE_MB, DEFAULT_STAGE_WEIGHTS
)
from ..config.manager import ConfigManager # For loading defaults if needed
from .types import (
    ProcessingMode, PipelineConfig, PipelineResult, AudioConfig,
    ExtractionResult, ChunkingResult, FilterResult, PDFMetadata
)
from ..processing.pdf_extractor import PDFProcessor
from ..processing.text_preprocessor import TextPreprocessor
from ..processing.text_chunker import TextChunker
from ..processing.response_filter import ChunkedResponseFilter
from ..formatting.base_formatter import get_formatter, BaseFormatter
from ..io.file_handler import FileHandler
from ..io.checkpoints import CheckpointManager
from ..models.backends.base import LLMBackend, AudioBackend
from ..models.registry import get_model_entry, ModelEntry
from .errors import PipelineError, MissingDataError, ModelLoadError, FileProcessingError, GenerationError
from .stage_analyzer import StageAnalyzer, create_lifecycle_plan
from .model_lifecycle_manager import ModelLifecycleManager

logger = get_logger_conf(__name__)

# --- Stage Execution Helper ---
# (Moved to core/stages.py)
from .stages import PipelineStageExecutor

# --- Main Pipeline Class ---

class ProcessingPipeline:
    """Main processing pipeline for document-to-text/audio conversion."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initializes the pipeline with a given configuration.
        Backends (LLMBackend, AudioBackend) must be injected after initialization.
        """
        self.config = config or PipelineConfig()
        self.logger = get_logger_conf(f"{__name__}.Pipeline")
        
        # Backends are injected, not created
        self.llm_backend: Optional[LLMBackend] = None
        self.audio_backend: Optional[AudioBackend] = None

        # State tracking
        self.current_stage: Optional[str] = None
        self.stages_completed: List[str] = []
        self.checkpoint_name: Optional[str] = None
        self.stages_to_run: List[str] = self.config.stages if self.config.stages is not None else list(DEFAULT_PIPELINE_STAGES)

        # Config UUID for checkpoint-config linking
        self.config_uuid: Optional[str] = None

        # Model lifecycle management
        self.lifecycle_manager: Optional[ModelLifecycleManager] = None
        self._lifecycle_plan = None

        # Initialize components based on config
        self._initialize_components()
        self.logger.info(f"Initialized pipeline in {self.config.mode.value} mode")
        self.logger.info(f"Pipeline config stages: {self.config.stages}")
        self.logger.info(f"Pipeline stages to run: {self.stages_to_run}")

    def _initialize_components(self):
        """Initialize all pipeline components based on self.config."""
        self.file_handler = FileHandler(
            output_dir=self.config.output_dir or DEFAULT_OUTPUT_DIR,
            timestamp_outputs=self.config.timestamp_outputs
        )
        self.pdf_processor = PDFProcessor(
            preserve_layout=self.config.preserve_pdf_layout,
            max_chars=MAX_CHARS_PER_FILE,
            max_size_mb=MAX_PDF_SIZE_MB
        )
        self.text_preprocessor = TextPreprocessor()
        self.text_chunker = TextChunker(
            target_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
            strategy=self.config.chunking_strategy
        )
        
        # Get ModelEntry for filter configuration
        model_entry_for_filter: Optional[ModelEntry] = None
        if self.config.model_provider.startswith("local"):
            model_entry_for_filter = get_model_entry(self.config.model_specifier)
            if model_entry_for_filter is None:
                 logger.warning(f"Model {self.config.model_specifier} not in registry. Filter may not handle <think> tokens correctly.")
                 model_entry_for_filter = ModelEntry(
                     name=Path(self.config.model_specifier).name,
                     model_id=self.config.model_specifier,
                     author="unknown"
                 )
        
        self.response_filter = ChunkedResponseFilter(
            model_config=model_entry_for_filter
        )
        
        # Get formatter
        self.formatter = get_formatter(
            self.config.mode,
            self.config.markdown_style # Pass style override if present
        )
        
        # Memory monitor
        self.memory_monitor = MemoryMonitor(self.logger)
        
        # Checkpoint manager and executor are initialized per-file
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.stage_executor: Optional[PipelineStageExecutor] = None

    def set_stages_to_run(self, stages: List[str]):
        """Explicitly set which stages to run."""
        self.stages_to_run = [s for s in stages if s in DEFAULT_PIPELINE_STAGES]
        self.logger.info(f"Pipeline stages set to run: {self.stages_to_run}")
        # Reinitialize lifecycle plan if already created
        if self._lifecycle_plan is not None:
            self._initialize_lifecycle_manager()

    def _initialize_lifecycle_manager(self):
        """Initialize the model lifecycle manager based on stages to run."""
        # Create lifecycle plan
        self._lifecycle_plan = create_lifecycle_plan(self.stages_to_run, verbose=False)

        # Create lifecycle manager
        enable_dynamic = getattr(self.config.layer_split_config, 'auto_discover_splits', True)
        self.lifecycle_manager = ModelLifecycleManager(
            self._lifecycle_plan,
            enable_dynamic_loading=enable_dynamic
        )

        # Register backends if they exist
        if self.llm_backend is not None:
            self.lifecycle_manager.set_text_backend(self.llm_backend)
        if self.audio_backend is not None:
            self.lifecycle_manager.set_audio_backend(self.audio_backend)

        self.logger.info(f"Lifecycle manager initialized (enabled={enable_dynamic})")

        # Display lifecycle plan if verbose logging
        try:
            if hasattr(self.logger, 'level') and self.logger.level <= 10:  # DEBUG level
                self.logger.debug(self._lifecycle_plan.format_plan())
        except Exception:
            pass  # Skip if logger doesn't support level attribute

    def _save_checkpoint(self, input_path: Path, stage: str, data_payload: Dict[str, Any]):
        """Save a checkpoint for the current stage."""
        self.logger.debug(f"=== _save_checkpoint ENTRY: stage={stage} ===")
        if self.config.enable_checkpoints and self.checkpoint_manager:
            try:
                self.logger.debug(f"=== _save_checkpoint: Calling checkpoint_manager.save ===")
                self.checkpoint_manager.save(
                    input_path,
                    self.config,
                    stage,
                    data_payload,
                    config_uuid=self.config_uuid  # Link checkpoint to configuration
                )
                self.logger.debug(f"=== _save_checkpoint: checkpoint_manager.save returned ===")
            except Exception as e:
                self.logger.warning(f"Failed to save checkpoint for stage '{stage}': {e}")
        else:
            self.logger.debug(f"=== _save_checkpoint: Checkpoints disabled or no checkpoint manager ===")
        self.logger.debug(f"=== _save_checkpoint EXIT: stage={stage} ===")

    def _update_pipeline_progress(self, progress_manager: 'ProgressManager', completed_stages: List[str], current_stage: Optional[str] = None, stage_progress: float = 0.0):
        """
        Update pipeline progress based on completed stages and current stage progress.

        Args:
            progress_manager: The ProgressManager instance
            completed_stages: List of completed stage names
            current_stage: Name of current stage being processed
            stage_progress: Progress within current stage (0.0 to 1.0)
        """
        # Calculate completed weight
        completed_weight = sum(DEFAULT_STAGE_WEIGHTS.get(s, 0.0) for s in completed_stages)

        # Add partial progress of current stage
        if current_stage and current_stage in DEFAULT_STAGE_WEIGHTS:
            current_stage_weight = DEFAULT_STAGE_WEIGHTS[current_stage]
            completed_weight += current_stage_weight * stage_progress

        # Update pipeline progress
        progress_manager.update_pipeline(completed_weight)

    def _complete_stage(self, progress_manager: 'ProgressManager', stage_name: str):
        """
        Mark a stage as complete and update progress.

        Args:
            progress_manager: The ProgressManager instance
            stage_name: Name of the completed stage
        """
        self.logger.debug(f"=== _complete_stage ENTRY: stage={stage_name} ===")
        # Add to completed stages first
        if stage_name not in self.stages_completed:
            self.logger.debug(f"=== _complete_stage: Adding {stage_name} to stages_completed ===")
            self.stages_completed.append(stage_name)
        else:
            self.logger.debug(f"=== _complete_stage: {stage_name} already in stages_completed ===")

        # Remove the stage progress bar if it exists
        self.logger.debug(f"=== _complete_stage: Checking if {stage_name} in progress_manager.stage_tasks ===")
        if stage_name in progress_manager.stage_tasks:
            self.logger.debug(f"=== _complete_stage: Removing stage progress bar for {stage_name} ===")
            progress_manager.remove_stage(stage_name)
            self.logger.debug(f"=== _complete_stage: Stage progress bar removed ===")

        # Update pipeline progress
        self.logger.debug(f"=== _complete_stage: Calling _update_pipeline_progress ===")
        self._update_pipeline_progress(progress_manager, self.stages_completed)
        self.logger.debug(f"=== _complete_stage: _update_pipeline_progress returned ===")

        self.logger.debug(f"Completed stage: {stage_name}")
        self.logger.debug(f"=== _complete_stage EXIT: stage={stage_name} ===")

    @log_execution_time(logger_name=__name__)
    def process_file(self,
                    input_path: Path,
                    output_path_base: Optional[Path] = None) -> PipelineResult:
        """
        Process a single file through the configured pipeline stages.

        Args:
            input_path: Path to input file (.pdf, .txt, .md).
            output_path_base: Optional *base* output path (e.g., 'my_doc' or 'path/to/my_doc').
                              If None, name is derived from input_path.
                              Final extension is added by the 'save' stage.
        Returns:
            PipelineResult object.
        """
        input_path = Path(input_path)
        start_time = time.time()

        # Setup per-file state
        self.checkpoint_name = f"{input_path.stem}_{int(start_time)}"

        # Generate config UUID for checkpoint-config linking
        if self.config.enable_checkpoints:
            from ..io.checkpoint_config import CheckpointConfigManager
            config_mgr = CheckpointConfigManager()
            self.config_uuid = config_mgr.save_config(self.config)
            self.logger.info(f"Configuration UUID: {self.config_uuid}")

        # Initialize new checkpoint manager with resume capability
        resume_mode = self.config.checkpoint_resume_mode if self.config.enable_checkpoints else "disabled"
        self.checkpoint_manager = CheckpointManager(
            base_checkpoint_dir=None,  # Uses default: project_root/checkpoints
            resume_mode=resume_mode
        )

        # Initialize progress manager with stage weights
        progress_manager = ProgressManager(stage_weights=DEFAULT_STAGE_WEIGHTS)
        progress_manager.start()
        progress_manager.add_pipeline_progress()

        # Keep dual_tracker for backward compatibility with existing code
        # BUT: Disable its display to avoid console conflicts with Rich Live display
        dual_tracker = DualProgressTracker(logger=self.logger)
        dual_tracker.enabled = False  # CRITICAL: Disable to prevent console conflicts
        total_stages = len(self.stages_to_run)
        dual_tracker.set_overall_progress(0, total_stages)

        self.stage_executor = PipelineStageExecutor(self) # Re-init executor
        self.stages_completed = []
        self.current_stage = "setup"

        # Initialize lifecycle manager for this pipeline run
        if self.lifecycle_manager is None or self._lifecycle_plan is None:
            self._initialize_lifecycle_manager()

        ConsoleOutput.header(f"Processing: {input_path.name}")
        self.logger.info(f"Starting pipeline for: {input_path}")

        # Try to resume from checkpoint if enabled
        resume_data = None
        resume_from_stage = None
        resume_chunk_index = None
        restored_config = None
        if self.config.enable_checkpoints and resume_mode != "disabled":
            resume_result = self.checkpoint_manager.get_resume_checkpoint(
                input_path, self.config, self.stages_to_run
            )
            if resume_result:
                resume_from_stage, resume_data, resume_chunk_index, restored_config = resume_result
                chunk_str = f" (chunk {resume_chunk_index})" if resume_chunk_index is not None else ""
                ConsoleOutput.success(f"Resuming from stage: {resume_from_stage}{chunk_str}")
                self.logger.info(f"Loaded checkpoint data with keys: {list(resume_data.keys())}")

                # Optionally use restored config (keep current config's stages_to_run)
                if restored_config:
                    # Calculate remaining stages automatically
                    remaining_stages = self.checkpoint_manager.calculate_remaining_stages(
                        resume_from_stage, self.stages_to_run
                    )
                    self.logger.info(f"Auto-calculated remaining stages: {remaining_stages}")
                    # You could optionally apply restored_config here, but preserve stages_to_run
                    # self.config = restored_config
                    # self.config.stages = remaining_stages
        self.memory_monitor.start()

        # Initialize data payload (from resume or fresh)
        if resume_data:
            data_payload: Dict[str, Any] = resume_data.copy()
            # Ensure input_path is set
            data_payload['input_path'] = input_path
            self.logger.debug(f"Resumed data_payload keys: {list(data_payload.keys())}")
        else:
            data_payload: Dict[str, Any] = {'input_path': input_path}
        final_output_path: Optional[Path] = None
        final_audio_path: Optional[Path] = None
        success = False
        error_message: Optional[str] = None

        # Determine which stages to skip based on resume point
        stage_order = ["extract", "preprocess", "chunk", "process", "filter", "format", "save", "audio"]
        skip_stages = set()
        if resume_from_stage:
            try:
                resume_idx = stage_order.index(resume_from_stage)
                # Skip all stages before and including the resume stage
                skip_stages = set(stage_order[:resume_idx + 1])
                self.logger.info(f"Skipping stages: {skip_stages}")
            except ValueError:
                self.logger.warning(f"Unknown resume stage: {resume_from_stage}, starting fresh")

        try:
            # Track current stage index in stages_to_run (not absolute stage order)
            current_stage_num = 0

            # --- Stage 1: Extract ---
            if "extract" in self.stages_to_run and "extract" not in skip_stages:
                current_stage_num += 1
                dual_tracker.set_overall_progress(current_stage_num, len(self.stages_to_run))

                # Check if text already exists from checkpoint (resume case)
                if 'text' in data_payload and resume_from_stage:
                    try:
                        resume_idx = stage_order.index(resume_from_stage)
                        extract_idx = stage_order.index("extract")
                        if extract_idx < resume_idx:
                            self.logger.info("Extract stage already completed from checkpoint, using existing data")
                        else:
                            # Run extract normally
                            extract_result: Optional[ExtractionResult]
                            if input_path.suffix.lower() == '.pdf':
                                extract_result = self.stage_executor.execute("extract",
                                                                            lambda: self.pdf_processor.extract_text(input_path),
                                                                            data_payload, [])
                                if not extract_result: raise PipelineError("PDF extraction failed.")
                            elif input_path.suffix.lower() in ['.txt', '.md']:
                                 def read_text():
                                     try:
                                         text = input_path.read_text(encoding='utf-8')
                                         meta = PDFMetadata(file_path=input_path, num_pages=1, file_size_mb=input_path.stat().st_size / (1024*1024), raw_metadata={})
                                         return ExtractionResult(text=text, metadata=meta, page_texts=[text], extraction_method="text_read", warnings=[], char_count=len(text), word_count=len(text.split()))
                                     except Exception as e: raise FileProcessingError(f"Failed to read input text file: {e}", str(input_path)) from e

                                 extract_result = self.stage_executor.execute("extract", read_text, data_payload, [])
                            else:
                                raise FileProcessingError(f"Unsupported file type for extraction: {input_path.suffix}", str(input_path))

                            data_payload['text'] = extract_result.text
                            data_payload['metadata'] = extract_result.metadata
                            data_payload['stats_extraction'] = {"chars": extract_result.char_count, "pages": extract_result.metadata.num_pages if extract_result.metadata else 1}
                            self._save_checkpoint(input_path, "extract", data_payload)
                            self._complete_stage(progress_manager, "extract")
                    except ValueError:
                        # resume_from_stage not in stage_order, run normally
                        extract_result: Optional[ExtractionResult]
                        if input_path.suffix.lower() == '.pdf':
                            extract_result = self.stage_executor.execute("extract",
                                                                        lambda: self.pdf_processor.extract_text(input_path),
                                                                        data_payload, [])
                            if not extract_result: raise PipelineError("PDF extraction failed.")
                        elif input_path.suffix.lower() in ['.txt', '.md']:
                             def read_text():
                                 try:
                                     text = input_path.read_text(encoding='utf-8')
                                     meta = PDFMetadata(file_path=input_path, num_pages=1, file_size_mb=input_path.stat().st_size / (1024*1024), raw_metadata={})
                                     return ExtractionResult(text=text, metadata=meta, page_texts=[text], extraction_method="text_read", warnings=[], char_count=len(text), word_count=len(text.split()))
                                 except Exception as e: raise FileProcessingError(f"Failed to read input text file: {e}", str(input_path)) from e

                             extract_result = self.stage_executor.execute("extract", read_text, data_payload, [])
                        else:
                            raise FileProcessingError(f"Unsupported file type for extraction: {input_path.suffix}", str(input_path))

                        data_payload['text'] = extract_result.text
                        data_payload['metadata'] = extract_result.metadata
                        data_payload['stats_extraction'] = {"chars": extract_result.char_count, "pages": extract_result.metadata.num_pages if extract_result.metadata else 1}
                        self._save_checkpoint(input_path, "extract", data_payload)
                        self._complete_stage(progress_manager, "extract")
                else:
                    # No resume, run normally
                    extract_result: Optional[ExtractionResult]
                    if input_path.suffix.lower() == '.pdf':
                        extract_result = self.stage_executor.execute("extract",
                                                                    lambda: self.pdf_processor.extract_text(input_path),
                                                                    data_payload, [])
                        if not extract_result: raise PipelineError("PDF extraction failed.")
                    elif input_path.suffix.lower() in ['.txt', '.md']:
                         def read_text():
                             try:
                                 text = input_path.read_text(encoding='utf-8')
                                 meta = PDFMetadata(file_path=input_path, num_pages=1, file_size_mb=input_path.stat().st_size / (1024*1024), raw_metadata={})
                                 return ExtractionResult(text=text, metadata=meta, page_texts=[text], extraction_method="text_read", warnings=[], char_count=len(text), word_count=len(text.split()))
                             except Exception as e: raise FileProcessingError(f"Failed to read input text file: {e}", str(input_path)) from e

                         extract_result = self.stage_executor.execute("extract", read_text, data_payload, [])
                    else:
                        raise FileProcessingError(f"Unsupported file type for extraction: {input_path.suffix}", str(input_path))

                    data_payload['text'] = extract_result.text
                    data_payload['metadata'] = extract_result.metadata
                    data_payload['stats_extraction'] = {"chars": extract_result.char_count, "pages": extract_result.metadata.num_pages if extract_result.metadata else 1}
                    self._save_checkpoint(input_path, "extract", data_payload)
                    self._complete_stage(progress_manager, "extract")

            # --- Stage 2: Preprocess ---
            if "preprocess" in self.stages_to_run and "preprocess" not in skip_stages:
                current_stage_num += 1
                dual_tracker.set_overall_progress(current_stage_num, len(self.stages_to_run))

                # Check if preprocess data already exists from checkpoint (resume case)
                if 'text' in data_payload and resume_from_stage:
                    try:
                        resume_idx = stage_order.index(resume_from_stage)
                        preprocess_idx = stage_order.index("preprocess")
                        # If we're resuming from a later stage, preprocess is already done
                        if preprocess_idx < resume_idx:
                            self.logger.info("Preprocess stage already completed from checkpoint, using existing data")
                        else:
                            # Run preprocess normally
                            def run_preprocess():
                                text = data_payload.get('text')
                                if text is None:
                                    raise MissingDataError("preprocess", "text")
                                if self.config.clean_for_audio:
                                    text = self.text_preprocessor.clean_for_audio(text)
                                return self.text_preprocessor.preprocess_for_llm(text)

                            data_payload['text'] = self.stage_executor.execute("preprocess", run_preprocess, data_payload, ['text'])
                            data_payload['stats_preprocess'] = {"chars": len(data_payload['text'])}
                            self._save_checkpoint(input_path, "preprocess", data_payload)
                            self._complete_stage(progress_manager, "preprocess")
                    except ValueError:
                        # resume_from_stage not in stage_order, run normally
                        def run_preprocess():
                            text = data_payload.get('text')
                            if text is None:
                                raise MissingDataError("preprocess", "text")
                            if self.config.clean_for_audio:
                                text = self.text_preprocessor.clean_for_audio(text)
                            return self.text_preprocessor.preprocess_for_llm(text)

                        data_payload['text'] = self.stage_executor.execute("preprocess", run_preprocess, data_payload, ['text'])
                        data_payload['stats_preprocess'] = {"chars": len(data_payload['text'])}
                        self._save_checkpoint(input_path, "preprocess", data_payload)
                        self._complete_stage(progress_manager, "preprocess")
                else:
                    # No resume, run normally
                    def run_preprocess():
                        text = data_payload.get('text')
                        if text is None:
                            raise MissingDataError("preprocess", "text")
                        if self.config.clean_for_audio:
                            text = self.text_preprocessor.clean_for_audio(text)
                        return self.text_preprocessor.preprocess_for_llm(text)

                    data_payload['text'] = self.stage_executor.execute("preprocess", run_preprocess, data_payload, ['text'])
                    data_payload['stats_preprocess'] = {"chars": len(data_payload['text'])}
                    self.logger.info("About to save preprocess checkpoint...")
                    self._save_checkpoint(input_path, "preprocess", data_payload)
                    self.logger.info("Preprocess checkpoint saved, calling _complete_stage...")
                    self._complete_stage(progress_manager, "preprocess")
                    self.logger.info("_complete_stage returned for preprocess")

            self.logger.debug("=== CHECKPOINT 1: After preprocess stage block ===")
            self.logger.info(f"Current stages_to_run: {self.stages_to_run}")
            self.logger.info(f"Current skip_stages: {skip_stages}")
            self.logger.info(f"Checking chunk stage: 'chunk' in stages_to_run = {'chunk' in self.stages_to_run}")
            self.logger.info(f"Checking chunk stage: 'chunk' not in skip_stages = {'chunk' not in skip_stages}")

            # --- Stage 3: Chunk ---
            self.logger.debug("=== CHECKPOINT 2: About to check chunk stage condition ===")
            if "chunk" in self.stages_to_run and "chunk" not in skip_stages:
                self.logger.debug("=== CHECKPOINT 3: Entered chunk stage block ===")
                self.logger.debug("=== CHECKPOINT 4: Incrementing stage counter ===")
                current_stage_num += 1
                self.logger.info(f"Current stage num: {current_stage_num}/{len(self.stages_to_run)}")

                self.logger.debug("=== CHECKPOINT 5: Calling dual_tracker.set_overall_progress ===")
                dual_tracker.set_overall_progress(current_stage_num, len(self.stages_to_run))
                self.logger.debug("=== CHECKPOINT 6: dual_tracker.set_overall_progress returned ===")

                # Check if chunks already exist from checkpoint (resume case)
                self.logger.debug("=== CHECKPOINT 7: Checking resume condition ===")
                self.logger.info(f"'chunks' in data_payload: {'chunks' in data_payload}")
                self.logger.info(f"resume_from_stage: {resume_from_stage}")
                if 'chunks' in data_payload and resume_from_stage:
                    self.logger.debug("=== CHECKPOINT 8: Entered resume condition block ===")
                    try:
                        self.logger.debug("=== CHECKPOINT 9: Inside resume try block ===")
                        resume_idx = stage_order.index(resume_from_stage)
                        chunk_idx = stage_order.index("chunk")
                        self.logger.info(f"resume_idx={resume_idx}, chunk_idx={chunk_idx}")
                        if chunk_idx < resume_idx:
                            self.logger.debug("=== CHECKPOINT 10: Chunk already completed, skipping ===")
                            self.logger.info("Chunk stage already completed from checkpoint, using existing data")
                        else:
                            self.logger.debug("=== CHECKPOINT 11: Running chunk stage ===")
                            def run_chunk():
                                self.logger.debug("=== CHECKPOINT 11a: Inside run_chunk function ===")
                                if 'text' not in data_payload: raise MissingDataError("chunk", "text")
                                self.logger.debug(f"=== CHECKPOINT 11b: Calling text_chunker.chunk_text (text length: {len(data_payload['text'])}) ===")
                                result = self.text_chunker.chunk_text(data_payload['text'])
                                self.logger.debug(f"=== CHECKPOINT 11c: chunk_text returned {len(result.chunks) if result else 'None'} chunks ===")
                                return result

                            self.logger.debug("=== CHECKPOINT 12: Calling stage_executor.execute ===")
                            chunk_result = self.stage_executor.execute("chunk", run_chunk, data_payload, ['text'])
                            self.logger.debug(f"=== CHECKPOINT 13: stage_executor.execute returned ===")
                            data_payload['chunks'] = [c.text for c in chunk_result.chunks]
                            data_payload['stats_chunk'] = {"count": chunk_result.total_chunks, "avg_size": chunk_result.average_chunk_size}
                            self.logger.debug("=== CHECKPOINT 14: Saving chunk checkpoint ===")
                            self._save_checkpoint(input_path, "chunk", data_payload)
                            self.logger.debug("=== CHECKPOINT 15: Completing chunk stage ===")
                            self._complete_stage(progress_manager, "chunk")
                            self.logger.debug("=== CHECKPOINT 16: Chunk stage completed ===")
                    except ValueError as ve:
                        self.logger.debug(f"=== CHECKPOINT 17: ValueError in resume block: {ve} ===")
                        def run_chunk():
                            self.logger.debug("=== CHECKPOINT 17a: Inside run_chunk (ValueError path) ===")
                            if 'text' not in data_payload: raise MissingDataError("chunk", "text")
                            self.logger.debug(f"=== CHECKPOINT 17b: Calling text_chunker.chunk_text ===")
                            result = self.text_chunker.chunk_text(data_payload['text'])
                            self.logger.debug(f"=== CHECKPOINT 17c: chunk_text returned ===")
                            return result

                        self.logger.debug("=== CHECKPOINT 18: Calling stage_executor.execute (ValueError path) ===")
                        chunk_result = self.stage_executor.execute("chunk", run_chunk, data_payload, ['text'])
                        self.logger.debug("=== CHECKPOINT 19: stage_executor.execute returned (ValueError path) ===")
                        data_payload['chunks'] = [c.text for c in chunk_result.chunks]
                        data_payload['stats_chunk'] = {"count": chunk_result.total_chunks, "avg_size": chunk_result.average_chunk_size}
                        self.logger.debug("=== CHECKPOINT 20: Saving chunk checkpoint (ValueError path) ===")
                        self._save_checkpoint(input_path, "chunk", data_payload)
                        self.logger.debug("=== CHECKPOINT 21: Completing chunk stage (ValueError path) ===")
                        self._complete_stage(progress_manager, "chunk")
                        self.logger.debug("=== CHECKPOINT 22: Chunk stage completed (ValueError path) ===")
                else:
                    self.logger.debug("=== CHECKPOINT 23: No resume, running chunk normally ===")
                    def run_chunk():
                        self.logger.debug("=== CHECKPOINT 23a: Inside run_chunk (normal path) ===")
                        if 'text' not in data_payload: raise MissingDataError("chunk", "text")
                        self.logger.debug(f"=== CHECKPOINT 23b: Calling text_chunker.chunk_text ===")
                        result = self.text_chunker.chunk_text(data_payload['text'])
                        self.logger.debug(f"=== CHECKPOINT 23c: chunk_text returned ===")
                        return result

                    self.logger.debug("=== CHECKPOINT 24: Calling stage_executor.execute (normal path) ===")
                    chunk_result = self.stage_executor.execute("chunk", run_chunk, data_payload, ['text'])
                    self.logger.debug("=== CHECKPOINT 25: stage_executor.execute returned (normal path) ===")
                    data_payload['chunks'] = [c.text for c in chunk_result.chunks]
                    data_payload['stats_chunk'] = {"count": chunk_result.total_chunks, "avg_size": chunk_result.average_chunk_size}
                    self.logger.debug("=== CHECKPOINT 26: Saving chunk checkpoint (normal path) ===")
                    self._save_checkpoint(input_path, "chunk", data_payload)
                    self.logger.debug("=== CHECKPOINT 27: Completing chunk stage (normal path) ===")
                    self._complete_stage(progress_manager, "chunk")
                    self.logger.debug("=== CHECKPOINT 28: Chunk stage completed (normal path) ===")

            self.logger.debug("=== CHECKPOINT 29: After chunk stage block ===")

            # --- Stage 4: Process ---
            self.logger.debug("=== CHECKPOINT 30: About to check process stage condition ===")
            self.logger.info(f"'process' in stages_to_run: {'process' in self.stages_to_run}")
            self.logger.info(f"'process' not in skip_stages: {'process' not in skip_stages}")
            if "process" in self.stages_to_run and "process" not in skip_stages:
                 self.logger.debug("=== CHECKPOINT 31: Entered process stage block ===")
                 # Handle model lifecycle before stage
                 self.logger.debug("=== CHECKPOINT 32: Checking lifecycle manager ===")
                 if self.lifecycle_manager:
                     self.logger.debug("=== CHECKPOINT 33: Calling lifecycle_manager.handle_stage_transition ===")
                     self.lifecycle_manager.handle_stage_transition("process")
                     self.logger.debug("=== CHECKPOINT 34: lifecycle_manager.handle_stage_transition returned ===")

                 # Update overall progress
                 self.logger.debug("=== CHECKPOINT 35: Incrementing process stage counter ===")
                 current_stage_num += 1
                 self.logger.debug("=== CHECKPOINT 36: Calling dual_tracker.set_overall_progress ===")
                 dual_tracker.set_overall_progress(current_stage_num, len(self.stages_to_run))
                 self.logger.debug("=== CHECKPOINT 37: dual_tracker.set_overall_progress returned ===")

                 # Check if chunking was skipped
                 if 'chunks' not in data_payload:
                      if 'text' in data_payload:
                           self.logger.info("Chunking skipped, treating whole document as one chunk.")
                           data_payload['chunks'] = [data_payload['text']]
                      else:
                           raise MissingDataError("process", "text or chunks")

                 # Determine resume point for this stage
                 resume_from_chunk_process = None
                 resume_process_results = None
                 if resume_from_stage == "process" and resume_chunk_index is not None:
                     resume_from_chunk_process = resume_chunk_index
                     self.logger.info(f"Resuming process stage from chunk {resume_from_chunk_process}")

                     # Reconstruct GenerationResult objects from checkpoint data
                     if 'processed_chunks' in data_payload:
                         processed_chunks = data_payload['processed_chunks']
                         resume_process_results = []
                         for chunk_text in processed_chunks[:resume_from_chunk_process]:
                             resume_process_results.append(GenerationResult(
                                 raw_output=chunk_text,
                                 filtered_output=chunk_text,
                                 input_tokens=0,  # Stats not preserved in checkpoint
                                 output_tokens=0,
                                 generation_time=0.0,
                                 memory_used=0,
                                 device_map={},
                                 error_message=None
                             ))
                         self.logger.info(f"Reconstructed {len(resume_process_results)} results from checkpoint")

                 def run_process():
                     if self.llm_backend is None: raise PipelineError("LLM Backend not set.")
                     if self.llm_backend.model_handle is None:
                         logger.info("Loading LLM backend model for processing...")
                         if not self.llm_backend.load(trust_remote_code=True): # Pass trust_remote_code
                             raise ModelLoadError("Failed to load LLM backend.", self.llm_backend.model_specifier)

                     from ..models.backends.batch import BatchProcessor # Local import
                     batch_processor = BatchProcessor(
                         backend=self.llm_backend,
                         batch_size=self.config.batch_size,
                         enable_dynamic_batching=self.config.enable_dynamic_batching,
                         max_batch_size=self.config.max_batch_size
                     )

                     # For podcast mode, use two-stage generation: planning then dialogue
                     use_two_stage_podcast = (self.config.mode == ProcessingMode.PODCAST)

                     if use_two_stage_podcast:
                         from ..config.settings import PODCAST_PLANNING_PROMPT, PODCAST_GENERATION_PROMPT
                         # Stage 1: Create outline for each chunk using BATCHING
                         self.logger.info(f"Podcast mode: Creating outlines for {len(data_payload['chunks'])} chunks (batch_size={self.config.batch_size})...")

                         # Process outlines in batches
                         outlines = []
                         for batch_start in range(0, len(data_payload['chunks']), self.config.batch_size):
                             batch_end = min(batch_start + self.config.batch_size, len(data_payload['chunks']))
                             batch_chunks = data_payload['chunks'][batch_start:batch_end]

                             # Check if backend supports batch generation
                             if self.config.batch_size > 1 and hasattr(self.llm_backend, '_generate_batch_request'):
                                 # Batch processing
                                 self.logger.info(f"Generating outlines for batch {batch_start}-{batch_end} (batch mode)")
                                 batch_results = self.llm_backend._generate_batch_request(
                                     prompts=batch_chunks,
                                     system_prompt=PODCAST_PLANNING_PROMPT,
                                     hyperparams=self.config.get_hyperparameters()
                                 )
                                 for result in batch_results:
                                     outline_text = result.filtered_output if result.filtered_output else result.raw_output
                                     outlines.append(outline_text)
                             else:
                                 # Sequential fallback
                                 self.logger.info(f"Generating outlines for chunks {batch_start}-{batch_end} (sequential mode)")
                                 for chunk_text in batch_chunks:
                                     outline_result = self.llm_backend.process_with_chat_template(
                                         system_prompt=PODCAST_PLANNING_PROMPT,
                                         user_message=chunk_text,
                                         hyperparams=self.config.get_hyperparameters()
                                     )
                                     outline_text = outline_result.filtered_output if outline_result.filtered_output else outline_result.raw_output
                                     outlines.append(outline_text)

                             self.logger.info(f"Created {len(outlines)} outlines so far...")

                         # Store outlines for use in generation
                         data_payload['podcast_outlines'] = outlines
                         system_prompt = PODCAST_GENERATION_PROMPT
                     else:
                         system_prompt = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT

                     # Set up stage progress tracking
                     total_chunks = len(data_payload['chunks'])
                     dual_tracker.set_stage_progress(0, total_chunks, "Processing chunks")

                     # Add rich progress bar for process stage
                     progress_manager.add_stage_progress("process", total_chunks)

                     # Define checkpoint callback (optimized to avoid full payload copy)
                     def save_process_checkpoint(chunk_idx, results, extra_data, **kwargs):
                         # Only copy essential data for checkpoint (avoid copying entire payload)
                         checkpoint_data = {
                             'input_path': data_payload['input_path'],
                             'chunks': data_payload['chunks'],
                             'process_checkpoint_index': chunk_idx,
                             'completed_stages': self.stages_completed.copy()  # Add completed stages for progress calculation
                         }

                         # Add optional data if present
                         if 'text' in data_payload:
                             checkpoint_data['text'] = data_payload['text']
                         if 'metadata' in data_payload:
                             checkpoint_data['metadata'] = data_payload['metadata']

                         # Convert results to raw outputs
                         processed_so_far = [r.raw_output if not r.error_message else data_payload['chunks'][i]
                                            for i, r in enumerate(results)]
                         checkpoint_data['processed_chunks'] = processed_so_far

                         # Calculate completion percentage for notification display
                         total_chunks = len(data_payload['chunks'])
                         completion_pct = None
                         try:
                             from ..config.settings import DEFAULT_STAGE_WEIGHTS
                             completion_pct = self.checkpoint_manager.calculate_pipeline_completion(
                                 current_stage="process",
                                 current_chunk=chunk_idx,
                                 total_chunks=total_chunks,
                                 stage_weights=DEFAULT_STAGE_WEIGHTS,
                                 completed_stages=self.stages_completed
                             )
                             self.logger.debug(f"Calculated completion_pct={completion_pct}% for process checkpoint")
                         except Exception as e:
                             self.logger.debug(f"Could not calculate completion percentage: {e}")
                             completion_pct = None

                         try:
                             success = self.checkpoint_manager.save(
                                 input_path=input_path,
                                 config=self.config,
                                 stage="process",
                                 data=checkpoint_data,
                                 chunk_index=chunk_idx,
                                 total_chunks=len(data_payload['chunks']),
                                 config_uuid=self.config_uuid
                             )

                             # Display checkpoint notification if save succeeded
                             if success and progress_manager and hasattr(progress_manager, 'display_checkpoint_info'):
                                 # Get checkpoint path to extract filename (use calculated completion_pct)
                                 checkpoint_path = self.checkpoint_manager._get_checkpoint_path(
                                     input_path, self.config, "process",
                                     chunk_index=chunk_idx,
                                     total_chunks=total_chunks,
                                     completion_pct=completion_pct  # Use the completion_pct we calculated above
                                 )

                                 # Get file size
                                 import os
                                 if checkpoint_path.exists():
                                     compressed_size = checkpoint_path.stat().st_size
                                     compressed_size_mb = compressed_size / (1024 * 1024)

                                     # Estimate uncompressed size (roughly 1.7x compressed)
                                     uncompressed_size_mb = compressed_size_mb * 1.7 if self.checkpoint_manager.enable_compression else compressed_size_mb
                                     compression_ratio = compressed_size_mb / uncompressed_size_mb if uncompressed_size_mb > 0 else 1.0

                                     # Get timestamp
                                     from datetime import datetime
                                     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                     # Display notification
                                     progress_manager.display_checkpoint_info({
                                         'filename': checkpoint_path.name,
                                         'chunk_index': chunk_idx,
                                         'total_chunks': len(data_payload['chunks']),
                                         'uncompressed_size_mb': uncompressed_size_mb,
                                         'compressed_size_mb': compressed_size_mb,
                                         'compression_ratio': compression_ratio,
                                         'hash': 'N/A',  # Hash calculation not currently implemented
                                         'timestamp': timestamp
                                     })
                         except TypeError as e:
                             # Handle case where save method signature doesn't match expected
                             self.logger.warning(f"Failed to save checkpoint: {e}")

                     # Prepare texts for processing
                     # For podcast mode with outlines, we need to combine outline + content
                     if use_two_stage_podcast:
                         # Format each chunk with its outline for the generation stage
                         # NOTE: Do NOT use labels like "OUTLINE:" or "CONTENT:" as they leak into output
                         # Instead, structure implicitly: outline first, then content separated by newlines
                         texts_to_process = []
                         for i, chunk_text in enumerate(data_payload['chunks']):
                             outline = outlines[i] if i < len(outlines) else ""
                             # Format WITHOUT labels - just structure with separation
                             formatted_text = f"{outline}\n\n---\n\n{chunk_text}"
                             texts_to_process.append(formatted_text)
                     else:
                         texts_to_process = data_payload['chunks']

                     # Run batch (sequentially) with checkpointing
                     # For podcast mode, ALWAYS remove thinking tags (models like DeepSeek-R1 generate them automatically)
                     # For other modes, let the filter stage handle it based on config
                     should_remove_thinking = use_two_stage_podcast or self.config.remove_thinking

                     results = batch_processor.process_batch(
                         texts=texts_to_process,
                         system_prompt=system_prompt,
                         hyperparams=self.config.get_hyperparameters(),
                         remove_thinking=should_remove_thinking,
                         checkpoint_callback=save_process_checkpoint if self.config.enable_checkpoints else None,
                         checkpoint_interval=self.config.checkpoint_interval,
                         resume_from_chunk=resume_from_chunk_process,
                         resume_results=resume_process_results,
                         dual_tracker=dual_tracker,
                         progress_manager=progress_manager  # Pass progress manager for rich progress bars
                     )

                     # Process results, handle errors, and fallback
                     processed_chunks = []
                     errors = 0
                     for i, res in enumerate(results):
                         if res.error_message or "Error:" in res.filtered_output:
                             errors += 1
                             logger.warning(f"Chunk {i+1} processing failed: {res.error_message or res.filtered_output}")
                             if self.config.fallback_on_error:
                                 processed_chunks.append(data_payload['chunks'][i]) # Fallback
                             else:
                                 raise GenerationError(f"Chunk {i+1} failed: {res.error_message}", self.llm_backend.model_specifier)
                         else:
                             processed_chunks.append(res.raw_output)

                     if errors > 0: ConsoleOutput.warning(f"{errors} chunks failed processing. Used fallback.")
                     return processed_chunks, results # Return both raw chunks and GenerationResult list

                 # Store results
                 processed_chunks, gen_results = self.stage_executor.execute("process", run_process, data_payload, ['chunks'])
                 data_payload['processed_chunks'] = processed_chunks
                 data_payload['stats_process'] = {
                     "input_tokens": sum(r.input_tokens for r in gen_results),
                     "output_tokens": sum(r.output_tokens for r in gen_results),
                     "total_time_sec": sum(r.generation_time for r in gen_results)
                 }

                 # Save final stage checkpoint (not mid-stage)
                 self._save_checkpoint(input_path, "process", data_payload)

                 # Mark process stage as complete
                 self._complete_stage(progress_manager, "process")

            # --- Stage 5: Filter ---
            if "filter" in self.stages_to_run and "filter" not in skip_stages:
                # Handle model lifecycle before stage
                if self.lifecycle_manager:
                    self.lifecycle_manager.handle_stage_transition("filter")

                current_stage_num += 1
                dual_tracker.set_overall_progress(current_stage_num, len(self.stages_to_run))

                # Determine input for filtering - check all possible sources
                self.logger.debug(f"Filter stage - data_payload keys: {list(data_payload.keys())}")
                chunks_to_filter = data_payload.get('processed_chunks')
                self.logger.debug(f"processed_chunks: {chunks_to_filter is not None and len(chunks_to_filter) if chunks_to_filter else 'None'}")

                # If no processed_chunks, try chunks
                if not chunks_to_filter:
                    chunks_to_filter = data_payload.get('chunks')
                    self.logger.debug(f"chunks: {chunks_to_filter is not None and len(chunks_to_filter) if chunks_to_filter else 'None'}")

                # If no chunks, try text as single chunk
                if not chunks_to_filter and 'text' in data_payload:
                    chunks_to_filter = [data_payload['text']]
                    self.logger.debug("Using text as single chunk")

                # If still no data, check if we already have filtered_text (from checkpoint)
                if not chunks_to_filter and 'filtered_text' in data_payload:
                    # Already filtered from checkpoint, skip this stage
                    self.logger.info("Filtered text already exists from checkpoint, skipping filter stage")
                    self.stages_completed.append("filter")
                else:
                    # Run the filter
                    def run_filter():
                        if not chunks_to_filter:
                            raise MissingDataError("filter", "processed_chunks, chunks, or text")
                        # Use the merged filter logic
                        return self.response_filter.filter_and_merge(
                            chunks_to_filter,
                            remove_thinking=self.config.remove_thinking,
                            remove_acknowledgments=True # Always remove acks
                        )

                    data_payload['filtered_text'] = self.stage_executor.execute("filter", run_filter, data_payload, [])
                    data_payload['stats_filter'] = {"chars": len(data_payload['filtered_text'])}

                    # Save checkpoint
                    self._save_checkpoint(input_path, "filter", data_payload)
                    self._complete_stage(progress_manager, "filter")

            # --- Stage 6: Format ---
            if "format" in self.stages_to_run and "format" not in skip_stages:
                current_stage_num += 1
                dual_tracker.set_overall_progress(current_stage_num, len(self.stages_to_run))
                def run_format():
                    # Find the best text to format
                    text_to_format = (
                        data_payload.get('filtered_text') or
                        (data_payload.get('processed_chunks') and "\n\n".join(data_payload['processed_chunks'])) or
                        data_payload.get('text')
                    )
                    if text_to_format is None: raise MissingDataError("format", "text")
                    
                    return self.formatter.format(
                        text_to_format,
                        add_emotions=self.config.add_emotions,
                        add_structure=True
                    )
                
                data_payload['formatted_text'] = self.stage_executor.execute("format", run_format, data_payload, [])
                data_payload['stats_format'] = {"chars": len(data_payload['formatted_text'])}

                # Save checkpoint
                self._save_checkpoint(input_path, "format", data_payload)
                self._complete_stage(progress_manager, "format")

            # --- Stage 7: Save ---
            if "save" in self.stages_to_run and "save" not in skip_stages:
                # Handle model lifecycle before stage
                if self.lifecycle_manager:
                    self.lifecycle_manager.handle_stage_transition("save")

                current_stage_num += 1
                dual_tracker.set_overall_progress(current_stage_num, len(self.stages_to_run))
                def run_save():
                    text_to_save = data_payload.get('formatted_text', data_payload.get('filtered_text'))
                    # Fallback logic
                    if text_to_save is None:
                         chunks_source = data_payload.get('processed_chunks', data_payload.get('chunks'))
                         if chunks_source: text_to_save = "\n\n".join(chunks_source)
                         else: text_to_save = data_payload.get('text')
                         
                    if text_to_save is None: raise MissingDataError("save", "any text content")
                    
                    # Determine path
                    path = self.file_handler.get_output_path(
                        input_path=input_path,
                        base_path_override=output_path_base,
                        suffix=f"_{self.config.mode.value}",
                        extension=f".{self.config.output_format}"
                    )
                    
                    # Build metadata
                    meta = {
                        "source_file": str(input_path.resolve()),
                        "processing_mode": self.config.mode.value,
                        "model_used": self.config.model_name_for_metadata,
                        "stages_completed": self.stages_completed + ["save"]
                    }
                    if data_payload.get('metadata'):
                         meta["original_pages"] = data_payload['metadata'].num_pages
                         
                    return self.file_handler.save_text(
                        text_to_save, path, self.config.output_format, meta
                    )
                
                final_output_path = self.stage_executor.execute("save", run_save, data_payload, [])
                if not final_output_path:
                     raise PipelineError("Failed to save output file.", "save")

                # Save checkpoint
                self._save_checkpoint(input_path, "save", data_payload)
                self._complete_stage(progress_manager, "save")

            # --- Stage 8: Audio (Custom, not in default list) ---
            if ("audio" in self.stages_to_run or self.config.generate_audio) and "audio" not in skip_stages: # Check both
                if self.audio_backend is None:
                     ConsoleOutput.warning("Audio generation requested but no audio backend is set. Skipping.")
                else:
                    # Handle model lifecycle before stage
                    if self.lifecycle_manager:
                        self.lifecycle_manager.handle_stage_transition("audio")

                    # Update overall progress
                    current_stage_num += 1
                    dual_tracker.set_overall_progress(current_stage_num, len(self.stages_to_run))

                    self.current_stage = "audio"
                    ConsoleOutput.section(f"Stage: Audio Generation")

                    # Get text for audio from pipeline stages
                    text_for_audio = data_payload.get('formatted_text', data_payload.get('filtered_text'))
                    if text_for_audio is None: # Fallback (same as save stage)
                         chunks_source = data_payload.get('processed_chunks', data_payload.get('chunks'))
                         if chunks_source: text_for_audio = "\n\n".join(chunks_source)
                         else: text_for_audio = data_payload.get('text')

                    # If no text from pipeline stages, try reading from input file directly
                    # This allows audio-only processing from existing markdown/text files
                    if text_for_audio is None:
                         if input_path.suffix.lower() in ['.txt', '.md']:
                              try:
                                   self.logger.info(f"No text from pipeline stages, reading directly from input file: {input_path}")
                                   text_for_audio = input_path.read_text(encoding='utf-8')
                                   ConsoleOutput.info(f"Read {len(text_for_audio)} characters from {input_path.name}")
                              except Exception as e:
                                   raise FileProcessingError(f"Failed to read input file for audio generation: {e}", str(input_path)) from e
                         else:
                              raise MissingDataError("audio", "any text content (no pipeline text and input is not .txt/.md)")

                    # Clean text for audio (even if preprocess stage was skipped)
                    if self.config.clean_for_audio:
                         text_for_audio = self.text_preprocessor.clean_for_audio(text_for_audio)

                    # Determine audio output path
                    audio_output_path = self.file_handler.get_output_path(
                         input_path=input_path,
                         base_path_override=output_path_base, # Use same base name
                         suffix=f"_{self.config.mode.value}",
                         extension=f".{self.config.audio_config.output_format}"
                    )

                    # Load audio backend if needed
                    if self.audio_backend.model_handle is None:
                         if not self.audio_backend.load():
                              raise ModelLoadError("Failed to load audio backend.", self.audio_backend.model_specifier)

                    # Determine resume point for audio stage
                    resume_from_chunk_audio = None
                    if resume_from_stage == "audio" and resume_chunk_index is not None:
                        resume_from_chunk_audio = resume_chunk_index
                        self.logger.info(f"Resuming audio stage from chunk {resume_from_chunk_audio}")

                    # Define checkpoint callback for audio (optimized to avoid full payload copy)
                    def save_audio_checkpoint(chunk_idx, audio_arrays, extra_data, **kwargs):
                        self.logger.debug(f"Audio checkpoint callback called: chunk_idx={chunk_idx}, audio_arrays len={len(audio_arrays) if audio_arrays else 0}, extra_data keys={list(extra_data.keys()) if extra_data else []}")
                        # Only copy essential data for checkpoint
                        checkpoint_data = {
                            'input_path': data_payload['input_path'],
                            'audio_arrays': audio_arrays,
                            'audio_checkpoint_index': chunk_idx,
                            'completed_stages': self.stages_completed.copy()  # Add completed stages for progress calculation
                        }

                        # Add optional text data if present
                        if 'formatted_text' in data_payload:
                            checkpoint_data['formatted_text'] = data_payload['formatted_text']
                        elif 'filtered_text' in data_payload:
                            checkpoint_data['filtered_text'] = data_payload['filtered_text']

                        # Merge extra data
                        checkpoint_data.update(extra_data)

                        # Get total chunks from extra_data if available
                        total_audio_chunks = extra_data.get('total_segments', None)

                        # Calculate completion percentage for notification display
                        completion_pct = None
                        if total_audio_chunks is not None:
                            try:
                                from ..config.settings import DEFAULT_STAGE_WEIGHTS
                                completion_pct = self.checkpoint_manager.calculate_pipeline_completion(
                                    current_stage="audio",
                                    current_chunk=chunk_idx,
                                    total_chunks=total_audio_chunks,
                                    stage_weights=DEFAULT_STAGE_WEIGHTS,
                                    completed_stages=self.stages_completed
                                )
                                self.logger.debug(f"Calculated completion_pct={completion_pct}% for audio checkpoint")
                            except Exception as e:
                                self.logger.debug(f"Could not calculate completion percentage: {e}")
                                completion_pct = None

                        try:
                            success = self.checkpoint_manager.save(
                                input_path=input_path,
                                config=self.config,
                                stage="audio",
                                data=checkpoint_data,
                                chunk_index=chunk_idx,
                                total_chunks=total_audio_chunks,
                                config_uuid=self.config_uuid
                            )

                            # Display checkpoint notification if save succeeded
                            if success and progress_manager and hasattr(progress_manager, 'display_checkpoint_info'):
                                # Get checkpoint path to extract filename (use calculated completion_pct)
                                checkpoint_path = self.checkpoint_manager._get_checkpoint_path(
                                    input_path, self.config, "audio",
                                    chunk_index=chunk_idx,
                                    total_chunks=total_audio_chunks,
                                    completion_pct=completion_pct  # Use the completion_pct we calculated above
                                )

                                # Get file size
                                import os
                                if checkpoint_path.exists():
                                    compressed_size = checkpoint_path.stat().st_size
                                    compressed_size_mb = compressed_size / (1024 * 1024)

                                    # Estimate uncompressed size (roughly 1.7x compressed)
                                    uncompressed_size_mb = compressed_size_mb * 1.7 if self.checkpoint_manager.enable_compression else compressed_size_mb
                                    compression_ratio = compressed_size_mb / uncompressed_size_mb if uncompressed_size_mb > 0 else 1.0

                                    # Get timestamp
                                    from datetime import datetime
                                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                    # Display notification
                                    progress_manager.display_checkpoint_info({
                                        'filename': checkpoint_path.name,
                                        'chunk_index': chunk_idx,
                                        'total_chunks': total_audio_chunks,
                                        'uncompressed_size_mb': uncompressed_size_mb,
                                        'compressed_size_mb': compressed_size_mb,
                                        'compression_ratio': compression_ratio,
                                        'hash': 'N/A',  # Hash calculation not currently implemented
                                        'timestamp': timestamp
                                    })

                            # Update pipeline progress during audio generation
                            if total_audio_chunks is not None and total_audio_chunks > 0:
                                stage_progress = chunk_idx / total_audio_chunks  # Progress within audio stage (0.0 to 1.0)
                                self._update_pipeline_progress(
                                    progress_manager,
                                    self.stages_completed,
                                    current_stage="audio",
                                    stage_progress=stage_progress
                                )
                                self.logger.debug(f"Updated pipeline progress: audio stage {chunk_idx}/{total_audio_chunks} ({stage_progress*100:.1f}%)")
                        except TypeError as e:
                            # Handle case where save method signature doesn't match expected
                            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

                    # Generate with checkpointing support
                    audio_result = self.audio_backend.generate_audio(
                        text=text_for_audio,
                        output_path=audio_output_path,
                        checkpoint_callback=save_audio_checkpoint if self.config.enable_checkpoints else None,
                        checkpoint_interval=self.config.checkpoint_interval,
                        resume_from_chunk=resume_from_chunk_audio,
                        dual_tracker=dual_tracker
                    )

                    if audio_result:
                         final_audio_path = audio_result.audio_path
                         data_payload['audio_result'] = audio_result
                         ConsoleOutput.success(f"Audio saved: {final_audio_path.name}")
                    else:
                         ConsoleOutput.error("Audio generation failed.")
                         # Don't fail the whole pipeline, just log it
                         data_payload['audio_result'] = None

                    # Save final stage checkpoint (not mid-stage)
                    self._save_checkpoint(input_path, "audio", data_payload)
                    self._complete_stage(progress_manager, "audio")
                    self.memory_monitor.check("after audio generation")

            success = True
            
        except (PipelineError, ModelLoadError, FileProcessingError, MissingDataError) as e:
            error_message = str(e) # These errors are already formatted
            self.logger.error(error_message, exc_info=True)
            ConsoleOutput.error(error_message)
        except Exception as e:
            error_message = f"Pipeline failed at stage '{self.current_stage or 'unknown'}': {e}"
            self.logger.error(error_message, exc_info=True)
            ConsoleOutput.error(error_message)
        finally:
            # Stop progress manager
            progress_manager.stop()

            self.memory_monitor.check(f"pipeline end for {input_path.name}")

            # Cleanup lifecycle manager (unload any remaining models)
            if self.lifecycle_manager:
                try:
                    self.lifecycle_manager.cleanup()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup lifecycle manager: {e}")

            # Unloading is handled by the caller (run_cli_processing or MenuSystem)

            # Cleanup old checkpoints if pipeline succeeded
            if success and self.checkpoint_manager and self.config.enable_checkpoints:
                try:
                    self.checkpoint_manager.cleanup_old_checkpoints(
                        input_path,
                        keep_latest=self.config.checkpoint_cleanup_keep
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup old checkpoints: {e}")

        processing_time = time.time() - start_time
        statistics = self._gather_statistics(data_payload)
        
        # Add timings if llm_backend was used
        if self.llm_backend and hasattr(self.llm_backend, 'last_generation_time'): # Assuming backend tracks this
             statistics['llm_total_gen_time'] = self.llm_backend.last_generation_time # This needs to be implemented in backend

        result = PipelineResult(
            success=success,
            input_file=input_path,
            output_file=final_output_path,
            audio_file=final_audio_path,
            processing_time=processing_time,
            stages_completed=self.stages_completed.copy(),
            error_message=error_message if not success else None,
            statistics=statistics
        )
        self.file_handler.record_processed_file(result)
        return result

    def _gather_statistics(self, data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather all statistics from data_payload into a single dictionary.

        Args:
            data_payload: The data payload dictionary containing stats_* keys

        Returns:
            A dictionary containing all collected statistics
        """
        statistics = {}

        # Collect all stats_* keys from data_payload
        for key, value in data_payload.items():
            if key.startswith('stats_'):
                # Remove 'stats_' prefix and add to statistics
                stat_name = key.replace('stats_', '')
                statistics[stat_name] = value

        return statistics

    def process_batch(self,
                     input_files: List[Path],
                     output_dir: Path,
                     parallel: bool = False) -> List[PipelineResult]:
        """Processes a list of files sequentially."""
        if parallel:
             ConsoleOutput.warning("Parallel processing is not yet implemented. Running sequentially.")

        ConsoleOutput.header(f"Batch Processing: {len(input_files)} files")
        results = []

        # Set the base output directory for this batch
        self.file_handler.output_dir = output_dir.resolve()
        self.file_handler.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Batch output directory set to: {self.file_handler.output_dir}")

        for i, input_file in enumerate(input_files, 1):
            ConsoleOutput.subsection(f"--- Processing File {i}/{len(input_files)}: {input_file.name} ---")
            
            # output_path_base is None, so process_file will use file_handler
            # to generate a name based on the input_file in the output_dir.
            result = self.process_file(input_file, output_path_base=None) 
            results.append(result)

            if not result.success:
                 ConsoleOutput.error(f"Failed to process {input_file.name}: {result.error_message}")
            
            # Unload local models between files
            if self.llm_backend and self.llm_backend.provider_identifier.startswith("local"):
                ConsoleOutput.info(f"Unloading local text model ({self.llm_backend.model_specifier})...")
                self.llm_backend.unload()
            if self.audio_backend and self.audio_backend.provider_identifier.startswith("local"):
                 ConsoleOutput.info(f"Unloading local audio model ({self.audio_backend.model_specifier})...")
                 self.audio_backend.unload()
                 
            self.memory_monitor.check(f"after file {i} (models unloaded)")

        # Generate summary
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.processing_time for r in results)
        ConsoleOutput.header("Batch Processing Complete")
        ConsoleOutput.info(f"Successfully processed: {successful} / {len(results)}")
        ConsoleOutput.info(f"Total time: {total_time:.2f}s")

        return results

    def cleanup(self):
        """Saves report and cleans cache, without touching models."""
        self.logger.info("Running pipeline cleanup (saving report).")
        if hasattr(self, 'file_handler') and self.file_handler:
             if self.file_handler.processed_files:
                 try:
                     report_path = self.file_handler.save_processing_report()
                     if report_path:
                          ConsoleOutput.info(f"Processing report saved: {report_path.name}")
                 except Exception as e:
                     self.logger.error(f"Failed to save processing report: {e}")
             # Cache cleanup (optional, might be better done by manager)
             # self.file_handler.cleanup_old_files(days=7)
