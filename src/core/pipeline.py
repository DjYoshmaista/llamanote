# llamanote/core/pipeline.py
"""
Processing Pipeline Module
Orchestrates the multi-stage workflow from file input to final output.
Refactored from original processing_pipeline.py
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..utils.logger import get_logger_conf, LoggingProgress, MemoryMonitor, ConsoleOutput
from ..config.settings import (
    DEFAULT_PIPELINE_STAGES, PREPROCESS_PROMPT_PODCAST, DEFAULT_SYSTEM_PROMPT,
    INCLUDE_METADATA, TIMESTAMP_OUTPUTS, DEFAULT_OUTPUT_DIR, MAX_CHARS_PER_FILE,
    MAX_PDF_SIZE_MB, CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP, ENABLE_STAGE_CHECKPOINTS,
    MAX_RETRIES, FALLBACK_ON_ERROR
)
from ..config.manager import ConfigManager # For loading defaults if needed
from .types import (
    ProcessingMode, PipelineConfig, PipelineResult, AudioConfig,
    ExtractionResult, ChunkingResult, FilterResult, PDFMetadata,
    ChunkingStrategy
)
from ..processing.pdf_extractor import PDFProcessor
from ..processing.text_preprocessor import TextPreprocessor # Removed PDFTextCleaner
from ..processing.text_chunker import TextChunker
from ..processing.response_filter import ChunkedResponseFilter
from ..formatting.base_formatter import get_formatter, BaseFormatter
from ..io.file_handler import FileHandler
from ..io.checkpoints import CheckpointManager
from ..models.backends.base import LLMBackend, AudioBackend
from ..models.registry import get_model_entry, ModelEntry
from .errors import PipelineError, MissingDataError, PDFExtractionError, GenerationError, FileProcessingError

# Import the stage functions
from . import stages 

logger = get_logger_conf(__name__)

# --- Stage Execution Helper ---
# (This class is now refactored to use the imported stage functions)
class PipelineStageExecutor:
    """Helper class to manage stage execution, checkpointing, and logging."""
    def __init__(self, pipeline: 'ProcessingPipeline'):
        self.pipeline = pipeline # Reference to parent pipeline
        self.config = pipeline.config
        self.logger = pipeline.logger
        self.checkpoint_manager = pipeline.checkpoint_manager
    
    def execute(self, 
                stage_name: str, 
                stage_func: Any, 
                data_payload: Dict[str, Any], 
                required_keys: List[str]) -> Any:
        """
        Executes a pipeline stage with checkpointing and error handling.
        
        Args:
            stage_name: The name of the stage (e.g., "extract").
            stage_func: The function (lambda or method) to call for this stage.
            data_payload: The dictionary holding all pipeline data.
            required_keys: List of keys that must be in data_payload before running.

        Returns:
            The result of the stage_func.
        """
        if self.pipeline:
            self.pipeline.current_stage = stage_name
            
        ConsoleOutput.section(f"Stage: {stage_name.capitalize()}")

        # 1. Checkpoint Load
        if self.config.enable_checkpoints:
            if self.checkpoint_manager is None:
                raise PipelineError("CheckpointManager not initialized.", stage=stage_name)
            
            checkpoint_data = self.checkpoint_manager.load(stage_name)
            if checkpoint_data is not None:
                self.logger.info(f"Loaded {stage_name} checkpoint")
                ConsoleOutput.info(f"Loaded from checkpoint: {stage_name}")
                if stage_name not in self.pipeline.stages_completed:
                    self.pipeline.stages_completed.append(stage_name)
                
                # Add memory check after loading checkpoint
                if hasattr(self.pipeline, 'memory_monitor'):
                    self.pipeline.memory_monitor.check(f"after loading {stage_name} checkpoint")
                
                return checkpoint_data # Return loaded data

        # 2. Check dependencies
        for key in required_keys:
            if key not in data_payload:
                raise MissingDataError(stage=stage_name, missing_key=key)
        
        # 3. Execute Stage
        self.logger.info(f"Running stage: {stage_name}...")
        try:
            result = stage_func()
        except Exception as e:
             self.logger.error(f"Error during stage '{stage_name}': {e}", exc_info=True)
             raise PipelineError(f"Stage '{stage_name}' failed: {e}", stage=stage_name) from e
             
        # 4. Checkpoint Save
        if self.config.enable_checkpoints and result is not None:
            self.checkpoint_manager.save(stage_name, result)
            self.logger.debug(f"Saved checkpoint for stage {stage_name}")
        
        if stage_name not in self.pipeline.stages_completed:
            self.pipeline.stages_completed.append(stage_name)
            
        # 5. Add memory check after stage execution
        if hasattr(self.pipeline, 'memory_monitor'):
             self.pipeline.memory_monitor.check(f"after running {stage_name}")

        return result


# --- Main Pipeline Class ---

class ProcessingPipeline:
    """Main processing pipeline for document-to-text/audio conversion."""

    def __init__(self, 
                 config: Optional[PipelineConfig] = None,
                 llm_backend: Optional[LLMBackend] = None,
                 audio_backend: Optional[AudioBackend] = None
                 ):
        """
        Initializes the pipeline with a given configuration.
        Backends (LLMBackend, AudioBackend) must be injected after initialization.
        """
        self.config = config or PipelineConfig()
        self.logger = get_logger_conf(f"{__name__}.Pipeline")
        
        # Backends are injected
        self.llm_backend: Optional[LLMBackend] = llm_backend
        self.audio_backend: Optional[AudioBackend] = audio_backend

        # State tracking
        self.current_stage: Optional[str] = None
        self.stages_completed: List[str] = []
        self.checkpoint_name: Optional[str] = None
        self.stages_to_run: List[str] = self.config.stages or list(DEFAULT_PIPELINE_STAGES)
        
        # Add audio stage if requested in config but not in list
        if self.config.generate_audio and "audio" not in self.stages_to_run:
             self.stages_to_run.append("audio")

        # Initialize components based on config
        self._initialize_components()
        self.logger.info(f"Initialized pipeline in {self.config.mode.value} mode")
        self.logger.info(f"Stages to run: {self.stages_to_run}")

    def _initialize_components(self):
        """Initialize all pipeline components based on self.config."""
        self.file_handler = FileHandler(
            output_dir=self.config.output_dir, # Already a Path from Config
            timestamp_outputs=self.config.timestamp_outputs
        )
        self.pdf_processor = PDFProcessor(
            preserve_layout=self.config.preserve_pdf_layout,
            max_chars=MAX_CHARS_PER_FILE,
            max_size_mb=MAX_PDF_SIZE_MB
        )
        # self.text_cleaner = PDFTextCleaner() # <-- REMOVED
        self.text_preprocessor = TextPreprocessor() # <-- This class now has .clean_for_audio
        
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
                 self.logger.warning(f"Model {self.config.model_specifier} not in registry. Filter may not handle <think> tokens correctly.")
                 model_entry_for_filter = ModelEntry(
                     name=Path(self.config.model_specifier).name,
                     model_id=self.config.model_specifier,
                     author="unknown",
                     max_context=4096, # Use a safe default
                     optimal_chunk_size=1000 # Use a safe default
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
        self.stages_to_run = [s for s in stages if s in DEFAULT_PIPELINE_STAGES or s == "audio"] # Allow 'audio'
        # Always add 'audio' if config.generate_audio is True
        if self.config.generate_audio and "audio" not in self.stages_to_run:
             self.stages_to_run.append("audio")
        self.logger.info(f"Pipeline stages set to run: {self.stages_to_run}")

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
        self.checkpoint_manager = CheckpointManager(self.file_handler.cache_dir, self.checkpoint_name)
        self.stage_executor = PipelineStageExecutor(self) # Re-init executor
        self.stages_completed = []
        self.current_stage = "setup"

        ConsoleOutput.header(f"Processing: {input_path.name}")
        self.logger.info(f"Starting pipeline for: {input_path}")
        self.memory_monitor.start()

        data_payload: Dict[str, Any] = {'input_path': input_path}
        final_output_path: Optional[Path] = None
        final_audio_path: Optional[Path] = None
        success = False
        error_message: Optional[str] = None

        try:
            # --- Stage 1: Extract ---
            if "extract" in self.stages_to_run:
                extract_result = self.stage_executor.execute(
                    "extract", 
                    lambda: stages.run_extraction_stage(input_path, self.pdf_processor),
                    data_payload, 
                    []
                )
                if not extract_result: raise PipelineError("Extraction failed or returned None.", "extract")
                
                data_payload['text'] = extract_result.text
                data_payload['metadata'] = extract_result.metadata
                data_payload['stats_extraction'] = {"chars": extract_result.char_count, "pages": extract_result.metadata.num_pages if extract_result.metadata else 1}

            # --- Stage 2: Preprocess ---
            if "preprocess" in self.stages_to_run:
                def run_preprocess_lambda():
                    text = data_payload.get('text')
                    if text is None: 
                        raise MissingDataError("preprocess", "text")

                    # Preprocess for LLM (general cleaning)
                    preprocessed_text = self.text_preprocessor.preprocess_for_llm(text)

                    # Apply audio-specific cleaning if required by mode
                    if self.config.clean_for_audio:
                        self.logger.debug("Applying audio-specific cleaning...")
                        text_for_audio = self.text_preprocessor.clean_for_audio(preprocessed_text)

                    return text_for_audio

                data_payload['text'] = self.stage_executor.execute(
                    "preprocess", run_preprocess_lambda, data_payload, ['text']
                )
                data_payload['stats_preprocess'] = {"chars": len(data_payload['text'])}

            # --- Stage 3: Chunk ---
            if "chunk" in self.stages_to_run:
                def run_chunk_lambda():
                     if 'text' not in data_payload: raise MissingDataError("chunk", "text")
                     # Update chunker config from model entry if possible
                     model_entry = get_model_entry(self.config.model_specifier)
                     if model_entry:
                         self.text_chunker.target_size = model_entry.optimal_chunk_size
                         self.logger.info(f"Set chunk size to {model_entry.optimal_chunk_size} based on model registry.")
                     else:
                         self.text_chunker.target_size = self.config.chunk_size # Use config default
                     
                     return stages.run_chunking_stage(data_payload['text'], self.text_chunker)
                
                chunk_result = self.stage_executor.execute("chunk", run_chunk_lambda, data_payload, ['text'])
                data_payload['chunks'] = [c.text for c in chunk_result.chunks]
                data_payload['stats_chunk'] = {"count": chunk_result.total_chunks, "avg_size": chunk_result.average_chunk_size}
            
            # --- Stage 4: Process ---
            if "process" in self.stages_to_run:
                 # Check if chunking was skipped
                 if 'chunks' not in data_payload:
                      if 'text' in data_payload:
                           self.logger.info("Chunking skipped, treating whole document as one chunk.")
                           data_payload['chunks'] = [data_payload['text']]
                      else:
                           raise MissingDataError("process", "text or chunks")
                 
                 def run_process_lambda():
                     if self.llm_backend is None: raise PipelineError("LLM Backend not set.")
                     # Load model *inside* the executor (handles checkpoints)
                     return stages.run_processing_stage(
                         data_payload['chunks'],
                         self.llm_backend,
                         self.config
                     )

                 # Store results
                 processed_chunks = self.stage_executor.execute("process", run_process_lambda, data_payload, ['chunks'])
                 data_payload['processed_chunks'] = processed_chunks
                 # TODO: Stats like token usage are harder to get this way unless run_processing_stage returns them
                 # Let's modify run_processing_stage to return (processed_chunks, stats)
                 
                 # --- THIS IS A CHANGE FROM THE PROVIDED FILE ---
                 # This part requires modifying `run_processing_stage` to return
                 # a tuple: (List[str], Dict[str, Any])
                 # Assuming stages.py is modified as such:
                 # processed_chunks, gen_stats = self.stage_executor.execute(...)
                 # data_payload['processed_chunks'] = processed_chunks
                 # data_payload['stats_process'] = gen_stats
                 # --- For now, we'll assume it just returns the list ---
                 data_payload['stats_process'] = {"chunks_processed": len(processed_chunks)}


            # --- Stage 5: Filter ---
            if "filter" in self.stages_to_run:
                def run_filter_lambda():
                    # Find the best text to filter (prefer processed chunks)
                    chunks_to_filter = data_payload.get('processed_chunks')
                    if chunks_to_filter is None:
                         chunks_to_filter = data_payload.get('chunks') # Fallback to raw chunks
                    if not chunks_to_filter and 'text' in data_payload: 
                         chunks_to_filter = [data_payload['text']] # Fallback to whole text
                    
                    if not chunks_to_filter: raise MissingDataError("filter", "processed_chunks or chunks or text")
                    
                    return stages.run_filtering_stage(
                        chunks_to_filter,
                        self.response_filter,
                        self.config
                    )
                
                data_payload['filtered_text'] = self.stage_executor.execute("filter", run_filter_lambda, data_payload, [])
                data_payload['stats_filter'] = {"chars": len(data_payload['filtered_text'])}


            # --- Stage 6: Format ---
            if "format" in self.stages_to_run:
                def run_format_lambda():
                    # Find the best text to format
                    text_to_format = data_payload.get('filtered_text')
                    if text_to_format is None:
                         chunks_source = data_payload.get('processed_chunks', data_payload.get('chunks'))
                         if chunks_source: text_to_format = "\n\n".join(chunks_source)
                         else: text_to_format = data_payload.get('text')
                          
                    if text_to_format is None: raise MissingDataError("format", "text")
                    
                    return stages.run_formatting_stage(
                        text_to_format,
                        self.formatter,
                        self.config
                    )
                
                data_payload['formatted_text'] = self.stage_executor.execute("format", run_format_lambda, data_payload, [])
                data_payload['stats_format'] = {"chars": len(data_payload['formatted_text'])}

            # --- Stage 7: Save ---
            if "save" in self.stages_to_run:
                 def run_save_lambda():
                     text_to_save = data_payload.get('formatted_text', data_payload.get('filtered_text'))
                     # Fallback logic
                     if text_to_save is None:
                          chunks_source = data_payload.get('processed_chunks', data_payload.get('chunks'))
                          if chunks_source: text_to_save = "\n\n".join(chunks_source)
                          else: text_to_save = data_payload.get('text')
                          
                     if text_to_save is None: raise MissingDataError("save", "any text content")
                     
                     return stages.run_save_stage(
                         text_to_save,
                         input_path,
                         output_path_base,
                         data_payload.get('metadata'), # Pass PDF metadata
                         self.file_handler,
                         self.config
                     )
                 
                 final_output_path = self.stage_executor.execute("save", run_save_lambda, data_payload, [])
                 if not final_output_path:
                      raise PipelineError("Failed to save output file.", "save")
                 data_payload['output_file'] = final_output_path
            
            # --- Stage 8: Audio (Custom, not in default list) ---
            if "audio" in self.stages_to_run: # Check if explicitly requested
                if self.audio_backend is None:
                     ConsoleOutput.warning("Audio generation requested but no audio backend is set. Skipping.")
                elif 'output_file' not in data_payload:
                     # This check is new: we need the *text file* to exist first
                     ConsoleOutput.warning("Audio generation requires 'save' stage to be run first. Skipping audio.")
                else:
                    def run_audio_lambda():
                        text_file_path = data_payload.get('output_file')
                        if not text_file_path or not Path(text_file_path).exists():
                             raise MissingDataError("audio", "saved text file (run 'save' stage first)")
                        
                        return stages.run_audio_stage(
                            text_file_path=text_file_path,
                            audio_backend=self.audio_backend,
                            config=self.config,
                            text_preprocessor=self.text_preprocessor # Pass the preprocessor
                        )
                    
                    audio_result = self.stage_executor.execute("audio", run_audio_lambda, data_payload, ['output_file'])
                    
                    if audio_result:
                         final_audio_path = audio_result.audio_path
                         data_payload['audio_result'] = audio_result
                         data_payload['stats_audio'] = {
                             "duration_s": audio_result.duration_seconds,
                             "path": str(audio_result.audio_path)
                         }
                    else:
                         ConsoleOutput.error("Audio generation stage failed.")
                         # Don't fail the whole pipeline, just log it
                         data_payload['stats_audio'] = {"error": "Generation failed"}

            success = True
            
        except (PipelineError, ModelLoadError, FileProcessingError, MissingDataError) as e:
            error_message = str(e) # These errors are already formatted
            self.logger.error(error_message, exc_info=True)
            ConsoleOutput.error(error_message)
        except Exception as e:
            error_message = f"Pipeline failed at stage '{self.current_stage or 'unknown'}': {e}"
            self.logger.critical(error_message, exc_info=True, save_context=True) # Save context for unexpected errors
            ConsoleOutput.error(error_message)
        finally:
            self.memory_monitor.check(f"pipeline end for {input_path.name}")
            # Unloading is handled by the caller (run_cli_processing or MenuSystem)
            
        processing_time = time.time() - start_time
        statistics = self._gather_statistics(data_payload)
        
        # Add timings from executor
        if self.stage_executor:
             statistics['stage_timings_sec'] = self.stage_executor.stage_timings

        result = PipelineResult(
            success=success,
            input_file=input_path,
            output_file=data_payload.get('output_file'),
            audio_file=final_audio_path,
            processing_time=processing_time,
            stages_completed=self.stages_completed.copy(),
            error_message=error_message if not success else None,
            statistics=statistics
        )
        self.file_handler.record_processed_file(result)
        return result
        

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
        
        self.cleanup() # Save report at the end of the batch
        
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

    def _gather_statistics(self, data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Collects stats from the data_payload for the final report."""
        stats = {}
        if 'stats_extraction' in data_payload:
            stats.update(data_payload['stats_extraction'])
        if 'stats_preprocess' in data_payload:
            stats.update(data_payload['stats_preprocess'])
        if 'stats_chunk' in data_payload:
            stats.update(data_payload['stats_chunk'])
        if 'stats_process' in data_payload:
            stats.update(data_payload['stats_process'])
        if 'stats_filter' in data_payload:
            stats.update(data_payload['stats_filter'])
        if 'stats_format' in data_Mpayload:
            stats.update(data_payload['stats_format'])
        if 'stats_audio' in data_payload:
            stats['audio_generation'] = data_payload['stats_audio']
            
        stats['output_format'] = self.config.output_format
        return stats
