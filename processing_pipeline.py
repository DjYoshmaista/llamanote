"""
Processing Pipeline Module
Main pipeline that orchestrates the PDF processing workflow
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from loggerConf import get_logger_conf, LoggingProgress, MemoryMonitor, ConsoleOutput
from config import (
    PREPROCESS_PROMPT,
    MODELS,
    DEFAULT_MODEL,
    MEMORY_PROFILES,
    MARKDOWN_STYLES,
    PIPELINE_STAGES, # Import this
    ENABLE_STAGE_CHECKPOINTS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    FALLBACK_ON_ERROR
)
from pdf_processor import PDFProcessor, PDFTextCleaner
from text_processor import TextChunker, TextPreprocessor, ChunkingStrategy
# Updated imports for new backend structure
from llm_handler import (
    LLMBackend, GenerationResult, QuantizationConfig, 
    LayerSplitConfig, get_llm_backend
)
from response_filter import ChunkedResponseFilter
from markdown_formatter import MarkdownFormatter, PodcastFormatter, TechnicalFormatter
from file_handler import FileHandler
from hyperparameters import HyperparameterConfig
from model_registry import get_registry, get_model_config # Import registry functions

logger = get_logger_conf(__name__)

class ProcessingMode(Enum):
    """Processing modes for different use cases"""
    PODCAST = "podcast"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    SUMMARY = "summary"
    CUSTOM = "custom"

@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline"""
    mode: ProcessingMode = ProcessingMode.PODCAST
    model_name: str = DEFAULT_MODEL # This is now for metadata (e.g., "local:qwen3-4b")
    model_provider: str = "local" # 'local', 'local_gguf', 'openai', 'google', etc.
    model_specifier: str = DEFAULT_MODEL # The actual model ID or path
    memory_profile: str = "medium_vram" # Used to configure local HF models
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY
    chunk_size: int = 1000 # Use a default, can be overridden
    markdown_style: str = "podcast"
    system_prompt: Optional[str] = None
    remove_thinking: bool = True
    preserve_layout: bool = True
    clean_for_audio: bool = True
    add_emotions: bool = True
    enable_checkpoints: bool = ENABLE_STAGE_CHECKPOINTS
    max_retries: int = MAX_RETRIES
    output_format: str = "markdown"
    hyperparameters: Optional[HyperparameterConfig] = field(default_factory=HyperparameterConfig)
    # Configs for local models
    quantization_config: Optional[QuantizationConfig] = None
    layer_split_config: Optional[LayerSplitConfig] = None


@dataclass
class PipelineResult:
    """Result from pipeline processing"""
    success: bool
    input_file: Path
    output_file: Optional[Path]
    processing_time: float
    stages_completed: List[str]
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = None


class ProcessingPipeline:
    """Main processing pipeline for PDF to formatted text"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize processing pipeline
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.logger = get_logger_conf(f"{__name__}.Pipeline")
        
        # Backends are now injected *after* initialization
        self.llm_backend: Optional[LLMBackend] = None 
        
        # Initialize components
        self._initialize_components()
        
        # Track processing state
        self.current_stage: Optional[str] = None
        self.stages_completed: List[str] = []
        self.checkpoint_name: Optional[str] = None
        self.stages_to_run: List[str] = list(PIPELINE_STAGES) # Default to all stages
        
        self.logger.info(f"Initialized pipeline in {self.config.mode.value} mode")
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        self.file_handler = FileHandler()
        
        self.pdf_processor = PDFProcessor(
            preserve_layout=self.config.preserve_layout
        )
        
        self.text_cleaner = PDFTextCleaner()
        self.text_preprocessor = TextPreprocessor()
        self.text_chunker = TextChunker(
            target_size=self.config.chunk_size,
            strategy=self.config.chunking_strategy
        )
        
        # Model manager is NOT initialized here anymore
        # self.model_manager = ... # REMOVED

        # Response filter
        # Try to get model config from registry for filter settings
        model_config_for_filter = None
        if self.config.model_provider == "local":
             model_entry = get_model_config(self.config.model_specifier)
             if model_entry:
                  # Convert ModelEntry to ModelConfig
                  model_config_for_filter = ModelConfig(
                       name=model_entry.name,
                       model_id=model_entry.model_id,
                       supports_thinking=model_entry.supports_thinking,
                       thinking_tokens=model_entry.thinking_tokens or [],
                       max_context=model_entry.max_context,
                       optimal_chunk_size=model_entry.optimal_chunk_size,
                       temperature=model_entry.temperature,
                       top_p=model_entry.top_p,
                       max_new_tokens=model_entry.max_new_tokens,
                       quantization_support=model_entry.quantization_support or ["4bit", "8bit"]
                  )
        
        self.response_filter = ChunkedResponseFilter(
            model_config=model_config_for_filter # Will use defaults if None
        )
        
        # Markdown formatter
        if self.config.mode == ProcessingMode.PODCAST:
            self.formatter = PodcastFormatter()
        elif self.config.mode == ProcessingMode.TECHNICAL:
            self.formatter = TechnicalFormatter()
        else:
            self.formatter = MarkdownFormatter(style=self.config.markdown_style)
            
        # Memory monitor
        self.memory_monitor = MemoryMonitor(self.logger)
        
    def set_stages_to_run(self, stages: List[str]):
        """Explicitly set which stages to run."""
        self.stages_to_run = stages
        self.logger.info(f"Pipeline stages set to: {self.stages_to_run}")

    def process_file(self, 
                    input_path: Path,
                    output_path: Optional[Path] = None) -> PipelineResult:
        """
        Process a single PDF file through the pipeline
        
        Args:
            input_path: Path to input PDF
            output_path: Optional output path (base name, extension added by save stage)
            
        Returns:
            PipelineResult with processing information
        """
        input_path = Path(input_path)
        start_time = time.time()
        
        self.checkpoint_name = f"{input_path.stem}_{int(start_time)}"
        self.stages_completed = [] # Reset for each file
        
        ConsoleOutput.header(f"Processing: {input_path.name}")
        self.logger.info(f"Starting pipeline for: {input_path}")
        
        self.memory_monitor.start()
        
        # Data payload that passes from stage to stage
        data_payload = {}
        
        try:
            # Stage 1: Extract text from PDF
            if "extract" in self.stages_to_run:
                extracted_data = self._stage_extract(input_path)
                if not extracted_data:
                    return self._create_error_result(
                        input_path, "Failed to extract text from PDF"
                    )
                data_payload['text'] = extracted_data.text
                data_payload['metadata'] = extracted_data.metadata
                data_payload['page_texts'] = extracted_data.page_texts
                data_payload['extraction_warnings'] = extracted_data.warnings
            else:
                 # If not extracting, try to read the input file as text
                 self.logger.info("Skipping extraction. Attempting to read input as text.")
                 try:
                     data_payload['text'] = input_path.read_text(encoding='utf-8')
                     data_payload['metadata'] = None # No PDF metadata
                     data_payload['page_texts'] = []
                 except Exception as e:
                     return self._create_error_result(input_path, f"Failed to read input file as text: {e}")

                
            # Stage 2: Preprocess text
            if "preprocess" in self.stages_to_run:
                preprocessed_text = self._stage_preprocess(data_payload['text'])
                data_payload['text'] = preprocessed_text
            
            # Stage 3: Chunk text
            if "chunk" in self.stages_to_run:
                chunks = self._stage_chunk(data_payload['text'])
                data_payload['chunks'] = chunks
            
            # Stage 4: Process chunks with LLM
            if "process" in self.stages_to_run:
                if 'chunks' not in data_payload:
                    ConsoleOutput.error("Cannot run 'process' stage, 'chunk' stage was skipped.")
                    raise ValueError("Chunking stage required before processing stage.")
                if self.llm_backend is None:
                     ConsoleOutput.error("Cannot run 'process' stage, LLM backend is not set.")
                     raise ValueError("LLMBackend not injected into pipeline.")
                     
                processed_chunks = self._stage_process(data_payload['chunks'])
                data_payload['processed_chunks'] = processed_chunks
            
            # Stage 5: Filter responses
            if "filter" in self.stages_to_run:
                if 'processed_chunks' not in data_payload:
                     # If process was skipped, filter the raw chunks
                     if 'chunks' in data_payload:
                          data_payload['processed_chunks'] = data_payload['chunks']
                     else:
                          ConsoleOutput.error("Cannot run 'filter' stage, 'process' or 'chunk' stage was skipped.")
                          raise ValueError("Processing or chunking stage required before filtering stage.")
                          
                filtered_chunks = self._stage_filter(data_payload['processed_chunks'])
                data_payload['filtered_chunks'] = filtered_chunks
            
            # Stage 6: Format output
            if "format" in self.stages_to_run:
                chunks_to_format = data_payload.get('filtered_chunks', data_payload.get('processed_chunks', data_payload.get('chunks')))
                if chunks_to_format is None:
                    # If no chunks, format the raw text
                    chunks_to_format = [data_payload['text']]
                
                formatted_text = self._stage_format(chunks_to_format)
                data_payload['formatted_text'] = formatted_text
            
            # Stage 7: Save output
            saved_path = None
            if "save" in self.stages_to_run:
                text_to_save = data_payload.get('formatted_text')
                if text_to_save is None:
                    # Fallback to saving the last available text form
                    text_to_save = data_payload.get('filtered_chunks')
                    if text_to_save: text_to_save = "\n\n".join(text_to_save)
                    else: 
                         text_to_save = data_payload.get('processed_chunks')
                         if text_to_save: text_to_save = "\n\n".join(text_to_save)
                         else:
                              text_to_save = data_payload.get('text')
                    
                    if text_to_save is None:
                         ConsoleOutput.error("No text available to save.")
                         raise ValueError("No text data found in payload to save.")

                saved_path = self._stage_save(text_to_save, input_path, output_path, data_payload.get('metadata'))
                data_payload['output_file'] = saved_path
            
            # Calculate statistics
            processing_time = time.time() - start_time
            statistics = self._gather_statistics(data_payload)
            
            ConsoleOutput.success(f"Processing complete in {processing_time:.2f}s")
            if saved_path:
                ConsoleOutput.info(f"Output saved to: {saved_path}")
            
            # Record processed file
            self.file_handler.record_processed_file(
                input_path=input_path,
                output_path=saved_path, # This might be None if 'save' wasn't run
                format=self.config.output_format,
                processing_time=processing_time,
                metadata=statistics,
                success=True
            )
            
            return PipelineResult(
                success=True,
                input_file=input_path,
                output_file=saved_path,
                processing_time=processing_time,
                stages_completed=self.stages_completed.copy(),
                statistics=statistics
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed at stage '{self.current_stage}': {e}", exc_info=True)
            ConsoleOutput.error(f"Pipeline failed at stage '{self.current_stage}': {e}")
            processing_time = time.time() - start_time
            
            # Record failure
            self.file_handler.record_processed_file(
                input_path=input_path,
                output_path=None,
                format=self.config.output_format,
                processing_time=processing_time,
                metadata=data_payload.get('metadata', {}), # Save any metadata we got
                success=False,
                error_message=f"Failed at stage {self.current_stage}: {str(e)}"
            )
            
            return self._create_error_result(input_path, str(e))
            
        finally:
            # Memory check at end, but don't unload model here
            self.memory_monitor.check(f"pipeline completion for {input_path.name}")
            
    def _stage_extract(self, input_path: Path) -> Optional[Any]: # Changed return type
        """Stage 1: Extract text from PDF"""
        self.current_stage = "extract"
        ConsoleOutput.section("Stage 1: Extracting text")
        
        # Check for checkpoint
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded extraction checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint # Returns ExtractionResult object
                
        # Extract text
        result = self.pdf_processor.extract_text(input_path)
        
        if not result:
            return None
            
        # Log extraction info
        ConsoleOutput.info(f"Extracted {result.char_count:,} characters from {result.metadata.num_pages} pages")
        
        # Save checkpoint (save the whole result object)
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                result, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        self.memory_monitor.check("after extraction")
        return result # Return ExtractionResult
        
    def _stage_preprocess(self, text: str) -> str:
        """Stage 2: Preprocess text"""
        self.current_stage = "preprocess"
        ConsoleOutput.section("Stage 2: Preprocessing text")
        
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded preprocessing checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        if self.config.clean_for_audio and self.config.mode == ProcessingMode.PODCAST:
            text = self.text_cleaner.clean_for_audio(text)
            ConsoleOutput.info("Applied audio-specific cleaning")
            
        text = self.text_preprocessor.preprocess_for_llm(text)
        
        ConsoleOutput.info(f"Preprocessed text: {len(text):,} characters")
        
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                text, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        return text
        
    def _stage_chunk(self, text: str) -> List[str]:
        """Stage 3: Chunk text"""
        self.current_stage = "chunk"
        ConsoleOutput.section("Stage 3: Chunking text")
        
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded chunking checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint # This should be List[str]
                
        result = self.text_chunker.chunk_text(text)
        chunks = [chunk.text for chunk in result.chunks]
        
        ConsoleOutput.info(f"Created {len(chunks)} chunks (avg size: {result.average_chunk_size:.0f} chars)")
        
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                chunks, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        return chunks
        
    def _stage_process(self, chunks: List[str]) -> List[str]:
        """Stage 4: Process chunks with LLM"""
        self.current_stage = "process"
        ConsoleOutput.section(f"Stage 4: Processing with LLM ({self.llm_backend.model_identifier})")
        
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded processing checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        # Model loading is now handled *outside* the pipeline, by the caller
        # We just check if the backend is assigned
        if self.llm_backend is None:
             raise RuntimeError("LLM Backend not set in pipeline. Cannot process.")
            
        system_prompt = self.config.system_prompt or PREPROCESS_PROMPT
        
        # Use BatchProcessor, which now just iterates (but could be optimized)
        # We need a BatchProcessor instance, or move logic here
        # Let's keep it simple and just iterate here
        
        processed_chunks = []
        with LoggingProgress(self.logger, "Processing chunks", len(chunks)) as progress:
            for i, chunk in enumerate(chunks):
                retry_count = 0
                success = False
                
                while retry_count < self.config.max_retries and not success:
                    try:
                        result = self.llm_backend.process_with_chat_template(
                            system_prompt=system_prompt,
                            user_message=chunk,
                            hyperparams=self.config.hyperparameters,
                            remove_thinking=False  # Filter in next stage
                        )
                        
                        processed_chunks.append(result.raw_output)
                        success = True
                        
                        progress.update(1, f"Chunk {i+1}/{len(chunks)}")
                        
                    except Exception as e:
                        retry_count += 1
                        self.logger.warning(f"Chunk {i+1} failed (attempt {retry_count}/{self.config.max_retries}): {e}")
                        
                        if retry_count < self.config.max_retries:
                            time.sleep(RETRY_DELAY_SECONDS * retry_count) # Exponential backoff
                        else:
                            self.logger.error(f"Chunk {i+1} failed permanently after {self.config.max_retries} attempts.")
                            if FALLBACK_ON_ERROR:
                                processed_chunks.append(chunk) # Use original chunk
                                self.logger.info(f"Using original chunk {i+1} as fallback")
                                success = True # Allow pipeline to continue
                                progress.update(1, f"Chunk {i+1}/{len(chunks)} - Fallback")
                            else:
                                raise RuntimeError(f"Failed to process chunk {i+1}: {e}")
                
                # Memory management (less critical for cloud, vital for local)
                if (i + 1) % 10 == 0:
                    self.memory_monitor.check(f"after processing {i+1} chunks")
                    
        ConsoleOutput.success(f"Processed {len(processed_chunks)} chunks")
        
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                processed_chunks, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        return processed_chunks
        
    def _stage_filter(self, chunks: List[str]) -> List[str]:
        """Stage 5: Filter responses"""
        self.current_stage = "filter"
        ConsoleOutput.section("Stage 5: Filtering responses")
        
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded filtering checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        if not self.config.remove_thinking:
            self.logger.info("Skipping filtering (remove_thinking=False)")
            self.stages_completed.append(self.current_stage)
            return chunks
            
        filtered_chunks = []
        total_removed = 0
        
        for i, chunk in enumerate(chunks):
            is_first = (i == 0)
            is_last = (i == len(chunks) - 1)
            
            # Pass model_config from the pipeline's config to the filter
            # (which was already set in _initialize_components)
            filter_result = self.response_filter.filter_chunk(
                chunk, i, is_first, is_last
            )
            
            filtered_chunks.append(filter_result)
            total_removed += len(chunk) - len(filter_result)
            
        merged_text = self.response_filter.merge_chunks(filtered_chunks)
        
        # Re-split by paragraph for formatting stage
        final_chunks = merged_text.split('\n\n')
        
        ConsoleOutput.info(f"Filtered {total_removed:,} characters of thinking/artifacts")
        
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                final_chunks, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        return final_chunks
        
    def _stage_format(self, chunks: List[str]) -> str:
        """Stage 6: Format output"""
        self.current_stage = "format"
        ConsoleOutput.section("Stage 6: Formatting output")
        
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded formatting checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        full_text = '\n\n'.join(chunks)
        
        # Apply formatting based on mode
        if self.config.mode == ProcessingMode.PODCAST:
            formatted = self.formatter.format_dialogue(
                full_text,
                auto_detect_speakers=True # TODO: Make configurable
            )
        elif self.config.mode == ProcessingMode.TECHNICAL:
            formatted = self.formatter.format_with_headers(
                full_text,
                auto_generate_toc=True # TODO: Make configurable
            )
        else: # Default/Narrative/Summary
            formatted = self.formatter.format_text(
                full_text,
                detect_emotions=self.config.add_emotions,
                add_structure=True
            )
            
        ConsoleOutput.info(f"Formatted output: {len(formatted):,} characters")
        
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                formatted, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        return formatted
        
    def _stage_save(self, 
                    text: str, 
                    input_path: Path, 
                    output_path_base: Optional[Path],
                    pdf_metadata: Optional[Any]) -> Path: # pdf_metadata is PDFMetadata object
        """Stage 7: Save output"""
        self.current_stage = "save"
        ConsoleOutput.section("Stage 7: Saving output")
        
        # Generate output path if not provided
        if output_path_base is None:
            suffix = f"_{self.config.mode.value}"
            extension = f".{self.config.output_format}"
            # Use FileHandler to generate path in the default output dir
            output_path = self.file_handler.get_output_path(
                input_path, suffix, extension
            )
        else:
            # Use the provided base path
            suffix = f"_{self.config.mode.value}"
            extension = f".{self.config.output_format}"
            # Add timestamp if enabled
            if self.file_handler.timestamp_outputs:
                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                 output_filename = f"{output_path_base.stem}{suffix}_{timestamp}{extension}"
            else:
                 output_filename = f"{output_path_base.stem}{suffix}{extension}"
            output_path = output_path_base.with_name(output_filename)


        # Prepare metadata for the output file header
        metadata_to_save = {
            "source_file": str(input_path.resolve()),
            "processing_mode": self.config.mode.value,
            "model_used": self.config.model_name, # This now includes provider
            "stages_completed": self.stages_completed + [self.current_stage]
        }
        
        # Add PDF metadata if available
        if pdf_metadata:
             try:
                 metadata_to_save["original_title"] = pdf_metadata.title
                 metadata_to_save["original_author"] = pdf_metadata.author
                 metadata_to_save["original_pages"] = pdf_metadata.num_pages
             except Exception as e:
                 logger.warning(f"Could not parse PDF metadata for saving: {e}")

        
        # Save file
        saved_path = self.file_handler.save_text(
            text, output_path, 
            format=self.config.output_format,
            metadata=metadata_to_save
        )
        
        self.stages_completed.append(self.current_stage)
        return saved_path
        
    def _gather_statistics(self, data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gather processing statistics from the data payload"""
        stats = { "stages_completed": len(self.stages_completed) }
        
        original_text = ""
        if 'metadata' in data_payload and data_payload['metadata'] is not None:
            stats["original_pages"] = data_payload['metadata'].num_pages
            # 'text' at the *start* of the payload is the extracted text
            extracted_text = data_payload.get('text_from_extract') # Need to save this explicitly
        
        # Let's refine payload management (this is a bit of a patch)
        # Better: Pass payload through and add keys
        
        # Simplified stats based on what _stage_save has
        formatted_text = data_payload.get('formatted_text', "")
        preprocessed_text = data_payload.get('text', "") # 'text' gets overwritten
        num_chunks = len(data_payload.get('chunks', []))
        processed_chars = sum(len(c) for c in data_payload.get('processed_chunks', []))
        
        stats["num_chunks"] = num_chunks
        stats["processed_chars"] = processed_chars
        stats["final_chars"] = len(formatted_text)
        
        # Cannot reliably get original_chars here unless saved separately
        # Let's assume 'text' in payload *after extract* is what we want
        # This part is fragile.
        
        return stats
        
    def _create_error_result(self, input_path: Path, error_message: str) -> PipelineResult:
        """Create an error result"""
        return PipelineResult(
            success=False,
            input_file=input_path,
            output_file=None,
            processing_time=0, # Will be set by the caller
            stages_completed=self.stages_completed.copy(),
            error_message=f"Failed at stage {self.current_stage}: {error_message}"
        )
        
    def process_batch(self, 
                     input_files: List[Path],
                     output_dir: Optional[Path] = None,
                     parallel: bool = False) -> List[PipelineResult]:
        """
        Process multiple files
        
        Args:
            input_files: List of input files
            output_dir: Directory to save files (if None, uses handler default)
            parallel: Whether to process in parallel (not implemented)
            
        Returns:
            List of processing results
        """
        if parallel:
             ConsoleOutput.warning("Parallel processing is not yet implemented. Running sequentially.")
             
        ConsoleOutput.header(f"Batch Processing: {len(input_files)} files")
        
        results = []
        # Use the pipeline's FileHandler's output_dir unless overridden
        if output_dir:
             self.file_handler.output_dir = output_dir
             
        for i, input_file in enumerate(input_files, 1):
            ConsoleOutput.info(f"\n--- Processing File {i}/{len(input_files)} ---")
            
            # Output path base is None, so process_file uses file_handler
            # to generate one based on the input_file.name
            result = self.process_file(input_file, output_path=None) 
            results.append(result)
            
            # Unload local models between files to conserve VRAM
            if self.llm_backend and self.llm_backend.provider_identifier.startswith("local"):
                ConsoleOutput.info("Unloading local model to conserve memory...")
                self.llm_backend.unload_model()
                self.memory_monitor.check(f"after unloading model from file {i}")
                
        # Generate summary
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.processing_time for r in results)
        
        ConsoleOutput.header("Batch Processing Complete")
        ConsoleOutput.info(f"Successful: {successful}/{len(results)}")
        ConsoleOutput.info(f"Total time: {total_time:.2f}s")
        
        # Report saving is now handled in cleanup()
        
        return results
        
    def cleanup_file_handler(self):
        """Saves report and cleans cache, without touching models."""
        self.logger.info("Cleaning up pipeline file handler")
        if self.file_handler:
             if self.file_handler.processed_files: # Only save if we did something
                 self.file_handler.save_processing_report()
             # Clean old cache files
             self.file_handler.cleanup_old_files(days=7)

    def cleanup(self):
        """Clean up resources (called by _execute_pipeline in menu)"""
        # Note: Model unloading is handled by the caller (_execute_pipeline)
        # This cleanup is for non-model resources, like saving the report.
        self.cleanup_file_handler()
        self.logger.info("Pipeline cleanup finished.")
