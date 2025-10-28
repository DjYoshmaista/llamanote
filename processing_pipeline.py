"""
Processing Pipeline Module
Main pipeline that orchestrates the PDF processing workflow
"""
import gc
import torch
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime # Added for timestamping output

# --- LlamaNote Modules ---
from loggerConf import get_logger_conf, LoggingProgress, MemoryMonitor, ConsoleOutput
from config_base import *
from config import ( # Keep specific config values if needed, but PipelineConfig comes from types
    PREPROCESS_PROMPT,
    MODELS,
    DEFAULT_MODEL,
    MEMORY_PROFILES,
    MARKDOWN_STYLES,
    PIPELINE_STAGES,
    ENABLE_STAGE_CHECKPOINTS,
    MAX_RETRIES,
    RETRY_DELAY_SECONDS,
    FALLBACK_ON_ERROR
)
# Import shared types from the new file
from pipeline_types import (
    ProcessingMode,
    PipelineConfig,
    PipelineResult,
    QuantizationConfig,
    LayerSplitConfig
)
from pdf_processor import PDFProcessor, PDFTextCleaner, PDFMetadata, ExtractionResult
from text_processor import TextChunker, TextPreprocessor, ChunkingStrategy, TextChunk
from llm_handler import (
    LLMBackend, GenerationResult, get_llm_backend
)
from response_filter import ChunkedResponseFilter, FilterResult
from markdown_formatter import MarkdownFormatter, PodcastFormatter, TechnicalFormatter
from file_handler import FileHandler
from hyperparameters import HyperparameterConfig
from model_registry import get_registry, get_model_config
from model_registry import ModelEntry

logger = get_logger_conf(__name__)
# Removed TIMESTAMP_OUTPUTS from here, should be handled by FileHandler/config_base

class ProcessingPipeline:
    """Main processing pipeline for PDF to formatted text"""

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize processing pipeline

        Args:
            config: Pipeline configuration from pipeline_types
        """
        self.config = config or PipelineConfig() # Use PipelineConfig from types
        self.logger = get_logger_conf(f"{__name__}.Pipeline")

        # Backend is now injected, not created here.
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
        self.file_handler = FileHandler(timestamp_outputs=TIMESTAMP_OUTPUTS) # Pass timestamp setting

        self.pdf_processor = PDFProcessor(
            preserve_layout=self.config.preserve_layout,
            max_chars=MAX_CHARS_PER_FILE, # Use base config value
            max_size_mb=MAX_PDF_SIZE_MB # Use base config value
        )

        self.text_cleaner = PDFTextCleaner()
        self.text_preprocessor = TextPreprocessor()
        self.text_chunker = TextChunker(
            target_size=self.config.chunk_size,
            strategy=self.config.chunking_strategy
        )

        # Response filter - needs ModelEntry potentially
        # *** FIX 3: Assign model_entry directly, don't re-construct ***
        model_config_for_filter = None
        if self.config.model_provider in ["local", "local_gguf"]: # Check for both local types
            model_config_for_filter = get_model_config(self.config.model_specifier) # Get from registry
            if model_config_for_filter is None:
                logger.warning(f"Model {self.config.model_specifier} not in registry. Filter may not work correctly.")
                # Create a basic default so it doesn't crash, including the required 'author'
                model_config_for_filter = ModelEntry(
                    name=Path(self.config.model_specifier).name,
                    model_id=self.config.model_specifier,
                    author="unknown" # Add the missing required argument
                )

        self.response_filter = ChunkedResponseFilter(
            model_config=model_config_for_filter # Pass the correct ModelEntry object or None
        )


        # Markdown formatter
        if self.config.mode == ProcessingMode.PODCAST:
            self.formatter = PodcastFormatter()
        elif self.config.mode == ProcessingMode.TECHNICAL:
            self.formatter = TechnicalFormatter()
        else:
            # Default to podcast style if not recognized, or use a new 'default' style
            style_name = self.config.markdown_style
            if style_name not in MARKDOWN_STYLES:
                logger.warning(f"Markdown style '{style_name}' not found, falling back to 'podcast'.")
                style_name = "podcast"
            self.formatter = MarkdownFormatter(style=style_name)

        # Memory monitor
        self.memory_monitor = MemoryMonitor(self.logger)

    def set_stages_to_run(self, stages: List[str]):
        """Explicitly set which stages to run."""
        self.stages_to_run = [s for s in stages if s in PIPELINE_STAGES] # Filter invalid stages
        self.logger.info(f"Pipeline stages set to run: {self.stages_to_run}")

    def process_file(self,
                    input_path: Path,
                    output_path: Optional[Path] = None) -> PipelineResult: # Use PipelineResult from types
        """
        Process a single PDF file through the pipeline

        Args:
            input_path: Path to input PDF or text file
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
        data_payload: Dict[str, Any] = {
            'input_path': input_path,
            'text_from_extract': None, # Store initial extraction separately
        }

        try:
            # Stage 1: Extract text
            if "extract" in self.stages_to_run:
                # Check if input is PDF, otherwise skip extraction
                if input_path.suffix.lower() == '.pdf':
                    extracted_data: Optional[ExtractionResult] = self._stage_extract(input_path)
                    if not extracted_data:
                        return self._create_error_result(
                            input_path, "Failed to extract text from PDF"
                        )
                    data_payload['text_from_extract'] = extracted_data.text
                    data_payload['text'] = extracted_data.text # Current text to process
                    data_payload['metadata'] = extracted_data.metadata
                    data_payload['page_texts'] = extracted_data.page_texts
                    data_payload['extraction_warnings'] = extracted_data.warnings
                elif input_path.suffix.lower() in ['.txt', '.md']:
                    self.logger.info("Input is text file, skipping PDF extraction.")
                    try:
                         text_content = input_path.read_text(encoding='utf-8')
                         data_payload['text_from_extract'] = text_content
                         data_payload['text'] = text_content
                         data_payload['metadata'] = None # No PDF metadata
                         data_payload['page_texts'] = []
                         self.stages_completed.append("extract") # Mark as completed conceptually
                    except Exception as e:
                         return self._create_error_result(input_path, f"Failed to read input text file: {e}")
                else:
                    return self._create_error_result(input_path, f"Unsupported file type for extraction: {input_path.suffix}")

            elif "extract" not in self.stages_to_run and 'text' not in data_payload:
                 # If extract is skipped, we must have text input from somewhere (e.g., previous run or direct input)
                 # Check if input file is text-based and read it
                 self.logger.info("Skipping extraction. Attempting to read input as text.")
                 if input_path.suffix.lower() in ['.txt', '.md']:
                     try:
                         text_content = input_path.read_text(encoding='utf-8')
                         data_payload['text_from_extract'] = text_content
                         data_payload['text'] = text_content
                         data_payload['metadata'] = None
                         data_payload['page_texts'] = []
                     except Exception as e:
                         return self._create_error_result(input_path, f"Failed to read input file as text when skipping extract: {e}")
                 else:
                     return self._create_error_result(input_path, "Extraction skipped, but input is not a text file.")


            # Stage 2: Preprocess text
            if "preprocess" in self.stages_to_run:
                if 'text' not in data_payload: return self._missing_data_error(input_path, "preprocess", "text")
                preprocessed_text = self._stage_preprocess(data_payload['text'])
                data_payload['text'] = preprocessed_text

            # Stage 3: Chunk text
            if "chunk" in self.stages_to_run:
                if 'text' not in data_payload: return self._missing_data_error(input_path, "chunk", "text")
                chunks_result = self._stage_chunk(data_payload['text']) # Returns ChunkingResult
                data_payload['chunk_result'] = chunks_result
                data_payload['chunks'] = [c.text for c in chunks_result.chunks] # Store just the text list too
            else:
                 # If chunking is skipped but processing is needed, treat the whole text as one chunk
                 if "process" in self.stages_to_run and 'text' in data_payload:
                      data_payload['chunks'] = [data_payload['text']]
                      self.logger.info("Chunking skipped, treating whole document as one chunk for processing.")


            # Stage 4: Process chunks with LLM
            if "process" in self.stages_to_run:
                if 'chunks' not in data_payload: return self._missing_data_error(input_path, "process", "chunks")
                if self.llm_backend is None:
                     ConsoleOutput.error("Cannot run 'process' stage, LLM backend is not set.")
                     return self._create_error_result(input_path, "LLMBackend not set.")
                
                # Check if backend is loaded
                if self.llm_backend.model is None:
                    self.logger.info(f"Loading LLM backend {self.llm_backend.model_identifier} for processing...")
                    if not self.llm_backend.load_model():
                        return self._create_error_result(input_path, "Failed to load LLM backend for processing.")

                processed_chunks = self._stage_process(data_payload['chunks'])
                data_payload['processed_chunks'] = processed_chunks # List of strings

            # Stage 5: Filter responses
            if "filter" in self.stages_to_run:
                chunks_to_filter = data_payload.get('processed_chunks')
                if chunks_to_filter is None:
                    # If process stage was skipped, maybe filter the raw chunks?
                    chunks_to_filter = data_payload.get('chunks')
                    if chunks_to_filter is None:
                         # Or maybe filter the preprocessed text?
                         text_to_filter = data_payload.get('text')
                         if text_to_filter:
                              chunks_to_filter = [text_to_filter] # Treat as one chunk
                         else:
                              return self._missing_data_error(input_path, "filter", "processed_chunks or chunks or text")
                
                filtered_chunks = self._stage_filter(chunks_to_filter) # Returns List[str] (paragraphs)
                data_payload['filtered_chunks'] = filtered_chunks

            # Stage 6: Format output
            if "format" in self.stages_to_run:
                # Decide which text to format based on completed stages
                text_source = None
                if 'filtered_chunks' in data_payload: text_source = data_payload['filtered_chunks']
                elif 'processed_chunks' in data_payload: text_source = data_payload['processed_chunks']
                elif 'chunks' in data_payload: text_source = data_payload['chunks']
                elif 'text' in data_payload: text_source = [data_payload['text']] # Treat as one chunk
                else: return self._missing_data_error(input_path, "format", "any text content")

                # If source is list of chunks, join them first
                text_to_format = "\n\n".join(text_source) if isinstance(text_source, list) else text_source

                formatted_text = self._stage_format(text_to_format)
                data_payload['formatted_text'] = formatted_text

            # Stage 7: Save output
            saved_path = None
            if "save" in self.stages_to_run:
                text_to_save = data_payload.get('formatted_text')
                if text_to_save is None:
                    # Fallback to saving the last available text form
                    last_available_text = None
                    if 'filtered_chunks' in data_payload: last_available_text = "\n\n".join(data_payload['filtered_chunks'])
                    elif 'processed_chunks' in data_payload: last_available_text = "\n\n".join(data_payload['processed_chunks'])
                    elif 'chunks' in data_payload: last_available_text = "\n\n".join(data_payload['chunks'])
                    elif 'text' in data_payload: last_available_text = data_payload['text']

                    if last_available_text is None:
                         return self._missing_data_error(input_path, "save", "any text content")
                    text_to_save = last_available_text
                    self.logger.warning("Formatting stage skipped or failed, saving last available text content.")

                
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
            self.logger.error(f"Pipeline failed at stage '{self.current_stage or 'initialization'}': {e}", exc_info=True)
            ConsoleOutput.error(f"Pipeline failed at stage '{self.current_stage or 'initialization'}': {e}")
            processing_time = time.time() - start_time

            # Record failure
            self.file_handler.record_processed_file(
                input_path=input_path,
                output_path=None,
                format=self.config.output_format,
                processing_time=processing_time,
                metadata=data_payload.get('metadata', {}), # Save any metadata we got
                success=False,
                error_message=f"Failed at stage {self.current_stage or 'initialization'}: {str(e)}"
            )

            return self._create_error_result(input_path, str(e))

        finally:
            self.memory_monitor.check(f"pipeline completion for {input_path.name}")
            # Do NOT unload model here, caller manages backend lifecycle
            pass


    def _stage_extract(self, input_path: Path) -> Optional[ExtractionResult]:
        """Stage 1: Extract text from PDF"""
        self.current_stage = "extract"
        ConsoleOutput.section("Stage 1: Extracting text")

        # Check for checkpoint
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint and isinstance(checkpoint, ExtractionResult):
                self.logger.info("Loaded extraction checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint

        # Extract text
        result = self.pdf_processor.extract_text(input_path)

        if not result:
            return None

        ConsoleOutput.info(f"Extracted {result.char_count:,} characters from {result.metadata.num_pages} pages")
        for warning in result.warnings:
             ConsoleOutput.warning(warning)

        # Save checkpoint (save the whole result object)
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                result, self.checkpoint_name, self.current_stage
            )

        self.stages_completed.append(self.current_stage)
        self.memory_monitor.check("after extraction")
        return result

    def _stage_preprocess(self, text: str) -> str:
        """Stage 2: Preprocess text"""
        self.current_stage = "preprocess"
        ConsoleOutput.section("Stage 2: Preprocessing text")

        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint and isinstance(checkpoint, str):
                self.logger.info("Loaded preprocessing checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint

        cleaned_text = text # Start with input text
        if self.config.clean_for_audio and self.config.mode == ProcessingMode.PODCAST:
            cleaned_text = self.text_cleaner.clean_for_audio(cleaned_text)
            ConsoleOutput.info("Applied audio-specific cleaning")

        preprocessed_text = self.text_preprocessor.preprocess_for_llm(cleaned_text)

        ConsoleOutput.info(f"Preprocessed text: {len(preprocessed_text):,} characters")

        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                preprocessed_text, self.checkpoint_name, self.current_stage
            )

        self.stages_completed.append(self.current_stage)
        return preprocessed_text

    def _stage_chunk(self, text: str) -> 'ChunkingResult': # Use type hint from text_processor
        """Stage 3: Chunk text"""
        self.current_stage = "chunk"
        ConsoleOutput.section("Stage 3: Chunking text")

        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            # Need to check if loaded checkpoint is the correct type (ChunkingResult)
            if checkpoint and hasattr(checkpoint, 'chunks') and hasattr(checkpoint, 'strategy_used'):
                self.logger.info("Loaded chunking checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint

        result = self.text_chunker.chunk_text(text)
        # chunks = [chunk.text for chunk in result.chunks] # We need the full result for checkpoint

        ConsoleOutput.info(f"Created {len(result.chunks)} chunks (avg size: {result.average_chunk_size:.0f} chars)")

        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                result, self.checkpoint_name, self.current_stage # Save the ChunkingResult object
            )

        self.stages_completed.append(self.current_stage)
        return result

    def _stage_process(self, chunks: List[str]) -> List[str]:
        """Stage 4: Process chunks with LLM"""
        self.current_stage = "process"
        ConsoleOutput.section(f"Stage 4: Processing with LLM ({self.llm_backend.model_identifier})")

        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint and isinstance(checkpoint, list):
                self.logger.info("Loaded processing checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint

        if self.llm_backend is None:
             raise RuntimeError("LLM Backend not set in pipeline. Cannot process.")
        
        # Ensure model is loaded (it should be, but check)
        if self.llm_backend.model is None:
            self.logger.warning("LLM backend model was not loaded. Attempting to load now.")
            if not self.llm_backend.load_model():
                self.logger.error("Failed to load LLM backend model during processing stage.")
                raise RuntimeError("Failed to load LLM backend model.")

        system_prompt = self.config.system_prompt or PREPROCESS_PROMPT

        processed_chunks = []
        with LoggingProgress(self.logger, f"Processing {len(chunks)} chunks", len(chunks)) as progress:
            for i, chunk in enumerate(chunks):
                retry_count = 0
                success = False
                result_text = chunk # Default to original if fallback enabled

                while retry_count <= self.config.max_retries and not success:
                    try:
                        self.memory_monitor.check(f"before processing chunk {i+1}")
                        result: GenerationResult = self.llm_backend.process_with_chat_template(
                            system_prompt=system_prompt,
                            user_message=chunk,
                            hyperparams=self.config.hyperparameters,
                            remove_thinking=False  # Filter in next stage
                        )

                        # Check if result indicates an error (e.g., API error)
                        if "[Error:" in result.raw_output or "[Blocked" in result.raw_output:
                             raise RuntimeError(f"LLM API Error: {result.raw_output}")

                        result_text = result.raw_output
                        processed_chunks.append(result_text)
                        success = True
                        progress.update(1, f"Chunk {i+1}/{len(chunks)} OK")

                    except Exception as e:
                        retry_count += 1
                        self.logger.warning(f"Chunk {i+1} failed (attempt {retry_count}/{self.config.max_retries}): {e}")

                        if retry_count <= self.config.max_retries:
                            ConsoleOutput.warning(f"Retrying chunk {i+1} (attempt {retry_count})...")
                            time.sleep(RETRY_DELAY_SECONDS * (2**(retry_count-1))) # Exponential backoff
                        else:
                            self.logger.error(f"Chunk {i+1} failed permanently after {self.config.max_retries} attempts.")
                            if FALLBACK_ON_ERROR:
                                processed_chunks.append(chunk) # Use original chunk
                                self.logger.info(f"Using original chunk {i+1} as fallback")
                                success = True # Allow pipeline to continue
                                progress.update(1, f"Chunk {i+1}/{len(chunks)} - Fallback")
                            else:
                                raise RuntimeError(f"Failed to process chunk {i+1}: {e}") # Propagate error if no fallback

                # Memory management (clear CUDA cache periodically for local models)
                if self.llm_backend.provider_identifier.startswith("local") and torch.cuda.is_available():
                    if (i + 1) % 5 == 0: # Clear every 5 chunks
                         gc.collect()
                         torch.cuda.empty_cache()
                         self.memory_monitor.check(f"after clearing cache (chunk {i+1})")

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
            if checkpoint and isinstance(checkpoint, list): # Should be list of strings
                self.logger.info("Loaded filtering checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint

        if not self.config.remove_thinking:
            self.logger.info("Skipping filtering (remove_thinking=False)")
            self.stages_completed.append(self.current_stage)
            return chunks # Return the input chunks directly

        filtered_paragraph_chunks = []
        total_removed_chars = 0
        original_total_chars = sum(len(c) for c in chunks)

        with LoggingProgress(self.logger, "Filtering chunks", len(chunks)) as progress:
            for i, chunk in enumerate(chunks):
                is_first = (i == 0)
                is_last = (i == len(chunks) - 1)

                # Filter individual chunk (might remove thinking tags)
                filter_result: FilterResult = self.response_filter.filter.filter(chunk) # Use the base filter here
                filtered_text = filter_result.filtered_text
                total_removed_chars += filter_result.removal_ratio * len(chunk)

                # Store the cleaned chunk text
                filtered_paragraph_chunks.append(filtered_text)
                progress.update(1)

        # Use the ChunkedResponseFilter's merge logic to smooth boundaries
        merged_text = self.response_filter.merge_chunks(filtered_paragraph_chunks)

        # Re-split into paragraphs for consistency with potential checkpoint format
        final_paragraphs = merged_text.split('\n\n')
        final_paragraphs = [p.strip() for p in final_paragraphs if p.strip()] # Clean empty paragraphs

        ConsoleOutput.info(f"Filtered ~{total_removed_chars:.0f} characters of thinking/artifacts "
                           f"({(total_removed_chars / max(original_total_chars, 1)) * 100:.1f}%)")

        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                final_paragraphs, self.checkpoint_name, self.current_stage
            )

        self.stages_completed.append(self.current_stage)
        return final_paragraphs # Return list of filtered paragraphs

    def _stage_format(self, text_to_format: str) -> str: # Takes joined text now
        """Stage 6: Format output"""
        self.current_stage = "format"
        ConsoleOutput.section("Stage 6: Formatting output")

        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint and isinstance(checkpoint, str):
                self.logger.info("Loaded formatting checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint


        # Apply formatting based on mode
        formatted_text = ""
        if self.config.mode == ProcessingMode.PODCAST:
            # Assuming self.formatter is PodcastFormatter
            formatted_text = self.formatter.format_dialogue(
                text_to_format,
                auto_detect_speakers=True
            )
        elif self.config.mode == ProcessingMode.TECHNICAL:
             # Assuming self.formatter is TechnicalFormatter
             formatted_text = self.formatter.format_with_headers(
                 text_to_format,
                 auto_generate_toc=True
             )
        else: # Default/Narrative/Summary
            formatted_text = self.formatter.format_text(
                text_to_format,
                detect_emotions=self.config.add_emotions,
                add_structure=True
            )

        ConsoleOutput.info(f"Formatted output: {len(formatted_text):,} characters")

        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                formatted_text, self.checkpoint_name, self.current_stage
            )

        self.stages_completed.append(self.current_stage)
        return formatted_text

    def _stage_save(self,
                    text: str,
                    input_path: Path,
                    output_path_base: Optional[Path],
                    pdf_metadata: Optional[PDFMetadata]) -> Path: # pdf_metadata type hint
        """Stage 7: Save output"""
        self.current_stage = "save"
        ConsoleOutput.section("Stage 7: Saving output")

        # Generate output path using FileHandler logic
        # If output_path_base is provided (e.g., from CLI), use it
        # Otherwise, generate a path based on the input file
        if output_path_base:
             # Ensure the directory exists
             output_path_base.parent.mkdir(parents=True, exist_ok=True)
             # Apply suffix and extension
             output_path = output_path_base.with_suffix(f".{self.config.output_format}")
             # Note: This logic assumes output_path_base does *not* include the suffix/timestamp
             # A better way might be for get_output_path to handle an optional base name
        else:
             output_path = self.file_handler.get_output_path(
                input_path=input_path,
                suffix=f"_{self.config.mode.value}",
                extension=f".{self.config.output_format}"
             )


        # Prepare metadata for the output file header
        metadata_to_save = {
            "source_file": str(input_path.resolve()),
            "processing_mode": self.config.mode.value,
            "model_used": self.config.model_name, # Includes provider:specifier
            "stages_run": self.stages_to_run, # Record which stages were configured to run
            "stages_completed": self.stages_completed + [self.current_stage] # Record stages actually finished
        }

        # Add PDF metadata if available
        if pdf_metadata and isinstance(pdf_metadata, PDFMetadata):
             try:
                 if pdf_metadata.title: metadata_to_save["original_title"] = pdf_metadata.title
                 if pdf_metadata.author: metadata_to_save["original_author"] = pdf_metadata.author
                 metadata_to_save["original_pages"] = pdf_metadata.num_pages
             except Exception as e:
                 logger.warning(f"Could not parse PDF metadata for saving: {e}")


        # Save file using FileHandler
        saved_path = self.file_handler.save_text(
            text, output_path,
            format=self.config.output_format, # Use config format
            metadata=metadata_to_save if INCLUDE_METADATA else None # Use base config flag
        )

        if not saved_path:
             raise IOError(f"Failed to save output file to {output_path}")

        self.stages_completed.append(self.current_stage)
        return saved_path

    def _gather_statistics(self, data_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Gather processing statistics from the data payload"""
        stats = {
             "stages_run": self.stages_to_run,
             "stages_completed": self.stages_completed
        }

        if 'metadata' in data_payload and data_payload['metadata'] is not None:
            stats["original_pages"] = data_payload['metadata'].num_pages
            stats["original_chars_extracted"] = len(data_payload.get('text_from_extract', ''))

        if 'chunk_result' in data_payload:
             chunk_res = data_payload['chunk_result']
             stats["num_chunks"] = chunk_res.total_chunks
             stats["avg_chunk_size"] = chunk_res.average_chunk_size
             stats["chunk_strategy"] = chunk_res.strategy_used.value

        stats["chars_after_preprocess"] = len(data_payload.get('text', '')) # 'text' holds preprocessed text after stage 2
        stats["chars_after_llm"] = sum(len(c) for c in data_payload.get('processed_chunks', []))
        stats["chars_after_filter"] = sum(len(c) for c in data_payload.get('filtered_chunks', [])) # Chunks are paragraphs here
        stats["final_formatted_chars"] = len(data_payload.get('formatted_text', ''))

        return stats

    def _create_error_result(self, input_path: Path, error_message: str) -> PipelineResult:
        """Create an error result, ensuring stage is set."""
        stage = self.current_stage or "unknown"
        full_error = f"Failed at stage {stage}: {error_message}"
        self.logger.error(f"Creating error result for {input_path.name}: {full_error}")
        return PipelineResult(
            success=False,
            input_file=input_path,
            output_file=None,
            processing_time=0, # Will be set by the caller's timer
            stages_completed=self.stages_completed.copy(),
            error_message=full_error,
            statistics={"stages_completed": self.stages_completed.copy()}
        )

    def _missing_data_error(self, input_path: Path, stage: str, missing_key: str) -> PipelineResult:
        """Create an error result for missing data between stages."""
        self.current_stage = stage
        error_msg = f"Missing required data '{missing_key}' for stage '{stage}'. A previous stage might have been skipped or failed."
        ConsoleOutput.error(error_msg)
        return self._create_error_result(input_path, error_msg)

    def process_batch(self,
                     input_files: List[Path],
                     output_dir: Optional[Path] = None,
                     parallel: bool = False) -> List[PipelineResult]:
        """
        Process multiple files sequentially.

        Args:
            input_files: List of input files (PDF or text)
            output_dir: Directory to save output files (overrides handler default)
            parallel: (Not implemented)

        Returns:
            List of processing results
        """
        if parallel:
             ConsoleOutput.warning("Parallel processing is not yet implemented. Running sequentially.")

        ConsoleOutput.header(f"Batch Processing: {len(input_files)} files")

        results = []
        original_output_dir = self.file_handler.output_dir # Store original
        if output_dir:
             self.file_handler.output_dir = output_dir.resolve()
             self.file_handler.output_dir.mkdir(parents=True, exist_ok=True)
             self.logger.info(f"Using specified output directory for batch: {self.file_handler.output_dir}")

        for i, input_file in enumerate(input_files, 1):
            ConsoleOutput.subsection(f"--- Processing File {i}/{len(input_files)}: {input_file.name} ---")

            # Output path base is None, process_file uses file_handler to generate full path
            result = self.process_file(input_file, output_path=None)
            results.append(result)

            # --- Important: Unload local models between files ---
            if self.llm_backend and self.llm_backend.provider_identifier.startswith("local"):
                ConsoleOutput.info("Unloading local text model to conserve memory...")
                self.llm_backend.unload_model()
                self.memory_monitor.check(f"after unloading model from file {i}")
                # The model will be reloaded (if needed) in the next call to process_file
                # inside _stage_process, if the 'process' stage is active.

        # Restore original output dir if it was overridden
        self.file_handler.output_dir = original_output_dir

        # Generate summary
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.processing_time for r in results)

        ConsoleOutput.header("Batch Processing Complete")
        ConsoleOutput.info(f"Successful: {successful}/{len(results)}")
        ConsoleOutput.info(f"Total time: {total_time:.2f}s")

        # Report saving is now handled by cleanup_file_handler or cleanup
        return results

    def cleanup_file_handler(self):
        """Saves report and cleans cache, without touching models."""
        self.logger.info("Cleaning up pipeline file handler")
        if hasattr(self, 'file_handler') and self.file_handler:
             if self.file_handler.processed_files: # Only save if we did something
                 try:
                     report_path = self.file_handler.save_processing_report()
                     ConsoleOutput.info(f"Processing report saved: {report_path.name}")
                 except Exception as e:
                     self.logger.error(f"Failed to save processing report: {e}")
             # Clean old cache files
             try:
                 self.file_handler.cleanup_old_files(days=7)
             except Exception as e:
                 self.logger.error(f"Failed to cleanup old files: {e}")

    def cleanup(self):
        """Clean up resources (called by _execute_pipeline in menu)"""
        # Note: Model unloading is handled by the caller (_execute_pipeline)
        # This cleanup is for non-model resources, like saving the report.
        self.cleanup_file_handler()
        self.logger.info("Pipeline cleanup finished.")
