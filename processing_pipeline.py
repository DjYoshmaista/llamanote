"""
Processing Pipeline Module
Main pipeline that orchestrates the PDF processing workflow
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from logging_config import get_logger, LoggingProgress, MemoryMonitor, ConsoleOutput
from config import (
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
from pdf_processor import PDFProcessor, PDFTextCleaner
from text_processor import TextChunker, TextPreprocessor, ChunkingStrategy
from llm_handler import ModelManager, BatchProcessor
from response_filter import ChunkedResponseFilter
from markdown_formatter import MarkdownFormatter, PodcastFormatter, TechnicalFormatter
from file_handler import FileHandler

logger = get_logger(__name__)


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
    model_name: str = DEFAULT_MODEL
    memory_profile: str = "medium_vram"
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY
    chunk_size: int = 1000
    markdown_style: str = "podcast"
    system_prompt: Optional[str] = None
    remove_thinking: bool = True
    preserve_layout: bool = False
    clean_for_audio: bool = True
    add_emotions: bool = True
    enable_checkpoints: bool = ENABLE_STAGE_CHECKPOINTS
    max_retries: int = MAX_RETRIES
    output_format: str = "markdown"


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
        self.logger = get_logger(f"{__name__}.Pipeline")
        
        # Initialize components
        self._initialize_components()
        
        # Track processing state
        self.current_stage = None
        self.stages_completed = []
        self.checkpoint_name = None
        
        self.logger.info(f"Initialized pipeline in {self.config.mode.value} mode")
        
    def _initialize_components(self):
        """Initialize all pipeline components"""
        # File handler
        self.file_handler = FileHandler()
        
        # PDF processor
        self.pdf_processor = PDFProcessor(
            preserve_layout=self.config.preserve_layout
        )
        
        # Text processors
        self.text_cleaner = PDFTextCleaner()
        self.text_preprocessor = TextPreprocessor()
        self.text_chunker = TextChunker(
            target_size=self.config.chunk_size,
            strategy=self.config.chunking_strategy
        )
        
        # Model manager
        memory_config = MEMORY_PROFILES.get(self.config.memory_profile)
        self.model_manager = ModelManager(
            model_name=self.config.model_name,
            memory_config=memory_config
        )
        
        # Response filter
        self.response_filter = ChunkedResponseFilter(
            model_config=MODELS.get(self.config.model_name)
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
        
    def process_file(self, 
                    input_path: Path,
                    output_path: Optional[Path] = None) -> PipelineResult:
        """
        Process a single PDF file through the pipeline
        
        Args:
            input_path: Path to input PDF
            output_path: Optional output path
            
        Returns:
            PipelineResult with processing information
        """
        input_path = Path(input_path)
        start_time = time.time()
        
        # Generate checkpoint name
        self.checkpoint_name = f"{input_path.stem}_{int(start_time)}"
        
        ConsoleOutput.header(f"Processing: {input_path.name}")
        self.logger.info(f"Starting pipeline for: {input_path}")
        
        # Start memory monitoring
        self.memory_monitor.start()
        
        try:
            # Stage 1: Extract text from PDF
            extracted_text = self._stage_extract(input_path)
            if not extracted_text:
                return self._create_error_result(
                    input_path, "Failed to extract text from PDF"
                )
                
            # Stage 2: Preprocess text
            preprocessed_text = self._stage_preprocess(extracted_text)
            
            # Stage 3: Chunk text
            chunks = self._stage_chunk(preprocessed_text)
            
            # Stage 4: Process chunks with LLM
            processed_chunks = self._stage_process(chunks)
            
            # Stage 5: Filter responses
            filtered_chunks = self._stage_filter(processed_chunks)
            
            # Stage 6: Format output
            formatted_text = self._stage_format(filtered_chunks)
            
            # Stage 7: Save output
            output_path = self._stage_save(formatted_text, input_path, output_path)
            
            # Calculate statistics
            processing_time = time.time() - start_time
            statistics = self._gather_statistics(
                extracted_text, preprocessed_text, 
                chunks, processed_chunks, formatted_text
            )
            
            ConsoleOutput.success(f"Processing complete in {processing_time:.2f}s")
            ConsoleOutput.info(f"Output saved to: {output_path}")
            
            # Record processed file
            self.file_handler.record_processed_file(
                input_path=input_path,
                output_path=output_path,
                format=self.config.output_format,
                processing_time=processing_time,
                metadata=statistics,
                success=True
            )
            
            return PipelineResult(
                success=True,
                input_file=input_path,
                output_file=output_path,
                processing_time=processing_time,
                stages_completed=self.stages_completed.copy(),
                statistics=statistics
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            processing_time = time.time() - start_time
            
            # Record failure
            self.file_handler.record_processed_file(
                input_path=input_path,
                output_path=None,
                format=self.config.output_format,
                processing_time=processing_time,
                metadata={},
                success=False,
                error_message=str(e)
            )
            
            return self._create_error_result(input_path, str(e))
            
        finally:
            # Clean up
            self.memory_monitor.check("pipeline completion")
            
    def _stage_extract(self, input_path: Path) -> Optional[str]:
        """Stage 1: Extract text from PDF"""
        self.current_stage = "extract"
        ConsoleOutput.section("Stage 1: Extracting text from PDF")
        
        # Check for checkpoint
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded extraction checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        # Extract text
        result = self.pdf_processor.extract_text(input_path)
        
        if not result:
            return None
            
        extracted_text = result.text
        
        # Log extraction info
        ConsoleOutput.info(f"Extracted {result.char_count:,} characters from {result.metadata.num_pages} pages")
        
        # Save checkpoint
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                extracted_text, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        self.memory_monitor.check("after extraction")
        return extracted_text
        
    def _stage_preprocess(self, text: str) -> str:
        """Stage 2: Preprocess text"""
        self.current_stage = "preprocess"
        ConsoleOutput.section("Stage 2: Preprocessing text")
        
        # Check for checkpoint
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded preprocessing checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        # Clean for specific use case
        if self.config.clean_for_audio and self.config.mode == ProcessingMode.PODCAST:
            text = self.text_cleaner.clean_for_audio(text)
            ConsoleOutput.info("Applied audio-specific cleaning")
            
        # General preprocessing
        text = self.text_preprocessor.preprocess_for_llm(text)
        
        ConsoleOutput.info(f"Preprocessed text: {len(text):,} characters")
        
        # Save checkpoint
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
        
        # Check for checkpoint
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded chunking checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        # Chunk text
        result = self.text_chunker.chunk_text(text)
        chunks = [chunk.text for chunk in result.chunks]
        
        ConsoleOutput.info(f"Created {len(chunks)} chunks (avg size: {result.average_chunk_size:.0f} chars)")
        
        # Save checkpoint
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                chunks, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        return chunks
        
    def _stage_process(self, chunks: List[str]) -> List[str]:
        """Stage 4: Process chunks with LLM"""
        self.current_stage = "process"
        ConsoleOutput.section("Stage 4: Processing with LLM")
        
        # Check for checkpoint
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded processing checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        # Load model if not already loaded
        if self.model_manager.model is None:
            ConsoleOutput.info("Loading model...")
            self.model_manager.load_model()
            self.memory_monitor.check("after model loading")
            
        # Get system prompt
        system_prompt = self.config.system_prompt or PREPROCESS_PROMPT
        
        # Process chunks
        processed_chunks = []
        
        with LoggingProgress(self.logger, "Processing chunks", len(chunks)) as progress:
            for i, chunk in enumerate(chunks):
                retry_count = 0
                success = False
                
                while retry_count < self.config.max_retries and not success:
                    try:
                        # Process chunk
                        result = self.model_manager.process_with_chat_template(
                            system_prompt=system_prompt,
                            user_message=chunk,
                            remove_thinking=False  # We'll filter later
                        )
                        
                        processed_chunks.append(result.raw_output)
                        success = True
                        
                        progress.update(1, f"Chunk {i+1}/{len(chunks)}")
                        ConsoleOutput.progress_bar(i+1, len(chunks), prefix="Processing")
                        
                    except Exception as e:
                        retry_count += 1
                        self.logger.warning(f"Chunk {i} failed (attempt {retry_count}): {e}")
                        
                        if retry_count < self.config.max_retries:
                            time.sleep(RETRY_DELAY_SECONDS)
                        else:
                            # Use fallback or original chunk
                            if FALLBACK_ON_ERROR:
                                processed_chunks.append(chunk)
                                self.logger.info(f"Using original chunk {i} as fallback")
                            else:
                                raise
                                
                # Memory management
                if (i + 1) % 10 == 0:
                    self.memory_monitor.check(f"after processing {i+1} chunks")
                    
        ConsoleOutput.success(f"Processed {len(processed_chunks)} chunks")
        
        # Save checkpoint
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
        
        # Check for checkpoint
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
            
        # Filter each chunk
        filtered_chunks = []
        total_removed = 0
        
        for i, chunk in enumerate(chunks):
            is_first = (i == 0)
            is_last = (i == len(chunks) - 1)
            
            filtered = self.response_filter.filter_chunk(
                chunk, i, is_first, is_last
            )
            
            filtered_chunks.append(filtered)
            total_removed += len(chunk) - len(filtered)
            
        # Merge chunks smoothly
        merged_text = self.response_filter.merge_chunks(filtered_chunks)
        
        # Re-split for consistency
        filtered_chunks = merged_text.split('\n\n')
        
        ConsoleOutput.info(f"Filtered {total_removed:,} characters of thinking/artifacts")
        
        # Save checkpoint
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                filtered_chunks, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        return filtered_chunks
        
    def _stage_format(self, chunks: List[str]) -> str:
        """Stage 6: Format output"""
        self.current_stage = "format"
        ConsoleOutput.section("Stage 6: Formatting output")
        
        # Check for checkpoint
        if self.config.enable_checkpoints:
            checkpoint = self.file_handler.load_checkpoint(
                self.checkpoint_name, self.current_stage
            )
            if checkpoint:
                self.logger.info("Loaded formatting checkpoint")
                self.stages_completed.append(self.current_stage)
                return checkpoint
                
        # Join chunks
        full_text = '\n\n'.join(chunks)
        
        # Apply formatting based on mode
        if self.config.mode == ProcessingMode.PODCAST:
            formatted = self.formatter.format_dialogue(
                full_text,
                auto_detect_speakers=True
            )
        elif self.config.mode == ProcessingMode.TECHNICAL:
            formatted = self.formatter.format_with_headers(
                full_text,
                auto_generate_toc=True
            )
        else:
            formatted = self.formatter.format_text(
                full_text,
                detect_emotions=self.config.add_emotions,
                add_structure=True
            )
            
        ConsoleOutput.info(f"Formatted output: {len(formatted):,} characters")
        
        # Save checkpoint
        if self.config.enable_checkpoints:
            self.file_handler.save_checkpoint(
                formatted, self.checkpoint_name, self.current_stage
            )
            
        self.stages_completed.append(self.current_stage)
        return formatted
        
    def _stage_save(self, text: str, input_path: Path, 
                   output_path: Optional[Path]) -> Path:
        """Stage 7: Save output"""
        self.current_stage = "save"
        ConsoleOutput.section("Stage 7: Saving output")
        
        # Generate output path if not provided
        if output_path is None:
            suffix = f"_{self.config.mode.value}"
            extension = ".md" if self.config.output_format == "markdown" else f".{self.config.output_format}"
            output_path = self.file_handler.get_output_path(
                input_path, suffix, extension
            )
            
        # Prepare metadata
        metadata = {
            "source_file": str(input_path),
            "processing_mode": self.config.mode.value,
            "model_used": self.config.model_name,
            "stages_completed": self.stages_completed
        }
        
        # Save file
        saved_path = self.file_handler.save_text(
            text, output_path, 
            format=self.config.output_format,
            metadata=metadata
        )
        
        self.stages_completed.append(self.current_stage)
        return saved_path
        
    def _gather_statistics(self, *args) -> Dict[str, Any]:
        """Gather processing statistics"""
        extracted_text, preprocessed_text, chunks, processed_chunks, formatted_text = args
        
        return {
            "original_chars": len(extracted_text),
            "preprocessed_chars": len(preprocessed_text),
            "num_chunks": len(chunks),
            "processed_chars": sum(len(c) for c in processed_chunks),
            "final_chars": len(formatted_text),
            "compression_ratio": len(formatted_text) / len(extracted_text),
            "stages_completed": len(self.stages_completed)
        }
        
    def _create_error_result(self, input_path: Path, error_message: str) -> PipelineResult:
        """Create an error result"""
        return PipelineResult(
            success=False,
            input_file=input_path,
            output_file=None,
            processing_time=0,
            stages_completed=self.stages_completed.copy(),
            error_message=error_message
        )
        
    def process_batch(self, 
                     input_files: List[Path],
                     parallel: bool = False) -> List[PipelineResult]:
        """
        Process multiple files
        
        Args:
            input_files: List of input files
            parallel: Whether to process in parallel (not implemented)
            
        Returns:
            List of processing results
        """
        ConsoleOutput.header(f"Batch Processing: {len(input_files)} files")
        
        results = []
        for i, input_file in enumerate(input_files, 1):
            ConsoleOutput.info(f"\nFile {i}/{len(input_files)}: {input_file.name}")
            
            result = self.process_file(input_file)
            results.append(result)
            
            # Unload model between files if memory is tight
            if self.config.memory_profile in ["low_vram", "cpu_only"]:
                self.model_manager.unload_model()
                
        # Generate summary
        successful = sum(1 for r in results if r.success)
        total_time = sum(r.processing_time for r in results)
        
        ConsoleOutput.header("Batch Processing Complete")
        ConsoleOutput.info(f"Successful: {successful}/{len(results)}")
        ConsoleOutput.info(f"Total time: {total_time:.2f}s")
        
        # Save processing report
        self.file_handler.save_processing_report()
        
        return results
        
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up pipeline resources")
        
        # Unload model
        if self.model_manager:
            self.model_manager.unload_model()
            
        # Clean old files
        self.file_handler.cleanup_old_files(days=7)
