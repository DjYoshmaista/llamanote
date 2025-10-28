# llamanote/core/stages.py
"""
Pipeline Stage Definitions for LlamaNote Enhanced

This module contains the logic for each individual step in the processing pipeline.
The ProcessingPipeline class will call these functions in order, passing
the data payload and managing state (checkpoints, error handling).
"""

import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from ..utils.logger import get_logger_conf, LoggingProgress, ConsoleOutput
from ..utils.decorators import retry, log_execution_time
from ..config.settings import FALLBACK_ON_ERROR, RETRY_DELAY_SECONDS, MAX_RETRIES, INCLUDE_METADATA, DEFAULT_SYSTEM_PROMPT
from ..core.types import (
    ProcessingMode, PipelineConfig, ExtractionResult, 
    ChunkingResult, TextChunk, GenerationResult, FilterResult, PDFMetadata
)
from ..core.errors import PipelineError, MissingDataError, PDFExtractionError, GenerationError

# Import components required by stages
from ..processing.pdf_extractor import PDFProcessor
from ..processing.text_preprocessor import TextPreprocessor, PDFTextCleaner
from ..processing.text_chunker import TextChunker
from ..processing.response_filter import ChunkedResponseFilter
from ..formatting.base_formatter import BaseFormatter, get_formatter
from ..io.file_handler import FileHandler
from ..models.backends.base import LLMBackend, AudioBackend

logger = get_logger_conf(__name__)

# --- Stage 1: Extraction ---

@log_execution_time(logger_name=__name__)
def run_extraction_stage(
    input_path: Path,
    pdf_processor: PDFProcessor
) -> ExtractionResult:
    """
    Handles file reading and text extraction.
    Supports PDF, TXT, and MD files.
    """
    logger.info(f"Stage 'extract': Processing file {input_path.name}")
    
    if input_path.suffix.lower() == '.pdf':
        try:
            result = pdf_processor.extract_text(input_path)
            if not result:
                raise PDFExtractionError("PDF processor returned no result.", str(input_path))
            
            ConsoleOutput.info(f"Extracted {result.char_count:,} chars from {result.metadata.num_pages} pages using {result.extraction_method}.")
            for warning in result.warnings:
                ConsoleOutput.warning(warning)
            return result
        
        except Exception as e:
            logger.error(f"Extraction failed for PDF {input_path.name}: {e}", exc_info=True)
            raise PipelineError(f"Extraction failed: {e}", stage="extract") from e

    elif input_path.suffix.lower() in ['.txt', '.md']:
        logger.info("Input is text file, reading content.")
        try:
            text_content = input_path.read_text(encoding='utf-8')
            # Create a minimal ExtractionResult
            meta = PDFMetadata(
                file_path=input_path,
                num_pages=1, # N/A, but set to 1
                file_size_mb=input_path.stat().st_size / (1024 * 1024),
                title=input_path.stem
            )
            return ExtractionResult(
                text=text_content,
                metadata=meta,
                page_texts=[text_content],
                extraction_method="text_read",
                warnings=[],
                char_count=len(text_content),
                word_count=len(text_content.split())
            )
        except Exception as e:
            logger.error(f"Failed to read input text file {input_path.name}: {e}", exc_info=True)
            raise FileProcessingError(f"Failed to read input file: {e}", str(input_path)) from e
    else:
        raise FileProcessingError(f"Unsupported file type: {input_path.suffix}", str(input_path))


# --- Stage 2: Preprocessing ---

@log_execution_time(logger_name=__name__)
def run_preprocess_stage(
    text: str, 
    config: PipelineConfig,
    text_cleaner: PDFTextCleaner,
    text_preprocessor: TextPreprocessor
) -> str:
    """Handles text cleaning and normalization."""
    logger.info("Stage 'preprocess': Cleaning extracted text.")
    
    cleaned_text = text
    if config.clean_for_audio and config.mode == ProcessingMode.PODCAST:
        cleaned_text = text_cleaner.clean_for_audio(cleaned_text)
        logger.debug("Applied audio-specific cleaning.")
    
    preprocessed_text = text_preprocessor.preprocess_for_llm(cleaned_text)
    
    ConsoleOutput.info(f"Preprocessing complete. Text length: {len(preprocessed_text):,} chars")
    return preprocessed_text


# --- Stage 3: Chunking ---

@log_execution_time(logger_name=__name__)
def run_chunking_stage(
    text: str,
    text_chunker: TextChunker
) -> ChunkingResult:
    """Handles splitting text into chunks."""
    logger.info(f"Stage 'chunk': Chunking text using {text_chunker.default_strategy.value} strategy.")
    
    result = text_chunker.chunk_text(text) # Uses chunker's configured strategy
    
    if not result.chunks:
         logger.warning("Chunking resulted in 0 chunks. This may cause issues in the process stage.")
         ConsoleOutput.warning("Text chunking produced no output. Subsequent steps may fail.")
    else:
        ConsoleOutput.info(f"Created {len(result.chunks)} chunks (avg size: {result.average_chunk_size:.0f} chars)")
        
    return result

# --- Stage 4: Processing (LLM) ---

@log_execution_time(logger_name=__name__)
def run_processing_stage(
    chunks: List[str],
    llm_backend: LLMBackend,
    config: PipelineConfig
) -> List[str]:
    """Handles sending chunks to the LLM and collecting responses."""
    if not llm_backend:
        raise PipelineError("LLM Backend not provided to processing stage.", stage="process")
        
    if not llm_backend.model_handle: # Check if model/client is loaded
        logger.info(f"Loading LLM backend {llm_backend.model_identifier} for processing...")
        if not llm_backend.load():
            raise ModelLoadError(f"Failed to load LLM backend: {llm_backend.model_identifier}")
        logger.info(f"LLM backend {llm_backend.model_identifier} loaded.")
            
    system_prompt = config.system_prompt or (
        PREPROCESS_PROMPT_PODCAST if config.mode == ProcessingMode.PODCAST else DEFAULT_SYSTEM_PROMPT
    )

    processed_chunks: List[str] = []
    
    ConsoleOutput.info(f"Processing {len(chunks)} chunks with {llm_backend.model_identifier}...")

    # We wrap the core generation call in a retry mechanism
    @retry(max_attempts=config.max_retries, delay_seconds=RETRY_DELAY_SECONDS, 
           exceptions_to_catch=(GenerationError, TimeoutError, IOError), # Add specific errors
           logger_name=__name__)
    def _process_chunk_with_retry(chunk: str) -> GenerationResult:
        return llm_backend.process_chat(
            system_prompt=system_prompt,
            user_message=chunk,
            hyperparams=config.hyperparameters,
            # Pass remove_thinking=False, as we do this in a separate stage
            remove_thinking=False 
        )

    with LoggingProgress(logger, f"Processing {len(chunks)} chunks", len(chunks)) as progress:
        for i, chunk in enumerate(chunks):
            result_text = None
            try:
                result = _process_chunk_with_retry(chunk)
                
                if result.error_message:
                    # Handle non-exception errors (e.g., API blocks)
                    raise GenerationError(result.error_message, llm_backend.model_specifier)
                    
                result_text = result.raw_output
                progress.update(1, f"Chunk {i+1}/{len(chunks)} complete ({result.output_tokens} tokens)")

            except Exception as e:
                logger.error(f"Chunk {i+1} failed permanently after {config.max_retries} attempts: {e}")
                if config.fallback_on_error:
                    ConsoleOutput.warning(f"Failed to process chunk {i+1}. Using original text as fallback.")
                    result_text = f"[[LLM_PROCESSING_FAILED: {e}]]\n\n{chunk}" # Add error and original text
                else:
                    logger.critical(f"Pipeline failed at chunk {i+1}. Fallback disabled.")
                    raise PipelineError(f"Failed to process chunk {i+1}: {e}", stage="process") from e
            
            processed_chunks.append(result_text)

            # Optional memory cleanup for local models
            if llm_backend.provider_identifier.startswith("local") and (i + 1) % 5 == 0:
                if torch and torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()

    ConsoleOutput.success(f"Finished processing {len(processed_chunks)} chunks.")
    return processed_chunks

# --- Stage 5: Filtering ---

@log_execution_time(logger_name=__name__)
def run_filtering_stage(
    processed_chunks: List[str],
    response_filter: ChunkedResponseFilter,
    config: PipelineConfig
) -> str:
    """Handles filtering artifacts and merging chunks."""
    logger.info("Stage 'filter': Filtering and merging processed chunks.")
    
    if not config.remove_thinking:
        logger.info("Filtering skipped (remove_thinking=False). Joining chunks.")
        # Still need to join the chunks
        merged_text = '\n\n'.join(chunk for chunk in processed_chunks if chunk)
        # Apply basic whitespace normalization
        merged_text = re.sub(r'\n{3,}', '\n\n', merged_text).strip()
        return merged_text

    # Use the ChunkedResponseFilter's combined filter-and-merge method
    merged_text = response_filter.filter_and_merge(
        chunks=processed_chunks,
        remove_thinking=True,
        remove_acknowledgments=True # Only first chunk ack is removed inside
    )
    
    ConsoleOutput.info(f"Filtering complete. Final text length: {len(merged_text):,} chars")
    return merged_text


# --- Stage 6: Formatting ---

@log_execution_time(logger_name=__name__)
def run_formatting_stage(
    filtered_text: str,
    formatter: BaseFormatter,
    config: PipelineConfig
) -> str:
    """Applies final presentation formatting (Markdown, etc.)."""
    logger.info(f"Stage 'format': Applying '{config.markdown_style or config.mode.value}' formatting.")
    
    formatted_text = formatter.format(
        filtered_text,
        add_emotions=config.add_emotions,
        auto_detect_speakers=(config.mode == ProcessingMode.PODCAST)
    )
    
    ConsoleOutput.info(f"Formatting complete. Final length: {len(formatted_text):,} chars")
    return formatted_text


# --- Stage 7: Saving ---

@log_execution_time(logger_name=__name__)
def run_save_stage(
    text_to_save: str,
    input_path: Path,
    output_path_base: Optional[Path], # This is the user-specified base name/path
    metadata: Optional[PDFMetadata],
    file_handler: FileHandler, # The pipeline's file handler instance
    config: PipelineConfig
) -> Path:
    """Handles saving the final text output."""
    logger.info(f"Stage 'save': Saving output as {config.output_format}.")

    # Generate the final output path
    # If output_path_base is set (e.g., via -o), use it
    # Otherwise, generate a name based on the input file
    if output_path_base:
        # Check if it was a directory (from -o <dir>) or a specific file (from -o <file>)
        if output_path_base.suffix: # It's a file path
            output_path = file_handler.path_generator.handle_existing_file(output_path_base)
            # Ensure suffix matches format
            output_path = output_path.with_suffix(f".{config.output_format}")
        else: # It was a directory path
            # Re-init generator for this specific directory? No, FileHandler's dir was updated.
             output_path = file_handler.get_output_path(
                 input_path=input_path,
                 suffix=f"_{config.mode.value}",
                 extension=f".{config.output_format}",
                 base_path_override=None # Already handled by setting handler.output_dir
             )
    else:
        # Default behavior: save in handler's output_dir with generated name
        output_path = file_handler.get_output_path(
            input_path=input_path,
            suffix=f"_{config.mode.value}",
            extension=f".{config.output_format}"
        )


    # Prepare metadata for the output file header
    metadata_to_save = {
        "source_file": str(input_path.resolve()),
        "processing_mode": config.mode.value,
        "model_used": config.model_name_for_metadata,
        "stages_completed": config.stages
    }
    
    if metadata: # Add PDF-specific metadata if it exists
        try:
            if metadata.title: metadata_to_save["original_title"] = metadata.title
            if metadata.author: metadata_to_save["original_author"] = metadata.author
            metadata_to_save["original_pages"] = metadata.num_pages
        except Exception as e:
            logger.warning(f"Could not parse PDF metadata for saving: {e}")

    # Save file using FileHandler
    saved_path = file_handler.save_text(
        text_to_save,
        output_path,
        format=config.output_format,
        metadata=metadata_to_save if config.include_metadata else None
    )

    if not saved_path:
        raise FileProcessingError(f"Failed to save output file to {output_path}", str(output_path))

    ConsoleOutput.success(f"File saved: {saved_path.name}")
    return saved_path


# --- Stage 8: Audio Generation (New stage) ---

@log_execution_time(logger_name=__name__)
def run_audio_stage(
    text_file_path: Path,
    audio_backend: AudioBackend,
    config: PipelineConfig
) -> Optional[AudioResult]:
    """Handles generating audio from a text file."""
    if not audio_backend:
        logger.warning("Audio generation stage called, but no audio backend is configured.")
        return None
        
    if not text_file_path or not text_file_path.exists():
         logger.error(f"Cannot generate audio: source text file not found at {text_file_path}")
         return None

    logger.info(f"Stage 'audio': Generating audio for {text_file_path.name}")
    ConsoleOutput.info(f"Generating audio for {text_file_path.name}...")

    try:
        text_content = text_file_path.read_text(encoding='utf-8')
        
        # Clean text for TTS (remove metadata header, speaker tags, etc.)
        # This is a basic cleanup; TextPreprocessor has a more robust one
        text_content = re.sub(r'^---.*?---', '', text_content, flags=re.DOTALL | re.MULTILINE).strip()
        if config.mode == ProcessingMode.PODCAST:
             # Remove speaker tags like **[Speaker Host]:**
             text_content = re.sub(r'^\*\*\[Speaker.*?\]:\*\*\n*', '', text_content, flags=re.MULTILINE)
        
        # Remove other common markdown
        text_content = re.sub(r'(\*\*|\*|`|#+\s|🎉|🤔|😐|😄|😲|❓|⚠️|📝|💡|📖|💬|🎬|🖼️)', '', text_content)
        # Consolidate whitespace
        text_content = re.sub(r'\n{2,}', '\n', text_content).strip()

        if not text_content:
            logger.warning(f"Skipping audio generation for {text_file_path.name}: No content after cleaning.")
            return None

        # Define output path
        audio_output_path = text_file_path.with_suffix(f".{config.audio_config.output_format}")
        
        # Ensure audio model is loaded (for local models)
        if not audio_backend.model_handle:
            if not audio_backend.load():
                raise ModelLoadError(f"Failed to load audio backend {audio_backend.model_identifier}", audio_backend.model_specifier)
        
        audio_result = audio_backend.generate_audio(
            text=text_content,
            output_path=audio_output_path,
            chunk_text=(audio_backend.provider_identifier == 'local_audio')
        )
        
        if audio_result:
            ConsoleOutput.success(f"Audio saved: {audio_result.audio_path.name} ({audio_result.duration_formatted})")
            return audio_result
        else:
            raise GenerationError(f"Audio generation failed for unknown reason.", audio_backend.model_specifier)

    except Exception as e:
        logger.error(f"Audio generation failed for {text_file_path.name}: {e}", exc_info=True)
        ConsoleOutput.error(f"Audio generation failed for {text_file_path.name}: {e}")
        return None
