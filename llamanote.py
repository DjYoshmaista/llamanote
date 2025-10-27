#!/usr/bin/env python3
"""
LlamaNote Enhanced - Main Script
Advanced PDF to formatted text processor with thinking model support
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import json

from logging_config import ConsoleOutput, get_logger
from config import (
    MODELS,
    MEMORY_PROFILES,
    MARKDOWN_STYLES,
    DEFAULT_MODEL,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_MAX
)
from processing_pipeline import (
    ProcessingPipeline,
    PipelineConfig,
    ProcessingMode
)
from text_processor import ChunkingStrategy
from file_handler import BatchFileManager

# Initialize logger
logger = get_logger("llamanote_main")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LlamaNote Enhanced - Convert PDFs to formatted text with AI processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                    # Process single PDF with defaults
  %(prog)s *.pdf -m podcast                # Process all PDFs in podcast mode
  %(prog)s dir/ -r -m technical            # Process directory recursively in technical mode
  %(prog)s doc.pdf -o output.md --model qwen3-4b --chunk-size 1500
  %(prog)s batch.pdf --memory-profile low_vram --no-thinking
        """
    )
    
    # Input files
    parser.add_argument(
        "input",
        nargs="+",
        help="Input PDF file(s) or directory"
    )
    
    # Output options
    parser.add_argument(
        "-o", "--output",
        help="Output file path (for single file processing)"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["markdown", "text", "json", "html"],
        default="markdown",
        help="Output format (default: markdown)"
    )
    
    # Processing mode
    parser.add_argument(
        "-m", "--mode",
        choices=["podcast", "technical", "narrative", "summary", "custom"],
        default="podcast",
        help="Processing mode (default: podcast)"
    )
    
    # Model options
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--memory-profile",
        choices=list(MEMORY_PROFILES.keys()),
        default="medium_vram",
        help="Memory optimization profile (default: medium_vram)"
    )
    
    # Chunking options
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE_DEFAULT,
        help=f"Target chunk size in characters (default: {CHUNK_SIZE_DEFAULT})"
    )
    
    parser.add_argument(
        "--chunk-strategy",
        choices=[s.value for s in ChunkingStrategy],
        default="word_boundary",
        help="Chunking strategy (default: word_boundary)"
    )
    
    # Formatting options
    parser.add_argument(
        "--markdown-style",
        choices=list(MARKDOWN_STYLES.keys()),
        default="podcast",
        help="Markdown formatting style (default: podcast)"
    )
    
    parser.add_argument(
        "--no-emotions",
        action="store_true",
        help="Disable emotional markers in output"
    )
    
    # Processing options
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable removal of thinking tokens (keep raw output)"
    )
    
    parser.add_argument(
        "--preserve-layout",
        action="store_true",
        help="Preserve PDF layout during extraction"
    )
    
    parser.add_argument(
        "--no-audio-clean",
        action="store_true",
        help="Disable audio-specific cleaning for podcast mode"
    )
    
    parser.add_argument(
        "--system-prompt",
        help="Custom system prompt for LLM processing"
    )
    
    # Batch processing
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively"
    )
    
    parser.add_argument(
        "--pattern",
        default="*.pdf",
        help="File pattern for directory processing (default: *.pdf)"
    )
    
    # Performance options
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable stage checkpointing"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for failed chunks (default: 3)"
    )
    
    # Utility options
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List memory profiles and exit"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--config",
        help="Load configuration from JSON file"
    )
    
    return parser.parse_args()


def list_models():
    """List available models"""
    ConsoleOutput.header("Available Models")
    
    for key, model in MODELS.items():
        print(f"\n{key}:")
        print(f"  Name: {model.name}")
        print(f"  ID: {model.model_id}")
        print(f"  Supports thinking: {model.supports_thinking}")
        print(f"  Max context: {model.max_context:,} tokens")
        print(f"  Optimal chunk size: {model.optimal_chunk_size}")
        
        if model.thinking_tokens:
            print(f"  Thinking tokens: {', '.join(model.thinking_tokens[:3])}...")


def list_memory_profiles():
    """List memory profiles"""
    ConsoleOutput.header("Memory Profiles")
    
    for key, profile in MEMORY_PROFILES.items():
        print(f"\n{key}:")
        print(f"  Quantization: {profile.quantization_type}")
        print(f"  Max GPU memory: {profile.max_gpu_memory}")
        print(f"  Max CPU memory: {profile.max_cpu_memory}")
        print(f"  Flash attention: {profile.use_flash_attention}")
        print(f"  Batch size: {profile.batch_size}")


def load_config_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        ConsoleOutput.error(f"Failed to load config file: {e}")
        sys.exit(1)


def validate_arguments(args) -> bool:
    """Validate command line arguments"""
    # Check chunk size
    if not CHUNK_SIZE_MIN <= args.chunk_size <= CHUNK_SIZE_MAX:
        ConsoleOutput.error(
            f"Chunk size must be between {CHUNK_SIZE_MIN} and {CHUNK_SIZE_MAX}"
        )
        return False
        
    # Check input paths
    for input_path in args.input:
        path = Path(input_path)
        if not path.exists():
            ConsoleOutput.error(f"Input path does not exist: {input_path}")
            return False
            
    # Check single file output
    if args.output and len(args.input) > 1:
        ConsoleOutput.warning(
            "Output path specified but multiple inputs provided. "
            "Output path will be ignored for batch processing."
        )
        args.output = None
        
    return True


def create_pipeline_config(args) -> PipelineConfig:
    """Create pipeline configuration from arguments"""
    # Load base config from file if provided
    if args.config:
        config_data = load_config_file(args.config)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None and key in config_data:
                config_data[key] = value
    else:
        config_data = vars(args)
        
    # Convert string to enum
    mode = ProcessingMode[config_data.get("mode", "podcast").upper()]
    strategy = ChunkingStrategy[config_data.get("chunk_strategy", "word_boundary").upper()]
    
    return PipelineConfig(
        mode=mode,
        model_name=config_data.get("model", DEFAULT_MODEL),
        memory_profile=config_data.get("memory_profile", "medium_vram"),
        chunking_strategy=strategy,
        chunk_size=config_data.get("chunk_size", CHUNK_SIZE_DEFAULT),
        markdown_style=config_data.get("markdown_style", "podcast"),
        system_prompt=config_data.get("system_prompt"),
        remove_thinking=not config_data.get("no_thinking", False),
        preserve_layout=config_data.get("preserve_layout", False),
        clean_for_audio=not config_data.get("no_audio_clean", False),
        add_emotions=not config_data.get("no_emotions", False),
        enable_checkpoints=not config_data.get("no_checkpoints", False),
        max_retries=config_data.get("max_retries", 3),
        output_format=config_data.get("format", "markdown")
    )


def process_single_file(pipeline: ProcessingPipeline, 
                       input_path: Path,
                       output_path: Optional[Path] = None):
    """Process a single file"""
    result = pipeline.process_file(input_path, output_path)
    
    if result.success:
        ConsoleOutput.success(f"✓ Successfully processed: {input_path.name}")
        ConsoleOutput.info(f"  Output: {result.output_file}")
        ConsoleOutput.info(f"  Time: {result.processing_time:.2f}s")
        
        if result.statistics:
            ConsoleOutput.info(f"  Compression: {result.statistics['compression_ratio']:.1%}")
    else:
        ConsoleOutput.error(f"✗ Failed to process: {input_path.name}")
        ConsoleOutput.error(f"  Error: {result.error_message}")
        
    return result.success


def process_batch(pipeline: ProcessingPipeline,
                 input_paths: List[Path]):
    """Process multiple files"""
    results = pipeline.process_batch(input_paths)
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    total_time = sum(r.processing_time for r in results)
    
    ConsoleOutput.header("Processing Summary")
    ConsoleOutput.info(f"Total files: {len(results)}")
    ConsoleOutput.success(f"Successful: {successful}")
    
    if failed > 0:
        ConsoleOutput.error(f"Failed: {failed}")
        
    ConsoleOutput.info(f"Total time: {total_time:.2f}s")
    ConsoleOutput.info(f"Average time: {total_time/len(results):.2f}s per file")
    
    return successful == len(results)


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Handle utility commands
    if args.list_models:
        list_models()
        return 0
        
    if args.list_profiles:
        list_memory_profiles()
        return 0
        
    # Validate arguments
    if not validate_arguments(args):
        return 1
        
    # Print header
    ConsoleOutput.header("LlamaNote Enhanced PDF Processor")
    
    # Collect input files
    batch_manager = BatchFileManager()
    input_files = batch_manager.collect_input_files(
        args.input,
        recursive=args.recursive,
        pattern=args.pattern
    )
    
    if not input_files:
        ConsoleOutput.error("No valid input files found")
        return 1
        
    ConsoleOutput.info(f"Found {len(input_files)} file(s) to process")
    
    # Create pipeline configuration
    config = create_pipeline_config(args)
    
    # Display configuration
    if args.verbose:
        ConsoleOutput.section("Configuration")
        print(f"  Mode: {config.mode.value}")
        print(f"  Model: {config.model_name}")
        print(f"  Memory profile: {config.memory_profile}")
        print(f"  Chunk size: {config.chunk_size}")
        print(f"  Chunk strategy: {config.chunking_strategy.value}")
        print(f"  Output format: {config.output_format}")
        print(f"  Remove thinking: {config.remove_thinking}")
        print()
        
    # Create pipeline
    try:
        pipeline = ProcessingPipeline(config)
    except Exception as e:
        ConsoleOutput.error(f"Failed to initialize pipeline: {e}")
        return 1
        
    # Process files
    try:
        if len(input_files) == 1:
            # Single file processing
            output_path = Path(args.output) if args.output else None
            success = process_single_file(pipeline, input_files[0], output_path)
        else:
            # Batch processing
            success = process_batch(pipeline, input_files)
            
    except KeyboardInterrupt:
        ConsoleOutput.warning("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        ConsoleOutput.error(f"Unexpected error: {e}")
        logger.error("Fatal error", exc_info=True)
        return 1
    finally:
        # Cleanup
        pipeline.cleanup()
        
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
