#!/usr/bin/env python3
"""
LlamaNote Enhanced - Main Script with Model Browser Integration
Advanced PDF to formatted text processor with model browsing and configuration
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional
import json

from menu_system import MenuSystem
from loggerConf import ConsoleOutput, get_logger_conf
from config import (
    MEMORY_PROFILES,
    MARKDOWN_STYLES,
    DEFAULT_MODEL,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_MAX,
    QUANTIZATION_OPTIONS,
    DEFAULT_QUANTIZATION,
    MODELS
)
from config_base import OFFLOAD_DIR, FALLBACK_MODEL, DEFAULT_MODEL
from model_registry import get_registry, get_model_config, refresh_registry_from_hub
from processing_pipeline import (
    ProcessingPipeline,
    PipelineConfig,
    ProcessingMode
)
from text_processor import ChunkingStrategy
from file_handler import BatchFileManager
from model_hub import ModelHub, InteractiveModelBrowser
from hyperparameters import (
    HyperparameterConfig,
    InteractiveHyperparameterEditor,
    get_hyperparameter_help
)
from llm_handler import AdvancedModelManager, QuantizationConfig, LayerSplitConfig
from llamacpp_backend import LLAMACPP_AVAILABLE
if LLAMACPP_AVAILABLE:
    from llamacpp_backend import (
        LlamaCppBackend,
        LlamaCppConfig,
        GGUFModelManager,
        LLAMACPP_AVAILABLE
    )

# Initialize logger
logger = get_logger_conf("llamanote_main")


# Helper function to create the parser (to ensure MODELS is defined)
def parse_arguments(models_dict):
    """Creates the argument parser, using the provided models_dict for choices."""
    parser = argparse.ArgumentParser(
        description="LlamaNote Enhanced - Convert PDFs to formatted text with AI processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.pdf                    # Process single PDF with defaults
  %(prog)s                                # Start interactive menu
  %(prog)s *.pdf -m podcast                # Process all PDFs in podcast mode
  %(prog)s doc.pdf --browse-models         # Browse and select model
  %(prog)s doc.pdf --model-id "Qwen/Qwen2-4B-Instruct"  # Use specific model
  %(prog)s doc.pdf --quantization 4bit --gpu-layers 32  # 4-bit quant, 32 GPU layers
  %(prog)s doc.pdf --configure-hyperparams  # Interactive hyperparameter setup
  %(prog)s doc.pdf --use-gguf path/to/model.gguf  # Use GGUF model
        """
    )
    # Input files (now optional for menu mode)
    parser.add_argument(
        "input",
        nargs="*", # Changed from '+' to '*' to make it optional
        help="Input PDF file(s) or directory. If omitted, starts interactive menu."
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--browse-models",
        action="store_true",
        help="Browse and select model interactively (CLI mode only)" # Clarify CLI only
    )
    model_group.add_argument(
        "--model",
        choices=list(models_dict.keys()) if models_dict else [DEFAULT_MODEL, FALLBACK_MODEL], # Use passed dict
        help=f"Use predefined model (default: {DEFAULT_MODEL})"
    )
    model_group.add_argument(
        "--model-id",
        help="HuggingFace model ID (e.g., 'Qwen/Qwen2-4B-Instruct')"
    )
    model_group.add_argument(
        "--use-gguf",
        type=Path,
        help="Use GGUF/GGML model file (requires llama-cpp-python)"
    )

    parser.add_argument(
        "--refresh-registry",
        action="store_true",
        help="Refresh model registry from HuggingFace"
    )
    
    parser.add_argument(
        "--export-registry",
        type=Path,
        help="Export model registry to JSON file"
    )
    
    parser.add_argument(
        "--import-registry",
        type=Path,
        help="Import model registry from JSON file"
    )
    
    parser.add_argument(
        "--registry-stats",
        action="store_true",
        help="Show model registry statistics"
    )    
    # Quantization options
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_OPTIONS,
        default=DEFAULT_QUANTIZATION,
        help=f"Quantization method (default: {DEFAULT_QUANTIZATION})"
    )
    
    parser.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Compute dtype for quantization (default: bfloat16)"
    )
    
    # Layer splitting
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1=auto, 0=CPU only)"
    )
    
    parser.add_argument(
        "--no-layer-split",
        action="store_true",
        help="Disable automatic layer splitting"
    )
    
    parser.add_argument(
        "--max-gpu-memory",
        default="10GB",
        help="Maximum GPU memory per device (e.g., '10GB', '8GB')"
    )
    
    parser.add_argument(
        "--max-cpu-memory",
        default="30GB",
        help="Maximum CPU/RAM memory (default: 30GB)"
    )
    
    # Hyperparameters
    parser.add_argument(
        "--configure-hyperparams",
        action="store_true",
        help="Interactive hyperparameter configuration"
    )
    
    parser.add_argument(
        "--hyperparams-file",
        type=Path,
        help="Load hyperparameters from JSON file"
    )
    
    parser.add_argument(
        "--hyperparams-preset",
        choices=["creative", "balanced", "precise", "deterministic", "long_form"],
        help="Use hyperparameter preset"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        help="Sampling temperature (0.0-2.0)"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        help="Nucleus sampling top-p (0.0-1.0)"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        help="Top-k sampling"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        help="Repetition penalty (1.0=none)"
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
    
    # Processing options
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable removal of thinking tokens"
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
    
    # Utility options
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available predefined models"
    )
    
    parser.add_argument(
        "--search-models",
        help="Search HuggingFace for models"
    )
    
    parser.add_argument(
        "--list-gguf",
        action="store_true",
        help="List available GGUF models"
    )
    
    parser.add_argument(
        "--hyperparam-help",
        help="Get help for a specific hyperparameter"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def handle_model_browsing(args) -> Optional[str]:
    """Handle interactive model browsing"""
    ConsoleOutput.header("Model Browser")
    
    hub = ModelHub()
    browser = InteractiveModelBrowser(hub)
    
    # Determine task based on mode
    task_map = {
        "podcast": "text-generation",
        "technical": "text-generation",
        "narrative": "text-generation",
        "summary": "summarization"
    }
    task = task_map.get(args.mode, "text-generation")
    
    model_info = browser.browse(task=task, library="transformers")
    
    if model_info:
        ConsoleOutput.success(f"Selected: {model_info.model_id}")
        
        # Ask if user wants to download
        if not hub.is_model_cached(model_info.model_id):
            download = input("Model not cached. Download now? (y/n): ").strip().lower()
            if download == 'y':
                hub.download_model(model_info.model_id)
        
        return model_info.model_id
    
    return None


def handle_hyperparameter_configuration(args) -> HyperparameterConfig:
    """Handle hyperparameter configuration"""
    # Start with preset if specified
    if args.hyperparams_preset:
        hyperparams = HyperparameterConfig.create_preset(args.hyperparams_preset)
        ConsoleOutput.info(f"Loaded '{args.hyperparams_preset}' preset")
    # Load from file if specified
    elif args.hyperparams_file:
        hyperparams = HyperparameterConfig.load(args.hyperparams_file)
        ConsoleOutput.info(f"Loaded hyperparameters from {args.hyperparams_file}")
    else:
        hyperparams = HyperparameterConfig()
    
    # Apply CLI overrides
    if args.temperature is not None:
        hyperparams.temperature = args.temperature
    if args.top_p is not None:
        hyperparams.top_p = args.top_p
    if args.top_k is not None:
        hyperparams.top_k = args.top_k
    if args.max_tokens is not None:
        hyperparams.max_new_tokens = args.max_tokens
    if args.repetition_penalty is not None:
        hyperparams.repetition_penalty = args.repetition_penalty
    
    # Interactive configuration if requested
    if args.configure_hyperparams:
        editor = InteractiveHyperparameterEditor(hyperparams)
        hyperparams = editor.edit()
    
    # Validate
    errors = hyperparams.validate()
    if errors:
        ConsoleOutput.warning("Hyperparameter validation warnings:")
        for error in errors:
            print(f"  - {error}")
    
    return hyperparams


def create_quantization_config(args) -> QuantizationConfig:
    """Create quantization configuration from arguments"""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    
    return QuantizationConfig(
        method=args.quantization,
        compute_dtype=dtype_map.get(args.compute_dtype, torch.bfloat16),
        use_double_quant=True,
        quant_type="nf4"
    )


def create_layer_split_config(args) -> LayerSplitConfig:
    """Create layer split configuration from arguments"""
    # Parse GPU memory specification
    max_gpu_memory = {}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            max_gpu_memory[i] = args.max_gpu_memory
    
    return LayerSplitConfig(
        enabled=not args.no_layer_split,
        gpu_layers=args.gpu_layers,
        max_gpu_memory=max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=OFFLOAD_DIR
    )


def create_pipeline_with_custom_model(args, hyperparams: HyperparameterConfig) -> ProcessingPipeline:
    """Create pipeline with custom model configuration"""
    # Get model ID
    model_id = args.model_id
    
    if args.browse_models:
        model_id = handle_model_browsing(args)
        if not model_id:
            ConsoleOutput.error("No model selected")
            sys.exit(1)
    
    # Create quantization and layer split configs
    quant_config = create_quantization_config(args)
    split_config = create_layer_split_config(args)
    
    # Create model manager
    ConsoleOutput.section("Initializing Model")
    print(f"Model ID: {model_id}")
    print(f"Quantization: {quant_config.method}")
    print(f"GPU Layers: {split_config.gpu_layers}")
    print(f"Max GPU Memory: {args.max_gpu_memory}")
    
    model_manager = AdvancedModelManager(
        model_id=model_id,
        quantization_config=quant_config,
        layer_split_config=split_config,
        hyperparameters=hyperparams
    )
    
    # Load model
    ConsoleOutput.info("Loading model...")
    model_manager.load_model(trust_remote_code=True)
    
    # Create pipeline config
    mode = ProcessingMode[args.mode.upper()]
    strategy = ChunkingStrategy[args.chunk_strategy.upper()]
    
    config = PipelineConfig(
        mode=mode,
        model_name=None,  # Using custom model
        chunking_strategy=strategy,
        chunk_size=args.chunk_size,
        system_prompt=args.system_prompt,
        remove_thinking=not args.no_thinking,
        output_format=args.format
    )
    
    # Create custom pipeline with our model manager
    pipeline = ProcessingPipeline(config)
    pipeline.model_manager = model_manager  # Override with our custom manager
    
    return pipeline



def run_cli_processing(args, parser): # Added parser argument
    """Contains the original logic for processing based on CLI arguments."""
    # Check if input was actually provided (even though nargs='*', parser allows empty list)
    if not args.input:
        ConsoleOutput.error("No input files specified for CLI processing.")
        parser.print_help() # Show help message
        return 1

    # Configure hyperparameters
    hyperparams = handle_hyperparameter_configuration(args)

    # Print header
    ConsoleOutput.header("LlamaNote PDF Processor")

    # Collect input files
    batch_manager = BatchFileManager()
    input_files = batch_manager.collect_input_files(args.input, recursive=args.recursive)

    if not input_files:
        ConsoleOutput.error("No valid input files found")
        return 1

    ConsoleOutput.info(f"Found {len(input_files)} file(s) to process")

    # Determine pipeline creation based on args
    pipeline = None
    try:
        if args.model_id or args.browse_models:
            pipeline = create_pipeline_with_custom_model(args, hyperparams)
        elif args.use_gguf:
            if not LLAMACPP_AVAILABLE:
                ConsoleOutput.error("llama-cpp-python not installed.  Install with pip install llama-cpp-python")
                return 1
            pipeline = create_gguf_pipeline(args, hyperparams)
        else:
            # Use standard pipeline with predefined model
            pipeline = create_standard_pipeline(args, hyperparams)

        if pipeline is None:
            ConsoleOutput.error("Failed to initialize pipeline.")
            return 1

        # Process files
        successs = False
        try:
            if len(input_files) == 1:
                output_path = Path(args.output) if args.output else None
                success = process_single_file(pipeline, input_files[0], output_path)
            else:
                success = process_batch(pipeline, input_files)

        except KeyboardIntterrupt:
            ConsoleOutput.warning("\nProcessing interrupted by user")
            return 130
        except Exception as e:
            ConsoleOutput.error(f"Unexpected error during processing: {e}")
            logger.error("Fatal error during processing", exc_info=True)
            return 1
        finally:
            if pipeline:
                pipeline.cleanup()

        return 0 if success else 1

    except Exception as e:
        print(f"Exception: {e}")


def main():
    # Main entry point - decides between CLI and menu
    # Initialize registry first to potentially populate MODELS for parser choices
    try:
        get_registry() # Initialize registry singleton
        # Explicitly reload MODELS in config after registry init, before parsing
        from config import reload_models
        reload_models()
        # Now import the potentially updated MODELS
        from config import MODELS
    except Exception as e:
        print(f"Warning: Could not initialize model registry: {e}")
        # Define MODELS minimally if registry fails so parser doesn't crash
        MODELS = {DEFAULT_MODEL: None, FALLBACK_MODEL: None}

    # Now parse arguments after MODELS is defined
    parser = parse_arguments(MODELS)
    args = parser.parse_args()

    # Check for Utility commands
    utility_args_passed = False
    
    # Handle utility commands
    if args.list_models:
        list_models()
        return 0
    
    if args.refresh_registry:
        refresh_registry()
        return 0
    
    if args.export_registry:
        export_registry(args.export_registry)
        return 0
    
    if args.import_registry:
        import_registry(args.import_registry)
        return 0
    
    if args.registry_stats:
        show_registry_stats()
        return 0

    # Handle utility commands
    if args.list_models:
        list_models()
        return 0
    
    if args.search_models:
        search_models(args.search_models)
        return 0
    
    if args.list_gguf:
        list_gguf_models()
        return 0
    
    if args.hyperparam_help:
        print(get_hyperparameter_help(args.hyperparam_help))
        return 0
    
    if utility_args_passed:
        # If any utility command was run, exit cleanly
        print("\nUtility command finished")
        return 0

    # --- Decide between CLI processing and Interactive Menu ---
    if args.input:
        # If input files ARE provided, run CLI processing
        print("Input files provided.  Running in command-line mode...")
        return run_cli_processing(args, parser) # Pass parser for help message
    else:
        # If NO input files and NO utility commands were run, start the menu
        print("No input files specified.  Starting interactive menu..")
        try:
            menu = MenuSystem()
            menu.run() # This will block unitl the mneu exits or runs processing
            return 0
        except Exception as e:
            ConsoleOutput.error(f"Failed to start menu system: {e}")
            logger.error("Menu system failed", exc_info=True)
            return 1

    # Input files (now optional for menu mode)
    parser.add_argument(
        "input",
        nargs="*", # Changed from '+' to '*' to make it optional
        help="Input PDF file(s) or directory. If omitted, starts interactive menu."
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--browse-models",
        action="store_true",
        help="Browse and select model interactively (CLI mode only)" # Clarify CLI only
    )
    model_group.add_argument(
        "--model",
        choices=list(models_dict.keys()) if models_dict else [DEFAULT_MODEL, FALLBACK_MODEL], # Use passed dict
        help=f"Use predefined model (default: {DEFAULT_MODEL})"
    )

    # Validate input
    if not args.input:
        ConsoleOutput.error("No input files specified")
        return 1
    if args.input:
        if not args.input and not (args.list_models or args.refresh_registry or args.export_registry or args.import_registry or args.registry_stats or args.search_models or args.list_gguf or args.hyperparam_help):
                # If no input files and no utility commands, show help and exit
            parser.print_help()
            return 1
        return run_cli_processing(args)
    # If no input files and no utility args, run the interactive menu
    else:
        # Prevent running menu if a utility arg was passed without input files
        utility_args_passed = any([
                                  args.list_models, args.refresh_registry, args.export_registry,
                                  args.import_registry, args.registry_stats, args.list_models,
                                  args.search_moddels, args.list_gguf, args.hyperparam_help
            ])
        if not utility_args_passed:
            print("No input files provided.  Starting interactive menu...")
            menu = MenuSystem()
            menu.run()
            return 0
        else:
            # A utility command was likely intended but ran because no input was given
            print("Utility command finished.")
            return 0

def refresh_registry():
    # Refresh model registry from HuggingFace
    ConsoleOutput.header("Refreshing Model Registry")
    
    search_terms = input("Enter search terms (comma-separated, or press Enter for defaults): ").strip()
    if search_terms:
        terms = [t.strip() for t in search_terms.split(',')]
    else:
        terms = None
    
    limit = input("Models per search term (default=100): ").strip()
    limit = int(limit) if limit.isdigit() else 100
    
    ConsoleOutput.info("Searching HuggingFace...")
    count = refresh_registry_from_hub(search_terms=terms, limit=limit)
    
    ConsoleOutput.success(f"Added/updated {count} models in registry")

def export_registry(path: Path):
    # Export registry to JSON
    registry = get_registry()
    registry.export_to_json(path)
    ConsoleOutput.success(f"Exported registry to {path}")


def import_registry(path: Path):
    # Import registry from JSON
    registry = get_registry()
    registry.import_from_json(path)
    ConsoleOutput.success(f"Imported registry from {path}")


def show_registry_stats():
    # Show registry statistics
    registry = get_registry()
    stats = registry.get_statistics()
    
    ConsoleOutput.header("Model Registry Statistics")
    print(f"Total models: {stats['total_models']}")
    print(f"Predefined models: {stats['predefined_models']}")
    print(f"Cached models: {stats['cached_models']}")
    print(f"Thinking models: {stats['thinking_models']}")
    print(f"Total downloads: {stats['total_downloads']:,}")
    print(f"Total likes: {stats['total_likes']:,}")

def list_models():
    # List available models
    registry = get_registry()
    
    ConsoleOutput.header("Available Models")
    
    # Show predefined models first
    predefined = registry.list_models(predefined_only=True)
    if predefined:
        ConsoleOutput.section("Predefined Models")
        for model in predefined:
            print(f"\n{model.model_id}")
            print(f"  Name: {model.name}")
            print(f"  Context: {model.max_context:,} tokens")
            print(f"  Thinking: {model.supports_thinking}")
            print(f"  Cached: {model.is_cached}")
    
    # Show all models
    print("\n")
    all_models = registry.list_models(predefined_only=False)
    ConsoleOutput.section(f"Total Models in Registry: {len(all_models)}")
    
    stats = registry.get_statistics()
    print(f"  Predefined: {stats['predefined_models']}")
    print(f"  Cached: {stats['cached_models']}")
    print(f"  Thinking models: {stats['thinking_models']}")
    print(f"\nUse --search-models to find more models")
    print(f"Use --refresh-registry to update from HuggingFace")

def search_models(search_term: str):
    # Search HuggingFace for models
    from model_hub import quick_model_search
    
    ConsoleOutput.header(f"Searching for: {search_term}")
    
    results = quick_model_search(search_term, limit=20)
    
    if not results:
        ConsoleOutput.warning("No models found")
        return
    
    for i, model in enumerate(results, 1):
        print(f"\n{i}. {model.model_id}")
        print(f"   Downloads: {model.downloads:,} | Size: {model.format_size()}")
        if model.description:
            print(f"   {model.description[:100]}...")


def list_gguf_models():
    # List available GGUF models
    if not LLAMACPP_AVAILABLE:
        ConsoleOutput.error("llama-cpp-python not installed")
        return
    
    manager = GGUFModelManager()
    models = manager.list_available_models()
    
    ConsoleOutput.header("Available GGUF Models")
    
    if not models:
        ConsoleOutput.warning("No GGUF models found in cache")
        print(f"Cache directory: {manager.cache_dir}")
        return
    
    for model_path in models:
        info = manager.get_model_info(model_path)
        print(f"\n{info['name']}")
        print(f"  Size: {info['size_mb']:.1f} MB")
        if 'quantization' in info:
            print(f"  Quantization: {info['quantization']}")
        print(f"  Path: {model_path}")


def create_standard_pipeline(args, hyperparams: HyperparameterConfig) -> ProcessingPipeline:
    # Create standard pipeline with predefined model
    from processing_pipeline import PipelineConfig
    
    mode = ProcessingMode[args.mode.upper()]
    strategy = ChunkingStrategy[args.chunk_strategy.upper()]
    
    config = PipelineConfig(
        mode=mode,
        model_name=args.model or DEFAULT_MODEL,
        memory_profile="medium_vram",
        chunking_strategy=strategy,
        chunk_size=args.chunk_size,
        system_prompt=args.system_prompt,
        remove_thinking=not args.no_thinking,
        output_format=args.format
    )
    
    pipeline = ProcessingPipeline(config)
    
    # Update hyperparameters
    pipeline.model_manager.hyperparams = hyperparams
    
    return pipeline


def create_gguf_pipeline(args, hyperparams: HyperparameterConfig):
    # Create pipeline using GGUF model
    # This would need additional implementation
    # For now, raise not implemented
    raise NotImplementedError("GGUF pipeline creation not yet implemented")


def process_single_file(pipeline, input_path: Path, output_path: Optional[Path] = None):
    # Process a single file
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


def process_batch(pipeline, input_paths: List[Path]):
    # Process multiple files
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

if __name__ == "__main__":
    #Ensure registry is loaded before parsing args that might need it
    try:
        get_registry() # Initialize the registry singleton
    except Exception as e:
        print(f"Warning: Could not initialize model registry: {e}")

    sys.exit(main())

"""
    # Configure hyperparameters
    hyperparams = handle_hyperparameter_configuration(args)
    
    # Print header
    ConsoleOutput.header("LlamaNote Enhanced PDF Processor")
    
    # Collect input files
    batch_manager = BatchFileManager()
    input_files = batch_manager.collect_input_files(
        args.input,
        recursive=args.recursive
    )
    
    if not input_files:
        ConsoleOutput.error("No valid input files found")
        return 1
    
    ConsoleOutput.info(f"Found {len(input_files)} file(s) to process")
    
    # Create pipeline
    try:
        if args.model_id or args.browse_models:
            pipeline = create_pipeline_with_custom_model(args, hyperparams)
        elif args.use_gguf:
            if not LLAMACPP_AVAILABLE:
                ConsoleOutput.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
                return 1
            pipeline = create_gguf_pipeline(args, hyperparams)
        else:
            # Use standard pipeline with predefined model
            pipeline = create_standard_pipeline(args, hyperparams)
    
    except Exception as e:
        ConsoleOutput.error(f"Failed to initialize pipeline: {e}")
        logger.error("Pipeline initialization failed", exc_info=True)
        return 1
    
    # Process files
    try:
        if len(input_files) == 1:
            output_path = Path(args.output) if args.output else None
            success = process_single_file(pipeline, input_files[0], output_path)
        else:
            success = process_batch(pipeline, input_files)
    
    except KeyboardInterrupt:
        ConsoleOutput.warning("\nProcessing interrupted by user")
        return 130
    except Exception as e:
        ConsoleOutput.error(f"Unexpected error: {e}")
        logger.error("Fatal error", exc_info=True)
        return 1
    finally:
        pipeline.cleanup()
    
    return 0 if success else 1
"""
