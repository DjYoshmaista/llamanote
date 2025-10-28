#!/usr/bin/env python3
"""
LlamaNote Enhanced - Main Script with Model Browser Integration
Advanced PDF to formatted text processor with model browsing and configuration
Supports both local and cloud LLM backends.
"""

import sys
import argparse
import torch
from pathlib import Path
from typing import List, Optional, Dict
import json
import time # Added for backend init timing

# --- Core LlamaNote Modules ---
from menu_system import MenuSystem, SUPPORTED_CLOUD_PROVIDERS # Import provider list
from loggerConf import ConsoleOutput, get_logger_conf
from config_manager import ConfigManager # Added for API key loading
from config import (
    MEMORY_PROFILES,
    MARKDOWN_STYLES,
    # DEFAULT_MODEL is used as fallback, MODELS dict still used for legacy CLI arg choices
    DEFAULT_MODEL,
    MODELS,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_MAX,
    QUANTIZATION_OPTIONS,
    DEFAULT_QUANTIZATION,
    PREPROCESS_PROMPT # Import if needed for pipeline config default
)
from config_base import * 
from model_registry import get_registry, get_model_config, refresh_registry_from_hub
from processing_pipeline import (
    ProcessingPipeline,
    PipelineConfig,
    ProcessingMode,
    PipelineResult # Import PipelineResult for type hinting
)
from text_processor import ChunkingStrategy
from file_handler import BatchFileManager, FileHandler # Import FileHandler for direct use if needed
from model_hub import ModelHub, InteractiveModelBrowser, ModelInfo as HFModelInfo
from hyperparameters import (
    HyperparameterConfig,
    InteractiveHyperparameterEditor,
    get_hyperparameter_help
)
# --- Backend Abstraction ---
from llm_handler import (
    get_llm_backend, # Import the factory function
    LLMBackend, # Import base class for type hinting
    QuantizationConfig, # Keep for local backend config
    LayerSplitConfig, # Keep for local backend config
    LLMBackend, # Keep as alias or specific use if needed
    GenerationResult # Import for type hinting
)
# --- GGUF Backend (Optional) ---
from llamacpp_backend import LLAMACPP_AVAILABLE
if LLAMACPP_AVAILABLE:
    from llamacpp_backend import (
        LlamaCppBackend,
        LlamaCppConfig,
        GGUFModelManager,
    )
else:
    # Define dummy classes if llama-cpp-python is not installed
    # to prevent NameErrors in type hints or conditional checks
    LlamaCppBackend = None
    LlamaCppConfig = None
    GGUFModelManager = None


# Initialize logger
logger = get_logger_conf("llamanote_main")


# Helper function to create the parser (to ensure MODELS is defined)
def parse_arguments(models_dict):
    """Creates the argument parser, using the provided models_dict for choices."""
    parser = argparse.ArgumentParser(
        description="LlamaNote Enhanced - Convert PDFs to formatted text with AI processing (local & cloud)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s document.pdf                    # Process single PDF with local default ({DEFAULT_MODEL})
  %(prog)s                                # Start interactive menu
  %(prog)s *.pdf -m podcast                # Process all PDFs in podcast mode (local default)
  %(prog)s doc.pdf --browse-models         # Browse and select local model
  %(prog)s doc.pdf --model-id "Qwen/Qwen2-4B-Instruct"  # Use specific local HF model
  %(prog)s doc.pdf --quantization 4bit --gpu-layers 32  # Local: 4-bit quant, 32 GPU layers
  %(prog)s doc.pdf --configure-hyperparams  # Interactive hyperparameter setup
  %(prog)s doc.pdf --use-gguf path/to/model.gguf  # Use local GGUF model
  %(prog)s doc.pdf --cloud-provider openai --cloud-model gpt-4o # Use OpenAI GPT-4o
  %(prog)s doc.pdf --cloud-provider anthropic --cloud-model claude-3-opus-20240229 # Use Claude 3 Opus
        """
    )
    # Input files (now optional for menu mode)
    parser.add_argument(
        "input",
        nargs="*", # Changed from '+' to '*' to make it optional
        help="Input PDF file(s) or directory. If omitted, starts interactive menu."
    )

    # --- Local Model Selection ---
    local_model_group = parser.add_argument_group('Local Model Options')
    local_model_exclusive_group = local_model_group.add_mutually_exclusive_group()
    local_model_exclusive_group.add_argument(
        "--browse-models",
        action="store_true",
        help="Browse and select local HuggingFace model interactively (CLI mode only)"
    )
    local_model_exclusive_group.add_argument(
        "--model", # Represents the short key for predefined local models
        choices=list(models_dict.keys()) if models_dict else [DEFAULT_MODEL, FALLBACK_MODEL],
        help=f"Use predefined local model key (default: {DEFAULT_MODEL})"
    )
    local_model_exclusive_group.add_argument(
        "--model-id", # Represents the full HuggingFace ID for local models
        help="HuggingFace model ID for local processing (e.g., 'Qwen/Qwen2-4B-Instruct')"
    )
    local_model_exclusive_group.add_argument(
        "--use-gguf",
        type=Path,
        help="Use local GGUF/GGML model file (requires llama-cpp-python)"
    )

    # --- Cloud Model Selection ---
    cloud_group = parser.add_argument_group('Cloud Provider Options (CLI)')
    cloud_group.add_argument(
        "--cloud-provider",
        choices=SUPPORTED_CLOUD_PROVIDERS, # Use the list defined in menu_system
        help="Specify cloud provider to use (requires API key configured via menu or api_keys.json)"
    )
    cloud_group.add_argument(
        "--cloud-model",
        help="Specify the model name/ID for the chosen cloud provider (e.g., 'gpt-4o', 'claude-3-opus-20240229')"
    )

    # --- Model Registry ---
    registry_group = parser.add_argument_group('Model Registry Options')
    registry_group.add_argument(
        "--refresh-registry",
        action="store_true",
        help="Refresh model registry from HuggingFace"
    )
    registry_group.add_argument(
        "--export-registry",
        type=Path,
        help="Export model registry to JSON file"
    )
    registry_group.add_argument(
        "--import-registry",
        type=Path,
        help="Import model registry from JSON file"
    )
    registry_group.add_argument(
        "--registry-stats",
        action="store_true",
        help="Show model registry statistics"
    )

    # --- Local Model Configuration (Quantization, Layer Splitting) ---
    local_config_group = parser.add_argument_group('Local Model Configuration')
    local_config_group.add_argument(
        "--quantization",
        choices=QUANTIZATION_OPTIONS,
        default=DEFAULT_QUANTIZATION,
        help=f"Quantization method for local models (default: {DEFAULT_QUANTIZATION})"
    )
    local_config_group.add_argument(
        "--compute-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Compute dtype for local quantization (default: bfloat16)"
    )
    local_config_group.add_argument(
        "--gpu-layers",
        type=int,
        default=DEFAULT_GPU_LAYERS,
        help="Number of layers to offload to GPU for local models (-1=auto, 0=CPU only)"
    )
    local_config_group.add_argument(
        "--no-layer-split",
        action="store_true",
        help="Disable automatic layer splitting for local models"
    )
    local_config_group.add_argument(
        "--max-gpu-memory",
        default="10GB",
        help="Maximum GPU memory per device for local models (e.g., '10GB', '8GB')"
    )
    local_config_group.add_argument(
        "--max-cpu-memory",
        default="30GB",
        help="Maximum CPU/RAM memory for local models (default: 30GB)"
    )

    # --- Hyperparameters ---
    hyperparam_group = parser.add_argument_group('Hyperparameter Options')
    hyperparam_group.add_argument(
        "--configure-hyperparams",
        action="store_true",
        help="Interactive hyperparameter configuration"
    )
    hyperparam_group.add_argument(
        "--hyperparams-file",
        type=Path,
        help="Load hyperparameters from JSON file"
    )
    hyperparam_group.add_argument(
        "--hyperparams-preset",
        choices=["creative", "balanced", "precise", "deterministic", "long_form"],
        help="Use hyperparameter preset"
    )
    # Direct hyperparameter overrides
    hyperparam_group.add_argument("--temperature", type=float, help="Sampling temperature (0.0-2.0)")
    hyperparam_group.add_argument("--top-p", type=float, help="Nucleus sampling top-p (0.0-1.0)")
    hyperparam_group.add_argument("--top-k", type=int, help="Top-k sampling")
    hyperparam_group.add_argument("--max-tokens", type=int, help="Maximum new tokens to generate")
    hyperparam_group.add_argument("--repetition-penalty", type=float, help="Repetition penalty (1.0=none)")

    # --- Output Options ---
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "-o", "--output",
        help="Output file path (for single file processing, overrides default naming)"
    )
    output_group.add_argument(
        "-f", "--format",
        choices=OUTPUT_FORMAT_OPTIONS,
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format (default: {DEFAULT_OUTPUT_FORMAT})"
    )

    # --- Processing Mode ---
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument(
        "-m", "--mode",
        choices=[mode.value for mode in ProcessingMode], # Use enum values
        default=ProcessingMode.PODCAST.value,
        help=f"Processing mode (default: {ProcessingMode.PODCAST.value})"
    )
    processing_group.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE_DEFAULT,
        help=f"Target chunk size in characters ({CHUNK_SIZE_MIN}-{CHUNK_SIZE_MAX}, default: {CHUNK_SIZE_DEFAULT})"
    )
    processing_group.add_argument(
        "--chunk-strategy",
        choices=[s.value for s in ChunkingStrategy],
        default=ChunkingStrategy.WORD_BOUNDARY.value,
        help=f"Chunking strategy (default: {ChunkingStrategy.WORD_BOUNDARY.value})"
    )
    processing_group.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable removal of thinking tokens (if supported by model)"
    )
    processing_group.add_argument(
        "--system-prompt",
        help="Custom system prompt for LLM processing"
    )
    processing_group.add_argument( # Added argument for audio generation via CLI
        "--generate-audio",
        action="store_true",
        help="Generate audio file after text processing (uses configured audio model)"
    )

    # --- Batch Processing ---
    batch_group = parser.add_argument_group('Batch Processing Options')
    batch_group.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively"
    )

    # --- Utility Options ---
    utility_group = parser.add_argument_group('Utility Options')
    utility_group.add_argument(
        "--list-models",
        action="store_true",
        help="List available models in the registry"
    )
    utility_group.add_argument(
        "--search-models",
        help="Search HuggingFace Hub for models matching a term"
    )
    utility_group.add_argument(
        "--list-gguf",
        action="store_true",
        help="List available GGUF models in cache (if llama-cpp-python installed)"
    )
    utility_group.add_argument(
        "--hyperparam-help",
        help="Get help for a specific hyperparameter (e.g., --hyperparam-help temperature)"
    )
    utility_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (sets logging level to DEBUG)"
    )

    return parser

# --- Helper Functions (Quantization, Layer Split, Hyperparams) ---

def handle_hyperparameter_configuration(args) -> HyperparameterConfig:
    """Handle hyperparameter configuration from CLI args or interactive session."""
    if args.hyperparams_preset:
        hyperparams = HyperparameterConfig.create_preset(args.hyperparams_preset)
        ConsoleOutput.info(f"Loaded '{args.hyperparams_preset}' preset")
    elif args.hyperparams_file:
        try:
            hyperparams = HyperparameterConfig.load(args.hyperparams_file)
            ConsoleOutput.info(f"Loaded hyperparameters from {args.hyperparams_file}")
        except Exception as e:
            ConsoleOutput.warning(f"Could not load hyperparams file: {e}. Using defaults.")
            hyperparams = HyperparameterConfig()
    else:
        hyperparams = HyperparameterConfig() # Start with defaults

    # Apply direct CLI overrides
    if args.temperature is not None: hyperparams.temperature = args.temperature
    if args.top_p is not None: hyperparams.top_p = args.top_p
    if args.top_k is not None: hyperparams.top_k = args.top_k
    if args.max_tokens is not None: hyperparams.max_new_tokens = args.max_tokens
    if args.repetition_penalty is not None: hyperparams.repetition_penalty = args.repetition_penalty

    if args.configure_hyperparams:
        editor = InteractiveHyperparameterEditor(hyperparams)
        hyperparams = editor.edit()

    errors = hyperparams.validate()
    if errors:
        ConsoleOutput.warning("Hyperparameter validation warnings:")
        for error in errors: print(f"  - {error}")

    return hyperparams

def create_quantization_config(args) -> QuantizationConfig:
    """Create quantization configuration from arguments (for local backend)."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    compute_dtype = dtype_map.get(args.compute_dtype, torch.bfloat16)

    # Need to pass all relevant bnb args
    return QuantizationConfig(
        method=args.quantization,
        compute_dtype=compute_dtype,
        use_double_quant=True if args.quantization == "4bit" else False, # Only relevant for 4bit
        quant_type="nf4", # Common default for 4bit
        bnb_4bit_use_double_quant=True if args.quantization == "4bit" else False # Explicitly pass
        # Add other BitsAndBytes args if needed based on quantization method
    )

def create_layer_split_config(args) -> LayerSplitConfig:
    """Create layer split configuration from arguments (for local backend)."""
    max_gpu_memory = {}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        # Apply the same limit to all GPUs specified via CLI
        # More complex multi-GPU setups might need per-device args
        for i in range(num_gpus):
            max_gpu_memory[i] = args.max_gpu_memory
    else: # Ensure it's an empty dict if no GPUs
         max_gpu_memory = {}

    return LayerSplitConfig(
        enabled=not args.no_layer_split and args.gpu_layers != 0, # Disable if explicitly told or CPU only
        gpu_layers=args.gpu_layers,
        max_gpu_memory=max_gpu_memory,
        max_cpu_memory=args.max_cpu_memory,
        offload_folder=OFFLOAD_DIR # Use base config path
        # offload_state_dict might need an arg if needed
    )

# --- Utility Functions ---

def list_models():
    """List available models from the registry."""
    registry = get_registry()
    ConsoleOutput.header("Available Models in Registry")

    predefined = registry.list_models(predefined_only=True, sort_by="name")
    if predefined:
        ConsoleOutput.section("Predefined Models (use key like '--model qwen3-4b')")
        for model in predefined:
            key_info = f" (Key: {model.short_key})" if model.short_key else ""
            print(f"- {model.name}{key_info}")
            print(f"  ID: {model.model_id}")
            print(f"  Context: {model.max_context:,} | Thinking: {model.supports_thinking}")
            print(f"  Cached: {model.is_cached}")

    all_models = registry.list_models(sort_by="name")
    other_models = [m for m in all_models if not m.is_predefined]
    if other_models:
        ConsoleOutput.section("Other Models in Registry (use ID like '--model-id Org/ModelName')")
        for model in other_models[:20]: # Show a limited number
            print(f"- {model.model_id} (Downloads: {model.downloads:,})")
            print(f"  Cached: {model.is_cached}")
        if len(other_models) > 20:
            print(f"  ... and {len(other_models) - 20} more.")

    print("\nUse --search-models TERM to find more on HuggingFace Hub.")
    print("Use --refresh-registry to update registry from Hub.")

def search_models(search_term: str):
    """Search HuggingFace Hub for models."""
    hub = ModelHub()
    ConsoleOutput.header(f"Searching HuggingFace Hub for: '{search_term}'")
    results = hub.search_models(search_term=search_term, limit=20)
    if not results:
        ConsoleOutput.warning("No models found matching the search term.")
        return
    ConsoleOutput.info(f"Found {len(results)} potential models:")
    for i, model in enumerate(results, 1):
        print(f"{i:2}. {model.model_id}")
        print(f"    Downloads: {model.downloads:,} | Likes: {model.likes} | Size: {model.format_size()}")
        if model.description:
            print(f"    Desc: {model.description[:80]}...")
    print("\nUse the full Model ID (e.g., 'Org/ModelName') with --model-id to use one.")
    print("Run --refresh-registry to potentially add these to your local registry.")


def list_gguf_models():
    """List available GGUF models from cache."""
    if not LLAMACPP_AVAILABLE:
        ConsoleOutput.error("llama-cpp-python is not installed. Cannot list GGUF models.")
        print("Install with: pip install llama-cpp-python")
        return

    manager = GGUFModelManager()
    models = manager.list_available_models()
    ConsoleOutput.header("Available GGUF Models (in cache)")
    if not models:
        ConsoleOutput.warning("No GGUF models found in cache.")
        print(f"Cache directory searched: {manager.cache_dir}")
        return

    for model_path in models:
        info = manager.get_model_info(model_path)
        print(f"- {info['name']}")
        print(f"  Size: {info['size_mb']:.1f} MB | Format: {info['format']}")
        if 'quantization' in info: print(f"  Quantization: {info['quantization']}")
        print(f"  Path: {model_path}")
    print("\nUse --use-gguf PATH/TO/MODEL.gguf to use one.")

def refresh_registry():
    """Refresh model registry from HuggingFace Hub."""
    ConsoleOutput.header("Refreshing Model Registry from HuggingFace Hub")
    terms_input = input("Enter search terms (comma-separated, Enter for defaults 'instruct,chat'): ").strip()
    terms = [t.strip() for t in terms_input.split(',')] if terms_input else ["instruct", "chat"]
    limit_input = input("Max models per term (default 100): ").strip()
    limit = int(limit_input) if limit_input.isdigit() else 100

    ConsoleOutput.info(f"Searching for terms: {terms} (limit {limit} each)...")
    try:
        count = refresh_registry_from_hub(search_terms=terms, limit=limit)
        ConsoleOutput.success(f"Registry updated. Added/updated {count} models.")
    except Exception as e:
        ConsoleOutput.error(f"Failed to refresh registry: {e}")
        logger.error("Registry refresh failed", exc_info=True)

def export_registry(path: Path):
    """Export registry to JSON file."""
    try:
        registry = get_registry()
        registry.export_to_json(path)
        ConsoleOutput.success(f"Exported model registry to {path}")
    except Exception as e:
        ConsoleOutput.error(f"Failed to export registry: {e}")

def import_registry(path: Path):
    """Import registry from JSON file."""
    try:
        if not path.exists():
            ConsoleOutput.error(f"Import file not found: {path}")
            return
        registry = get_registry()
        registry.import_from_json(path)
        ConsoleOutput.success(f"Imported model registry from {path}")
    except Exception as e:
        ConsoleOutput.error(f"Failed to import registry: {e}")

def show_registry_stats():
    """Show statistics about the model registry."""
    try:
        registry = get_registry()
        stats = registry.get_statistics()
        ConsoleOutput.header("Model Registry Statistics")
        print(f"Total models in registry: {stats['total_models']}")
        print(f"Predefined models: {stats['predefined_models']}")
        print(f"Cached local models: {stats['cached_models']}")
        print(f"Models supporting 'thinking': {stats['thinking_models']}")
        print(f"Total recorded downloads (approx): {stats['total_downloads']:,}")
        print(f"Total recorded likes (approx): {stats['total_likes']:,}")
    except Exception as e:
        ConsoleOutput.error(f"Failed to get registry stats: {e}")

# --- CLI Processing Logic ---

def run_cli_processing(args, parser):
    """Handles processing when run directly from the command line with arguments."""
    if not args.input:
        ConsoleOutput.error("No input files specified for CLI processing.")
        parser.print_help()
        return 1

    ConsoleOutput.header("LlamaNote PDF Processor - CLI Mode")

    # 1. Load API Keys (needed for backend selection)
    config_manager = ConfigManager()
    api_keys = config_manager.load_cloud_keys()

    # 2. Determine Backend Provider and Model Specifier
    provider = "local" # Default
    model_specifier = DEFAULT_MODEL # Default key
    local_backend_type = "transformers" # Default local type

    if args.cloud_provider:
        if not args.cloud_model:
            ConsoleOutput.error("--cloud-model is required when --cloud-provider is used.")
            return 1
        if args.cloud_provider not in api_keys and args.cloud_provider != "huggingface_token": # HF Token might be optional depending on backend
             # Recheck just before use, but warn early
             ConsoleOutput.warning(f"API key for '{args.cloud_provider}' not found in configuration. Execution might fail.")
             # Proceed anyway, let backend handle missing key error if critical
        provider = args.cloud_provider
        model_specifier = args.cloud_model
        ConsoleOutput.info(f"Using Cloud Backend: {provider} with model {model_specifier}")
    elif args.use_gguf:
        if not LLAMACPP_AVAILABLE:
            ConsoleOutput.error("Cannot use --use-gguf. llama-cpp-python is not installed.")
            return 1
        provider = "local"
        model_specifier = str(args.use_gguf.resolve()) # Use the absolute path
        local_backend_type = "llamacpp"
        ConsoleOutput.info(f"Using Local GGUF Backend: {model_specifier}")
    elif args.model_id:
        provider = "local"
        model_specifier = args.model_id
        local_backend_type = "transformers"
        ConsoleOutput.info(f"Using Local HF Backend (by ID): {model_specifier}")
    elif args.model: # Predefined local model key
        provider = "local"
        local_backend_type = "transformers"
        # Find the actual HF model ID from the short key
        registry = get_registry()
        entry = registry.get_by_key(args.model)
        if entry:
            model_specifier = entry.model_id
            ConsoleOutput.info(f"Using Local HF Backend (by key '{args.model}'): {model_specifier}")
        else:
            ConsoleOutput.warning(f"Predefined key '{args.model}' not found in registry.")
            # Fallback logic (could try DEFAULT_MODEL key, then FALLBACK_MODEL key)
            entry = registry.get_by_key(DEFAULT_MODEL)
            if entry:
                 ConsoleOutput.warning(f"Falling back to default key '{DEFAULT_MODEL}': {entry.model_id}")
                 model_specifier = entry.model_id
            else: # Ultimate fallback
                 ConsoleOutput.error(f"Default model key '{DEFAULT_MODEL}' also not found. Cannot determine model.")
                 return 1
    else:
         # No specific model chosen, use default local model
         provider = "local"
         local_backend_type = "transformers"
         registry = get_registry()
         entry = registry.get_by_key(DEFAULT_MODEL)
         if entry:
             model_specifier = entry.model_id
             ConsoleOutput.info(f"Using Default Local HF Backend ('{DEFAULT_MODEL}'): {model_specifier}")
         else:
             ConsoleOutput.error(f"Default model key '{DEFAULT_MODEL}' not found. Cannot determine model.")
             return 1


    # 3. Configure Hyperparameters
    hyperparams = handle_hyperparameter_configuration(args)

    # 4. Initialize Backend
    text_backend: Optional[LLMBackend] = None
    try:
        ConsoleOutput.section("Initializing Text Backend")
        # Prepare kwargs specific to local backends
        local_kwargs = {}
        if provider == "local":
            if local_backend_type == "transformers":
                local_kwargs['quantization_config'] = create_quantization_config(args)
                local_kwargs['layer_split_config'] = create_layer_split_config(args)
                # You might need memory_config here too depending on LocalTransformerBackend init
                # local_kwargs['memory_config'] = MEMORY_PROFILES.get(...)
            elif local_backend_type == "llamacpp":
                 # GGUF backend might need different config (e.g., n_gpu_layers directly)
                 gguf_config = LlamaCppConfig(
                     model_path=Path(model_specifier),
                     n_ctx=hyperparams.max_length or 2048, # Use max_length if set, else default
                     n_gpu_layers=args.gpu_layers if args.gpu_layers >= 0 else -1 # Pass GPU layers
                     # Add other LlamaCppConfig settings if needed
                 )
                 # Instantiate LlamaCppBackend directly here instead of using the factory
                 # (Or enhance the factory to handle GGUF paths)
                 text_backend = LlamaCppBackend(gguf_config)
                 # Loading happens via text_backend.load_model() below

        # Use factory for non-GGUF cases, or instantiate GGUF backend directly
        if local_backend_type != "llamacpp":
             text_backend = get_llm_backend(
                 provider=provider,
                 model_specifier=model_specifier,
                 api_keys=api_keys,
                 hyperparameters=hyperparams,
                 **local_kwargs
             )

        # Load the model (local) or initialize client (cloud/gguf)
        load_success = text_backend.load_model() # GGUF backend's load_model takes args like verbose
        if local_backend_type == "llamacpp" and not load_success:
             raise RuntimeError("Failed to load GGUF model.")

        ConsoleOutput.success("Text backend initialized.")

    except Exception as e:
        ConsoleOutput.error(f"Failed to initialize text backend: {e}")
        logger.error("Text backend init failed (CLI)", exc_info=True)
        return 1


    # 5. Collect Input Files
    batch_manager = BatchFileManager()
    input_files = batch_manager.collect_input_files(args.input, recursive=args.recursive)
    if not input_files:
        ConsoleOutput.error("No valid input files found.")
        if text_backend: text_backend.unload_model()
        return 1
    ConsoleOutput.info(f"Found {len(input_files)} file(s) to process.")


    # 6. Initialize Pipeline
    pipeline: Optional[ProcessingPipeline] = None
    try:
        pipeline_config = PipelineConfig(
            mode=ProcessingMode(args.mode), # Convert string back to enum
            model_provider=provider,
            model_specifier=model_specifier,
            # Pass model_name for metadata/logging purposes
            model_name=f"{provider}:{model_specifier}",
            chunking_strategy=ChunkingStrategy(args.chunk_strategy), # Convert string back to enum
            chunk_size=args.chunk_size,
            system_prompt=args.system_prompt,
            remove_thinking=not args.no_thinking,
            output_format=args.format,
            hyperparameters=hyperparams # Pass the configured hyperparameters
            # memory_profile is less relevant now as backend handles specifics
        )
        pipeline = ProcessingPipeline(pipeline_config)
        pipeline.llm_backend = text_backend # Inject the initialized backend

        # 7. Process Files
        ConsoleOutput.section("Starting Processing")
        overall_success = False
        if len(input_files) == 1:
            # Determine output path for single file
            # If -o is specified, use it directly. Otherwise, let FileHandler decide.
            output_p = Path(args.output) if args.output else None
            # The process_file method now needs to handle None output_path
            result = pipeline.process_file(input_files[0], output_path=output_p)
            overall_success = result.success
        else:
            if args.output:
                 ConsoleOutput.warning("-o/--output is ignored during batch processing. Files saved to default directory.")
            results = pipeline.process_batch(input_files)
            overall_success = all(r.success for r in results)

        # 8. Audio Generation (Conditional)
        if args.generate_audio:
             ConsoleOutput.warning("--generate-audio via CLI is not fully implemented yet.")
             # TODO: Add logic similar to _execute_pipeline in menu_system
             # - Determine audio provider/model (need CLI args or use defaults)
             # - Initialize audio backend (factory function needed)
             # - Find saved text files from 'results' or 'result' object
             # - Loop and call audio_backend.generate_audio

        return 0 if overall_success else 1

    except KeyboardInterrupt:
        ConsoleOutput.warning("\nProcessing interrupted by user.")
        return 130 # Standard exit code for Ctrl+C
    except Exception as e:
        ConsoleOutput.error(f"An unexpected error occurred during CLI processing: {e}")
        logger.error("CLI processing failed", exc_info=True)
        # Attempt to save error context if possible
        if text_backend and hasattr(text_backend, 'logger'):
             text_backend.logger.error(f"CLI processing failed: {e}", exc_info=True, save_context=True)
        return 1
    finally:
        # Cleanup backend and pipeline resources
        if text_backend:
            text_backend.unload_model()
        if pipeline:
            pipeline.cleanup_file_handler() # Save report if implemented


# --- Main Entry Point ---

def main():
    """Main execution function: parses args, runs utilities, CLI, or Menu."""
    try:
        get_registry() # Initialize registry singleton first
        from config import reload_models # Function to update MODELS dict
        reload_models() # Update MODELS based on registry
        from config import MODELS # Import the potentially updated dict
    except Exception as e:
        print(f"Warning: Could not initialize/load model registry: {e}")
        MODELS = {DEFAULT_MODEL: None, FALLBACK_MODEL: None} # Minimal fallback

    parser = parse_arguments(MODELS)
    args = parser.parse_args()

    # Set verbosity level
    if args.verbose:
         logging.getLogger("llamanote").setLevel(logging.DEBUG)
         # Optionally set other loggers (like transformers) if needed
         # logging.getLogger("transformers").setLevel(logging.INFO)
         logger.debug("Verbose mode enabled.")


    # --- Handle Utility Commands ---
    utility_commands = {
        'list_models': list_models,
        'search_models': lambda: search_models(args.search_models) if args.search_models else parser.error("--search-models requires an argument"),
        'list_gguf': list_gguf_models,
        'hyperparam_help': lambda: print(get_hyperparameter_help(args.hyperparam_help)) if args.hyperparam_help else parser.error("--hyperparam-help requires an argument"),
        'refresh_registry': refresh_registry,
        'export_registry': lambda: export_registry(args.export_registry) if args.export_registry else parser.error("--export-registry requires a path"),
        'import_registry': lambda: import_registry(args.import_registry) if args.import_registry else parser.error("--import-registry requires a path"),
        'registry_stats': show_registry_stats,
    }

    utility_arg_passed = False
    for arg_name, func in utility_commands.items():
        if getattr(args, arg_name, None):
            utility_arg_passed = True
            func() # Execute the utility function
            break # Assume only one utility command per run

    if utility_arg_passed:
        # If any utility command was run, exit cleanly
        print("\nUtility command finished.")
        return 0

    # --- Decide between CLI processing and Interactive Menu ---
    if args.input:
        # If input files ARE provided, run CLI processing
        print("Input files provided. Running in command-line mode...")
        return run_cli_processing(args, parser)
    else:
        # If NO input files and NO utility commands were run, start the menu
        print("No input files specified. Starting interactive menu...")
        try:
            # Ensure registry is loaded again here in case it failed initially
            try:
                 get_registry()
                 from config import reload_models
                 reload_models()
            except Exception as e:
                 ConsoleOutput.warning(f"Registry reload failed before menu: {e}")

            menu = MenuSystem()
            menu.run() # This blocks until the menu exits or runs processing
            return 0
        except Exception as e:
            ConsoleOutput.error(f"Failed to start menu system: {e}")
            logger.error("Menu system failed", exc_info=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
