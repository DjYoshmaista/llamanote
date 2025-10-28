#!/usr/bin/env python3
"""
LlamaNote Enhanced - Main Script with Model Browser Integration
Advanced PDF to formatted text processor with model browsing and configuration
Supports both local and cloud LLM backends.
"""

import sys
import argparse
import logging # Import standard logging for level setting
from pathlib import Path
from typing import List, Optional, Dict
import json
import time # Added for backend init timing

# --- Core LlamaNote Modules ---
from menu_system import MenuSystem, SUPPORTED_CLOUD_PROVIDERS # Import provider list
from loggerConf import ConsoleOutput, get_logger_conf
from config_manager import ConfigManager # Added for API key loading
# Import shared types from pipeline_types
from pipeline_types import (
    ProcessingMode,
    PipelineConfig,
    PipelineResult,
    QuantizationConfig,
    LayerSplitConfig,
    GenerationResult
)
from config import ( # Keep specific config values if needed
    MEMORY_PROFILES,
    MARKDOWN_STYLES,
    DEFAULT_MODEL, # Used as fallback key
    # MODELS dict is less critical now, registry is primary
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MIN,
    CHUNK_SIZE_MAX,
    QUANTIZATION_OPTIONS,
    DEFAULT_QUANTIZATION,
    PREPROCESS_PROMPT
)
from config_base import * # Import base paths and constants
from model_registry import get_registry, get_model_config, refresh_registry_from_hub, ModelEntry
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
    LLMBackend # Import base class for type hinting
    # Other specific types imported from pipeline_types
)
# --- GGUF Backend (Optional) ---
# Import check and types directly from llamacpp_backend
from llamacpp_backend import LLAMACPP_AVAILABLE
if LLAMACPP_AVAILABLE:
    from llamacpp_backend import (
        LlamaCppBackend, # Import the actual class
        LlamaCppConfig,
        GGUFModelManager,
    )
else:
    # Define dummy classes if llama-cpp-python is not installed
    LlamaCppBackend = None
    LlamaCppConfig = None
    GGUFModelManager = None

# Import torch check from menu_system helper or directly
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# Initialize logger
logger = get_logger_conf("llamanote_main")

# Helper function to create the parser
def parse_arguments():
    """Creates the argument parser."""
    # Try to get predefined model keys for choices, handle potential errors
    try:
        registry = get_registry()
        predefined_models = registry.list_models(predefined_only=True)
        model_keys = [entry.short_key for entry in predefined_models if entry.short_key]
        default_key = DEFAULT_MODEL
    except Exception as e:
        logger.warning(f"Could not load predefined models from registry for CLI choices: {e}. Using basic defaults.")
        model_keys = [DEFAULT_MODEL, FALLBACK_MODEL] # Fallback keys
        default_key = DEFAULT_MODEL

    parser = argparse.ArgumentParser(
        description="LlamaNote Enhanced - Convert PDFs/Text to formatted text with AI processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s document.pdf                    # Process single PDF with local default ({default_key})
  %(prog)s                                # Start interactive menu
  %(prog)s *.pdf -m podcast                # Process all PDFs in podcast mode (local default)
  %(prog)s doc.pdf --browse-models         # Browse and select local model
  %(prog)s doc.pdf --model-id "Qwen/Qwen2-1.5B-Instruct"  # Use specific local HF model
  %(prog)s doc.pdf --quantization 4bit --gpu-layers -1   # Local: 4-bit quant, auto GPU layers (HF/GGUF)
  %(prog)s doc.pdf --configure-hyperparams  # Interactive hyperparameter setup
  %(prog)s doc.pdf --use-gguf path/to/model.gguf  # Use local GGUF model
  %(prog)s doc.pdf --cloud-provider openai --cloud-model gpt-4o # Use OpenAI GPT-4o
  %(prog)s doc.pdf --cloud-provider anthropic --cloud-model claude-3-5-sonnet-20240620 # Use Claude 3.5 Sonnet
        """
    )
    # Input files (optional for menu mode)
    parser.add_argument(
        "input",
        nargs="*",
        help="Input PDF/Text file(s) or directory. If omitted, starts interactive menu."
    )

    # --- Local Model Selection ---
    local_model_group = parser.add_argument_group('Local Model Options (Select ONE)')
    local_model_exclusive_group = local_model_group.add_mutually_exclusive_group()
    local_model_exclusive_group.add_argument(
        "--browse-models",
        action="store_true",
        help="Browse and select local HuggingFace model interactively before processing (CLI mode only)"
    )
    local_model_exclusive_group.add_argument(
        "--model", # Represents the short key for predefined local models
        choices=model_keys,
        help=f"Use predefined local model key (e.g., '{default_key}'). See --list-models."
    )
    local_model_exclusive_group.add_argument(
        "--model-id", # Represents the full HuggingFace ID for local models
        help="HuggingFace model ID for local Transformers processing (e.g., 'Org/ModelName')"
    )
    local_model_exclusive_group.add_argument(
        "--use-gguf",
        type=Path,
        metavar="PATH_TO_GGUF",
        help="Path to local GGUF/GGML model file (requires llama-cpp-python)"
    )

    # --- Cloud Model Selection ---
    cloud_group = parser.add_argument_group('Cloud Provider Options (CLI)')
    cloud_group.add_argument(
        "--cloud-provider",
        choices=SUPPORTED_CLOUD_PROVIDERS,
        help="Specify cloud provider (requires API key configured via menu or api_keys.json)"
    )
    cloud_group.add_argument(
        "--cloud-model",
        help="Model name/ID for the chosen cloud provider (e.g., 'gpt-4o', 'claude-3-opus-20240229')"
    )

    # --- Model Registry ---
    registry_group = parser.add_argument_group('Model Registry Options (Run Separately)')
    registry_group.add_argument(
        "--list-models",
        action="store_true",
        help="List available models in the local registry and exit"
    )
    registry_group.add_argument(
        "--refresh-registry",
        action="store_true",
        help="Refresh model registry from HuggingFace Hub based on common terms and exit"
    )
    registry_group.add_argument(
        "--search-models",
        metavar="TERM",
        help="Search HuggingFace Hub for models matching TERM and exit"
    )
    registry_group.add_argument(
        "--list-gguf",
        action="store_true",
        help="List cached GGUF models and exit (if llama-cpp-python installed)"
    )
    registry_group.add_argument(
        "--export-registry", type=Path, metavar="FILE.json", help="Export model registry to JSON file and exit")
    registry_group.add_argument(
        "--import-registry", type=Path, metavar="FILE.json", help="Import model registry from JSON file and exit")
    registry_group.add_argument(
        "--registry-stats", action="store_true", help="Show model registry statistics and exit")


    # --- Local Model Configuration (Quantization, Layer Splitting) ---
    local_config_group = parser.add_argument_group('Local Model Hardware Configuration')
    local_config_group.add_argument(
        "--quantization",
        choices=QUANTIZATION_OPTIONS,
        default=DEFAULT_QUANTIZATION,
        help=f"Quantization for local Transformers models (default: {DEFAULT_QUANTIZATION})"
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
        help="Num GPU layers for local GGUF models (-1=auto/all, 0=CPU)"
    )
    local_config_group.add_argument(
        "--no-layer-split",
        action="store_true",
        help="Disable automatic layer splitting for local Transformers models"
    )
    local_config_group.add_argument(
        "--max-gpu-memory",
        default="10GB",
        metavar="SIZE",
        help="Max GPU memory per device for local Transformers (e.g., '8GB', '10000MB')"
    )
    local_config_group.add_argument(
        "--max-cpu-memory",
        default="30GB",
        metavar="SIZE",
        help="Max CPU/RAM for local Transformers (e.g., '30GB')"
    )

    # --- Hyperparameters ---
    hyperparam_group = parser.add_argument_group('Hyperparameter Options')
    hyperparam_group.add_argument(
        "--configure-hyperparams",
        action="store_true",
        help="Interactive hyperparameter setup before processing (CLI mode only)"
    )
    hyperparam_group.add_argument(
        "--hyperparams-file", type=Path, metavar="FILE.json", help="Load hyperparameters from JSON file")
    hyperparam_group.add_argument(
        "--hyperparams-preset",
        choices=["creative", "balanced", "precise", "deterministic", "long_form"],
        help="Use a predefined hyperparameter preset"
    )
    # Direct overrides
    hyperparam_group.add_argument("--temperature", type=float, help="Override sampling temperature")
    hyperparam_group.add_argument("--top-p", type=float, help="Override nucleus sampling top-p")
    hyperparam_group.add_argument("--top-k", type=int, help="Override top-k sampling")
    hyperparam_group.add_argument("--max-tokens", type=int, help="Override maximum new tokens")
    hyperparam_group.add_argument("--repetition-penalty", type=float, help="Override repetition penalty")
    hyperparam_group.add_argument(
        "--hyperparam-help", metavar="PARAM_NAME", help="Show detailed help for a specific hyperparameter and exit")


    # --- Output Options ---
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        "-o", "--output", help="Output file path (for single input file only)")
    output_group.add_argument(
        "-d", "--output-dir", type=Path, help=f"Output directory (default: {OUTPUT_DIR})")
    output_group.add_argument(
        "-f", "--format", choices=OUTPUT_FORMAT_OPTIONS, default=DEFAULT_OUTPUT_FORMAT, help="Output format for text files")

    # --- Processing Mode ---
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument(
        "-m", "--mode", choices=[mode.value for mode in ProcessingMode], default=ProcessingMode.PODCAST.value, help="Processing mode")
    processing_group.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE_DEFAULT, help=f"Target text chunk size ({CHUNK_SIZE_MIN}-{CHUNK_SIZE_MAX})")
    processing_group.add_argument(
        "--chunk-strategy", choices=[s.value for s in ChunkingStrategy], default=ChunkingStrategy.WORD_BOUNDARY.value, help="Chunking strategy")
    processing_group.add_argument(
        "--no-thinking", action="store_true", help="Disable removal of <thinking> tokens")
    processing_group.add_argument(
        "--system-prompt", help="Custom system prompt for LLM processing (overrides mode default)")
    processing_group.add_argument(
        "--generate-audio", action="store_true", help="Generate audio after text processing (requires 'save' stage)")
    # TODO: Add CLI args for audio model selection/config if needed


    # --- Batch Processing ---
    batch_group = parser.add_argument_group('Batch Processing Options')
    batch_group.add_argument(
        "-r", "--recursive", action="store_true", help="Process directories recursively")

    # --- Utility ---
    utility_group = parser.add_argument_group('Utility Options')
    utility_group.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level)"
    )

    return parser

# --- Helper Functions (Quantization, Layer Split, Hyperparams - Copied from menu_system) ---
# These need to be defined *before* run_cli_processing uses them.

def create_quantization_config(args: argparse.Namespace) -> QuantizationConfig:
    """Create quantization configuration from arguments (or state)."""
    dtype_map = {
        "bfloat16": torch.bfloat16 if TORCH_AVAILABLE else None,
        "float16": torch.float16 if TORCH_AVAILABLE else None,
        "float32": torch.float32 if TORCH_AVAILABLE else None,
    }
    compute_dtype_str = getattr(args, 'compute_dtype', 'bfloat16')
    compute_dtype = dtype_map.get(compute_dtype_str, dtype_map.get('bfloat16')) # Default to bfloat16 if torch available
    quant_method = getattr(args, 'quantization', DEFAULT_QUANTIZATION)

    if compute_dtype is None and quant_method != "none":
        logger.warning(f"Torch not available, cannot set compute_dtype. Quantization might fail.")

    return QuantizationConfig(
        method=quant_method,
        compute_dtype=compute_dtype,
        use_double_quant=(quant_method == "4bit"),
        quant_type="nf4",
        bnb_4bit_use_double_quant=(quant_method == "4bit")
    )

def create_layer_split_config(args: argparse.Namespace) -> LayerSplitConfig:
    """Create layer split configuration from arguments (or state)."""
    max_gpu_mem_str = getattr(args, 'max_gpu_memory', "10GB")
    max_cpu_mem_str = getattr(args, 'max_cpu_memory', "30GB")
    gpu_layers_count = getattr(args, 'gpu_layers', DEFAULT_GPU_LAYERS)
    no_split = getattr(args, 'no_layer_split', not ENABLE_LAYER_SPLITTING)

    max_gpu_memory = {}
    cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
    if cuda_available:
        try:
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                max_gpu_memory[i] = max_gpu_mem_str
        except Exception as e:
             logger.warning(f"Could not get CUDA device count: {e}. Assuming 0 GPUs.")
             max_gpu_memory = {}
    else:
        max_gpu_memory = {}

    split_enabled = not no_split and gpu_layers_count != 0 and cuda_available

    return LayerSplitConfig(
        enabled=split_enabled,
        gpu_layers=gpu_layers_count,
        max_gpu_memory=max_gpu_memory,
        max_cpu_memory=max_cpu_mem_str,
        offload_folder=Path(OFFLOAD_DIR)
    )

def handle_hyperparameter_configuration(args) -> HyperparameterConfig:
    """Handle hyperparameter configuration from CLI args or interactive session."""
    hyperparams = HyperparameterConfig() # Start with defaults

    if args.hyperparams_preset:
        hyperparams = HyperparameterConfig.create_preset(args.hyperparams_preset)
        ConsoleOutput.info(f"Loaded '{args.hyperparams_preset}' hyperparameter preset")
    elif args.hyperparams_file:
        try:
            hyperparams = HyperparameterConfig.load(args.hyperparams_file)
            ConsoleOutput.info(f"Loaded hyperparameters from {args.hyperparams_file}")
        except FileNotFoundError:
            ConsoleOutput.error(f"Hyperparams file not found: {args.hyperparams_file}. Using defaults.")
        except Exception as e:
            ConsoleOutput.warning(f"Could not load hyperparams file ({e}). Using defaults.")

    # Apply direct CLI overrides AFTER loading presets/files
    overrides = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_tokens,
        "repetition_penalty": args.repetition_penalty
    }
    for key, value in overrides.items():
        if value is not None:
             setattr(hyperparams, key, value)
             ConsoleOutput.info(f"Overriding hyperparameter: {key} = {value}")


    if args.configure_hyperparams:
        editor = InteractiveHyperparameterEditor(hyperparams)
        hyperparams = editor.edit()

    errors = hyperparams.validate()
    if errors:
        ConsoleOutput.warning("Hyperparameter validation warnings:")
        for error in errors: print(f"  - {error}")

    return hyperparams

class ModelRegistryUtils:
    """Utility functions for model registry operations."""
    
    @staticmethod
    def list_models():
        """List available models from the registry."""
        registry = get_registry()
        ConsoleOutput.header("Available Models in Registry")
        
        predefined = registry.list_models(predefined_only=True, sort_by="name")
        if predefined:
            ConsoleOutput.section("Predefined Models (use key with --model)")
            for model in predefined:
                key_info = f" (Key: {model.short_key})" if model.short_key else ""
                print(f"- {model.name}{key_info}")
                print(f"  ID: {model.model_id}")
                print(f"  Context: {model.max_context:,} | Thinking: {'Yes' if model.supports_thinking else 'No'}")
        else:
            print("No predefined models found in registry.")
        
        all_models = registry.list_models(sort_by="name")
        other_models = [m for m in all_models if not m.is_predefined]
        if other_models:
            ConsoleOutput.section("Other Models in Registry (use ID with --model-id)")
            max_show = 20
            for model in other_models[:max_show]:
                dl_info = f" (Downloads: {model.downloads:,})" if model.downloads > 0 else ""
                print(f"- {model.model_id}{dl_info}")
            if len(other_models) > max_show:
                print(f"  ... and {len(other_models) - max_show} more.")
        
        print("\nUse --search-models TERM to find more on HuggingFace Hub.")
        print("Use --refresh-registry to update registry from Hub.")
    
    @staticmethod
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
            dl_info = f"Downloads: {model.downloads:,}" if model.downloads else "Downloads: N/A"
            size_info = f"Size: {model.format_size()}" if model.model_size_mb else "Size: N/A"
            print(f"{i:2}. {model.model_id}")
            print(f"    {dl_info} | Likes: {model.likes} | {size_info}")
            if model.description:
                print(f"    Desc: {model.description[:80]}...")
        print("\nUse the full Model ID (e.g., 'Org/ModelName') with --model-id to select.")
    
    @staticmethod
    def list_gguf_models():
        """List available GGUF models from cache."""
        if not LLAMACPP_AVAILABLE:
            ConsoleOutput.error("llama-cpp-python is not installed. Cannot list GGUF models.")
            print("Install with: pip install llama-cpp-python")
            return
        
        try:
            manager = GGUFModelManager(cache_dir=CACHE_DIR / "gguf_models")
            models = manager.list_available_models()
            ConsoleOutput.header("Available GGUF Models (in cache)")
            if not models:
                ConsoleOutput.warning("No GGUF models found in cache.")
                print(f"Cache directory searched: {manager.cache_dir}")
                return
            
            for model_path in models:
                info = manager.get_model_info(model_path)
                print(f"- {info.get('name', model_path.name)}")
                print(f"  Size: {info.get('size_mb', 0):.1f} MB | Format: {info.get('format', 'N/A')}")
                if 'quantization' in info:
                    print(f"  Quantization: {info['quantization']}")
                print(f"  Path: {model_path}")
            print("\nUse --use-gguf PATH/TO/MODEL.gguf to select.")
        except Exception as e:
            ConsoleOutput.error(f"Error listing GGUF models: {e}")
    
    @staticmethod
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

list_models = ModelRegistryUtils.list_models
search_models = ModelRegistryUtils.search_models
list_gguf_models = ModelRegistryUtils.list_gguf_models
refresh_registry = ModelRegistryUtils.refresh_registry

def export_registry(path: Path):
    """Export registry to JSON file."""
    try:
        registry = get_registry()
        registry.export_to_json(path)
        ConsoleOutput.success(f"Exported model registry to {path}")
    except Exception as e:
        ConsoleOutput.error(f"Failed to export registry: {e}")
        logger.error("Registry export failed", exc_info=True)


def import_registry(path: Path):
    """Import registry from JSON file."""
    try:
        if not path.is_file():
            ConsoleOutput.error(f"Import file not found or is not a file: {path}")
            return
        registry = get_registry()
        registry.import_from_json(path)
        ConsoleOutput.success(f"Imported model registry from {path}")
    except json.JSONDecodeError:
        ConsoleOutput.error(f"Failed to import registry: Invalid JSON file at {path}")
    except Exception as e:
        ConsoleOutput.error(f"Failed to import registry: {e}")
        logger.error("Registry import failed", exc_info=True)


def show_registry_stats():
    """Show statistics about the model registry."""
    try:
        registry = get_registry()
        stats = registry.get_statistics()
        ConsoleOutput.header("Model Registry Statistics")
        print(f"Total models in registry : {stats.get('total_models', 0)}")
        print(f"Predefined models        : {stats.get('predefined_models', 0)}")
        print(f"Cached local models      : {stats.get('cached_models', 0)} (Note: cache status may need update)")
        print(f"Models supporting 'think': {stats.get('thinking_models', 0)}")
        print(f"Total downloads (Hub)    : {stats.get('total_downloads', 0):,}")
        print(f"Total likes (Hub)        : {stats.get('total_likes', 0):,}")
    except Exception as e:
        ConsoleOutput.error(f"Failed to get registry stats: {e}")
        logger.error("Registry stats failed", exc_info=True)


# --- CLI Processing Logic ---

def run_cli_processing(args, parser):
    """Handles processing when run directly from the command line with arguments."""
    if not args.input:
        ConsoleOutput.error("No input files specified for CLI processing.")
        parser.print_help()
        return 1

    ConsoleOutput.header("LlamaNote PDF Processor - CLI Mode")

    # 1. Load API Keys
    config_manager = ConfigManager()
    api_keys = config_manager.load_cloud_keys()

    # 2. Determine Backend Provider and Model Specifier
    provider = "local" # Default
    model_specifier = DEFAULT_MODEL # Default key, will resolve to ID/path
    local_backend_type = "transformers" # Default local type
    selected_model_entry: Optional[ModelEntry] = None # To store ModelEntry for local models

    registry = get_registry() # Get registry instance

    if args.cloud_provider:
        if not args.cloud_model:
            ConsoleOutput.error("--cloud-model is required when --cloud-provider is used.")
            return 1
        provider = args.cloud_provider
        model_specifier = args.cloud_model
        ConsoleOutput.info(f"Using Cloud Backend: {provider} / {model_specifier}")
        if provider not in api_keys:
             ConsoleOutput.warning(f"API key for '{provider}' not found. Execution might fail.")

    elif args.use_gguf:
        if not LLAMACPP_AVAILABLE:
            ConsoleOutput.error("Cannot use --use-gguf. llama-cpp-python is not installed.")
            return 1
        provider = "local_gguf"
        model_specifier = str(args.use_gguf.resolve()) # Use the absolute path
        local_backend_type = "gguf" # Changed for clarity
        ConsoleOutput.info(f"Using Local GGUF Backend: {Path(model_specifier).name}")
        # GGUF doesn't need a ModelEntry from registry in the same way

    elif args.model_id:
        provider = "local"
        model_specifier = args.model_id
        local_backend_type = "transformers"
        ConsoleOutput.info(f"Using Local HF Backend (by ID): {model_specifier}")
        selected_model_entry = registry.get_model(model_specifier)
        if not selected_model_entry:
             ConsoleOutput.warning(f"Model ID '{model_specifier}' not found in registry. Using default settings for it.")
             # Create a temporary minimal entry
             selected_model_entry = ModelEntry(model_id=model_specifier, name=model_specifier.split('/')[-1], author="Unknown")

    elif args.model: # Predefined local model key
        provider = "local"
        local_backend_type = "transformers"
        selected_model_entry = registry.get_by_key(args.model)
        if selected_model_entry:
            model_specifier = selected_model_entry.model_id
            ConsoleOutput.info(f"Using Local HF Backend (by key '{args.model}'): {model_specifier}")
        else:
            ConsoleOutput.error(f"Predefined key '{args.model}' not found in registry. Use --list-models.")
            # Fallback to default key
            default_entry = registry.get_by_key(DEFAULT_MODEL)
            if default_entry:
                 ConsoleOutput.warning(f"Falling back to default key '{DEFAULT_MODEL}': {default_entry.model_id}")
                 model_specifier = default_entry.model_id
                 selected_model_entry = default_entry
            else:
                 ConsoleOutput.error("Default model key also not found. Cannot proceed.")
                 return 1
    else:
         # No specific model chosen, use default local model key
         provider = "local"
         local_backend_type = "transformers"
         default_entry = registry.get_by_key(DEFAULT_MODEL)
         if default_entry:
             model_specifier = default_entry.model_id
             selected_model_entry = default_entry
             ConsoleOutput.info(f"Using Default Local HF Backend ('{DEFAULT_MODEL}'): {model_specifier}")
         else:
             ConsoleOutput.error(f"Default model key '{DEFAULT_MODEL}' not found in registry. Cannot proceed.")
             return 1


    # --- Interactive Model Selection (if --browse-models) ---
    if args.browse_models:
         if provider != "local" or local_backend_type != "transformers":
              ConsoleOutput.warning("--browse-models is only for local HuggingFace Transformers models. Ignoring.")
         else:
              browser = InteractiveModelBrowser(ModelHub()) # Use default ModelHub
              hf_model_info = browser.browse(task="text-generation", library="transformers")
              if hf_model_info:
                   model_specifier = hf_model_info.model_id
                   ConsoleOutput.info(f"Selected via browser: {model_specifier}")
                   # Update selected_model_entry
                   selected_model_entry = registry.get_model(model_specifier)
                   if not selected_model_entry:
                        registry.add_from_model_info(hf_model_info) # Add to registry if new
                        selected_model_entry = registry.get_model(model_specifier)
              else:
                   ConsoleOutput.warning("No model selected via browser. Using previous selection or default.")
                   # Keep the previously determined model_specifier and selected_model_entry


    # 3. Configure Hyperparameters
    hyperparams = handle_hyperparameter_configuration(args)


    # 4. Initialize Backend
    text_backend: Optional[LLMBackend] = None
    try:
        ConsoleOutput.section("Initializing Text Backend")
        # Prepare kwargs for backend factory
        backend_kwargs = {
            "provider": provider,
            "model_specifier": model_specifier,
            "api_keys": api_keys,
            "hyperparameters": hyperparams,
        }
        # Add local-specific configs if needed
        if provider == "local":
            if local_backend_type == "transformers":
                backend_kwargs['quantization_config'] = create_quantization_config(args)
                backend_kwargs['layer_split_config'] = create_layer_split_config(args)
                # Pass ModelConfig derived from ModelEntry
                if selected_model_entry:
                    from config import ModelConfig # Local import needed here
                    backend_kwargs['model_config'] = ModelConfig(
                         name=selected_model_entry.name, model_id=selected_model_entry.model_id,
                         supports_thinking=selected_model_entry.supports_thinking, thinking_tokens=selected_model_entry.thinking_tokens or [],
                         max_context=selected_model_entry.max_context, optimal_chunk_size=selected_model_entry.optimal_chunk_size,
                         temperature=selected_model_entry.temperature, top_p=selected_model_entry.top_p,
                         max_new_tokens=selected_model_entry.max_new_tokens,
                         quantization_support=selected_model_entry.quantization_support or ["4bit", "8bit"]
                    )
                else:
                    # Should not happen if logic above is correct, but raise error if it does
                    raise ValueError("ModelEntry not available for local Transformers backend.")

        elif provider == "local_gguf":
            # Pass layer split config to factory for GGUF
            backend_kwargs['layer_split_config'] = create_layer_split_config(args)

        # Use the factory function
        text_backend = get_llm_backend(**backend_kwargs)

        if text_backend is None:
             raise RuntimeError(f"Failed to create backend for provider '{provider}'. Check logs.")


        # Load the model (local) or initialize client (cloud)
        load_success = text_backend.load_model(trust_remote_code=True) # Pass trust_remote_code for HF
        if not load_success:
             raise RuntimeError(f"Failed to load/initialize backend: {text_backend.model_identifier}")

        ConsoleOutput.success("Text backend initialized.")

    except Exception as e:
        ConsoleOutput.error(f"Failed to initialize text backend: {e}")
        logger.error("Text backend init failed (CLI)", exc_info=True)
        return 1


    # 5. Collect Input Files
    batch_manager = BatchFileManager() # Use default FileHandler settings
    try:
        input_files = batch_manager.collect_input_files(args.input, recursive=args.recursive)
    except Exception as e:
        ConsoleOutput.error(f"Error collecting input files: {e}")
        if text_backend: text_backend.unload_model() # Cleanup backend
        return 1

    if not input_files:
        ConsoleOutput.error("No valid input files found at specified paths.")
        if text_backend: text_backend.unload_model()
        return 1
    ConsoleOutput.info(f"Found {len(input_files)} file(s) to process.")


    # 6. Initialize Pipeline
    pipeline: Optional[ProcessingPipeline] = None
    try:
        # Create Quantization/LayerSplit configs again for PipelineConfig (or pass them)
        q_config = create_quantization_config(args)
        ls_config = create_layer_split_config(args)

        pipeline_config = PipelineConfig(
            mode=ProcessingMode(args.mode),
            model_provider=provider,
            model_specifier=model_specifier,
            model_name=f"{provider}:{Path(model_specifier).name if provider=='local_gguf' else model_specifier}", # Metadata name
            chunking_strategy=ChunkingStrategy(args.chunk_strategy),
            chunk_size=args.chunk_size,
            system_prompt=args.system_prompt or PREPROCESS_PROMPT, # Use default if not provided
            remove_thinking=not args.no_thinking,
            output_format=args.format,
            hyperparameters=hyperparams,
            quantization_config=q_config, # Pass config
            layer_split_config=ls_config, # Pass config
            # preserve_layout can be added to CLI args if needed
        )
        pipeline = ProcessingPipeline(pipeline_config)
        pipeline.llm_backend = text_backend # Inject the initialized backend

        # Explicitly set stages to run (all text stages for CLI by default)
        pipeline.set_stages_to_run(list(PIPELINE_STAGES)) # Run all text stages

        # 7. Process Files
        ConsoleOutput.section("Starting Processing")
        overall_success = False
        output_dir = args.output_dir or Path(OUTPUT_DIR) # Use arg or default base

        if len(input_files) == 1:
            # Handle single file output path
            output_p = Path(args.output) if args.output else None # Base name or full path
            if output_p and output_p.is_dir():
                 ConsoleOutput.error(f"Specified output path '{args.output}' is a directory. Please provide a filename for single file processing.")
                 raise ValueError("Output path cannot be a directory for single file.")
            if output_p and not output_p.parent.exists():
                 try: output_p.parent.mkdir(parents=True, exist_ok=True)
                 except Exception as e: ConsoleOutput.warning(f"Could not create output directory {output_p.parent}: {e}")

            # If -o gives a name without dir, combine with output_dir
            if output_p and not output_p.is_absolute() and output_p.parent == Path('.'):
                 final_output_base = output_dir / output_p.name
            elif output_p: # User provided full path or path with directory
                 final_output_base = output_p
            else: # No -o, let pipeline decide name in output_dir
                 final_output_base = None # Signal pipeline to use default naming in output_dir
                 pipeline.file_handler.output_dir = output_dir # Ensure pipeline saves to correct dir


            result = pipeline.process_file(input_files[0], output_path=final_output_base)
            all_results = [result] # Store result for potential audio stage
            overall_success = result.success
        else: # Batch processing
            if args.output:
                 ConsoleOutput.warning("-o/--output is ignored during batch processing. Files saved to directory specified by --output-dir or default.")
            results = pipeline.process_batch(input_files, output_dir=output_dir)
            all_results = results
            overall_success = all(r.success for r in results)

        # 8. Audio Generation (Conditional)
        if args.generate_audio:
             # Basic implementation sketch - needs refinement
             ConsoleOutput.section("🔊 Starting Audio Generation (CLI) 🔊")
             audio_backend: Optional[AudioBackend] = None
             try:
                  # --- Determine Audio Backend ---
                  # For CLI, we need args or use defaults from AppState/Config
                  # Example: using defaults from AppState (adjust as needed)
                  temp_state = AppState() # Get defaults
                  audio_provider = temp_state.audio_model_provider
                  audio_specifier = temp_state.audio_model_specifier
                  audio_config = temp_state.audio_config
                  ConsoleOutput.info(f"Using default audio backend: {audio_provider} / {audio_specifier}")

                  # Check API key if cloud
                  if audio_provider != 'local' and audio_provider not in api_keys:
                       raise ValueError(f"API key for audio provider '{audio_provider}' missing.")

                  audio_backend = get_audio_backend(
                      provider=audio_provider,
                      model_specifier=audio_specifier,
                      api_keys=api_keys,
                      config=audio_config
                  )
                  if audio_backend is None: raise RuntimeError("Could not create audio backend.")
                  if not audio_backend.load_model(): raise RuntimeError("Failed to load audio backend.")

                  # --- Find Text Files and Generate Audio ---
                  text_files = [res.output_file for res in all_results if res.success and res.output_file and res.output_file.exists()]
                  if not text_files:
                       ConsoleOutput.warning("No successful text output files found to generate audio from.")
                  else:
                       ConsoleOutput.info(f"Generating audio for {len(text_files)} file(s)...")
                       audio_success_count = 0
                       for i, txt_file in enumerate(text_files):
                            ConsoleOutput.info(f"Audio {i+1}/{len(text_files)}: {txt_file.name}")
                            try:
                                text = txt_file.read_text('utf-8')
                                # Basic cleaning (remove front matter, etc.) - reuse logic from menu
                                text = re.sub(r'^---.*?---', '', text, flags=re.DOTALL | re.MULTILINE).strip()
                                if pipeline_config.mode == ProcessingMode.PODCAST:
                                     text = re.sub(r'^\*\*\[Speaker.*?\]:\*\*\n*', '', text, flags=re.MULTILINE)
                                text = re.sub(r'(\*\*|\*|`|#+\s)', '', text) # Remove basic markdown

                                if not text.strip():
                                     ConsoleOutput.warning("Skipping empty text.")
                                     continue

                                audio_out_path = txt_file.with_suffix(f".{audio_config.output_format}")
                                audio_res = audio_backend.generate_audio(text, audio_out_path)
                                if audio_res:
                                     ConsoleOutput.success(f" -> Saved: {audio_res.audio_path.name}")
                                     audio_success_count += 1
                                else:
                                     ConsoleOutput.error(" -> Audio generation failed.")
                                     overall_success = False
                            except Exception as audio_e:
                                 ConsoleOutput.error(f" -> Error generating audio: {audio_e}")
                                 logger.error(f"Audio gen failed for {txt_file.name}", exc_info=True)
                                 overall_success = False
                       ConsoleOutput.info(f"Audio generation complete: {audio_success_count} successful.")

             except Exception as audio_setup_e:
                  ConsoleOutput.error(f"Failed to set up audio generation: {audio_setup_e}")
                  logger.error("Audio setup failed (CLI)", exc_info=True)
                  overall_success = False # Mark as failure if audio was requested but setup failed
             finally:
                  if audio_backend: audio_backend.unload_model() # Unload audio model


        return 0 if overall_success else 1

    except KeyboardInterrupt:
        ConsoleOutput.warning("\nProcessing interrupted by user.")
        return 130 # Standard exit code for Ctrl+C
    except Exception as e:
        ConsoleOutput.error(f"An unexpected error occurred during CLI processing: {e}")
        logger.critical("CLI processing failed", exc_info=True)
        # Attempt to save error context if text_backend logger exists
        if text_backend and hasattr(text_backend, 'logger') and hasattr(text_backend.logger, 'error'):
             try: text_backend.logger.error(f"CLI processing failed: {e}", exc_info=True, save_context=True)
             except: pass # Avoid errors during error handling
        return 1
    finally:
        # Cleanup backend and pipeline resources
        ConsoleOutput.info("Cleaning up resources...")
        if text_backend:
            try: text_backend.unload_model()
            except Exception as unload_e: logger.warning(f"Error during text backend cleanup: {unload_e}")
        if pipeline:
            try: pipeline.cleanup_file_handler() # Save report if files were processed
            except Exception as cleanup_e: logger.warning(f"Error during pipeline cleanup: {cleanup_e}")
        ConsoleOutput.info("Cleanup complete.")


# --- Main Entry Point ---

def main():
    """Main execution function: parses args, runs utilities, CLI, or Menu."""
    try:
        get_registry() # Initialize registry singleton first
    except Exception as e:
        print(f"Warning: Could not initialize model registry: {e}")
        # Continue without registry functionality if it fails

    parser = parse_arguments()
    args = parser.parse_args()

    # Set verbosity level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger("llamanote").setLevel(log_level)
    # Optionally adjust other loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logger.info(f"Log level set to {logging.getLevelName(log_level)}")


    # --- Handle Utility Commands First ---
    utility_commands = {
        'list_models': list_models,
        'search_models': lambda: search_models(args.search_models) if args.search_models else parser.error("--search-models requires a search term"),
        'list_gguf': list_gguf_models,
        'hyperparam_help': lambda: print(get_hyperparameter_help(args.hyperparam_help)) if args.hyperparam_help else parser.error("--hyperparam-help requires a parameter name"),
        'refresh_registry': refresh_registry,
        'export_registry': lambda: export_registry(args.export_registry) if args.export_registry else parser.error("--export-registry requires a path"),
        'import_registry': lambda: import_registry(args.import_registry) if args.import_registry else parser.error("--import-registry requires a path"),
        'registry_stats': show_registry_stats,
    }

    utility_arg_passed = False
    for arg_name, func in utility_commands.items():
        arg_value = getattr(args, arg_name, None)
        # Check for boolean flags or if the argument has a value
        if (isinstance(arg_value, bool) and arg_value) or (not isinstance(arg_value, bool) and arg_value is not None):
            utility_arg_passed = True
            try:
                func() # Execute the utility function
            except Exception as util_e:
                ConsoleOutput.error(f"Error running utility command '{arg_name}': {util_e}")
                logger.error(f"Utility command '{arg_name}' failed", exc_info=True)
                return 1 # Exit with error code
            break # Assume only one utility command per run

    if utility_arg_passed:
        # If any utility command was run, exit cleanly
        print("\nUtility command finished.")
        return 0

    # --- Decide between CLI processing and Interactive Menu ---
    if args.input:
        # If input files ARE provided, run CLI processing
        ConsoleOutput.info("Input files provided. Running in command-line mode...")
        exit_code = run_cli_processing(args, parser)
        return exit_code
    else:
        # If NO input files and NO utility commands were run, start the menu
        ConsoleOutput.info("No input files specified. Starting interactive menu...")
        try:
            # Re-initialize registry here to ensure it's fresh for the menu
            try: get_registry()
            except Exception: pass # Ignore if fails again

            menu = MenuSystem()
            menu.run() # This blocks until the menu exits
            return 0
        except Exception as e:
            ConsoleOutput.error(f"Failed to start menu system: {e}")
            logger.critical("Menu system failed", exc_info=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
