# llamanote/cli.py
"""
Command-Line Interface (CLI) Handler for LlamaNote Enhanced
"""

import argparse
import sys
import time
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.utils.logger import ConsoleOutput, get_logger_conf
from src.config.manager import ConfigManager
from src.config.profiles import list_memory_profiles, create_configs_from_memory_profile
from src.config.presets import get_hyperparameter_preset, list_hyperparameter_presets
from src.config.settings import (
    DEFAULT_MODEL_KEY, FALLBACK_MODEL_KEY, QUANTIZATION_OPTIONS,
    DEFAULT_QUANTIZATION, DEFAULT_GPU_LAYERS, ENABLE_LAYER_SPLITTING,
    DEFAULT_PIPELINE_STAGES, OUTPUT_FORMAT_OPTIONS, DEFAULT_OUTPUT_FORMAT,
    DEFAULT_OUTPUT_DIR, PREPROCESS_PROMPT_PODCAST, SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_TTS_PROVIDERS
)
from src.core.types import ProcessingMode, PipelineConfig, ChunkingStrategy, PipelineResult
from src.core.pipeline import ProcessingPipeline
from src.models.registry import get_registry, ModelEntry
from src.models.hyperparameters import HyperparameterConfig, InteractiveHyperparameterEditor, get_hyperparameter_help
from src.models.hub import ModelHub, ModelHubInfo
from src.models.backends import get_llm_backend, get_audio_backend, LLMBackend, AudioBackend
from src.io.batch_manager import BatchFileManager
from src.core.errors import ModelLoadError, ConfigurationError, FileProcessingError, MissingDataError

# Conditional import for GGUF
try:
    from models.backends.local_gguf import GGUFModelManager, LLAMACPP_AVAILABLE
except ImportError:
    LLAMACPP_AVAILABLE = False
    GGUFModelManager = None

logger = get_logger_conf(__name__)

def parse_arguments() -> argparse.ArgumentParser:
    """Creates the argument parser for the CLI."""
    
    # Try to get predefined model keys for choices
    try:
        registry = get_registry()
        predefined_models = registry.list_models(is_predefined=True, sort_by="name")
        model_keys = [entry.short_key for entry in predefined_models if entry.short_key]
        default_key = DEFAULT_MODEL_KEY
    except Exception as e:
        logger.warning(f"Could not load predefined models for CLI help: {e}. Using basic defaults.")
        model_keys = [DEFAULT_MODEL_KEY, FALLBACK_MODEL_KEY]
        default_key = DEFAULT_MODEL_KEY

    parser = argparse.ArgumentParser(
        description="LlamaNote Enhanced - Process PDFs/text into structured notes or audio.",
        formatter_class=argparse.RawTextHelpFormatter, # Use RawText for better formatting
        epilog=f"""
Examples:
  Run interactively:
    python -m llamanote

  Process a single file with defaults (local Qwen model, podcast format):
    python -m llamanote ./docs/my_file.pdf

  Process multiple files and output to a specific directory:
    python -m llamanote ./docs/file1.pdf ./docs/another.txt -d ./my_outputs

  Process a whole directory recursively:
    python -m llamanote ./docs/ -r -m technical

  Use a specific local Hugging Face model with 4-bit quantization:
    python -m llamanote file.pdf --model-id "NousResearch/Llama-2-7b-chat-hf" --quantization 4bit

  Use a local GGUF model with full GPU offload:
    python -m llamanote file.pdf --use-gguf /path/to/model.gguf --gpu-layers -1

  Use an OpenAI model (requires API key set via menu):
    python -m llamanote file.pdf --cloud-provider openai --cloud-model gpt-4o

  Generate text and audio (using default audio models):
    python -m llamanote file.pdf --generate-audio

  List available models in your registry:
    python -m llamanote --list-models

  Search Hugging Face for new models to add:
    python -m llamanote --search-models "gemma-2"
"""
    )
    
    # --- Primary Arguments ---
    parser.add_argument(
        "input",
        nargs="*",
        help="Path(s) to input PDF/text file(s) or directory. If omitted, starts interactive menu."
    )
    # Add near other related arguments in parse_arguments()
    other_group.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search input directories."
    )
    utility_group.add_argument( # Add to utility group
        "--browse-models",
        action="store_true",
        help="Interactively browse Hugging Face Hub models (for local_hf)."
    )
    # Import missing items at the top
    from src.config.settings import (
        # ... existing imports ...
        CACHE_DIR, CHUNK_OVERLAP, MAX_RETRIES, FALLBACK_ON_ERROR, # Add missing
        PREPROCESS_PROMPT_PODCAST, DEFAULT_SYSTEM_PROMPT
    )
    from src.models.hyperparameters import get_hyperparameter_help # Add missing import
    # Remove unused import:
    # from src.models.hub import InteractiveModelBrowser # Not used
    # Make sure these are imported for the error handling block:
    from src.core.errors import ConfigurationError, FileProcessingError, MissingDataError, ModelLoadError # Ensure ModelLoadError is there
    # Ensure PipelineResult is imported from types (Fix #4)
    # Inside run_cli_processing, update relevant constants if needed or confirm they come from pipeline_config
    # e.g., CHUNK_OVERLAP, MAX_RETRIES, FALLBACK_ON_ERROR are used to create PipelineConfig, which is good.
    # CACHE_DIR might be needed for ModelHub instantiation if not handled by registry defaults. Add import if needed.    
    # --- Utility Actions (run instead of processing) ---
    utility_group = parser.add_argument_group('Utility Commands (run separately)')
    utility_group.add_argument(
        "--list-models",
        action="store_true",
        help="List models in the local registry and exit."
    )
    utility_group.add_argument(
        "--refresh-registry",
        action="store_true",
        help="Refresh model registry from HuggingFace Hub and exit."
    )
    utility_group.add_argument(
        "--search-models",
        metavar="TERM",
        help="Search HuggingFace Hub for models matching TERM and exit."
    )
    utility_group.add_argument(
        "--list-gguf",
        action="store_true",
        help="List cached GGUF models and exit (requires llama-cpp-python)."
    )
    utility_group.add_argument(
        "--export-registry", type=Path, metavar="FILE.json", help="Export model registry to JSON file and exit.")
    utility_group.add_argument(
        "--import-registry", type=Path, metavar="FILE.json", help="Import/merge model registry from JSON file and exit.")
    utility_group.add_argument(
        "--registry-stats", action="store_true", help="Show model registry statistics and exit.")
    utility_group.add_argument(
        "--hyperparam-help", metavar="PARAM", help="Show detailed help for a specific hyperparameter and exit.")

    # --- Processing Mode & Output ---
    proc_group = parser.add_argument_group('Processing & Output')
    proc_group.add_argument(
        "-m", "--mode", 
        choices=[mode.value for mode in ProcessingMode], 
        default=ProcessingMode.PODCAST.value, 
        help=f"Processing mode (default: {ProcessingMode.PODCAST.value})"
    )
    proc_group.add_argument(
        "-f", "--format", 
        choices=OUTPUT_FORMAT_OPTIONS, 
        default=DEFAULT_OUTPUT_FORMAT, 
        help=f"Output format (default: {DEFAULT_OUTPUT_FORMAT})"
    )
    proc_group.add_argument(
        "-o", "--output", 
        help="Output file path (for single file input) or directory (for batch input)."
    )
    proc_group.add_argument(
        "-d", "--output-dir", 
        type=Path, 
        default=DEFAULT_OUTPUT_DIR, 
        help=f"Default output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    proc_group.add_argument(
        "--no-timestamp", 
        action="store_true", 
        help="Disable adding timestamps to output filenames."
    )
    proc_group.add_argument(
        "--no-metadata", 
        action="store_true", 
        help="Do not include a metadata header in the output file."
    )

    # --- Model Selection (Mutually Exclusive) ---
    model_group = parser.add_argument_group('Model Selection (Choose One)')
    model_exclusive_group = model_group.add_mutually_exclusive_group()
    model_exclusive_group.add_argument(
        "--model", 
        choices=model_keys, 
        default=default_key,
        help="Use a predefined local model key (e.g., 'qwen3-4b')."
    )
    model_exclusive_group.add_argument(
        "--model-id", 
        help="Use a specific Hugging Face model ID (e.g., 'google/gemma-2-9b-it')."
    )
    model_exclusive_group.add_argument(
        "--use-gguf", 
        type=Path, 
        metavar="FILE.gguf",
        help="Use a local GGUF model file."
    )
    model_exclusive_group.add_argument(
        "--cloud-provider", 
        choices=[p for p in SUPPORTED_LLM_PROVIDERS if "local" not in p],
        help="Use a cloud provider (e.g., 'openai', 'anthropic', 'google')."
    )

    # --- Model Configuration ---
    model_config_group = parser.add_argument_group('Model Configuration')
    model_config_group.add_argument(
        "--cloud-model", 
        help="Model name for the cloud provider (e.g., 'gpt-4o', 'claude-3-5-sonnet-20240620')."
    )
    model_config_group.add_argument(
        "--quantization",
        choices=QUANTIZATION_OPTIONS,
        default=None, # Default is handled by memory profile
        help=f"Override quantization for local HF models (e.g., 4bit, 8bit, none)."
    )
    model_config_group.add_argument(
        "--gpu-layers",
        type=int,
        default=None, # Default is handled by memory profile
        help="Override GPU layers for local GGUF models (-1=all, 0=CPU)."
    )
    model_config_group.add_argument(
        "--memory-profile",
        choices=[p[0] for p in list_memory_profiles()],
        default="medium_vram",
        help="Set hardware constraints (default: medium_vram)."
    )

    # --- Hyperparameters ---
    hyperparam_group = parser.add_argument_group('Hyperparameter Tuning')
    hyperparam_group.add_argument(
        "--configure-hyperparams",
        action="store_true",
        help="Interactively edit hyperparameters before running."
    )
    hyperparam_group.add_argument(
        "--hyperparams-preset",
        choices=[p[0] for p in list_hyperparameter_presets()],
        help="Use a predefined hyperparameter preset (e.g., 'creative', 'precise')."
    )
    hyperparam_group.add_argument("--temperature", type=float, help="Override temperature (e.g., 0.7)")
    hyperparam_group.add_argument("--top-p", type=float, help="Override top_p (e.g., 0.9)")
    hyperparam_group.add_argument("--top-k", type=int, help="Override top_k (e.g., 50)")
    hyperparam_group.add_argument("--max-tokens", type=int, help="Override max_new_tokens")
    
    # --- Other Processing Options ---
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument(
        "--chunk-size", 
        type=int, 
        default=CHUNK_SIZE_DEFAULT, 
        help=f"Target text chunk size (default: {CHUNK_SIZE_DEFAULT})"
    )
    other_group.add_argument(
        "--chunk-strategy", 
        choices=[s.value for s in ChunkingStrategy], 
        default=ChunkingStrategy.WORD_BOUNDARY.value, 
        help="Chunking strategy (default: word_boundary)"
    )
    other_group.add_argument(
        "--system-prompt", 
        help="Path to a text file containing a custom system prompt."
    )
    other_group.add_argument(
        "--no-thinking-filter",
        action="store_true",
        help="Disable removal of <think>...</think> tokens."
    )
    other_group.add_argument(
        "--no-layout",
        action="store_true",
        help="Disable layout preservation during PDF text extraction."
    )
    other_group.add_argument(
        "--no-audio-clean",
        action="store_true",
        help="Disable audio-specific text cleaning (e.g., removing citations)."
    )
    other_group.add_argument(
        "--generate-audio", 
        action="store_true", 
        help="Generate audio file (TTS) from the final text output."
    )
    other_group.add_argument(
        "--audio-provider",
        choices=SUPPORTED_TTS_PROVIDERS,
        default="local_audio",
        help="Audio generation backend (default: local_audio)"
    )
    other_group.add_argument(
        "--audio-model",
        default="microsoft/speecht5_tts",
        help="Model ID/name for audio generation (default: microsoft/speecht5_tts)"
    )
    other_group.add_argument(
        "-v", "--verbose", 
        action="store_true", 
        help="Enable verbose logging to console (DEBUG level)"
    )
    other_group.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="Disable saving/loading intermediate processing steps."
    )
    
    return parser

def handle_utility_commands(args: argparse.Namespace, parser: argparse.ArgumentParser) -> bool:
    """Executes utility commands that exit immediately. Returns True if a command was run."""
    
    registry = get_registry() # Get singleton instance
    
    if args.list_models:
        ConsoleOutput.header("Available Models in Registry")
        predefined = registry.list_models(is_predefined=True, sort_by="name")
        if predefined:
            ConsoleOutput.section("Predefined Models (use key with --model)")
            for model in predefined:
                key_info = f"(Key: {model.short_key})"
                print(f"- {model.name} {key_info}\n  ID: {model.model_id}")
        
        other_models = registry.list_models(is_predefined=False, sort_by="downloads")
        if other_models:
            ConsoleOutput.section("Other Models in Registry (use ID with --model-id)")
            for model in other_models[:20]: # Limit display
                print(f"- {model.model_id} (Downloads: {model.downloads})")
            if len(other_models) > 20:
                print(f"  ... and {len(other_models) - 20} more.")
        return True

    if args.search_models:
        hub = ModelHub(cache_dir=CACHE_DIR / "model_hub")
        ConsoleOutput.header(f"Searching Hugging Face Hub for: '{args.search_models}'")
        results = hub.search_models(search_term=args.search_models, limit=20)
        if not results:
            ConsoleOutput.warning("No models found.")
        else:
            for i, model in enumerate(results, 1):
                print(f"{i:2}. {model.display_name}")
                print(f"    ID: {model.model_id}")
                print(f"    Downloads: {model.downloads:,} | Likes: {model.likes}")
                if model.description:
                    print(f"    Desc: {model.description}...")
            print("\nUse the full Model ID with --model-id to use one of these models.")
        return True

    if args.list_gguf:
        if not LLAMACPP_AVAILABLE:
            ConsoleOutput.error("llama-cpp-python is not installed. Cannot list GGUF models.")
            print("Install with: pip install llama-cpp-python")
            return True
        manager = GGUFModelManager(cache_dir=CACHE_DIR / "gguf_models")
        models = manager.list_available_models()
        ConsoleOutput.header("Available GGUF Models (in cache)")
        if not models:
            ConsoleOutput.warning(f"No GGUF models found in {manager.cache_dir}")
        else:
            for model_path in models:
                info = manager.get_model_info(model_path)
                print(f"- {info['name']} ({info['size_mb']:.1f} MB)")
                print(f"  Path: {info['path']}")
        return True
        
    if args.refresh_registry:
        from .models.registry import refresh_registry_from_hub # Local import
        ConsoleOutput.info("Refreshing model registry from Hugging Face Hub...")
        count = refresh_registry_from_hub(limit=100)
        ConsoleOutput.success(f"Registry refresh complete. Added/updated {count} models.")
        return True
        
    if args.export_registry:
        registry.export_to_json(args.export_registry)
        ConsoleOutput.success(f"Registry exported to {args.export_registry}")
        return True

    if args.import_registry:
        registry.import_from_json(args.import_registry)
        ConsoleOutput.success(f"Registry imported from {args.import_registry}")
        return True

    if args.registry_stats:
        stats = registry.get_statistics()
        ConsoleOutput.header("Model Registry Statistics")
        for key, value in stats.items():
            print(f"- {key.replace('_', ' ').title()}: {value}")
        return True
        
    if args.hyperparam_help:
        print(get_hyperparameter_help(args.hyperparam_help))
        return True

    return False # No utility command was run

def run_cli_processing(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """
    Handles the main processing logic when run from the command line.
    Returns an exit code (0 for success, 1 for failure).
    """
    config_manager = ConfigManager()
    registry = get_registry()
    
    # 1. Load API Keys
    api_keys = config_manager.load_cloud_keys()

    # 2. Determine Model Provider and Specifier
    provider = "local_hf" # Default
    model_specifier = ""
    model_entry: Optional[ModelEntry] = None

    if args.cloud_provider:
        provider = args.cloud_provider
        if not args.cloud_model:
            ConsoleOutput.error(f"--cloud-model is required when using --cloud-provider {provider}")
            return 1
        model_specifier = args.cloud_model
        if provider not in api_keys:
             ConsoleOutput.warning(f"API key for '{provider}' not found. Execution may fail if required.")
    elif args.use_gguf:
        if not LLAMACPP_AVAILABLE:
            ConsoleOutput.error("Cannot use --use-gguf. llama-cpp-python is not installed.")
            return 1
        provider = "local_gguf"
        model_specifier = str(args.use_gguf.resolve())
    elif args.model_id:
        provider = "local_hf"
        model_specifier = args.model_id
        model_entry = registry.get_model(model_specifier)
    else: # Use --model key (default or specified)
        provider = "local_hf"
        model_entry = registry.get_by_key(args.model)
        if not model_entry:
            ConsoleOutput.error(f"Predefined model key '{args.model}' not found in registry.")
            ConsoleOutput.info("Attempting to fall back to default model...")
            model_entry = registry.get_by_key(DEFAULT_MODEL_KEY)
            if not model_entry:
                 ConsoleOutput.error(f"Default model '{DEFAULT_MODEL_KEY}' also not found. Cannot proceed.")
                 return 1
        model_specifier = model_entry.model_id

    # 3. Handle --browse-models
    if args.browse_models:
         if provider != "local_hf":
              ConsoleOutput.warning("--browse-models is only for local HuggingFace models. Ignoring.")
         else:
              ConsoleOutput.info("Opening interactive model browser...")
              hub = ModelHub(cache_dir=CACHE_DIR / "model_hub")
              browser = InteractiveModelBrowser(hub) # Assumes InteractiveModelBrowser is in model_hub
              hf_model_info: Optional[ModelHubInfo] = browser.browse(task="text-generation", library="transformers")
              if hf_model_info:
                   model_specifier = hf_model_info.model_id
                   ConsoleOutput.info(f"Selected model from browser: {model_specifier}")
                   # Add/update in registry and get the entry
                   registry.add_from_model_info(hf_model_info, save=True)
                   model_entry = registry.get_model(model_specifier)
              else:
                   ConsoleOutput.warning("No model selected. Using previous selection.")
                   if not model_entry: # Make sure we still have a valid entry
                        model_entry = registry.get_by_key(DEFAULT_MODEL_KEY)
                        model_specifier = model_entry.model_id
                        ConsoleOutput.info(f"Reverting to default model: {model_specifier}")


    # 4. Resolve ModelEntry for local_hf
    if provider == "local_hf" and not model_entry:
        model_entry = registry.get_model(model_specifier)
        if not model_entry:
            ConsoleOutput.warning(f"Model ID '{model_specifier}' not in registry. Fetching info...")
            hub = ModelHub(cache_dir=CACHE_DIR / "model_hub")
            info = hub.get_model_info(model_specifier)
            if info:
                registry.add_from_model_info(info, save=True)
                model_entry = registry.get_model(model_specifier)
            else:
                ConsoleOutput.warning(f"Could not fetch info for '{model_specifier}'. Using default settings.")
                model_entry = ModelEntry(model_id=model_specifier, name=model_specifier.split('/')[-1], author="Unknown")

    # 5. Configure Hardware
    quant_config, layer_split_config = create_configs_from_memory_profile(args.memory_profile)
    # Apply CLI overrides
    if args.quantization:
        quant_config.method = args.quantization
        logger.info(f"Overriding quantization to: {args.quantization}")
    if args.gpu_layers is not None:
        layer_split_config.gpu_layers = args.gpu_layers
        logger.info(f"Overriding GPU layers to: {args.gpu_layers}")

    # 6. Configure Hyperparameters
    hyperparams = handle_hyperparameter_configuration(args)

    # 7. Configure System Prompt
    system_prompt_str = PREPROCESS_PROMPT_PODCAST # Default
    if args.system_prompt:
        try:
            system_prompt_str = Path(args.system_prompt).read_text(encoding='utf-T8')
            ConsoleOutput.info(f"Loaded custom system prompt from {args.system_prompt}")
        except Exception as e:
            ConsoleOutput.error(f"Failed to read system prompt file '{args.system_prompt}': {e}. Using default.")
    elif args.mode != ProcessingMode.PODCAST.value:
         system_prompt_str = DEFAULT_SYSTEM_PROMPT # Use generic one for other modes

    # 8. Initialize PipelineConfig
    try:
        pipeline_config = PipelineConfig(
            mode=ProcessingMode(args.mode),
            model_provider=provider,
            model_specifier=model_specifier,
            model_name=f"{provider}:{Path(model_specifier).name if provider=='local_gguf' else model_specifier}",
            chunking_strategy=ChunkingStrategy(args.chunk_strategy),
            chunk_size=args.chunk_size,
            chunk_overlap=CHUNK_OVERLAP, # Using constant from settings
            system_prompt=system_prompt_str,
            remove_thinking=not args.no_thinking_filter,
            preserve_pdf_layout=not args.no_layout,
            clean_for_audio=not args.no_audio_clean,
            add_emotions=(args.mode == ProcessingMode.PODCAST.value),
            enable_checkpoints=not args.no_checkpoints,
            max_retries=MAX_RETRIES,
            fallback_on_error=FALLBACK_ON_ERROR,
            output_format=args.format,
            output_dir=args.output_dir or DEFAULT_OUTPUT_DIR,
            timestamp_outputs=not args.no_timestamp,
            include_metadata=not args.no_metadata,
            hyperparameters=hyperparams,
            quantization_config=quant_config,
            layer_split_config=layer_split_config,
            generate_audio=args.generate_audio,
            audio_provider=args.audio_provider,
            audio_specifier=args.audio_model,
            audio_config=None # Let pipeline init default AudioConfig
        )
    except ValueError as e:
        ConsoleOutput.error(f"Invalid configuration value: {e}")
        return 1

    # 9. Initialize Backends
    text_backend: Optional[LLMBackend] = None
    audio_backend: Optional[AudioBackend] = None
    pipeline: Optional[ProcessingPipeline] = None
    
    try:
        # --- Text Backend ---
        ConsoleOutput.section("Initializing Text Backend")
        backend_kwargs = {
            "provider": provider,
            "model_specifier": model_specifier,
            "api_keys": api_keys,
            "hyperparameters": hyperparams,
            "model_config": model_entry, # Pass ModelEntry (None for cloud/GGUF)
            "quantization_config": quant_config,
            "layer_split_config": layer_split_config,
        }
        text_backend = get_llm_backend(**backend_kwargs)
        if text_backend is None:
             raise ModelLoadError(f"Could not create backend for provider '{provider}'.")
        
        if not text_backend.load(trust_remote_code=True):
             raise ModelLoadError(f"Failed to load/initialize backend: {text_backend.model_identifier}")
        ConsoleOutput.success("Text backend initialized.")

        # --- Audio Backend (if needed) ---
        if pipeline_config.generate_audio:
            ConsoleOutput.section("Initializing Audio Backend")
            audio_backend = get_audio_backend(
                provider=pipeline_config.audio_provider,
                model_specifier=pipeline_config.audio_specifier,
                api_keys=api_keys,
                config=pipeline_config.audio_config # Pass the default config
            )
            if audio_backend is None:
                 raise ModelLoadError(f"Could not create audio backend for provider '{pipeline_config.audio_provider}'.")
            
            if not audio_backend.load():
                 raise ModelLoadError(f"Failed to load/initialize audio backend: {audio_backend.model_identifier}")
            ConsoleOutput.success("Audio backend initialized.")
            
        # 10. Initialize Pipeline and Inject Backends
        pipeline = ProcessingPipeline(pipeline_config)
        pipeline.llm_backend = text_backend
        pipeline.audio_backend = audio_backend # Pass audio backend to pipeline

        # 11. Collect Input Files
        batch_manager = BatchFileManager(supported_formats=SUPPORTED_FORMATS)
        input_files = batch_manager.collect_input_files(args.input, recursive=args.recursive)

        if not input_files:
            ConsoleOutput.error("No valid input files found at specified paths.")
            return 1
        ConsoleOutput.info(f"Found {len(input_files)} file(s) to process.")

        # 12. Determine Output Path(s) and Run
        ConsoleOutput.section("Starting Processing")
        
        output_base_path: Optional[Path] = None
        output_dir: Optional[Path] = args.output_dir

        if args.output:
            # User specified -o/--output
            p_out = Path(args.output)
            if len(input_files) == 1 and not p_out.is_dir():
                # Single file, and -o is a file path
                output_base_path = p_out
                output_dir = p_out.parent # Use this directory
            else:
                # Batch processing, or -o is a directory
                output_dir = p_out
                output_base_path = None # Let pipeline name files within output_dir
        
        # Update pipeline's output dir
        pipeline.file_handler.output_dir = output_dir.resolve()
        _ensure_directory(pipeline.file_handler.output_dir)

        # 13. Process Files
        results: List[PipelineResult] = []
        overall_success = True

        for i, input_file in enumerate(input_files, 1):
            ConsoleOutput.subsection(f"--- Processing File {i}/{len(input_files)}: {input_file.name} ---")
            
            # For single file, pass the specific output path
            # For batch, pass None so it generates based on input name
            file_output_base = output_base_path if len(input_files) == 1 else None 

            result = pipeline.process_file(input_file, output_path_base=file_output_base)
            results.append(result)
            
            if not result.success:
                overall_success = False
                ConsoleOutput.error(f"Failed to process {input_file.name}: {result.error_message}")
            
            # Unload local models between files in batch mode
            if len(input_files) > 1 and text_backend.provider_identifier.startswith("local"):
                ConsoleOutput.info(f"Unloading model {text_backend.model_specifier}...")
                text_backend.unload()
                pipeline.memory_monitor.check(f"after unloading model from file {i}")


        return 0 if overall_success else 1

    except (ModelLoadError, ConfigurationError, FileProcessingError, MissingDataError) as e:
        ConsoleOutput.error(f"A critical error occurred: {e}")
        logger.critical(f"CLI processing failed: {e}", exc_info=True)
        return 1
    except KeyboardInterrupt:
        ConsoleOutput.warning("\nProcessing interrupted by user.")
        return 130 # Standard exit code for Ctrl+C
    except Exception as e:
        ConsoleOutput.error(f"An unexpected critical error occurred: {e}")
        logger.critical("CLI processing failed", exc_info=True)
        # Attempt to save error context
        if hasattr(logger, 'error'):
             logger.error(f"Critical failure in CLI: {e}", exc_info=True, save_context=True)
        return 1
    finally:
        # Cleanup backends and save final report
        ConsoleOutput.info("Cleaning up resources...")
        if text_backend:
            try: text_backend.unload()
            except Exception as unload_e: logger.warning(f"Error during text backend cleanup: {unload_e}")
        if audio_backend:
            try: audio_backend.unload()
            except Exception as unload_e: logger.warning(f"Error during audio backend cleanup: {unload_e}")
        if pipeline:
            try: pipeline.cleanup_file_handler() # Saves report
            except Exception as cleanup_e: logger.warning(f"Error during pipeline cleanup: {cleanup_e}")
        ConsoleOutput.info("Cleanup complete.")
