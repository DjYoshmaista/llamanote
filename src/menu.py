# llamanote/menu.py
"""
Menu System Module - Refactored
Interactive menu interface for LlamaNote Enhanced
"""

import sys
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# --- LlamaNote Modules ---
from .utils.logger import ConsoleOutput, get_logger_conf, LoggingProgress
from .utils.validators import validate_file_path, validate_directory_path
from .config.manager import ConfigCRUD, ConfigManager
from .config.settings import (
    DEFAULT_MODEL_KEY, SUPPORTED_FORMATS, DEFAULT_PIPELINE_STAGES,
    TIMESTAMP_OUTPUTS, PREPROCESS_PROMPT_PODCAST, DEFAULT_SYSTEM_PROMPT,
    SUPPORTED_LLM_PROVIDERS, SUPPORTED_TTS_PROVIDERS, DEFAULT_CACHE_DIR, DEFAULT_OUTPUT_DIR,
    DEFAULT_GPU_LAYERS, DEFAULT_QUANTIZATION, QUANTIZATION_OPTIONS, CHUNK_SIZE_DEFAULT,
    DEFAULT_CONFIG_DIR, CHUNK_OVERLAP, INCLUDE_METADATA, ENABLE_STAGE_CHECKPOINTS,
    MAX_RETRIES, FALLBACK_ON_ERROR
)
from .config.profiles import list_memory_profiles, create_configs_from_memory_profile, get_memory_profile
from .config.presets import list_hyperparameter_presets, get_hyperparameter_preset, PREPROCESS_PROMPT 

from .core.types import (
    ProcessingMode, PipelineConfig, PipelineResult,
    QuantizationConfig, LayerSplitConfig, AudioConfig, GenerationResult,
    ChunkingStrategy
)
from .core.pipeline import ProcessingPipeline
from .core.errors import ModelLoadError, PipelineError, FileProcessingError

from .models.hub import InteractiveModelBrowser, ModelHub, ModelHubInfo
from .models.backends import (
    get_llm_backend, get_audio_backend, LLMBackend, AudioBackend
)
from .models.registry import get_registry, ModelEntry
from .models.hyperparameters import HyperparameterConfig, InteractiveHyperparameterEditor
from .io.batch_manager import BatchFileManager
from .menu_checkpoint import CheckpointMenuManager
from .utils.file_browser import FileBrowser

# Conditional import for GGUF
try:
    from .models.backends.local_gguf import GGUFModelManager, LLAMACPP_AVAILABLE
except ImportError:
    LLAMACPP_AVAILABLE = False
    GGUFModelManager = None

# Conditional import for torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = get_logger_conf(__name__)

# --- Menu Data Structures ---

class MenuAction(Enum):
    """Actions to control menu flow."""
    BACK = "back"
    EXIT = "exit"
    CONTINUE = "continue"
    RUN = "run"

@dataclass
class MenuItem:
    """Represents an item in a menu."""
    key: str
    label: Union[str, Callable[[], str]] # Allow label to be a function
    action: Optional[Callable] = None
    submenu: Optional['Menu'] = None
    description: Optional[Union[str, Callable[[], str]]] = None # Allow description to be a function

class Menu:
    """A navigable, text-based menu."""
    def __init__(self, title: str, items: List[MenuItem], parent: Optional['Menu'] = None):
        self.title = title
        self.items = items
        self.parent = parent

    def display(self) -> MenuAction:
        """Displays the menu and handles user input."""
        while True:
            ConsoleOutput.header(self.title)
            for item in self.items:
                label_text = item.label() if callable(item.label) else item.label
                desc_text = item.description() if callable(item.description) else item.description
                desc_str = f" - {desc_text}" if desc_text else ""
                print(f"  {item.key}. {label_text}{desc_str}")

            nav_options = []
            if self.parent:
                nav_options.append("(b) Back")
            nav_options.append("(q) Quit")
            print("\n  " + " | ".join(nav_options))
            print("-" * 60)

            choice = input("Select option: ").strip().lower()

            if choice == 'q':
                confirm_quit = input("Are you sure you want to quit? (y/n): ").strip().lower()
                if confirm_quit == 'y':
                    return MenuAction.EXIT
                else:
                    continue
            if choice == 'b' and self.parent:
                return MenuAction.BACK

            selected_item = next((item for item in self.items if item.key.lower() == choice), None)

            if selected_item:
                result = MenuAction.CONTINUE
                if selected_item.submenu:
                    result = selected_item.submenu.display()
                elif selected_item.action:
                    action_result = selected_item.action()
                    result = action_result if isinstance(action_result, MenuAction) else MenuAction.CONTINUE

                if result == MenuAction.EXIT:
                    return MenuAction.EXIT # Propagate EXIT up
                if result == MenuAction.RUN:
                    return MenuAction.RUN # Propagate RUN up
                if result == MenuAction.BACK:
                    continue # Stay in this menu loop
            else:
                ConsoleOutput.warning("Invalid selection. Please try again.")
                time.sleep(1)


# --- Application State ---

@dataclass
class AppState:
    """Holds the current configuration state of the application menu."""
    input_files: List[Path] = field(default_factory=list)
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_filename: Optional[str] = None
    processing_mode: ProcessingMode = ProcessingMode.PODCAST
    text_model_provider: str = "local_hf"
    text_model_specifier: str = DEFAULT_MODEL_KEY # Use key, will be resolved
    audio_model_provider: str = "local_audio"
    audio_model_specifier: str = "microsoft/speecht5_tts"
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    stages_to_run: List[str] = field(default_factory=lambda: list(DEFAULT_PIPELINE_STAGES))
    run_audio_generation: bool = False
    cloud_api_keys: Dict[str, str] = field(default_factory=dict) # Loaded at runtime
    system_prompt: Optional[str] = PREPROCESS_PROMPT # Custom system prompt (if None, use defaults)

    # --- Checkpoint resume data ---
    checkpoint_data: Optional[Dict[str, Any]] = None  # Data from loaded checkpoint
    checkpoint_metadata: Optional[Dict[str, Any]] = None  # Metadata from loaded checkpoint

    # --- Local model compute settings ---
    memory_profile: str = "medium_vram"
    # These are now *derived* from the profile but can be overridden
    quantization: str = "4bit"
    gpu_layers: int = DEFAULT_GPU_LAYERS
    max_gpu_memory: str = "4GiB"
    max_cpu_memory: str = "24GiB"

    # --- Advanced memory optimization settings ---
    layer_split_config: LayerSplitConfig = field(default_factory=LayerSplitConfig)

    def __post_init__(self):
        """Update compute settings from default profile."""
        self.update_compute_settings(self.memory_profile, force=True)
        # Ensure layer_split_config is initialized with defaults
        if not hasattr(self, 'layer_split_config') or self.layer_split_config is None:
            self.layer_split_config = LayerSplitConfig()

    def update_compute_settings(self, profile_name: str, force: bool = False):
        """Updates compute settings based on a memory profile name."""
        if profile_name == self.memory_profile and not force:
            return # No change

        profile = get_memory_profile(profile_name)
        if not profile:
            logger.warning(f"Memory profile '{profile_name}' not found. Using 'medium_vram'.")
            profile = get_memory_profile("medium_vram")
            profile_name = "medium_vram"

        self.memory_profile = profile_name
        self.quantization = profile.quantization_method
        self.gpu_layers = profile.gpu_layers
        self.max_gpu_memory = profile.max_gpu_memory_per_device
        self.max_cpu_memory = profile.max_cpu_memory
        logger.info(f"Set memory profile to '{profile_name}'")


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the app state to a dictionary for saving."""
        data = asdict(self)
        # Convert non-serializable or sensitive types
        data["input_files"] = [str(f) for f in self.input_files]
        data["output_dir"] = str(self.output_dir)
        data["processing_mode"] = self.processing_mode.value
        data.pop("cloud_api_keys", None) # Don't save keys in presets
        # Store nested dataclasses as dicts
        data["hyperparameters"] = self.hyperparameters.to_dict()
        data["audio_config"] = self.audio_config.to_dict()
        # layer_split_config is already converted by asdict, but ensure it's serialized
        return data

    def from_dict(self, data: Dict[str, Any]):
        """Deserializes a dictionary into the app state."""
        try:
            # Preserve values not in preset
            current_inputs = self.input_files
            current_keys = self.cloud_api_keys

            self.input_files = [Path(p) for p in data.get("input_files", [])]
            self.output_dir = Path(data.get("output_dir", str(DEFAULT_OUTPUT_DIR)))
            self.output_filename = data.get("output_filename")
            
            try:
                self.processing_mode = ProcessingMode(data.get("processing_mode", "podcast"))
            except ValueError:
                self.processing_mode = ProcessingMode.PODCAST

            self.text_model_provider = data.get("text_model_provider", "local_hf")
            self.text_model_specifier = data.get("text_model_specifier", DEFAULT_MODEL_KEY)
            self.audio_model_provider = data.get("audio_model_provider", "local_audio")
            self.audio_model_specifier = data.get("audio_model_specifier", "microsoft/speecht5_tts")
            
            self.hyperparameters = HyperparameterConfig(**data.get("hyperparameters", {}))
            self.audio_config = AudioConfig(**data.get("audio_config", {}))

            self.stages_to_run = data.get("stages_to_run", list(DEFAULT_PIPELINE_STAGES))
            self.run_audio_generation = data.get("run_audio_generation", False)
            self.system_prompt = data.get("system_prompt", PREPROCESS_PROMPT)

            # Load compute settings
            self.memory_profile = data.get("memory_profile", "medium_vram")
            self.quantization = data.get("quantization", DEFAULT_QUANTIZATION)
            self.gpu_layers = data.get("gpu_layers", DEFAULT_GPU_LAYERS)
            self.max_gpu_memory = data.get("max_gpu_memory", "10GB")
            self.max_cpu_memory = data.get("max_cpu_memory", "30GB")

            # Load layer_split_config
            layer_split_data = data.get("layer_split_config", {})
            if layer_split_data:
                self.layer_split_config = LayerSplitConfig(**layer_split_data)
            else:
                self.layer_split_config = LayerSplitConfig()

            # Restore preserved values
            self.input_files = current_inputs
            self.cloud_api_keys = current_keys

            ConsoleOutput.success("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading configuration state: {e}", exc_info=True)
            ConsoleOutput.error(f"Error loading configuration: {e}. State may be partial.")

    def get_pipeline_config(self) -> PipelineConfig:
        """Creates a PipelineConfig based on current AppState."""
        logger.info(f"Getting pipeline config. AppState stages_to_run: {self.stages_to_run}")
        
        # 1. Resolve Model Specifier
        # If using a key (like 'qwen3-4b'), resolve it to the full ID
        resolved_specifier = self.text_model_specifier
        model_entry = None
        if self.text_model_provider == "local_hf":
            model_entry = get_registry().find_entry(self.text_model_specifier)
            if model_entry:
                resolved_specifier = model_entry.model_id
            else:
                logger.warning(f"Could not find model key '{self.text_model_specifier}' in registry. Using it as a direct Model ID.")
                resolved_specifier = self.text_model_specifier
        
        # 2. Build Hardware Configs
        # Use the layer_split_config from AppState (which includes advanced memory settings)
        # Get base configs from profile for quantization
        quant_config, _ = create_configs_from_memory_profile(self.memory_profile)
        # Apply quantization override
        quant_config.method = self.quantization

        # Use the state's layer_split_config which has advanced memory optimization settings
        # but also apply profile-based overrides for backwards compatibility
        layer_split_config = self.layer_split_config
        layer_split_config.gpu_layers = self.gpu_layers
        layer_split_config.max_cpu_memory = self.max_cpu_memory
        if layer_split_config.max_gpu_memory: # Update all GPUs if dict exists
            # Ensure keys are integers, not strings
            layer_split_config.max_gpu_memory = {int(i): self.max_gpu_memory for i in layer_split_config.max_gpu_memory.keys()}
        
        # 3. Determine System Prompt
        system_prompt_str = self.system_prompt # Check if user set one
        if not system_prompt_str:
            system_prompt_str = PREPROCESS_PROMPT_PODCAST if self.processing_mode == ProcessingMode.PODCAST else DEFAULT_SYSTEM_PROMPT

        # 4. Create PipelineConfig
        return PipelineConfig(
            mode=self.processing_mode,
            model_provider=self.text_model_provider,
            model_specifier=resolved_specifier,
            system_prompt=system_prompt_str,
            
            output_format=self.audio_config.output_format if self.run_audio_generation else "markdown",
            output_dir=self.output_dir,
            timestamp_outputs=TIMESTAMP_OUTPUTS,
            include_metadata=INCLUDE_METADATA,

            chunking_strategy=ChunkingStrategy.WORD_BOUNDARY, # TODO: Make configurable
            chunk_size=model_entry.optimal_chunk_size if model_entry else CHUNK_SIZE_DEFAULT, # TODO: Make configurable
            chunk_overlap=CHUNK_OVERLAP, # TODO: Make configurable
            
            preserve_pdf_layout=True, # TODO: Make configurable
            clean_for_audio=(self.processing_mode == ProcessingMode.PODCAST),
            add_emotions=(self.processing_mode == ProcessingMode.PODCAST),

            hyperparameters=self.hyperparameters,
            quantization_config=quant_config,
            layer_split_config=layer_split_config,
            
            generate_audio=self.run_audio_generation,
            audio_provider=self.audio_model_provider,
            audio_specifier=self.audio_model_specifier,
            audio_config=self.audio_config,
            
            enable_checkpoints=ENABLE_STAGE_CHECKPOINTS,
            max_retries=MAX_RETRIES,
            fallback_on_error=FALLBACK_ON_ERROR
        )

# --- Main Menu System Class ---

class MenuSystem:
    def __init__(self):
        self.state = AppState()
        self.config_manager = ConfigManager(base_dir=DEFAULT_CONFIG_DIR)
        self.model_hub = ModelHub(cache_dir=DEFAULT_CACHE_DIR / "model_hub")
        self.registry = get_registry() # Get singleton instance
        self.file_manager = BatchFileManager(supported_formats=SUPPORTED_FORMATS)
        self.checkpoint_manager = CheckpointMenuManager()  # Add checkpoint manager

        # Load API keys on startup
        self.state.cloud_api_keys = self.config_manager.load_cloud_keys()
        logger.info(f"Loaded API keys for: {list(self.state.cloud_api_keys.keys())}")

        self.main_menu = self._build_main_menu()
        logger.info("Initialized MenuSystem")

    def run(self):
        """Starts the main menu loop."""
        action = MenuAction.CONTINUE
        while action != MenuAction.EXIT:
             action = self.main_menu.display()
             
             if action == MenuAction.RUN:
                 run_success = self._execute_pipeline()
                 if run_success:
                      ConsoleOutput.success("\nPipeline finished successfully.")
                 else:
                      ConsoleOutput.error("\nPipeline finished with errors.")
                 input("Press Enter to return to the main menu...")
                 action = MenuAction.CONTINUE # Go back to main menu
                 
        ConsoleOutput.info("Exiting LlamaNote.")

    # --- Build Menus ---

    def _build_main_menu(self) -> Menu:
        """Builds the top-level menu."""
        items = [
            MenuItem("1", 
                     "Select Input & Stages", 
                     self._select_input_menu,
                     description=lambda: f"Files: {len(self.state.input_files)}, Audio: {'ON' if self.state.run_audio_generation else 'OFF'}"),
            MenuItem("2", 
                     "Model Settings", 
                     submenu=self._build_model_settings_menu(), 
                     description=lambda: f"Txt: {self.state.text_model_specifier} | Audio: {self.state.audio_model_specifier}"),
            MenuItem("3", 
                     "Cloud API Keys", 
                     submenu=self._build_cloud_settings_menu(), 
                     description=lambda: f"Keys loaded for: {', '.join(self.state.cloud_api_keys.keys()) or 'None'}"),
            MenuItem("4", 
                     "Output Settings", 
                     self._output_settings_menu, 
                     description=lambda: f"Dir: ...{str(self.state.output_dir)[-30:]}"),
            MenuItem("5", 
                     "Load/Save Preset",
                     submenu=self._build_preset_menu()),
            MenuItem("6",
                     "View Current Setup",
                     self._view_current_setup,
                     description="Review all settings"),
            MenuItem("7",
                     "Checkpoint Management",
                     self._checkpoint_management_action,
                     description="Manage pipeline checkpoints"),
            MenuItem("R",
                     "RUN PIPELINE",
                     self._confirm_and_run,
                     description="Start processing"),
        ]
        
        main_menu = Menu("LlamaNote Enhanced - Main Menu", items, parent=None)
        
        # Assign parents to submenus
        for item in items:
            if item.submenu:
                item.submenu.parent = main_menu
                
        return main_menu

    def _build_model_settings_menu(self) -> Menu:
        """Builds the menu for configuring models."""
        items = [
            MenuItem(
                "1",
                "Set Text Model (LLM)",
                self._set_text_model_menu,
                description=lambda: f"Provider: {self.state.text_model_provider} | Model: {self.state.text_model_specifier}"
            ),
            MenuItem(
                "2",
                "Set Local Hardware Profile",
                self._set_local_hardware_profile,
                description=lambda: f"Profile: {self.state.memory_profile} (Quant: {self.state.quantization}, GPU Layers: {self.state.gpu_layers})"
            ),
            MenuItem(
                "3",
                "Advanced Memory Optimization",
                self._memory_optimization_menu,
                description="Configure CPU/disk offloading, OOM handling, cache management"
            ),
            MenuItem(
                "4",
                "Set Text Hyperparameters",
                self._set_text_hyperparameters,
                description="e.g., Temperature, Top-p, Max Tokens"
            ),
            MenuItem(
                "5",
                "Set Audio Model (TTS)",
                self._set_audio_model_menu,
                description=lambda: f"Provider: {self.state.audio_model_provider} | Model: {self.state.audio_model_specifier}"
            ),
            MenuItem(
                "6",
                "Configure Audio Settings",
                self._set_audio_config,
                description="e.g., Voice, Speed, Format"
            ),
        ]
        return Menu("Model Settings", items)

    def _build_cloud_settings_menu(self) -> Menu:
        """Builds the menu for managing API keys."""
        items = []
        all_providers = sorted(list(set(SUPPORTED_LLM_PROVIDERS + SUPPORTED_TTS_PROVIDERS) - {"local_hf", "local_gguf", "local_audio"}))
        
        for i, provider in enumerate(all_providers):
            # Clean name for display (e.g., 'openai_audio' -> 'OpenAI (Audio)')
            provider_name = provider.replace('_', ' ').title()
            
            items.append(MenuItem(
                key=str(i + 1),
                label=lambda p=provider, pn=provider_name: f"Set/Update {pn} Key",
                description=lambda p=provider: "✓ Set" if p in self.state.cloud_api_keys else "X Not Set",
                action=lambda p=provider: self._set_provider_key(p)
            ))
            
        items.append(MenuItem("V", "View Configured Keys", self._view_keys))
        return Menu("Cloud Provider API Keys", items)

    def _build_preset_menu(self) -> Menu:
        """Builds the menu for loading/saving configuration presets."""
        items = [
            MenuItem("1", "Load Preset", self._load_configuration, 
                     description=f"Load settings from {self.config_manager.get_dir('preset').name}"),
            MenuItem("2", "Save Current Settings", self._save_configuration,
                     description="Save all settings (except files/keys)"),
        ]
        return Menu("Load/Save Configuration", items)

    # --- Menu Action Implementations ---

    def _select_input_menu(self):
        """Chain of actions: Select Files -> Select Stages."""
        action = self._select_input_files()
        if action == MenuAction.BACK:
            return MenuAction.CONTINUE # Stay in main menu
        
        action = self._select_stages_menu()
        if action == MenuAction.BACK:
            return MenuAction.CONTINUE # Stay in main menu
            
        return MenuAction.CONTINUE

    def _select_input_files(self):
        """Handles selection of input file(s) with enhanced file browser and checkpoint support."""
        while True:
            ConsoleOutput.section("Select Input Source")
            print(f"Current selection: {len(self.state.input_files)} file(s)")
            print("\nOptions:")
            print("  1. Browse by File Type (PDF, Text, Markdown, Transcript, Checkpoint)")
            print("  2. Enter Path(s) Manually")
            print("  3. Load from Checkpoint")
            print("  4. Clear File List")
            print("  B. Back")
            print("-" * 60)

            choice = input("Select option: ").strip().upper()

            if choice == 'B':
                return MenuAction.BACK
            elif choice == '1':
                result = FileBrowser.select_file_type_and_browse()
                if result:
                    file_path, file_type = result
                    # Handle different file types
                    if file_type == 'checkpoint':
                        self._load_from_checkpoint_file(file_path)
                    else:
                        self._add_file_and_auto_populate_stages(file_path, file_type)
                    input("\nPress Enter to continue...")
            elif choice == '2':
                self._manual_path_entry()
            elif choice == '3':
                self._browse_and_load_checkpoint()
            elif choice == '4':
                self.state.input_files = []
                ConsoleOutput.info("Input file list cleared.")
                input("\nPress Enter to continue...")
            else:
                ConsoleOutput.error("Invalid option")

            # After any action, ask if user wants to continue or go back
            if len(self.state.input_files) > 0:
                continue_choice = input("\nAdd more files? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    return MenuAction.CONTINUE

    def _manual_path_entry(self):
        """Manual path entry (original behavior)."""
        print("\nEnter one or more file paths or directory paths, separated by commas.")
        path_input = input("Path(s): ").strip()

        if not path_input:
            ConsoleOutput.warning("No input provided.")
            return

        paths = [p.strip().strip('"\'') for p in path_input.split(',')]

        # Ask about recursive search if any directory is given
        recursive = False
        if any(Path(os.path.expanduser(p)).is_dir() for p in paths):
             rec_choice = input("Search directories recursively? (y/n) [y]: ").strip().lower()
             recursive = (rec_choice != 'n')

        try:
            selected_files = self.file_manager.collect_input_files(paths, recursive=recursive)
        except Exception as e:
            ConsoleOutput.error(f"Error collecting files: {e}")
            logger.error(f"File collection failed: {e}", exc_info=True)
            return

        if not selected_files:
            ConsoleOutput.warning("No supported files found at the specified path(s).")
            return

        # Merge with existing list, ensuring no duplicates
        new_files_added = 0
        current_set = set(self.state.input_files)
        for f in selected_files:
            if f not in current_set:
                self.state.input_files.append(f)
                current_set.add(f)
                new_files_added += 1

        ConsoleOutput.success(f"Added {new_files_added} new file(s).")
        ConsoleOutput.info(f"Total files to process: {len(self.state.input_files)}")

    def _add_file_and_auto_populate_stages(self, file_path: Path, file_type: str):
        """Add file and auto-populate stages based on file type."""
        # Add file to list
        if file_path not in self.state.input_files:
            self.state.input_files.append(file_path)
            ConsoleOutput.success(f"Added: {file_path.name}")
        else:
            ConsoleOutput.info(f"File already in list: {file_path.name}")

        # Auto-populate stages based on file type
        if file_type in ['pdf', 'text', 'markdown']:
            # Full pipeline
            self.state.stages_to_run = list(DEFAULT_PIPELINE_STAGES)
            ConsoleOutput.info("📋 Auto-populated all stages (full pipeline)")
        elif file_type == 'transcript':
            # Only format, save, and audio
            self.state.stages_to_run = ['format', 'save']
            self.state.run_audio_generation = True
            ConsoleOutput.info("📋 Auto-populated stages: Format, Save, Audio")

        ConsoleOutput.info(f"Total files: {len(self.state.input_files)}")

    def _browse_and_load_checkpoint(self):
        """Browse checkpoints and load selected one."""
        result = self.checkpoint_manager.browse_checkpoints_menu()
        if result:
            checkpoint_path, metadata, data = result
            self._load_checkpoint_data(checkpoint_path, metadata, data)

    def _load_from_checkpoint_file(self, checkpoint_path: Path):
        """Load data from a checkpoint file."""
        result = self.checkpoint_manager.checkpoint_manager.load(checkpoint_path)
        if not result:
            ConsoleOutput.error(f"Failed to load checkpoint: {checkpoint_path.name}")
            return

        metadata, data = result
        self._load_checkpoint_data(checkpoint_path, metadata, data)

    def _load_checkpoint_data(self, checkpoint_path: Path, metadata: Dict[str, Any], data: Dict[str, Any]):
        """Load checkpoint data into the current state."""
        ConsoleOutput.success(f"Loaded checkpoint: {checkpoint_path.name}")

        # Get input file from metadata
        input_file_str = metadata.get('input_file')
        if input_file_str:
            input_file = Path(input_file_str)
            if input_file.exists():
                if input_file not in self.state.input_files:
                    self.state.input_files.append(input_file)
            else:
                ConsoleOutput.warning(f"Original input file not found: {input_file}")

        # Determine which stages are complete and which need to run
        stage = metadata.get('stage')
        stage_order = ["extract", "preprocess", "chunk", "process", "filter", "format", "save", "audio"]

        try:
            stage_idx = stage_order.index(stage)
            # Set stages to run from the next stage onwards
            remaining_stages = stage_order[stage_idx + 1:]
            self.state.stages_to_run = [s for s in remaining_stages if s != "audio"]

            # Check if audio was completed
            if 'audio_result' in data and stage == 'audio':
                self.state.run_audio_generation = False
                ConsoleOutput.info("✅ Audio already generated")
            elif 'audio' in remaining_stages:
                self.state.run_audio_generation = True

            ConsoleOutput.info(f"📋 Checkpoint at stage: {stage}")
            ConsoleOutput.info(f"📋 Remaining stages: {', '.join(self.state.stages_to_run)}")

        except ValueError:
            ConsoleOutput.warning(f"Unknown stage: {stage}")

        # Load configuration from checkpoint
        config = metadata.get('config', {})

        # Load model settings
        text_model = config.get('text_model', '')
        if ':' in text_model:
            provider, model = text_model.split(':', 1)
            self.state.text_model_provider = provider
            self.state.text_model_specifier = model
            ConsoleOutput.info(f"📝 Loaded text model: {text_model}")

        # Load mode
        mode_str = config.get('mode')
        if mode_str:
            try:
                self.state.processing_mode = ProcessingMode(mode_str)
            except:
                pass

        # Load output format
        output_fmt = config.get('output_format')
        if output_fmt:
            self.state.output_format = output_fmt

        # Store checkpoint data for pipeline resume
        self.state.checkpoint_data = data
        self.state.checkpoint_metadata = metadata

        ConsoleOutput.success("Checkpoint loaded successfully!")


    def _select_stages_menu(self):
        """Allows user to toggle which pipeline stages run."""
        all_text_stages = list(DEFAULT_PIPELINE_STAGES)

        while True:
            ConsoleOutput.header("Select Processing Stages")
            print("Text Processing Stages:")
            for i, stage in enumerate(all_text_stages):
                 included = "[X]" if stage in self.state.stages_to_run else "[ ]"
                 print(f"  {i+1}. {included} {stage.capitalize()}")

            audio_included = "[X]" if self.state.run_audio_generation else "[ ]"
            # Audio can now work without save stage (reads from .txt/.md files directly)
            has_text_stages = bool(self.state.stages_to_run)
            audio_note = "" if has_text_stages else " (will read from input file)"
            print(f"\nAudio Generation Stage:")
            print(f"  A. {audio_included} Generate Audio{audio_note}")

            print("\nOptions:")
            print(f"  Enter number (1-{len(all_text_stages)}) or 'A' to toggle a stage.")
            print("  'all'  - Select all stages (Text + Audio)")
            print("  'text' - Select text stages only")
            print("  'none' - Deselect all stages")
            print("  'b'    - Back to Main Menu")
            print("-" * 60)

            choice = input("Toggle stage or command: ").strip().lower()

            if choice == 'b':
                # Final check before leaving
                # Audio no longer strictly requires save stage - it can read .txt/.md files directly
                if not self.state.stages_to_run and not self.state.run_audio_generation:
                     ConsoleOutput.warning("No stages are selected. Nothing will be processed.")
                     confirm = input("Are you sure you want to continue? (y/n) [n]: ").strip().lower()
                     if confirm != 'y':
                         continue # Stay in this menu
                return MenuAction.BACK # Go back to main menu
            
            elif choice == 'all':
                self.state.stages_to_run = list(DEFAULT_PIPELINE_STAGES)
                self.state.run_audio_generation = True
            elif choice == 'text':
                self.state.stages_to_run = list(DEFAULT_PIPELINE_STAGES)
                self.state.run_audio_generation = False
            elif choice == 'none':
                self.state.stages_to_run = []
                self.state.run_audio_generation = False
            elif choice == 'a':
                self.state.run_audio_generation = not self.state.run_audio_generation
                status = "ENABLED" if self.state.run_audio_generation else "DISABLED"
                ConsoleOutput.info(f"Audio Generation {status}.")
            elif choice.isdigit():
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_text_stages):
                        stage_name = all_text_stages[idx]
                        if stage_name in self.state.stages_to_run:
                            self.state.stages_to_run.remove(stage_name)
                            ConsoleOutput.info(f"Stage '{stage_name}' disabled.")
                        else:
                            # Add stage back in, maintaining order
                            new_stages = []
                            stages_set = set(self.state.stages_to_run)
                            stages_set.add(stage_name)
                            for s in DEFAULT_PIPELINE_STAGES:
                                if s in stages_set:
                                    new_stages.append(s)
                            self.state.stages_to_run = new_stages
                            ConsoleOutput.info(f"Stage '{stage_name}' enabled.")
                    else:
                        ConsoleOutput.warning("Invalid stage number.")
                except ValueError:
                    ConsoleOutput.warning("Invalid input.")
            else:
                ConsoleOutput.warning("Invalid command.")


    def _set_provider_key(self, provider: str):
        """Sets or clears an API key for a given provider."""
        provider_name = provider.replace('_', ' ').title()
        ConsoleOutput.section(f"Set API Key for {provider_name}")
        current_key = self.state.cloud_api_keys.get(provider)
        
        if current_key:
            masked_key = '*' * (len(current_key) - 4) + current_key[-4:] if len(current_key) > 4 else '*' * len(current_key)
            print(f"Current key: {masked_key}")
        else:
            print("Current key: Not set")

        new_key = input(f"Enter new API key for {provider_name} (leave blank to clear, 'keep' to exit): ").strip()

        if new_key.lower() == 'keep':
            return MenuAction.CONTINUE

        final_key = new_key if new_key else None # Use None to signal clearing

        if self.config_manager.cloud_keys.save_key(provider, final_key):
             # Update runtime state immediately after successful save
             self.state.cloud_api_keys = self.config_manager.cloud_keys.load_active_keys()
             action = "cleared" if final_key is None else "saved"
             ConsoleOutput.success(f"API key for {provider_name} {action}.")
        else:
             ConsoleOutput.error(f"Failed to save/clear key for {provider_name}.")

        input("Press Enter to continue...")
        return MenuAction.CONTINUE


    def _view_keys(self):
        """Displays which providers have keys configured."""
        ConsoleOutput.section("Configured API Keys")
        # Reload from file to show the latest saved state
        configured_keys = self.config_manager.cloud_keys.load_active_keys()
        self.state.cloud_api_keys = configured_keys # Sync runtime state

        if not configured_keys:
            print("No API keys are currently configured in the file.")
            print(f"(Config file location: {self.config_manager.cloud_keys.keys_file})")
        else:
            print("Providers with keys configured:")
            for provider, key in configured_keys.items():
                 # Mask the key for display
                 masked_key = '*' * (len(key) - 4) + key[-4:] if len(key) > 4 else '*' * len(key)
                 provider_name = provider.replace('_', ' ').title()
                 print(f"  - {provider_name:<16}: {masked_key}")
            print(f"\nKeys loaded from: {self.config_manager.cloud_keys.keys_file}")

        input("\nPress Enter to continue...")
        return MenuAction.CONTINUE

    # --- Model Selection (Refined) ---
    def _set_text_model_menu(self):
         """Top-level menu to choose between local and cloud text models."""
         while True:
            ConsoleOutput.section("Set Text Processing Model")
            print(f"  1. Use Local Model (HF Transformers) {self.state.text_model_provider == 'local_hf' and '[Current]' or ''}")
            print(f"  2. Use Local Model (GGUF) {self.state.text_model_provider == 'local_gguf' and '[Current]' or ''}")
            print(f"  3. Use Cloud Model (OpenAI) {self.state.text_model_provider == 'openai' and '[Current]' or ''}")
            print(f"  4. Use Cloud Model (Anthropic) {self.state.text_model_provider == 'anthropic' and '[Current]' or ''}")
            print(f"  5. Use Cloud Model (Google) {self.state.text_model_provider == 'google' and '[Current]' or ''}")
            print("\n  b. Back to Model Settings Menu")
            print("-" * 60)

            choice = input("Select option (1-5, or b): ").strip().lower()
            
            result = None
            if choice == '1': result = self._select_local_hf_model()
            elif choice == '2': result = self._select_local_gguf_model()
            elif choice == '3': result = self._select_cloud_model("openai")
            elif choice == '4': result = self._select_cloud_model("anthropic")
            elif choice == '5': result = self._select_cloud_model("google")
            elif choice == 'b': return MenuAction.BACK 
            else:
                ConsoleOutput.warning("Invalid choice.")
                continue

            if result == MenuAction.CONTINUE: # A model was successfully selected
                return MenuAction.CONTINUE # Return to Model Settings menu
            # If result was BACK, the loop continues

    def _select_local_hf_model(self) -> MenuAction:
        """Handles selection of a local HF Transformers model."""
        while True:
            ConsoleOutput.subsection("Select Local Hugging Face Model")
            print("  1. Select from Predefined List")
            print("  2. Select from Cached Models")
            print("  3. Search Hugging Face Hub")
            print("  4. Enter Model ID Manually")
            print(f"\nCurrent: {self.state.text_model_specifier if self.state.text_model_provider == 'local_hf' else 'None'}")
            print("\n  b. Back")
            print("-" * 60)
            
            choice = input("Select option: ").strip().lower()
            model_entry: Optional[ModelEntry] = None

            if choice == 'b': return MenuAction.BACK

            elif choice == '1': # Predefined
                model_entry = self._select_from_model_list(
                    self.registry.list_models(is_predefined=True, sort_by="name"),
                    "Predefined Models"
                )
            elif choice == '2': # Cached
                 model_entry = self._select_from_model_list(
                    self.registry.list_models(is_cached=True, sort_by="name"),
                    "Cached Models"
                )
            elif choice == '3': # Search
                browser = InteractiveModelBrowser(self.model_hub)
                hf_info: Optional[ModelHubInfo] = browser.browse(task="text-generation", library="transformers")
                if hf_info:
                    # Add/update registry and get the ModelEntry
                    self.registry.add_models(ModelEntry.from_model_info(hf_info), save=True)
                    model_entry = self.registry.get_model(hf_info.model_id)
            elif choice == '4': # Manual
                 model_id = input("Enter Hugging Face Model ID (e.g., 'Org/ModelName'): ").strip()
                 if model_id:
                     model_entry = self.registry.find_entry(model_id)
                     if not model_entry:
                         # Not in registry, try to fetch from hub
                         ConsoleOutput.info(f"Fetching info for '{model_id}'...")
                         hf_info = self.model_hub.get_model_info(model_id)
                         if hf_info:
                             ConsoleOutput.success(f"Found model: {hf_info.display_name}")
                             self.registry.add_models(ModelEntry.from_model_info(hf_info), save=True)
                             model_entry = self.registry.get_model(hf_info.model_id)
                         else:
                             ConsoleOutput.error(f"Could not find model '{model_id}' on Hugging Face Hub.")
                             input("Press Enter...")
                 
            else:
                 ConsoleOutput.warning("Invalid selection.")
                 continue

            if model_entry:
                 return self._confirm_model_selection("local_hf", model_entry) # Returns CONTINUE or BACK


    def _select_local_gguf_model(self) -> MenuAction:
        """Handles selection of a local GGUF model."""
        if not LLAMACPP_AVAILABLE:
            ConsoleOutput.error("llama-cpp-python is not installed.")
            ConsoleOutput.info("Please install it (e.g., `pip install llama-cpp-python`) to use GGUF models.")
            input("Press Enter to continue...")
            return MenuAction.BACK

        manager = GGUFModelManager()
        
        while True:
            ConsoleOutput.subsection("Select Local GGUF Model")
            print("  1. Select from Cached GGUF Models")
            print("  2. Enter GGUF File Path Manually")
            print("  3. Download GGUF Model from Hugging Face")
            current_model = Path(self.state.text_model_specifier).name if self.state.text_model_provider == 'local_gguf' else 'None'
            print(f"\nCurrent: {current_model}")
            print(f"(Cache Dir: {manager.cache_dir})")
            print("\n  b. Back")
            print("-" * 60)
            
            choice = input("Select option: ").strip().lower()
            model_path: Optional[Path] = None

            if choice == 'b': return MenuAction.BACK

            elif choice == '1': # Select from cache
                gguf_models = manager.list_available_models()
                if not gguf_models:
                    ConsoleOutput.warning("No GGUF models found in cache.")
                    input("Press Enter...")
                    continue
                print("\nAvailable GGUF models:")
                model_map = {}
                for i, p in enumerate(gguf_models):
                     info = manager.get_model_info(p)
                     print(f"  {i+1}. {info['name']} ({info['size_mb']:.1f} MB)")
                     model_map[str(i+1)] = p
                
                model_choice = input(f"Select model (1-{len(gguf_models)} or b): ").strip().lower()
                if model_choice == 'b': continue
                if model_choice in model_map:
                    model_path = model_map[model_choice]
                else:
                     ConsoleOutput.warning("Invalid number.")

            elif choice == '2': # Manual path
                gguf_path_str = input("Enter full path to .gguf model file: ").strip().strip('"\'')
                path_obj = Path(os.path.expanduser(gguf_path_str))
                is_valid, msg = validate_file_path(path_obj, allowed_extensions=['.gguf'])
                if is_valid:
                    model_path = path_obj
                else:
                    ConsoleOutput.error(f"Invalid path: {msg}")
                    input("Press Enter...")
            
            elif choice == '3': # Download
                 repo_id = input("Enter Hugging Face Repo ID (e.g., 'QuantFactory/Meta-Llama-3.1-8B-Instruct-GGUF'): ").strip()
                 file_name = input("Enter *exact* GGUF filename (e.g., 'Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf'): ").strip()
                 if repo_id and file_name:
                      model_path = manager.download_model_from_hf(repo_id, file_name)
                      if not model_path:
                           ConsoleOutput.error("Download failed. Please check repository and file name.")
                           input("Press Enter...")
                 else:
                      ConsoleOutput.warning("Repo ID and filename are required.")

            else:
                 ConsoleOutput.warning("Invalid selection.")
                 continue

            if model_path:
                # GGUF models don't use ModelEntry, just the path
                return self._confirm_model_selection("local_gguf", str(model_path.resolve()))
                
    def _select_cloud_model(self, provider: str) -> MenuAction:
        """Handles selection of a specific cloud model."""
        if provider not in self.state.cloud_api_keys:
            ConsoleOutput.warning(f"No API key set for {provider.title()}.")
            print("Please add a key in the 'Cloud API Keys' menu first.")
            input("Press Enter to continue...")
            return MenuAction.BACK # Go back to model type menu

        ConsoleOutput.subsection(f"Set {provider.title()} Model")
        example = "e.g., "
        if provider == "openai": example += "gpt-4o"
        elif provider == "anthropic": example += "claude-3-5-sonnet-20240620"
        elif provider == "google": example += "models/gemini-1.5-pro-latest"
        
        print(f"Current: {self.state.text_model_specifier if self.state.text_model_provider == provider else 'None'}")
        model_specifier = input(f"Enter model name/ID ({example}): ").strip()

        if not model_specifier:
            ConsoleOutput.warning("No model name entered. Selection cancelled.")
            input("Press Enter...")
            return MenuAction.BACK # Go back to model type menu
        
        # We don't verify cloud model names here, just accept the string
        return self._confirm_model_selection(provider, model_specifier)


    def _confirm_model_selection(self, provider: str, specifier_or_entry: Union[str, ModelEntry]) -> MenuAction:
        """Centralizes setting the model and resetting hyperparameters."""
        
        old_provider = self.state.text_model_provider
        old_specifier = self.state.text_model_specifier
        
        new_specifier: str
        new_hp = HyperparameterConfig() # Start with defaults

        if isinstance(specifier_or_entry, ModelEntry):
            model_entry = specifier_or_entry
            new_specifier = model_entry.model_id
            # Use model-specific defaults
            new_hp.temperature = model_entry.temperature
            new_hp.top_p = model_entry.top_p
            new_hp.max_new_tokens = model_entry.max_new_tokens
            ConsoleOutput.success(f"Text model set to: {provider}: {model_entry.name} ({new_specifier})")
        else:
            new_specifier = specifier_or_entry
            ConsoleOutput.success(f"Text model set to: {provider}: {new_specifier}")

        self.state.text_model_provider = provider
        self.state.text_model_specifier = new_specifier

        # Ask to reset hyperparams only if the model *actually* changed
        if old_provider != provider or old_specifier != new_specifier:
             reset_choice = input(f"Reset hyperparameters to model/preset defaults? (y/n) [y]: ").strip().lower()
             if reset_choice != 'n':
                 self.state.hyperparameters = new_hp
                 ConsoleOutput.info("Hyperparameters reset.")
             else:
                 ConsoleOutput.info("Keeping existing hyperparameter settings.")
                 
        input("Press Enter to continue...")
        return MenuAction.CONTINUE # Signal success

    def _select_from_model_list(self, model_list: List[ModelEntry], title: str) -> Optional[ModelEntry]:
         """Generic helper to display a list of ModelEntry items and get user selection."""
         if not model_list:
             ConsoleOutput.warning(f"No models found in '{title}'.")
             input("Press Enter...")
             return None

         print(f"\n--- {title} ---")
         model_map = {}
         for i, entry in enumerate(model_list, 1):
             print(f"  {i}. {entry.name} (by {entry.author})")
             print(f"     ID: {entry.model_id}")
             model_map[str(i)] = entry
         print("\n  b. Back")
         
         choice = input(f"Select model (1-{len(model_list)} or b): ").strip().lower()
         
         if choice == 'b': return None
         if choice in model_map:
             return model_map[choice]
         else:
             ConsoleOutput.warning("Invalid selection.")
             return None


    def _set_local_hardware_profile(self):
        """Interactive configuration for local model hardware settings."""
        if self.state.text_model_provider not in ["local_hf", "local_gguf"]:
             ConsoleOutput.warning("Hardware settings only apply to local (HF or GGUF) models.")
             input("Press Enter...")
             return MenuAction.CONTINUE

        while True:
            ConsoleOutput.header("Local Hardware Settings")
            print(f"Current Profile: {self.state.memory_profile}")
            print(f"  - Quantization: {self.state.quantization}")
            print(f"  - GPU Layers (GGUF): {self.state.gpu_layers}")
            print(f"  - Max GPU Mem (HF): {self.state.max_gpu_memory}")
            print(f"  - Max CPU Mem (HF): {self.state.max_cpu_memory}")
            print("-" * 60)
            
            print("Select a profile (recommended):")
            profiles = list_memory_profiles()
            profile_map = {}
            for i, (name, desc) in enumerate(profiles, 1):
                 print(f"  {i}. {name}: {desc}")
                 profile_map[str(i)] = name
            
            print("\nOr, customize individual settings (advanced):")
            is_gguf = (self.state.text_model_provider == 'local_gguf')
            
            if is_gguf:
                print("  G. GPU Layers (-1=all, 0=CPU)")
            else: # Transformers
                print("  Q. Quantization")
                print("  M. Max GPU Memory (per device)")
                print("  C. Max CPU Memory (offload)")

            print("\n  b. Back to Model Settings")
            print("-" * 60)
            
            choice = input("Select profile, option, or 'b' to go back: ").strip().lower()

            try:
                if choice == 'b':
                    return MenuAction.BACK
                
                # Profile selection
                if choice in profile_map:
                    profile_name = profile_map[choice]
                    self.state.update_compute_settings(profile_name, force=True)
                    ConsoleOutput.success(f"Hardware profile set to '{profile_name}'.")

                # Individual setting overrides
                elif choice == 'g' and is_gguf:
                    val = input(f"Enter number of GPU layers (-1 for auto/all, 0 for CPU) [current: {self.state.gpu_layers}]: ").strip()
                    if val: self.state.gpu_layers = int(val)
                
                elif choice == 'q' and not is_gguf:
                    val = input(f"Enter quantization ({', '.join(QUANTIZATION_OPTIONS)}) [current: {self.state.quantization}]: ").strip().lower()
                    if val in QUANTIZATION_OPTIONS:
                        self.state.quantization = val
                    elif val:
                         ConsoleOutput.warning(f"Invalid option. Must be one of: {', '.join(QUANTIZATION_OPTIONS)}")

                elif choice == 'm' and not is_gguf:
                     val = input(f"Enter Max GPU Memory per GPU (e.g., 8GiB, 10GB) [current: {self.state.max_gpu_memory}]: ").strip()
                     if val:
                         if re.match(r"^\d+(\.\d+)?(GB|GiB|MB|MiB)$", val, re.IGNORECASE):
                             self.state.max_gpu_memory = val
                         else:
                             ConsoleOutput.warning("Invalid format. Use numbers followed by GB, GiB, MB, or MiB.")

                elif choice == 'c' and not is_gguf:
                    val = input(f"Enter Max CPU Memory (e.g., 30GB) [current: {self.state.max_cpu_memory}]: ").strip()
                    if val:
                         if re.match(r"^\d+(\.\d+)?(GB|GiB|MB|MiB)$", val, re.IGNORECASE):
                             self.state.max_cpu_memory = val
                         else:
                             ConsoleOutput.warning("Invalid format. Use numbers followed by GB, GiB, MB, or MiB.")
                
                else:
                    ConsoleOutput.warning("Invalid selection.")

            except ValueError:
                ConsoleOutput.warning("Invalid input value type (e.g., expected a number).")
            except Exception as e:
                 ConsoleOutput.error(f"Error updating setting: {e}")

    def _memory_optimization_menu(self):
        """Interactive menu for advanced memory optimization settings."""
        while True:
            ConsoleOutput.header("Advanced Memory Optimization Settings")
            print("These settings apply to local models (HF and Audio)\n")

            # Get current settings from state
            lsc = self.state.layer_split_config
            ac = self.state.audio_config

            print("Current Settings:")
            print(f"  CPU Offloading:         {'ENABLED' if lsc.enabled else 'DISABLED'}")
            print(f"  Auto OOM Handling:      {'ENABLED' if lsc.auto_oom_handling else 'DISABLED'}")
            print(f"  Max GPU Memory:         {list(lsc.max_gpu_memory.values())[0] if lsc.max_gpu_memory else 'N/A'}")
            print(f"  Max CPU Memory:         {lsc.max_cpu_memory}")
            print(f"  Disk Offloading:        {'ENABLED' if ac.enable_disk_offload else 'DISABLED'}")
            print(f"  Cache Clearing (Audio): {'ENABLED' if ac.clear_cache_between_chunks else 'DISABLED'}")
            print(f"  Low CPU Mem Mode:       {'ENABLED' if lsc.low_cpu_mem_usage else 'DISABLED'}")
            print("-" * 70)

            print("\nConfiguration Options:")
            print("  1. Toggle CPU Offloading")
            print("  2. Toggle Auto OOM Handling (Progressive Layer Offloading)")
            print("  3. Set Max GPU Memory")
            print("  4. Set Max CPU Memory")
            print("  5. Toggle Disk Offloading")
            print("  6. Toggle Cache Clearing Between Audio Chunks")
            print("  7. Toggle Low CPU Memory Mode")
            print("  8. Reset to Recommended Defaults (4GB VRAM + 28GB RAM)")
            print("\n  b. Back to Model Settings")
            print("-" * 70)

            choice = input("Select option: ").strip().lower()

            try:
                if choice == 'b':
                    return MenuAction.BACK

                elif choice == '1':
                    lsc.enabled = not lsc.enabled
                    ConsoleOutput.success(f"CPU Offloading {'ENABLED' if lsc.enabled else 'DISABLED'}")

                elif choice == '2':
                    lsc.auto_oom_handling = not lsc.auto_oom_handling
                    ConsoleOutput.success(f"Auto OOM Handling {'ENABLED' if lsc.auto_oom_handling else 'DISABLED'}")
                    if lsc.auto_oom_handling:
                        ConsoleOutput.info("The system will automatically reduce GPU memory allocation if CUDA OOM errors occur")

                elif choice == '3':
                    current_val = list(lsc.max_gpu_memory.values())[0] if lsc.max_gpu_memory else "4GB"
                    val = input(f"Enter Max GPU Memory (e.g., 4GB, 3.5GB) [current: {current_val}]: ").strip()
                    if val:
                        if re.match(r"^\d+(\.\d+)?(GB|GiB|MB|MiB)$", val, re.IGNORECASE):
                            lsc.max_gpu_memory = {0: val}
                            ConsoleOutput.success(f"Max GPU Memory set to {val}")
                        else:
                            ConsoleOutput.warning("Invalid format. Use numbers followed by GB, GiB, MB, or MiB.")

                elif choice == '4':
                    val = input(f"Enter Max CPU Memory (e.g., 28GB) [current: {lsc.max_cpu_memory}]: ").strip()
                    if val:
                        if re.match(r"^\d+(\.\d+)?(GB|GiB|MB|MiB)$", val, re.IGNORECASE):
                            lsc.max_cpu_memory = val
                            ConsoleOutput.success(f"Max CPU Memory set to {val}")
                        else:
                            ConsoleOutput.warning("Invalid format. Use numbers followed by GB, GiB, MB, or MiB.")

                elif choice == '5':
                    ac.enable_disk_offload = not ac.enable_disk_offload
                    ConsoleOutput.success(f"Disk Offloading {'ENABLED' if ac.enable_disk_offload else 'DISABLED'}")
                    if ac.enable_disk_offload:
                        ConsoleOutput.warning("Disk offloading is SLOW but saves RAM. Only use if necessary.")

                elif choice == '6':
                    ac.clear_cache_between_chunks = not ac.clear_cache_between_chunks
                    ConsoleOutput.success(f"Cache Clearing {'ENABLED' if ac.clear_cache_between_chunks else 'DISABLED'}")

                elif choice == '7':
                    lsc.low_cpu_mem_usage = not lsc.low_cpu_mem_usage
                    ConsoleOutput.success(f"Low CPU Memory Mode {'ENABLED' if lsc.low_cpu_mem_usage else 'DISABLED'}")

                elif choice == '8':
                    # Reset to recommended defaults
                    lsc.enabled = True
                    lsc.auto_oom_handling = True
                    lsc.max_gpu_memory = {0: "4GB"}
                    lsc.max_cpu_memory = "28GB"
                    lsc.low_cpu_mem_usage = True
                    lsc.offload_state_dict = True
                    ac.enable_disk_offload = False
                    ac.clear_cache_between_chunks = True
                    ac.quantization = "4bit"
                    ac.enable_cpu_offload = True
                    ConsoleOutput.success("Reset to recommended defaults for 4GB VRAM + 32GB RAM system")

                else:
                    ConsoleOutput.warning("Invalid selection.")

            except ValueError:
                ConsoleOutput.warning("Invalid input value.")
            except Exception as e:
                ConsoleOutput.error(f"Error updating setting: {e}")

    def _set_text_hyperparameters(self):
        """Opens the interactive hyperparameter editor."""
        ConsoleOutput.section(f"Configure Text Hyperparameters for {self.state.text_model_provider}: {self.state.text_model_specifier}")
        
        # Load defaults from model entry if they exist
        current_model_entry = get_registry().find_entry(self.state.text_model_specifier)
        if current_model_entry:
             # Check if current settings are just the default, if so, load model-specific defaults
             if self.state.hyperparameters == HyperparameterConfig():
                  model_defaults = current_model_entry.get_hyperparameter_defaults()
                  for key, value in model_defaults.items():
                       if value is not None:
                            setattr(self.state.hyperparameters, key, value)
                  ConsoleOutput.info("Loaded model-specific default hyperparameters.")
        
        editor = InteractiveHyperparameterEditor(self.state.hyperparameters)
        self.state.hyperparameters = editor.edit() # This starts the interactive editing loop
        ConsoleOutput.success("Text hyperparameters updated for this session.")
        return MenuAction.CONTINUE # Stay in Model Settings menu

    def _set_audio_model_menu(self):
        """Top-level menu to choose between local and cloud audio models."""
        while True:
            ConsoleOutput.section("Set Audio Generation Model (TTS)")
            print(f"Audio Generation Currently: {'ENABLED' if self.state.run_audio_generation else 'DISABLED'}")
            print(f"  1. Use Local Model (HF) {self.state.audio_model_provider == 'local_audio' and '[Current]' or ''}")
            print(f"  2. Use Cloud Model (OpenAI) {self.state.audio_model_provider == 'openai_audio' and '[Current]' or ''}")
            # Add other cloud providers here
            print("  T. Toggle Audio Generation (Currently: " + ("ON" if self.state.run_audio_generation else "OFF") + ")")
            
            current_model_display = f"{self.state.audio_model_provider}: {self.state.audio_model_specifier}"
            print(f"\nCurrent Model: {current_model_display}")
            print("\n  b. Back to Model Settings Menu")
            print("-" * 60)

            choice = input("Select option (1, 2, T, or b): ").strip().lower()

            result = None
            if choice == '1':
                result = self._select_local_audio_model()
            elif choice == '2':
                result = self._select_cloud_audio_model("openai_audio")
            elif choice == 't':
                 self.state.run_audio_generation = not self.state.run_audio_generation
                 status = "ENABLED" if self.state.run_audio_generation else "DISABLED"
                 ConsoleOutput.info(f"Audio Generation {status}.")
            elif choice == 'b':
                return MenuAction.BACK
            else:
                ConsoleOutput.warning("Invalid choice.")
                continue

            if result == MenuAction.CONTINUE: # A model was successfully selected
                return MenuAction.CONTINUE # Return to Model Settings menu

    def _select_local_audio_model(self) -> MenuAction:
        """Handles selection of a local HF TTS model."""
        while True:
            ConsoleOutput.subsection("Select Local Audio Model (TTS)")
            print("  1. Browse Hugging Face Hub (text-to-speech)")
            print("  2. Enter Hugging Face Model ID directly")
            print("     Examples: microsoft/speecht5_tts, suno/bark-small, facebook/mms-tts-eng")
            current_local = self.state.audio_model_specifier if self.state.audio_model_provider == 'local_audio' else 'N/A'
            print(f"\nCurrent Local Model: {current_local}")
            print("\n  b. Back to Audio Model Type Selection")
            print("-" * 60)

            choice = input("Select option: ").strip().lower()
            new_model_id = None
            new_provider = "local_audio"

            if choice == 'b': return MenuAction.BACK

            if choice == '1':
                browser = InteractiveModelBrowser(self.model_hub)
                selected_hf_model: Optional[ModelHubInfo] = browser.browse(task="text-to-speech")
                if selected_hf_model:
                     new_model_id = selected_hf_model.model_id
            elif choice == '2':
                 model_id_input = input("Enter Hugging Face Model ID (TTS): ").strip()
                 if model_id_input:
                     if '/' in model_id_input and len(model_id_input) > 3:
                         new_model_id = model_id_input
                         # We can't easily verify if it's a TTS model without loading
                         ConsoleOutput.info(f"Set model ID to {new_model_id}. Will attempt to load at runtime.")
                     else:
                         ConsoleOutput.warning("Invalid model ID format (expected Org/ModelName).")
            else:
                ConsoleOutput.warning("Invalid choice.")
                continue

            if new_model_id:
                self.state.audio_model_provider = new_provider
                self.state.audio_model_specifier = new_model_id
                ConsoleOutput.success(f"Audio model set to: {new_provider}: {self.state.audio_model_specifier}")
                input("Press Enter...")
                return MenuAction.CONTINUE

    def _select_cloud_audio_model(self, provider: str) -> MenuAction:
        """Handles selection of a cloud TTS provider and model."""
        if provider not in self.state.cloud_api_keys:
            ConsoleOutput.warning(f"No API key set for {provider.title()}.")
            print("Please add a key in the 'Cloud API Keys' menu first.")
            input("Press Enter to continue...")
            return MenuAction.BACK # Go back to model type menu

        ConsoleOutput.subsection(f"Set {provider.title()} Model")
        
        example = ""
        default_model = ""
        if provider == "openai_audio":
             example = "e.g., tts-1, tts-1-hd"
             default_model = "tts-1"

        print(f"Current: {self.state.audio_model_specifier if self.state.audio_model_provider == provider else 'None'}")
        model_specifier = input(f"Enter model name ({example}) [default: {default_model}]: ").strip()
        
        if not model_specifier:
            model_specifier = default_model # Use default if empty

        if model_specifier:
            self.state.audio_model_provider = provider
            self.state.audio_model_specifier = model_specifier
            ConsoleOutput.success(f"Audio model set to: {provider}: {model_specifier}")
            input("Press Enter...")
            return MenuAction.CONTINUE
        else:
            ConsoleOutput.warning("Model name cannot be empty.")
            input("Press Enter...")
            return MenuAction.BACK

    def _set_audio_config(self):
        """Interactive configuration for AudioConfig."""
        ConsoleOutput.section("Configure Audio Settings")
        cfg = self.state.audio_config # Get mutable dataclass instance

        while True:
            ConsoleOutput.header("Audio Settings")
            is_local = (self.state.audio_model_provider == 'local_audio')
            print(f"  Provider: {self.state.audio_model_provider}")
            print(f"  Model: {self.state.audio_model_specifier}")
            print("-" * 60)
            print(f"  1. Output Format : {cfg.output_format} (wav, mp3, flac)")
            print(f"  2. Sample Rate   : {cfg.sample_rate} Hz (Target SR)")
            print(f"  3. Speed Factor  : {cfg.speed:.1f} (0.5 to 2.0)")
            print(f"  4. Pitch Shift   : {cfg.pitch_shift} semitones (Post-processing only)")
            print(f"  5. Volume Norm.  : {'Enabled' if cfg.volume_normalize else 'Disabled'} (Post-processing)")

            option_num = 6
            provider_options_map = {}
            
            if is_local:
                 print(f"  {option_num}. Half Precision: {'Enabled' if cfg.use_half_precision else 'Disabled'} (Local CUDA only)")
                 provider_options_map[str(option_num)] = 'use_half_precision'
                 option_num += 1
                 print(f"  {option_num}. Speaker Embed.: {cfg.speaker_embedding or 'Default'} (SpeechT5 only)")
                 provider_options_map[str(option_num)] = 'speaker_embedding'
                 option_num += 1
                 print(f"  {option_num}. Text Chunk Size:{cfg.chunk_size} chars (Local processing)")
                 provider_options_map[str(option_num)] = 'chunk_size'
                 option_num += 1
                 print(f"  {option_num}. Quantization   : {cfg.quantization} (none, 4bit, 8bit)")
                 provider_options_map[str(option_num)] = 'quantization'
                 option_num += 1
            else: # Cloud
                 print(f"  {option_num}. Cloud Voice   : {cfg.cloud_voice} (Provider-specific, e.g., 'alloy')")
                 provider_options_map[str(option_num)] = 'cloud_voice'
                 option_num += 1

            print("\n  s. Save audio settings to preset")
            print("  l. Load audio settings from preset")
            print("  b. Back to Model Settings Menu")
            print("-" * 60)

            choice = input("Select setting to change (or command): ").strip().lower()

            try:
                if choice == 'b':
                    return MenuAction.CONTINUE # Go back to model settings menu

                elif choice == 's':
                    self._save_config_preset(self.config_manager.audio_configs, cfg.to_dict(), "audio preset")
                
                elif choice == 'l':
                    loaded_data = self._load_config_preset(self.config_manager.audio_configs, "audio preset")
                    if loaded_data:
                         try:
                             self.state.audio_config = AudioConfig(**loaded_data)
                             ConsoleOutput.success("Audio preset loaded.")
                         except TypeError as e:
                             ConsoleOutput.error(f"Error applying preset: {e}. Preset might be incompatible.")

                elif choice == '1': # Output Format
                    fmt = input(f"Enter format (wav, mp3, flac) [current: {cfg.output_format}]: ").strip().lower()
                    if fmt in ["wav", "mp3", "flac"]: cfg.output_format = fmt
                    elif fmt: ConsoleOutput.warning("Invalid format.")
                elif choice == '2': # Sample Rate
                    sr = input(f"Enter target sample rate (e.g., 16000, 24000, 44100) [current: {cfg.sample_rate}]: ").strip()
                    if sr.isdigit() and 8000 <= int(sr) <= 48000: cfg.sample_rate = int(sr)
                    elif sr: ConsoleOutput.warning("Invalid sample rate (use 8000-48000).")
                elif choice == '3': # Speed Factor
                    spd_str = input(f"Enter speed factor (e.g., 1.0, 1.2) [current: {cfg.speed:.1f}]: ").strip()
                    if spd_str:
                         spd_f = float(spd_str)
                         if 0.5 <= spd_f <= 2.0: cfg.speed = spd_f # Broaden range slightly
                         else: ConsoleOutput.warning("Speed factor must be between 0.5 and 2.0.")
                elif choice == '4': # Pitch Shift
                    pitch_str = input(f"Enter pitch shift in semitones (-12 to 12) [current: {cfg.pitch_shift}]: ").strip()
                    if pitch_str:
                         pitch_i = int(pitch_str)
                         if -12 <= pitch_i <= 12: cfg.pitch_shift = pitch_i
                         else: ConsoleOutput.warning("Pitch shift out of range (-12 to 12).")
                elif choice == '5': # Volume Norm
                    cfg.volume_normalize = not cfg.volume_normalize
                    ConsoleOutput.info(f"Volume normalization {'Enabled' if cfg.volume_normalize else 'Disabled'}.")

                # Provider-specific options
                elif choice in provider_options_map:
                     setting = provider_options_map[choice]
                     if setting == 'cloud_voice':
                          voice = input(f"Enter provider-specific voice name [current: {cfg.cloud_voice}]: ").strip()
                          if voice: cfg.cloud_voice = voice
                     elif setting == 'use_half_precision':
                          if TORCH_AVAILABLE and torch.cuda.is_available():
                               cfg.use_half_precision = not cfg.use_half_precision
                               ConsoleOutput.info(f"Half precision {'Enabled' if cfg.use_half_precision else 'Disabled'}.")
                          else:
                               ConsoleOutput.warning("Half precision requires PyTorch and CUDA.")
                               cfg.use_half_precision = False
                     elif setting == 'speaker_embedding':
                          spk = input(f"Enter path/ID for speaker embedding (blank=clear) [current: {cfg.speaker_embedding or 'Default'}]: ").strip()
                          cfg.speaker_embedding = spk if spk else None
                          ConsoleOutput.info(f"Speaker embedding set to: {cfg.speaker_embedding or 'Default'}")
                     elif setting == 'chunk_size':
                          cs_str = input(f"Enter text chunk size for TTS (chars) [current: {cfg.chunk_size}]: ").strip()
                          if cs_str.isdigit() and 50 <= int(cs_str) <= 10000: cfg.chunk_size = int(cs_str)
                          elif cs_str: ConsoleOutput.warning("Invalid chunk size (use 50-10000).")
                     elif setting == 'quantization':
                          quant = input(f"Enter quantization (none, 4bit, 8bit) [current: {cfg.quantization}]: ").strip().lower()
                          if quant in ["none", "4bit", "8bit"]: cfg.quantization = quant
                          elif quant: ConsoleOutput.warning("Invalid quantization.")

                else:
                    ConsoleOutput.warning("Invalid selection.")

            except ValueError:
                ConsoleOutput.warning("Invalid input value type (e.g., expected a number).")
            except Exception as e:
                 ConsoleOutput.error(f"Error updating setting: {e}")
                 input("Press Enter...")


    def _output_settings_menu(self):
        """Sets the output directory and optional base filename."""
        ConsoleOutput.section("Output Settings")
        
        # Validate and set output directory
        current_dir_str = str(self.state.output_dir.resolve())
        print(f"Current Output Directory: {current_dir_str}")
        new_dir_str = input(f"Enter new output directory (leave blank to keep current): ").strip()
        
        if new_dir_str:
            try:
                # Resolve relative to CWD if not absolute
                new_dir_path = Path(os.path.expanduser(new_dir_str))
                if not new_dir_path.is_absolute():
                    new_dir_path = Path.cwd() / new_dir_path
                
                # Validate and create
                is_valid, msg = validate_directory_path(new_dir_path, ensure_writable=True)
                if is_valid:
                    self.state.output_dir = new_dir_path.resolve()
                    ConsoleOutput.success(f"Output directory set to: {self.state.output_dir}")
                else:
                    ConsoleOutput.error(f"Invalid directory: {msg}")
            except Exception as e:
                ConsoleOutput.error(f"Error setting directory: {e}")

        # Set base filename (only relevant for single file processing)
        if len(self.state.input_files) > 1:
            ConsoleOutput.info("Note: Base filename is ignored for batch processing (uses input names).")
            self.state.output_filename = None
        else:
             current_name_prompt = f"(current: {self.state.output_filename})" if self.state.output_filename else "(blank = use input file name)"
             new_name = input(f"Enter base output filename (no extension, {current_name_prompt}): ").strip()
             
             if new_name:
                 # Sanitize filename
                 new_name = re.sub(r'[\\/*?:"<>|]', '_', new_name)
                 self.state.output_filename = new_name
                 ConsoleOutput.success(f"Base output filename set to: {self.state.output_filename}")
             elif new_name == "" and self.state.output_filename is not None:
                 ConsoleOutput.info("Base filename cleared. Will use input file name.")
                 self.state.output_filename = None
        
        input("Press Enter to continue...")
        return MenuAction.CONTINUE

    def _save_configuration(self):
        """Saves the current AppState to a named preset."""
        ConsoleOutput.section("Save Configuration Preset")

        presets = self.config_manager.presets.list()
        if presets:
            print("Existing presets:", ", ".join(presets))
        
        name = input("\nEnter name for this configuration (e.g., 'my_podcast_setup'): ").strip()
        if not name:
            ConsoleOutput.warning("Save cancelled. No name provided.")
            input("Press Enter to continue...")
            return MenuAction.CONTINUE
            
        try:
            config_data = self.state.to_dict()
            # Remove input files from saved preset
            config_data.pop("input_files", None)

            if self.config_manager.presets.save(name, config_data):
                ConsoleOutput.success(f"Configuration preset '{name}' saved successfully.")
            else:
                ConsoleOutput.error(f"Failed to save configuration preset '{name}'.")
        except Exception as e:
            ConsoleOutput.error(f"Error saving configuration preset: {e}")
            logger.error(f"Failed to save config preset '{name}'", exc_info=True)
            
        input("Press Enter to continue...")
        return MenuAction.CONTINUE

    def _load_config_preset(self, manager: ConfigCRUD, config_type: str) -> Optional[Dict[str, Any]]:
        """Helper to display and load a config preset from a specific manager."""
        ConsoleOutput.section(f"Load {config_type.title()} Preset")
        
        presets = manager.list()
        if not presets:
            ConsoleOutput.warning(f"No saved {config_type} presets found.")
            print(f"(Looked in: {manager.dir_path})")
            input("Press Enter to continue...")
            return None

        print(f"Available {config_type} presets:")
        for i, name in enumerate(presets):
            print(f"  {i+1}. {name}")
        print("\n  b. Back")
        print("-" * 60)
        
        choice = input(f"Select preset to load (1-{len(presets)} or b): ").strip().lower()
        
        if choice == 'b':
            return None
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                selected_name = presets[idx]
                config_data = manager.load(selected_name)
                if config_data:
                    ConsoleOutput.success(f"Loaded preset '{selected_name}'.")
                    return config_data
                else:
                    ConsoleOutput.error(f"Failed to load preset '{selected_name}'. File might be empty or corrupt.")
            else:
                ConsoleOutput.warning("Invalid selection.")
        except ValueError:
            ConsoleOutput.warning("Invalid input. Please enter a number.")
        except Exception as e:
            ConsoleOutput.error(f"Error loading preset: {e}")
            logger.error(f"Failed to load preset", exc_info=True)

        input("Press Enter to continue...")
        return None

    def _load_configuration(self):
        """Loads an AppState from a saved preset."""
        config_data = self._load_config_preset(self.config_manager.presets, "Configuration")
        if config_data:
             self.state.from_dict(config_data) # This method preserves files/keys
        
        return MenuAction.CONTINUE # Return to main menu

    def _view_current_setup(self):
        """Displays the current configuration state."""
        ConsoleOutput.section("Current Processing Setup Summary")

        # Refresh keys just in case
        self.state.cloud_api_keys = self.config_manager.load_cloud_keys()

        # Input Files
        print(f"Input File(s) ({len(self.state.input_files)}):")
        if not self.state.input_files:
             print("  (None selected)")
        else:
             max_files_to_show = 5
             for i, f in enumerate(self.state.input_files):
                 if i < max_files_to_show:
                     print(f"  - {f.name}")
                 elif i == max_files_to_show:
                     print(f"  ... and {len(self.state.input_files) - max_files_to_show} more.")
                     break
        
        # Output Settings
        print(f"\nOutput Directory: {self.state.output_dir.resolve()}")
        if len(self.state.input_files) == 1 and self.state.output_filename:
             print(f"Base Output Filename: {self.state.output_filename}")
        
        # Text Processing
        print(f"\n--- Text Processing ---")
        print(f"Mode         : {self.state.processing_mode.value}")
        print(f"Provider     : {self.state.text_model_provider}")
        model_display = Path(self.state.text_model_specifier).name if self.state.text_model_provider=='local_gguf' else self.state.text_model_specifier
        print(f"Model        : {model_display}")

        # Local hardware settings
        if self.state.text_model_provider == "local_hf":
            print(f"HW Profile   : {self.state.memory_profile} (Quant: {self.state.quantization}, GPU: {self.state.max_gpu_memory}, CPU: {self.state.max_cpu_memory})")
        elif self.state.text_model_provider == "local_gguf":
            print(f"HW Profile   : {self.state.memory_profile} (GPU Layers: {self.state.gpu_layers})")

        print(f"Hyperparams  : (Temp: {self.state.hyperparameters.temperature}, Top_p: {self.state.hyperparameters.top_p}, Max_Tokens: {self.state.hyperparameters.max_new_tokens or 'Auto'})")
        
        # Stages
        print(f"\n--- Pipeline Stages ---")
        text_stages_str = ', '.join([s.capitalize() for s in self.state.stages_to_run])
        print(f"Text Stages: {text_stages_str if text_stages_str else 'None'}")
        
        audio_status = 'ENABLED' if self.state.run_audio_generation else 'DISABLED'
        print(f"Audio Stage: {audio_status}")

        if self.state.run_audio_generation:
             print(f"\n--- Audio Generation ---")
             print(f"Provider     : {self.state.audio_model_provider}")
             print(f"Model        : {self.state.audio_model_specifier}")
             print(f"Config       : (Voice: {self.state.audio_config.cloud_voice}, Speed: {self.state.audio_config.speed}x, Format: {self.state.audio_config.output_format})")
        
        # Warnings/Errors Check
        print("-" * 60)
        errors = self._validate_setup()
        if errors:
             ConsoleOutput.error("Setup incomplete - Cannot Run:")
             for err in errors: print(f"  - {err}")
        else:
             ConsoleOutput.success("Setup appears valid and ready to run.")
             
        input("\nPress Enter to continue...")
        return MenuAction.CONTINUE
        
    def _validate_setup(self) -> List[str]:
        """Checks the current state for errors before running."""
        errors = []
        text_stages_selected = bool(self.state.stages_to_run)

        if not self.state.input_files:
            errors.append("No input files selected.")
        if not text_stages_selected and not self.state.run_audio_generation:
            errors.append("No processing stages (text or audio) selected.")
            
        # Text model checks
        if text_stages_selected:
            if not self.state.text_model_specifier:
                errors.append("No text model specified.")
            elif self.state.text_model_provider not in ["local_hf", "local_gguf"]:
                if self.state.text_model_provider not in self.state.cloud_api_keys:
                    errors.append(f"API key for text provider '{self.state.text_model_provider}' is missing.")
            elif self.state.text_model_provider == 'local_gguf':
                 if not Path(self.state.text_model_specifier).is_file():
                      errors.append(f"GGUF model file not found: {self.state.text_model_specifier}")
        
        # Audio model checks
        if self.state.run_audio_generation:
            # Check if audio-only mode (no text stages) or normal mode (with text stages)
            if text_stages_selected:
                # Normal mode: text stages are running, save stage is required
                if 'save' not in self.state.stages_to_run:
                     errors.append("Audio generation with text processing requires the 'save' stage to be enabled.")
            else:
                # Audio-only mode: no text stages, input must be .txt or .md
                if self.state.input_files:
                    input_file = Path(self.state.input_files[0])
                    if input_file.suffix.lower() not in ['.txt', '.md']:
                        errors.append("Audio-only mode (no text stages) requires input file to be .txt or .md format.")

            if not self.state.audio_model_specifier:
                errors.append("No audio model specified.")
            elif self.state.audio_model_provider not in ["local_audio"]:
                 if self.state.audio_model_provider not in self.state.cloud_api_keys:
                    errors.append(f"API key for audio provider '{self.state.audio_model_provider}' is missing.")
        
        return errors

    def _checkpoint_management_action(self):
        """Checkpoint management submenu."""
        self.checkpoint_manager.checkpoint_management_menu()
        return MenuAction.CONTINUE

    def _confirm_and_run(self) -> MenuAction:
        """Confirms the setup and returns RUN action if confirmed."""
        ConsoleOutput.section("Confirm and Run")
        
        errors = self._validate_setup()
        if errors:
            ConsoleOutput.error("Cannot run. Please fix these issues:")
            for err in errors:
                print(f"  - {err}")
            input("\nPress Enter to return to the menu...")
            return MenuAction.CONTINUE

        print("Current Configuration:")
        print(f"  Input Files: {len(self.state.input_files)}")
        print(f"  Text Model:  {self.state.text_model_provider}: {self.state.text_model_specifier}")
        print(f"  Stages:      {', '.join(self.state.stages_to_run)}")
        print(f"  Audio Gen:   {'Yes' if self.state.run_audio_generation else 'No'}")
        if self.state.run_audio_generation:
             print(f"  Audio Model: {self.state.audio_model_provider}: {self.state.audio_model_specifier}")
        print(f"  Output Dir:  {self.state.output_dir.resolve()}")
        print("-" * 60)
        
        confirm = input("Proceed with processing? (y/n): ").strip().lower()
        if confirm == 'y':
            return MenuAction.RUN # Signal to the main loop to execute
        else:
            ConsoleOutput.info("Processing cancelled.")
            return MenuAction.CONTINUE

    def _execute_pipeline(self) -> bool:
         """Executes the text and/or audio pipeline based on the current state."""
         ConsoleOutput.header("🚀 Starting Pipeline Execution 🚀")
         
         # 1. Get configuration objects from state
         try:
            pipeline_config = self.state.get_pipeline_config()
         except Exception as e:
             ConsoleOutput.error(f"Failed to build pipeline configuration: {e}")
             logger.error("Failed to build PipelineConfig", exc_info=True)
             return False

         text_backend: Optional[LLMBackend] = None
         audio_backend: Optional[AudioBackend] = None
         pipeline: Optional[ProcessingPipeline] = None
         all_results: List[PipelineResult] = []
         overall_success = True

         try:
             # 2. Initialize Text Backend (only if 'process' stage is selected)
             # Only the 'process' stage actually needs the LLM backend
             needs_llm = 'process' in pipeline_config.stages
             if needs_llm:
                 ConsoleOutput.info(f"Initializing Text Backend ({pipeline_config.model_provider})...")
                 start_init = time.time()
                 
                 # Get ModelEntry for local HF models (needed for filter/token calc)
                 model_entry_for_backend = None
                 if pipeline_config.model_provider == "local_hf":
                      model_entry_for_backend = get_registry().find_entry(pipeline_config.model_specifier)
                      if not model_entry_for_backend:
                           # Create a placeholder if not in registry (e.g., loaded from CLI)
                           model_entry_for_backend = ModelEntry(
                                model_id=pipeline_config.model_specifier,
                                name=Path(pipeline_config.model_specifier).name,
                                author="Unknown",
                                max_context=4096, # Use a safe default
                                optimal_chunk_size=1000 # Use a safe default
                           )
                           logger.warning(f"Using default ModelEntry for {pipeline_config.model_specifier}")

                 text_backend = get_llm_backend(
                     provider=pipeline_config.model_provider,
                     model_specifier=pipeline_config.model_specifier,
                     api_keys=self.state.cloud_api_keys,
                     hyperparameters=pipeline_config.get_hyperparameters(),
                     model_entry=model_entry_for_backend, # Pass ModelEntry (only used by local_hf)
                     quantization_config=pipeline_config.quantization_config,
                     layer_split_config=pipeline_config.layer_split_config
                 )

                 if text_backend is None:
                      raise ModelLoadError(f"Could not create text backend for {pipeline_config.model_provider}.")
                 
                 # Loading is now deferred to the 'process' stage, but we can init client here
                 if not pipeline_config.model_provider.startswith("local"):
                     if not text_backend.load(): # This just inits the client for cloud
                         raise ModelLoadError(f"Failed to initialize cloud backend: {text_backend.model_identifier}")

                 init_time = time.time() - start_init
                 ConsoleOutput.success(f"Text backend initialized in {init_time:.2f}s.")

             # 3. Initialize Audio Backend (if audio gen is selected)
             if pipeline_config.generate_audio:
                 ConsoleOutput.info(f"Initializing Audio Backend ({pipeline_config.audio_provider})...")
                 start_init_audio = time.time()
                 audio_backend = get_audio_backend(
                     provider=pipeline_config.audio_provider,
                     model_specifier=pipeline_config.audio_specifier,
                     api_keys=self.state.cloud_api_keys,
                     config=pipeline_config.audio_config,
                     layer_split_config=pipeline_config.layer_split_config
                 )
                 if audio_backend is None:
                     raise ModelLoadError(f"Could not create audio backend for {pipeline_config.audio_provider}.")
                 
                 # Defer local audio model loading to its stage
                 if not audio_backend.provider_identifier.startswith("local"):
                     if not audio_backend.load():
                         raise ModelLoadError(f"Failed to initialize cloud audio backend.")
                 
                 init_time_audio = time.time() - start_init_audio
                 ConsoleOutput.success(f"Audio backend initialized in {init_time_audio:.2f}s.")

             # 4. Initialize Pipeline object and inject dependencies
             pipeline = ProcessingPipeline(config=pipeline_config)
             pipeline.llm_backend = text_backend  # Inject (can be None)
             pipeline.audio_backend = audio_backend  # Inject (can be None)

             # Set stages to run directly from the app state
             stages_to_run = self.state.stages_to_run.copy()
             if self.state.run_audio_generation and 'audio' not in stages_to_run:
                 stages_to_run.append('audio')
             pipeline.set_stages_to_run(stages_to_run)

             # 5. Run Processing
             if len(self.state.input_files) == 1:
                 # Single file
                 output_base = self.state.output_filename or None # Let pipeline handle default if None
                 output_p_base = self.state.output_dir / output_base if output_base else None
                 
                 result = pipeline.process_file(self.state.input_files[0], output_path_base=output_p_base)
                 all_results.append(result)
             else:
                 # Batch processing
                 batch_results = pipeline.process_batch(
                     input_files=self.state.input_files,
                     output_dir=self.state.output_dir
                 )
                 all_results.extend(batch_results)
            
             # Check overall success
             if not all(r.success for r in all_results):
                  overall_success = False

         except (ModelLoadError, PipelineError, FileProcessingError) as e:
            ConsoleOutput.error(f"🚨 Pipeline execution failed: {e}")
            logger.critical("Pipeline execution failed", exc_info=True)
            overall_success = False
         except KeyboardInterrupt:
            ConsoleOutput.warning("\nProcessing interrupted by user.")
            logger.warning("Pipeline execution interrupted.")
            overall_success = False
         except Exception as e:
             ConsoleOutput.error(f"🚨 An unexpected error occurred during execution: {e}")
             logger.critical("Pipeline execution failed with unexpected error", exc_info=True)
             overall_success = False
         finally:
             # Cleanup: Unload models, save report
             ConsoleOutput.info("Cleaning up resources...")
             if text_backend:
                 try: text_backend.unload()
                 except Exception as unload_err: logger.warning(f"Error unloading text backend: {unload_err}")
             if audio_backend:
                 try: audio_backend.unload()
                 except Exception as unload_err: logger.warning(f"Error unloading audio backend: {unload_err}")
             if pipeline:
                 try:
                     pipeline.cleanup() # Saves report, cleans caches
                 except Exception as cleanup_err:
                     logger.error(f"Error during final pipeline cleanup: {cleanup_err}")
             
             ConsoleOutput.info("Cleanup complete.")

         return overall_success
