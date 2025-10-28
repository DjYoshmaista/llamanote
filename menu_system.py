"""
Menu System Module - Refacted
Interactive menu interface for LlamaNote Enhanced
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging # Import logging for setting level in verbose mode

# --- LlamaNote Modules ---
from loggerConf import ConsoleOutput, get_logger_conf
from config_manager import ConfigManager
from config_base import (
    BASE_DIR, OUTPUT_DIR, CACHE_DIR, OFFLOAD_DIR,
    DEFAULT_MODEL, FALLBACK_MODEL,
    ENABLE_LAYER_SPLITTING, DEFAULT_GPU_LAYERS,
    DEFAULT_QUANTIZATION, QUANTIZATION_OPTIONS,
    SUPPORTED_FORMATS, PIPELINE_STAGES, TIMESTAMP_OUTPUTS # Added TIMESTAMP_OUTPUTS
)
# Import shared types from pipeline_types
from pipeline_types import (
    ProcessingMode,
    PipelineConfig,
    PipelineResult,
    QuantizationConfig,
    LayerSplitConfig,
    GenerationResult
)
from model_hub import InteractiveModelBrowser, ModelHub, ModelInfo as HFModelInfo
from audio_generator import (
    AudioConfig, AudioResult, AudioBackend,
    LocalAudioBackend, OpenAITTSBackend, get_audio_backend, LoggingProgress # Import LoggingProgress here
)
from processing_pipeline import ProcessingPipeline # Keep this for instantiation
from text_processor import ChunkingStrategy
from model_registry import get_registry, ModelEntry
from hyperparameters import HyperparameterConfig, InteractiveHyperparameterEditor, get_hyperparameter_help
from llm_handler import (
    LLMBackend, get_llm_backend # Keep LLMBackend for type hint, factory for creation
    # QuantizationConfig, LayerSplitConfig are now in pipeline_types
)
from huggingface_search import HuggingFaceSearch # Import fix
from file_handler import BatchFileManager

# --- GGUF Backend (Optional) ---
try:
    # Need LlamaCppBackend for type hinting if used, Config for creation
    from llamacpp_backend import LlamaCppBackend, LlamaCppConfig, GGUFModelManager
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    LlamaCppBackend = None
    LlamaCppConfig = None
    GGUFModelManager = None

# Import torch only if needed for local hardware settings check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None # Make torch usable in conditional checks


logger = get_logger_conf(__name__)

# --- Menu Data Structures ---

class MenuAction(Enum):
    BACK = "back"
    EXIT = "exit"
    CONTINUE = "continue"
    RUN = "run"

@dataclass
class MenuItem:
    key: str
    label: Union[str, Callable[[], str]] # Allow label to be a function
    action: Optional[Callable] = None
    submenu: Optional['Menu'] = None
    description: Optional[Union[str, Callable[[], str]]] = None # Allow description to be a function


class Menu:
    def __init__(self, title: str, items: List[MenuItem], parent: Optional['Menu'] = None):
        self.title = title
        self.items = items
        self.parent = parent

    def display(self) -> MenuAction:
        while True:
            ConsoleOutput.header(self.title)
            for item in self.items:
                label_text = item.label() if callable(item.label) else item.label
                desc_text = item.description() if callable(item.description) else item.description
                desc_str = f" - {desc_text}" if desc_text else ""
                print(f"  {item.key}. {label_text}{desc_str}")

            nav_options = []
            if self.parent:
                nav_options.append("b. Back")
            nav_options.append("q. Quit")
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
                    # Ensure action_result is handled correctly
                    if isinstance(action_result, MenuAction):
                         result = action_result
                    # If action returns None or something else, default to CONTINUE
                    else:
                         result = MenuAction.CONTINUE


                if result == MenuAction.EXIT:
                    return MenuAction.EXIT # Propagate EXIT up
                elif result == MenuAction.RUN:
                    return MenuAction.RUN # Propagate RUN up
                # If result is BACK or CONTINUE, loop back to display this menu again
                elif result == MenuAction.BACK:
                    continue
                elif result == MenuAction.CONTINUE:
                    continue # Explicitly continue loop

            else:
                ConsoleOutput.warning("Invalid selection")


    def add_item(self, item: MenuItem):
        self.items.append(item)

# --- Menu System State ---

SUPPORTED_CLOUD_PROVIDERS = [
    "openai",
    "anthropic",
    "google",
    # "cohere", # Add back when implemented
    # "huggingface_token", # This is often an API key for HF Inference API or specific services
    # "openrouter", # Add back when implemented
]
TTS_PROVIDERS = ["local", "openai"] # Add more as backends are implemented

@dataclass
class AppState:
    """Holds the current configuration state of the application menu."""
    input_files: List[Path] = field(default_factory=list)
    output_dir: Path = Path(OUTPUT_DIR)
    output_filename: Optional[str] = None
    processing_mode: ProcessingMode = ProcessingMode.PODCAST
    text_model_provider: str = "local"
    text_model_specifier: str = DEFAULT_MODEL # This should be a key initially
    audio_model_provider: str = "local"
    audio_model_specifier: str = "microsoft/speecht5_tts" # Example default TTS
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    stages_to_run: List[str] = field(default_factory=lambda: list(PIPELINE_STAGES))
    run_audio_generation: bool = False
    cloud_api_keys: Dict[str, str] = field(default_factory=dict)

    # --- Local model compute settings ---
    quantization: str = DEFAULT_QUANTIZATION
    gpu_layers: int = DEFAULT_GPU_LAYERS
    max_gpu_memory: str = "10GB"
    max_cpu_memory: str = "30GB"

    def __post_init__(self):
        # Resolve default model key to ID on init if possible
        registry = get_registry()
        default_entry = registry.get_by_key(DEFAULT_MODEL)
        if default_entry:
            self.text_model_specifier = default_entry.model_id
        else:
            # Handle case where default key doesn't resolve (registry empty?)
            logger.warning(f"Default model key '{DEFAULT_MODEL}' not found in registry. Using key as specifier.")
            self.text_model_specifier = DEFAULT_MODEL # Keep the key as specifier

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the app state to a dictionary for saving."""
        # Need to handle Path objects and Enums correctly
        return {
            "input_files": [str(f.resolve()) for f in self.input_files], # Save absolute paths as strings
            "output_dir": str(self.output_dir.resolve()),
            "output_filename": self.output_filename,
            "processing_mode": self.processing_mode.value, # Save enum value
            "text_model_provider": self.text_model_provider,
            "text_model_specifier": self.text_model_specifier,
            "audio_model_provider": self.audio_model_provider,
            "audio_model_specifier": self.audio_model_specifier,
            "hyperparameters": asdict(self.hyperparameters),
            "audio_config": asdict(self.audio_config), # Use helper if needed for complex types
            "stages_to_run": self.stages_to_run, # Use current state directly
            "run_audio_generation": self.run_audio_generation, # Use current state directly
            "quantization": self.quantization,
            "gpu_layers": self.gpu_layers,
            "max_gpu_memory": self.max_gpu_memory,
            "max_cpu_memory": self.max_cpu_memory
        }

    def from_dict(self, data: Dict[str, Any]):
        """Deserializes a dictionary into the app state."""
        try:
            self.input_files = [Path(p) for p in data.get("input_files", []) if Path(p).exists()] # Check existence
            self.output_dir = Path(data.get("output_dir", OUTPUT_DIR))
            self.output_filename = data.get("output_filename")
            # Handle potential invalid enum value during load
            try: self.processing_mode = ProcessingMode(data.get("processing_mode", ProcessingMode.PODCAST.value))
            except ValueError: self.processing_mode = ProcessingMode.PODCAST

            self.text_model_provider = data.get("text_model_provider", "local")
            self.text_model_specifier = data.get("text_model_specifier", self.text_model_specifier) # Keep default if not in file
            self.audio_model_provider = data.get("audio_model_provider", "local")
            self.audio_model_specifier = data.get("audio_model_specifier", "microsoft/speecht5_tts")
            # Load nested dataclasses carefully
            self.hyperparameters = HyperparameterConfig(**data.get("hyperparameters", {}))
            self.audio_config = AudioConfig(**data.get("audio_config", {}))
            self.stages_to_run = data.get("stages_to_run", list(PIPELINE_STAGES))
            self.run_audio_generation = data.get("run_audio_generation", False)
            self.quantization = data.get("quantization", DEFAULT_QUANTIZATION)
            self.gpu_layers = data.get("gpu_layers", DEFAULT_GPU_LAYERS)
            self.max_gpu_memory = data.get("max_gpu_memory", "10GB")
            self.max_cpu_memory = data.get("max_cpu_memory", "30GB")

            ConsoleOutput.success("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading configuration state: {e}", exc_info=True)
            ConsoleOutput.error(f"Error loading configuration: {e}. State may be partial or defaults used.")
            # Optionally reset to defaults on load error?
            # self.__init__() # Reset to defaults


    def get_pipeline_config(self) -> PipelineConfig:
        """Creates a PipelineConfig based on current AppState."""
        # Use the specifier from state (which should be resolved ID or path)
        model_name_meta = f"{self.state.text_model_provider}:{self.state.text_model_specifier}"

        # Create mock args namespace for the helper functions
        # This is slightly awkward, consider moving helpers or refactoring them
        mock_args = argparse.Namespace(
            quantization=self.quantization,
            compute_dtype='bfloat16', # Make configurable if needed
            gpu_layers=self.gpu_layers,
            no_layer_split=(not ENABLE_LAYER_SPLITTING), # Use base config
            max_gpu_memory=self.max_gpu_memory,
            max_cpu_memory=self.max_cpu_memory
        )

        # Create Quantization and LayerSplit configs using helpers
        q_config = create_quantization_config(mock_args)
        ls_config = create_layer_split_config(mock_args)

        return PipelineConfig(
            mode=self.processing_mode,
            model_name=model_name_meta, # For logging/metadata
            model_provider=self.text_model_provider,
            model_specifier=self.text_model_specifier, # Actual ID/path
            quantization_config=q_config, # Pass generated config
            layer_split_config=ls_config, # Pass generated config
            # memory_profile is redundant now
            chunking_strategy=ChunkingStrategy.WORD_BOUNDARY, # Make configurable
            chunk_size=1000, # Make configurable
            markdown_style=self.processing_mode.value, # Or separate config
            output_format="markdown", # Make configurable
            hyperparameters=self.hyperparameters # Pass current hyperparams
        )


class MenuSystem:
    def __init__(self):
        self.state = AppState()
        self.config_manager = ConfigManager()
        # No need for separate HF search, ModelHub includes it
        self.model_hub = ModelHub(cache_dir=CACHE_DIR / "model_hub")
        self.registry = get_registry() # Get registry instance

        # Load API keys on startup
        self.state.cloud_api_keys = self.config_manager.load_cloud_keys()
        logger.info(f"Loaded API keys for providers: {list(self.state.cloud_api_keys.keys())}")

        self.main_menu = self._build_main_menu()
        logger.info("Initialized MenuSystem")

    def run(self):
        """Starts the main menu loop."""
        while True: # Keep running the menu until EXIT
             result = self.main_menu.display()
             if result == MenuAction.EXIT:
                 ConsoleOutput.info("Exiting LlamaNote.")
                 break # Exit the loop
             elif result == MenuAction.RUN:
                 run_success = self._execute_pipeline()
                 if run_success:
                      ConsoleOutput.success("Processing finished successfully.")
                 else:
                      ConsoleOutput.error("Processing finished with errors.")
                 input("Press Enter to return to the main menu...")
                 # Loop continues, displaying main menu again
             # Handle BACK from main menu (should not happen if parent is None)
             elif result == MenuAction.BACK:
                  ConsoleOutput.warning("Already at main menu.")
             # Handle CONTINUE (just loop)
             elif result == MenuAction.CONTINUE:
                  pass

    # --- Build Menus ---

    def _build_main_menu(self) -> Menu:
        """Builds the top-level menu."""
        items = [
            MenuItem("1", "Select Input & Stages", self._select_input_menu, description=lambda: f"{len(self.state.input_files)} file(s) selected"),
            MenuItem("2", "Model Settings", submenu=self._build_model_settings_menu(), description="Configure text and audio models"),
            MenuItem("3", "Cloud Provider Settings", submenu=self._build_cloud_settings_menu(), description="Manage API keys"),
            MenuItem("4", "Output Settings", self._output_settings_menu, description=lambda: f"Saving to: {self.state.output_dir.name}"),
            MenuItem("5", "Load Configuration", self._load_configuration, description="Load a saved settings preset"),
            MenuItem("6", "Save Configuration", self._save_configuration, description="Save current settings as a preset"),
            MenuItem("7", "View Current Setup", self._view_current_setup, description="Review selections"),
            MenuItem("8", "Run Processing", self._confirm_and_run, description="Execute the pipeline"),
        ]
        # Assign self.main_menu *before* creating submenus that need it as parent
        main_menu = Menu("LlamaNote Enhanced - Main Menu", items, parent=None)
        self.main_menu = main_menu # Set self.main_menu here

        # Now build submenus and assign parent
        model_submenu = self._build_model_settings_menu()
        cloud_submenu = self._build_cloud_settings_menu()
        items[1].submenu = model_submenu # Assign submenu to main menu item
        items[2].submenu = cloud_submenu
        model_submenu.parent = main_menu # Set parent for submenus
        cloud_submenu.parent = main_menu

        return main_menu


    def _build_model_settings_menu(self) -> Menu:
        """Builds the menu for configuring models."""
        items = [
            MenuItem(
                "1",
                "Set Text Processing Model",
                self._set_text_model_menu,
                description=lambda: f"Current: {self.state.text_model_provider} - {self.state.text_model_specifier}"
            ),
            MenuItem(
                "2",
                "Set Text Model Hyperparameters",
                self._set_text_hyperparameters,
                description="Configure generation settings"
            ),
            MenuItem( # New item for local model settings
                "3",
                "Set Local Model Hardware Settings",
                self._set_local_hardware_config,
                description=lambda: f"Quant: {self.state.quantization}, GPU Lyr: {self.state.gpu_layers}"
            ),
            MenuItem(
                "4",
                "Set Audio Generation Model",
                self._set_audio_model_menu,
                description=lambda: f"Audio Gen: {'ON' if self.state.run_audio_generation else 'OFF'} | {self.state.audio_model_provider}: {self.state.audio_model_specifier}"
            ),
            MenuItem(
                "5",
                "Configure Audio Settings",
                self._set_audio_config,
                description="Set sample rate, format, voice, etc."
            ),
        ]
        # Parent is assigned when this menu is added to the main menu
        menu = Menu("Model Settings", items, parent=None)
        return menu

    def _build_cloud_settings_menu(self) -> Menu:
        """Builds the menu for managing API keys."""
        items = []
        # Dynamically build based on SUPPORTED_CLOUD_PROVIDERS
        for i, provider in enumerate(SUPPORTED_CLOUD_PROVIDERS):
            # Use lambdas to capture the provider variable correctly
            items.append(MenuItem(
                key=str(i + 1),
                label=lambda p=provider: f"Set/Update {p.replace('_', ' ').title()} Key {'(Set)' if p in self.state.cloud_api_keys else '(Not Set)'}",
                action=lambda p=provider: self._set_provider_key(p)
            ))
        items.append(MenuItem(str(len(SUPPORTED_CLOUD_PROVIDERS) + 1), "View Configured Keys", self._view_keys))
        # Parent is assigned when this menu is added to the main menu
        menu = Menu("Cloud Provider API Keys", items, parent=None)
        return menu


    # --- Menu Actions ---

    # ... (Keep _save_configuration, _load_configuration as they are) ...
    # Need to adjust state saving/loading in these methods if AppState structure changes significantly

    def _save_configuration(self):
        """Saves the current AppState to a named preset."""
        ConsoleOutput.section("Save Configuration Preset")

        presets = self.config_manager.list_configs(dir_type="preset")
        if presets:
            print("Existing presets:")
            for p in presets:
                print(f"  - {p}")
        else:
            print("No existing presets found.")

        name = input("\nEnter name for this configuration (e.g., 'my_podcast_setup'): ").strip()
        if not name:
            ConsoleOutput.warning("Save cancelled. No name provided.")
            input("Press Enter to continue...")
            return MenuAction.CONTINUE

        # Basic sanitization
        name = re.sub(r'[\\/*?:"<>|]', '', name).replace(" ", "_")
        if not name:
            ConsoleOutput.error("Invalid name after sanitization. Save cancelled.")
            input("Press Enter to continue...")
            return MenuAction.CONTINUE

        try:
            # Get current state, exclude non-serializable or irrelevant parts
            config_data = self.state.to_dict()
            # Don't save input files or API keys in presets
            config_data.pop("input_files", None)
            config_data.pop("cloud_api_keys", None)

            success = self.config_manager.save_config(name, config_data, dir_type="preset")
            if success:
                ConsoleOutput.success(f"Configuration preset '{name}' saved successfully.")
            else:
                ConsoleOutput.error(f"Failed to save configuration preset '{name}'.")
        except Exception as e:
            ConsoleOutput.error(f"Error saving configuration preset: {e}")
            logger.error(f"Failed to save config preset '{name}'", exc_info=True)

        input("Press Enter to continue...")
        return MenuAction.CONTINUE

    def _load_configuration(self):
        """Loads an AppState from a saved preset."""
        ConsoleOutput.section("Load Configuration Preset")

        presets = self.config_manager.list_configs(dir_type="preset")
        if not presets:
            ConsoleOutput.warning("No saved presets found.")
            print(f"(Looked in: {self.config_manager.get_dir('preset')})")
            input("Press Enter to continue...")
            return MenuAction.CONTINUE

        print("Available presets:")
        for i, name in enumerate(presets):
            print(f"  {i+1}. {name}")
        print("\n  b. Back")
        print("-" * 60)

        choice = input(f"Select preset to load (1-{len(presets)} or b): ").strip().lower()

        if choice == 'b':
            return MenuAction.CONTINUE

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                selected_name = presets[idx]
                config_data = self.config_manager.load_config(selected_name, dir_type="preset")
                if config_data:
                    # Preserve current input files and API keys
                    current_inputs = self.state.input_files
                    current_keys = self.state.cloud_api_keys

                    self.state.from_dict(config_data) # Load preset data

                    # Restore preserved items
                    self.state.input_files = current_inputs
                    self.state.cloud_api_keys = current_keys # Keep runtime keys

                    ConsoleOutput.success(f"Loaded preset '{selected_name}'. Input files and API keys remain unchanged.")
                else:
                    ConsoleOutput.error(f"Failed to load preset '{selected_name}'. File might be empty or corrupt.")
            else:
                ConsoleOutput.warning("Invalid selection.")
        except ValueError:
            ConsoleOutput.warning("Invalid input. Please enter a number.")
        except Exception as e:
            ConsoleOutput.error(f"Error loading preset: {e}")
            logger.error(f"Failed to load preset '{selected_name}'", exc_info=True)

        input("Press Enter to continue...")
        return MenuAction.CONTINUE # Go back to main menu


    def _select_input_menu(self):
        """Handles selection of input file(s) or directory."""
        ConsoleOutput.section("Select Input Source")
        path_str = input("Enter file path OR folder path containing files: ").strip()
        if not path_str:
            return MenuAction.CONTINUE

        path_str = os.path.expanduser(path_str) # Handle ~/ paths
        path = Path(path_str)

        selected_files = []
        if not path.exists():
            ConsoleOutput.error(f"Path does not exist: {path_str}")
            input("Press Enter to continue...")
            return MenuAction.CONTINUE

        if path.is_file():
             if path.suffix.lower() in SUPPORTED_FORMATS:
                selected_files = [path]
                ConsoleOutput.success(f"Selected single file: {path.name}")
             else:
                 ConsoleOutput.error(f"Unsupported file type: {path.suffix}. Supported: {', '.join(SUPPORTED_FORMATS)}")
                 input("Press Enter to continue...")
                 return MenuAction.CONTINUE

        elif path.is_dir():
            ConsoleOutput.info(f"Scanning directory (recursive): {path}")
            fm = BatchFileManager() # Use default FileHandler settings

            # Collect files using BatchFileManager's logic
            try:
                 # Pass the directory as a single Path object in a list
                 supported_files = fm.collect_input_files([path], recursive=True)
            except Exception as e:
                 ConsoleOutput.error(f"Error scanning directory: {e}")
                 supported_files = []


            if not supported_files:
                ConsoleOutput.warning("No supported files found in this directory or subdirectories.")
                self.state.input_files = []
                input("Press Enter to continue...")
                return MenuAction.CONTINUE

            print(f"Found {len(supported_files)} supported files.")
            print("  1. Process ALL files found")
            print("  2. Select specific files from the list")
            print("\n  b. Back")
            choice = input("Select option (1-2 or b): ").strip().lower()

            if choice == '1':
                selected_files = supported_files
                ConsoleOutput.success(f"Selected {len(selected_files)} files for batch processing.")
            elif choice == '2':
                 print("\nAvailable files:")
                 for i, file_path in enumerate(supported_files):
                     try: display_path = file_path.relative_to(path)
                     except ValueError: display_path = file_path # Show full path if not relative
                     print(f"  {i+1:3d}. {display_path}")

                 file_choices_str = input(f"Enter file numbers to process (comma-separated, e.g., 1,3,5): ").strip()
                 try:
                     indices = [int(n.strip()) - 1 for n in file_choices_str.split(',') if n.strip().isdigit()]
                     valid_indices = [idx for idx in indices if 0 <= idx < len(supported_files)]
                     if valid_indices:
                         selected_files = [supported_files[idx] for idx in valid_indices]
                         ConsoleOutput.success(f"Selected {len(selected_files)} specific file(s):")
                         # Display selected files concisely
                         names = [f.name for f in selected_files]
                         if len(names) <= 5: print(f"  Files: {', '.join(names)}")
                         else: print(f"  Files: {', '.join(names[:3])}, ..., {names[-1]}")
                     else:
                         ConsoleOutput.warning("No valid file numbers selected.")
                 except ValueError:
                     ConsoleOutput.warning("Invalid input format for file numbers.")
            elif choice == 'b':
                 return MenuAction.CONTINUE
            else:
                 ConsoleOutput.warning("Invalid choice.")
        else:
            ConsoleOutput.error(f"Path is not a file or directory: {path_str}")

        self.state.input_files = selected_files

        # Only proceed to stages if files were selected
        if self.state.input_files:
            return self._select_stages_menu()
        else:
            # If no files selected (e.g., user backed out), return to main menu
            return MenuAction.CONTINUE


    def _select_stages_menu(self):
        """Allows user to toggle which pipeline stages run."""
        all_text_stages = list(PIPELINE_STAGES)

        while True:
            ConsoleOutput.header("Select Processing Stages")
            print("Text Processing Stages:")
            for i, stage in enumerate(all_text_stages):
                 included = "[X]" if stage in self.state.stages_to_run else "[ ]"
                 print(f"  {i+1}. {included} {stage.capitalize()}") # Use capitalize for readability

            audio_included = "[X]" if self.state.run_audio_generation else "[ ]"
            print(f"\nAudio Generation Stage:")
            # Clarify dependency
            audio_dependency_met = 'save' in self.state.stages_to_run
            dependency_note = "" if audio_dependency_met else " (Requires 'Save' stage)"
            print(f"  A. {audio_included} Generate Audio{dependency_note}")

            print("\nOptions:")
            print(f"  Enter number (1-{len(all_text_stages)}) or 'A' to toggle a stage.")
            print("  'all'    - Select all stages (Text + Audio).")
            print("  'text'   - Select all text stages only.")
            print("  'none'   - Deselect all stages.")
            print("  'done'   - Confirm selection and go back.")
            print("-" * 60)

            choice = input("Toggle stage or command: ").strip().lower()

            if choice == 'done':
                # Final check for audio dependency
                if self.state.run_audio_generation and not audio_dependency_met:
                     ConsoleOutput.warning("Audio generation enabled, but 'Save' stage is disabled. Audio will likely fail.")
                     confirm = input("Proceed anyway? (y/n): ").strip().lower()
                     if confirm != 'y': continue # Re-prompt stage selection
                break # Exit loop if 'done'
            elif choice == 'all':
                self.state.stages_to_run = list(PIPELINE_STAGES)
                self.state.run_audio_generation = True
                ConsoleOutput.info("All text and audio stages selected.")
            elif choice == 'text':
                self.state.stages_to_run = list(PIPELINE_STAGES)
                self.state.run_audio_generation = False # Explicitly disable audio
                ConsoleOutput.info("All text stages selected, audio disabled.")
            elif choice == 'none':
                self.state.stages_to_run = []
                self.state.run_audio_generation = False
                ConsoleOutput.info("All stages deselected.")
            elif choice == 'a':
                self.state.run_audio_generation = not self.state.run_audio_generation
                status = "ENABLED" if self.state.run_audio_generation else "DISABLED"
                ConsoleOutput.info(f"Audio Generation {status}.")
                if self.state.run_audio_generation and not audio_dependency_met:
                     ConsoleOutput.warning("Warning: Audio generation requires the 'Save' stage to be enabled.")
            elif choice.isdigit():
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_text_stages):
                        stage_name = all_text_stages[idx]
                        if stage_name in self.state.stages_to_run:
                            self.state.stages_to_run.remove(stage_name)
                            ConsoleOutput.info(f"Stage '{stage_name}' disabled.")
                            # Check audio dependency if 'save' is disabled
                            if stage_name == 'save' and self.state.run_audio_generation:
                                 ConsoleOutput.warning("Disabling 'Save' stage, which is required for Audio Generation.")
                        else:
                            # Add stage back in the correct order
                            temp_stages = []
                            stages_set = set(self.state.stages_to_run)
                            stages_set.add(stage_name)
                            for s in PIPELINE_STAGES: # Iterate in predefined order
                                if s in stages_set:
                                    temp_stages.append(s)
                            self.state.stages_to_run = temp_stages
                            ConsoleOutput.info(f"Stage '{stage_name}' enabled.")
                    else:
                        ConsoleOutput.warning("Invalid stage number.")
                except ValueError:
                    ConsoleOutput.warning("Invalid input.")
            else:
                ConsoleOutput.warning("Invalid command.")

        ConsoleOutput.success("Stages selection confirmed.")
        return MenuAction.CONTINUE # Go back to main menu


    def _set_provider_key(self, provider: str):
        """Sets or clears an API key for a given provider."""
        ConsoleOutput.section(f"Set API Key for {provider.replace('_', ' ').title()}")
        current_key = self.state.cloud_api_keys.get(provider)
        masked_key = ""
        if current_key:
            masked_key = '*' * (len(current_key) - 4) + current_key[-4:] if len(current_key) > 4 else '*' * len(current_key)
            print(f"Current key: {masked_key}")
        else:
            print("Current key: Not set")

        new_key = input("Enter new API key (leave blank to clear, 'keep' to keep current): ").strip()

        if new_key.lower() == 'keep':
            ConsoleOutput.info("Keeping current key.")
            input("Press Enter...")
            return MenuAction.CONTINUE

        final_key = new_key if new_key else None # Use None to signal clearing

        if self.config_manager.save_cloud_key(provider, final_key):
             # Update runtime state immediately after successful save
             self.state.cloud_api_keys = self.config_manager.load_cloud_keys()
             action = "cleared" if final_key is None else "saved"
             ConsoleOutput.success(f"API key for {provider} {action}.")
        else:
             ConsoleOutput.error(f"Failed to save/clear key for {provider}.")

        input("Press Enter to continue...")
        return MenuAction.CONTINUE


    def _view_keys(self):
        """Displays which providers have keys configured."""
        ConsoleOutput.section("Configured API Keys")
        # Always reload from file to show the latest saved state
        configured_keys = self.config_manager.load_cloud_keys()
        self.state.cloud_api_keys = configured_keys # Sync runtime state

        if not configured_keys:
            print("No API keys are currently configured in the file.")
            print(f"(Config file location: {self.config_manager.cloud_keys_file})")
        else:
            print("Providers with keys configured:")
            for provider, key in configured_keys.items():
                 # Mask the key for display
                 masked_key = '*' * (len(key) - 4) + key[-4:] if len(key) > 4 else '*' * len(key)
                 print(f"  - {provider.replace('_', ' ').title():<15}: {masked_key}")
            print(f"\nKeys loaded from: {self.config_manager.cloud_keys_file}")

        input("\nPress Enter to continue...")
        return MenuAction.CONTINUE

    # --- Model Selection (Refined) ---
    def _set_text_model_menu(self):
         """Top-level menu to choose between local and cloud text models."""
         while True:
             ConsoleOutput.section("Set Text Processing Model")
             print("  1. Use Local Model (Hugging Face Transformers / GGUF)")
             print("  2. Use Cloud Model (OpenAI, Anthropic, Google, etc.)")
             current_model_display = f"{self.state.text_model_provider}: {self.state.text_model_specifier}"
             print(f"\nCurrent Model: {current_model_display}")
             print("\n  b. Back to Model Settings Menu")
             print("-" * 60)

             choice = input("Select option (1-2, or b): ").strip().lower()

             result = MenuAction.CONTINUE # Default action if sub-menu returns
             if choice == '1':
                 result = self._select_local_text_model() # This now returns action
             elif choice == '2':
                 result = self._select_cloud_text_model() # This now returns action
             elif choice == 'b':
                 return MenuAction.BACK # Go back to Model Settings Menu
             else:
                 ConsoleOutput.warning("Invalid choice.")

             # If sub-menu returned BACK, continue this loop, otherwise return to parent
             if result != MenuAction.BACK:
                  return MenuAction.CONTINUE # Return to Model Settings Menu after selection


    def _select_local_text_model(self) -> MenuAction: # Return MenuAction
        """Handles selection of a local HF model or GGUF."""
        while True:
            ConsoleOutput.subsection("Select Local Text Model")
            print("  1. Select from Predefined List (Recommended)")
            print("  2. Browse Hugging Face Hub (Transformers)")
            print("  3. Enter Hugging Face Model ID directly (Transformers)")
            if LLAMACPP_AVAILABLE:
                print("  4. Select GGUF Model from cache")
                print("  5. Enter path to GGUF Model file")

            current_provider = self.state.text_model_provider
            current_specifier = self.state.text_model_specifier
            current_local = f"{current_provider}: {current_specifier}" if current_provider in ['local', 'local_gguf'] else 'N/A (Cloud model selected)'
            print(f"\nCurrent Local Model: {current_local}")
            print("\n  b. Back to Text Model Type Selection")
            print("-" * 60)

            choice = input("Select option: ").strip().lower()
            new_model_id = None
            new_provider = "local" # Default for HF models

            if choice == '1':
                predefined = self.registry.list_models(predefined_only=True, sort_by="name")
                if not predefined:
                    ConsoleOutput.warning("No predefined models found in registry.")
                    input("Press Enter...")
                    continue
                print("\nAvailable predefined models:")
                model_map = {}
                for i, entry in enumerate(predefined):
                    print(f"  {i+1}. {entry.name} ({entry.model_id})")
                    model_map[str(i+1)] = entry.model_id
                model_map['b'] = 'back'

                model_choice = input(f"Select model (1-{len(predefined)} or b): ").strip().lower()
                if model_choice == 'b': continue
                if model_choice in model_map:
                    new_model_id = model_map[model_choice]
                    new_provider = "local" # Predefined are transformers models
                else:
                    ConsoleOutput.warning("Invalid selection.")

            elif choice == '2':
                browser = InteractiveModelBrowser(self.model_hub)
                selected_hf_model: Optional[HFModelInfo] = browser.browse(task="text-generation", library="transformers")
                if selected_hf_model:
                    new_model_id = selected_hf_model.model_id
                    new_provider = "local"
                    # Add/update in registry if selected
                    if not self.registry.get_model(new_model_id):
                        self.registry.add_from_model_info(selected_hf_model)
                        ConsoleOutput.info(f"Added {new_model_id} to local registry.")

            elif choice == '3':
                model_id_input = input("Enter Hugging Face Model ID (e.g., Org/ModelName): ").strip()
                if model_id_input:
                     if '/' not in model_id_input:
                         ConsoleOutput.warning("Invalid HF Model ID format (expected 'Org/ModelName').")
                     else:
                        # Basic check if model exists on Hub (optional but helpful)
                        hf_info = self.model_hub.get_model_info(model_id_input)
                        if hf_info:
                            new_model_id = model_id_input
                            new_provider = "local"
                            if not self.registry.get_model(new_model_id):
                                self.registry.add_from_model_info(hf_info)
                        else:
                             ConsoleOutput.error(f"Model '{model_id_input}' not found on Hugging Face Hub.")
                             input("Press Enter...")


            elif choice == '4' and LLAMACPP_AVAILABLE:
                manager = GGUFModelManager(cache_dir=CACHE_DIR / "gguf_models")
                gguf_models = manager.list_available_models()
                if not gguf_models:
                    ConsoleOutput.warning("No GGUF models found in cache.")
                    print(f"(Searched in: {manager.cache_dir})")
                    input("Press Enter...")
                    continue
                print("\nAvailable GGUF models:")
                model_map = {}
                for i, p in enumerate(gguf_models):
                     print(f"  {i+1}. {p.name}")
                     model_map[str(i+1)] = p
                model_map['b'] = 'back'

                gguf_choice = input(f"Select GGUF model number (1-{len(gguf_models)} or b): ").strip().lower()
                if gguf_choice == 'b': continue
                if gguf_choice in model_map:
                    new_model_id = str(model_map[gguf_choice].resolve())
                    new_provider = "local_gguf"
                else:
                     ConsoleOutput.warning("Invalid number.")

            elif choice == '5' and LLAMACPP_AVAILABLE:
                gguf_path_str = input("Enter full path to GGUF model file: ").strip()
                gguf_path = Path(os.path.expanduser(gguf_path_str))
                if gguf_path.is_file() and gguf_path.suffix.lower() in ['.gguf', '.ggml', '.bin']:
                    new_model_id = str(gguf_path.resolve())
                    new_provider = "local_gguf"
                else:
                    ConsoleOutput.error("Invalid GGUF file path or extension.")
                    input("Press Enter...")

            elif choice == 'b':
                return MenuAction.BACK # Go back to the text model type selection menu

            else:
                ConsoleOutput.warning("Invalid choice.")
                continue # Re-prompt this menu

            # If a new model was selected, update state and return CONTINUE
            if new_model_id:
                old_provider = self.state.text_model_provider
                old_specifier = self.state.text_model_specifier
                self.state.text_model_provider = new_provider
                self.state.text_model_specifier = new_model_id
                ConsoleOutput.success(f"Text model set to: {new_provider}: {self.state.text_model_specifier}")

                # Ask to reset hyperparams only if the model *actually* changed
                if old_provider != new_provider or old_specifier != new_model_id:
                     reset_hp = input("Reset hyperparameters to default for new model? (y/n) [y]: ").strip().lower()
                     if reset_hp != 'n':
                         self.state.hyperparameters = HyperparameterConfig()
                         ConsoleOutput.info("Hyperparameters reset to defaults.")
                input("Press Enter to continue...")
                return MenuAction.CONTINUE # Signal successful selection, go back to calling menu


    def _select_cloud_text_model(self) -> MenuAction: # Return MenuAction
        """Handles selection of a cloud provider and model."""
        while True:
            ConsoleOutput.subsection("Select Cloud Text Model")
            # Filter providers with keys
            configured_providers = [p for p in SUPPORTED_CLOUD_PROVIDERS if p in self.state.cloud_api_keys]

            if not configured_providers:
                ConsoleOutput.warning("No cloud providers configured with API keys.")
                print("Please configure keys in the 'Cloud Provider Settings' menu.")
                input("Press Enter to continue...")
                return MenuAction.BACK # Back to text model type selection

            print("Configured Cloud Providers:")
            provider_map = {}
            for i, provider in enumerate(configured_providers):
                print(f"  {i+1}. {provider.replace('_', ' ').title()}")
                provider_map[str(i+1)] = provider

            print("\n  b. Back to Text Model Type Selection")
            print("-" * 60)

            provider_choice = input(f"Select provider (1-{len(provider_map)} or b): ").strip().lower()

            if provider_choice == 'b':
                return MenuAction.BACK
            elif provider_choice in provider_map:
                selected_provider = provider_map[provider_choice]

                # Provide better examples based on provider
                example_prompt = "e.g., "
                if selected_provider == "openai": example_prompt += "gpt-4o / gpt-4-turbo / gpt-3.5-turbo"
                elif selected_provider == "anthropic": example_prompt += "claude-3-5-sonnet-20240620 / claude-3-opus-20240229"
                elif selected_provider == "google": example_prompt += "models/gemini-1.5-pro-latest"
                # Add examples for other providers when implemented
                else: example_prompt = "Enter model name/ID specific to this provider"

                model_specifier = input(f"Enter model name/ID for {selected_provider.title()} ({example_prompt}): ").strip()

                if model_specifier:
                    old_provider = self.state.text_model_provider
                    old_specifier = self.state.text_model_specifier
                    self.state.text_model_provider = selected_provider
                    self.state.text_model_specifier = model_specifier
                    ConsoleOutput.success(f"Text model set to: {selected_provider}: {model_specifier}")

                    if old_provider != selected_provider or old_specifier != model_specifier:
                        reset_hp = input("Reset hyperparameters to default? (y/n) [y]: ").strip().lower()
                        if reset_hp != 'n':
                            self.state.hyperparameters = HyperparameterConfig()
                            ConsoleOutput.info("Hyperparameters reset to defaults.")
                    input("Press Enter to continue...")
                    return MenuAction.CONTINUE # Success, return to calling menu
                else:
                    ConsoleOutput.warning("Model name cannot be empty.")
            else:
                ConsoleOutput.warning("Invalid provider selection.")

            # Pause before re-prompting this menu
            input("Press Enter to continue...")


    def _set_text_hyperparameters(self):
        """Opens the interactive hyperparameter editor."""
        ConsoleOutput.section(f"Configure Text Hyperparameters for {self.state.text_model_provider}: {self.state.text_model_specifier}")
        editor = InteractiveHyperparameterEditor(self.state.hyperparameters)
        self.state.hyperparameters = editor.edit() # This starts the interactive editing loop
        ConsoleOutput.success("Text hyperparameters updated for this session.")
        input("Press Enter to return...")
        return MenuAction.CONTINUE # Stay in Model Settings menu

    def _set_local_hardware_config(self):
        """Interactive configuration for local model hardware settings."""
        if self.state.text_model_provider not in ["local", "local_gguf"]:
             ConsoleOutput.warning("Hardware settings only apply to local models.")
             input("Press Enter...")
             return MenuAction.CONTINUE

        ConsoleOutput.section("Configure Local Model Hardware Settings")

        while True:
            ConsoleOutput.header("Local Hardware Settings")
            print(f"  Provider: {self.state.text_model_provider}")
            print(f"  Model: {self.state.text_model_specifier}")
            print("-" * 60)

            is_gguf = (self.state.text_model_provider == 'local_gguf')

            options = {}
            if is_gguf:
                print("  Note: Settings for GGUF model (via llama-cpp-python)")
                print(f"  1. GPU Layers (n_gpu_layers) : {self.state.gpu_layers} (-1 = auto/all, 0 = CPU only)")
                options['1'] = 'gpu_layers'
                print(f"  2. Quantization (inferred) : (Determined by GGUF file)")
            else: # Transformers
                print("  Note: Settings for Hugging Face Transformers model")
                print(f"  1. Quantization           : {self.state.quantization} (Options: {', '.join(QUANTIZATION_OPTIONS)})")
                options['1'] = 'quantization'
                print(f"  2. Max GPU Memory         : {self.state.max_gpu_memory} (e.g., 8GB, 10000MB per GPU)")
                options['2'] = 'max_gpu_memory'
                print(f"  3. Max CPU Memory (Offload) : {self.state.max_cpu_memory} (e.g., 30GB)")
                options['3'] = 'max_cpu_memory'

            print("\n  b. Back to Model Settings")
            print("-" * 60)

            choice = input("Select setting to change (or b to back): ").strip().lower()

            try:
                if choice == 'b':
                    return MenuAction.BACK

                if choice not in options:
                     ConsoleOutput.warning("Invalid selection.")
                     continue

                setting_to_change = options[choice]

                if setting_to_change == 'gpu_layers':
                    val = input(f"Enter number of GPU layers (-1 for auto/all, 0 for CPU) [current: {self.state.gpu_layers}]: ").strip()
                    if not val: continue
                    self.state.gpu_layers = int(val) # Let int() raise ValueError
                    ConsoleOutput.success(f"GPU Layers set to: {self.state.gpu_layers}")
                elif setting_to_change == 'quantization':
                    val = input(f"Enter quantization ({', '.join(QUANTIZATION_OPTIONS)}) [current: {self.state.quantization}]: ").strip().lower()
                    if not val: continue
                    if val in QUANTIZATION_OPTIONS:
                        self.state.quantization = val
                        ConsoleOutput.success(f"Quantization set to: {val}")
                    else:
                        ConsoleOutput.warning(f"Invalid option. Must be one of: {', '.join(QUANTIZATION_OPTIONS)}")
                elif setting_to_change == 'max_gpu_memory':
                     val = input(f"Enter Max GPU Memory per GPU (e.g., 8GiB, 10GB) [current: {self.state.max_gpu_memory}]: ").strip()
                     if not val: continue
                     # Basic validation, refine as needed
                     if re.match(r"^\d+(\.\d+)?(GB|GiB|MB|MiB)$", val, re.IGNORECASE):
                         self.state.max_gpu_memory = val
                         ConsoleOutput.success(f"Max GPU Memory set to: {val}")
                     else:
                         ConsoleOutput.warning("Invalid format. Use numbers followed by GB, GiB, MB, or MiB.")
                elif setting_to_change == 'max_cpu_memory':
                    val = input(f"Enter Max CPU Memory (e.g., 30GB) [current: {self.state.max_cpu_memory}]: ").strip()
                    if not val: continue
                    if re.match(r"^\d+(\.\d+)?(GB|GiB|MB|MiB)$", val, re.IGNORECASE):
                        self.state.max_cpu_memory = val
                        ConsoleOutput.success(f"Max CPU Memory set to: {val}")
                    else:
                         ConsoleOutput.warning("Invalid format. Use numbers followed by GB, GiB, MB, or MiB.")

            except ValueError:
                ConsoleOutput.warning("Invalid input value type (e.g., expected a number).")
            except Exception as e:
                 ConsoleOutput.error(f"Error updating setting: {e}")


    def _set_audio_model_menu(self):
        """Top-level menu to choose between local and cloud audio models."""
        while True:
            ConsoleOutput.section("Set Audio Generation Model (TTS)")
            print(f"Audio Generation Currently: {'ENABLED' if self.state.run_audio_generation else 'DISABLED'}")
            print("  1. Use Local Model (Hugging Face)")
            print("  2. Use Cloud Model (e.g., OpenAI TTS)")
            print("  T. Toggle Audio Generation ON/OFF")
            current_model_display = f"{self.state.audio_model_provider}: {self.state.audio_model_specifier}"
            print(f"\nCurrent Model: {current_model_display}")
            print("\n  b. Back to Model Settings Menu")
            print("-" * 60)

            choice = input("Select option (1-2, T, or b): ").strip().lower()

            result = MenuAction.CONTINUE # Default
            if choice == '1':
                result = self._select_local_audio_model()
            elif choice == '2':
                result = self._select_cloud_audio_model()
            elif choice == 't':
                 self.state.run_audio_generation = not self.state.run_audio_generation
                 status = "ENABLED" if self.state.run_audio_generation else "DISABLED"
                 ConsoleOutput.info(f"Audio Generation {status}.")
                 input("Press Enter...") # Pause to show status
            elif choice == 'b':
                return MenuAction.BACK
            else:
                ConsoleOutput.warning("Invalid choice.")

            # If sub-menu returned BACK, continue loop, else return to parent
            if result != MenuAction.BACK:
                 return MenuAction.CONTINUE


    def _select_local_audio_model(self) -> MenuAction: # Return action
        """Handles selection of a local HF TTS model."""
        while True:
            ConsoleOutput.subsection("Select Local Audio Model (TTS)")
            print("  1. Browse Hugging Face Hub (text-to-speech)")
            print("  2. Enter Hugging Face Model ID directly")
            # Example popular/good models
            print("     Examples: microsoft/speecht5_tts, suno/bark-small, facebook/mms-tts-eng")
            current_local = self.state.audio_model_specifier if self.state.audio_model_provider == 'local' else 'N/A'
            print(f"\nCurrent Local Model: {current_local}")
            print("\n  b. Back to Audio Model Type Selection")
            print("-" * 60)

            choice = input("Select option: ").strip().lower()
            new_model_id = None

            if choice == '1':
                browser = InteractiveModelBrowser(self.model_hub)
                # Filter specifically for TTS tasks
                selected_hf_model: Optional[HFModelInfo] = browser.browse(task="text-to-speech")
                if selected_hf_model:
                     new_model_id = selected_hf_model.model_id
            elif choice == '2':
                 model_id_input = input("Enter Hugging Face Model ID (TTS): ").strip()
                 if model_id_input:
                     if '/' in model_id_input and len(model_id_input) > 3:
                         # Optional: Add basic check if model exists on Hub
                         info = self.model_hub.get_model_info(model_id_input)
                         if info:
                              new_model_id = model_id_input
                         else:
                              ConsoleOutput.error(f"Model '{model_id_input}' not found on Hub.")
                              input("Press Enter...")
                     else:
                         ConsoleOutput.warning("Invalid model ID format (expected Org/ModelName).")
            elif choice == 'b':
                 return MenuAction.BACK

            else:
                ConsoleOutput.warning("Invalid choice.")
                continue

            if new_model_id:
                self.state.audio_model_provider = "local"
                self.state.audio_model_specifier = new_model_id
                # self.state.audio_config.model_id = new_model_id # AudioConfig doesn't need model_id
                ConsoleOutput.success(f"Audio model set to: local: {self.state.audio_model_specifier}")
                input("Press Enter...")
                return MenuAction.CONTINUE


    def _select_cloud_audio_model(self) -> MenuAction: # Return action
        """Handles selection of a cloud TTS provider and model."""
        while True:
            ConsoleOutput.subsection("Select Cloud Audio Model (TTS)")
            # Filter providers that have keys AND are in TTS_PROVIDERS
            configured_tts_providers = [
                p for p in TTS_PROVIDERS
                if p != 'local' and p in self.state.cloud_api_keys
            ]

            if not configured_tts_providers:
                ConsoleOutput.warning("No cloud TTS providers configured with API keys.")
                supported_cloud_tts = [p for p in TTS_PROVIDERS if p != 'local']
                print(f"Supported & Configurable Cloud TTS Providers: {', '.join(supported_cloud_tts)}")
                print("Please configure keys in the 'Cloud Provider Settings' menu.")
                input("Press Enter to continue...")
                return MenuAction.BACK

            print("Configured & Supported Cloud TTS Providers:")
            provider_map = {}
            for i, provider in enumerate(configured_tts_providers):
                print(f"  {i+1}. {provider.replace('_', ' ').title()}")
                provider_map[str(i+1)] = provider
            print("\n  b. Back to Audio Model Type Selection")
            print("-" * 60)

            provider_choice = input(f"Select provider (1-{len(provider_map)} or b): ").strip().lower()

            if provider_choice == 'b':
                return MenuAction.BACK
            elif provider_choice in provider_map:
                selected_provider = provider_map[provider_choice]
                model_specifier = ""

                # Provider-specific model examples/prompts
                if selected_provider == "openai":
                    model_specifier = input(f"Enter model name (e.g., tts-1, tts-1-hd) [default: tts-1]: ").strip()
                    if not model_specifier: model_specifier = "tts-1"
                # Add prompts for other cloud TTS providers here when implemented
                # elif selected_provider == "google_tts": ...
                else:
                    # Generic prompt if no specific examples known
                    model_specifier = input(f"Enter model name/ID for {selected_provider.title()}: ").strip()

                if model_specifier:
                    self.state.audio_model_provider = selected_provider
                    self.state.audio_model_specifier = model_specifier
                    # self.state.audio_config.model_id = None # Not needed
                    ConsoleOutput.success(f"Audio model set to: {selected_provider}: {model_specifier}")
                    input("Press Enter...")
                    return MenuAction.CONTINUE
                else:
                    ConsoleOutput.warning("Model name cannot be empty.")
            else:
                ConsoleOutput.warning("Invalid provider selection.")

            # Pause before re-prompting this menu
            input("Press Enter to continue...")


    def _set_audio_config(self):
        """Interactive configuration for AudioConfig."""
        ConsoleOutput.section("Configure Audio Settings")
        cfg = self.state.audio_config

        while True:
            ConsoleOutput.header("Audio Settings")
            is_local = self.state.audio_model_provider == 'local'
            print(f"  Provider: {self.state.audio_model_provider}")
            print(f"  Model: {self.state.audio_model_specifier}")
            print("-" * 60)
            print(f"  1. Output Format : {cfg.output_format} (wav, mp3, flac)")
            print(f"  2. Sample Rate   : {cfg.sample_rate} Hz (Target SR; cloud/local models may resample)")
            print(f"  3. Speed Factor  : {cfg.speed:.1f} (1.0 = normal; applied via post-processing or API)")
            print(f"  4. Pitch Shift   : {cfg.pitch_shift} semitones (0 = normal; post-processing only)")
            print(f"  5. Volume Norm.  : {'Enabled' if cfg.volume_normalize else 'Disabled'} (Post-processing)")

            option_num = 6
            local_options = {}
            cloud_options = {}

            if not is_local:
                 print(f"  {option_num}. Cloud Voice   : {cfg.cloud_voice} (Provider-specific, e.g., 'alloy' for OpenAI)")
                 cloud_options[str(option_num)] = 'cloud_voice'
                 option_num += 1
            else:
                 print(f"  {option_num}. Half Precision: {'Enabled' if cfg.use_half_precision else 'Disabled'} (Local CUDA only)")
                 local_options[str(option_num)] = 'use_half_precision'
                 option_num += 1
                 print(f"  {option_num}. Speaker Embed.: {cfg.speaker_embedding or 'Default/Not Used'} (Local SpeechT5 path/ID)")
                 local_options[str(option_num)] = 'speaker_embedding'
                 option_num += 1
                 print(f"  {option_num}. Text Chunk Size:{cfg.chunk_size} chars (Local processing)")
                 local_options[str(option_num)] = 'chunk_size'
                 option_num += 1

            print("\n  s. Save audio settings to preset")
            print("  l. Load audio settings from preset")
            print("  b. Back to Model Settings Menu")
            print("-" * 60)

            choice = input("Select setting to change (or command): ").strip().lower()

            try:
                if choice == 'b':
                    return MenuAction.BACK
                elif choice == 's':
                    preset_name = input("Enter name for audio preset: ").strip()
                    if preset_name:
                        # Sanitize name
                        preset_name = re.sub(r'[\\/*?:"<>|]', '', preset_name).replace(" ", "_")
                        if not preset_name:
                             ConsoleOutput.error("Invalid preset name after sanitization.")
                             continue
                        # Use ConfigManager to save
                        save_data = cfg.to_dict() # Get dict representation
                        if self.config_manager.save_config(preset_name, save_data, dir_type="audio"):
                            ConsoleOutput.success(f"Audio preset '{preset_name}' saved.")
                        else:
                            ConsoleOutput.error("Failed to save preset.")
                    else:
                        ConsoleOutput.warning("Invalid name, save cancelled.")
                elif choice == 'l':
                    presets = self.config_manager.list_configs(dir_type="audio")
                    if not presets:
                        ConsoleOutput.warning("No saved audio presets found.")
                        continue
                    print("Available audio presets:")
                    for i, name in enumerate(presets): print(f"  {i+1}. {name}")
                    preset_choice = input(f"Select preset (1-{len(presets)} or b): ").strip().lower()
                    if preset_choice == 'b': continue
                    if preset_choice.isdigit():
                         idx = int(preset_choice) - 1
                         if 0 <= idx < len(presets):
                              loaded_data = self.config_manager.load_config(presets[idx], dir_type="audio")
                              if loaded_data:
                                   # Update state's audio_config from loaded data
                                   try:
                                        self.state.audio_config = AudioConfig(**loaded_data)
                                        ConsoleOutput.success(f"Loaded audio preset '{presets[idx]}'.")
                                   except TypeError as te:
                                        ConsoleOutput.error(f"Error applying preset '{presets[idx]}': {te}. Preset might be incompatible.")
                                        logger.error(f"Audio preset load error", exc_info=True)
                              else:
                                   ConsoleOutput.error(f"Failed to load preset '{presets[idx]}'.")
                         else: ConsoleOutput.warning("Invalid selection.")
                    else: ConsoleOutput.warning("Invalid input.")

                elif choice == '1': # Output Format
                    fmt = input(f"Enter format (wav, mp3, flac) [current: {cfg.output_format}]: ").strip().lower()
                    if fmt in ["wav", "mp3", "flac"]: cfg.output_format = fmt
                    elif fmt: ConsoleOutput.warning("Invalid format.")
                elif choice == '2': # Sample Rate
                    sr = input(f"Enter target sample rate (e.g., 16000, 24000) [current: {cfg.sample_rate}]: ").strip()
                    if sr.isdigit() and 8000 <= int(sr) <= 48000: cfg.sample_rate = int(sr)
                    elif sr: ConsoleOutput.warning("Invalid sample rate (use 8000-48000).")
                elif choice == '3': # Speed Factor
                    spd_str = input(f"Enter speed factor (e.g., 1.0, 1.2, 0.8) [current: {cfg.speed:.1f}]: ").strip()
                    if spd_str:
                         spd_f = float(spd_str)
                         if 0.5 <= spd_f <= 2.0: cfg.speed = spd_f
                         else: ConsoleOutput.warning("Speed factor out of range (0.5-2.0).")
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
                elif choice in cloud_options:
                     setting = cloud_options[choice]
                     if setting == 'cloud_voice':
                          voice = input(f"Enter provider-specific voice name [current: {cfg.cloud_voice}]: ").strip()
                          if voice: cfg.cloud_voice = voice
                elif choice in local_options:
                     setting = local_options[choice]
                     if setting == 'use_half_precision':
                          if TORCH_AVAILABLE and torch.cuda.is_available():
                               cfg.use_half_precision = not cfg.use_half_precision
                               ConsoleOutput.info(f"Half precision {'Enabled' if cfg.use_half_precision else 'Disabled'}.")
                          else:
                               ConsoleOutput.warning("Half precision requires PyTorch and CUDA.")
                               cfg.use_half_precision = False
                     elif setting == 'speaker_embedding':
                          spk = input(f"Enter path/ID for speaker embedding (blank=clear) [current: {cfg.speaker_embedding or 'Default'}]: ").strip()
                          cfg.speaker_embedding = spk if spk else None
                          ConsoleOutput.info(f"Speaker embedding set to: {cfg.speaker_embedding or 'Default/Not Used'}")
                     elif setting == 'chunk_size':
                          cs_str = input(f"Enter text chunk size for TTS (chars) [current: {cfg.chunk_size}]: ").strip()
                          if cs_str.isdigit() and 500 <= int(cs_str) <= 10000: cfg.chunk_size = int(cs_str)
                          elif cs_str: ConsoleOutput.warning("Invalid chunk size (use 500-10000).")

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
        try:
             # Resolve relative paths against BASE_DIR or CWD
             if not self.state.output_dir.is_absolute():
                  resolved_dir = (BASE_DIR / self.state.output_dir).resolve()
             else:
                  resolved_dir = self.state.output_dir.resolve()

             # Attempt to create and test writability
             resolved_dir.mkdir(parents=True, exist_ok=True)
             test_file = resolved_dir / f".llamanote_test_{int(time.time())}"
             test_file.touch()
             test_file.unlink()
             self.state.output_dir = resolved_dir # Update state only if valid and writable

        except Exception as e:
             ConsoleOutput.warning(f"Could not use/create output directory '{self.state.output_dir}', defaulting to 'output'. Error: {e}")
             self.state.output_dir = (BASE_DIR / "output").resolve()
             try: # Ensure default is created
                  self.state.output_dir.mkdir(parents=True, exist_ok=True)
             except Exception as default_e:
                  ConsoleOutput.error(f"CRITICAL: Could not create default output directory '{self.state.output_dir}'. Cannot save files. Error: {default_e}")
                  # This is a critical error, might need to prevent running
                  input("Press Enter to continue (saving might fail)...")
                  return MenuAction.CONTINUE


        print(f"Current Output Directory: {self.state.output_dir}")
        if self.state.output_filename:
             print(f"Current Base Filename: {self.state.output_filename} (Timestamps/suffixes added automatically)")
        else:
             print("Current Base Filename: Default (derived from input filename)")

        new_dir_str = input(f"Enter new output directory path (leave blank to keep current): ").strip()
        if new_dir_str:
            try:
                # Resolve relative to current working directory or make absolute
                new_dir_path = Path(os.path.expanduser(new_dir_str))
                if not new_dir_path.is_absolute():
                    new_dir_path = Path.cwd() / new_dir_path
                new_dir_path = new_dir_path.resolve()

                new_dir_path.mkdir(parents=True, exist_ok=True)
                # Test writability again
                test_file = new_dir_path / f".llamanote_test_{int(time.time())}"
                test_file.touch()
                test_file.unlink()
                self.state.output_dir = new_dir_path
                ConsoleOutput.success(f"Output directory set to: {self.state.output_dir}")
            except Exception as e:
                ConsoleOutput.error(f"Invalid or unwritable directory path: {e}")
                input("Press Enter to continue...")

        num_inputs = len(self.state.input_files)
        if num_inputs == 1:
             current_name_prompt = f"(current: {self.state.output_filename})" if self.state.output_filename else "(blank for default based on input)"
             new_name = input(f"Enter base output filename (NO extension, {current_name_prompt}): ").strip()
             if new_name:
                 # Sanitize filename (allow alphanumeric, underscore, hyphen)
                 new_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', new_name)
                 self.state.output_filename = new_name
                 ConsoleOutput.success(f"Base output filename set to: {self.state.output_filename}")
             elif new_name == "" and self.state.output_filename is not None:
                 # User explicitly cleared the filename
                 ConsoleOutput.info("Using default output filename based on input.")
                 self.state.output_filename = None
             # If blank and was already None, do nothing
        elif num_inputs > 1:
             ConsoleOutput.info("Base output filename is ignored during batch processing (uses input names).")
             self.state.output_filename = None

        return MenuAction.CONTINUE

    def _save_configuration(self):
        """Saves the current AppState to a named preset."""
        ConsoleOutput.section("Save Configuration Preset")
        
        presets = self.config_manager.list_configs(dir_type="preset")
        if presets:
            print("Existing presets:")
            for p in presets:
                print(f"  - {p}")
        else:
            print("No existing presets found.")

        name = input("\nEnter name for this configuration (e.g., 'my_podcast_setup'): ").strip()
        if not name:
            ConsoleOutput.warning("Save cancelled. No name provided.")
            input("Press Enter to continue...")
            return MenuAction.CONTINUE
            
        name = re.sub(r'[\\/*?:"<>|]', '', name).replace(" ", "_")
        if not name:
            ConsoleOutput.error("Invalid name. Save cancelled.")
            input("Press Enter to continue...")
            return MenuAction.CONTINUE
            
        try:
            config_data = self.state.to_dict()
            # Remove input files from saved preset
            config_data["input_files"] = []
            success = self.config_manager.save_config(name, config_data, dir_type="preset")
            if success:
                ConsoleOutput.success(f"Configuration '{name}' saved successfully.")
            else:
                ConsoleOutput.error(f"Failed to save configuration '{name}'.")
        except Exception as e:
            ConsoleOutput.error(f"Error saving configuration: {e}")
            logger.error(f"Failed to save config '{name}'", exc_info=True)
            
        input("Press Enter to continue...")
        return MenuAction.CONTINUE

    def _load_configuration(self):
        """Loads an AppState from a saved preset."""
        ConsoleOutput.section("Load Configuration Preset")
        
        presets = self.config_manager.list_configs(dir_type="preset")
        if not presets:
            ConsoleOutput.warning("No saved presets found.")
            print(f"(Looked in: {self.config_manager.get_dir('preset')})")
            input("Press Enter to continue...")
            return MenuAction.CONTINUE

        print("Available presets:")
        for i, name in enumerate(presets):
            print(f"  {i+1}. {name}")
        print("\n  b. Back")
        print("-" * 60)
        
        choice = input(f"Select preset to load (1-{len(presets)} or b): ").strip().lower()
        
        if choice == 'b':
            return MenuAction.CONTINUE
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                selected_name = presets[idx]
                config_data = self.config_manager.load_config(selected_name, dir_type="preset")
                if config_data:
                    # Save current input files, as they are not part of the preset
                    current_inputs = self.state.input_files
                    self.state.from_dict(config_data)
                    self.state.input_files = current_inputs # Restore input files
                    
                    ConsoleOutput.success(f"Loaded preset '{selected_name}'.")
                    # Re-load API keys as they are not saved in presets
                    self.state.cloud_api_keys = self.config_manager.load_cloud_keys()
                else:
                    ConsoleOutput.error(f"Failed to load preset '{selected_name}'.")
            else:
                ConsoleOutput.warning("Invalid selection.")
        except ValueError:
            ConsoleOutput.warning("Invalid input. Please enter a number.")
        except Exception as e:
            ConsoleOutput.error(f"Error loading preset: {e}")
            logger.error(f"Failed to load preset", exc_info=True)

        input("Press Enter to continue...")
        return MenuAction.CONTINUE # Return to main menu

    def _view_current_setup(self):
        """Displays the current configuration state."""
        ConsoleOutput.section("Current Processing Setup Summary")

        # Reload keys just in case they were changed outside
        self.state.cloud_api_keys = self.config_manager.load_cloud_keys()

        # Input Files
        print(f"Input File(s):")
        if self.state.input_files:
             max_files_to_show = 5
             for i, f in enumerate(self.state.input_files):
                 if i < max_files_to_show:
                     # Show relative path if possible, else full path
                     try: display_path = f.relative_to(Path.cwd())
                     except ValueError: display_path = f.resolve()
                     print(f"  - {display_path}")
                 elif i == max_files_to_show:
                     print(f"  ... and {len(self.state.input_files) - max_files_to_show} more.")
                     break
        else:
             print("  ⚠️ None selected. Please select input file(s).")

        # Output Settings
        print(f"\nOutput Directory: {self.state.output_dir.resolve()}")
        # Get output format from pipeline config helper
        temp_pipeline_config = self.state.get_pipeline_config()
        text_output_format = temp_pipeline_config.output_format or 'md' # Default to md if somehow None

        if len(self.state.input_files) == 1 and self.state.output_filename:
             example_suffix = f"_{self.state.processing_mode.value}"
             example_ext = f".{text_output_format}"
             ts_example = "_<timestamp>" if TIMESTAMP_OUTPUTS else ""
             print(f"Base Output Filename: {self.state.output_filename}")
             print(f"  (Example text output: {self.state.output_filename}{example_suffix}{ts_example}{example_ext})")
        elif len(self.state.input_files) >= 1:
             example_input_name = self.state.input_files[0].stem
             example_suffix = f"_{self.state.processing_mode.value}"
             example_ext = f".{text_output_format}"
             ts_example = "_<timestamp>" if TIMESTAMP_OUTPUTS else ""
             print(f"Base Output Filename: Default (based on input)")
             print(f"  (Example text output for '{self.state.input_files[0].name}': {example_input_name}{example_suffix}{ts_example}{example_ext})")


        # Text Processing
        print(f"\n--- Text Processing ---")
        print(f"Mode: {self.state.processing_mode.value}")
        print(f"Provider: {self.state.text_model_provider}")
        model_display = self.state.text_model_specifier
        if self.state.text_model_provider == 'local_gguf':
             model_display = Path(model_display).name # Show only filename for GGUF
        print(f"Model: {model_display}")

        # Show local hardware settings if local
        if self.state.text_model_provider == "local":
            print(f"Local Hardware Config (Transformers):")
            print(f"  - Quantization: {self.state.quantization}")
            print(f"  - Max GPU Mem : {self.state.max_gpu_memory} | Max CPU Mem: {self.state.max_cpu_memory}")
        elif self.state.text_model_provider == "local_gguf":
            print(f"Local Hardware Config (GGUF):")
            print(f"  - GPU Layers  : {self.state.gpu_layers}")

        print(f"Hyperparameters (Key Settings):")
        hp_dict = self.state.hyperparameters.to_dict()
        print(f"  - temperature: {hp_dict.get('temperature', 'Default'):.2f}")
        print(f"  - top_p: {hp_dict.get('top_p', 'Default'):.2f}")
        print(f"  - max_new_tokens: {hp_dict.get('max_new_tokens') or 'Auto'}")
        print(f"  (Use 'Model Settings -> Hyperparameters' menu to see/edit all)")

        # Stages
        print(f"\n--- Pipeline Stages ---")
        text_stages_to_run = [s.capitalize() for s in PIPELINE_STAGES if s in self.state.stages_to_run]
        print(f"Text Stages to Run: {', '.join(text_stages_to_run) if text_stages_to_run else 'None'}")
        audio_status = '✅ Yes' if self.state.run_audio_generation else '❌ No'
        audio_dep_met = 'save' in self.state.stages_to_run
        audio_note = "" if audio_dep_met or not self.state.run_audio_generation else " (⚠️ Requires 'Save' stage)"
        print(f"Generate Audio After Text: {audio_status}{audio_note}")

        # Audio Generation (if enabled)
        if self.state.run_audio_generation:
             print(f"\n--- Audio Generation ---")
             print(f"Provider: {self.state.audio_model_provider}")
             print(f"Model: {self.state.audio_model_specifier}")
             cfg = self.state.audio_config
             print(f"Output Format: {cfg.output_format}")
             print(f"Sample Rate: {cfg.sample_rate} Hz")
             if self.state.audio_model_provider != 'local':
                  print(f"Cloud Voice: {cfg.cloud_voice}")
             else:
                  print(f"Half Precision: {'Enabled' if cfg.use_half_precision else 'Disabled'}")
             print(f"  (Use 'Model Settings -> Configure Audio' menu to see/edit all)")

        # Warnings/Errors Check
        print("-" * 60)
        errors = []
        warnings = []
        text_stages_selected = bool(text_stages_to_run) # Check if list is non-empty

        if not self.state.input_files: errors.append("Input files are missing.")
        if not text_stages_selected and not self.state.run_audio_generation: errors.append("No processing or audio stages selected.")

        if text_stages_selected:
            if not self.state.text_model_specifier: errors.append("No text model specified.")
            elif self.state.text_model_provider not in ['local', 'local_gguf'] and self.state.text_model_provider not in self.state.cloud_api_keys:
                 errors.append(f"API key for text provider '{self.state.text_model_provider}' is missing.")
            if self.state.text_model_provider == 'local_gguf' and not Path(self.state.text_model_specifier).is_file():
                 errors.append(f"GGUF model file not found: {self.state.text_model_specifier}")

        if self.state.run_audio_generation:
            if not self.state.audio_model_specifier: errors.append("No audio model specified.")
            elif self.state.audio_model_provider != 'local' and self.state.audio_model_provider not in self.state.cloud_api_keys:
                 errors.append(f"API key for audio provider '{self.state.audio_model_provider}' is missing.")
            if not audio_dep_met:
                 warnings.append("Audio generation enabled, but 'Save' stage disabled. Audio might fail.")

        if errors:
             ConsoleOutput.error("Setup incomplete - Cannot Run:")
             for err in errors: print(f"  - {err}")
        if warnings:
             ConsoleOutput.warning("Potential Issues:")
             for warn in warnings: print(f"  - {warn}")
        if not errors and not warnings:
             ConsoleOutput.success("Setup appears complete and valid.")

        input("\nPress Enter to continue...")
        return MenuAction.CONTINUE


    def _confirm_and_run(self) -> MenuAction:
        """Confirms the setup and returns RUN action if confirmed."""
        ConsoleOutput.section("Confirm and Run Processing")

        # Rerun validation logic from _view_current_setup
        errors = []
        warnings = []
        text_stages_selected = any(s in self.state.stages_to_run for s in PIPELINE_STAGES)
        audio_dep_met = 'save' in self.state.stages_to_run

        if not self.state.input_files: errors.append("Input files missing.")
        if not text_stages_selected and not self.state.run_audio_generation: errors.append("No stages selected.")

        if text_stages_selected:
            if not self.state.text_model_specifier: errors.append("Text model missing.")
            elif self.state.text_model_provider not in ['local', 'local_gguf'] and self.state.text_model_provider not in self.state.cloud_api_keys:
                 errors.append(f"API key missing for text provider '{self.state.text_model_provider}'.")
            if self.state.text_model_provider == 'local_gguf' and not Path(self.state.text_model_specifier).is_file():
                 errors.append(f"GGUF model file not found: {Path(self.state.text_model_specifier).name}")

        if self.state.run_audio_generation:
            if not self.state.audio_model_specifier: errors.append("Audio model missing.")
            elif self.state.audio_model_provider != 'local' and self.state.audio_model_provider not in self.state.cloud_api_keys:
                 errors.append(f"API key missing for audio provider '{self.state.audio_model_provider}'.")
            if not audio_dep_met: warnings.append("Audio enabled, but 'Save' stage disabled.")

        # Display summary *briefly* before asking
        print("Summary:")
        print(f"- Input: {len(self.state.input_files)} file(s)")
        print(f"- Text Model: {self.state.text_model_provider} / {Path(self.state.text_model_specifier).name if self.state.text_model_provider=='local_gguf' else self.state.text_model_specifier}")
        print(f"- Text Stages: {', '.join(s.capitalize() for s in self.state.stages_to_run) if self.state.stages_to_run else 'None'}")
        print(f"- Audio Gen: {'Yes' if self.state.run_audio_generation else 'No'}")
        if self.state.run_audio_generation: print(f"- Audio Model: {self.state.audio_model_provider} / {self.state.audio_model_specifier}")
        print(f"- Output Dir: {self.state.output_dir.name}")
        print("-" * 60)


        if errors:
             ConsoleOutput.error("Cannot run due to errors:")
             for err in errors: print(f"  - {err}")
             input("\nPlease correct the setup. Press Enter...")
             return MenuAction.CONTINUE # Go back if validation fails

        if warnings:
            ConsoleOutput.warning("Potential issues detected:")
            for warn in warnings: print(f"  - {warn}")

        confirm = input("\n➡️ Proceed with processing? (y/n): ").strip().lower()
        if confirm == 'y':
            return MenuAction.RUN # Signal to the main loop to execute
        else:
            ConsoleOutput.info("Processing cancelled.")
            input("Press Enter...")
            return MenuAction.CONTINUE


    def _execute_pipeline(self) -> bool: # Return success status
         """Executes the text and/or audio pipeline based on the current state."""
         ConsoleOutput.header("🚀 Starting Processing Pipeline 🚀")
         overall_success = True # Assume success unless error occurs

         if not self.state.input_files:
             ConsoleOutput.error("Execution failed: No input files specified.")
             return False

         text_backend: Optional[LLMBackend] = None
         audio_backend: Optional[AudioBackend] = None
         pipeline: Optional[ProcessingPipeline] = None
         all_results: List[PipelineResult] = []

         try:
             # --- Get Pipeline Config ---
             pipeline_config = self.state.get_pipeline_config()

             # --- Text Processing Backend Setup ---
             text_stages_selected = any(s in self.state.stages_to_run for s in PIPELINE_STAGES)
             if text_stages_selected:
                 ConsoleOutput.info(f"Initializing Text Backend ({pipeline_config.model_provider})...")
                 start_init = time.time()
                 try:
                     # Prepare kwargs for local backends, including ModelConfig
                     local_kwargs = {}
                     model_conf_for_local = None
                     if pipeline_config.model_provider == "local":
                         model_entry = get_registry().get_model(pipeline_config.model_specifier)
                         if model_entry:
                             from config import ModelConfig # Local import
                             model_conf_for_local = ModelConfig(
                                 name=model_entry.name, model_id=model_entry.model_id,
                                 supports_thinking=model_entry.supports_thinking, thinking_tokens=model_entry.thinking_tokens or [],
                                 max_context=model_entry.max_context, optimal_chunk_size=model_entry.optimal_chunk_size,
                                 temperature=model_entry.temperature, top_p=model_entry.top_p,
                                 max_new_tokens=model_entry.max_new_tokens,
                                 quantization_support=model_entry.quantization_support or ["4bit", "8bit"]
                             )
                         else:
                             raise ValueError(f"ModelConfig not found in registry for {pipeline_config.model_specifier}")

                         local_kwargs['model_config'] = model_conf_for_local # Pass resolved ModelConfig
                         local_kwargs['quantization_config'] = pipeline_config.quantization_config
                         local_kwargs['layer_split_config'] = pipeline_config.layer_split_config

                     # Use the factory function
                     text_backend = get_llm_backend(
                         provider=pipeline_config.model_provider,
                         model_specifier=pipeline_config.model_specifier,
                         api_keys=self.state.cloud_api_keys,
                         hyperparameters=pipeline_config.hyperparameters,
                         **local_kwargs # Pass specific kwargs for local backends
                     )

                     if text_backend is None:
                          raise ValueError(f"Could not create text backend for {pipeline_config.model_provider}. Check logs.")

                     ConsoleOutput.info(f"Loading/Initializing {text_backend.model_identifier}...")
                     load_success = text_backend.load_model(trust_remote_code=True)
                     if not load_success:
                         raise RuntimeError(f"Failed to load/initialize backend: {text_backend.model_identifier}")

                     init_time = time.time() - start_init
                     ConsoleOutput.success(f"Text backend ready in {init_time:.2f}s.")

                 except Exception as e:
                     ConsoleOutput.error(f"Failed to initialize text backend: {e}")
                     logger.error("Text backend initialization failed", exc_info=True)
                     # Don't proceed if text processing selected but backend failed
                     return False


             # --- Audio Generation Backend Setup ---
             if self.state.run_audio_generation:
                 ConsoleOutput.info(f"Initializing Audio Backend ({self.state.audio_model_provider})...")
                 start_init_audio = time.time()
                 try:
                     audio_backend = get_audio_backend(
                         provider=self.state.audio_model_provider,
                         model_specifier=self.state.audio_model_specifier,
                         api_keys=self.state.cloud_api_keys,
                         config=self.state.audio_config
                     )

                     if audio_backend is None:
                         raise ValueError(f"Could not create audio backend for {self.state.audio_model_provider}.")

                     ConsoleOutput.info(f"Loading/Initializing {audio_backend.model_identifier}...")
                     if not audio_backend.load_model():
                         raise RuntimeError(f"Failed to load/initialize audio backend.")

                     init_time_audio = time.time() - start_init_audio
                     ConsoleOutput.success(f"Audio backend ready in {init_time_audio:.2f}s.")

                 except Exception as e:
                     ConsoleOutput.error(f"Failed to initialize audio backend: {e}. Audio generation will be skipped.")
                     logger.error("Audio backend initialization failed", exc_info=True)
                     audio_backend = None # Ensure audio is skipped


             # --- Run Pipeline ---
             pipeline = ProcessingPipeline(pipeline_config)
             if text_backend:
                 pipeline.llm_backend = text_backend # Inject the backend

             pipeline.set_stages_to_run(self.state.stages_to_run)

             if text_stages_selected:
                  if not text_backend:
                      ConsoleOutput.error("Text processing stages skipped - backend not available.")
                      overall_success = False # Mark run as failed if stages skipped due to backend error
                  else:
                      ConsoleOutput.header("📝 Starting Text Processing 📝")
                      if len(self.state.input_files) == 1:
                          output_base = self.state.output_filename or None # Let pipeline handle default if None
                          output_p_base = self.state.output_dir / output_base if output_base else None
                          result = pipeline.process_file(self.state.input_files[0], output_path=output_p_base)
                          all_results.append(result)
                      else:
                          batch_results = pipeline.process_batch(self.state.input_files, output_dir=self.state.output_dir)
                          all_results.extend(batch_results)
             else: # No text stages selected
                  ConsoleOutput.info("No text processing stages selected.")
                  # Populate all_results for audio stage if needed
                  if self.state.run_audio_generation:
                       for f in self.state.input_files:
                            if f.suffix.lower() in ['.md', '.txt']:
                                 all_results.append(PipelineResult(success=True, input_file=f, output_file=f, # Use input as output
                                                                    processing_time=0, stages_completed=[], statistics={}))
                            else:
                                 ConsoleOutput.warning(f"Cannot generate audio from non-text file '{f.name}' when text stages are skipped.")


             # --- Run Audio Generation ---
             if self.state.run_audio_generation and audio_backend:
                 ConsoleOutput.header("🔊 Starting Audio Generation 🔊")
                 text_files_to_convert = [res.output_file for res in all_results if res.success and res.output_file and res.output_file.exists() and res.output_file.suffix.lower() in ['.md', '.txt']]

                 if not text_files_to_convert:
                     ConsoleOutput.warning("No valid text output files found to generate audio from (check if 'save' stage ran successfully).")
                 else:
                     ConsoleOutput.info(f"Found {len(text_files_to_convert)} file(s) for audio conversion...")
                     successful_audio_count = 0
                     with LoggingProgress(logger, "Generating audio", len(text_files_to_convert)) as progress:
                         for i, text_file in enumerate(text_files_to_convert):
                             ConsoleOutput.subsection(f"Audio for: {text_file.name} ({i+1}/{len(text_files_to_convert)})")
                             try:
                                 text_content = text_file.read_text(encoding='utf-8')
                                 # Basic cleaning for TTS (remove front matter, speaker tags if podcast)
                                 text_content = re.sub(r'^---.*?---', '', text_content, flags=re.DOTALL | re.MULTILINE).strip()
                                 if self.state.processing_mode == ProcessingMode.PODCAST:
                                      text_content = re.sub(r'^\*\*\[Speaker.*?\]:\*\*\n*', '', text_content, flags=re.MULTILINE)
                                 # Basic markdown removal
                                 text_content = re.sub(r'(\*\*|\*|`|#+\s)', '', text_content)

                                 if not text_content.strip():
                                      ConsoleOutput.warning("Skipping empty text file.")
                                      progress.update(1)
                                      continue

                                 audio_output_path = text_file.with_suffix(f".{self.state.audio_config.output_format}")
                                 audio_result = audio_backend.generate_audio(
                                     text=text_content,
                                     output_path=audio_output_path,
                                     chunk_text=(self.state.audio_model_provider == 'local')
                                 )
                                 if audio_result:
                                     ConsoleOutput.success(f"Audio saved: {audio_result.audio_path.name} ({audio_result.duration_formatted})")
                                     successful_audio_count += 1
                                 else:
                                     ConsoleOutput.error(f"Audio generation failed for {text_file.name}")
                                     overall_success = False # Mark run as failed if any audio fails

                             except Exception as audio_err:
                                  ConsoleOutput.error(f"Error during audio generation for {text_file.name}: {audio_err}")
                                  logger.error(f"Audio generation error", exc_info=True)
                                  overall_success = False
                             finally:
                                 progress.update(1)
                     ConsoleOutput.info(f"Audio generation finished: {successful_audio_count} successful.")
             elif self.state.run_audio_generation and not audio_backend:
                 ConsoleOutput.warning("Audio generation skipped as backend failed to initialize.")
                 overall_success = False # Mark as failed if audio was requested but couldn't run

             # Check overall text processing success
             if text_stages_selected and not all(r.success for r in all_results):
                  overall_success = False


         except KeyboardInterrupt:
            ConsoleOutput.warning("\nProcessing interrupted by user.")
            logger.warning("Pipeline execution interrupted.")
            overall_success = False
         except Exception as e:
             ConsoleOutput.error(f"🚨 Pipeline execution encountered a critical error: {e}")
             logger.critical("Pipeline execution failed", exc_info=True)
             overall_success = False
         finally:
             # Cleanup: Unload models, save report
             ConsoleOutput.info("Cleaning up resources...")
             if text_backend:
                 try: text_backend.unload_model()
                 except Exception as unload_err: logger.warning(f"Error unloading text backend: {unload_err}")
             if audio_backend:
                 try: audio_backend.unload_model()
                 except Exception as unload_err: logger.warning(f"Error unloading audio backend: {unload_err}")
             if pipeline: # Save report via pipeline's cleanup
                 try:
                     pipeline.cleanup() # Saves report, cleans file handler cache
                 except Exception as cleanup_err:
                     logger.error(f"Error during final pipeline cleanup: {cleanup_err}")
             else: # If pipeline failed early, try saving report via FileHandler directly
                  try:
                       if hasattr(self, 'file_handler') and self.file_handler.processed_files:
                            self.file_handler.save_processing_report()
                  except Exception as report_err:
                       logger.error(f"Error saving report during fallback cleanup: {report_err}")

             ConsoleOutput.info("Cleanup complete.")

         return overall_success


# --- Helper Functions (Quantization, Layer Split) ---

def create_quantization_config(args: argparse.Namespace) -> QuantizationConfig:
    """Create quantization configuration from arguments (or state)."""
    dtype_map = {
        "bfloat16": torch.bfloat16 if TORCH_AVAILABLE else None,
        "float16": torch.float16 if TORCH_AVAILABLE else None,
        "float32": torch.float32 if TORCH_AVAILABLE else None,
    }
    # Use getattr to safely access attributes, providing defaults
    compute_dtype_str = getattr(args, 'compute_dtype', 'bfloat16')
    compute_dtype = dtype_map.get(compute_dtype_str, dtype_map['bfloat16'])
    quant_method = getattr(args, 'quantization', DEFAULT_QUANTIZATION)

    # Need to handle case where torch is not available
    if compute_dtype is None and quant_method != "none":
        logger.warning(f"Torch not available, cannot set compute_dtype. Quantization might fail.")
        # Fallback to a type that doesn't require torch? Not really possible for bnb.
        # Let the backend handle the error if torch is missing.

    return QuantizationConfig(
        method=quant_method,
        compute_dtype=compute_dtype, # Pass potentially None type
        use_double_quant=True if quant_method == "4bit" else False,
        quant_type="nf4",
        bnb_4bit_use_double_quant=True if quant_method == "4bit" else False
    )

def create_layer_split_config(args: argparse.Namespace) -> LayerSplitConfig:
    """Create layer split configuration from arguments (or state)."""
    max_gpu_mem_str = getattr(args, 'max_gpu_memory', "10GB")
    max_cpu_mem_str = getattr(args, 'max_cpu_memory', "30GB")
    gpu_layers_count = getattr(args, 'gpu_layers', DEFAULT_GPU_LAYERS)
    # Check if 'no_layer_split' exists, default to False if not present
    no_split = getattr(args, 'no_layer_split', not ENABLE_LAYER_SPLITTING)

    max_gpu_memory = {}
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                max_gpu_memory[i] = max_gpu_mem_str
        except Exception as e:
             logger.warning(f"Could not get CUDA device count: {e}. Assuming 0 GPUs.")
             max_gpu_memory = {}
    else:
        max_gpu_memory = {} # Ensure empty dict if no CUDA

    # Determine 'enabled' based on args and CUDA availability
    split_enabled = not no_split and gpu_layers_count != 0 and bool(max_gpu_memory)

    return LayerSplitConfig(
        enabled=split_enabled,
        gpu_layers=gpu_layers_count,
        max_gpu_memory=max_gpu_memory,
        max_cpu_memory=max_cpu_mem_str,
        offload_folder=Path(OFFLOAD_DIR) # Ensure it's a Path
    )
