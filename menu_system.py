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

# --- LlamaNote Modules ---
from loggerConf import ConsoleOutput, get_logger_conf
from config_manager import ConfigManager
from config_base import (
    BASE_DIR, OUTPUT_DIR, CACHE_DIR, OFFLOAD_DIR,
    DEFAULT_MODEL, FALLBACK_MODEL, 
    ENABLE_LAYER_SPLITTING, DEFAULT_GPU_LAYERS,
    DEFAULT_QUANTIZATION, QUANTIZATION_OPTIONS,
    SUPPORTED_FORMATS, PIPELINE_STAGES
)
from model_hub import InteractiveModelBrowser, ModelHub, ModelInfo as HFModelInfo
from audio_generator import (
    AudioConfig, AudioResult, AudioBackend, 
    LocalAudioBackend, OpenAITTSBackend, get_audio_backend
)
from processing_pipeline import (
    ProcessingPipeline, PipelineConfig, ProcessingMode, 
    PipelineResult
)
from text_processor import ChunkingStrategy
from model_registry import get_registry, ModelEntry
from hyperparameters import HyperparameterConfig, InteractiveHyperparameterEditor, get_hyperparameter_help
from llm_handler import (
    LLMBackend, get_llm_backend, QuantizationConfig, LayerSplitConfig
)
from huggingface_search import HuggingFaceSearch # Import fix
from file_handler import BatchFileManager

# --- GGUF Backend (Optional) ---
try:
    from llamacpp_backend import LlamaCppBackend, LlamaCppConfig, GGUFModelManager
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    LlamaCppBackend = None
    LlamaCppConfig = None
    GGUFModelManager = None


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
                # --- FIX for lambda display ---
                # Resolve dynamic labels and descriptions
                label_text = item.label() if callable(item.label) else item.label
                desc_text = item.description() if callable(item.description) else item.description
                
                desc_str = f" - {desc_text}" if desc_text else ""
                print(f"  {item.key}. {label_text}{desc_str}")
            # --- End FIX ---

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
                    result = action_result if isinstance(action_result, MenuAction) else MenuAction.CONTINUE

                if result == MenuAction.EXIT:
                    return MenuAction.EXIT # Propagate EXIT up
                elif result == MenuAction.RUN:
                    return MenuAction.RUN # Propagate RUN up
                elif result == MenuAction.BACK:
                    continue

            else:
                ConsoleOutput.warning("Invalid selection")
            # time.sleep(0.5) # Removed for faster interaction

    def add_item(self, item: MenuItem):
        self.items.append(item)

# --- Menu System State ---

SUPPORTED_CLOUD_PROVIDERS = [
    "openai",
    "anthropic",
    "google",
    "cohere",
    "huggingface_token", 
    "openrouter",
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
    text_model_specifier: str = DEFAULT_MODEL
    audio_model_provider: str = "local"
    audio_model_specifier: str = "microsoft/speecht5_tts"
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the app state to a dictionary for saving."""
        return {
            "input_files": [str(f) for f in self.input_files], # Save as strings
            "output_dir": str(self.output_dir),
            "output_filename": self.output_filename,
            "processing_mode": self.processing_mode.value, # Save enum value
            "text_model_provider": self.text_model_provider,
            "text_model_specifier": self.text_model_specifier,
            "audio_model_provider": self.audio_model_provider,
            "audio_model_specifier": self.audio_model_specifier,
            "hyperparameters": asdict(self.hyperparameters),
            "audio_config": asdict(self.audio_config),
            "stages_to_run": self.state.stages_to_run,
            "run_audio_generation": self.state.run_audio_generation,
            "quantization": self.quantization,
            "gpu_layers": self.gpu_layers,
            "max_gpu_memory": self.max_gpu_memory,
            "max_cpu_memory": self.max_cpu_memory
        }

    def from_dict(self, data: Dict[str, Any]):
        """Deserializes a dictionary into the app state."""
        try:
            self.input_files = [Path(p) for p in data.get("input_files", [])]
            self.output_dir = Path(data.get("output_dir", "output"))
            self.output_filename = data.get("output_filename")
            self.processing_mode = ProcessingMode(data.get("processing_mode", "podcast"))
            self.text_model_provider = data.get("text_model_provider", "local")
            self.text_model_specifier = data.get("text_model_specifier", DEFAULT_MODEL)
            self.audio_model_provider = data.get("audio_model_provider", "local")
            self.audio_model_specifier = data.get("audio_model_specifier", "microsoft/speecht5_tts")
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
            ConsoleOutput.error(f"Error loading configuration: {e}. State may be partial.")

    def get_pipeline_config(self) -> PipelineConfig:
        """Creates a PipelineConfig based on current AppState."""
        model_name_meta = f"{self.text_model_provider}:{self.text_model_specifier}"

        # Create mock args namespace just for the helpers
        mock_args = argparse.Namespace(
            quantization=self.quantization,
            compute_dtype='bfloat16', # TODO: Make this configurable in AppState
            gpu_layers=self.gpu_layers,
            no_layer_split=False, # TODO: Make this configurable in AppState
            max_gpu_memory=self.max_gpu_memory,
            max_cpu_memory=self.max_cpu_memory
        )

        return PipelineConfig(
            mode=self.processing_mode,
            model_name=model_name_meta,
            model_provider=self.text_model_provider,
            model_specifier=self.text_model_specifier,
            quantization_config=create_quantization_config(mock_args),
            layer_split_config=create_layer_split_config(mock_args),
            memory_profile="medium_vram", # TODO: This is now redundant, remove later
            chunking_strategy=ChunkingStrategy.WORD_BOUNDARY, # TODO: Make configurable in AppState
            chunk_size=1000, # TODO: Make configurable in AppState
            markdown_style=self.processing_mode.value,
            output_format="markdown", # TODO: Make configurable in AppState
            hyperparameters=self.hyperparameters
        )


class MenuSystem:
    def __init__(self):
        self.state = AppState()
        self.config_manager = ConfigManager()
        self.hf_search = HuggingFaceSearch(cache_dir=CACHE_DIR / "hf_search") # Use main cache dir
        self.model_hub = ModelHub(cache_dir=CACHE_DIR / "model_hub") # Use main cache dir
        self.registry = get_registry() # Get registry instance

        self.state.cloud_api_keys = self.config_manager.load_cloud_keys()
        logger.info(f"Loaded API keys for providers: {list(self.state.cloud_api_keys.keys())}")

        self.main_menu = self._build_main_menu()
        logger.info("Initialized MenuSystem")

    def run(self):
        """Starts the main menu loop."""
        result = self.main_menu.display()
        if result == MenuAction.EXIT:
            ConsoleOutput.info("Exiting LlamaNote.")
        elif result == MenuAction.RUN:
             self._execute_pipeline()
             ConsoleOutput.info("Processing finished. Returning to main menu.")
             self.run() # Loop back to main menu after running
        else:
             ConsoleOutput.info("Exiting LlamaNote.")

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
        main_menu = Menu("LlamaNote Enhanced - Main Menu", items, parent=None)
        
        for item in items:
            if item.submenu:
                item.submenu.parent = main_menu
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
                description=lambda: f"Quant: {self.state.quantization}, GPUs: {self.state.gpu_layers}"
            ),
            MenuItem(
                "4",
                "Set Audio Generation Model",
                self._set_audio_model_menu,
                description=lambda: f"Current: {self.state.audio_model_provider} - {self.state.audio_model_specifier}"
            ),
            MenuItem(
                "5",
                "Configure Audio Settings",
                self._set_audio_config, 
                description="Set sample rate, format, voice, etc."
            ),
        ]
        menu = Menu("Model Settings", items, parent=self.main_menu)
        return menu

    def _build_cloud_settings_menu(self) -> Menu:
        """Builds the menu for managing API keys."""
        items = []
        for i, provider in enumerate(SUPPORTED_CLOUD_PROVIDERS):
            items.append(MenuItem(
                key=str(i + 1),
                label=lambda p=provider: f"Set/Update {p.replace('_', ' ').title()} Key {'(Set)' if p in self.state.cloud_api_keys else '(Not Set)'}",
                action=lambda p=provider: self._set_provider_key(p)
            ))
        items.append(MenuItem(str(len(SUPPORTED_CLOUD_PROVIDERS) + 1), "View Configured Keys", self._view_keys))
        menu = Menu("Cloud Provider API Keys", items, parent=self.main_menu)
        return menu


    # --- Menu Actions ---
    
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

    def _select_input_menu(self):
        """Handles selection of input file(s) or directory."""
        ConsoleOutput.section("Select Input Source")
        path_str = input("Enter PDF file path OR folder path containing files: ").strip()
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
                 ConsoleOutput.error(f"Unsupported file type: {path.suffix}. Supported: {SUPPORTED_FORMATS}")
                 input("Press Enter to continue...")
                 return MenuAction.CONTINUE
                 
        elif path.is_dir():
            ConsoleOutput.info(f"Scanning directory (recursive): {path}")
            fm = BatchFileManager()
            
            all_files_in_dir = []
            for ext in SUPPORTED_FORMATS:
                 all_files_in_dir.extend(path.rglob(f"*{ext}"))
            
            supported_files = sorted(list(set(all_files_in_dir)))

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
                     except ValueError: display_path = file_path
                     print(f"  {i+1:3d}. {display_path}")
                 
                 file_choices_str = input(f"Enter file numbers to process (comma-separated, e.g., 1,3,5): ").strip()
                 try:
                     indices = [int(n.strip()) - 1 for n in file_choices_str.split(',') if n.strip().isdigit()]
                     valid_indices = [idx for idx in indices if 0 <= idx < len(supported_files)]
                     if valid_indices:
                         selected_files = [supported_files[idx] for idx in valid_indices]
                         ConsoleOutput.success(f"Selected {len(selected_files)} specific file(s):")
                         for f in selected_files[:5]: print(f"  - {f.name}")
                         if len(selected_files) > 5: print("  ...")
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

        if self.state.input_files:
            return self._select_stages_menu() 
        else:
            return MenuAction.CONTINUE


    def _select_stages_menu(self):
        """Allows user to toggle which pipeline stages run."""
        all_text_stages = list(PIPELINE_STAGES) 

        while True:
            ConsoleOutput.header("Select Stages")
            print("Text Processing Stages:")
            for i, stage in enumerate(all_text_stages):
                 included = "[X]" if stage in self.state.stages_to_run else "[ ]"
                 print(f"  {i+1}. {included} {stage}")

            audio_included = "[X]" if self.state.run_audio_generation else "[ ]"
            print(f"\nAudio Generation Stage:")
            print(f"  A. {audio_included} Generate Audio (Runs after text processing if 'save' is enabled)")

            print("\nOptions:")
            print("  Enter number (1-7) or 'A' to toggle a stage.")
            print("  'text'   - Select all text stages.")
            print("  'none'   - Deselect all text stages.")
            print("  'done'   - Confirm selection and go back.")
            print("-" * 60)

            choice = input("Toggle stage or command: ").strip().lower()

            if choice == 'done':
                break 
            elif choice == 'text':
                self.state.stages_to_run = list(PIPELINE_STAGES)
                ConsoleOutput.info("All text stages selected.")
            elif choice == 'none':
                self.state.stages_to_run = []
                ConsoleOutput.info("All text stages deselected.")
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
                        else:
                            temp_stages = []
                            stages_set = set(self.state.stages_to_run)
                            stages_set.add(stage_name)
                            for s in PIPELINE_STAGES:
                                if s in stages_set:
                                    temp_stages.append(s)
                            self.state.stages_to_run = temp_stages
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
        if current_key:
            masked_key = '*' * (len(current_key) - 4) + current_key[-4:] if len(current_key) > 4 else '*' * len(current_key)
            print(f"Current key: {masked_key}")
        else:
            print("Current key: Not set")

        new_key = input("Enter new API key (leave blank to clear, 'keep' to keep current): ").strip()

        if new_key.lower() == 'keep':
            return MenuAction.CONTINUE
        
        if not new_key:
            # Clear the key
            if self.config_manager.save_cloud_key(provider, None):
                if provider in self.state.cloud_api_keys:
                    del self.state.cloud_api_keys[provider]
                ConsoleOutput.success(f"API key for {provider} cleared.")
            else:
                ConsoleOutput.error(f"Failed to clear key for {provider}.")
        else:
            # Save the new key
            if self.config_manager.save_cloud_key(provider, new_key):
                self.state.cloud_api_keys[provider] = new_key # Update runtime state
                ConsoleOutput.success(f"API key for {provider} saved.")
            else:
                ConsoleOutput.error(f"Failed to save key for {provider}.")

        # Force reload keys from file to ensure state is sync
        self.state.cloud_api_keys = self.config_manager.load_cloud_keys()
        return MenuAction.CONTINUE


    def _view_keys(self):
        """Displays which providers have keys configured."""
        ConsoleOutput.section("Configured API Keys")
        configured_keys = self.config_manager.load_cloud_keys()
        self.state.cloud_api_keys = configured_keys # Update state

        if not configured_keys:
            print("No API keys are currently configured in the file.")
            print(f"(Config file location: {self.config_manager.cloud_keys_file})")
        else:
            print("Providers with keys configured:")
            for provider, key in configured_keys.items():
                 masked_key = '*' * (len(key) - 4) + key[-4:] if len(key) > 4 else '*' * len(key)
                 print(f"  - {provider.replace('_', ' ').title()}: {masked_key}")
        input("\nPress Enter to continue...")
        return MenuAction.CONTINUE

    # --- Model Selection ---
    def _set_text_model_menu(self):
         """Top-level menu to choose between local and cloud text models."""
         while True: 
             ConsoleOutput.section("Set Text Processing Model")
             print("  1. Use Local Model (Hugging Face / GGUF)")
             print("  2. Use Cloud Model (OpenAI, Anthropic, Google, etc.)")
             current_model_display = f"{self.state.text_model_provider}: {self.state.text_model_specifier}"
             print(f"\nCurrent Model: {current_model_display}")
             print("\n  b. Back")
             print("-" * 60)

             choice = input("Select option (1-2, or b): ").strip().lower()

             if choice == '1':
                 self._select_local_text_model() # This is now its own loop
             elif choice == '2':
                 self._select_cloud_text_model() # This is now its own loop
             elif choice == 'b':
                 return MenuAction.BACK # Go back to Model Settings Menu
             else:
                 ConsoleOutput.warning("Invalid choice.")


    def _select_local_text_model(self):
        """Handles selection of a local HF model or GGUF."""
        while True:
            ConsoleOutput.subsection("Select Local Text Model")
            print("  1. Select from Predefined List (Recommended)")
            print("  2. Browse Hugging Face Hub (Transformers)")
            print("  3. Enter Hugging Face Model ID directly (Transformers)")
            if LLAMACPP_AVAILABLE:
                print("  4. Select GGUF Model from cache")
                print("  5. Enter path to GGUF Model file")
            
            current_local = self.state.text_model_specifier if self.state.text_model_provider in ['local', 'local_gguf'] else 'N/A'
            print(f"\nCurrent Local Model: {current_local}")
            print("\n  b. Back")
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
                for i, entry in enumerate(predefined):
                    print(f"  {i+1}. {entry.name} ({entry.model_id})")
                
                model_choice = input(f"Select model (1-{len(predefined)} or b): ").strip()
                if model_choice == 'b': continue
                try:
                    idx = int(model_choice) - 1
                    if 0 <= idx < len(predefined):
                        new_model_id = predefined[idx].model_id
                        new_provider = "local" # Predefined are transformers models
                    else:
                        ConsoleOutput.warning("Invalid number.")
                except ValueError:
                    ConsoleOutput.warning("Invalid input.")

            elif choice == '2':
                browser = InteractiveModelBrowser(self.model_hub)
                selected_hf_model: Optional[HFModelInfo] = browser.browse(task="text-generation", library="transformers")
                if selected_hf_model:
                    new_model_id = selected_hf_model.model_id
                    new_provider = "local"
                    if not self.registry.get_model(new_model_id):
                        self.registry.add_from_model_info(selected_hf_model)
                        ConsoleOutput.info(f"Added {new_model_id} to local registry.")

            elif choice == '3':
                model_id_input = input("Enter Hugging Face Model ID: ").strip()
                if model_id_input:
                     if '/' not in model_id_input:
                         ConsoleOutput.warning("Invalid HF Model ID format (expected 'Org/ModelName').")
                     else:
                        new_model_id = model_id_input
                        new_provider = "local"
                        if not self.registry.get_model(new_model_id):
                            hf_info = self.model_hub.get_model_info(new_model_id)
                            if hf_info: self.registry.add_from_model_info(hf_info)
                            else: self.registry.add_model(ModelEntry(model_id=model_id_input, name=model_id_input.split('/')[-1], author="Unknown"))

            elif choice == '4' and LLAMACPP_AVAILABLE:
                manager = GGUFModelManager(cache_dir=CACHE_DIR / "gguf_models")
                gguf_models = manager.list_available_models()
                if not gguf_models:
                    ConsoleOutput.warning("No GGUF models found in cache.")
                    input("Press Enter to continue...")
                    continue
                print("\nAvailable GGUF models:")
                for i, p in enumerate(gguf_models): print(f"  {i+1}. {p.name}")
                gguf_choice = input(f"Select GGUF model number (1-{len(gguf_models)} or b): ").strip()
                if gguf_choice == 'b': continue
                try:
                    idx = int(gguf_choice) - 1
                    if 0 <= idx < len(gguf_models):
                        new_model_id = str(gguf_models[idx].resolve())
                        new_provider = "local_gguf"
                    else: ConsoleOutput.warning("Invalid number.")
                except ValueError: ConsoleOutput.warning("Invalid input.")

            elif choice == '5' and LLAMACPP_AVAILABLE:
                gguf_path_str = input("Enter full path to GGUF model file: ").strip()
                gguf_path = Path(os.path.expanduser(gguf_path_str))
                if gguf_path.is_file() and gguf_path.suffix.lower() in ['.gguf', '.ggml', '.bin']:
                    new_model_id = str(gguf_path.resolve())
                    new_provider = "local_gguf"
                else:
                    ConsoleOutput.error("Invalid GGUF file path or extension.")
                    input("Press Enter to continue...")

            elif choice == 'b':
                return # Go back to the text model type selection

            else:
                ConsoleOutput.warning("Invalid choice.")
                continue # Re-prompt this menu

            # If a new model was selected, update state
            if new_model_id:
                old_provider = self.state.text_model_provider
                old_specifier = self.state.text_model_specifier
                self.state.text_model_provider = new_provider
                self.state.text_model_specifier = new_model_id
                ConsoleOutput.success(f"Text model set to: {new_provider}: {self.state.text_model_specifier}")
                
                if old_provider != new_provider or old_specifier != new_model_id:
                     reset_hp = input("Reset hyperparameters to default for new model? (y/n) [y]: ").strip().lower()
                     if reset_hp != 'n':
                         self.state.hyperparameters = HyperparameterConfig() 
                         ConsoleOutput.info("Hyperparameters reset to defaults.")
                return # Go back after successful selection


    def _select_cloud_text_model(self):
        """Handles selection of a cloud provider and model."""
        while True:
            ConsoleOutput.subsection("Select Cloud Text Model")
            # Filter providers with keys
            configured_providers = [p for p in SUPPORTED_CLOUD_PROVIDERS if p in self.state.cloud_api_keys and p != "huggingface_token"]

            if not configured_providers:
                ConsoleOutput.warning("No cloud providers configured with API keys.")
                print("Please configure keys in the 'Cloud Provider Settings' menu.")
                input("Press Enter to continue...")
                return # Back to text model type selection

            print("Configured Cloud Providers:")
            provider_map = {}
            # Start numbering from 1
            for i, provider in enumerate(configured_providers):
                print(f"  {i+1}. {provider.replace('_', ' ').title()}")
                provider_map[str(i+1)] = provider
            
            print("\n  b. Back to Text Model Type Selection")
            print("-" * 60)

            provider_choice = input(f"Select provider (1-{len(provider_map)} or b): ").strip().lower()

            if provider_choice == 'b':
                return
            elif provider_choice in provider_map:
                selected_provider = provider_map[provider_choice]
                
                example_prompt = "e.g., "
                if selected_provider == "openai":
                    example_prompt += "gpt-4o / gpt-4-turbo / gpt-3.5-turbo"
                elif selected_provider == "anthropic":
                    example_prompt += "claude-3-opus-20240229"
                elif selected_provider == "google":
                    example_prompt += "models/gemini-1.5-pro-latest"
                elif selected_provider == "openrouter":
                    example_prompt += "anthropic/claude-3-haiku"
                else:
                    example_prompt = "Enter model name/ID"
                
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
                    return # Go back after success
                else:
                    ConsoleOutput.warning("Model name cannot be empty.")
            else:
                ConsoleOutput.warning("Invalid provider selection.")
            
            input("Press Enter to continue...") # Pause before re-prompting


    def _set_text_hyperparameters(self):
        """Opens the interactive hyperparameter editor."""
        ConsoleOutput.section(f"Configure Text Hyperparameters for {self.state.text_model_provider}: {self.state.text_model_specifier}")
        editor = InteractiveHyperparameterEditor(self.state.hyperparameters)
        self.state.hyperparameters = editor.edit() # This starts the interactive editing loop
        ConsoleOutput.success("Text hyperparameters updated for this session.")
        return MenuAction.CONTINUE # Stay in Model Settings menu

    def _set_local_hardware_config(self):
        """Interactive configuration for local model hardware settings."""
        ConsoleOutput.section("Configure Local Model Hardware Settings")
        
        while True:
            ConsoleOutput.header("Local Hardware Settings")
            print(f"  Provider: {self.state.text_model_provider}")
            print(f"  Model: {self.state.text_model_specifier}")
            print("-" * 60)
            
            is_gguf = (self.state.text_model_provider == 'local_gguf')
            
            if is_gguf:
                print("  Note: Settings for GGUF model (via llama-cpp-python)")
                print(f"  1. GPU Layers (n_gpu_layers) : {self.state.gpu_layers} (-1 = all, 0 = CPU only)")
                # Quantization is baked into GGUF, but we can show it
                print(f"  2. Quantization (inferred) : (Set by GGUF file)") 
            else: # Transformers
                print("  Note: Settings for Hugging Face Transformers model")
                print(f"  1. Quantization           : {self.state.quantization} (Options: none, 4bit, 8bit)")
                print(f"  2. Max GPU Memory         : {self.state.max_gpu_memory} (e.g., 8GB, 10000MB)")
                print(f"  3. Max CPU Memory (Offload) : {self.state.max_cpu_memory} (e.g., 30GB)")
                
            print("\n  b. Back to Model Settings")
            print("-" * 60)
            
            choice = input("Select setting to change (or b to back): ").strip().lower()
            
            try:
                if choice == 'b':
                    return MenuAction.BACK
                
                if is_gguf:
                    if choice == '1':
                        val = input(f"Enter number of GPU layers (-1 for all) [current: {self.state.gpu_layers}]: ").strip()
                        if not val: continue
                        self.state.gpu_layers = int(val)
                        ConsoleOutput.success(f"GPU Layers set to: {self.state.gpu_layers}")
                    else:
                        ConsoleOutput.warning("Invalid selection for GGUF model.")
                
                else: # Transformers settings
                    if choice == '1':
                        val = input(f"Enter quantization (none, 4bit, 8bit) [current: {self.state.quantization}]: ").strip().lower()
                        if not val: continue
                        if val in QUANTIZATION_OPTIONS:
                            self.state.quantization = val
                            ConsoleOutput.success(f"Quantization set to: {val}")
                        else:
                            ConsoleOutput.warning(f"Invalid option. Must be one of: {QUANTIZATION_OPTIONS}")
                    elif choice == '2':
                         val = input(f"Enter Max GPU Memory (e.g., 8GiB, 10GB) [current: {self.state.max_gpu_memory}]: ").strip()
                         if not val: continue
                         # Basic validation, refine as needed
                         if val.endswith(("GB", "GiB", "MB", "MiB")):
                             self.state.max_gpu_memory = val
                             ConsoleOutput.success(f"Max GPU Memory set to: {val}")
                         else:
                             ConsoleOutput.warning("Invalid format. Use GB, GiB, MB, etc.")
                    elif choice == '3':
                        val = input(f"Enter Max CPU Memory (e.g., 30GB) [current: {self.state.max_cpu_memory}]: ").strip()
                        if not val: continue
                        if val.endswith(("GB", "GiB", "MB", "MiB")):
                            self.state.max_cpu_memory = val
                            ConsoleOutput.success(f"Max CPU Memory set to: {val}")
                        else:
                             ConsoleOutput.warning("Invalid format. Use GB, GiB, MB, etc.")
                    else:
                        ConsoleOutput.warning("Invalid selection for Transformers model.")
                        
            except ValueError:
                ConsoleOutput.warning("Invalid input value type.")
            except Exception as e:
                 ConsoleOutput.error(f"Error updating setting: {e}")
                 
                 
    def _set_audio_model_menu(self):
        """Top-level menu to choose between local and cloud audio models."""
        while True:
            ConsoleOutput.section("Set Audio Generation Model (TTS)")
            print("  1. Use Local Model (Hugging Face)")
            print("  2. Use Cloud Model (e.g., OpenAI TTS)")
            current_model_display = f"{self.state.audio_model_provider}: {self.state.audio_model_specifier}"
            print(f"\nCurrent Model: {current_model_display}")
            print("\n  b. Back to Model Settings")
            print("-" * 60)

            choice = input("Select option (1-2, or b): ").strip().lower()

            if choice == '1':
                self._select_local_audio_model()
            elif choice == '2':
                self._select_cloud_audio_model()
            elif choice == 'b':
                return MenuAction.BACK 
            else:
                ConsoleOutput.warning("Invalid choice.")

    def _select_local_audio_model(self):
        """Handles selection of a local HF TTS model."""
        while True:
            ConsoleOutput.subsection("Select Local Audio Model (TTS)")
            print("  1. Browse Hugging Face Hub (text-to-speech)")
            print("  2. Enter Hugging Face Model ID directly")
            current_local = self.state.audio_model_specifier if self.state.audio_model_provider == 'local' else 'N/A'
            print(f"\nCurrent Local Model: {current_local}")
            print("\n  b. Back to Audio Model Type Selection")
            print("-" * 60)

            choice = input("Select option: ").strip().lower()
            new_model_id = None

            if choice == '1':
                browser = InteractiveModelBrowser(self.model_hub)
                selected_hf_model: Optional[HFModelInfo] = browser.browse(task="text-to-speech")
                if selected_hf_model:
                     new_model_id = selected_hf_model.model_id
            elif choice == '2':
                 model_id_input = input("Enter Hugging Face Model ID (TTS): ").strip()
                 if model_id_input:
                     if '/' in model_id_input and len(model_id_input) > 3:
                         new_model_id = model_id_input
                     else:
                         ConsoleOutput.warning("Invalid model ID format.")
            elif choice == 'b':
                 return

            else:
                ConsoleOutput.warning("Invalid choice.")
                continue

            if new_model_id:
                self.state.audio_model_provider = "local"
                self.state.audio_model_specifier = new_model_id
                self.state.audio_config.model_id = new_model_id 
                ConsoleOutput.success(f"Audio model set to: local: {self.state.audio_model_specifier}")
                return


    def _select_cloud_audio_model(self):
        """Handles selection of a cloud TTS provider and model."""
        while True:
            ConsoleOutput.subsection("Select Cloud Audio Model (TTS)")
            available_tts_providers = [p for p in TTS_PROVIDERS if p != 'local' and p in self.state.cloud_api_keys]

            if not available_tts_providers:
                ConsoleOutput.warning("No cloud TTS providers configured with API keys.")
                print(f"Supported & Configurable TTS Providers: {', '.join(p for p in TTS_PROVIDERS if p != 'local')}")
                print("Please configure keys in the 'Cloud Provider Settings' menu.")
                input("Press Enter to continue...")
                return

            print("Configured & Supported Cloud TTS Providers:")
            provider_map = {}
            for i, provider in enumerate(available_tts_providers):
                print(f"  {i+1}. {provider.replace('_', ' ').title()}")
                provider_map[str(i+1)] = provider
            print("\n  b. Back to Audio Model Type Selection")
            print("-" * 60)

            provider_choice = input(f"Select provider (1-{len(provider_map)} or b): ").strip().lower()

            if provider_choice == 'b':
                return
            elif provider_choice in provider_map:
                selected_provider = provider_map[provider_choice]
                model_specifier = ""
                
                if selected_provider == "openai":
                    model_specifier = input(f"Enter model name for {selected_provider.title()} (e.g., tts-1, tts-1-hd) [default: tts-1]: ").strip()
                    if not model_specifier: model_specifier = "tts-1"
                else:
                    model_specifier = input(f"Enter model name/ID for {selected_provider.title()}: ").strip()

                if model_specifier:
                    self.state.audio_model_provider = selected_provider
                    self.state.audio_model_specifier = model_specifier
                    self.state.audio_config.model_id = None 
                    ConsoleOutput.success(f"Audio model set to: {selected_provider}: {model_specifier}")
                    return
                else:
                    ConsoleOutput.warning("Model name cannot be empty.")
            else:
                ConsoleOutput.warning("Invalid provider selection.")
            
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
            print(f"  2. Sample Rate   : {cfg.sample_rate} Hz (Target SR; cloud models may have fixed output SR)")
            print(f"  3. Speed Factor  : {cfg.speed} (1.0 = normal; applied via post-processing or API if supported)")
            print(f"  4. Pitch Shift   : {cfg.pitch_shift} semitones (0 = normal; post-processing only)")
            print(f"  5. Volume Norm.  : {'Enabled' if cfg.volume_normalize else 'Disabled'} (Post-processing)")

            if not is_local:
                 print(f"  6. Cloud Voice   : {cfg.cloud_voice} (Provider-specific, e.g., 'alloy', 'echo' for OpenAI)")
                 print("  7. (Local setting N/A)")
                 print("  8. (Local setting N/A)")
            else:
                 print(f"  6. Half Precision: {'Enabled' if cfg.use_half_precision else 'Disabled'} (Local CUDA only)")
                 print(f"  7. Speaker Embed.: {cfg.speaker_embedding or 'Default/Not Used'} (Local SpeechT5)")
                 print(f"  8. Text Chunk Size:{cfg.chunk_size} chars (Local processing)")
            
            print("\n  s. Save audio settings to preset")
            print("  l. Load audio settings from preset")
            print("  b. Back to Model Settings")
            print("-" * 60)

            choice = input("Select setting to change (or b, s, l): ").strip().lower()

            try:
                if choice == 'b':
                    return MenuAction.BACK 
                elif choice == 's':
                    preset_name = input("Enter name for audio preset: ").strip()
                    if preset_name:
                        self.config_manager.save_config(preset_name, cfg.to_dict(), dir_type="audio")
                        ConsoleOutput.success(f"Audio preset '{preset_name}' saved.")
                    else:
                        ConsoleOutput.warning("Invalid name, save cancelled.")
                elif choice == 'l':
                    presets = self.config_manager.list_configs(dir_type="audio")
                    if not presets:
                        ConsoleOutput.warning("No saved audio presets found.")
                        continue
                    print("Available audio presets:")
                    for i, name in enumerate(presets): print(f"  {i+1}. {name}")
                    preset_choice = input(f"Select preset (1-{len(presets)} or b): ").strip()
                    if preset_choice == 'b': continue
                    if preset_choice.isdigit():
                         idx = int(preset_choice) - 1
                         if 0 <= idx < len(presets):
                              loaded_data = self.config_manager.load_config(presets[idx], dir_type="audio")
                              if loaded_data:
                                   self.state.audio_config = AudioConfig(**loaded_data)
                                   ConsoleOutput.success(f"Loaded audio preset '{presets[idx]}'.")
                              else:
                                   ConsoleOutput.error("Failed to load preset.")
                         else: ConsoleOutput.warning("Invalid selection.")
                    else: ConsoleOutput.warning("Invalid input.")

                elif choice == '1':
                    fmt = input(f"Enter new format (wav, mp3, flac) [current: {cfg.output_format}]: ").strip().lower()
                    if fmt in ["wav", "mp3", "flac"]: cfg.output_format = fmt
                    elif fmt: ConsoleOutput.warning("Invalid format.")
                elif choice == '2':
                    sr = input(f"Enter new target sample rate (e.g., 16000) [current: {cfg.sample_rate}]: ").strip()
                    if sr and sr.isdigit() and int(sr) > 0: cfg.sample_rate = int(sr)
                    elif sr: ConsoleOutput.warning("Invalid sample rate.")
                elif choice == '3':
                    spd = input(f"Enter new speed factor (0.5-2.0) [current: {cfg.speed}]: ").strip()
                    if spd:
                        spd_f = float(spd)
                        if 0.5 <= spd_f <= 2.0: cfg.speed = spd_f
                        else: ConsoleOutput.warning("Speed factor out of range (0.5-2.0).")
                elif choice == '4':
                    pitch = input(f"Enter pitch shift in semitones (-12 to 12) [current: {cfg.pitch_shift}]: ").strip()
                    if pitch:
                        pitch_i = int(pitch)
                        if -12 <= pitch_i <= 12: cfg.pitch_shift = pitch_i
                        else: ConsoleOutput.warning("Pitch shift out of range (-12 to 12).")
                elif choice == '5':
                    cfg.volume_normalize = not cfg.volume_normalize
                    ConsoleOutput.info(f"Volume normalization {'Enabled' if cfg.volume_normalize else 'Disabled'}.")
                elif choice == '6':
                     if not is_local: # Cloud Voice
                          voice = input(f"Enter provider-specific voice (current: {cfg.cloud_voice}): ").strip()
                          if voice: cfg.cloud_voice = voice
                     else: # Local model precision
                          if torch.cuda.is_available():
                               cfg.use_half_precision = not cfg.use_half_precision
                               ConsoleOutput.info(f"Half precision {'Enabled' if cfg.use_half_precision else 'Disabled'}.")
                          else:
                               ConsoleOutput.warning("Half precision requires CUDA.")
                               cfg.use_half_precision = False
                elif choice == '7' and is_local: # Speaker Embedding
                     spk = input(f"Enter path/ID for speaker embedding (current: {cfg.speaker_embedding or 'Default'}, blank to clear): ").strip()
                     cfg.speaker_embedding = spk if spk else None
                     ConsoleOutput.info(f"Speaker embedding set to: {cfg.speaker_embedding or 'Default'}")
                elif choice == '8' and is_local: # Chunk size
                     cs = input(f"Enter text chunk size (current: {cfg.chunk_size}): ").strip()
                     if cs and cs.isdigit() and 500 <= int(cs) <= 10000: cfg.chunk_size = int(cs)
                     elif cs: ConsoleOutput.warning("Invalid chunk size (use 500-10000).")
                else:
                    ConsoleOutput.warning("Invalid selection.")

            except ValueError:
                ConsoleOutput.warning("Invalid input value type.")
            except Exception as e:
                 ConsoleOutput.error(f"Error updating setting: {e}")
                 input("Press Enter...")


    def _output_settings_menu(self):
        """Sets the output directory and optional base filename."""
        ConsoleOutput.section("Output Settings")
        try:
             if not self.state.output_dir.is_absolute():
                  self.state.output_dir = BASE_DIR / self.state.output_dir
             self.state.output_dir = self.state.output_dir.resolve()
             self.state.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
             ConsoleOutput.warning(f"Could not resolve output directory '{self.state.output_dir}', defaulting to 'output'. Error: {e}")
             self.state.output_dir = (BASE_DIR / "output").resolve()
             self.state.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Current Output Directory: {self.state.output_dir}")
        if self.state.output_filename:
             print(f"Current Base Filename: {self.state.output_filename} (Timestamps/suffixes added automatically)")
        else:
             print("Current Base Filename: Default (derived from input filename)")

        new_dir_str = input(f"Enter new output directory path (leave blank to keep current): ").strip()
        if new_dir_str:
            try:
                new_dir_path = Path(os.path.expanduser(new_dir_str)).resolve()
                new_dir_path.mkdir(parents=True, exist_ok=True) 
                # Test writability
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
             current_name_prompt = f"(current: {self.state.output_filename})" if self.state.output_filename else "(leave blank for default)"
             new_name = input(f"Enter base output filename (WITHOUT extension, {current_name_prompt}): ").strip()
             if new_name:
                 new_name = re.sub(r'[\\/*?:"<>|]', '', new_name) # Sanitize
                 self.state.output_filename = new_name
                 ConsoleOutput.success(f"Base output filename set to: {self.state.output_filename}")
             elif new_name == "" and self.state.output_filename:
                 ConsoleOutput.info("Keeping current base filename.")
             elif new_name == "":
                 ConsoleOutput.info("Using default output filename based on input.")
                 self.state.output_filename = None # Explicitly set to default
        elif num_inputs > 1:
             ConsoleOutput.info("Base output filename is ignored during batch processing (uses input names).")
             self.state.output_filename = None 

        return MenuAction.CONTINUE


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
                     print(f"  - {f.name}")
                 elif i == max_files_to_show:
                     print(f"  ... and {len(self.state.input_files) - max_files_to_show} more.")
                     break
        else:
             print("  ⚠️ None selected. Please select input file(s).")

        # Output Settings
        print(f"\nOutput Directory: {self.state.output_dir.resolve()}")
        # Get format from pipeline config (which needs AppState)
        temp_pipeline_config = self.state.get_pipeline_config()
        text_output_format = temp_pipeline_config.output_format or 'md'

        if len(self.state.input_files) == 1 and self.state.output_filename:
             example_suffix = f"_{self.state.processing_mode.value}"
             example_ext = f".{text_output_format}"
             print(f"Base Output Filename: {self.state.output_filename}")
             print(f"  (Example text output: {self.state.output_filename}{example_suffix}_<timestamp>{example_ext})")
        else:
             print(f"Base Output Filename: Default (based on input filename)")

        # Text Processing
        print(f"\n--- Text Processing ---")
        print(f"Mode: {self.state.processing_mode.value}")
        print(f"Provider: {self.state.text_model_provider}")
        print(f"Model: {self.state.text_model_specifier}")
        
        # Show local hardware settings if local
        if self.state.text_model_provider == "local":
            print(f"Local Hardware Config:")
            print(f"  - Quantization: {self.state.quantization}")
            print(f"  - Max GPU Mem: {self.state.max_gpu_memory} | Max CPU Mem: {self.state.max_cpu_memory}")
        elif self.state.text_model_provider == "local_gguf":
            print(f"Local Hardware Config (GGUF):")
            print(f"  - GPU Layers: {self.state.gpu_layers}")

        print(f"Hyperparameters:")
        hp_dict = self.state.hyperparameters.to_dict()
        print(f"  - temperature: {hp_dict.get('temperature', 'Default')}")
        print(f"  - top_p: {hp_dict.get('top_p', 'Default')}")
        print(f"  - max_new_tokens: {hp_dict.get('max_new_tokens') or 'Auto'}")
        print(f"  (Use 'Model Settings -> Hyperparameters' menu to see/edit all)")

        # Stages
        print(f"\n--- Pipeline Stages ---")
        text_stages_to_run = [s for s in PIPELINE_STAGES if s in self.state.stages_to_run]
        print(f"Text Stages to Run: {', '.join(text_stages_to_run) if text_stages_to_run else 'None'}")
        print(f"Generate Audio After Text: {'✅ Yes' if self.state.run_audio_generation else '❌ No'}")

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

        # Warnings
        print("-" * 60)
        has_warnings = False
        if not self.state.input_files:
             ConsoleOutput.warning("Setup incomplete: Input files are missing.")
             has_warnings = True
        if not text_stages_to_run and not self.state.run_audio_generation:
             ConsoleOutput.warning("Setup incomplete: No processing or audio stages selected.")
             has_warnings = True
        if text_stages_selected:
            if not self.state.text_model_specifier:
                 ConsoleOutput.warning("Setup incomplete: No text model specified.")
                 has_warnings = True
            elif self.state.text_model_provider != 'local' and self.state.text_model_provider != 'local_gguf' and self.state.text_model_provider not in self.state.cloud_api_keys:
                 ConsoleOutput.warning(f"API key for text provider '{self.state.text_model_provider}' is missing.")
                 has_warnings = True
        if self.state.run_audio_generation:
            if not self.state.audio_model_specifier:
                 ConsoleOutput.warning("Setup incomplete: No audio model specified.")
                 has_warnings = True
            elif self.state.audio_model_provider != 'local' and self.state.audio_model_provider not in self.state.cloud_api_keys:
                 ConsoleOutput.warning(f"API key for audio provider '{self.state.audio_model_provider}' is missing.")
                 has_warnings = True
        
        if not has_warnings:
             ConsoleOutput.info("Setup appears complete.")

        input("\nPress Enter to continue...")
        return MenuAction.CONTINUE


    def _confirm_and_run(self) -> MenuAction:
        """Confirms the setup and returns RUN action if confirmed."""
        ConsoleOutput.section("Confirm and Run Processing")
        
        # Validation Checks
        text_stages_selected = any(s in self.state.stages_to_run for s in PIPELINE_STAGES)
        runnable = True
        warnings = []
        errors = []

        if not self.state.input_files:
             errors.append("No input files selected.")
             runnable = False
        if not text_stages_selected and not self.state.run_audio_generation:
             errors.append("No processing or audio stages selected.")
             runnable = False
        
        # Text Backend Checks
        if text_stages_selected:
            if not self.state.text_model_specifier:
                 errors.append("No text model specified.")
                 runnable = False
            elif self.state.text_model_provider != 'local' and self.state.text_model_provider != 'local_gguf' and self.state.text_model_provider not in self.state.cloud_api_keys:
                 errors.append(f"API key for text provider '{self.state.text_model_provider}' is missing.")
                 runnable = False

        # Audio Backend Checks
        if self.state.run_audio_generation:
            if not self.state.audio_model_specifier:
                 errors.append("No audio model specified.")
                 runnable = False
            elif self.state.audio_model_provider != 'local' and self.state.audio_model_provider not in self.state.cloud_api_keys:
                 errors.append(f"API key for audio provider '{self.state.audio_model_provider}' is missing.")
                 runnable = False
            
            # Check dependency on 'save' stage
            if 'save' not in self.state.stages_to_run:
                 if not all(f.suffix.lower() in ['.md', '.txt'] for f in self.state.input_files):
                      warnings.append("Audio generation requested, but 'save' stage is not selected. Audio may fail if input files are not text.")

        # Display current setup *before* asking to proceed
        self._view_current_setup()

        if not runnable:
             ConsoleOutput.error("Cannot run due to errors:")
             for err in errors:
                 print(f"  - {err}")
             input("\nPlease correct the setup. Press Enter to return to the main menu...")
             return MenuAction.CONTINUE # Go back if validation fails
             
        if warnings:
            ConsoleOutput.warning("Potential issues detected:")
            for warn in warnings:
                print(f"  - {warn}")

        confirm = input("\n➡️ Proceed with processing? (y/n): ").strip().lower()
        if confirm == 'y':
            return MenuAction.RUN # Signal to the main loop to execute
        else:
            ConsoleOutput.info("Processing cancelled.")
            return MenuAction.CONTINUE


    def _execute_pipeline(self):
         """Executes the text and/or audio pipeline based on the current state."""
         ConsoleOutput.header("🚀 Starting Processing Pipeline 🚀")

         if not self.state.input_files:
             ConsoleOutput.error("Execution failed: No input files specified.")
             return

         text_backend: Optional[LLMBackend] = None
         audio_backend: Optional[AudioBackend] = None
         pipeline: Optional[ProcessingPipeline] = None
         all_results: List[PipelineResult] = [] # Store text processing results
         
         # Reload keys just in case
         self.state.cloud_api_keys = self.config_manager.load_cloud_keys()

         try:
             # --- Get Pipeline Config ---
             pipeline_config = self.state.get_pipeline_config()

             # --- Text Processing Setup ---
             text_stages_selected = any(s in self.state.stages_to_run for s in PIPELINE_STAGES)
             if text_stages_selected:
                 ConsoleOutput.info(f"Initializing Text Processing Backend ({pipeline_config.model_provider})...")
                 start_init = time.time()
                 try:
                     local_kwargs = {}
                     if pipeline_config.model_provider == "local":
                         local_kwargs['quantization_config'] = pipeline_config.quantization_config
                         local_kwargs['layer_split_config'] = pipeline_config.layer_split_config
                         # We need the original model config from the registry for the filter/calculator
                         model_entry = self.registry.get_model(pipeline_config.model_specifier)
                         if not model_entry:
                             # This should ideally not happen if selected via menu, but handle anyway
                             raise ValueError(f"Model {pipeline_config.model_specifier} not found in registry for local processing.")
                         # Convert ModelEntry to the old ModelConfig dataclass
                         from config import ModelConfig # Ensure this is imported
                         model_conf_for_local = ModelConfig(
                             name=model_entry.name, model_id=model_entry.model_id,
                             supports_thinking=model_entry.supports_thinking,
                             thinking_tokens=model_entry.thinking_tokens or [],
                             max_context=model_entry.max_context,
                             optimal_chunk_size=model_entry.optimal_chunk_size,
                             temperature=model_entry.temperature,
                             top_p=model_entry.top_p,
                             max_new_tokens=model_entry.max_new_tokens,
                             quantization_support=model_entry.quantization_support or ["4bit", "8bit"]
                         )
                         local_kwargs['model_config'] = model_conf_for_local
                         
                         text_backend = get_llm_backend(
                             provider=pipeline_config.model_provider,
                             model_specifier=pipeline_config.model_specifier,
                             api_keys=self.state.cloud_api_keys,
                             hyperparameters=pipeline_config.hyperparameters,
                             **local_kwargs
                         )
                     elif pipeline_config.model_provider == "local_gguf":
                         if not (LLAMACPP_AVAILABLE and LlamaCppConfig and LlamaCppBackend):
                             raise ImportError("llama-cpp-python is not installed, cannot use GGUF model.")
                         gguf_config = LlamaCppConfig(
                             model_path=Path(pipeline_config.model_specifier),
                             n_ctx=pipeline_config.hyperparameters.max_length or 2048,
                             n_gpu_layers=pipeline_config.layer_split_config.gpu_layers if pipeline_config.layer_split_config else -1,
                             n_batch=pipeline_config.hyperparameters.batch_size if hasattr(pipeline_config.hyperparameters, 'batch_size') else 512, # Example of mapping
                             # Add other LlamaCppConfig settings if needed, potentially from AppState
                         )
                         text_backend = LlamaCppBackend(
                             config=gguf_config,
                             hyperparameters=pipeline_config.hyperparameters
                         )
                     else: # Cloud providers
                          text_backend = get_llm_backend(
                              provider=pipeline_config.model_provider,
                              model_specifier=pipeline_config.model_specifier,
                              api_keys=self.state.cloud_api_keys,
                              hyperparameters=pipeline_config.hyperparameters
                          )

                     if text_backend is None:
                          raise ValueError(f"Could not create text backend for provider {pipeline_config.model_provider}. Check API keys and model names.")

                     ConsoleOutput.info(f"Loading/Initializing {text_backend.model_identifier}...")
                     if not text_backend.load_model(trust_remote_code=True): # Pass trust_remote_code
                         raise RuntimeError(f"Failed to load/initialize backend: {text_backend.model_identifier}")

                     init_time = time.time() - start_init
                     ConsoleOutput.success(f"Text backend initialized successfully in {init_time:.2f}s.")

                 except Exception as e:
                     ConsoleOutput.error(f"Failed to initialize text backend: {e}")
                     logger.error("Text backend initialization failed", exc_info=True)
                     input("Press Enter to return to menu...")
                     return # Stop execution if backend fails

             # --- Audio Generation Setup ---
             if self.state.run_audio_generation:
                 ConsoleOutput.info(f"Initializing Audio Generation Backend ({self.state.audio_model_provider})...")
                 start_init_audio = time.time()
                 try:
                     audio_backend = get_audio_backend(
                         provider=self.state.audio_model_provider,
                         model_specifier=self.state.audio_model_specifier,
                         api_keys=self.state.cloud_api_keys,
                         config=self.state.audio_config 
                     )

                     if audio_backend is None:
                         raise ValueError(f"Could not create audio backend for provider {self.state.audio_model_provider}.")

                     ConsoleOutput.info(f"Loading/Initializing {audio_backend.model_identifier}...")
                     if not audio_backend.load_model():
                         raise RuntimeError(f"Failed to load/initialize audio backend.")

                     init_time_audio = time.time() - start_init_audio
                     ConsoleOutput.success(f"Audio backend initialized successfully in {init_time_audio:.2f}s.")

                 except Exception as e:
                     ConsoleOutput.error(f"Failed to initialize audio backend: {e}")
                     logger.error("Audio backend initialization failed", exc_info=True)
                     audio_backend = None # Ensure audio is skipped if init fails
                     input("Press Enter to continue...")


             # --- Run Text Processing ---
             pipeline = ProcessingPipeline(pipeline_config)
             if text_backend:
                 pipeline.llm_backend = text_backend # Inject the backend
             
             pipeline.set_stages_to_run(self.state.stages_to_run) # Tell pipeline which stages to run

             if text_stages_selected:
                  if not text_backend:
                      ConsoleOutput.error("Text processing stages skipped as backend is not available.")
                  else:
                      ConsoleOutput.header("📝 Starting Text Processing 📝")
                      ConsoleOutput.info(f"Running stages: {', '.join(self.state.stages_to_run)}")
                      
                      if len(self.state.input_files) == 1:
                          output_base = self.state.output_filename or self.state.input_files[0].stem
                          output_p_base = self.state.output_dir / output_base # Base path without suffix
                          result = pipeline.process_file(self.state.input_files[0], output_path=output_p_base)
                          all_results.append(result)
                      else:
                          # output_filename is ignored in batch mode
                          batch_results = pipeline.process_batch(self.state.input_files, output_dir=self.state.output_dir)
                          all_results.extend(batch_results)
             else:
                  ConsoleOutput.info("No text processing stages selected. Skipping.")
                  # If only audio is selected, we need to populate all_results with input files
                  # if they are text, so the audio stage can find them.
                  if self.state.run_audio_generation:
                       for f in self.state.input_files:
                            if f.suffix.lower() in ['.md', '.txt']:
                                 # Create a dummy result object
                                 all_results.append(PipelineResult(
                                      success=True,
                                      input_file=f,
                                      output_file=f, # Use the input file as the "output" text file
                                      processing_time=0,
                                      stages_completed=[],
                                      error_message=None,
                                      statistics=None
                                 ))


             # --- Run Audio Generation ---
             if self.state.run_audio_generation:
                 if not audio_backend:
                     ConsoleOutput.error("Audio generation selected, but backend failed to initialize. Skipping audio.")
                 else:
                     ConsoleOutput.header("🔊 Starting Audio Generation 🔊")
                     
                     text_files_to_convert = [res.output_file for res in all_results if res.success and res.output_file and res.output_file.exists()]

                     if not text_files_to_convert:
                         ConsoleOutput.warning("No valid text output files found to generate audio from.")
                     else:
                         ConsoleOutput.info(f"Found {len(text_files_to_convert)} file(s) to convert to audio...")
                         successful_audio_count = 0
                         with LoggingProgress(logger, "Generating audio files", len(text_files_to_convert)) as progress:
                             for i, text_file in enumerate(text_files_to_convert):
                                 ConsoleOutput.subsection(f"Audio for: {text_file.name} ({i+1}/{len(text_files_to_convert)})")
                                 try:
                                     with open(text_file, 'r', encoding='utf-8') as f:
                                         text_content = f.read()
                                         
                                     # Clean text for TTS: remove markdown metadata
                                     text_content = re.sub(r'^---.*?---', '', text_content, flags=re.DOTALL | re.MULTILINE).strip()
                                     
                                     # Clean for podcast mode: remove speaker tags
                                     if self.state.processing_mode == ProcessingMode.PODCAST:
                                          text_content = re.sub(r'^\*\*\[Speaker.*?\]:\*\*\n*', '', text_content, flags=re.MULTILINE)
                                     
                                     # Clean for technical mode: remove note/warning emojis/prefixes
                                     if self.state.processing_mode == ProcessingMode.TECHNICAL:
                                         text_content = re.sub(r'📝 \*\*Note:\*\* ', '', text_content)
                                         text_content = re.sub(r'⚠️ \*\*Warning:\*\* ', '', text_content)
                                         text_content = re.sub(r'💡 \*\*Tip:\*\* ', '', text_content)
                                         text_content = re.sub(r'❗ \*\*Important:\*\* ', '', text_content)

                                     # Remove markdown formatting like *, **, `
                                     text_content = re.sub(r'(\*\*|\*|`|#+\s)', '', text_content)

                                     if not text_content.strip():
                                          ConsoleOutput.warning("Text file is empty after cleaning, skipping audio generation.")
                                          progress.update(1)
                                          continue

                                     # Determine audio output path
                                     audio_output_path = text_file.with_suffix(f".{self.state.audio_config.output_format}")

                                     audio_result = audio_backend.generate_audio(
                                         text=text_content,
                                         output_path=audio_output_path,
                                         chunk_text=(self.state.audio_model_provider == 'local') # Only chunk for local models
                                     )

                                     if audio_result:
                                         ConsoleOutput.success(f"Audio saved: {audio_result.audio_path.name} ({audio_result.duration_formatted})")
                                         successful_audio_count += 1
                                     else:
                                         ConsoleOutput.error(f"Failed to generate audio for {text_file.name}")

                                 except FileNotFoundError:
                                     ConsoleOutput.error(f"Text file not found: {text_file}")
                                 except Exception as audio_err:
                                      ConsoleOutput.error(f"Error during audio generation for {text_file.name}: {audio_err}")
                                      logger.error(f"Audio generation error for {text_file.name}", exc_info=True)
                                 finally:
                                     progress.update(1)

                         ConsoleOutput.info(f"Audio generation finished. Successfully generated {successful_audio_count} audio file(s).")
             
             # --- Final Summary ---
             ConsoleOutput.header("✅ Processing Run Finished ✅")
             success_count = sum(1 for r in all_results if r.success)
             fail_count = len(all_results) - success_count
             if text_stages_selected:
                 print(f"Text Processing: {success_count} succeeded, {fail_count} failed.")
             if self.state.run_audio_generation:
                 print(f"Audio Generation: Check logs above for status.") # Already printed success count
             
             if pipeline and text_stages_selected:
                 report_path = pipeline.file_handler.save_processing_report()
                 print(f"Processing report saved to: {report_path}")

         except KeyboardInterrupt:
            ConsoleOutput.warning("\nProcessing interrupted by user.")
            logger.warning("Pipeline execution interrupted by user.")
         except Exception as e:
             ConsoleOutput.error(f"🚨 Pipeline execution encountered a critical error: {e}")
             logger.critical("Pipeline execution failed", exc_info=True)
         finally:
             # Cleanup backends and pipeline resources
             ConsoleOutput.info("Cleaning up resources...")
             if text_backend:
                 try: text_backend.unload_model()
                 except Exception as unload_err: logger.warning(f"Error unloading text backend: {unload_err}")
             if audio_backend:
                 try: audio_backend.unload_model()
                 except Exception as unload_err: logger.warning(f"Error unloading audio backend: {unload_err}")
             # Ensure file handler report is saved even if pipeline object wasn't fully used
             if self.state.input_files and not (pipeline and 'save' in self.state.stages_to_run):
                 # Manually save a partial report if needed, or just let pipeline's cleanup handle it
                 if pipeline:
                     try: pipeline.cleanup()
                     except Exception as cleanup_err: logger.warning(f"Error during final pipeline cleanup: {cleanup_err}")
                 else:
                     # If pipeline never even started, just log
                     logger.info("Pipeline did not initialize, no report to save.")
             
             ConsoleOutput.info("Cleanup complete. Returning to main menu.")
             time.sleep(2) # Pause to let user see final messages


# --- Helper Functions (Quantization, Layer Split) ---
# (Duplicated from llamanote.py for use in menu system)

def create_quantization_config(args: argparse.Namespace) -> QuantizationConfig:
    """Create quantization configuration from arguments (for local backend)."""
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    compute_dtype_str = getattr(args, 'compute_dtype', 'bfloat16')
    compute_dtype = dtype_map.get(compute_dtype_str, torch.bfloat16)
    quant_method = getattr(args, 'quantization', DEFAULT_QUANTIZATION)
    
    return QuantizationConfig(
        method=quant_method,
        compute_dtype=compute_dtype,
        use_double_quant=True if quant_method == "4bit" else False,
        quant_type="nf4",
        bnb_4bit_use_double_quant=True if quant_method == "4bit" else False
    )

def create_layer_split_config(args: argparse.Namespace) -> LayerSplitConfig:
    """Create layer split configuration from arguments (for local backend)."""
    max_gpu_mem_str = getattr(args, 'max_gpu_memory', "10GB")
    max_cpu_mem_str = getattr(args, 'max_cpu_memory', "30GB")
    gpu_layers_count = getattr(args, 'gpu_layers', DEFAULT_GPU_LAYERS)
    no_split = getattr(args, 'no_layer_split', not ENABLE_LAYER_SPLITTING)

    max_gpu_memory = {}
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus): 
            max_gpu_memory[i] = max_gpu_mem_str
    
    return LayerSplitConfig(
        enabled=not no_split and gpu_layers_count != 0,
        gpu_layers=gpu_layers_count,
        max_gpu_memory=max_gpu_memory,
        max_cpu_memory=max_cpu_mem_str,
        offload_folder=OFFLOAD_DIR
    )
