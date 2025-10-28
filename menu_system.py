"""
Menu System Module - Refactored
Interactive menu interface for LlamaNote Enhanced
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Assuming these modules exist and provide the necessary classes/functions
from loggerConf import ConsoleOutput, get_logger_conf
from config_manager import ConfigManager
from config_base import DEFAULT_MODEL
from huggingface_search import HuggingFaceSearch, FilterBuilder, SearchFilters, ModelTask, ModelInfo as HFModelInfo
from audio_generator import AudioConfig, AudioGenerator
from processing_pipeline import ProcessingPipeline, PipelineConfig, ProcessingMode, PIPELINE_STAGES
from text_processor import ChunkingStrategy
from model_registry import get_registry, ModelEntry
from model_hub import InteractiveModelBrowser, ModelHub
from hyperparameters import HyperparameterConfig, InteractiveHyperparameterEditor, get_hyperparameter_help

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
    label: str
    action: Optional[Callable] = None
    submenu: Optional['Menu'] = None
    description: Optional[str] = None

class Menu:
    def __init__(self, title: str, items: List[MenuItem], parent: Optional['Menu'] = None):
        self.title = title
        self.items = items
        self.parent = parent

    def display(self) -> MenuAction:
        while True:
            ConsoleOutput.header(self.title)
            for item in self.items:
                desc = f" - {item.description}" if item.description else ""
                print(f"  {item.key}. {item.label}{desc}")

            nav_options = []
            if self.parent:
                nav_options.append("b. Back")
            nav_options.append("q. Quit")
            print("\n  " + " | ".join(nav_options))
            print("-" * 60)

            choice = input("Select option: ").strip().lower()

            if choice == 'q':
                return MenuAction.EXIT
            if choice == 'b' and self.parent:
                return MenuAction.BACK

            selected_item = next((item for item in self.items if item.key.lower() == choice), None)

            if selected_item:
                result = MenuAction.CONTINUE
                if selected_item.submenu:
                    result = selected_item.submenu.display()
                elif selected_item.action:
                    result = selected_item.action() or MenuAction.CONTINUE # Ensure actions return a MenuAction or None

                if result == MenuAction.EXIT:
                    return MenuAction.EXIT
                elif result == MenuAction.RUN: # Propagate RUN action up
                    return MenuAction.RUN
                # CONTINUE or BACK means stay in the current/parent loop
            else:
                ConsoleOutput.warning("Invalid selection")
            time.sleep(0.5) # Small pause

    def add_item(self, item: MenuItem):
        self.items.append(item)

# --- Menu System State ---

@dataclass
class AppState:
    input_files: List[Path] = field(default_factory=list)
    output_dir: Path = Path("output")
    output_filename: Optional[str] = None # For single file output base name
    processing_mode: ProcessingMode = ProcessingMode.PODCAST
    text_model_id: str = DEFAULT_MODEL # Default text processing model
    audio_model_id: str = "microsoft/speecht5_tts" # Default audio model
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    audio_config: AudioConfig = field(default_factory=AudioConfig)
    stages_to_run: List[str] = field(default_factory=lambda: list(PIPELINE_STAGES))
    run_audio_generation: bool = False # Separate flag for audio stage

    def get_pipeline_config(self) -> PipelineConfig:
        # Creates a PipelineConfig based on current AppState
        return PipelineConfig(
            mode=self.processing_mode,
            model_name=self.text_model_id, # Use the selected model ID
            # memory_profile might need adjustment based on selected model
            # For simplicity, using a default. Could be made configurable.
            memory_profile="medium_vram",
            chunking_strategy=ChunkingStrategy.WORD_BOUNDARY, # Default, make configurable?
            chunk_size=1000, # Default, make configurable?
            markdown_style=self.processing_mode.value, # Match style to mode
            output_format="markdown" # Default, make configurable?
        )

# --- Main Menu System Class ---

class MenuSystem:
    def __init__(self):
        self.state = AppState()
        self.config_manager = ConfigManager() # For saving/loading presets if needed
        self.hf_search = HuggingFaceSearch()
        self.model_hub = ModelHub()
        self.registry = get_registry() # Get registry instance

        # 1. Create the main menu object first, potentially empty or with non-submenu items
        self.main_menu = Menu("LlamaNote - Main Menu", [])

        # 2. Build submenus, passing the now-existing self.main_menu as parent
        model_settings_submenu = self._build_model_settings_menu()

        # 3. Build the main menu items, including the submenus
        main_menu_items = [
            MenuItem("1", "Select Input and Stages", self._select_input_menu, description="Choose PDF(s) and processing steps"),
            MenuItem("2", "Model Settings", submenu=model_settings_submenu, description="Configure text and audio model(s)"),
            MenuItem("3", "Output Settings", self._output_settings_menu, description="Set Output Location"),
            MenuItem("4", "View Current Setup", self._view_current_setup, description="Review selections"),
            MenuItem("5", "Run Processing", self._confirm_and_run, description="Execute the pipeline"),
        ]

        # 4. Assign the items to the main menu
        self.main_menu.items = main_menu_items
        logger.info("Initialized MenuSystem")

    def run(self):
        result = self.main_menu.display()
        if result == MenuAction.EXIT:
            ConsoleOutput.info("Exiting LlamaNote.")
        elif result == MenuAction.RUN:
             self._execute_pipeline()
             ConsoleOutput.info("Processing finished. Exiting.")
        else:
             ConsoleOutput.info("Exiting LlamaNote.")

    # --- Build Menus ---

    def _build_main_menu(self) -> Menu:
        items = [
            MenuItem("1", "Select Input & Stages", self._select_input_menu, description="Choose PDF(s) and processing steps"),
            MenuItem("2", "Model Settings", submenu=self._build_model_settings_menu(), description="Configure text and audio models"),
            MenuItem("3", "Output Settings", self._output_settings_menu, description="Set output location"),
            MenuItem("4", "View Current Setup", self._view_current_setup, description="Review selections"),
            MenuItem("5", "Run Processing", self._confirm_and_run, description="Execute the pipeline"),
        ]
        return Menu("LlamaNote Enhanced - Main Menu", items)

    def _build_model_settings_menu(self) -> Menu:
        items = [
            MenuItem("1", "Set Text Processing Model", self._set_text_model_menu, description=f"Current: {self.state.text_model_id}"),
            MenuItem("2", "Set Text Model Hyperparameters", self._set_text_hyperparameters, description="Configure generation settings"),
            MenuItem("3", "Set Audio Generation Model", self._set_audio_model_menu, description=f"Current: {self.state.audio_model_id}"),
            # MenuItem("4", "Set Audio Model Hyperparameters", self._set_audio_hyperparameters, description="Configure TTS settings"), # Add if needed
        ]
        # Dynamically update description before display if needed, or pass self.state to Menu.display
        menu = Menu("Model Settings", items, parent=self.main_menu)
        # Hacky way to update descriptions - better to pass state down
        menu.items[0].action = lambda: self._update_menu_description_and_run(menu.items[0], f"Current: {self.state.text_model_id}", self._set_text_model_menu)
        menu.items[2].action = lambda: self._update_menu_description_and_run(menu.items[2], f"Current: {self.state.audio_model_id}", self._set_audio_model_menu)
        return menu

    # Helper to update menu item description before running action
    def _update_menu_description_and_run(self, item: MenuItem, description: str, action: Callable):
        item.description = description
        return action()

    # --- Menu Actions ---

    def _select_input_menu(self):
        ConsoleOutput.section("Select Input Source")
        path_str = input("Enter PDF file path OR folder path: ").strip()
        if not path_str:
            return MenuAction.CONTINUE

        path = Path(path_str)

        if path.is_file() and path.suffix.lower() == ".pdf":
            self.state.input_files = [path]
            ConsoleOutput.success(f"Selected file: {path.name}")
        elif path.is_dir():
            ConsoleOutput.info(f"Selected directory: {path}")
            pdfs_in_dir = list(path.rglob("*.pdf"))
            if not pdfs_in_dir:
                ConsoleOutput.warning("No PDF files found in this directory (recursive).")
                self.state.input_files = []
                return MenuAction.CONTINUE

            print(f"Found {len(pdfs_in_dir)} PDF files.")
            print("  1. Process ALL PDF files found (recursive)")
            print("  2. Select a single PDF file from the directory") # Simplified - no TUI browser
            choice = input("Select option (1-2): ").strip()

            if choice == '1':
                self.state.input_files = pdfs_in_dir
                ConsoleOutput.success(f"Selected {len(pdfs_in_dir)} files for batch processing.")
            elif choice == '2':
                 # Simplified selection - list and choose by number
                 print("\nAvailable PDF files:")
                 for i, pdf_path in enumerate(pdfs_in_dir[:20]): # Show first 20
                     print(f"  {i+1}. {pdf_path.relative_to(path)}")
                 if len(pdfs_in_dir) > 20:
                     print("  ...")

                 file_choice = input(f"Select file number (1-{min(len(pdfs_in_dir), 20)}): ").strip()
                 try:
                     idx = int(file_choice) - 1
                     if 0 <= idx < min(len(pdfs_in_dir), 20):
                         self.state.input_files = [pdfs_in_dir[idx]]
                         ConsoleOutput.success(f"Selected file: {pdfs_in_dir[idx].name}")
                     else:
                         ConsoleOutput.warning("Invalid selection.")
                         self.state.input_files = []
                 except ValueError:
                     ConsoleOutput.warning("Invalid input.")
                     self.state.input_files = []
            else:
                 ConsoleOutput.warning("Invalid choice.")
                 self.state.input_files = []

        else:
            ConsoleOutput.error("Invalid path. Please provide a valid PDF file or directory.")
            self.state.input_files = []

        if self.state.input_files:
            self._select_stages_menu() # Chain to stage selection

        return MenuAction.CONTINUE

    def _select_stages_menu(self):
        ConsoleOutput.section("Select Processing Stages")
        all_stages = list(PIPELINE_STAGES)
        print("Available stages:")
        for i, stage in enumerate(all_stages):
             included = "[X]" if stage in self.state.stages_to_run else "[ ]"
             print(f"  {i+1}. {included} {stage}")

        # Audio Generation Toggle
        audio_included = "[X]" if self.state.run_audio_generation else "[ ]"
        print(f"  A. {audio_included} generate_audio (Runs after 'save')")

        print("\nEnter numbers/A to toggle, 'all' to select all text stages, 'none' to deselect all, 'done' to confirm.")

        while True:
            choice = input("Toggle stage (e.g., '1', 'a', 'all', 'done'): ").strip().lower()
            if choice == 'done':
                break
            elif choice == 'all':
                self.state.stages_to_run = list(PIPELINE_STAGES)
                self.state.run_audio_generation = False # Keep audio separate toggle
            elif choice == 'none':
                self.state.stages_to_run = []
                self.state.run_audio_generation = False
            elif choice == 'a':
                self.state.run_audio_generation = not self.state.run_audio_generation
            elif choice.isdigit():
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(all_stages):
                        stage_name = all_stages[idx]
                        if stage_name in self.state.stages_to_run:
                            self.state.stages_to_run.remove(stage_name)
                        else:
                            # Add stage back in correct order
                            temp_stages = []
                            for s in PIPELINE_STAGES:
                                if s in self.state.stages_to_run or s == stage_name:
                                    temp_stages.append(s)
                            self.state.stages_to_run = temp_stages

                    else:
                        ConsoleOutput.warning("Invalid stage number.")
                except ValueError:
                    ConsoleOutput.warning("Invalid input.")
            else:
                ConsoleOutput.warning("Invalid input.")

            # Redisplay stages
            ConsoleOutput.section("Select Processing Stages")
            for i, stage in enumerate(all_stages):
                included = "[X]" if stage in self.state.stages_to_run else "[ ]"
                print(f"  {i+1}. {included} {stage}")
            audio_included = "[X]" if self.state.run_audio_generation else "[ ]"
            print(f"  A. {audio_included} generate_audio (Runs after 'save')")
            print("\nEnter numbers/A to toggle, 'all'/'none', 'done'.")


        ConsoleOutput.success("Stages selected.")
        return MenuAction.CONTINUE


    def _set_text_model_menu(self):
         ConsoleOutput.section("Set Text Processing Model")
         print("  1. Search Hugging Face Hub")
         print("  2. Enter Model ID directly")
         print(f"  Current Model: {self.state.text_model_id}")

         choice = input("Select option (1-2, or b to back): ").strip().lower()

         if choice == '1':
             # Use InteractiveModelBrowser for searching
             browser = InteractiveModelBrowser(self.model_hub)
             # Filter for text generation models
             selected_hf_model: Optional[HFModelInfo] = browser.browse(task="text-generation", library="transformers")
             if selected_hf_model:
                 # Check if model exists in our registry, add if not
                 model_entry = self.registry.get_model(selected_hf_model.model_id)
                 if not model_entry:
                     self.registry.add_from_model_info(selected_hf_model)
                     ConsoleOutput.info(f"Added {selected_hf_model.model_id} to local registry.")
                 self.state.text_model_id = selected_hf_model.model_id
                 ConsoleOutput.success(f"Text model set to: {self.state.text_model_id}")
                 # Reset hyperparameters when model changes? Ask user?
                 reset_hp = input("Reset hyperparameters to default for new model? (y/n, default y): ").strip().lower()
                 if reset_hp != 'n':
                     self.state.hyperparameters = HyperparameterConfig()
                     ConsoleOutput.info("Hyperparameters reset to defaults.")

         elif choice == '2':
             model_id = input("Enter Hugging Face Model ID: ").strip()
             if model_id:
                 # Validate? For now, just set it. Add to registry if needed.
                 model_entry = self.registry.get_model(model_id)
                 if not model_entry:
                      # Try fetching info to add basic entry
                      hf_info = self.model_hub.get_model_info(model_id)
                      if hf_info:
                          self.registry.add_from_model_info(hf_info)
                          ConsoleOutput.info(f"Added {model_id} to local registry.")
                      else:
                          ConsoleOutput.warning(f"Could not fetch info for {model_id}. Adding with defaults.")
                          # Add a minimal entry if fetch fails
                          self.registry.add_model(ModelEntry(model_id=model_id, name=model_id.split('/')[-1], author="Unknown"))

                 self.state.text_model_id = model_id
                 ConsoleOutput.success(f"Text model set to: {self.state.text_model_id}")
                 reset_hp = input("Reset hyperparameters to default for new model? (y/n, default y): ").strip().lower()
                 if reset_hp != 'n':
                     self.state.hyperparameters = HyperparameterConfig()
                     ConsoleOutput.info("Hyperparameters reset to defaults.")
         elif choice == 'b':
             return MenuAction.BACK
         else:
             ConsoleOutput.warning("Invalid choice.")

         return MenuAction.CONTINUE


    def _set_text_hyperparameters(self):
        ConsoleOutput.section(f"Configure Hyperparameters for {self.state.text_model_id}")
        editor = InteractiveHyperparameterEditor(self.state.hyperparameters)
        self.state.hyperparameters = editor.edit() # This starts the interactive editing loop
        ConsoleOutput.success("Hyperparameters updated.")
        # Need to save this config? Or just use it for the run? For now, just use for run.
        return MenuAction.CONTINUE

    def _set_audio_model_menu(self):
        ConsoleOutput.section("Set Audio Generation Model (TTS)")
        print("  1. Search Hugging Face Hub")
        print("  2. Enter Model ID directly")
        print(f"  Current Model: {self.state.audio_model_id}")

        choice = input("Select option (1-2, or b to back): ").strip().lower()

        if choice == '1':
            browser = InteractiveModelBrowser(self.model_hub)
            # Filter for TTS models
            selected_hf_model: Optional[HFModelInfo] = browser.browse(task="text-to-speech")
            if selected_hf_model:
                self.state.audio_model_id = selected_hf_model.model_id
                self.state.audio_config.model_id = selected_hf_model.model_id # Update audio config too
                ConsoleOutput.success(f"Audio model set to: {self.state.audio_model_id}")
                # Potentially download model here or let AudioGenerator handle it

        elif choice == '2':
            model_id = input("Enter Hugging Face Model ID (TTS): ").strip()
            if model_id:
                # Add basic validation or just set
                self.state.audio_model_id = model_id
                self.state.audio_config.model_id = model_id
                ConsoleOutput.success(f"Audio model set to: {self.state.audio_model_id}")
        elif choice == 'b':
            return MenuAction.BACK
        else:
            ConsoleOutput.warning("Invalid choice.")

        return MenuAction.CONTINUE

    def _output_settings_menu(self):
        ConsoleOutput.section("Output Settings")
        print(f"Current Output Directory: {self.state.output_dir.resolve()}")
        if self.state.output_filename:
             print(f"Current Base Filename: {self.state.output_filename} (Timestamps/suffixes will be added)")
        else:
             print("Current Base Filename: Default (derived from input filename)")

        new_dir = input("Enter new output directory (leave blank to keep current): ").strip()
        if new_dir:
            try:
                path = Path(new_dir)
                path.mkdir(parents=True, exist_ok=True) # Ensure it exists
                self.state.output_dir = path
                ConsoleOutput.success(f"Output directory set to: {self.state.output_dir.resolve()}")
            except Exception as e:
                ConsoleOutput.error(f"Invalid directory path: {e}")

        # Setting base filename only makes sense for single file processing
        if len(self.state.input_files) == 1:
             new_name = input("Enter base output filename (without extension, leave blank for default): ").strip()
             self.state.output_filename = new_name if new_name else None
             if self.state.output_filename:
                 ConsoleOutput.success(f"Base output filename set to: {self.state.output_filename}")
             else:
                 ConsoleOutput.info("Using default output filename based on input.")
        elif self.state.input_files:
             ConsoleOutput.info("Output filename is ignored during batch processing.")
             self.state.output_filename = None # Ensure it's None for batch

        return MenuAction.CONTINUE

    def _view_current_setup(self):
        ConsoleOutput.section("Current Processing Setup")
        print(f"Input File(s):")
        if self.state.input_files:
             for f in self.state.input_files[:5]: # Show first 5
                 print(f"  - {f.name}")
             if len(self.state.input_files) > 5:
                 print(f"  ... and {len(self.state.input_files) - 5} more.")
        else:
             print("  None selected.")

        print(f"\nOutput Directory: {self.state.output_dir.resolve()}")
        if len(self.state.input_files) == 1 and self.state.output_filename:
             print(f"Base Output Filename: {self.state.output_filename}")
        else:
             print(f"Base Output Filename: Default (based on input)")

        print(f"\nProcessing Mode: {self.state.processing_mode.value}")
        print(f"Text Model: {self.state.text_model_id}")
        print(f"Text Hyperparameters:")
        hp_dict = self.state.hyperparameters.to_dict()
        # Display only a few key hyperparameters for brevity
        print(f"  - temperature: {hp_dict.get('temperature', 'Default')}")
        print(f"  - top_p: {hp_dict.get('top_p', 'Default')}")
        print(f"  - max_new_tokens: {hp_dict.get('max_new_tokens', 'Auto')}")
        print(f"  (Use Model Settings menu to see/edit all)")

        print(f"\nStages to Run: {', '.join(self.state.stages_to_run)}")
        print(f"Generate Audio After: {'Yes' if self.state.run_audio_generation else 'No'}")
        if self.state.run_audio_generation:
             print(f"Audio Model: {self.state.audio_model_id}")
             # Add more audio config details if needed

        input("\nPress Enter to continue...")
        return MenuAction.CONTINUE

    def _confirm_and_run(self) -> MenuAction:
        ConsoleOutput.section("Confirm and Run")
        self._view_current_setup() # Show summary again

        if not self.state.input_files:
             ConsoleOutput.error("No input files selected. Please select input first.")
             input("Press Enter to continue...")
             return MenuAction.CONTINUE
        if not self.state.stages_to_run and not self.state.run_audio_generation:
             ConsoleOutput.error("No processing stages selected.")
             input("Press Enter to continue...")
             return MenuAction.CONTINUE

        confirm = input("\nProceed with processing? (y/n): ").strip().lower()
        if confirm == 'y':
            return MenuAction.RUN # Signal to the main loop to execute
        else:
            ConsoleOutput.info("Processing cancelled.")
            return MenuAction.CONTINUE

    def _execute_pipeline(self):
         """Executes the pipeline based on the current state."""
         ConsoleOutput.header("Starting Processing Pipeline")

         if not self.state.input_files:
             ConsoleOutput.error("Execution failed: No input files specified.")
             return

         pipeline_config = self.state.get_pipeline_config()
         # Apply selected hyperparameters
         pipeline_config.hyperparameters = self.state.hyperparameters # Assuming PipelineConfig can take hyperparameters

         # Initialize pipeline (outside the loop for batch)
         pipeline = None
         audio_gen = None
         try:
             pipeline = ProcessingPipeline(pipeline_config)
             # Potentially modify pipeline to run only selected stages
             # pipeline.set_stages_to_run(self.state.stages_to_run) # Needs implementation in ProcessingPipeline

             ConsoleOutput.info(f"Processing {len(self.state.input_files)} file(s)...")

             all_results: List[PipelineResult] = []

             # --- Process Files ---
             if 'save' in self.state.stages_to_run: # Only run text pipeline if save stage is included
                 if len(self.state.input_files) == 1:
                     output_base = self.state.output_filename or self.state.input_files[0].stem
                     # Construct output path (let FileHandler add timestamp/suffix later)
                     output_p = self.state.output_dir / f"{output_base}" # FileHandler will add suffix/ext
                     result = pipeline.process_file(self.state.input_files[0], output_path=output_p)
                     all_results.append(result)
                 else:
                     # Batch processing - output path determined by FileHandler based on input names
                     batch_results = pipeline.process_batch(self.state.input_files)
                     all_results.extend(batch_results)
             else:
                 ConsoleOutput.warning("Skipping text processing pipeline as 'save' stage not selected.")


             # --- Audio Generation ---
             if self.state.run_audio_generation:
                 ConsoleOutput.header("Starting Audio Generation")
                 # Get list of text files to convert (successful results from previous step)
                 text_files_to_convert = []
                 if 'save' in self.state.stages_to_run:
                     text_files_to_convert = [res.output_file for res in all_results if res.success and res.output_file]
                 elif self.state.input_files and all(f.suffix.lower() in ['.md', '.txt'] for f in self.state.input_files):
                     # If only audio stage is selected and inputs are text files
                     text_files_to_convert = self.state.input_files
                 else:
                      ConsoleOutput.error("Audio generation requires text files. Run text processing first or provide .md/.txt files.")


                 if text_files_to_convert:
                     try:
                         # Initialize Audio Generator with current config
                         self.state.audio_config.model_id = self.state.audio_model_id
                         audio_gen = AudioGenerator(self.state.audio_config)
                         if not audio_gen.model and not audio_gen.pipeline:
                             ConsoleOutput.info(f"Loading audio model: {audio_gen.config.model_id}")
                             if not audio_gen.load_model():
                                 raise RuntimeError("Failed to load audio model.")

                         ConsoleOutput.info(f"Generating audio for {len(text_files_to_convert)} file(s)...")
                         for text_file in text_files_to_convert:
                             ConsoleOutput.section(f"Generating audio for {text_file.name}")
                             try:
                                 with open(text_file, 'r', encoding='utf-8') as f:
                                     text_content = f.read()
                                 # Determine audio output path
                                 audio_output_path = text_file.with_suffix(f".{audio_gen.config.output_format}")
                                 audio_result = audio_gen.generate_audio(text_content, audio_output_path)
                                 if audio_result:
                                     ConsoleOutput.success(f"Audio saved to {audio_result.audio_path}")
                                 else:
                                     ConsoleOutput.error(f"Failed to generate audio for {text_file.name}")
                             except Exception as audio_err:
                                  ConsoleOutput.error(f"Error generating audio for {text_file.name}: {audio_err}")
                                  logger.error(f"Audio generation error for {text_file.name}", exc_info=True)

                     except Exception as e:
                         ConsoleOutput.error(f"Audio generation setup failed: {e}")
                         logger.error("Audio generation setup failed", exc_info=True)
                     finally:
                         if audio_gen:
                             audio_gen.unload_model()
                 else:
                      ConsoleOutput.warning("No successful text files available to generate audio from.")

         except Exception as e:
             ConsoleOutput.error(f"Pipeline execution failed: {e}")
             logger.error("Pipeline execution failed", exc_info=True)
         finally:
             if pipeline:
                 pipeline.cleanup() # Unloads models etc.


# --- Entry Point ---
# (This part should ideally be in llamanote.py as modified earlier)
# if __name__ == "__main__":
#     menu = MenuSystem()
#     menu.run()
