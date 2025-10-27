"""
Menu System Module
Interactive menu interface for LlamaNote Enhanced
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from logging_config import ConsoleOutput, get_logger
from config_manager import ConfigManager
from huggingface_search import HuggingFaceSearch, FilterBuilder, SearchFilters, ModelTask
from audio_generator import AudioConfig, AudioGenerator
from processing_pipeline import ProcessingPipeline, PipelineConfig, ProcessingMode
from text_processor import ChunkingStrategy

logger = get_logger(__name__)


class MenuAction(Enum):
    """Menu action types"""
    BACK = "back"
    EXIT = "exit"
    CONTINUE = "continue"


@dataclass
class MenuItem:
    """Menu item definition"""
    key: str
    label: str
    action: Optional[Callable] = None
    submenu: Optional['Menu'] = None
    description: Optional[str] = None


class Menu:
    """Interactive menu"""
    
    def __init__(self, title: str, items: List[MenuItem], 
                parent: Optional['Menu'] = None):
        """
        Initialize menu
        
        Args:
            title: Menu title
            items: Menu items
            parent: Parent menu for navigation
        """
        self.title = title
        self.items = items
        self.parent = parent
        
    def display(self):
        """Display menu and get selection"""
        while True:
            ConsoleOutput.header(self.title)
            
            # Display items
            for item in self.items:
                if item.description:
                    print(f"  {item.key}. {item.label} - {item.description}")
                else:
                    print(f"  {item.key}. {item.label}")
                    
            # Add navigation options
            if self.parent:
                print(f"  b. Back")
            print(f"  q. Quit")
            print()
            
            # Get selection
            choice = input("Select option: ").strip().lower()
            
            # Handle navigation
            if choice == 'q':
                return MenuAction.EXIT
            elif choice == 'b' and self.parent:
                return MenuAction.BACK
                
            # Find and execute item
            for item in self.items:
                if choice == item.key.lower():
                    if item.submenu:
                        result = item.submenu.display()
                        if result == MenuAction.EXIT:
                            return MenuAction.EXIT
                    elif item.action:
                        result = item.action()
                        if result == MenuAction.EXIT:
                            return MenuAction.EXIT
                    break
            else:
                ConsoleOutput.warning("Invalid selection")
                
    def add_item(self, item: MenuItem):
        """Add item to menu"""
        self.items.append(item)
        
    def remove_item(self, key: str):
        """Remove item from menu"""
        self.items = [i for i in self.items if i.key != key]


class MenuSystem:
    """Main menu system for LlamaNote Enhanced"""
    
    def __init__(self):
        """Initialize menu system"""
        self.config_manager = ConfigManager()
        self.current_config = self.config_manager.get_default_config()
        self.pipeline = None
        self.audio_generator = None
        self.hf_search = HuggingFaceSearch()
        
        # Build menu structure
        self.main_menu = self._build_main_menu()
        
        logger.info("Initialized MenuSystem")
        
    def _build_main_menu(self) -> Menu:
        """Build main menu structure"""
        main = Menu("LlamaNote Enhanced - Main Menu", [])
        
        # Process PDF
        main.add_item(MenuItem(
            "1", "Process PDF",
            action=self.process_pdf_menu,
            description="Convert PDF to formatted text"
        ))
        
        # Generate Audio
        main.add_item(MenuItem(
            "2", "Generate Audio",
            action=self.audio_generation_menu,
            description="Convert text to speech"
        ))
        
        # Search Models
        main.add_item(MenuItem(
            "3", "Search TTS Models",
            action=self.search_models_menu,
            description="Search and download models from HuggingFace"
        ))
        
        # Configuration
        config_menu = self._build_config_menu(main)
        main.add_item(MenuItem(
            "4", "Configuration",
            submenu=config_menu,
            description="Manage configurations and settings"
        ))
        
        # Batch Processing
        main.add_item(MenuItem(
            "5", "Batch Processing",
            action=self.batch_processing_menu,
            description="Process multiple files"
        ))
        
        # Settings
        settings_menu = self._build_settings_menu(main)
        main.add_item(MenuItem(
            "6", "Settings",
            submenu=settings_menu,
            description="Application settings"
        ))
        
        return main
        
    def _build_config_menu(self, parent: Menu) -> Menu:
        """Build configuration menu"""
        menu = Menu("Configuration Management", [], parent)
        
        menu.add_item(MenuItem(
            "1", "Load Configuration",
            action=self.load_configuration,
            description="Load saved configuration"
        ))
        
        menu.add_item(MenuItem(
            "2", "Save Configuration",
            action=self.save_configuration,
            description="Save current configuration"
        ))
        
        menu.add_item(MenuItem(
            "3", "List Configurations",
            action=self.list_configurations,
            description="Show all saved configurations"
        ))
        
        menu.add_item(MenuItem(
            "4", "Delete Configuration",
            action=self.delete_configuration,
            description="Delete a saved configuration"
        ))
        
        menu.add_item(MenuItem(
            "5", "Import/Export",
            action=self.import_export_menu,
            description="Import or export configurations"
        ))
        
        menu.add_item(MenuItem(
            "6", "Manage Filters",
            action=self.manage_filters_menu,
            description="Manage search filter presets"
        ))
        
        menu.add_item(MenuItem(
            "7", "Backup/Restore",
            action=self.backup_restore_menu,
            description="Backup or restore all configurations"
        ))
        
        return menu
        
    def _build_settings_menu(self, parent: Menu) -> Menu:
        """Build settings menu"""
        menu = Menu("Settings", [], parent)
        
        menu.add_item(MenuItem(
            "1", "Processing Settings",
            action=self.processing_settings,
            description="Configure PDF processing"
        ))
        
        menu.add_item(MenuItem(
            "2", "Audio Settings",
            action=self.audio_settings,
            description="Configure audio generation"
        ))
        
        menu.add_item(MenuItem(
            "3", "Model Settings",
            action=self.model_settings,
            description="Configure language models"
        ))
        
        # Path configuration submenu
        paths_menu = self._build_paths_menu(menu)
        menu.add_item(MenuItem(
            "4", "Path Configuration",
            submenu=paths_menu,
            description="Configure directory paths"
        ))
        
        menu.add_item(MenuItem(
            "5", "View Current Settings",
            action=self.view_current_settings,
            description="Display all current settings"
        ))
        
        return menu
        
    def _build_paths_menu(self, parent: Menu) -> Menu:
        """Build paths configuration menu"""
        menu = Menu("Path Configuration", [], parent)
        
        menu.add_item(MenuItem(
            "1", "Set Config Directory",
            action=lambda: self.set_custom_path("config"),
            description="Set configuration directory"
        ))
        
        menu.add_item(MenuItem(
            "2", "Set Filter Directory",
            action=lambda: self.set_custom_path("filter"),
            description="Set filter presets directory"
        ))
        
        menu.add_item(MenuItem(
            "3", "Set Audio Directory",
            action=lambda: self.set_custom_path("audio"),
            description="Set audio configs directory"
        ))
        
        menu.add_item(MenuItem(
            "4", "Set Output Directory",
            action=lambda: self.set_custom_path("output"),
            description="Set default output directory"
        ))
        
        menu.add_item(MenuItem(
            "5", "View Current Paths",
            action=self.view_current_paths,
            description="Display all configured paths"
        ))
        
        menu.add_item(MenuItem(
            "6", "Reset to Defaults",
            action=self.reset_paths_to_default,
            description="Reset all paths to defaults"
        ))
        
        return menu
        
    def run(self):
        """Run the menu system"""
        ConsoleOutput.header("Welcome to LlamaNote Enhanced")
        result = self.main_menu.display()
        
        if result == MenuAction.EXIT:
            ConsoleOutput.info("Goodbye!")
            sys.exit(0)
            
    # Menu Actions
    
    def process_pdf_menu(self):
        """Process PDF menu"""
        ConsoleOutput.section("Process PDF")
        
        # Get input file
        input_path = input("Enter PDF file path: ").strip()
        if not input_path:
            return
            
        input_path = Path(input_path)
        if not input_path.exists():
            ConsoleOutput.error("File not found")
            input("Press Enter to continue...")
            return
            
        # Select processing mode
        print("\nSelect processing mode:")
        modes = list(ProcessingMode)
        for i, mode in enumerate(modes, 1):
            print(f"  {i}. {mode.value.title()}")
            
        mode_choice = input("Mode (1-{}, default=1): ".format(len(modes))).strip()
        mode = modes[int(mode_choice) - 1] if mode_choice.isdigit() else ProcessingMode.PODCAST
        
        # Configure options
        config = PipelineConfig(mode=mode)
        
        # Quick settings
        print("\nQuick settings:")
        print("  1. Use current settings")
        print("  2. Configure options")
        
        if input("Choice (1-2, default=1): ").strip() == "2":
            config = self._configure_pipeline()
            
        # Process file
        try:
            pipeline = ProcessingPipeline(config)
            result = pipeline.process_file(input_path)
            
            if result.success:
                ConsoleOutput.success(f"Processing complete!")
                ConsoleOutput.info(f"Output: {result.output_file}")
                ConsoleOutput.info(f"Time: {result.processing_time:.2f}s")
                
                # Ask about audio generation
                if input("\nGenerate audio? (y/n): ").strip().lower() == 'y':
                    self.generate_audio_from_file(result.output_file)
            else:
                ConsoleOutput.error(f"Processing failed: {result.error_message}")
                
        except Exception as e:
            ConsoleOutput.error(f"Error: {e}")
            
        input("\nPress Enter to continue...")
        
    def audio_generation_menu(self):
        """Audio generation menu"""
        ConsoleOutput.section("Generate Audio")
        
        # Get input
        print("Select input source:")
        print("  1. Text file")
        print("  2. Enter text directly")
        print("  3. Use last processed file")
        
        choice = input("Choice (1-3): ").strip()
        
        if choice == "1":
            file_path = input("Enter text file path: ").strip()
            if not file_path:
                return
                
            file_path = Path(file_path)
            if not file_path.exists():
                ConsoleOutput.error("File not found")
                input("Press Enter to continue...")
                return
                
            with open(file_path, 'r') as f:
                text = f.read()
                
        elif choice == "2":
            print("Enter text (end with empty line):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            text = "\n".join(lines)
            
        elif choice == "3":
            # Find last output file
            output_dir = Path("output")
            if output_dir.exists():
                files = sorted(output_dir.glob("*.md"), key=lambda p: p.stat().st_mtime)
                if files:
                    file_path = files[-1]
                    with open(file_path, 'r') as f:
                        text = f.read()
                    ConsoleOutput.info(f"Using: {file_path}")
                else:
                    ConsoleOutput.error("No processed files found")
                    return
            else:
                ConsoleOutput.error("No output directory")
                return
        else:
            return
            
        # Generate audio
        self._generate_audio(text)
        input("\nPress Enter to continue...")
        
    def search_models_menu(self):
        """Search TTS models menu"""
        ConsoleOutput.section("Search TTS Models")
        
        print("Search options:")
        print("  1. Popular models")
        print("  2. Search with filters")
        print("  3. Search by language")
        print("  4. Use filter preset")
        
        choice = input("Choice (1-4): ").strip()
        
        if choice == "1":
            models = self.hf_search.get_popular_tts_models(20)
            
        elif choice == "2":
            builder = FilterBuilder()
            filters = builder.interactive_build()
            models = self.hf_search.search_models(filters)
            
            # Ask to save filter
            if input("\nSave filter preset? (y/n): ").strip().lower() == 'y':
                name = input("Preset name: ").strip()
                if name:
                    builder.save_filters(name, filters)
                    ConsoleOutput.success(f"Saved filter preset: {name}")
                    
        elif choice == "3":
            language = input("Language code (e.g., en, es, fr): ").strip()
            models = self.hf_search.search_by_language(language)
            
        elif choice == "4":
            presets = self.config_manager.list_configs("filter")
            if not presets:
                ConsoleOutput.warning("No filter presets found")
                return
                
            print("\nAvailable presets:")
            for i, preset in enumerate(presets, 1):
                print(f"  {i}. {preset}")
                
            idx = input("Select preset: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(presets):
                builder = FilterBuilder()
                filters = builder.load_filters(presets[int(idx) - 1])
                if filters:
                    models = self.hf_search.search_models(filters)
                else:
                    ConsoleOutput.error("Failed to load preset")
                    return
            else:
                return
        else:
            return
            
        # Display results
        if models:
            ConsoleOutput.info(f"\nFound {len(models)} models:")
            
            for i, model in enumerate(models[:50], 1):  # Limit display
                print(f"{i:3}. {model.model_id}")
                print(f"     Downloads: {model.downloads:,} | Likes: {model.likes}")
                if model.language:
                    print(f"     Languages: {', '.join(model.language[:3])}")
                    
            # Model selection
            if input("\nSelect a model? (y/n): ").strip().lower() == 'y':
                idx = input("Model number: ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(models):
                    selected = models[int(idx) - 1]
                    self._handle_model_selection(selected)
        else:
            ConsoleOutput.warning("No models found")
            
        input("\nPress Enter to continue...")
        
    def batch_processing_menu(self):
        """Batch processing menu"""
        ConsoleOutput.section("Batch Processing")
        
        # Get input source
        print("Select input source:")
        print("  1. Directory")
        print("  2. File list")
        print("  3. Pattern match")
        
        choice = input("Choice (1-3): ").strip()
        
        files = []
        
        if choice == "1":
            dir_path = input("Directory path: ").strip()
            if dir_path:
                dir_path = Path(dir_path)
                if dir_path.exists():
                    recursive = input("Search recursively? (y/n): ").strip().lower() == 'y'
                    pattern = input("File pattern (default=*.pdf): ").strip() or "*.pdf"
                    
                    if recursive:
                        files = list(dir_path.rglob(pattern))
                    else:
                        files = list(dir_path.glob(pattern))
                        
        elif choice == "2":
            print("Enter file paths (empty line to finish):")
            while True:
                path = input().strip()
                if not path:
                    break
                path = Path(path)
                if path.exists():
                    files.append(path)
                    
        elif choice == "3":
            pattern = input("Pattern (e.g., docs/*.pdf): ").strip()
            if pattern:
                files = list(Path().glob(pattern))
                
        if not files:
            ConsoleOutput.warning("No files found")
            input("Press Enter to continue...")
            return
            
        ConsoleOutput.info(f"Found {len(files)} files")
        
        # Configure processing
        config = self._configure_pipeline()
        
        # Process files
        try:
            pipeline = ProcessingPipeline(config)
            results = pipeline.process_batch(files)
            
            # Summary
            successful = sum(1 for r in results if r.success)
            ConsoleOutput.info(f"\nProcessed {successful}/{len(results)} files successfully")
            
            # Generate audio for all?
            if successful > 0 and input("\nGenerate audio for all? (y/n): ").strip().lower() == 'y':
                for result in results:
                    if result.success and result.output_file:
                        ConsoleOutput.info(f"Generating audio for: {result.output_file.name}")
                        self.generate_audio_from_file(result.output_file)
                        
        except Exception as e:
            ConsoleOutput.error(f"Batch processing failed: {e}")
            
        input("\nPress Enter to continue...")
        
    def load_configuration(self):
        """Load a saved configuration"""
        configs = self.config_manager.list_configs("pipeline")
        
        if not configs:
            ConsoleOutput.warning("No saved configurations")
            input("Press Enter to continue...")
            return
            
        print("\nAvailable configurations:")
        for i, config in enumerate(configs, 1):
            print(f"  {i}. {config}")
            
        choice = input("Select configuration: ").strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(configs):
            config_data = self.config_manager.load_pipeline_config(configs[int(choice) - 1])
            if config_data:
                self.current_config = config_data
                ConsoleOutput.success("Configuration loaded")
            else:
                ConsoleOutput.error("Failed to load configuration")
                
        input("Press Enter to continue...")
        
    def save_configuration(self):
        """Save current configuration"""
        name = input("Configuration name: ").strip()
        
        if name:
            if self.config_manager.save_pipeline_config(name, self.current_config):
                ConsoleOutput.success(f"Configuration saved: {name}")
            else:
                ConsoleOutput.error("Failed to save configuration")
                
        input("Press Enter to continue...")
        
    def list_configurations(self):
        """List all configurations"""
        ConsoleOutput.section("Saved Configurations")
        
        for config_type in ["pipeline", "audio", "filter", "model"]:
            configs = self.config_manager.list_configs(config_type)
            if configs:
                print(f"\n{config_type.title()} Configurations:")
                for config in configs:
                    print(f"  - {config}")
                    
        input("\nPress Enter to continue...")
        
    def delete_configuration(self):
        """Delete a configuration"""
        print("Select configuration type:")
        print("  1. Pipeline")
        print("  2. Audio")
        print("  3. Filter")
        print("  4. Model")
        
        type_choice = input("Type (1-4): ").strip()
        type_map = {"1": "pipeline", "2": "audio", "3": "filter", "4": "model"}
        
        if type_choice in type_map:
            config_type = type_map[type_choice]
            configs = self.config_manager.list_configs(config_type)
            
            if configs:
                print(f"\n{config_type.title()} configurations:")
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
                    
                choice = input("Select to delete: ").strip()
                
                if choice.isdigit() and 1 <= int(choice) <= len(configs):
                    if self.config_manager.delete_config(configs[int(choice) - 1], config_type):
                        ConsoleOutput.success("Configuration deleted")
                    else:
                        ConsoleOutput.error("Failed to delete")
            else:
                ConsoleOutput.warning(f"No {config_type} configurations found")
                
        input("\nPress Enter to continue...")
        
    def import_export_menu(self):
        """Import/export configurations"""
        print("Select action:")
        print("  1. Export configuration")
        print("  2. Import configuration")
        
        choice = input("Choice (1-2): ").strip()
        
        if choice == "1":
            # Export
            configs = self.config_manager.list_configs()
            if configs:
                print("\nConfigurations:")
                for i, config in enumerate(configs, 1):
                    print(f"  {i}. {config}")
                    
                idx = input("Select to export: ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(configs):
                    export_path = input("Export path: ").strip()
                    if export_path:
                        if self.config_manager.export_config(
                            configs[int(idx) - 1], Path(export_path)
                        ):
                            ConsoleOutput.success("Configuration exported")
                        else:
                            ConsoleOutput.error("Export failed")
                            
        elif choice == "2":
            # Import
            import_path = input("Import file path: ").strip()
            if import_path:
                import_path = Path(import_path)
                if import_path.exists():
                    name = input("Configuration name (optional): ").strip()
                    if self.config_manager.import_config(import_path, name):
                        ConsoleOutput.success("Configuration imported")
                    else:
                        ConsoleOutput.error("Import failed")
                else:
                    ConsoleOutput.error("File not found")
                    
        input("\nPress Enter to continue...")
        
    def manage_filters_menu(self):
        """Manage search filters"""
        ConsoleOutput.section("Manage Search Filters")
        
        print("Options:")
        print("  1. Create new filter")
        print("  2. List filters")
        print("  3. Test filter")
        print("  4. Delete filter")
        
        choice = input("Choice (1-4): ").strip()
        
        if choice == "1":
            builder = FilterBuilder()
            filters = builder.interactive_build()
            name = input("\nFilter name: ").strip()
            if name:
                builder.save_filters(name, filters)
                ConsoleOutput.success(f"Filter saved: {name}")
                
        elif choice == "2":
            filters = self.config_manager.list_configs("filter")
            if filters:
                print("\nSaved filters:")
                for f in filters:
                    print(f"  - {f}")
            else:
                ConsoleOutput.warning("No saved filters")
                
        elif choice == "3":
            filters = self.config_manager.list_configs("filter")
            if filters:
                print("\nFilters:")
                for i, f in enumerate(filters, 1):
                    print(f"  {i}. {f}")
                    
                idx = input("Select filter: ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(filters):
                    builder = FilterBuilder()
                    filter_obj = builder.load_filters(filters[int(idx) - 1])
                    if filter_obj:
                        models = self.hf_search.search_models(filter_obj)
                        ConsoleOutput.info(f"Found {len(models)} models")
                        
        elif choice == "4":
            filters = self.config_manager.list_configs("filter")
            if filters:
                print("\nFilters:")
                for i, f in enumerate(filters, 1):
                    print(f"  {i}. {f}")
                    
                idx = input("Select to delete: ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(filters):
                    if self.config_manager.delete_config(filters[int(idx) - 1], "filter"):
                        ConsoleOutput.success("Filter deleted")
                        
        input("\nPress Enter to continue...")
        
    def backup_restore_menu(self):
        """Backup/restore menu"""
        print("Select action:")
        print("  1. Create backup")
        print("  2. Restore backup")
        print("  3. List backups")
        
        choice = input("Choice (1-3): ").strip()
        
        if choice == "1":
            name = input("Backup name (optional): ").strip()
            backup_dir = self.config_manager.backup_configs(name)
            if backup_dir:
                ConsoleOutput.success(f"Backup created: {backup_dir}")
            else:
                ConsoleOutput.error("Backup failed")
                
        elif choice == "2":
            backups = self.config_manager.list_backups()
            if backups:
                print("\nAvailable backups:")
                for i, backup in enumerate(backups, 1):
                    print(f"  {i}. {backup}")
                    
                idx = input("Select backup: ").strip()
                if idx.isdigit() and 1 <= int(idx) <= len(backups):
                    if self.config_manager.restore_backup(backups[int(idx) - 1]):
                        ConsoleOutput.success("Backup restored")
                    else:
                        ConsoleOutput.error("Restore failed")
            else:
                ConsoleOutput.warning("No backups found")
                
        elif choice == "3":
            backups = self.config_manager.list_backups()
            if backups:
                print("\nBackups:")
                for backup in backups:
                    print(f"  - {backup}")
            else:
                ConsoleOutput.warning("No backups found")
                
        input("\nPress Enter to continue...")
        
    def processing_settings(self):
        """Configure processing settings"""
        ConsoleOutput.section("Processing Settings")
        
        print("Current settings:")
        print(f"  Mode: {self.current_config.get('mode', 'podcast')}")
        print(f"  Model: {self.current_config.get('model', 'qwen3-4b')}")
        print(f"  Chunk size: {self.current_config.get('chunk_size', 1000)}")
        print(f"  Output format: {self.current_config.get('output_format', 'markdown')}")
        
        if input("\nModify settings? (y/n): ").strip().lower() == 'y':
            # Mode
            modes = ["podcast", "technical", "narrative", "summary", "custom"]
            print("\nProcessing modes:")
            for i, mode in enumerate(modes, 1):
                print(f"  {i}. {mode}")
            choice = input("Mode: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(modes):
                self.current_config['mode'] = modes[int(choice) - 1]
                
            # Model
            from config import MODELS
            models = list(MODELS.keys())
            print("\nModels:")
            for i, model in enumerate(models, 1):
                print(f"  {i}. {model}")
            choice = input("Model: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(models):
                self.current_config['model'] = models[int(choice) - 1]
                
            # Chunk size
            size = input("\nChunk size (100-5000): ").strip()
            if size.isdigit():
                self.current_config['chunk_size'] = int(size)
                
            # Output format
            formats = ["markdown", "text", "json", "html"]
            print("\nOutput formats:")
            for i, fmt in enumerate(formats, 1):
                print(f"  {i}. {fmt}")
            choice = input("Format: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(formats):
                self.current_config['output_format'] = formats[int(choice) - 1]
                
            self.config_manager.set_default_config(self.current_config)
            ConsoleOutput.success("Settings updated")
            
        input("\nPress Enter to continue...")
        
    def audio_settings(self):
        """Configure audio settings"""
        ConsoleOutput.section("Audio Settings")
        
        audio_config = self.current_config.get('audio', {})
        
        print("Current audio settings:")
        print(f"  Model: {audio_config.get('model_id', 'microsoft/speecht5_tts')}")
        print(f"  Sample rate: {audio_config.get('sample_rate', 16000)}")
        print(f"  Format: {audio_config.get('output_format', 'wav')}")
        print(f"  Speed: {audio_config.get('speed', 1.0)}")
        
        if input("\nModify settings? (y/n): ").strip().lower() == 'y':
            # Model selection
            print("\nCommon TTS models:")
            print("  1. microsoft/speecht5_tts (fast, good quality)")
            print("  2. suno/bark (expressive, slower)")
            print("  3. facebook/mms-tts-eng (multilingual)")
            print("  4. Custom model ID")
            
            choice = input("Choice (1-4): ").strip()
            if choice == "1":
                audio_config['model_id'] = "microsoft/speecht5_tts"
            elif choice == "2":
                audio_config['model_id'] = "suno/bark"
            elif choice == "3":
                audio_config['model_id'] = "facebook/mms-tts-eng"
            elif choice == "4":
                model_id = input("Model ID: ").strip()
                if model_id:
                    audio_config['model_id'] = model_id
                    
            # Sample rate
            rate = input("\nSample rate (8000/16000/22050/44100): ").strip()
            if rate.isdigit():
                audio_config['sample_rate'] = int(rate)
                
            # Output format
            fmt = input("Format (wav/mp3/flac): ").strip().lower()
            if fmt in ["wav", "mp3", "flac"]:
                audio_config['output_format'] = fmt
                
            # Speed
            speed = input("Speed (0.5-2.0): ").strip()
            try:
                audio_config['speed'] = float(speed)
            except:
                pass
                
            self.current_config['audio'] = audio_config
            self.config_manager.set_default_config(self.current_config)
            ConsoleOutput.success("Audio settings updated")
            
        input("\nPress Enter to continue...")
        
    def model_settings(self):
        """Configure model settings"""
        ConsoleOutput.section("Model Settings")
        
        from config import MEMORY_PROFILES
        
        print("Memory profiles:")
        for name, profile in MEMORY_PROFILES.items():
            print(f"\n{name}:")
            print(f"  Quantization: {profile.quantization_type}")
            print(f"  Max GPU: {profile.max_gpu_memory}")
            print(f"  Max CPU: {profile.max_cpu_memory}")
            
        profile = input("\nSelect profile: ").strip()
        if profile in MEMORY_PROFILES:
            self.current_config['memory_profile'] = profile
            self.config_manager.set_default_config(self.current_config)
            ConsoleOutput.success(f"Using {profile} profile")
            
        input("\nPress Enter to continue...")
        
    def set_custom_path(self, path_type: str):
        """Set a custom path"""
        current = self.config_manager.get_dir(path_type)
        print(f"Current {path_type} path: {current}")
        
        new_path = input("New path (or Enter to keep current): ").strip()
        if new_path:
            new_path = Path(new_path)
            new_path.mkdir(parents=True, exist_ok=True)
            self.config_manager.set_custom_path(path_type, new_path)
            ConsoleOutput.success(f"Path updated: {new_path}")
            
        input("\nPress Enter to continue...")
        
    def view_current_paths(self):
        """View all configured paths"""
        ConsoleOutput.section("Configured Paths")
        
        print(f"Base directory: {self.config_manager.base_dir}")
        print(f"Config: {self.config_manager.get_dir('config')}")
        print(f"Filter: {self.config_manager.get_dir('filter')}")
        print(f"Audio: {self.config_manager.get_dir('audio')}")
        print(f"Model: {self.config_manager.get_dir('model')}")
        print(f"Pipeline: {self.config_manager.get_dir('pipeline')}")
        
        if self.config_manager.custom_paths:
            print("\nCustom paths:")
            for name, path in self.config_manager.custom_paths.items():
                print(f"  {name}: {path}")
                
        input("\nPress Enter to continue...")
        
    def reset_paths_to_default(self):
        """Reset all paths to defaults"""
        if input("Reset all paths to defaults? (y/n): ").strip().lower() == 'y':
            self.config_manager.custom_paths = {}
            self.config_manager._save_custom_paths()
            ConsoleOutput.success("Paths reset to defaults")
            
        input("\nPress Enter to continue...")
        
    def view_current_settings(self):
        """View all current settings"""
        ConsoleOutput.section("Current Settings")
        
        print(json.dumps(self.current_config, indent=2))
        
        input("\nPress Enter to continue...")
        
    # Helper methods
    
    def _configure_pipeline(self) -> PipelineConfig:
        """Configure pipeline interactively"""
        config = PipelineConfig()
        
        # Mode
        modes = list(ProcessingMode)
        print("\nProcessing mode:")
        for i, mode in enumerate(modes, 1):
            print(f"  {i}. {mode.value}")
        choice = input("Mode (default=1): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(modes):
            config.mode = modes[int(choice) - 1]
            
        # Model
        from config import MODELS
        model_names = list(MODELS.keys())
        print("\nModel:")
        for i, name in enumerate(model_names, 1):
            print(f"  {i}. {name}")
        choice = input("Model (default=1): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(model_names):
            config.model_name = model_names[int(choice) - 1]
            
        # Chunk size
        size = input("\nChunk size (default=1000): ").strip()
        if size.isdigit():
            config.chunk_size = int(size)
            
        # Chunking strategy
        strategies = list(ChunkingStrategy)
        print("\nChunking strategy:")
        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy.value}")
        choice = input("Strategy (default=1): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(strategies):
            config.chunking_strategy = strategies[int(choice) - 1]
            
        # Other options
        config.remove_thinking = input("\nRemove thinking tokens? (y/n, default=y): ").strip().lower() != 'n'
        config.add_emotions = input("Add emotion markers? (y/n, default=y): ").strip().lower() != 'n'
        config.clean_for_audio = input("Clean for audio? (y/n, default=y): ").strip().lower() != 'n'
        
        return config
        
    def _generate_audio(self, text: str):
        """Generate audio from text"""
        # Initialize generator if needed
        if not self.audio_generator:
            audio_config = AudioConfig()
            
            # Get audio settings
            audio_settings = self.current_config.get('audio', {})
            audio_config.model_id = audio_settings.get('model_id', audio_config.model_id)
            audio_config.sample_rate = audio_settings.get('sample_rate', audio_config.sample_rate)
            audio_config.output_format = audio_settings.get('output_format', audio_config.output_format)
            
            self.audio_generator = AudioGenerator(audio_config)
            
        # Load model if needed
        if not self.audio_generator.model and not self.audio_generator.pipeline:
            ConsoleOutput.info(f"Loading TTS model: {self.audio_generator.config.model_id}")
            if not self.audio_generator.load_model():
                ConsoleOutput.error("Failed to load TTS model")
                return
                
        # Generate audio
        output_path = input("Output audio file (default=output.wav): ").strip()
        if not output_path:
            output_path = "output.wav"
            
        ConsoleOutput.info("Generating audio...")
        result = self.audio_generator.generate_audio(text, Path(output_path))
        
        if result:
            ConsoleOutput.success(f"Audio generated: {result.audio_path}")
            ConsoleOutput.info(f"Duration: {result.duration_formatted}")
            ConsoleOutput.info(f"Processing time: {result.processing_time:.2f}s")
        else:
            ConsoleOutput.error("Audio generation failed")
            
    def generate_audio_from_file(self, file_path: Path):
        """Generate audio from a file"""
        with open(file_path, 'r') as f:
            text = f.read()
            
        # Output path based on input
        audio_path = file_path.with_suffix('.wav')
        
        if not self.audio_generator:
            audio_config = AudioConfig()
            audio_settings = self.current_config.get('audio', {})
            audio_config.model_id = audio_settings.get('model_id', audio_config.model_id)
            self.audio_generator = AudioGenerator(audio_config)
            
        if not self.audio_generator.model and not self.audio_generator.pipeline:
            ConsoleOutput.info(f"Loading TTS model: {self.audio_generator.config.model_id}")
            if not self.audio_generator.load_model():
                ConsoleOutput.error("Failed to load TTS model")
                return
                
        result = self.audio_generator.generate_audio(text, audio_path)
        
        if result:
            ConsoleOutput.success(f"Audio saved: {result.audio_path}")
            
    def _handle_model_selection(self, model_info):
        """Handle selected model from search"""
        print(f"\nSelected: {model_info.model_id}")
        print(f"Downloads: {model_info.downloads:,}")
        print(f"Likes: {model_info.likes}")
        
        print("\nOptions:")
        print("  1. Use for audio generation")
        print("  2. Save as preference")
        print("  3. Download only")
        print("  4. Cancel")
        
        choice = input("Choice (1-4): ").strip()
        
        if choice == "1":
            # Set as current audio model
            if 'audio' not in self.current_config:
                self.current_config['audio'] = {}
            self.current_config['audio']['model_id'] = model_info.model_id
            ConsoleOutput.success(f"Set audio model: {model_info.model_id}")
            
        elif choice == "2":
            # Save preference
            name = input("Preference name: ").strip()
            if name:
                self.config_manager.save_model_preference(name, {
                    "model_id": model_info.model_id,
                    "task": model_info.task,
                    "info": model_info.__dict__
                })
                ConsoleOutput.success(f"Saved model preference: {name}")
                
        elif choice == "3":
            # Download
            from huggingface_search import ModelDownloader
            downloader = ModelDownloader()
            path = downloader.download_model(model_info.model_id)
            if path:
                ConsoleOutput.success(f"Downloaded to: {path}")
            else:
                ConsoleOutput.error("Download failed")
