# config_manager.py
import os
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigManager:
    def __init__(self):
        self.user_home = Path.home()
        self.default_config_dir = self.user_home / ".config" / "llamaNote" / "configs"
        self.default_config_dir.mkdir(parents=True, exist_ok=True)
        self.current_config = None
        
        # Default configuration
        self.default_config = {
            "preprocessing": {
                "model_type": "local",
                "model_name": "google/gemma-3-270m",
                "provider": "huggingface"
            },
            "transcription": {
                "model_type": "local", 
                "model_name": "Qwen/Qwen3-4B-Instruct-2507",
                "provider": "huggingface"
            },
            "enhancement": {
                "model_type": "local",
                "model_name": "Qwen/Qwen3-4B-Instruct-2507", 
                "provider": "huggingface"
            },
            "text_to_speech": {
                "model_type": "local",
                "model_name": "parler-tts/parler-tts-mini-v1",
                "provider": "huggingface"
            },
            "cloud_settings": {
                "openai_api_key": "",
                "anthropic_api_key": "",
                "google_api_key": "",
                "mistral_api_key": "",
                "deepseek_api_key": "",
                "openrouter_api_key": "",
                "qwen_api_key": ""
            }
        }
        
        # Available providers and models
        self.available_providers = {
            "local": {
                "huggingface": [
                    "google/gemma-3-270m",
                    "Qwen/Qwen3-4B-Instruct-2507", 
                    "Llama-3.2-1B-Instruct",
                    "mistralai/Mistral-7B-Instruct-v0.2",
                    "microsoft/DialoGPT-medium"
                ]
            },
            "cloud": {
                "openai": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "google": ["gemini-pro", "gemini-ultra"],
                "mistral": ["mistral-large", "mistral-medium", "mistral-small"],
                "deepseek": ["deepseek-chat"],
                "openrouter": [
                    "anthropic/claude-3-opus",
                    "openai/gpt-4-turbo", 
                    "google/gemini-pro"
                ],
                "qwen": ["qwen-plus", "qwen-turbo"]
            }
        }

    def get_config_path(self, config_name: str) -> Path:
        """Get full path for a config file"""
        return self.default_config_dir / f"{config_name}.json"

    def create_config(self, config_name: str, config_data: Dict[str, Any]) -> bool:
        """Create a new configuration file"""
        try:
            config_path = self.get_config_path(config_name)
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"✓ Configuration '{config_name}' created successfully!")
            return True
        except Exception as e:
            print(f"✗ Error creating configuration: {str(e)}")
            return False

    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load a configuration file"""
        try:
            config_path = self.get_config_path(config_name)
            if not config_path.exists():
                print(f"✗ Configuration '{config_name}' not found!")
                return None
                
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.current_config = config
            print(f"✓ Configuration '{config_name}' loaded successfully!")
            return config
        except Exception as e:
            print(f"✗ Error loading configuration: {str(e)}")
            return None

    def delete_config(self, config_name: str) -> bool:
        """Delete a configuration file"""
        try:
            config_path = self.get_config_path(config_name)
            if config_path.exists():
                config_path.unlink()
                print(f"✓ Configuration '{config_name}' deleted successfully!")
                return True
            else:
                print(f"✗ Configuration '{config_name}' not found!")
                return False
        except Exception as e:
            print(f"✗ Error deleting configuration: {str(e)}")
            return False

    def list_configs(self) -> list:
        """List all available configurations"""
        config_files = list(self.default_config_dir.glob("*.json"))
        return [f.stem for f in config_files]

    def update_config_dir(self, new_path: str) -> bool:
        """Update the default configuration directory"""
        try:
            new_dir = Path(new_path)
            new_dir.mkdir(parents=True, exist_ok=True)
            
            # Move existing configs if any
            for config_file in self.default_config_dir.glob("*.json"):
                config_file.rename(new_dir / config_file.name)
                
            self.default_config_dir = new_dir
            print(f"✓ Configuration directory updated to: {new_path}")
            return True
        except Exception as e:
            print(f"✗ Error updating configuration directory: {str(e)}")
            return False

    def get_model_config(self, step: str) -> Dict[str, Any]:
        """Get model configuration for a specific step"""
        if self.current_config and step in self.current_config:
            return self.current_config[step]
        return self.default_config[step]

    def interactive_config_creation(self) -> Dict[str, Any]:
        """Interactive configuration creation wizard"""
        print("\n" + "="*60)
        print("CONFIGURATION CREATION WIZARD")
        print("="*60)
        
        config = self.default_config.copy()
        
        # Configure each step
        steps = ["preprocessing", "transcription", "enhancement", "text_to_speech"]
        
        for step in steps:
            print(f"\n--- Configuring {step.upper()} Model ---")
            
            # Model type selection
            print("\nAvailable model types:")
            print("1. Local (HuggingFace)")
            print("2. Cloud (API-based)")
            model_type_choice = input("Select model type (1-2, default=1): ").strip() or "1"
            
            if model_type_choice == "2":
                config[step]["model_type"] = "cloud"
                self.configure_cloud_model(step, config)
            else:
                config[step]["model_type"] = "local"
                self.configure_local_model(step, config)
        
        # Configure cloud API keys
        self.configure_cloud_keys(config)
        
        return config

    def configure_local_model(self, step: str, config: Dict[str, Any]):
        """Configure local model settings"""
        print("\nAvailable local models:")
        models = self.available_providers["local"]["huggingface"]
        for i, model in enumerate(models, 1):
            print(f"{i}. {model}")
        
        try:
            choice = int(input(f"Select model (1-{len(models)}, default=1): ").strip() or "1")
            if 1 <= choice <= len(models):
                config[step]["model_name"] = models[choice-1]
                config[step]["provider"] = "huggingface"
            else:
                print("Invalid choice, using default.")
        except ValueError:
            print("Invalid input, using default.")

    def configure_cloud_model(self, step: str, config: Dict[str, Any]):
        """Configure cloud model settings"""
        print("\nAvailable cloud providers:")
        providers = list(self.available_providers["cloud"].keys())
        for i, provider in enumerate(providers, 1):
            print(f"{i}. {provider}")
        
        try:
            choice = int(input(f"Select provider (1-{len(providers)}): ").strip())
            if 1 <= choice <= len(providers):
                provider = providers[choice-1]
                config[step]["provider"] = provider
                
                # Model selection for provider
                models = self.available_providers["cloud"][provider]
                print(f"\nAvailable {provider} models:")
                for i, model in enumerate(models, 1):
                    print(f"{i}. {model}")
                
                model_choice = int(input(f"Select model (1-{len(models)}): ").strip())
                if 1 <= model_choice <= len(models):
                    config[step]["model_name"] = models[model_choice-1]
                else:
                    print("Invalid choice, using first model.")
                    config[step]["model_name"] = models[0]
            else:
                print("Invalid choice, using local model as fallback.")
                config[step]["model_type"] = "local"
                config[step]["provider"] = "huggingface"
                config[step]["model_name"] = self.default_config[step]["model_name"]
        except ValueError:
            print("Invalid input, using local model as fallback.")
            config[step]["model_type"] = "local"
            config[step]["provider"] = "huggingface"
            config[step]["model_name"] = self.default_config[step]["model_name"]

    def configure_cloud_keys(self, config: Dict[str, Any]):
        """Configure cloud API keys"""
        print("\n--- Cloud API Keys Configuration ---")
        print("Leave blank to skip or keep existing value.")
        
        for provider in self.available_providers["cloud"]:
            key_name = f"{provider}_api_key"
            current_value = config["cloud_settings"].get(key_name, "")
            masked_value = f"{current_value[:4]}...{current_value[-4:]}" if current_value else "Not set"
            
            new_value = input(f"{provider.upper()} API Key [{masked_value}]: ").strip()
            if new_value:
                config["cloud_settings"][key_name] = new_value
