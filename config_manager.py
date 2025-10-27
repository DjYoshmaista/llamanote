"""
Configuration Manager Module
Manages configuration files, filters, and settings
"""

import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import shutil

from logging_config import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """Manages application configurations and settings"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize configuration manager
        
        Args:
            base_dir: Base directory for configurations
        """
        self.base_dir = base_dir or Path.home() / ".config" / "llamanote"
        
        # Setup directory structure
        self.config_dir = self.base_dir / "config"
        self.filter_dir = self.base_dir / "filter"
        self.preset_dir = self.base_dir / "preset"
        self.pipeline_dir = self.base_dir / "pipeline"
        self.audio_dir = self.base_dir / "audio"
        self.model_dir = self.base_dir / "model"
        
        # Create directories
        for dir_path in [self.config_dir, self.filter_dir, self.preset_dir,
                         self.pipeline_dir, self.audio_dir, self.model_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Default config file
        self.default_config_file = self.config_dir / "default.json"
        
        # Custom paths configuration
        self.paths_config_file = self.config_dir / "paths.json"
        self.custom_paths = self._load_custom_paths()
        
        logger.info(f"Initialized ConfigManager at {self.base_dir}")
        
    def _load_custom_paths(self) -> Dict[str, Path]:
        """Load custom path configurations"""
        if self.paths_config_file.exists():
            try:
                with open(self.paths_config_file, 'r') as f:
                    paths = json.load(f)
                return {k: Path(v) for k, v in paths.items()}
            except Exception as e:
                logger.error(f"Failed to load custom paths: {e}")
                
        return {}
        
    def set_custom_path(self, name: str, path: Path):
        """Set a custom path"""
        self.custom_paths[name] = Path(path)
        self._save_custom_paths()
        
    def _save_custom_paths(self):
        """Save custom path configurations"""
        try:
            paths = {k: str(v) for k, v in self.custom_paths.items()}
            with open(self.paths_config_file, 'w') as f:
                json.dump(paths, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save custom paths: {e}")
            
    def get_dir(self, dir_type: str) -> Path:
        """Get directory path (custom or default)"""
        if dir_type in self.custom_paths:
            return self.custom_paths[dir_type]
            
        dir_map = {
            "config": self.config_dir,
            "filter": self.filter_dir,
            "preset": self.preset_dir,
            "pipeline": self.pipeline_dir,
            "audio": self.audio_dir,
            "model": self.model_dir
        }
        
        return dir_map.get(dir_type, self.base_dir / dir_type)
        
    def save_config(self, name: str, config: Dict[str, Any], 
                   dir_type: str = "config") -> bool:
        """
        Save configuration to file
        
        Args:
            name: Configuration name
            config: Configuration dictionary
            dir_type: Directory type
            
        Returns:
            Success status
        """
        try:
            config_dir = self.get_dir(dir_type)
            config_file = config_dir / f"{name}.json"
            
            # Add metadata
            config["_metadata"] = {
                "created": datetime.now().isoformat(),
                "version": "1.0",
                "type": dir_type
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Saved {dir_type} configuration: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
            
    def load_config(self, name: str, dir_type: str = "config") -> Optional[Dict[str, Any]]:
        """
        Load configuration from file
        
        Args:
            name: Configuration name
            dir_type: Directory type
            
        Returns:
            Configuration dictionary or None
        """
        try:
            config_dir = self.get_dir(dir_type)
            config_file = config_dir / f"{name}.json"
            
            if not config_file.exists():
                logger.warning(f"Configuration not found: {name}")
                return None
                
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Remove metadata for return
            config.pop("_metadata", None)
            
            logger.info(f"Loaded {dir_type} configuration: {name}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return None
            
    def list_configs(self, dir_type: str = "config") -> List[str]:
        """List available configurations"""
        config_dir = self.get_dir(dir_type)
        configs = []
        
        for file in config_dir.glob("*.json"):
            configs.append(file.stem)
            
        return sorted(configs)
        
    def delete_config(self, name: str, dir_type: str = "config") -> bool:
        """Delete a configuration"""
        try:
            config_dir = self.get_dir(dir_type)
            config_file = config_dir / f"{name}.json"
            
            if config_file.exists():
                config_file.unlink()
                logger.info(f"Deleted {dir_type} configuration: {name}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete configuration: {e}")
            return False
            
    def export_config(self, name: str, export_path: Path, 
                     dir_type: str = "config") -> bool:
        """Export configuration to external file"""
        try:
            config = self.load_config(name, dir_type)
            if not config:
                return False
                
            with open(export_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Exported {name} to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False
            
    def import_config(self, import_path: Path, name: Optional[str] = None,
                     dir_type: str = "config") -> bool:
        """Import configuration from external file"""
        try:
            with open(import_path, 'r') as f:
                config = json.load(f)
                
            name = name or import_path.stem
            return self.save_config(name, config, dir_type)
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return False
            
    # Specialized methods for different configuration types
    
    def save_pipeline_config(self, name: str, config: Dict[str, Any]) -> bool:
        """Save pipeline configuration"""
        return self.save_config(name, config, "pipeline")
        
    def load_pipeline_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load pipeline configuration"""
        return self.load_config(name, "pipeline")
        
    def save_audio_config(self, name: str, config: Dict[str, Any]) -> bool:
        """Save audio configuration"""
        return self.save_config(name, config, "audio")
        
    def load_audio_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Load audio configuration"""
        return self.load_config(name, "audio")
        
    def save_filter_preset(self, name: str, filters: Dict[str, Any]) -> bool:
        """Save filter preset"""
        return self.save_config(name, filters, "filter")
        
    def load_filter_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Load filter preset"""
        return self.load_config(name, "filter")
        
    def save_model_preference(self, name: str, model_info: Dict[str, Any]) -> bool:
        """Save model preference"""
        return self.save_config(name, model_info, "model")
        
    def load_model_preference(self, name: str) -> Optional[Dict[str, Any]]:
        """Load model preference"""
        return self.load_config(name, "model")
        
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        if self.default_config_file.exists():
            return self.load_config("default", "config") or {}
            
        # Return hardcoded defaults
        return {
            "mode": "podcast",
            "model": "qwen3-4b",
            "memory_profile": "medium_vram",
            "chunk_size": 1000,
            "output_format": "markdown",
            "audio_model": "microsoft/speecht5_tts",
            "sample_rate": 16000
        }
        
    def set_default_config(self, config: Dict[str, Any]) -> bool:
        """Set default configuration"""
        return self.save_config("default", config, "config")
        
    def backup_configs(self, backup_name: Optional[str] = None) -> Optional[Path]:
        """Backup all configurations"""
        try:
            backup_name = backup_name or datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.base_dir / "backups" / backup_name
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all config directories
            for dir_name in ["config", "filter", "preset", "pipeline", "audio", "model"]:
                src = self.get_dir(dir_name)
                if src.exists():
                    dst = backup_dir / dir_name
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    
            logger.info(f"Created backup: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
            
    def restore_backup(self, backup_name: str) -> bool:
        """Restore configurations from backup"""
        try:
            backup_dir = self.base_dir / "backups" / backup_name
            
            if not backup_dir.exists():
                logger.error(f"Backup not found: {backup_name}")
                return False
                
            # Restore each directory
            for dir_name in ["config", "filter", "preset", "pipeline", "audio", "model"]:
                src = backup_dir / dir_name
                if src.exists():
                    dst = self.get_dir(dir_name)
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    
            logger.info(f"Restored backup: {backup_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
            
    def list_backups(self) -> List[str]:
        """List available backups"""
        backup_base = self.base_dir / "backups"
        
        if not backup_base.exists():
            return []
            
        backups = []
        for backup_dir in backup_base.iterdir():
            if backup_dir.is_dir():
                backups.append(backup_dir.name)
                
        return sorted(backups, reverse=True)
