#!/usr/bin/env python3
# src/io/model_abbreviations.py
"""
Model Name Abbreviation Manager
Provides configurable model name shortening for checkpoint filenames.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


class ModelAbbreviationManager:
    """
    Manages model name abbreviations for checkpoint filename generation.

    Supports:
    - User-configurable mappings from JSON file
    - Automatic fallback patterns for unknown models
    - Separate mappings for text and audio models
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the abbreviation manager.

        Args:
            config_path: Path to model_abbreviations.json file.
                        If None, uses default location in src/config/
        """
        if config_path is None:
            # Default to src/config/model_abbreviations.json
            config_path = Path(__file__).parent.parent / "config" / "model_abbreviations.json"

        self.config_path = Path(config_path)
        self.text_models: Dict[str, str] = {}
        self.audio_models: Dict[str, str] = {}

        # Load abbreviations from file
        self._load_abbreviations()

        logger.debug(f"Loaded {len(self.text_models)} text model abbreviations")
        logger.debug(f"Loaded {len(self.audio_models)} audio model abbreviations")

    def _load_abbreviations(self):
        """Load abbreviations from JSON file."""
        if not self.config_path.exists():
            logger.warning(f"Abbreviation config not found: {self.config_path}")
            logger.warning("Creating default abbreviation file")
            self._create_default_config()
            return

        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)

            self.text_models = data.get('text_models', {})
            self.audio_models = data.get('audio_models', {})

            logger.info(f"Loaded model abbreviations from {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to load abbreviations: {e}")
            # Use empty dicts as fallback
            self.text_models = {}
            self.audio_models = {}

    def _create_default_config(self):
        """Create default abbreviation configuration file."""
        default_config = {
            "text_models": {
                "meta-llama/llama-3.2-3b": "llama_3.2_3b",
                "Qwen/Qwen2.5-3B-Instruct": "qwen2.5_3b",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "deepseek_r1_1.5b"
            },
            "audio_models": {
                "microsoft/speecht5_tts": "ms_speecht5",
                "meta-llama/AudioLlama": "metallama_audiollama"
            }
        }

        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)

            self.text_models = default_config['text_models']
            self.audio_models = default_config['audio_models']

            logger.info(f"Created default abbreviation config at {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to create default config: {e}")

    def abbreviate_text_model(self, full_name: str) -> str:
        """
        Abbreviate a text model name.

        Args:
            full_name: Full model name (e.g., "meta-llama/llama-3.2-3b")

        Returns:
            Abbreviated name (e.g., "llama_3.2_3b")
        """
        # Check if we have a configured abbreviation
        if full_name in self.text_models:
            return self.text_models[full_name]

        # Apply automatic fallback pattern
        abbreviated = self._apply_fallback_pattern(full_name)
        logger.debug(f"Auto-abbreviated text model: {full_name} -> {abbreviated}")
        return abbreviated

    def abbreviate_audio_model(self, full_name: str) -> str:
        """
        Abbreviate an audio model name.

        Args:
            full_name: Full model name (e.g., "microsoft/speecht5_tts")

        Returns:
            Abbreviated name (e.g., "ms_speecht5")
        """
        # Check if we have a configured abbreviation
        if full_name in self.audio_models:
            return self.audio_models[full_name]

        # Apply automatic fallback pattern
        abbreviated = self._apply_fallback_pattern(full_name)
        logger.debug(f"Auto-abbreviated audio model: {full_name} -> {abbreviated}")
        return abbreviated

    def _apply_fallback_pattern(self, model_name: str) -> str:
        """
        Apply automatic abbreviation pattern for unknown models.

        Rules:
        1. Remove common organization prefixes (meta-llama/, microsoft/, Qwen/, etc.)
        2. Replace hyphens and dots with underscores
        3. Remove common suffixes (-Instruct, -Chat, -v0.2, etc.)
        4. Lowercase everything
        5. Truncate if longer than 30 characters

        Args:
            model_name: Full model name

        Returns:
            Abbreviated model name
        """
        # Remove organization prefix
        name = model_name
        prefixes = [
            'meta-llama/', 'microsoft/', 'Qwen/', 'deepseek-ai/', 'google/',
            'mistralai/', 'facebook/', 'suno/', 'coqui/', 'huggingface/',
            'openai/', 'anthropic/', 'EleutherAI/'
        ]
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        # Replace hyphens and dots with underscores
        name = name.replace('-', '_').replace('.', '_')

        # Remove common suffixes (case-insensitive)
        suffixes = ['_Instruct', '_Chat', '_Base', '_v0_1', '_v0_2', '_v1', '_v2', '_tts', '_stt']
        for suffix in suffixes:
            if name.lower().endswith(suffix.lower()):
                name = name[:-(len(suffix))]
                break

        # Lowercase
        name = name.lower()

        # Remove double underscores
        while '__' in name:
            name = name.replace('__', '_')

        # Truncate if too long
        if len(name) > 30:
            # Try to keep important parts (model family + size)
            # E.g., "very_long_model_name_3b" -> "very_long_model_3b"
            parts = name.split('_')
            if len(parts) > 2:
                # Keep first part and last 2 parts
                name = '_'.join([parts[0]] + parts[-2:])

            # If still too long, just truncate
            if len(name) > 30:
                name = name[:30]

        return name

    def add_abbreviation(self, model_type: str, full_name: str, abbrev: str):
        """
        Add a new abbreviation to the configuration.

        Args:
            model_type: "text" or "audio"
            full_name: Full model name
            abbrev: Abbreviated name

        Raises:
            ValueError: If model_type is invalid
        """
        if model_type == "text":
            self.text_models[full_name] = abbrev
        elif model_type == "audio":
            self.audio_models[full_name] = abbrev
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'text' or 'audio'")

        logger.info(f"Added {model_type} model abbreviation: {full_name} -> {abbrev}")

    def save_abbreviations(self):
        """Save current abbreviations to JSON file."""
        try:
            data = {
                "text_models": self.text_models,
                "audio_models": self.audio_models
            }

            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved abbreviations to {self.config_path}")

        except Exception as e:
            logger.error(f"Failed to save abbreviations: {e}")
            raise

    def get_all_abbreviations(self) -> Dict[str, Dict[str, str]]:
        """
        Get all abbreviations.

        Returns:
            Dictionary with 'text_models' and 'audio_models' keys
        """
        return {
            "text_models": self.text_models.copy(),
            "audio_models": self.audio_models.copy()
        }
