# llamanote/config/presets.py
"""
Hyperparameter Preset Management for LlamaNote Enhanced
Loads predefined hyperparameter configurations from a JSON file.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from ..models.hyperparameters import HyperparameterConfig

from ..utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

# Location of the presets file (relative to this file)
PRESETS_FILE = Path(__file__).parent / "hyperparameter_presets.json"

# Original preprocessing prompt (kept as is)
PREPROCESS_PROMPT = """
You are a world class text pre-processor, here is the raw data from a PDF. Please parse and return it in a way that is crispy and usable to send to a podcast writer.
The raw data is riddled with new line breaks, LaTeX math, and you will see fluff that you should remove completely. Remove, or alternatively translate, any details or data that would be lost, useless, misunderstood, or simply lost in translation from a pure text and raw data format to the audio podcast format.
Remember, the podcast could be on any one topic, or even on a myriad of topics, so the issues listed above are not necessarily exhaustive in scope.
Take care with what you remove, and do so intelligently, yet creatively please.
DO NOT START SUMMARIZING THIS. This should be a rule which is constantly and consistently at the forefront of your logic and processing as you preprocess the data into usable text. YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED.
Be very smart, yet aggressive, with removing details. You will get a running portion of the text and keep returning the processed text.
PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPITALIZATION LENDS ITSELF TO
ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions, period, end of discussion. Okay?

Here's the text:
"""

# --- Default Presets (Hardcoded fallback if file is missing/invalid) ---
DEFAULT_PRESETS = {
    "creative": {
        "description": "High creativity, diverse output (higher temperature).",
        "temperature": 0.9, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.1, "do_sample": True
    },
    "balanced": {
        "description": "Good balance of quality and diversity (default settings).",
        "temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.0, "do_sample": True
    },
    "precise": {
        "description": "Focused, more deterministic output (lower temperature).",
        "temperature": 0.3, "top_p": 0.85, "top_k": 20, "repetition_penalty": 1.0, "do_sample": True
    },
    "deterministic": {
        "description": "Most focused, greedy decoding (no randomness).",
        "temperature": 0.0, "do_sample": False, "top_k": 1, "repetition_penalty": 1.0
    },
    "long_form": {
        "description": "Optimized for generating longer text, reduces repetition.",
        "temperature": 0.75, "top_p": 0.9, "max_new_tokens": 4096,
        "repetition_penalty": 1.15, "no_repeat_ngram_size": 3, "do_sample": True
    },
}

def _load_presets_from_file() -> Dict[str, Dict]:
    """Loads presets from the JSON file, with fallback to defaults."""
    if not PRESETS_FILE.exists():
        logger.warning(f"Hyperparameter presets file not found: {PRESETS_FILE}. Using defaults.")
        # Optionally create the default file here
        # try:
        #     with open(PRESETS_FILE, 'w') as f:
        #         json.dump(DEFAULT_PRESETS, f, indent=2)
        #     logger.info(f"Created default presets file at {PRESETS_FILE}")
        # except Exception as e:
        #     logger.error(f"Could not create default presets file: {e}")
        return DEFAULT_PRESETS
    try:
        with open(PRESETS_FILE, 'r') as f:
            presets = json.load(f)
        # Basic validation: ensure it's a dict and values are dicts
        if not isinstance(presets, dict) or not all(isinstance(v, dict) for v in presets.values()):
             raise ValueError("Invalid format in presets file.")
        logger.info(f"Loaded {len(presets)} hyperparameter presets from {PRESETS_FILE}")
        return presets
    except (json.JSONDecodeError, ValueError, Exception) as e:
        logger.error(f"Failed to load or parse presets file {PRESETS_FILE}: {e}. Using defaults.")
        return DEFAULT_PRESETS

# Load presets when the module is imported
_PRESETS_DATA = _load_presets_from_file()

def get_hyperparameter_preset(name: str) -> Optional['HyperparameterConfig']:
    """
    Retrieves a HyperparameterConfig instance for a given preset name.

    Args:
        name: The name of the preset (case-insensitive).

    Returns:
        A HyperparameterConfig instance or None if the preset doesn't exist.
    """
    preset_data = _PRESETS_DATA.get(name.lower())
    if preset_data:
        try:
            # Local import to avoid circular dependency
            from ..models.hyperparameters import HyperparameterConfig
            # Create config, ignoring extra keys like 'description'
            config_params = {k: v for k, v in preset_data.items() if k != 'description'}
            return HyperparameterConfig(**config_params)
        except TypeError as e:
            logger.warning(f"Error creating HyperparameterConfig for preset '{name}': {e}. Preset data might be invalid.")
            return None # Return None if preset data doesn't match dataclass fields
    logger.warning(f"Hyperparameter preset '{name}' not found.")
    return None

def list_hyperparameter_presets() -> List[Tuple[str, str]]:
    """
    Returns a list of available preset names and their descriptions.

    Returns:
        List of tuples: [(preset_name, description), ...].
    """
    return [
        (name, data.get("description", "No description available."))
        for name, data in _PRESETS_DATA.items()
    ]

def add_hyperparameter_preset(name: str, config: 'HyperparameterConfig', description: str = "") -> bool:
    """
    Adds or updates a preset in memory and saves to the file. (Use with caution)

    Args:
        name: Name for the preset.
        config: The HyperparameterConfig to save.
        description: Optional description for the preset.

    Returns:
        True if saved successfully, False otherwise.
    """
    name_lower = name.lower()
    preset_data = config.to_dict() # Use the config's serialization method
    preset_data["description"] = description
    _PRESETS_DATA[name_lower] = preset_data

    # Save updated presets back to file
    try:
        with open(PRESETS_FILE, 'w') as f:
            json.dump(_PRESETS_DATA, f, indent=2, sort_keys=True)
        logger.info(f"Added/Updated hyperparameter preset '{name_lower}' and saved to file.")
        return True
    except Exception as e:
        logger.error(f"Failed to save updated presets file {PRESETS_FILE}: {e}")
        # Optionally revert the in-memory change:
        # _PRESETS_DATA.pop(name_lower, None) # Or reload from file
        return False
