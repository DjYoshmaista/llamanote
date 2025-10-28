# llamanote/models/hyperparameters.py
"""
Hyperparameter Management Module
Defines hyperparameter data structures and interactive editor.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import json
from pathlib import Path

from ..utils.logger import get_logger_conf, ConsoleOutput
from ..config.presets import get_hyperparameter_preset, list_hyperparameter_presets

logger = get_logger_conf(__name__)

class HyperparameterType(Enum):
    """Types of hyperparameters"""
    SAMPLING = "sampling"
    LENGTH = "length"
    PENALTY = "penalty"
    BEAM_SEARCH = "beam_search"
    TOKEN_CONTROL = "token_control"
    CACHE = "cache"
    ADVANCED = "advanced"


@dataclass
class HyperparameterDef:
    """Definition of a hyperparameter"""
    name: str
    description: str
    type: HyperparameterType
    default_value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    valid_values: Optional[List[Any]] = None
    mathematical_definition: Optional[str] = None

    def validate(self, value: Any) -> bool:
        """Validate a value for this hyperparameter"""
        if self.valid_values is not None:
            return value in self.valid_values
        
        if self.min_value is not None and isinstance(value, (int, float)):
            if value < self.min_value:
                return False
        
        if self.max_value is not None and isinstance(value, (int, float)):
            if value > self.max_value:
                return False
        
        return True

    def get_help_text(self) -> str:
        """Generates a detailed help string for this parameter."""
        help_text = [
            f"--- Help for: {self.name} ---",
            f"Description: {self.description}",
            f"Type: {self.type.value}",
            f"Default: {self.default_value}"
        ]
        if self.min_value is not None or self.max_value is not None:
            help_text.append(f"Range: [{self.min_value}, {self.max_value}]")
        if self.valid_values is not None:
            help_text.append(f"Valid Values: {self.valid_values}")
        if self.mathematical_definition:
            help_text.append(f"Details: {self.mathematical_definition}")
        return "\n".join(help_text)


# Define all hyperparameters with detailed information
HYPERPARAMETER_DEFINITIONS: Dict[str, HyperparameterDef] = {
    # ============ SAMPLING PARAMETERS ============
    "temperature": HyperparameterDef(
        name="temperature",
        description="Controls randomness. Lower is more deterministic, higher is more creative.",
        type=HyperparameterType.SAMPLING,
        default_value=0.7, min_value=0.0, max_value=2.0
    ),
    "top_p": HyperparameterDef(
        name="top_p",
        description="Nucleus sampling. Considers the smallest set of tokens whose probability sums to p.",
        type=HyperparameterType.SAMPLING,
        default_value=0.9, min_value=0.0, max_value=1.0
    ),
    "top_k": HyperparameterDef(
        name="top_k",
        description="Sample from the k most likely tokens. Set to 0 to disable.",
        type=HyperparameterType.SAMPLING,
        default_value=50, min_value=0, max_value=1000
    ),
    "do_sample": HyperparameterDef(
        name="do_sample",
        description="Whether to use sampling (True) or greedy decoding (False).",
        type=HyperparameterType.SAMPLING,
        default_value=True, valid_values=[True, False]
    ),
    "typical_p": HyperparameterDef(
        name="typical_p",
        description="Locally typical sampling probability. 1.0 disables it.",
        type=HyperparameterType.SAMPLING,
        default_value=1.0, min_value=0.0, max_value=1.0
    ),
    
    # ============ LENGTH PARAMETERS ============
    "max_new_tokens": HyperparameterDef(
        name="max_new_tokens",
        description="Maximum number of new tokens to generate. (None = auto-calculate)",
        type=HyperparameterType.LENGTH,
        default_value=None, min_value=1, max_value=32768
    ),
    "max_length": HyperparameterDef(
        name="max_length",
        description="[DEPRECATED] Maximum total sequence length (input + output). Use max_new_tokens instead.",
        type=HyperparameterType.LENGTH,
        default_value=None, min_value=1, max_value=32768
    ),
    "min_length": HyperparameterDef(
        name="min_length",
        description="Minimum sequence length (including prompt).",
        type=HyperparameterType.LENGTH,
        default_value=0, min_value=0, max_value=32768
    ),
    "min_new_tokens": HyperparameterDef(
        name="min_new_tokens",
        description="Minimum number of new tokens to generate.",
        type=HyperparameterType.LENGTH,
        default_value=None, min_value=0, max_value=32768
    ),
    
    # ============ PENALTY PARAMETERS ============
    "repetition_penalty": HyperparameterDef(
        name="repetition_penalty",
        description="Penalty for repeating tokens. > 1.0 discourages repetition.",
        type=HyperparameterType.PENALTY,
        default_value=1.0, min_value=0.0, max_value=10.0
    ),
    "length_penalty": HyperparameterDef(
        name="length_penalty",
        description="Exponential penalty for sequence length (for beam search). > 1.0 favors longer sequences.",
        type=HyperparameterType.PENALTY,
        default_value=1.0, min_value=0.0, max_value=10.0
    ),
    "no_repeat_ngram_size": HyperparameterDef(
        name="no_repeat_ngram_size",
        description="Prevent repeating n-grams of this size. 0 to disable.",
        type=HyperparameterType.PENALTY,
        default_value=0, min_value=0, max_value=10
    ),
    
    # ============ BEAM SEARCH PARAMETERS ============
    "num_beams": HyperparameterDef(
        name="num_beams",
        description="Number of beams for beam search. 1 = no beam search.",
        type=HyperparameterType.BEAM_SEARCH,
        default_value=1, min_value=1, max_value=20
    ),
    "num_beam_groups": HyperparameterDef(
        name="num_beam_groups",
        description="Number of groups for diverse beam search. Must divide num_beams.",
        type=HyperparameterType.BEAM_SEARCH,
        default_value=1, min_value=1, max_value=20
    ),
    "diversity_penalty": HyperparameterDef(
        name="diversity_penalty",
        description="Penalty for tokens chosen by other beam groups. Requires num_beam_groups > 1.",
        type=HyperparameterType.BEAM_SEARCH,
        default_value=0.0, min_value=0.0, max_value=10.0
    ),
    
    # ============ TOKEN CONTROL (Usually set automatically) ============
    "pad_token_id": HyperparameterDef(
        name="pad_token_id",
        description="ID of padding token.",
        type=HyperparameterType.TOKEN_CONTROL,
        default_value=None
    ),
    "eos_token_id": HyperparameterDef(
        name="eos_token_id",
        description="ID of end-of-sequence token.",
        type=HyperparameterType.TOKEN_CONTROL,
        default_value=None
    ),
    "bos_token_id": HyperparameterDef(
        name="bos_token_id",
        description="ID of beginning-of-sequence token.",
        type=HyperparameterType.TOKEN_CONTROL,
        default_value=None
    ),
    
    # ============ CACHE AND OPTIMIZATION ============
    "use_cache": HyperparameterDef(
        name="use_cache",
        description="Use past key/value cache for faster generation.",
        type=HyperparameterType.CACHE,
        default_value=True, valid_values=[True, False]
    ),
}

# --- Configuration Dataclass ---

@dataclass
class HyperparameterConfig:
    """Complete hyperparameter configuration for LLM generation"""
    
    # Sampling parameters
    temperature: float = field(default=HYPERPARAMETER_DEFINITIONS["temperature"].default_value)
    top_p: float = field(default=HYPERPARAMETER_DEFINITIONS["top_p"].default_value)
    top_k: int = field(default=HYPERPARAMETER_DEFINITIONS["top_k"].default_value)
    do_sample: bool = field(default=HYPERPARAMETER_DEFINITIONS["do_sample"].default_value)
    typical_p: float = field(default=HYPERPARAMETER_DEFINITIONS["typical_p"].default_value)
    
    # Length parameters
    max_new_tokens: Optional[int] = field(default=HYPERPARAMETER_DEFINITIONS["max_new_tokens"].default_value)
    max_length: Optional[int] = field(default=HYPERPARAMETER_DEFINITIONS["max_length"].default_value)
    min_length: int = field(default=HYPERPARAMETER_DEFINITIONS["min_length"].default_value)
    min_new_tokens: Optional[int] = field(default=HYPERPARAMETER_DEFINITIONS["min_new_tokens"].default_value)
    
    # Penalty parameters
    repetition_penalty: float = field(default=HYPERPARAMETER_DEFINITIONS["repetition_penalty"].default_value)
    length_penalty: float = field(default=HYPERPARAMETER_DEFINITIONS["length_penalty"].default_value)
    no_repeat_ngram_size: int = field(default=HYPERPARAMETER_DEFINITIONS["no_repeat_ngram_size"].default_value)
    encoder_repetition_penalty: float = field(default=HYPERPARAMETER_DEFINITIONS["encoder_repetition_penalty"].default_value)
    
    # Beam search parameters
    num_beams: int = field(default=HYPERPARAMETER_DEFINITIONS["num_beams"].default_value)
    num_beam_groups: int = field(default=HYPERPARAMETER_DEFINITIONS["num_beam_groups"].default_value)
    diversity_penalty: float = field(default=HYPERPARAMETER_DEFINITIONS["diversity_penalty"].default_value)
    early_stopping: bool = field(default=HYPERPARAMETER_DEFINITIONS["early_stopping"].default_value)
    num_return_sequences: int = field(default=HYPERPARAMETER_DEFINITIONS["num_return_sequences"].default_value)
    
    # Token control
    pad_token_id: Optional[int] = field(default=HYPERPARAMETER_DEFINITIONS["pad_token_id"].default_value)
    eos_token_id: Optional[int] = field(default=HYPERPARAMETER_DEFINITIONS["eos_token_id"].default_value)
    bos_token_id: Optional[int] = field(default=HYPERPARAMETER_DEFINITIONS["bos_token_id"].default_value)
    forced_bos_token_id: Optional[int] = field(default=HYPERPARAMETER_DEFINITIONS["forced_bos_token_id"].default_value)
    forced_eos_token_id: Optional[int] = field(default=HYPERPARAMETER_DEFINITIONS["forced_eos_token_id"].default_value)
    
    # Cache and optimization
    use_cache: bool = field(default=HYPERPARAMETER_DEFINITIONS["use_cache"].default_value)
    
    # Advanced parameters
    exponential_decay_length_penalty: Optional[tuple] = field(default=HYPERPARAMETER_DEFINITIONS["exponential_decay_length_penalty"].default_value)
    bad_words_ids: Optional[List[List[int]]] = field(default=HYPERPARAMETER_DEFINITIONS["bad_words_ids"].default_value)
    force_words_ids: Optional[List[List[int]]] = field(default=HYPERPARAMETER_DEFINITIONS["force_words_ids"].default_value)
    renormalize_logits: bool = field(default=HYPERPARAMETER_DEFINITIONS["renormalize_logits"].default_value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, removing None values for cleaner serialization."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        config_dict = self.to_dict() # Use to_dict to skip None values
        
        for param_name, value in config_dict.items():
            if param_name in HYPERPARAMETER_DEFINITIONS:
                param_def = HYPERPARAMETER_DEFINITIONS[param_name]
                if not param_def.validate(value):
                    errors.append(
                        f"{param_name}={value} is invalid. "
                        f"Valid: {param_def.valid_values or f'[{param_def.min_value}, {param_def.max_value}]'}"
                    )
        
        if self.num_return_sequences > self.num_beams:
            errors.append(f"num_return_sequences ({self.num_return_sequences}) must be <= num_beams ({self.num_beams})")
        if self.num_beam_groups > 1 and self.num_beams % self.num_beam_groups != 0:
            errors.append(f"num_beams ({self.num_beams}) must be divisible by num_beam_groups ({self.num_beam_groups})")
        
        return errors

    def save(self, filepath: Path):
        """Save configuration to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved hyperparameter config to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'HyperparameterConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded hyperparameter config from {filepath}")
        return cls(**data) # Unpack dictionary into dataclass fields
    
    @classmethod
    def create_preset(cls, preset_name: str) -> 'HyperparameterConfig':
        """Create a preset configuration from the loaded presets."""
        config_data = get_hyperparameter_preset(preset_name)
        if config_data:
            return config_data
        logger.warning(f"Preset '{preset_name}' not found. Returning default config.")
        return cls() # Return default config if preset name is invalid


class InteractiveHyperparameterEditor:
    """Interactive editor for hyperparameter configuration"""
    
    def __init__(self, config: Optional[HyperparameterConfig] = None):
        self.config = config or HyperparameterConfig()
        
        # Group parameters by category
        self.categories: Dict[str, List[str]] = {}
        for name, param_def in HYPERPARAMETER_DEFINITIONS.items():
            category = param_def.type.value
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(name)
            
    def edit(self) -> HyperparameterConfig:
        """Interactive editing session"""
        while True:
            ConsoleOutput.header("Hyperparameter Configuration Editor")
            
            # Dynamically create category menu
            category_keys = sorted(self.categories.keys())
            for i, category in enumerate(category_keys):
                print(f"  {i+1}. Edit {category.capitalize()} Parameters")
            
            print("\n  --- Other Actions ---")
            print(f"  P. Load Preset")
            print(f"  V. View Current Configuration")
            print(f"  D. Validate Configuration")
            print(f"  S. Save and Exit")
            print(f"  Q. Exit without Saving")
            
            choice = input("\nSelect category or action: ").strip().lower()
            
            if choice == 'q': return self.config # Return current (potentially unsaved) config
            if choice == 's': return self.config # (Saving happens in ConfigManager)
            if choice == 'p': self._load_preset()
            elif choice == 'v': self._view_config()
            elif choice == 'd': self._validate_config()
            elif choice.isdigit():
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(category_keys):
                        category_name = category_keys[idx]
                        self._edit_category(category_name, self.categories[category_name])
                    else:
                        ConsoleOutput.warning("Invalid category number.")
                except ValueError:
                    ConsoleOutput.warning("Invalid input.")
            else:
                ConsoleOutput.warning("Invalid choice.")
    
    def _edit_category(self, category_name: str, param_names: List[str]):
        """Helper to edit all parameters within a category."""
        ConsoleOutput.section(f"Editing {category_name.capitalize()} Parameters")
        for param_name in param_names:
            if hasattr(self.config, param_name):
                self._edit_param(param_name)
    
    def _edit_param(self, param_name: str):
        """Edit a single parameter with validation."""
        param_def = HYPERPARAMETER_DEFINITIONS[param_name]
        current_value = getattr(self.config, param_name)
        
        print(f"\nEditing: {param_name} (Current: {current_value})")
        print(f"  {param_def.description}")
        
        prompt = f"  New value (Default: {param_def.default_value}"
        if param_def.min_value is not None and param_def.max_value is not None:
            prompt += f", Range: [{param_def.min_value}, {param_def.max_value}]"
        if param_def.valid_values:
            prompt += f", Options: {param_def.valid_values}"
        prompt += ", '?' for help, Enter to keep): "
        
        new_value_str = input(prompt).strip()
        
        if not new_value_str:
            return # Keep current value
        
        if new_value_str == '?':
            print(param_def.get_help_text())
            input("  Press Enter to continue...")
            self._edit_param(param_name) # Retry editing this param
            return
        
        # Handle 'None' as a special string input
        if new_value_str.lower() == 'none':
            if param_name in ['max_new_tokens', 'max_length', 'min_new_tokens', 'pad_token_id', 'eos_token_id', 'bos_token_id', 'forced_bos_token_id', 'forced_eos_token_id']:
                 setattr(self.config, param_name, None)
                 print(f"  ✓ Set {param_name} to None.")
                 return
            else:
                 print(f"  ✗ '{param_name}' cannot be set to None.")
                 return

        try:
            # Convert to appropriate type based on default value
            target_type = type(param_def.default_value)
            if target_type == bool:
                converted_value = new_value_str.lower() in ('true', 'yes', '1', 'y')
            elif target_type == int:
                converted_value = int(new_value_str)
            elif target_type == float:
                converted_value = float(new_value_str)
            elif param_def.default_value is None: # Handle fields that default to None
                 # Try to guess type (e.g., int, float, or str)
                 try: converted_value = int(new_value_str)
                 except ValueError:
                     try: converted_value = float(new_value_str)
                     except ValueError: converted_value = new_value_str
            else:
                converted_value = new_value_str # Assume string if not bool/int/float

            # Validate
            if param_def.validate(converted_value):
                setattr(self.config, param_name, converted_value)
                print(f"  ✓ Updated {param_name} to: {converted_value}")
            else:
                ConsoleOutput.warning(f"  ✗ Value '{new_value_str}' is invalid for {param_name}.")
        
        except ValueError:
            ConsoleOutput.warning(f"  ✗ Invalid input type. Expected {type(param_def.default_value).__name__}.")
        except Exception as e:
             ConsoleOutput.error(f"  ✗ An error occurred: {e}")

    def _load_preset(self):
        """Load a preset configuration."""
        presets = list_hyperparameter_presets() # Get list of (name, desc) tuples
        if not presets:
             ConsoleOutput.warning("No presets defined.")
             return
             
        print("\nAvailable Presets:")
        preset_map = {}
        for i, (name, desc) in enumerate(presets, 1):
            print(f"  {i}. {name} - {desc}")
            preset_map[str(i)] = name
        print("  b. Back")
        
        choice = input(f"Select preset (1-{len(presets)} or b): ").strip().lower()
        
        if choice == 'b': return
        
        preset_name = preset_map.get(choice)
        if preset_name:
            preset_config = get_hyperparameter_preset(preset_name)
            if preset_config:
                 self.config = preset_config
                 ConsoleOutput.success(f"Loaded '{preset_name}' preset")
            else:
                 ConsoleOutput.error(f"Failed to load preset '{preset_name}'.") # Should not happen if list_presets worked
        else:
            ConsoleOutput.warning("Invalid selection.")
    
    def _view_config(self):
        """View current configuration"""
        ConsoleOutput.section("Current Hyperparameters")
        config_dict = self.config.to_dict()
        for key, value in config_dict.items():
            print(f"  {key:<30}: {value}")
        print("-" * 60)
        input("Press Enter to continue...")

    def _validate_config(self):
        """Validate current configuration"""
        errors = self.config.validate()
        if not errors:
            ConsoleOutput.success("✓ Configuration is valid.")
        else:
            ConsoleOutput.error("✗ Configuration has errors:")
            for error in errors:
                print(f"  - {error}")
        input("\nPress Enter to continue...")

def get_hyperparameter_help(param_name: str) -> str:
    """Get detailed help for a hyperparameter."""
    param_def = HYPERPARAMETER_DEFINITIONS.get(param_name)
    if param_def:
        return param_def.get_help_text()
    return f"Unknown hyperparameter: {param_name}"
