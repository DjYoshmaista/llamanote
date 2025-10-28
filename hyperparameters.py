"""
Hyperparameter Management Module
Comprehensive LLM hyperparameter definitions and configuration
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from enum import Enum
import json
from pathlib import Path

def _get_logger():
    # Lazy logger import to avoid circular dependency
    from loggerConf import get_logger_conf
    return get_logger_conf(__name__)

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


# Define all hyperparameters with detailed information
HYPERPARAMETER_DEFINITIONS = {
    # ============ SAMPLING PARAMETERS ============
    "temperature": HyperparameterDef(
        name="temperature",
        description="Controls randomness in sampling. Lower values make output more deterministic and focused, higher values more random and creative.",
        type=HyperparameterType.SAMPLING,
        default_value=0.7,
        min_value=0.0,
        max_value=2.0,
        mathematical_definition="P(token) = softmax(logits / temperature). As T→0, approaches argmax; as T→∞, approaches uniform distribution."
    ),
    
    "top_p": HyperparameterDef(
        name="top_p",
        description="Nucleus sampling: sample from smallest set of tokens whose cumulative probability exceeds p. Balances diversity and quality.",
        type=HyperparameterType.SAMPLING,
        default_value=0.9,
        min_value=0.0,
        max_value=1.0,
        mathematical_definition="Select tokens where Σ P(token_i) ≥ p, sorted by probability descending."
    ),
    
    "top_k": HyperparameterDef(
        name="top_k",
        description="Sample from top k most likely tokens. Limits vocabulary at each step to control randomness.",
        type=HyperparameterType.SAMPLING,
        default_value=50,
        min_value=1,
        max_value=1000,
        mathematical_definition="Consider only k tokens with highest P(token), zero out others before sampling."
    ),
    
    "do_sample": HyperparameterDef(
        name="do_sample",
        description="Whether to use sampling (True) or greedy decoding (False). Greedy always picks highest probability token.",
        type=HyperparameterType.SAMPLING,
        default_value=True,
        valid_values=[True, False],
        mathematical_definition="If False: token = argmax(P(token)); If True: token ~ P(token)"
    ),
    
    # ============ LENGTH PARAMETERS ============
    "max_new_tokens": HyperparameterDef(
        name="max_new_tokens",
        description="Maximum number of tokens to generate. Controls generation length independently of input.",
        type=HyperparameterType.LENGTH,
        default_value=2048,
        min_value=1,
        max_value=32768,
        mathematical_definition="Stop generation when generated_tokens ≥ max_new_tokens"
    ),
    
    "max_length": HyperparameterDef(
        name="max_length",
        description="Maximum total sequence length (input + output). Alternative to max_new_tokens.",
        type=HyperparameterType.LENGTH,
        default_value=None,
        min_value=1,
        max_value=32768,
        mathematical_definition="Stop when len(input_ids) + generated_tokens ≥ max_length"
    ),
    
    "min_length": HyperparameterDef(
        name="min_length",
        description="Minimum sequence length. Forces generation to continue until this length is reached.",
        type=HyperparameterType.LENGTH,
        default_value=0,
        min_value=0,
        max_value=32768,
        mathematical_definition="Prevent EOS token until len(sequence) ≥ min_length"
    ),
    
    "min_new_tokens": HyperparameterDef(
        name="min_new_tokens",
        description="Minimum number of new tokens to generate. Similar to min_length but relative to input.",
        type=HyperparameterType.LENGTH,
        default_value=None,
        min_value=0,
        max_value=32768,
        mathematical_definition="Prevent EOS token until generated_tokens ≥ min_new_tokens"
    ),
    
    # ============ PENALTY PARAMETERS ============
    "repetition_penalty": HyperparameterDef(
        name="repetition_penalty",
        description="Penalty for repeating tokens. Values > 1 discourage repetition, < 1 encourage it.",
        type=HyperparameterType.PENALTY,
        default_value=1.0,
        min_value=0.0,
        max_value=10.0,
        mathematical_definition="For token t: if t in generated_tokens, logit(t) /= repetition_penalty"
    ),
    
    "length_penalty": HyperparameterDef(
        name="length_penalty",
        description="Exponential penalty for sequence length in beam search. > 1 encourages longer sequences, < 1 encourages shorter.",
        type=HyperparameterType.PENALTY,
        default_value=1.0,
        min_value=0.0,
        max_value=10.0,
        mathematical_definition="score = log_prob / (length ^ length_penalty)"
    ),
    
    "no_repeat_ngram_size": HyperparameterDef(
        name="no_repeat_ngram_size",
        description="Prevent repeating n-grams of this size. Set to 0 to disable. Useful for avoiding repetitive text.",
        type=HyperparameterType.PENALTY,
        default_value=0,
        min_value=0,
        max_value=10,
        mathematical_definition="If n-gram of size n exists in generated text, set logit of next token completing that n-gram to -∞"
    ),
    
    "encoder_repetition_penalty": HyperparameterDef(
        name="encoder_repetition_penalty",
        description="Penalty for tokens already in the encoder input. Discourages copying from input.",
        type=HyperparameterType.PENALTY,
        default_value=1.0,
        min_value=0.0,
        max_value=10.0,
        mathematical_definition="For token t in encoder_input: logit(t) /= encoder_repetition_penalty"
    ),
    
    # ============ BEAM SEARCH PARAMETERS ============
    "num_beams": HyperparameterDef(
        name="num_beams",
        description="Number of beams for beam search. 1 means no beam search (greedy/sampling only). Higher values explore more hypotheses.",
        type=HyperparameterType.BEAM_SEARCH,
        default_value=1,
        min_value=1,
        max_value=20,
        mathematical_definition="Maintain top-k hypotheses at each step, where k = num_beams"
    ),
    
    "num_beam_groups": HyperparameterDef(
        name="num_beam_groups",
        description="Number of groups for diverse beam search. Must divide num_beams evenly. Encourages diverse outputs.",
        type=HyperparameterType.BEAM_SEARCH,
        default_value=1,
        min_value=1,
        max_value=20,
        mathematical_definition="Split beams into groups, apply diversity penalty between groups"
    ),
    
    "diversity_penalty": HyperparameterDef(
        name="diversity_penalty",
        description="Penalty for tokens chosen by other beam groups. Used with num_beam_groups > 1 to encourage diversity.",
        type=HyperparameterType.BEAM_SEARCH,
        default_value=0.0,
        min_value=0.0,
        max_value=10.0,
        mathematical_definition="For token t chosen by another group: logit(t) -= diversity_penalty"
    ),
    
    "early_stopping": HyperparameterDef(
        name="early_stopping",
        description="Whether to stop beam search when at least num_beams sentences are finished. Can be True, False, or 'never'.",
        type=HyperparameterType.BEAM_SEARCH,
        default_value=False,
        valid_values=[True, False, "never"],
        mathematical_definition="If True: stop when finished_beams ≥ num_beams"
    ),
    
    "num_return_sequences": HyperparameterDef(
        name="num_return_sequences",
        description="Number of independently computed returned sequences. Must be ≤ num_beams if using beam search.",
        type=HyperparameterType.BEAM_SEARCH,
        default_value=1,
        min_value=1,
        max_value=20,
        mathematical_definition="Return top-n scored sequences from beam search or sampling"
    ),
    
    # ============ TOKEN CONTROL PARAMETERS ============
    "pad_token_id": HyperparameterDef(
        name="pad_token_id",
        description="ID of padding token. Used to pad sequences to same length in batches.",
        type=HyperparameterType.TOKEN_CONTROL,
        default_value=None,
        min_value=0,
        mathematical_definition="Token used for padding: attention_mask[pad_positions] = 0"
    ),
    
    "eos_token_id": HyperparameterDef(
        name="eos_token_id",
        description="ID(s) of end-of-sequence token. Generation stops when this token is produced.",
        type=HyperparameterType.TOKEN_CONTROL,
        default_value=None,
        min_value=0,
        mathematical_definition="Stop generation when token = eos_token_id"
    ),
    
    "bos_token_id": HyperparameterDef(
        name="bos_token_id",
        description="ID of beginning-of-sequence token. Used to start generation.",
        type=HyperparameterType.TOKEN_CONTROL,
        default_value=None,
        min_value=0,
        mathematical_definition="If no input: generated_tokens[0] = bos_token_id"
    ),
    
    "forced_bos_token_id": HyperparameterDef(
        name="forced_bos_token_id",
        description="Force generation to start with this token ID. Overrides model's default.",
        type=HyperparameterType.TOKEN_CONTROL,
        default_value=None,
        min_value=0,
        mathematical_definition="Set logits[0, forced_bos_token_id] = ∞, others = -∞ at start"
    ),
    
    "forced_eos_token_id": HyperparameterDef(
        name="forced_eos_token_id",
        description="Force generation to end with this token at max_length. Ensures proper termination.",
        type=HyperparameterType.TOKEN_CONTROL,
        default_value=None,
        min_value=0,
        mathematical_definition="At max_length: set logits[forced_eos_token_id] = ∞"
    ),
    
    # ============ CACHE AND OPTIMIZATION ============
    "use_cache": HyperparameterDef(
        name="use_cache",
        description="Whether to use past key/value cache for faster generation. Recommended for long sequences.",
        type=HyperparameterType.CACHE,
        default_value=True,
        valid_values=[True, False],
        mathematical_definition="Store and reuse K,V matrices from previous steps: K_cached, V_cached"
    ),
    
    # ============ ADVANCED PARAMETERS ============
    "exponential_decay_length_penalty": HyperparameterDef(
        name="exponential_decay_length_penalty",
        description="Tuple (start_index, decay_factor) for exponentially increasing length penalty. Helps control very long generations.",
        type=HyperparameterType.ADVANCED,
        default_value=None,
        mathematical_definition="After start_index: penalty *= exp(decay_factor * (current_length - start_index))"
    ),
    
    "bad_words_ids": HyperparameterDef(
        name="bad_words_ids",
        description="List of token ids that are not allowed to be generated. Useful for filtering unwanted content.",
        type=HyperparameterType.ADVANCED,
        default_value=None,
        mathematical_definition="For token_id in bad_words_ids: logits[token_id] = -∞"
    ),
    
    "force_words_ids": HyperparameterDef(
        name="force_words_ids",
        description="List of token sequences that must appear in generated text. Enforces specific content.",
        type=HyperparameterType.ADVANCED,
        default_value=None,
        mathematical_definition="Constrained decoding: ensure all sequences in force_words_ids appear"
    ),
    
    "renormalize_logits": HyperparameterDef(
        name="renormalize_logits",
        description="Whether to renormalize logits after applying all processors. Ensures valid probability distribution.",
        type=HyperparameterType.ADVANCED,
        default_value=False,
        valid_values=[True, False],
        mathematical_definition="After modifications: logits = softmax(modified_logits)"
    ),
    
    "typical_p": HyperparameterDef(
        name="typical_p",
        description="Locally typical sampling: sample from tokens whose conditional probability is close to expected. Alternative to top_p.",
        type=HyperparameterType.SAMPLING,
        default_value=1.0,
        min_value=0.0,
        max_value=1.0,
        mathematical_definition="Sample tokens where |log P(token) - E[log P]| is minimal"
    ),
}


@dataclass
class HyperparameterConfig:
    """Complete hyperparameter configuration for LLM generation"""
    
    # Sampling parameters
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    typical_p: float = 1.0
    
    # Length parameters
    max_new_tokens: Optional[int] = 2048
    max_length: Optional[int] = None
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    
    # Penalty parameters
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    encoder_repetition_penalty: float = 1.0
    
    # Beam search parameters
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    early_stopping: bool = False
    num_return_sequences: int = 1
    
    # Token control
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    
    # Cache and optimization
    use_cache: bool = True
    
    # Advanced parameters
    exponential_decay_length_penalty: Optional[tuple] = None
    bad_words_ids: Optional[List[List[int]]] = None
    force_words_ids: Optional[List[List[int]]] = None
    renormalize_logits: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, removing None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Check each parameter against its definition
        config_dict = asdict(self)
        for param_name, value in config_dict.items():
            if value is None:
                continue
            
            if param_name in HYPERPARAMETER_DEFINITIONS:
                param_def = HYPERPARAMETER_DEFINITIONS[param_name]
                if not param_def.validate(value):
                    errors.append(
                        f"{param_name}={value} is invalid. "
                        f"Valid range: [{param_def.min_value}, {param_def.max_value}]"
                    )
        
        # Cross-parameter validation
        if self.num_return_sequences > self.num_beams:
            errors.append(
                f"num_return_sequences ({self.num_return_sequences}) "
                f"cannot exceed num_beams ({self.num_beams})"
            )
        
        if self.num_beam_groups > 1:
            if self.num_beams % self.num_beam_groups != 0:
                errors.append(
                    f"num_beams ({self.num_beams}) must be divisible by "
                    f"num_beam_groups ({self.num_beam_groups})"
                )
        
        if self.max_length is not None and self.max_new_tokens is not None:
            errors.append(
                "Cannot specify both max_length and max_new_tokens. Use max_new_tokens for modern usage."
            )
        
        return errors
    
    def save(self, filepath: Path):
        """Save configuration to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger = _get_logger()
        logger.info(f"Saved hyperparameter config to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'HyperparameterConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger = _get_logger()
        logger.info(f"Loaded hyperparameter config from {filepath}")
        return cls(**data)
    
    @classmethod
    def create_preset(cls, preset_name: str) -> 'HyperparameterConfig':
        """Create a preset configuration"""
        presets = {
            "creative": cls(
                temperature=0.9,
                top_p=0.95,
                top_k=100,
                repetition_penalty=1.1
            ),
            "balanced": cls(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.0
            ),
            "precise": cls(
                temperature=0.3,
                top_p=0.85,
                top_k=20,
                repetition_penalty=1.0
            ),
            "deterministic": cls(
                temperature=0.1,
                do_sample=False,
                top_k=1,
                repetition_penalty=1.0
            ),
            "long_form": cls(
                temperature=0.7,
                top_p=0.9,
                max_new_tokens=4096,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3
            ),
        }
        
        return presets.get(preset_name, cls())


class InteractiveHyperparameterEditor:
    """Interactive editor for hyperparameter configuration"""
    
    def __init__(self, config: Optional[HyperparameterConfig] = None):
        self.config = config or HyperparameterConfig()
    
    def edit(self) -> HyperparameterConfig:
        """Interactive editing session"""
        from loggerConf import ConsoleOutput
        
        ConsoleOutput.header("Hyperparameter Configuration Editor")
        
        while True:
            print("\nCategories:")
            print("1. Sampling Parameters (temperature, top_p, top_k)")
            print("2. Length Parameters (max_tokens, min_length)")
            print("3. Penalty Parameters (repetition, length_penalty)")
            print("4. Beam Search Parameters")
            print("5. Token Control")
            print("6. Advanced Parameters")
            print("7. Load Preset")
            print("8. View Current Configuration")
            print("9. Validate Configuration")
            print("10. Save and Exit")
            print("11. Exit without saving")
            
            choice = input("\nSelect category (1-11): ").strip()
            
            if choice == "1":
                self._edit_sampling()
            elif choice == "2":
                self._edit_length()
            elif choice == "3":
                self._edit_penalties()
            elif choice == "4":
                self._edit_beam_search()
            elif choice == "5":
                self._edit_token_control()
            elif choice == "6":
                self._edit_advanced()
            elif choice == "7":
                self._load_preset()
            elif choice == "8":
                self._view_config()
            elif choice == "9":
                self._validate_config()
            elif choice == "10":
                return self.config
            elif choice == "11":
                return self.config
    
    def _edit_sampling(self):
        """Edit sampling parameters"""
        print("\nSampling Parameters:")
        self._edit_param("temperature")
        self._edit_param("top_p")
        self._edit_param("top_k")
        self._edit_param("do_sample")
        self._edit_param("typical_p")
    
    def _edit_length(self):
        """Edit length parameters"""
        print("\nLength Parameters:")
        self._edit_param("max_new_tokens")
        self._edit_param("min_length")
        self._edit_param("min_new_tokens")
    
    def _edit_penalties(self):
        """Edit penalty parameters"""
        print("\nPenalty Parameters:")
        self._edit_param("repetition_penalty")
        self._edit_param("length_penalty")
        self._edit_param("no_repeat_ngram_size")
    
    def _edit_beam_search(self):
        """Edit beam search parameters"""
        print("\nBeam Search Parameters:")
        self._edit_param("num_beams")
        self._edit_param("num_beam_groups")
        self._edit_param("diversity_penalty")
        self._edit_param("early_stopping")
        self._edit_param("num_return_sequences")
    
    def _edit_token_control(self):
        """Edit token control parameters"""
        print("\nToken Control Parameters:")
        print("(Usually set automatically by tokenizer)")
        self._edit_param("pad_token_id")
        self._edit_param("eos_token_id")
        self._edit_param("bos_token_id")
    
    def _edit_advanced(self):
        """Edit advanced parameters"""
        print("\nAdvanced Parameters:")
        self._edit_param("use_cache")
        self._edit_param("renormalize_logits")
    
    def _edit_param(self, param_name: str):
        """Edit a single parameter"""
        if param_name not in HYPERPARAMETER_DEFINITIONS:
            return
        
        param_def = HYPERPARAMETER_DEFINITIONS[param_name]
        current_value = getattr(self.config, param_name)
        
        print(f"\n{param_name}:")
        print(f"  Description: {param_def.description}")
        print(f"  Current value: {current_value}")
        print(f"  Default: {param_def.default_value}")
        
        if param_def.min_value is not None and param_def.max_value is not None:
            print(f"  Range: [{param_def.min_value}, {param_def.max_value}]")
        
        new_value = input(f"  New value (press Enter to keep current): ").strip()
        
        if new_value:
            try:
                # Convert to appropriate type
                if isinstance(param_def.default_value, bool):
                    new_value = new_value.lower() in ('true', 'yes', '1', 'y')
                elif isinstance(param_def.default_value, int):
                    new_value = int(new_value)
                elif isinstance(param_def.default_value, float):
                    new_value = float(new_value)
                
                # Validate
                if param_def.validate(new_value):
                    setattr(self.config, param_name, new_value)
                    print(f"  ✓ Updated to: {new_value}")
                else:
                    print(f"  ✗ Invalid value")
            except ValueError:
                print(f"  ✗ Invalid input")
    
    def _load_preset(self):
        """Load a preset configuration"""
        print("\nAvailable Presets:")
        print("1. Creative - High creativity, diverse output")
        print("2. Balanced - Good balance of quality and diversity")
        print("3. Precise - Focused, deterministic output")
        print("4. Deterministic - Most focused, no randomness")
        print("5. Long Form - Optimized for long text generation")
        
        choice = input("Select preset (1-5): ").strip()
        
        preset_map = {
            "1": "creative",
            "2": "balanced",
            "3": "precise",
            "4": "deterministic",
            "5": "long_form"
        }
        
        if choice in preset_map:
            self.config = HyperparameterConfig.create_preset(preset_map[choice])
            print(f"✓ Loaded '{preset_map[choice]}' preset")
    
    def _view_config(self):
        """View current configuration"""
        print("\nCurrent Configuration:")
        print("=" * 50)
        
        config_dict = self.config.to_dict()
        for key, value in config_dict.items():
            print(f"{key}: {value}")
        
        print("=" * 50)
    
    def _validate_config(self):
        """Validate current configuration"""
        errors = self.config.validate()
        
        if not errors:
            print("\n✓ Configuration is valid")
        else:
            print("\n✗ Configuration has errors:")
            for error in errors:
                print(f"  - {error}")


def get_hyperparameter_help(param_name: str) -> str:
    """Get detailed help for a hyperparameter"""
    if param_name not in HYPERPARAMETER_DEFINITIONS:
        return f"Unknown hyperparameter: {param_name}"
    
    param_def = HYPERPARAMETER_DEFINITIONS[param_name]
    
    help_text = f"""
{param_name.upper()}
{'=' * 60}

Description:
  {param_def.description}

Type: {param_def.type.value}
Default Value: {param_def.default_value}
"""
    
    if param_def.min_value is not None or param_def.max_value is not None:
        help_text += f"Range: [{param_def.min_value}, {param_def.max_value}]\n"
    
    if param_def.valid_values is not None:
        help_text += f"Valid Values: {param_def.valid_values}\n"
    
    if param_def.mathematical_definition:
        help_text += f"\nMathematical Definition:\n  {param_def.mathematical_definition}\n"
    
    return help_text
