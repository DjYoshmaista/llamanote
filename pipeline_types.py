"""
Pipeline Types Module
Defines shared data structures for the processing pipeline and LLM handlers.
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# Need imports from other modules for type hints within dataclasses
from text_processor import ChunkingStrategy
from hyperparameters import HyperparameterConfig

# --- Enums ---

class ProcessingMode(Enum):
    """Processing modes for different use cases"""
    PODCAST = "podcast"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    SUMMARY = "summary"
    CUSTOM = "custom"

# --- Dataclasses previously in llm_handler.py ---

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    method: str = "8bit"  # none, 4bit, 8bit, 16bit
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = True  # For 4bit
    quant_type: str = "nf4"  # nf4 or fp4 for 4bit
    bnb_4bit_use_double_quant: bool = True # Explicit for bnb
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False

    def to_bnb_config(self) -> Optional['BitsAndBytesConfig']:
        """Convert to BitsAndBytes configuration"""
        # BitsAndBytesConfig import needs to be handled carefully
        # It's better if this logic stays within the backend that uses it.
        # For now, keep the structure but note the dependency.
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            # Handle cases where transformers isn't installed or BitsAndBytesConfig changes
            return None # Or raise a specific error

        if self.method == "none" or self.method == "16bit":
            return None

        if self.method == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.quant_type
            )
        elif self.method == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=self.llm_int8_threshold,
                llm_int8_skip_modules=self.llm_int8_skip_modules,
                llm_int8_enable_fp32_cpu_offload=self.llm_int8_enable_fp32_cpu_offload,
                llm_int8_has_fp16_weight=self.llm_int8_has_fp16_weight
            )

        # logger.warning(f"Unknown quantization method: {self.method}. No quantization applied.")
        # Cannot use logger here easily, maybe return None and let caller log.
        return None

@dataclass
class LayerSplitConfig:
    """Configuration for splitting model layers between devices"""
    enabled: bool = True
    gpu_layers: int = -1  # -1 means auto-detect (GGUF uses this, HF Transformers uses device_map)
    max_gpu_memory: Dict[int, str] = None  # e.g., {0: "10GB", 1: "10GB"}
    max_cpu_memory: str = "30GB"
    offload_folder: Optional[Path] = None
    offload_state_dict: bool = False

    def __post_init__(self):
        # Default max_gpu_memory logic requires torch, handle potential import error
        if self.max_gpu_memory is None:
            try:
                import torch
                if torch.cuda.is_available():
                    num_gpus = torch.cuda.device_count()
                    # Default to a reasonable amount per GPU if none specified
                    self.max_gpu_memory = {i: "10GB" for i in range(num_gpus)}
                else:
                    self.max_gpu_memory = {}
            except ImportError:
                 self.max_gpu_memory = {} # No GPU memory settings if torch isn't available

        # Resolve offload folder path if specified
        if self.offload_folder:
            self.offload_folder = Path(self.offload_folder).resolve()


    def get_max_memory_dict(self) -> Dict:
        """Get max memory dictionary for device mapping"""
        max_memory = {}
        if self.max_gpu_memory:
            max_memory.update(self.max_gpu_memory)
        max_memory["cpu"] = self.max_cpu_memory
        return max_memory

@dataclass
class GenerationResult:
    """Result from text generation"""
    raw_output: str
    filtered_output: str
    input_tokens: int
    output_tokens: int
    generation_time: float
    memory_used: float # In MB
    device_map: Optional[Dict] = None # Information about how model was distributed


# --- Dataclasses previously in processing_pipeline.py ---

@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline"""
    mode: ProcessingMode = ProcessingMode.PODCAST
    model_name: str = "default_model_metadata" # This is for metadata (e.g., "local:qwen3-4b")
    model_provider: str = "local" # 'local', 'local_gguf', 'openai', 'google', etc.
    model_specifier: str = "default_model_specifier" # The actual model ID or path
    memory_profile: str = "medium_vram" # Used to configure local HF models if LayerSplitConfig not provided
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY
    chunk_size: int = 1000 # Use a default, can be overridden
    markdown_style: str = "podcast"
    system_prompt: Optional[str] = None
    remove_thinking: bool = True
    preserve_layout: bool = True # Relevant for PDF extraction
    clean_for_audio: bool = True # Relevant for preprocessing
    add_emotions: bool = True # Relevant for formatting
    enable_checkpoints: bool = True # Global flag, maybe move elsewhere
    max_retries: int = 3 # Global flag, maybe move elsewhere
    output_format: str = "markdown"
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    # Configs for local models - now directly part of PipelineConfig
    quantization_config: Optional[QuantizationConfig] = None
    layer_split_config: Optional[LayerSplitConfig] = None


@dataclass
class PipelineResult:
    """Result from pipeline processing"""
    success: bool
    input_file: Path
    output_file: Optional[Path]
    processing_time: float
    stages_completed: List[str]
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = None
