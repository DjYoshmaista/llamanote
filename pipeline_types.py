"""
Shared Data Types for LlamaNote Pipeline
Contains dataclasses and enums used across multiple pipeline modules
to prevent circular imports.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from enum import Enum

# --- LlamaNote Modules ---
# Import only base types needed for definitions, avoid full module imports
from hyperparameters import HyperparameterConfig
from text_processor import ChunkingStrategy # Import Enum directly
from config_base import (
    DEFAULT_MODEL,
    PIPELINE_STAGES,
    ENABLE_STAGE_CHECKPOINTS,
    MAX_RETRIES,
    DEFAULT_QUANTIZATION,
    ENABLE_LAYER_SPLITTING,
    DEFAULT_GPU_LAYERS,
    OFFLOAD_DIR
)

# --- Enums ---

class ProcessingMode(Enum):
    """Processing modes for different use cases"""
    PODCAST = "podcast"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    SUMMARY = "summary"
    CUSTOM = "custom"

# --- Dataclasses ---

@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    method: str = DEFAULT_QUANTIZATION  # none, 4bit, 8bit, 16bit
    compute_dtype: torch.dtype = torch.bfloat16
    use_double_quant: bool = True  # For 4bit
    quant_type: str = "nf4"  # nf4 or fp4 for 4bit
    bnb_4bit_use_double_quant: bool = True # Explicit for bnb
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: Optional[List[str]] = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False

    def to_bnb_config(self): # Keep type hint Optional[BitsAndBytesConfig] out
        """Convert to BitsAndBytes configuration object."""
        # Avoid direct import of BitsAndBytesConfig here if possible
        # This method might need to be in llm_handler where BitsAndBytesConfig is imported
        # Or return a dictionary representation for llm_handler to use.
        # Let's return a dict for now to keep this file independent.
        if self.method == "none" or self.method == "16bit":
            return None
        elif self.method == "4bit":
            return {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": self.compute_dtype,
                "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
                "bnb_4bit_quant_type": self.quant_type
            }
        elif self.method == "8bit":
            return {
                "load_in_8bit": True,
                "llm_int8_threshold": self.llm_int8_threshold,
                "llm_int8_skip_modules": self.llm_int8_skip_modules,
                "llm_int8_enable_fp32_cpu_offload": self.llm_int8_enable_fp32_cpu_offload,
                "llm_int8_has_fp16_weight": self.llm_int8_has_fp16_weight
            }
        # Consider logging a warning if method is unknown, but avoid logger import here
        print(f"Warning: Unknown quantization method '{self.method}' in QuantizationConfig. Returning None.")
        return None


@dataclass
class LayerSplitConfig:
    """Configuration for splitting model layers between devices"""
    enabled: bool = ENABLE_LAYER_SPLITTING
    gpu_layers: int = DEFAULT_GPU_LAYERS  # -1 means auto-detect (GGUF uses this, HF Transformers uses device_map)
    max_gpu_memory: Dict[int, str] = None  # e.g., {0: "10GB", 1: "10GB"}
    max_cpu_memory: str = "30GB"
    offload_folder: Optional[Path] = Path(OFFLOAD_DIR)
    offload_state_dict: bool = False

    def __post_init__(self):
        # Avoid torch import here if possible. Let the user (e.g., MenuSystem) handle this.
        # We'll set a basic default, assuming at least one GPU might exist.
        if self.max_gpu_memory is None:
            self.max_gpu_memory = {0: "10GB"} # Simple default

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
    device_map: Optional[Dict] = None


@dataclass
class PipelineConfig:
    """Configuration for the processing pipeline"""
    mode: ProcessingMode = ProcessingMode.PODCAST
    model_name: str = DEFAULT_MODEL # This is now for metadata (e.g., "local:qwen3-4b")
    model_provider: str = "local" # 'local', 'local_gguf', 'openai', 'google', etc.
    model_specifier: str = DEFAULT_MODEL # The actual model ID or path
    memory_profile: str = "medium_vram" # Used to configure local HF models (potentially redundant)
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY
    chunk_size: int = 1000 # Use a default, can be overridden
    markdown_style: str = "podcast"
    system_prompt: Optional[str] = None
    remove_thinking: bool = True
    preserve_layout: bool = True # Corresponds to PDFProcessor setting
    clean_for_audio: bool = True
    add_emotions: bool = True # Corresponds to MarkdownFormatter setting
    enable_checkpoints: bool = ENABLE_STAGE_CHECKPOINTS
    max_retries: int = MAX_RETRIES
    output_format: str = "markdown"
    hyperparameters: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    # Configs for local models are now separate, passed during backend creation
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
