# llamanote/core/types.py
"""
Core Data Structures for LlamaNote Enhanced
Defines shared Enums and Dataclasses used across the pipeline and components.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from ..config.settings import DEFAULT_PIPELINE_STAGES

# Attempt import for type hint, but make it optional
try:
    import torch
    TorchDtype = torch.dtype
except ImportError:
    TorchDtype = Any # Fallback type if torch is not installed

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from ..models.hyperparameters import HyperparameterConfig

# --- Enums ---

class ProcessingMode(Enum):
    """Modes defining the overall goal of the processing pipeline."""
    PODCAST = "podcast"
    TECHNICAL = "technical"
    NARRATIVE = "narrative"
    SUMMARY = "summary"
    CUSTOM = "custom" # For user-defined prompts/styles

class ChunkingStrategy(Enum):
    """Strategies for dividing text into smaller chunks."""
    WORD_BOUNDARY = "word_boundary"
    SENTENCE_BOUNDARY = "sentence_boundary"
    PARAGRAPH_BOUNDARY = "paragraph_boundary"
    # SEMANTIC = "semantic" # Often requires additional models/complexity
    SLIDING_WINDOW = "sliding_window"
    # TOKEN_BASED = "token_based" # Requires tokenizer access

# --- Model Configuration Types ---

@dataclass
class QuantizationConfig:
    """Configuration for model quantization (primarily Transformers/BnB)."""
    method: str = "none"  # "none", "4bit", "8bit", "16bit" (fp16/bf16)
    compute_dtype: Optional[TorchDtype] = None # e.g., torch.bfloat16, torch.float16
    use_double_quant: bool = True  # Specific to 4bit BnB
    quant_type: str = "nf4"      # Specific to 4bit BnB ("nf4" or "fp4")
    # Add other BnB params if needed, e.g., llm_int8_threshold

    def __post_init__(self):
        # Set default compute_dtype based on method if not provided
        if self.compute_dtype is None:
            if self.method in ["4bit", "8bit", "16bit"]:
                try:
                    import torch
                    # Default to bfloat16 if available and supported, else float16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                         self.compute_dtype = torch.bfloat16
                    else:
                         self.compute_dtype = torch.float16
                except ImportError:
                    pass # Keep as None if torch not installed

    def to_bnb_config(self) -> Optional[Any]:
        """Convert to Hugging Face BitsAndBytesConfig if applicable."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            return None # Transformers not available

        if self.method == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
                bnb_4bit_use_double_quant=self.use_double_quant,
                bnb_4bit_quant_type=self.quant_type
            )
        elif self.method == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True
                # Add llm_int8 specific params here if needed
            )
        # 16bit doesn't use BnB config, handled via torch_dtype directly
        return None


@dataclass
class LayerSplitConfig:
    """Configuration for splitting model layers between devices (GPU/CPU/Disk)."""
    enabled: bool = True # Whether to attempt splitting
    gpu_layers: int = -1  # GGUF: Num layers on GPU (-1=all). Transformers: Hint (less effective).
    max_gpu_memory: Dict[int, str] = field(default_factory=lambda: {0: "4GB"}) # Transformers: Max mem per GPU (default: 4GB for dedicated GPU)
    max_cpu_memory: str = "28GB" # Transformers: Max RAM for offload (leave some for OS)
    offload_folder: Optional[Path] = None # Transformers: Disk offload directory
    offload_state_dict: bool = True # Transformers: Use state_dict for offloading (more memory efficient)
    low_cpu_mem_usage: bool = True # Enable low CPU memory usage mode
    auto_oom_handling: bool = True # Enable automatic CUDA OOM error handling with progressive layer offloading

    def get_max_memory_dict(self) -> Dict[Union[int, str], str]:
        """Formats memory limits for Transformers device_map='auto'."""
        max_memory = {}
        # Ensure GPU device keys are integers, not strings
        for k, v in self.max_gpu_memory.items():
            max_memory[int(k)] = v  # Convert to int in case it's a string
        max_memory["cpu"] = self.max_cpu_memory
        return max_memory


# --- Processing Component Result Types ---

@dataclass
class PDFMetadata:
    """Metadata extracted from a PDF document."""
    file_path: Path
    num_pages: int
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    file_size_mb: float = 0.0
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractionResult:
    """Result from PDF or text file extraction/reading stage."""
    text: str
    metadata: Optional[PDFMetadata] # None for non-PDF inputs
    page_texts: List[str] = field(default_factory=list) # Relevant for PDFs
    extraction_method: str = "unknown"
    warnings: List[str] = field(default_factory=list)
    char_count: int = 0
    word_count: int = 0

    def __post_init__(self):
        # Calculate counts if not provided
        if self.char_count == 0 and self.text:
            self.char_count = len(self.text)
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())


@dataclass
class TextChunk:
    """Represents a single chunk of text with metadata."""
    text: str
    index: int = 0
    start_pos: int = 0 # Character start position in original text
    end_pos: int = 0   # Character end position in original text
    word_count: int = 0
    char_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict) # For strategy, etc.

    def __post_init__(self):
        if self.char_count == 0 and self.text:
            self.char_count = len(self.text)
        if self.word_count == 0 and self.text:
            self.word_count = len(self.text.split())
        if self.end_pos == 0 and self.start_pos >= 0 and self.char_count > 0:
             self.end_pos = self.start_pos + self.char_count


@dataclass
class ChunkingResult:
    """Result from the text chunking stage."""
    chunks: List[TextChunk]
    total_chunks: int = 0
    strategy_used: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY
    average_chunk_size: float = 0.0
    overlap_used: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict) # e.g., original length

    def __post_init__(self):
        if not self.chunks: return
        self.total_chunks = len(self.chunks)
        total_chars = sum(c.char_count for c in self.chunks)
        self.average_chunk_size = total_chars / self.total_chunks if self.total_chunks > 0 else 0.0


@dataclass
class FilterResult:
    """Result from the response filtering stage."""
    original_text: str
    filtered_text: str
    removed_segments: List[str] = field(default_factory=list) # Descriptions of removed parts
    filter_stats: Dict[str, int] = field(default_factory=dict) # e.g., {'thinking_segments': 5}

    @property
    def removal_ratio(self) -> float:
        """Ratio of characters removed (0.0 to 1.0)."""
        if not self.original_text: return 0.0
        return 1.0 - (len(self.filtered_text) / len(self.original_text))


@dataclass
class GenerationResult:
    """Standardized result from any LLM backend generation."""
    raw_output: str          # The direct output from the model
    filtered_output: str     # Output after applying filters (e.g., removing <think>)
    input_tokens: int        # Number of tokens in the input prompt
    output_tokens: int       # Number of tokens generated by the model
    generation_time: float   # Time taken for generation in seconds
    memory_used: float = 0.0 # Estimated peak memory usage in MB (primarily for local models)
    device_map: Optional[Dict[str, Any]] = None # Info about device placement (local) or provider
    error_message: Optional[str] = None # If an error occurred during generation


@dataclass
class AudioConfig:
    """Configuration specific to audio generation."""
    # model_id/specifier is handled by AppState/PipelineConfig
    speaker_embedding: Optional[str] = None # Path or ID for local models like SpeechT5
    sample_rate: int = 24000 # Target sample rate (common for modern TTS)
    output_format: str = "wav" # "wav", "mp3", "flac"
    chunk_size: int = 1500  # Chars per chunk for local TTS (reduced for 4GB VRAM compatibility)
    speed: float = 1.0 # Playback speed factor (applied via API or post-processing)
    pitch_shift: int = 0  # Semitones for pitch shift (post-processing only)
    volume_normalize: bool = True # Apply normalization (post-processing)
    device: str = "auto"  # "auto", "cpu", "cuda" (primarily local models)
    use_half_precision: bool = True # Use float16/bfloat16 (local CUDA models)
    # Cloud specific settings
    cloud_voice: str = "alloy" # Default voice for providers like OpenAI
    quantization: str = "4bit" # "none", "4bit", "8bit" (default 4bit for memory efficiency)
    # Memory optimization
    enable_cpu_offload: bool = True # Enable CPU offloading for large models
    enable_disk_offload: bool = False # Enable disk offloading (slower but saves RAM)
    clear_cache_between_chunks: bool = True # Clear CUDA cache between audio chunks

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Need to handle potential non-serializable types if any added later
        return asdict(self)

@dataclass
class AudioResult:
    """Result from the audio generation stage."""
    audio_path: Path             # Path to the saved audio file
    duration_seconds: float      # Duration of the generated audio
    sample_rate: int             # Actual sample rate of the saved file
    num_samples: int             # Total number of audio samples
    model_used: str              # Identifier (e.g., "local:microsoft/speecht5_tts")
    processing_time: float       # Time taken for generation and saving
    chunks_processed: int        # Number of text chunks processed by TTS

    @property
    def duration_formatted(self) -> str:
        """Formatted duration string (MM:SS)."""
        minutes = int(self.duration_seconds // 60)
        seconds = int(self.duration_seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


# --- Pipeline Configuration and Result Types ---

@dataclass
class PipelineConfig:
    """Complete configuration for a processing pipeline run."""
    # Core settings
    mode: ProcessingMode = ProcessingMode.PODCAST
    model_provider: str = "local_hf" # "local_hf", "local_gguf", "openai", etc.
    model_specifier: str = "qwen3-4b" # HF ID, GGUF path, or Cloud model name
    system_prompt: Optional[str] = None # Overrides mode default
    stages: Optional[List[str]] = None # Stages to run, None means all defaults

    # Input/Output
    output_format: str = "markdown"
    output_dir: Optional[Path] = None # Overrides default output dir
    timestamp_outputs: bool = True
    include_metadata: bool = True

    # Processing details
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.WORD_BOUNDARY
    chunk_size: int = 1000
    chunk_overlap: int = 50 # Added from text_processor defaults
    preserve_pdf_layout: bool = True # From pdf_processor
    clean_for_audio: bool = True # From text_preprocessor/pipeline

    # Filtering/Formatting
    remove_thinking: bool = True
    markdown_style: Optional[str] = None # If None, derived from mode
    add_emotions: bool = True # For formatting

    # LLM Generation settings
    hyperparameters: Any = field(default=None) # Will be lazily initialized
    # Local Model Hardware settings
    quantization_config: Optional[QuantizationConfig] = None
    layer_split_config: Optional[LayerSplitConfig] = None

    # Audio Generation settings (if applicable)
    generate_audio: bool = False
    audio_config: Optional[AudioConfig] = None
    audio_provider: Optional[str] = None
    audio_specifier: Optional[str] = None

    # Pipeline execution settings
    enable_checkpoints: bool = True
    max_retries: int = 3
    fallback_on_error: bool = True

    @property
    def model_name_for_metadata(self) -> str:
        """Generates a string representation for metadata."""
        # Shorten GGUF path for metadata
        specifier_display = self.model_specifier
        if self.model_provider == 'local_gguf':
            specifier_display = Path(self.model_specifier).name
        return f"{self.model_provider}:{specifier_display}"

    def get_hyperparameters(self) -> 'HyperparameterConfig':
        """Lazily loads and returns HyperparameterConfig only when needed."""
        if self.hyperparameters is None:
            # Local import to avoid circular dependency
            from ..models.hyperparameters import HyperparameterConfig
            self.hyperparameters = HyperparameterConfig()
        return self.hyperparameters

    def __post_init__(self):
        # Ensure default dataclasses are created if None
        if self.quantization_config is None: self.quantization_config = QuantizationConfig()
        if self.layer_split_config is None: self.layer_split_config = LayerSplitConfig()
        if self.audio_config is None: self.audio_config = AudioConfig()
        if not self.stages: self.stages = list(DEFAULT_PIPELINE_STAGES) # Use constant
        # hyperparameters is now lazily initialized via get_hyperparameters() method


@dataclass
class PipelineResult:
    """Result from a full pipeline run for a single file."""
    success: bool
    input_file: Path
    output_file: Optional[Path] = None # Path to the final text output (e.g., .md)
    audio_file: Optional[Path] = None # Path to the final audio output (e.g., .wav)
    processing_time: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    statistics: Dict[str, Any] = field(default_factory=dict) # Chunks, timings per stage, etc.

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary for reporting."""
        d = asdict(self)
        d["input_file"] = str(self.input_file)
        d["output_file"] = str(self.output_file) if self.output_file else None
        d["audio_file"] = str(self.audio_file) if self.audio_file else None
        return d


@dataclass
class ProcessedFile:
    """Information about a processed file for tracking and reporting."""
    input_path: Path
    output_path: Optional[Path]
    format: str
    timestamp: str
    processing_time: float
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
