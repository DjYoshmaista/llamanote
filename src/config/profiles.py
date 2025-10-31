# llamanote/config/profiles.py
"""
Memory Profile Management for LlamaNote Enhanced
Defines memory profiles and provides functions to create hardware configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, TYPE_CHECKING

# Use TYPE_CHECKING to avoid circular import at module load time
if TYPE_CHECKING:
    from ..core.types import QuantizationConfig, LayerSplitConfig

from .settings import DEFAULT_QUANTIZATION, DEFAULT_GPU_LAYERS, QUANTIZATION_OPTIONS

# Attempt optional import for torch types
try:
    import torch
    TorchDtype = torch.dtype
except ImportError:
    torch = None
    TorchDtype = None # Use None or Any as fallback


@dataclass
class MemoryProfile:
    """Represents a memory usage profile with associated hardware settings."""
    name: str
    description: str
    # Quantization settings (can map directly to QuantizationConfig fields)
    quantization_method: str = DEFAULT_QUANTIZATION
    compute_dtype_str: str = "bfloat16" # Store as string for flexibility
    # Layer splitting settings (can map directly to LayerSplitConfig fields)
    gpu_layers: int = DEFAULT_GPU_LAYERS # Primarily for GGUF hint
    max_gpu_memory_per_device: str = "10GB" # For Transformers
    max_cpu_memory: str = "30GB" # For Transformers
    enable_splitting: bool = True # General flag

    def get_quantization_config(self) -> 'QuantizationConfig':
        """Creates a QuantizationConfig based on the profile."""
        # Local import to avoid circular dependency
        from ..core.types import QuantizationConfig
        compute_dtype: Optional[TorchDtype] = None
        if torch:
            dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
            compute_dtype = dtype_map.get(self.compute_dtype_str)

        # Ensure quantization method is valid
        method = self.quantization_method if self.quantization_method in QUANTIZATION_OPTIONS else "none"

        return QuantizationConfig(
            method=method,
            compute_dtype=compute_dtype,
            use_double_quant=(method == "4bit"),
            quant_type="nf4" # Default quant_type
        )

    def get_layer_split_config(self) -> 'LayerSplitConfig':
        """Creates a LayerSplitConfig based on the profile."""
        # Local import to avoid circular dependency
        from ..core.types import LayerSplitConfig
        max_gpu_mem_dict: Dict[int, str] = {}
        cuda_available = False
        num_gpus = 0

        if torch:
            try:
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    num_gpus = torch.cuda.device_count()
                    max_gpu_mem_dict = {i: self.max_gpu_memory_per_device for i in range(num_gpus)}
            except Exception:
                cuda_available = False # Treat error as no CUDA

        # Splitting is enabled only if requested AND CUDA is available AND not CPU only profile
        effective_enable_splitting = self.enable_splitting and cuda_available and self.max_gpu_memory_per_device != "0GB"

        # Determine gpu_layers: use profile value if splitting enabled, else 0
        effective_gpu_layers = self.gpu_layers if effective_enable_splitting else 0
        if not cuda_available:
             effective_gpu_layers = 0 # Force 0 if no CUDA detected

        return LayerSplitConfig(
            enabled=effective_enable_splitting,
            gpu_layers=effective_gpu_layers,
            max_gpu_memory=max_gpu_mem_dict,
            max_cpu_memory=self.max_cpu_memory
            # offload_folder and offload_state_dict could be added to profile if needed
        )

# Define the standard memory profiles using the dataclass
# This replaces the old MEMORY_PROFILES dictionary
_DEFINED_PROFILES: Dict[str, MemoryProfile] = {
    "low_vram": MemoryProfile(
        name="low_vram",
        description="Optimized for GPUs with ~4-6GB VRAM. Uses 4-bit quantization.",
        quantization_method="4bit",
        max_gpu_memory_per_device="4GB",
        max_cpu_memory="16GB",
        gpu_layers=-1 # GGUF hint: try to offload all possible
    ),
    "medium_vram": MemoryProfile(
        name="medium_vram",
        description="Balanced profile for GPUs with ~8-12GB VRAM. Uses default quantization.",
        quantization_method=DEFAULT_QUANTIZATION, # Typically 4bit or 8bit
        max_gpu_memory_per_device="10GB",
        max_cpu_memory="30GB",
        gpu_layers=DEFAULT_GPU_LAYERS # GGUF hint: use default auto
    ),
    "high_vram": MemoryProfile(
        name="high_vram",
        description="For GPUs with 16GB+ VRAM. Less aggressive quantization.",
        quantization_method="8bit", # Or "16bit"/"none" depending on preference
        compute_dtype_str="bfloat16", # Use higher precision if less quant
        max_gpu_memory_per_device="20GB",
        max_cpu_memory="64GB",
        gpu_layers=-1 # GGUF hint: try to offload all possible
    ),
    "cpu_only": MemoryProfile(
        name="cpu_only",
        description="Runs entirely on CPU. No quantization by default.",
        quantization_method="none",
        compute_dtype_str="float32", # CPU often prefers float32
        max_gpu_memory_per_device="0GB", # Explicitly no GPU memory
        max_cpu_memory="64GB", # Allow ample RAM
        gpu_layers=0, # GGUF: Force CPU
        enable_splitting=False # Disable splitting logic
    ),
}

def get_memory_profile(name: str) -> Optional[MemoryProfile]:
    """Retrieves a memory profile by name."""
    return _DEFINED_PROFILES.get(name.lower())

def list_memory_profiles() -> List[Tuple[str, str]]:
    """Returns a list of available memory profile names and descriptions."""
    return [(name, profile.description) for name, profile in _DEFINED_PROFILES.items()]

def create_configs_from_memory_profile(profile_name: str) -> Tuple[Optional['QuantizationConfig'], Optional['LayerSplitConfig']]:
    """
    Factory function to create QuantizationConfig and LayerSplitConfig from a profile name.

    Args:
        profile_name: The name of the memory profile (e.g., "low_vram").

    Returns:
        A tuple containing (QuantizationConfig, LayerSplitConfig), or (None, None) if profile not found.
    """
    # Local import to avoid circular dependency
    from ..core.types import QuantizationConfig, LayerSplitConfig

    profile = get_memory_profile(profile_name)
    if profile:
        return profile.get_quantization_config(), profile.get_layer_split_config()
    else:
        # Log warning (needs logger import)
        # logger.warning(f"Memory profile '{profile_name}' not found. Using defaults.")
        # Fallback to default configs (e.g., medium profile or create directly)
        default_profile = get_memory_profile("medium_vram") # Or define standalone defaults
        if default_profile:
             return default_profile.get_quantization_config(), default_profile.get_layer_split_config()
        else:
             # Absolute fallback if even "medium_vram" is missing
             return QuantizationConfig(), LayerSplitConfig() # Basic defaults
