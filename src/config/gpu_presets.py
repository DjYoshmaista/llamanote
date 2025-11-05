# llamanote/config/gpu_presets.py
"""
GPU Configuration Presets
Pre-configured settings optimized for different GPU memory sizes.
"""

import torch
from pathlib import Path
from typing import Tuple
from ..core.types import QuantizationConfig, LayerSplitConfig
from .settings import DEFAULT_OFFLOAD_DIR


def get_preset_for_vram(vram_gb: float) -> Tuple[QuantizationConfig, LayerSplitConfig, str]:
    """
    Get optimal preset configuration based on available VRAM.

    Args:
        vram_gb: Available VRAM in gigabytes

    Returns:
        Tuple of (QuantizationConfig, LayerSplitConfig, recommended_model_key)
    """
    if vram_gb < 2:
        return preset_2gb()
    elif vram_gb < 6:
        return preset_4gb()
    elif vram_gb < 12:
        return preset_8gb()
    else:
        return preset_12gb_plus()


def preset_2gb() -> Tuple[QuantizationConfig, LayerSplitConfig, str]:
    """
    Preset for 2GB VRAM GPUs (GTX 1050, etc.)
    - Use smallest models only
    - Aggressive quantization
    """
    quant_config = QuantizationConfig(
        method="4bit",
        compute_dtype=torch.float16,
        use_double_quant=True,
        quant_type="nf4"
    )

    split_config = LayerSplitConfig(
        enabled=True,
        max_gpu_memory={0: "1.5GB"},
        max_cpu_memory="28GB",
        offload_folder=DEFAULT_OFFLOAD_DIR,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        auto_oom_handling=True
    )

    recommended_model = "gemma3-270m"
    return quant_config, split_config, recommended_model


def preset_4gb() -> Tuple[QuantizationConfig, LayerSplitConfig, str]:
    """
    Preset for 4GB VRAM GPUs (GTX 1650 Ti, RTX 3050, etc.)
    - Optimized for DeepSeek-R1-Distill-Qwen-1.5B
    - 4-bit quantization required
    - Conservative memory limits with disk offload fallback
    """
    quant_config = QuantizationConfig(
        method="4bit",
        compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_double_quant=True,
        quant_type="nf4"
    )

    split_config = LayerSplitConfig(
        enabled=True,
        max_gpu_memory={0: "3.1GB"},  # Conservative limit leaving headroom
        max_cpu_memory="28GB",
        offload_folder=DEFAULT_OFFLOAD_DIR,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        auto_oom_handling=True
    )

    recommended_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    return quant_config, split_config, recommended_model


def preset_8gb() -> Tuple[QuantizationConfig, LayerSplitConfig, str]:
    """
    Preset for 8GB VRAM GPUs (RTX 3060, RTX 3070, etc.)
    - Can handle 4B models with 4-bit quantization
    - Good balance of speed and quality
    """
    quant_config = QuantizationConfig(
        method="4bit",
        compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_double_quant=True,
        quant_type="nf4"
    )

    split_config = LayerSplitConfig(
        enabled=True,
        max_gpu_memory={0: "7GB"},
        max_cpu_memory="28GB",
        offload_folder=DEFAULT_OFFLOAD_DIR,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        auto_oom_handling=True
    )

    recommended_model = "qwen3-4b"
    return quant_config, split_config, recommended_model


def preset_12gb_plus() -> Tuple[QuantizationConfig, LayerSplitConfig, str]:
    """
    Preset for 12GB+ VRAM GPUs (RTX 3080, RTX 4080, etc.)
    - Can handle 7B models with 4-bit quantization
    - Or 4B models with 8-bit/16-bit for better quality
    """
    quant_config = QuantizationConfig(
        method="4bit",  # Still recommended for larger models
        compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        use_double_quant=True,
        quant_type="nf4"
    )

    split_config = LayerSplitConfig(
        enabled=True,
        max_gpu_memory={0: "11GB"},
        max_cpu_memory="28GB",
        offload_folder=DEFAULT_OFFLOAD_DIR,
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        auto_oom_handling=True
    )

    recommended_model = "qwen3-4b"  # Can upgrade to 7B models if available
    return quant_config, split_config, recommended_model


def preset_cpu_only() -> Tuple[QuantizationConfig, LayerSplitConfig, str]:
    """
    Preset for CPU-only inference (no GPU or when GPU fails)
    - No quantization
    - Full CPU memory
    - Will be slow but functional
    """
    quant_config = QuantizationConfig(
        method="none",
        compute_dtype=torch.float32
    )

    split_config = LayerSplitConfig(
        enabled=True,
        max_gpu_memory={0: "0GB"},
        max_cpu_memory="28GB",
        offload_state_dict=False,
        low_cpu_mem_usage=False,
        auto_oom_handling=False
    )

    recommended_model = "gemma3-270m"
    return quant_config, split_config, recommended_model


def auto_detect_preset() -> Tuple[QuantizationConfig, LayerSplitConfig, str]:
    """
    Automatically detect GPU VRAM and return optimal preset.

    Returns:
        Tuple of (QuantizationConfig, LayerSplitConfig, recommended_model_key)
    """
    if not torch.cuda.is_available():
        return preset_cpu_only()

    try:
        props = torch.cuda.get_device_properties(0)
        total_vram_gb = props.total_memory / (1024**3)
        return get_preset_for_vram(total_vram_gb)
    except Exception:
        # Fallback to conservative preset
        return preset_4gb()


# Quick access to specific presets
PRESETS = {
    "2gb": preset_2gb,
    "4gb": preset_4gb,
    "8gb": preset_8gb,
    "12gb": preset_12gb_plus,
    "cpu": preset_cpu_only,
    "auto": auto_detect_preset
}


def get_preset(preset_name: str = "auto") -> Tuple[QuantizationConfig, LayerSplitConfig, str]:
    """
    Get a preset configuration by name.

    Args:
        preset_name: One of "2gb", "4gb", "8gb", "12gb", "cpu", or "auto"

    Returns:
        Tuple of (QuantizationConfig, LayerSplitConfig, recommended_model_key)

    Example:
        >>> quant, split, model = get_preset("4gb")
        >>> # Use with model loading...
    """
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")

    return PRESETS[preset_name]()


__all__ = [
    'get_preset',
    'get_preset_for_vram',
    'auto_detect_preset',
    'preset_2gb',
    'preset_4gb',
    'preset_8gb',
    'preset_12gb_plus',
    'preset_cpu_only',
    'PRESETS'
]
