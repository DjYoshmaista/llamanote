#!/usr/bin/env python3
"""
GPU Memory Optimizer for 4GB VRAM
Configures llamanote for optimal performance on GTX 1650 Ti (4GB VRAM)
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.types import QuantizationConfig, LayerSplitConfig
from src.config.settings import DEFAULT_CACHE_DIR, DEFAULT_OFFLOAD_DIR


def get_gpu_info():
    """Get GPU memory information."""
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    free_mem = torch.cuda.mem_get_info()[0]
    total_mem = props.total_memory

    return {
        'name': props.name,
        'total_gb': total_mem / (1024**3),
        'free_gb': free_mem / (1024**3),
        'compute_capability': f"{props.major}.{props.minor}",
        'supports_bf16': torch.cuda.is_bf16_supported()
    }


def get_optimal_config(vram_gb: float, target_model_size: str = "1.5B"):
    """
    Generate optimal configuration based on available VRAM.

    Args:
        vram_gb: Available VRAM in GB
        target_model_size: Target model size ("270M", "1.5B", "4B")

    Returns:
        tuple: (QuantizationConfig, LayerSplitConfig, recommended_model)
    """

    if target_model_size == "270M" or vram_gb < 2:
        # Small model - can use FP16
        quant_config = QuantizationConfig(
            method="none",
            compute_dtype=torch.float16
        )
        split_config = LayerSplitConfig(
            enabled=True,
            max_gpu_memory={0: f"{max(1.5, vram_gb - 0.5):.1f}GB"},
            max_cpu_memory="28GB",
            offload_state_dict=False,
            low_cpu_mem_usage=True,
            auto_oom_handling=True
        )
        recommended_model = "gemma3-270m"

    elif target_model_size == "1.5B" or (2 <= vram_gb < 6):
        # Medium model - need 4-bit quantization
        quant_config = QuantizationConfig(
            method="4bit",
            compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            use_double_quant=True,
            quant_type="nf4"
        )
        split_config = LayerSplitConfig(
            enabled=True,
            max_gpu_memory={0: f"{max(2.5, vram_gb - 0.5):.1f}GB"},
            max_cpu_memory="28GB",
            offload_folder=DEFAULT_OFFLOAD_DIR,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            auto_oom_handling=True
        )
        recommended_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    elif target_model_size == "4B" or vram_gb >= 6:
        # Large model - aggressive 4-bit + offloading
        quant_config = QuantizationConfig(
            method="4bit",
            compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            use_double_quant=True,
            quant_type="nf4"
        )
        split_config = LayerSplitConfig(
            enabled=True,
            max_gpu_memory={0: f"{vram_gb - 1.0:.1f}GB"},
            max_cpu_memory="28GB",
            offload_folder=DEFAULT_OFFLOAD_DIR,
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            auto_oom_handling=True
        )
        recommended_model = "qwen3-4b"
    else:
        # Fallback to smallest model
        quant_config = QuantizationConfig(method="4bit")
        split_config = LayerSplitConfig(
            enabled=True,
            max_gpu_memory={0: "2GB"},
            auto_oom_handling=True
        )
        recommended_model = "gemma3-270m"

    return quant_config, split_config, recommended_model


def print_config_code(quant_config, split_config, model_name):
    """Print Python code to use the configuration."""
    print("\n" + "="*70)
    print("OPTIMAL CONFIGURATION FOR YOUR GPU")
    print("="*70)
    print("\nAdd this to your model initialization code:\n")

    print("```python")
    print("from src.core.types import QuantizationConfig, LayerSplitConfig")
    print("import torch")
    print()
    print("# Quantization Configuration")
    print(f"quant_config = QuantizationConfig(")
    print(f"    method='{quant_config.method}',")
    if quant_config.compute_dtype:
        dtype_str = str(quant_config.compute_dtype).replace("torch.", "torch.")
        print(f"    compute_dtype={dtype_str},")
    if quant_config.method == "4bit":
        print(f"    use_double_quant={quant_config.use_double_quant},")
        print(f"    quant_type='{quant_config.quant_type}'")
    print(")")
    print()
    print("# Layer Split Configuration")
    print(f"split_config = LayerSplitConfig(")
    print(f"    enabled={split_config.enabled},")
    print(f"    max_gpu_memory={split_config.max_gpu_memory},")
    print(f"    max_cpu_memory='{split_config.max_cpu_memory}',")
    if split_config.offload_folder:
        print(f"    offload_folder=Path('{split_config.offload_folder}'),")
    print(f"    offload_state_dict={split_config.offload_state_dict},")
    print(f"    low_cpu_mem_usage={split_config.low_cpu_mem_usage},")
    print(f"    auto_oom_handling={split_config.auto_oom_handling}")
    print(")")
    print()
    print(f"# Recommended Model: {model_name}")
    print("```")
    print()


def test_configuration(quant_config, split_config):
    """Test if the configuration will likely work."""
    print("\n" + "="*70)
    print("CONFIGURATION TEST")
    print("="*70)

    checks = []

    # Check 1: CUDA available
    cuda_available = torch.cuda.is_available()
    checks.append(("CUDA Available", cuda_available, "Required for GPU acceleration"))

    # Check 2: Quantization method valid
    valid_quant = quant_config.method in ["none", "4bit", "8bit", "16bit"]
    checks.append(("Valid Quantization", valid_quant, f"Method: {quant_config.method}"))

    # Check 3: Memory limits reasonable
    max_gpu = split_config.max_gpu_memory.get(0, "0GB")
    gpu_gb = float(max_gpu.replace("GB", ""))
    reasonable_memory = 0.5 <= gpu_gb <= 6.0
    checks.append(("Reasonable GPU Limit", reasonable_memory, f"Set to: {max_gpu}"))

    # Check 4: BitsAndBytes available for quantization
    if quant_config.method in ["4bit", "8bit"]:
        try:
            import bitsandbytes
            bnb_available = True
        except ImportError:
            bnb_available = False
        checks.append(("BitsAndBytes Available", bnb_available,
                      "Required for 4bit/8bit quantization"))

    # Check 5: Offload directory writable
    if split_config.offload_folder:
        try:
            split_config.offload_folder.mkdir(parents=True, exist_ok=True)
            offload_writable = True
        except Exception:
            offload_writable = False
        checks.append(("Offload Directory", offload_writable,
                      f"Path: {split_config.offload_folder}"))

    # Print results
    print()
    all_passed = True
    for name, passed, details in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} | {name:25} | {details}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("✓ All checks passed! Configuration should work.")
    else:
        print("✗ Some checks failed. Review the issues above.")

    return all_passed


def main():
    print("="*70)
    print("GPU MEMORY OPTIMIZER FOR LLAMANOTE")
    print("="*70)

    # Get GPU info
    gpu_info = get_gpu_info()

    if gpu_info is None:
        print("\n✗ ERROR: No CUDA-capable GPU detected!")
        print("This optimizer requires an NVIDIA GPU with CUDA support.")
        print("\nFalling back to CPU-only configuration...")

        quant_config = QuantizationConfig(method="none", compute_dtype=torch.float32)
        split_config = LayerSplitConfig(
            enabled=True,
            max_gpu_memory={0: "0GB"},
            max_cpu_memory="28GB",
            auto_oom_handling=False
        )
        recommended_model = "gemma3-270m"

        print_config_code(quant_config, split_config, recommended_model)
        return

    # Print GPU info
    print(f"\nDetected GPU: {gpu_info['name']}")
    print(f"Total VRAM: {gpu_info['total_gb']:.2f} GB")
    print(f"Free VRAM: {gpu_info['free_gb']:.2f} GB")
    print(f"Compute Capability: {gpu_info['compute_capability']}")
    print(f"BF16 Support: {'Yes' if gpu_info['supports_bf16'] else 'No'}")

    # Determine target model size
    vram_gb = gpu_info['total_gb']

    if vram_gb < 2:
        target_size = "270M"
        print(f"\n⚠ WARNING: Only {vram_gb:.1f}GB VRAM detected.")
        print("Recommending smallest model (270M parameters).")
    elif vram_gb < 6:
        target_size = "1.5B"
        print(f"\n⚠ Limited VRAM ({vram_gb:.1f}GB) detected.")
        print("Recommending 1.5B model with 4-bit quantization.")
    else:
        target_size = "4B"
        print(f"\n✓ Good VRAM ({vram_gb:.1f}GB) detected.")
        print("Can handle larger models with 4-bit quantization.")

    # Get optimal configuration
    quant_config, split_config, recommended_model = get_optimal_config(vram_gb, target_size)

    # Print configuration
    print_config_code(quant_config, split_config, recommended_model)

    # Test configuration
    test_configuration(quant_config, split_config)

    # Additional recommendations
    print("\n" + "="*70)
    print("ADDITIONAL RECOMMENDATIONS")
    print("="*70)
    print()

    if vram_gb <= 4:
        print("• Close other GPU-accelerated applications before loading models")
        print("• Monitor GPU memory with: nvidia-smi -l 1")
        print("• Consider using smaller models for faster inference")

    if quant_config.method == "4bit":
        print("• 4-bit quantization provides ~3x memory savings")
        print("• Expect minor quality degradation (<5% typically)")

    if split_config.offload_folder:
        print(f"• Disk offloading enabled at: {split_config.offload_folder}")
        print("• Ensure sufficient disk space (5-10GB recommended)")

    print()
    print("For more details, see: GPU_OPTIMIZATION_GUIDE.md")
    print()


if __name__ == "__main__":
    main()
