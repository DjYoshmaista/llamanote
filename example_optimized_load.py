#!/usr/bin/env python3
"""
Example: Loading Models with Optimized Configuration for 4GB GPU

This script demonstrates how to load models using the optimized configuration
for your GTX 1650 Ti (4GB VRAM).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config.gpu_presets import get_preset, auto_detect_preset
from src.models.backends.local_hf import LocalModelLoader
from src.config.manager import ConfigManager
from src.utils.logger import get_logger_conf

logger = get_logger_conf(__name__)


def example_1_auto_detect():
    """Example 1: Automatically detect GPU and use optimal preset."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Auto-Detect GPU Configuration")
    print("="*70 + "\n")

    # Auto-detect optimal configuration
    quant_config, split_config, recommended_model = auto_detect_preset()

    print(f"Auto-detected configuration:")
    print(f"  - Quantization: {quant_config.method}")
    print(f"  - GPU Memory Limit: {split_config.max_gpu_memory}")
    print(f"  - Recommended Model: {recommended_model}")
    print()

    return quant_config, split_config, recommended_model


def example_2_specific_preset():
    """Example 2: Use specific preset for 4GB GPU."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Using 4GB GPU Preset")
    print("="*70 + "\n")

    # Get 4GB preset explicitly
    quant_config, split_config, recommended_model = get_preset("4gb")

    print(f"4GB GPU configuration:")
    print(f"  - Quantization: {quant_config.method}")
    print(f"  - Compute dtype: {quant_config.compute_dtype}")
    print(f"  - Double quantization: {quant_config.use_double_quant}")
    print(f"  - Quantization type: {quant_config.quant_type}")
    print(f"  - GPU Memory Limit: {split_config.max_gpu_memory}")
    print(f"  - CPU Memory Limit: {split_config.max_cpu_memory}")
    print(f"  - Offload to disk: {split_config.offload_folder is not None}")
    print(f"  - Auto OOM handling: {split_config.auto_oom_handling}")
    print(f"  - Recommended Model: {recommended_model}")
    print()

    return quant_config, split_config, recommended_model


def example_3_load_model():
    """Example 3: Actually load a model with optimized config."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Loading Model with Optimized Configuration")
    print("="*70 + "\n")

    # Get configuration
    quant_config, split_config, recommended_model = get_preset("4gb")

    # For this example, we'll use a smaller model that will definitely work
    # Change to "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" if you want to try 1.5B
    model_id = "google/gemma-3-270m"

    print(f"Loading model: {model_id}")
    print(f"This may take a few minutes on first run (downloading model)...\n")

    try:
        config_manager = ConfigManager()
        loader = LocalModelLoader(
            model_id=model_id,
            quant_config=quant_config,
            split_config=split_config,
            trust_remote_code=True,
            config_manager=config_manager
        )

        print("Starting model load...")
        model, tokenizer = loader.load()

        print("\n✓ Model loaded successfully!")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Tokenizer vocab size: {len(tokenizer)}")

        # Test generation
        print("\nTesting generation...")
        inputs = tokenizer("Hello, my name is", return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

        print("\n✓ Model is working correctly!")

        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        print("\n✓ Model unloaded and GPU memory cleared")

    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        logger.error(f"Failed to load model: {e}", exc_info=True)
        print("\nTroubleshooting tips:")
        print("1. Ensure you have enough disk space for model download")
        print("2. Check GPU is available: nvidia-smi")
        print("3. Try smaller model: google/gemma-3-270m")
        print("4. Check GPU_OPTIMIZATION_GUIDE.md for more details")


def example_4_compare_presets():
    """Example 4: Compare different preset configurations."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Comparing Different GPU Presets")
    print("="*70 + "\n")

    presets_to_compare = ["2gb", "4gb", "8gb", "cpu"]

    print(f"{'Preset':<10} | {'Quantization':<12} | {'GPU Memory':<12} | {'Model':<30}")
    print("-" * 80)

    for preset_name in presets_to_compare:
        quant, split, model = get_preset(preset_name)
        gpu_mem = split.max_gpu_memory.get(0, "N/A")
        print(f"{preset_name:<10} | {quant.method:<12} | {gpu_mem:<12} | {model:<30}")

    print()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("OPTIMIZED MODEL LOADING EXAMPLES")
    print("GTX 1650 Ti (4GB VRAM)")
    print("="*70)

    # Run examples
    example_1_auto_detect()
    example_2_specific_preset()
    example_4_compare_presets()

    # Ask user if they want to try loading a model
    print("\n" + "="*70)
    print("INTERACTIVE MODEL LOADING TEST")
    print("="*70 + "\n")

    response = input("Do you want to test loading a model? (y/n): ").strip().lower()

    if response == 'y':
        example_3_load_model()
    else:
        print("\nSkipping model load test.")
        print("To test later, run: python example_optimized_load.py")

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the configurations above")
    print("2. Import gpu_presets in your code:")
    print("   from src.config.gpu_presets import get_preset")
    print("3. Use the preset when initializing models:")
    print("   quant, split, model = get_preset('4gb')")
    print("4. See GPU_OPTIMIZATION_GUIDE.md for detailed documentation")
    print()


if __name__ == "__main__":
    main()
