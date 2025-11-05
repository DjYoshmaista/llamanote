#!/usr/bin/env python3
"""
Test script to verify 4-bit and 8-bit quantization with DeepSeek-R1-Distill-Qwen-1.5B
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project to path
sys.path.insert(0, '/home/yosh/gitrepos/llamanote-github')
from src.core.types import QuantizationConfig

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "CUDA not available"

def test_model_loading(model_id: str, quant_method: str):
    """Test loading a model with specified quantization"""
    print(f"\n{'='*60}")
    print(f"Testing {quant_method} quantization with {model_id}")
    print(f"{'='*60}")

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"Initial GPU memory: {get_gpu_memory_info()}")

    # Create quantization config
    quant_config = QuantizationConfig(method=quant_method)
    bnb_config = quant_config.to_bnb_config()

    print(f"\nQuantization config: {bnb_config}")

    try:
        print(f"\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=quant_config.compute_dtype,
            trust_remote_code=True
        )

        print(f"✓ Model loaded successfully!")
        print(f"GPU memory after loading: {get_gpu_memory_info()}")

        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(0) / (1024**3)
            print(f"Peak GPU memory: {peak:.2f}GB")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        # Test a simple generation
        print(f"\nTesting generation...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {response}")

        # Clean up
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        if torch.cuda.is_available():
            print(f"GPU memory at failure: {get_gpu_memory_info()}")
        return False

if __name__ == "__main__":
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Test both quantization methods
    results = {}

    for method in ["4bit", "8bit"]:
        try:
            success = test_model_loading(model_id, method)
            results[method] = "✓ Success" if success else "✗ Failed"
        except Exception as e:
            results[method] = f"✗ Error: {e}"

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    for method, result in results.items():
        print(f"{method}: {result}")
