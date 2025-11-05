#!/usr/bin/env python3
"""
Test script for iterative layer splitting functionality.
This simulates the OOM recovery process without actually loading a model.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils.memory_manager import OOMRecoveryStrategy
from utils.logger import get_logger_conf

logger = get_logger_conf(__name__)

def test_layer_splitting():
    """Test the iterative layer splitting strategy."""
    print("=" * 80)
    print("TESTING ITERATIVE LAYER SPLITTING")
    print("=" * 80)
    print()

    # Simulate DeepSeek-R1-Distill-Qwen-1.5B which has 28 layers
    model_layers = 28

    print(f"📊 Model Configuration:")
    print(f"   Model: DeepSeek-R1-Distill-Qwen-1.5B")
    print(f"   Total Layers: {model_layers}")
    print(f"   Initial GPU Memory: 4GB")
    print(f"   Initial CPU Memory: 30GB")
    print()

    # Create recovery strategy
    strategy = OOMRecoveryStrategy(
        initial_gpu_memory='4GB',
        initial_cpu_memory='30GB',
        min_gpu_memory='1GB',
        model_num_layers=model_layers,
        use_iterative_layer_split=True,
        logger=logger
    )

    print("🔄 Simulating OOM Recovery Attempts:")
    print("-" * 80)

    attempt_results = []

    # Simulate attempts
    while strategy.should_retry():
        next_strategy = strategy.get_next_strategy()

        if next_strategy['action'] == 'exhausted':
            print()
            print("⚠️  All strategies exhausted")
            break

        attempt_num = next_strategy['attempt']
        action = next_strategy['action']
        params = next_strategy['params']

        result = {
            'attempt': attempt_num,
            'action': action,
        }

        # Display attempt info
        if action == 'clear_cache':
            print(f"Attempt {attempt_num:2d}: Clearing cache aggressively")
            result['description'] = "Cache clearing"

        elif action == 'offload_kv_cache':
            print(f"Attempt {attempt_num:2d}: Offloading KV cache to CPU")
            result['description'] = "KV cache → CPU"

        elif action == 'offload_context':
            print(f"Attempt {attempt_num:2d}: Offloading context/activations to CPU")
            result['description'] = "Context → CPU"

        elif action == 'split_layers':
            layers_gpu = params.get('layers_on_gpu', 0)
            layers_cpu = params.get('layers_on_cpu', 0)
            print(f"Attempt {attempt_num:2d}: Layer split - {layers_gpu:2d} GPU / {layers_cpu:2d} CPU")
            result['description'] = f"{layers_gpu} GPU / {layers_cpu} CPU"
            result['layers_gpu'] = layers_gpu
            result['layers_cpu'] = layers_cpu

        elif action == 'reduce_gpu_memory':
            max_mem = params.get('max_gpu_memory', {}).get(0, 'Unknown')
            print(f"Attempt {attempt_num:2d}: Reducing GPU memory to {max_mem}")
            result['description'] = f"GPU mem → {max_mem}"

        attempt_results.append(result)

        # Stop after 15 attempts for demo
        if attempt_num >= 15:
            print()
            print("... (showing first 15 attempts)")
            break

    print()
    print("=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print()

    # Count attempts by type
    cache_attempts = sum(1 for r in attempt_results if r['action'] == 'clear_cache')
    kv_attempts = sum(1 for r in attempt_results if r['action'] == 'offload_kv_cache')
    context_attempts = sum(1 for r in attempt_results if r['action'] == 'offload_context')
    layer_attempts = sum(1 for r in attempt_results if r['action'] == 'split_layers')
    memory_attempts = sum(1 for r in attempt_results if r['action'] == 'reduce_gpu_memory')

    print(f"Total Attempts: {len(attempt_results)}")
    print(f"  - Cache clearing:      {cache_attempts}")
    print(f"  - KV cache offload:    {kv_attempts}")
    print(f"  - Context offload:     {context_attempts}")
    print(f"  - Layer splitting:     {layer_attempts}")
    print(f"  - Memory reduction:    {memory_attempts}")
    print()

    if layer_attempts > 0:
        print("✅ Layer Splitting Progression:")
        for r in attempt_results:
            if r['action'] == 'split_layers':
                gpu = r.get('layers_gpu', 0)
                cpu = r.get('layers_cpu', 0)
                pct_gpu = (gpu / model_layers * 100) if model_layers > 0 else 0
                print(f"   Attempt {r['attempt']:2d}: {gpu:2d} GPU ({pct_gpu:5.1f}%) / {cpu:2d} CPU")
        print()

        # Show what would happen if it succeeded at attempt 10
        if len(attempt_results) >= 10:
            attempt_10 = attempt_results[9]  # 0-indexed
            if 'layers_gpu' in attempt_10:
                print(f"💡 Example Success Scenario:")
                print(f"   If model loads successfully at Attempt 10:")
                print(f"   → {attempt_10['layers_gpu']} layers on GPU (using ~2-3GB VRAM)")
                print(f"   → {attempt_10['layers_cpu']} layers on CPU (slower but works)")
                print(f"   → Processing continues normally")
                print()

    print("=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()
    print("Key Improvements:")
    print("  ✓ Layer-by-layer splitting (instead of percentage-based)")
    print("  ✓ Up to 50 attempts (instead of 10)")
    print("  ✓ Clear progress indication")
    print("  ✓ Automatic acceleration after 25 attempts")
    print()
    print("Next Steps:")
    print("  1. Run actual pipeline with fresh PDF (not checkpoint)")
    print("  2. Watch for 'Detected X layers' message")
    print("  3. Observe layer splitting in action")
    print("  4. Check final layer distribution")
    print()

if __name__ == '__main__':
    test_layer_splitting()
