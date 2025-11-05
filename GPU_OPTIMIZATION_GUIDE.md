# GPU Optimization Guide for GTX 1650 Ti (4GB VRAM)

## Your Hardware
- **GPU**: NVIDIA GeForce GTX 1650 Ti
- **VRAM**: 4096 MB (4 GB)
- **Currently Free**: ~3.6 GB

## Problem Analysis

The DeepSeek-R1-Distill-Qwen-1.5B model (1.5B parameters) requires:
- **FP32**: ~6 GB
- **FP16**: ~3 GB
- **8-bit**: ~1.5 GB
- **4-bit**: ~0.75 GB

With 4GB VRAM, you need aggressive optimization strategies.

## Recommended Solutions

### Option 1: Use 4-bit Quantization (RECOMMENDED)
This is the most effective approach for your GPU.

**Benefits:**
- Model fits entirely in VRAM
- Reasonable inference speed
- Good quality retention

**Configuration:**
```python
from src.core.types import QuantizationConfig, LayerSplitConfig
import torch

# Optimal quantization config for 4GB GPU
quant_config = QuantizationConfig(
    method="4bit",
    compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    use_double_quant=True,
    quant_type="nf4"
)

# Layer split config with conservative GPU memory
split_config = LayerSplitConfig(
    enabled=True,
    gpu_layers=-1,  # Try to use GPU for all layers
    max_gpu_memory={0: "3.5GB"},  # Conservative limit (leave headroom)
    max_cpu_memory="28GB",
    offload_state_dict=True,
    low_cpu_mem_usage=True,
    auto_oom_handling=True  # Enable automatic OOM recovery
)
```

### Option 2: Use Smaller Models
Switch to more VRAM-friendly models:

**Recommended Models for 4GB VRAM:**
1. **google/gemma-3-270m** (270M params) - Fits easily, fast
2. **TinyLlama/TinyLlama-1.1B** (1.1B params) - With 4-bit
3. **microsoft/phi-2** (2.7B params) - With 4-bit + aggressive offload

**Update in code:**
```python
# In your config or when initializing
DEFAULT_MODEL_KEY = "gemma3-270m"  # Already defined in your presets
```

### Option 3: Layer Offloading Strategy
For models that still don't fit, use progressive layer offloading:

```python
split_config = LayerSplitConfig(
    enabled=True,
    max_gpu_memory={0: "3.2GB"},  # More aggressive limit
    max_cpu_memory="28GB",
    offload_folder=Path("./offload"),  # Enable disk offload
    offload_state_dict=True,
    low_cpu_mem_usage=True,
    auto_oom_handling=True
)
```

### Option 4: Use CPU-Only Mode
For models that absolutely won't fit:

```python
split_config = LayerSplitConfig(
    enabled=True,
    max_gpu_memory={0: "0GB"},  # Force CPU-only
    max_cpu_memory="28GB",
    offload_state_dict=False,
    low_cpu_mem_usage=False,
    auto_oom_handling=False
)
```

## Implementation in Your Codebase

### Quick Fix: Update Default Settings

Edit `src/config/settings.py`:
```python
# Line 43 - Change default quantization
DEFAULT_QUANTIZATION: str = "4bit"  # Already set correctly!

# Line 56 - Consider switching default model
DEFAULT_MODEL_KEY: str = "gemma3-270m"  # Use smaller model
```

### For Current DeepSeek Model

If you want to keep using DeepSeek-R1-Distill-Qwen-1.5B:

1. **Ensure 4-bit quantization is enabled**
2. **Set conservative GPU memory limit**: `3.5GB` instead of `4GB`
3. **Enable disk offloading** as fallback

## Testing Your Configuration

Run this test to verify your setup:

```bash
# From project root
python -c "
import torch
from src.core.types import QuantizationConfig, LayerSplitConfig
from src.models.backends.local_hf import LocalModelLoader
from src.config.manager import ConfigManager

print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')

# Test configuration
quant = QuantizationConfig(method='4bit', compute_dtype=torch.bfloat16)
split = LayerSplitConfig(
    enabled=True,
    max_gpu_memory={0: '3.5GB'},
    auto_oom_handling=True
)

print(f'Quantization: {quant.method}')
print(f'Max GPU Memory: {split.max_gpu_memory}')
print('Configuration looks good!')
"
```

## Memory Usage Tips

1. **Close other GPU applications** before loading models
2. **Monitor memory**: Use `nvidia-smi -l 1` in another terminal
3. **Clear cache between loads**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```
4. **Use gradient checkpointing** if fine-tuning (not needed for inference)

## Model-Specific Recommendations

### For DeepSeek-R1-Distill-Qwen-1.5B (1.5B)
- **Minimum**: 4-bit quantization, 3.5GB GPU limit
- **Optimal**: 4-bit + 8-bit layers offload
- **Expected Speed**: 10-20 tokens/sec

### For Qwen3-4B-Instruct (4B)
- **Too Large**: Won't fit even with 4-bit
- **Alternative**: Use with heavy CPU offload (slow)
- **Better Option**: Switch to smaller model

### For Gemma-3-270M (270M) ✅
- **Perfect Fit**: Works great on 4GB
- **No Quantization Needed**: Can use FP16
- **Expected Speed**: 50-100 tokens/sec

## Troubleshooting

### Still Getting OOM?
1. Reduce `max_gpu_memory` to `3.0GB`
2. Enable disk offloading
3. Try smaller model variant
4. Use CPU-only mode

### Model Loads But Is Slow?
- Layers are being offloaded to CPU/disk
- This is expected with limited VRAM
- Consider using smaller model

### Errors About Device Placement?
- Your current fixes to `dtype` (replacing `torch_dtype`) should resolve this
- Make sure you're using latest code with the fixes

## Next Steps

1. **Immediate**: Use gemma-3-270m for fast testing
2. **Production**: Decide between quality (1.5B with 4-bit) vs speed (270M)
3. **Future**: Consider upgrading to 8GB+ VRAM GPU for larger models

## Code Changes Summary

Your recent changes to replace `torch_dtype` with `dtype` are correct and will eliminate the deprecation warnings. The OOM issue is purely about model size vs available VRAM.

**Key takeaway**: With 4GB VRAM, stick to models ≤1.5B parameters with 4-bit quantization, or use smaller models like Gemma-3-270M.
