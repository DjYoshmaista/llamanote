# Quick Start Guide for 4GB GPU

**GPU**: NVIDIA GeForce GTX 1650 Ti (4GB VRAM)
**Status**: ✅ Optimized configurations ready

## What Was Fixed

### 1. Deprecated `torch_dtype` Parameter
- ✅ Replaced all `torch_dtype` with `dtype` in:
  - `src/models/backends/local_hf.py` (7 locations)
  - `src/models/backends/transcript_generator.py` (1 location)
  - `src/models/backends/local_audio.py` (2 locations)

### 2. OOM (Out of Memory) Issues
- Created optimized configurations for 4GB VRAM
- Added automatic GPU detection and preset selection
- Implemented conservative memory limits with disk offload fallback

## Quick Usage

### Option 1: Use Auto-Detection (Recommended)

```python
from src.config.gpu_presets import auto_detect_preset

# Automatically detect GPU and get optimal config
quant_config, split_config, recommended_model = auto_detect_preset()

# Use these configs when loading your model
```

### Option 2: Use Specific Preset

```python
from src.config.gpu_presets import get_preset

# Get optimized config for 4GB GPU
quant_config, split_config, recommended_model = get_preset("4gb")

# Configuration details:
# - 4-bit quantization (NF4)
# - 3.1GB GPU memory limit (leaves headroom)
# - 28GB CPU memory for offloading
# - Disk offload enabled
# - Auto OOM handling enabled
```

### Option 3: Run Optimizer Script

```bash
python optimize_for_4gb_gpu.py
```

This will:
- Detect your GPU
- Show optimal configuration
- Test if config will work
- Provide recommendations

## Recommended Models for Your GPU

### 1. Gemma-3-270M (Best for Speed)
- **Size**: 270M parameters
- **VRAM**: ~500MB with FP16
- **Speed**: Very fast (50-100 tokens/sec)
- **Quality**: Good for basic tasks

```python
model_id = "google/gemma-3-270m"
# Works without quantization
```

### 2. DeepSeek-R1-Distill-Qwen-1.5B (Best for Quality)
- **Size**: 1.5B parameters
- **VRAM**: ~800MB with 4-bit quantization
- **Speed**: Moderate (10-20 tokens/sec)
- **Quality**: Excellent

```python
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
# Requires 4-bit quantization
quant_config, split_config, _ = get_preset("4gb")
```

### 3. Qwen3-4B (Requires Offloading)
- **Size**: 4B parameters
- **VRAM**: Won't fit entirely in 4GB
- **Speed**: Slower (layers offloaded to CPU)
- **Quality**: Best

⚠️ Only recommended if you can tolerate slower inference

## Test Your Configuration

### Quick Test (No Model Loading)

```bash
python optimize_for_4gb_gpu.py
```

### Full Test (With Model Loading)

```bash
python example_optimized_load.py
```

This interactive script will:
1. Show auto-detected configuration
2. Explain the 4GB preset
3. Compare different presets
4. Optionally load and test a model

## Files Created

1. **`GPU_OPTIMIZATION_GUIDE.md`** - Comprehensive guide
2. **`optimize_for_4gb_gpu.py`** - Diagnostic and config generator
3. **`src/config/gpu_presets.py`** - Reusable preset configurations
4. **`example_optimized_load.py`** - Example usage code
5. **`QUICK_START_4GB_GPU.md`** - This file

## Optimal Configuration for Your GPU

```python
from src.core.types import QuantizationConfig, LayerSplitConfig
import torch
from pathlib import Path

quant_config = QuantizationConfig(
    method='4bit',
    compute_dtype=torch.bfloat16,  # Your GPU supports BF16
    use_double_quant=True,
    quant_type='nf4'
)

split_config = LayerSplitConfig(
    enabled=True,
    max_gpu_memory={0: '3.1GB'},  # Conservative limit
    max_cpu_memory='28GB',
    offload_folder=Path('./offload'),
    offload_state_dict=True,
    low_cpu_mem_usage=True,
    auto_oom_handling=True
)
```

## Tips for Success

1. **Close other GPU apps** before loading models
   ```bash
   nvidia-smi  # Check what's using GPU
   ```

2. **Monitor GPU memory** in real-time
   ```bash
   nvidia-smi -l 1  # Update every second
   ```

3. **Start with small models** to verify everything works
   - Try Gemma-3-270M first
   - Then move to DeepSeek-1.5B if needed

4. **Clear GPU cache** between loads
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

5. **Check disk space** for offloading (5-10GB recommended)

## Troubleshooting

### Still Getting OOM?
1. Reduce GPU memory limit to `2.5GB`
2. Use smaller model (Gemma-3-270M)
3. Try CPU-only mode: `get_preset("cpu")`

### Model loads but is slow?
- Normal! Layers are being offloaded to CPU/RAM
- For faster inference, use smaller model
- Or upgrade to GPU with more VRAM

### Import errors?
```bash
pip install torch transformers bitsandbytes accelerate
```

## Integration with Existing Code

### In Your Model Loading Code

```python
# Add at the top of your script/module
from src.config.gpu_presets import get_preset

# When creating backend/loader
quant_config, split_config, recommended_model = get_preset("4gb")

# Pass to your existing model initialization
backend = LocalHFBackend(
    model_id=recommended_model,
    model_entry=model_entry,
    quant_config=quant_config,  # Use optimized config
    split_config=split_config,   # Use optimized config
    hyperparams=hyperparams
)
```

### In CLI/Main Entry Point

```python
from src.config.gpu_presets import auto_detect_preset

# At startup, detect GPU and set defaults
quant_config, split_config, recommended_model = auto_detect_preset()
print(f"Detected GPU configuration, using: {recommended_model}")
```

## Performance Expectations

### Gemma-3-270M (No Quantization)
- Load time: 5-10 seconds
- Inference: 50-100 tokens/sec
- VRAM usage: ~500MB

### DeepSeek-1.5B (4-bit)
- Load time: 15-30 seconds
- Inference: 10-20 tokens/sec
- VRAM usage: ~800MB

### Qwen3-4B (4-bit + offload)
- Load time: 30-60 seconds
- Inference: 5-10 tokens/sec
- VRAM usage: ~3GB (rest on CPU)

## Next Steps

1. ✅ Run the optimizer to verify config: `python optimize_for_4gb_gpu.py`
2. ✅ Try example script: `python example_optimized_load.py`
3. ✅ Integrate `gpu_presets` into your workflow
4. 📖 Read `GPU_OPTIMIZATION_GUIDE.md` for deep dive
5. 🚀 Start using optimized configurations in production

## Summary

Your GTX 1650 Ti (4GB) can handle:
- ✅ Small models (<1B params) with no quantization
- ✅ Medium models (1-2B params) with 4-bit quantization
- ⚠️ Large models (3-4B params) with heavy offloading (slow)
- ❌ Very large models (7B+ params) - too large even with offloading

**Recommended workflow**: Start with Gemma-3-270M, upgrade to DeepSeek-1.5B if you need better quality.
