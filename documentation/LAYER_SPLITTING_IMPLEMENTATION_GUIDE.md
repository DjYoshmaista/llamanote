# Layer Splitting & Checkpoint Config Implementation Guide

## Overview

This guide documents the new layer splitting and checkpoint configuration system.

**Status**: Core modules created, integration pending

## New Modules Created

### 1. `src/utils/device_map_builder.py` ✅

**Purpose**: Creates explicit device maps for deterministic layer placement

**Key Classes**:
- `DeviceMapBuilder`: Builds custom device_map dicts
- `LayerSplitResult`: Stores successful split configurations
- `LayerSplitConfigManager`: Saves/loads split configs

**Usage Example**:
```python
from src.utils.device_map_builder import DeviceMapBuilder

builder = DeviceMapBuilder("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
device_map = builder.build_device_map(layers_on_gpu=20)

# device_map = {
#     'model.embed_tokens': 0,
#     'model.layers.0': 0,
#     ...
#     'model.layers.19': 0,  # Layer 20 on GPU
#     'model.layers.20': 'cpu',  # Remaining on CPU
#     ...
# }
```

### 2. `src/utils/layer_split_finder.py` ✅

**Purpose**: Automatically discovers optimal layer splits

**Key Classes**:
- `LayerSplitFinder`: Tests splits via binary search
- `find_and_save_layer_split()`: Convenience function

**Process**:
1. Binary search to find max layers that fit on GPU
2. Test forward pass to verify functionality
3. Find minimum GPU layers (for memory conservation)
4. Save both configurations

**Usage Example**:
```python
from src.utils.layer_split_finder import find_and_save_layer_split
from transformers import BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_4bit=True)

min_split, max_split = find_and_save_layer_split(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization_config=quant_config
)

print(f"Min split: {min_split.layers_on_gpu} GPU / {min_split.layers_on_cpu} CPU")
print(f"Max split: {max_split.layers_on_gpu} GPU / {max_split.layers_on_cpu} CPU")
```

### 3. `src/io/checkpoint_config.py` ✅

**Purpose**: Manages configuration files required for checkpoint resume

**Key Classes**:
- `CheckpointConfigManager`: UUID-based config management

**Features**:
- Generates UUID from resume-critical settings
- Reuses config files when UUIDs match (deduplication)
- Automatically loads correct config for checkpoint

**Usage Example**:
```python
from src.io.checkpoint_config import CheckpointConfigManager

config_mgr = CheckpointConfigManager()

# Save config (returns UUID)
uuid = config_mgr.save_config(pipeline_config)  # e.g., "a3f5b2c8-1234-5678-90ab-cdef12345678"

# Load config later
loaded_config = config_mgr.load_config(uuid)
```

## Integration Points

### A. Integrate Device Maps into Model Loading

**File**: `src/models/backends/local_hf.py`

**Current**:
```python
# Line 172 in build_load_config()
self.load_config["device_map"] = "auto"
```

**Proposed**:
```python
# Check if we have a saved device map for this configuration
from ..utils.device_map_builder import LayerSplitConfigManager

split_mgr = LayerSplitConfigManager()
quant_method = self.quant_config.method
gpu_memory_gb = list(self.split_config.max_gpu_memory.values())[0] if self.split_config.max_gpu_memory else "4GB"
gpu_memory_gb = float(gpu_memory_gb.replace("GB", ""))

# Try to load saved split
saved_splits = split_mgr.load_config(
    model_id=self.model_id,
    quant_method=quant_method,
    gpu_memory_gb=gpu_memory_gb
)

if saved_splits:
    min_split, max_split = saved_splits
    # Use the minimum split (most layers on GPU) for best performance
    self.load_config["device_map"] = min_split.device_map
    logger.info(f"Using saved layer split: {min_split.layers_on_gpu} GPU / {min_split.layers_on_cpu} CPU")
else:
    # Fall back to auto
    self.load_config["device_map"] = "auto"
```

### B. Integrate OOM Recovery with Layer Splitting

**File**: `src/utils/memory_manager.py`

**Current**: Uses `max_memory` dict to control allocation

**Proposed**: Add device_map support to OOM recovery

```python
# In OOMRecoveryStrategy.__init__
def __init__(self, ..., device_map_builder=None, ...):
    self.device_map_builder = device_map_builder

# In get_next_strategy() for split_layers action:
if self.device_map_builder:
    # Use explicit device map instead of max_memory
    device_map = self.device_map_builder.build_device_map(layers_on_gpu)
    strategy['params']['device_map'] = device_map
    # Remove max_memory when using explicit device_map
    strategy['params'].pop('max_memory', None)
```

### C. Link Checkpoints to Configuration UUIDs

**File**: `src/io/checkpoints.py`

**Add to checkpoint metadata**:
```python
# In CheckpointManager.save()
def save(self, ..., config_uuid: Optional[str] = None):
    metadata = {
        ...
        'config_uuid': config_uuid,  # NEW: Link to configuration
        ...
    }
```

**When loading checkpoint**:
```python
# In CheckpointManager.load()
checkpoint_data = ...
config_uuid = checkpoint_data['metadata'].get('config_uuid')

if config_uuid:
    # Load the configuration
    from .checkpoint_config import CheckpointConfigManager
    config_mgr = CheckpointConfigManager()
    required_config = config_mgr.load_config(config_uuid)

    # Return both data and required config
    return checkpoint_data['data'], required_config
```

### D. Update Pipeline to Save Config UUID

**File**: `src/core/pipeline.py`

**In __init__**:
```python
from ..io.checkpoint_config import CheckpointConfigManager

self.checkpoint_config_mgr = CheckpointConfigManager()
self.config_uuid = None
```

**Before processing**:
```python
# In process_file(), before starting pipeline
self.config_uuid = self.checkpoint_config_mgr.save_config(self.config)
logger.info(f"Configuration UUID: {self.config_uuid}")
```

**When saving checkpoints**:
```python
# In _save_checkpoint()
if self.checkpoint_manager:
    self.checkpoint_manager.save(
        ...,
        config_uuid=self.config_uuid  # Pass UUID to checkpoint
    )
```

### E. Add Menu Option for Checkpoint Resume

**File**: `src/menu.py`

**Add new menu option**:
```python
def show_checkpoint_resume_menu(self):
    """Show menu to select and resume from checkpoint."""
    print("\n" + "="*80)
    print("Resume from Checkpoint")
    print("="*80)

    # List available checkpoints
    checkpoints = self.checkpoint_manager.list_all_checkpoints()

    if not checkpoints:
        ConsoleOutput.warning("No checkpoints found")
        input("\nPress Enter to return...")
        return

    # Display checkpoints grouped by input file
    for idx, (input_file, stages) in enumerate(checkpoints.items(), 1):
        print(f"\n{idx}. {input_file}")
        for stage_info in stages:
            print(f"   - {stage_info['stage']}: {stage_info['timestamp']}")

    print("\n(b) Back")
    choice = input("\nSelect checkpoint: ").strip()

    if choice.lower() == 'b':
        return

    try:
        idx = int(choice) - 1
        selected_file = list(checkpoints.keys())[idx]

        # Load checkpoint
        checkpoint_data, required_config = self.checkpoint_manager.load_latest(selected_file)

        if required_config:
            # Apply configuration
            self.apply_checkpoint_config(required_config)
            ConsoleOutput.success(f"Loaded configuration for {selected_file}")

            # Ask if user wants to continue
            print("\nCheckpoint loaded. Ready to resume processing.")
            confirm = input("Start pipeline? (y/n): ").strip().lower()

            if confirm == 'y':
                # Set input file and run
                self.state.input_files = [Path(selected_file)]
                self.run_pipeline()

    except (ValueError, IndexError):
        ConsoleOutput.error("Invalid selection")
        input("\nPress Enter to return...")
```

## Why the Current System Falls Back to CPU

**Problem**: Even with our OOM recovery, models end up on CPU

**Root Cause**: `device_map="auto"` with `max_memory` is non-deterministic and conservative

**How Accelerate Works**:
```python
# With device_map="auto" and max_memory={0: "3GB", "cpu": "28GB"}:
# 1. Accelerate estimates memory for each layer
# 2. Places layers on GPU until memory limit reached
# 3. Places remaining layers on CPU
# 4. BUT: If initial placement fails (OOM), it gives up and moves everything to CPU
```

**Solution**: Use explicit device_map

```python
# Instead of:
device_map = "auto"
max_memory = {0: "3GB", "cpu": "28GB"}

# Use:
device_map = {
    'model.embed_tokens': 0,
    'model.layers.0': 0,
    'model.layers.1': 0,
    ...
    'model.layers.19': 0,  # 20 layers on GPU
    'model.layers.20': 'cpu',  # Rest on CPU
    ...
}
```

This forces Transformers to load exactly as specified, no guessing.

## Testing the New System

### 1. Test Device Map Builder

```bash
python -c "
from src.utils.device_map_builder import DeviceMapBuilder

builder = DeviceMapBuilder('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')
print(f'Total layers: {builder.num_layers}')

# Build device map with 20 layers on GPU
device_map = builder.build_device_map(20)
print(f'Device map created with {len([k for k, v in device_map.items() if v == 0])} modules on GPU')
"
```

### 2. Test Layer Split Finder

```bash
python -c "
from src.utils.layer_split_finder import LayerSplitFinder
from transformers import BitsAndBytesConfig

# Create quantization config
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype='bfloat16')

# Create finder
finder = LayerSplitFinder('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')

# Test a specific split
load_config = {
    'quantization_config': quant_config,
    'low_cpu_mem_usage': True,
    'trust_remote_code': True
}

success, gpu_mem = finder.test_split(20, load_config)
print(f'20 GPU layers: {'✓ Success' if success else '✗ Failed'}')
if success:
    print(f'GPU memory: {gpu_mem:.1f}MB')
"
```

### 3. Test Configuration UUID System

```bash
python -c "
from src.io.checkpoint_config import CheckpointConfigManager
from src.core.types import PipelineConfig, ProcessingMode

# Create config manager
config_mgr = CheckpointConfigManager()

# Create a pipeline config
config = PipelineConfig(
    mode=ProcessingMode.PODCAST,
    model_provider='local_hf',
    model_specifier='deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
)

# Save and get UUID
uuid = config_mgr.save_config(config)
print(f'Configuration UUID: {uuid}')

# Try saving again (should reuse)
uuid2 = config_mgr.save_config(config)
print(f'Second UUID: {uuid2}')
print(f'Same UUID: {uuid == uuid2}')

# Load config
loaded = config_mgr.load_config(uuid)
print(f'Loaded config: {loaded.model_specifier}')
"
```

## Implementation Checklist

- [x] Create device_map_builder.py
- [x] Create layer_split_finder.py
- [x] Create checkpoint_config.py
- [ ] Integrate device maps into local_hf.py
- [ ] Add device_map support to OOM recovery
- [ ] Link checkpoints to config UUIDs
- [ ] Update pipeline to save config UUIDs
- [ ] Add checkpoint resume menu option
- [ ] Test end-to-end functionality
- [ ] Update README.md

## Benefits

### 1. Deterministic Layer Placement
- Know exactly which layers are on GPU vs CPU
- Reproducible performance across runs
- No surprises from "auto" placement

### 2. Optimal Performance Discovery
- Automatically find best split for your hardware
- Save and reuse working configurations
- No manual trial-and-error

### 3. Proper Checkpoint Resume
- Checkpoints link to exact configuration needed
- No "missing data" errors
- Configurations deduplicated (same config = same UUID)

### 4. Better User Experience
- Load checkpoint → Auto-configure → Ready to run
- Clear about what's needed for resume
- Reuse proven configurations

## Current Limitations

1. **Integration Not Complete**: Core modules created but not yet integrated
2. **Testing Required**: Need to verify device maps work with all model architectures
3. **Menu Updates Needed**: Checkpoint resume menu not yet implemented
4. **Documentation**: README needs updates

## Next Steps

1. **Integrate device maps** into model loading (highest priority)
2. **Test with DeepSeek-1.5B** to verify layer splitting works
3. **Add checkpoint-config linking** to save/load
4. **Implement resume menu** for better UX
5. **Update documentation** once tested

## Temporary Workaround

Until integration is complete, you can manually test layer splitting:

```python
# In your code or Python REPL:
from src.utils.device_map_builder import DeviceMapBuilder
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Build device map
builder = DeviceMapBuilder("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
device_map = builder.build_device_map(layers_on_gpu=20)

# Load model with explicit device map
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map=device_map,  # Use explicit map instead of "auto"
    trust_remote_code=True
)

# Verify placement
print("Model loaded!")
for name, param in model.named_parameters():
    if 'layers.0' in name:
        print(f"{name}: {param.device}")  # Should be cuda:0
        break
    if 'layers.25' in name:
        print(f"{name}: {param.device}")  # Should be cpu
        break
```

This proves the concept works before full integration.
