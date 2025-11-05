# Quick Reference: Model Loading & Memory Management

## File Locations Reference

| Component | File | Key Classes/Functions |
|-----------|------|----------------------|
| **Pipeline Execution** | `src/core/pipeline.py` | `ProcessingPipeline`, `process_file()` |
| **Pipeline Stages** | `src/core/stages.py` | `PipelineStageExecutor`, individual stage functions |
| **Type Definitions** | `src/core/types.py` | `PipelineConfig`, `QuantizationConfig`, `LayerSplitConfig` |
| **LLM Backend Base** | `src/models/backends/base.py` | `LLMBackend`, `AudioBackend` |
| **Local HF Backend** | `src/models/backends/local_hf.py` | `LocalHFBackend`, `LocalModelLoader` |
| **Local Audio Backend** | `src/models/backends/local_audio.py` | `LocalAudioBackend` |
| **Backend Factory** | `src/models/backends/__init__.py` | `get_llm_backend()`, `get_audio_backend()` |
| **Memory Management** | `src/utils/memory_manager.py` | `CUDAMemoryManager`, `OOMRecoveryStrategy` |
| **Hyperparameters** | `src/models/hyperparameters.py` | `HyperparameterConfig`, `HyperparameterDef` |
| **Memory Profiles** | `src/config/profiles.py` | `MemoryProfile`, profile functions |
| **Settings** | `src/config/settings.py` | Constants, `MarkdownStyle` |
| **CLI Handler** | `src/cli.py` | `run_cli_processing()`, argument parsing |
| **Menu System** | `src/menu.py` | `AppState`, menu configuration |

## Key Code Snippets

### Load Text Model (CLI)
**Location:** `src/cli.py:568`
```python
text_backend = get_llm_backend(
    provider=provider,
    model_specifier=model_specifier,
    api_keys=api_keys,
    hyperparameters=hyperparams,
    model_entry=model_entry,
    quantization_config=quant_config,
    layer_split_config=layer_split_config
)
if not text_backend.load(trust_remote_code=True):
    raise ModelLoadError(...)
pipeline.llm_backend = text_backend
```

### Load Audio Model (CLI)
**Location:** `src/cli.py:575`
```python
audio_backend = get_audio_backend(
    provider=pipeline_config.audio_provider,
    model_specifier=pipeline_config.audio_specifier,
    api_keys=api_keys,
    config=pipeline_config.audio_config,
    layer_split_config=pipeline_config.layer_split_config
)
if not audio_backend.load():
    raise ModelLoadError(...)
pipeline.audio_backend = audio_backend
```

### Unload Model (Batch Processing)
**Location:** `src/pipeline.py:644`
```python
if len(input_files) > 1 and text_backend.provider_identifier.startswith("local"):
    text_backend.unload()
    audio_backend.unload()
```

### Configure Memory (CLI)
**Location:** `src/cli.py:490`
```python
quant_config, layer_split_config = create_configs_from_memory_profile(args.memory_profile)

# Override from CLI
if args.quantization:
    quant_config.method = args.quantization
if args.gpu_layers is not None:
    layer_split_config.gpu_layers = args.gpu_layers
```

### Memory Cleanup During Processing
**Location:** `src/core/stages.py:215`
```python
if llm_backend.provider_identifier.startswith("local") and (i + 1) % 5 == 0:
    if torch and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
```

### OOM Recovery During Load
**Location:** `src/models/backends/local_hf.py:204`
```python
model, final_params = with_oom_handling(
    load_fn=load_model_fn,
    recovery_strategy=recovery_strategy,
    logger=logger
)
```

## Configuration Paths

### Memory Profiles
```
medium_vram (default):
  - quantization: "4bit"
  - gpu_layers: -1
  - max_gpu_memory: 4GB per GPU
  - max_cpu_memory: 24GB
  - enable_disk_offload: False

low_vram:
  - quantization: "8bit"
  - gpu_layers: Limited
  - max_gpu_memory: 2GB per GPU
  - max_cpu_memory: 8GB
  - enable_disk_offload: True

high_vram:
  - quantization: "none"
  - gpu_layers: All
  - max_gpu_memory: 8GB+ per GPU
  - max_cpu_memory: 32GB+
  - enable_disk_offload: False
```

### CLI Arguments for Memory Control
```bash
# Memory profile
--memory-profile low_vram|medium_vram|high_vram|disk_offload

# Overrides
--quantization 4bit|8bit|16bit|none
--gpu-layers -1|0|8|16|32

# Hyperparameter control
--hyperparameter-preset balanced|fast|quality
--max-tokens 1024
--max-length 4096

# Audio settings
--generate-audio
--audio-model model_id
--audio-provider local_audio|openai_audio
```

## Model Loading Flow

```
1. CLI Start
   └─> Parse args (--model, --memory-profile, etc)

2. Configuration
   └─> Load memory profile
   └─> Create QuantizationConfig & LayerSplitConfig
   └─> Create PipelineConfig with all settings

3. Backend Creation (Factory Pattern)
   └─> get_llm_backend(provider, model_specifier, configs)
   └─> get_audio_backend(provider, model_specifier, configs)

4. Backend Initialization
   └─> Instantiate backend class
   └─> Set model_handle = None, is_loaded = False

5. Model Loading (Explicit)
   └─> backend.load()
   └─> For local models:
       ├─> Create LocalModelLoader
       ├─> build_load_config() with device mapping
       ├─> Load tokenizer
       ├─> Load model with OOM recovery
       ├─> Post-load setup (device_map capture, etc)
   └─> For cloud models:
       └─> Initialize API client

6. Pipeline Injection
   └─> pipeline.llm_backend = text_backend
   └─> pipeline.audio_backend = audio_backend

7. File Processing
   └─> pipeline.process_file(input_path)

8. Pipeline-Time Loading (if needed)
   └─> Process stage checks: if model_handle is None: load()

9. Memory Management During Generation
   └─> Per 5 chunks: gc.collect() + torch.cuda.empty_cache()
   └─> Checkpoint callbacks for recovery

10. Cleanup
    └─> Batch: Unload between files (if > 1 file)
    └─> CLI Finally: Always unload at end
```

## Memory Architecture Decisions

### Why Two-Stage Loading?
1. **CLI Stage:** Early load allows for:
   - Early error detection
   - Memory requirement verification
   - Centralized backend management

2. **Pipeline Stage:** Fallback load allows for:
   - Lazy evaluation if pipeline doesn't use LLM
   - Dynamic provider selection
   - Testing without actual model

### Why Explicit Unloading Between Batch Files?
- Preserves memory for next file
- Prevents OOM in long batch runs
- Allows recovery from batch failures
- Clear resource lifecycle management

### Why Per-5-Chunk Cleanup?
- Balances memory management overhead vs effectiveness
- Prevents KV cache accumulation
- Empirically determined sweet spot
- Minimal performance impact

### Why Checkpointing?
- Resume from any stage if failure occurs
- Per-chunk granularity for long documents
- Checkpoint saved every N chunks (default: 10)
- Auto-cleanup of old checkpoints (keep 3)

## Debugging Memory Issues

### Check Current Status
```python
# In any backend:
print(f"Loaded: {backend.is_loaded}")
print(f"Handle: {backend.model_handle is not None}")
print(f"Provider: {backend.provider_identifier}")

# GPU memory:
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM at load | Model too large for memory | Use smaller model, enable quantization, or disk offload |
| CUDA OOM during gen | KV cache grows | Reduce max_new_tokens or chunk size |
| Slow generation | CPU offloading | Use smaller chunks, increase max_gpu_memory |
| Memory not released | Models not unloaded | Check that unload() is called in finally blocks |
| Batch failure midway | No checkpointing | Enable checkpoints: --no-checkpoints NOT set |

## Configuration Best Practices

### For Memory-Constrained Systems (2-4GB VRAM)
```
--memory-profile low_vram
--quantization 8bit
--chunk-size 500
--max-tokens 512
```

### For Standard Systems (4-8GB VRAM)
```
--memory-profile medium_vram  # Default
--quantization 4bit           # Default
--chunk-size 1000
--max-tokens 1024
```

### For High-End Systems (8GB+ VRAM)
```
--memory-profile high_vram
--quantization none
--chunk-size 2000
--max-tokens 2048
--gpu-layers -1  # All layers on GPU
```

### For Disk Offloading (Very Limited VRAM)
```
--memory-profile disk_offload
--quantization 8bit
--chunk-size 300
--max-tokens 256
```

## Pipeline Stages Quick Reference

| Stage | Function | Loads | Memory |
|-------|----------|-------|--------|
| extract | Read PDF/text | File I/O | Minimal |
| preprocess | Clean text | Tokenizer (implicit) | Low |
| chunk | Split text | None | Minimal |
| **process** | **LLM gen** | **Text Model** | **High** |
| filter | Remove artifacts | Filter (local) | Low |
| format | Apply styles | Formatter | Low |
| save | Write file | File I/O | Minimal |
| **audio** | **TTS gen** | **Audio Model** | **Medium** |

## Key Metrics to Monitor

1. **GPU Memory Utilization:** `nvidia-smi` or `torch.cuda.memory_allocated()`
2. **CPU Memory:** `psutil.virtual_memory()` or `top`
3. **Tokens Per Second:** Logged during generation
4. **Stage Duration:** Logged by decorators
5. **Checkpoint Size:** Check `project_root/checkpoints/`

---

**Last Updated:** 2025-11-03
**Related Docs:** `ARCHITECTURE_MODEL_MEMORY.md`, `ARCHITECTURE_DIAGRAMS.txt`
