# LlamaNote Model Loading & Memory Management Architecture

## 1. TEXT MODEL (LLM) LOADING & PIPELINE INTEGRATION

### Loading Points

**Primary CLI Loading (src/cli.py, line 568):**
```python
# Step 1: Backend instantiation via factory
text_backend = get_llm_backend(
    provider="local_hf",          # or "local_gguf", "openai", "google", "anthropic"
    model_specifier="qwen3-4b",
    api_keys=api_keys,
    hyperparameters=hyperparams,
    quantization_config=quant_config,
    layer_split_config=layer_split_config
)

# Step 2: Explicit load call in CLI
if not text_backend.load(trust_remote_code=True):
    raise ModelLoadError(...)

# Step 3: Inject into pipeline
pipeline.llm_backend = text_backend
```

**Pipeline-time Loading (src/core/pipeline.py, line 296-300):**
- If LLM backend not loaded when process stage runs:
```python
if self.llm_backend.model_handle is None:
    logger.info("Loading LLM backend model for processing...")
    if not self.llm_backend.load(trust_remote_code=True):
        raise ModelLoadError(...)
```

### Backend Architecture (src/models/backends/)

**Base Class (base.py):**
- `LLMBackend` abstract class with:
  - `model_handle: Any` - Holds actual model/client
  - `is_loaded: bool` - Load state tracking
  - `load(**kwargs)` - Abstract method (provider-specific)
  - `unload()` - Abstract method (cleanup)
  - `process_with_chat_template()` - Public chat interface
  - `generate()` - Public generation interface

**LocalHFBackend (local_hf.py):**
```
LocalModelLoader helper:
  ├─ build_load_config() - Creates load parameters
  │  ├─ Device mapping strategy (auto/CPU/GPU layering)
  │  ├─ Quantization setup (4bit/8bit via BitsAndBytes)
  │  └─ Offload configuration (disk/CPU offload)
  └─ load() - Executes AutoModelForCausalLM.from_pretrained()
     ├─ Tokenizer loading
     ├─ Model loading with OOM recovery
     ├─ Post-load setup (device map capture, token resize)
     └─ Returns (model, tokenizer)

LocalHFBackend:
  ├─ load() - Delegates to LocalModelLoader
  ├─ unload() - cleanup_resources(model, tokenizer, clear_cuda=True)
  ├─ _chat_request() - Implements chat template generation
  ├─ _generate_request() - Implements raw generation
  └─ State tracking:
     - model_handle = model
     - tokenizer = AutoTokenizer
     - device_map = device placement info
     - is_loaded = True/False
```

**Other Backends:**
- `LocalGGUFBackend` - llama-cpp-python models
- `OpenAIBackend` - API client (lazy load)
- `GoogleAIBackend` - API client
- `AnthropicBackend` - API client

## 2. AUDIO MODEL (TTS) LOADING & PIPELINE INTEGRATION

### Loading Points

**CLI Loading (src/cli.py, line 575-587):**
```python
if pipeline_config.generate_audio:
    audio_backend = get_audio_backend(
        provider="local_audio",              # or "openai_audio"
        model_specifier="microsoft/speecht5_tts",
        api_keys=api_keys,
        config=audio_config,
        layer_split_config=layer_split_config
    )
    if not audio_backend.load():
        raise ModelLoadError(...)
    pipeline.audio_backend = audio_backend
```

**Pipeline-time Loading (src/core/pipeline.py, line 499-501):**
```python
if self.audio_backend.model_handle is None:
    if not self.audio_backend.load():
        raise ModelLoadError(...)
```

### LocalAudioBackend (src/models/backends/local_audio.py)

Supports models:
- SpeechT5 (microsoft/speecht5_tts)
- Bark (suno/bark)
- MMS (facebook/mms-tts-eng)
- VibeVoice (microsoft/vibevoice-1.5b)

```
LocalAudioBackend:
  ├─ load()
  │  ├─ Lazy imports transformers/datasets
  │  ├─ Model-specific loading:
  │  │  ├─ SpeechT5: Processor + Model + Vocoder
  │  │  ├─ Bark: Pipeline API
  │  │  ├─ VITS: Pipeline API
  │  │  └─ VibeVoice: Custom processor + model
  │  ├─ Speaker embeddings loading (if configured)
  │  ├─ GPU memory management
  │  └─ model_handle = pipeline or model
  │
  ├─ generate_audio()
  │  ├─ Text chunking (based on config.chunk_size)
  │  ├─ Per-chunk generation with checkpoint callbacks
  │  ├─ Audio array combination
  │  ├─ Post-processing (normalization, speed)
  │  └─ Save to output_path
  │
  └─ unload()
     ├─ Clear model, processor, vocoder, pipeline
     ├─ GPU memory cleanup
     └─ model_handle = None
```

## 3. CURRENT MODEL OFFLOADING/UNLOADING MECHANISMS

### Explicit Unloading

**Batch Processing (pipeline.py, line 634-640):**
```python
# Between files in batch mode, unload local models
if len(input_files) > 1 and text_backend.provider_identifier.startswith("local"):
    ConsoleOutput.info(f"Unloading model {text_backend.model_specifier}...")
    text_backend.unload()
    pipeline.memory_monitor.check(f"after unloading model from file {i}")

if audio_backend and audio_backend.provider_identifier.startswith("local"):
    ConsoleOutput.info(f"Unloading local audio model...")
    audio_backend.unload()
```

**CLI Cleanup (src/cli.py, line 669-674):**
```python
finally:
    if text_backend:
        try: text_backend.unload()
        except Exception: ...
    if audio_backend:
        try: audio_backend.unload()
        except Exception: ...
```

### Layer Offloading During Load

**LayerSplitConfig (src/core/types.py, line 92-101):**
```python
@dataclass
class LayerSplitConfig:
    enabled: bool = True
    gpu_layers: int = -1                    # All layers on GPU (GGUF)
    max_gpu_memory: Dict[int, str] = {0: "4GB"}  # Per-GPU memory limit
    max_cpu_memory: str = "28GB"            # CPU offload limit
    offload_folder: Optional[Path] = None   # Disk offload directory
    offload_state_dict: bool = True         # Efficient offloading mode
    low_cpu_mem_usage: bool = True          # Minimize RAM during load
    auto_oom_handling: bool = True          # Progressive OOM recovery
```

**Applied in LocalModelLoader.build_load_config():**
```python
if split_config.enabled:
    load_config["device_map"] = "auto"
    load_config["max_memory"] = split_config.get_max_memory_dict()
    load_config["low_cpu_mem_usage"] = True
    if split_config.offload_state_dict or offload_dir:
        load_config["offload_folder"] = str(offload_dir)
        load_config["offload_state_dict"] = True
```

### OOM Recovery (src/utils/memory_manager.py)

**OOMRecoveryStrategy:**
1. Aggressive cache clearing
2. KV cache offload to CPU
3. Context/activation offload to CPU
4. Progressive layer offloading
5. Max 10 attempts with min 1GB GPU requirement

**Applied during model load via with_oom_handling():**
```python
model, final_params = with_oom_handling(
    load_fn=load_model_fn,
    recovery_strategy=recovery_strategy,
    logger=logger
)
```

### Memory Cleanup

**Per-chunk cleanup (stages.py, line 215-218):**
```python
# After every 5 chunks during LLM processing
if llm_backend.provider_identifier.startswith("local") and (i + 1) % 5 == 0:
    if torch and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
```

## 4. KV-CACHE & CONTEXT WINDOW HANDLING

### Context Window Management

**HyperparameterConfig (src/models/hyperparameters.py):**
```python
max_length: int                    # Total sequence length (context + generation)
max_new_tokens: Optional[int]      # Max output tokens (generated)
```

**Dynamic Token Calculation (LocalHFBackend):**
```python
# Line 388 in local_hf.py
calculated_max = self.token_calculator.calculate_max_new_tokens(prompt)
max_new = min(hyperparams.max_new_tokens, calculated_max)
gen_config_dict['max_new_tokens'] = max_new
```

**DynamicTokenLimitCalculator:**
- Estimates available context: context_window - prompt_tokens
- Respects hyperparameter limits
- Provider-specific defaults (varies by backend)

### KV Cache in OOM Recovery

**OOMRecoveryStrategy mentions (memory_manager.py, line 192-198):**
```python
# Strategy 2: Offload KV cache to CPU
if self.attempt_count == 2 and not self.tried_cache_offload:
    strategy['action'] = 'offload_kv_cache'
    strategy['params'] = {'device': 'cpu'}
    self.tried_cache_offload = True
```

Note: KV cache offloading is mentioned but implementation details not exposed in public methods.

### No Explicit Per-Chunk KV Cache Management

- KV cache is managed implicitly by transformers/torch
- Cleared via `torch.cuda.empty_cache()` between chunks
- No explicit context window truncation between chunks

## 5. PIPELINE STAGE EXECUTION FLOW

### Stage Order (pipeline.py, line 198)

```
Stage Execution Sequence:
1. extract       - PDF/text file reading
2. preprocess    - Text cleaning for LLM
3. chunk         - Split into manageable chunks
4. process       - LLM generation (LOADS TEXT MODEL HERE)
5. filter        - Remove thinking tags, merge chunks
6. format        - Apply markdown formatting
7. save          - Write to output file
8. audio         - Audio generation (LOADS AUDIO MODEL HERE)
```

### Stage Dependencies

```
extract
  └─> text extracted
       └─> preprocess
            └─> cleaned text
                 └─> chunk
                      └─> text chunks
                           └─> process (LLM loads here)
                                └─> processed chunks
                                     └─> filter
                                          └─> merged text
                                               ├─> format
                                               │    └─> save
                                               │         └─> output file
                                               └─> audio (if enabled)
                                                    └─> audio file
```

### Checkpoint Resume Points

Each stage can have checkpoints that allow resuming from:
- extract, preprocess, chunk, process (with per-chunk resume index)
- filter, format, save, audio (with per-chunk resume index for long audio)

## 6. SETTINGS/CONFIGURATION SYSTEM

### Memory Profile System (src/config/profiles.py)

Predefined profiles:
- **low_vram** - 2GB GPU, 8GB RAM (8bit quant, limited CPU offload)
- **medium_vram** - 4GB GPU, 16GB RAM (4bit quant, moderate offload)
- **high_vram** - 8GB GPU, 32GB RAM (minimal quant, full GPU)
- **disk_offload** - Limited VRAM + disk swapping

Profile contains:
```python
quantization_method: str          # "none", "4bit", "8bit"
gpu_layers: int                   # How many layers on GPU
max_gpu_memory_per_device: str     # e.g., "4GiB"
max_cpu_memory: str               # e.g., "24GiB"
enable_disk_offload: bool
disk_offload_dir: Optional[Path]
```

### AppState (menu.py, line 142-174)

Menu system tracks:
```python
memory_profile: str = "medium_vram"
quantization: str = "4bit"
gpu_layers: int = DEFAULT_GPU_LAYERS  # -1 (auto)
max_gpu_memory: str = "4GiB"
max_cpu_memory: str = "24GiB"
layer_split_config: LayerSplitConfig
```

### PipelineConfig (core/types.py, line 259-330)

Core execution config:
```python
model_provider: str               # "local_hf", "local_gguf", "openai", etc.
model_specifier: str              # Model ID or path
quantization_config: Optional[QuantizationConfig]
layer_split_config: Optional[LayerSplitConfig]
hyperparameters: HyperparameterConfig
audio_config: Optional[AudioConfig]
enable_checkpoints: bool
checkpoint_interval: int          # Save checkpoint every N chunks
checkpoint_resume_mode: str       # "auto", "interactive", "disabled"
```

### AudioConfig (core/types.py, line 212-236)

Audio-specific settings:
```python
sample_rate: int = 24000
output_format: str = "wav"        # "wav", "mp3", "flac"
chunk_size: int = 1500            # Chars per audio chunk
speed: float = 1.0
volume_normalize: bool = True
device: str = "auto"
use_half_precision: bool = True
enable_cpu_offload: bool = True
enable_disk_offload: bool = False
clear_cache_between_chunks: bool = True
quantization: str = "4bit"        # For audio models
```

### CLI Configuration Points

**Memory Profile Selection (cli.py, line 490):**
```python
quant_config, layer_split_config = create_configs_from_memory_profile(args.memory_profile)

# CLI overrides:
if args.quantization:
    quant_config.method = args.quantization
if args.gpu_layers is not None:
    layer_split_config.gpu_layers = args.gpu_layers
```

**Hyperparameter Configuration (cli.py, line 500):**
```python
hyperparams = handle_hyperparameter_configuration(args)
# Can be from:
# - Preset (--hyperparameter-preset)
# - File (--hyperparameter-file)
# - Interactive editor
```

## 7. SUMMARY TABLE

| Component | Location | Responsibility |
|-----------|----------|-----------------|
| **Text Model Loading** | cli.py:568, pipeline.py:296 | Get backend, call load() |
| **Audio Model Loading** | cli.py:585, pipeline.py:499 | Get backend, call load() |
| **Backend Factory** | models/backends/__init__.py | Instantiate correct backend |
| **LocalHF Loading** | models/backends/local_hf.py | Transformers model loading |
| **Layer Splitting** | LocalModelLoader.build_load_config() | GPU/CPU/Disk allocation |
| **Quantization** | LocalModelLoader.build_load_config() | BitsAndBytes config |
| **OOM Recovery** | utils/memory_manager.py | Progressive offloading |
| **Model Unload** | cli.py:670, pipeline.py:637 | Explicit cleanup between files |
| **Memory Monitoring** | utils/logger.py MemoryMonitor | Track GPU/CPU/RAM usage |
| **Checkpoint System** | io/checkpoints.py | Resume from stage failures |
| **Context Management** | DynamicTokenLimitCalculator | Calculate max_new_tokens |
| **KV Cache** | Implicit in transformers | No explicit per-chunk mgmt |
| **Memory Profiles** | config/profiles.py | Preset quant/offload configs |

## 8. KEY FEATURES CURRENTLY PRESENT

✅ **Lazy Model Loading** - Models load on first use or explicit CLI call
✅ **Explicit Unloading** - Between batch files, at CLI cleanup
✅ **Layer Splitting** - GPU/CPU/Disk device mapping via "auto"
✅ **Quantization Support** - 4bit/8bit/16bit via BitsAndBytes
✅ **OOM Recovery** - Progressive offloading with 10-attempt strategy
✅ **Memory Monitoring** - GPU/CPU/RAM tracking throughout pipeline
✅ **Checkpoint System** - Resume from any stage after failure
✅ **Audio Model Chunking** - Text chunks for audio generation
✅ **Per-5-Chunk Cleanup** - GPU cache clearing during LLM processing
✅ **Memory Profiles** - Preset configurations for different hardware

## 9. NOTABLE GAPS/LIMITATIONS

⚠️ **KV Cache Management** - No explicit per-chunk KV cache offloading (transformers handles implicitly)
⚠️ **Context Window Truncation** - No sliding window for long documents
⚠️ **Unified Memory Config** - LayerSplitConfig used for both LLM and audio
⚠️ **No Selective Unloading** - Either fully load or unload entire model
⚠️ **No Model Caching** - Can't hold multiple models in memory simultaneously
⚠️ **Audio Model Memory** - Limited explicit optimization for audio vs text models
