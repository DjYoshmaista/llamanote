# Layer Split Configuration Validation - Feasibility Analysis
## Date: 2025-11-10

---

## Executive Summary

This document analyzes the feasibility and approach for implementing **runtime-parameter-aware layer split testing** as requested. The goal is to ensure that cached layer split configurations are validated against actual runtime conditions (batch_size, max_tokens, etc.) rather than just testing model loading capability.

**TL;DR Recommendation**:
- ✅ **Implement Approach #3** (Hybrid Reference + Validation) from LAYER_SPLIT_CACHING_ARCHITECTURES.md
- ⚠️ **Use Conservative Estimation** for runtime parameter validation instead of full generation tests
- ✅ **Add User Prompt** for choosing between cached configurations (max GPU vs min GPU)
- ✅ **Implement Fast Validation** using memory estimation formulas, not actual generation

---

## Problem Analysis

### Current Layer Split Testing Behavior

**File**: `src/utils/layer_split_finder.py` (lines 57-153)

**What It Currently Tests**:
```python
def test_split(self, layers_on_gpu: int, load_config: dict, test_forward_pass: bool = True):
    # 1. Builds device map with specified GPU layers
    device_map = self.builder.build_device_map(layers_on_gpu)

    # 2. Loads model with device map
    model = AutoModelForCausalLM.from_pretrained(
        self.model_id,
        **test_config
    )

    # 3. Tests basic forward pass with DUMMY INPUT
    if test_forward_pass:
        first_device = next(model.parameters()).device
        dummy_input = torch.randint(0, 1000, (1, 10), device=first_device)  # ← ONLY 10 TOKENS, BATCH_SIZE=1
        with torch.no_grad():
            _ = model(dummy_input)
```

**Critical Limitation**:
- ✅ Validates: Model loading and device placement
- ❌ Does NOT validate: Actual batch_size, max_new_tokens, KV cache memory consumption
- ❌ Does NOT validate: Real generation memory requirements
- ❌ Does NOT validate: Runtime hyperparameter compatibility

### Gap Between Current Testing and User Requirements

**User's Requirement** (from conversation):
> "Ensure that the layer splitting test and the configurations take into account the other settings, configurations, and hyperparameters set for the current configuration, setup, and model(s). Settings such as batch_size, max_tokens and other relevant settings..."

**What This Means**:
The cached layer split should be validated to work with:
1. **Configured batch_size** (e.g., 4 instead of 1)
2. **Configured max_new_tokens** (e.g., 2048 instead of 10)
3. **Other hyperparameters** (temperature, top_p, etc.)
4. **KV cache memory** (grows with sequence length)

**Example Failure Scenario**:
```
1. Layer split discovered with batch_size=1, dummy 10-token input
2. Works perfectly - 20 layers on GPU, passes test
3. User runs pipeline with batch_size=4, max_tokens=2048
4. OOM error during generation!
5. Reason: KV cache for batch_size=4 × 2048 tokens >> batch_size=1 × 10 tokens
```

---

## Feasibility Analysis: Two Approaches

### Approach A: Full Generation Testing (REJECTED)

**What It Would Entail**:
1. Load model with layer split configuration
2. Run actual generation with configured batch_size and max_tokens
3. Monitor peak memory usage during generation
4. Validate no OOM errors occur

**Implementation**:
```python
def test_split_with_generation(
    layers_on_gpu: int,
    batch_size: int,
    max_tokens: int,
    hyperparams: HyperparameterConfig
) -> Tuple[bool, float]:
    # Load model
    model = load_model_with_device_map(layers_on_gpu)
    tokenizer = load_tokenizer()

    # Create synthetic prompts (batch_size prompts)
    prompts = ["Test prompt " * 50] * batch_size  # ~50 tokens each

    # Tokenize
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    # Generate with actual max_tokens
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=hyperparams.temperature,
            # ... all hyperparameters
        )

    # Measure peak memory
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
    return True, peak_memory
```

**Pros**:
- ✅ 100% accurate validation
- ✅ Tests actual runtime conditions
- ✅ Detects all potential OOM scenarios

**Cons**:
- ❌ **VERY SLOW**: Each test takes 20-60 seconds for full generation
- ❌ **Binary search × generation time**: 5-10 splits tested × 30 seconds = **5-10 minutes** discovery time
- ❌ **Wasteful**: Generating actual text just to measure memory
- ❌ **High memory usage**: Requires almost full VRAM for each test
- ❌ **User experience**: Unacceptable wait time for discovery

**Time Complexity**:
```
Current discovery: ~20-30 seconds total (load-only tests)
Full generation discovery: ~5-10 minutes total (30 seconds per test × 10-15 tests)

Result: 10-20× SLOWER
```

**Verdict**: ❌ **REJECTED** - Too slow, poor user experience

---

### Approach B: Conservative Memory Estimation (RECOMMENDED)

**What It Would Entail**:
1. Use mathematical formulas to estimate memory requirements
2. Calculate KV cache size based on batch_size and max_tokens
3. Add safety margins for conservative estimates
4. Validate layer split against estimated memory needs

**Implementation** (leverages existing `MemoryEstimator`):
```python
def validate_split_for_runtime(
    layers_on_gpu: int,
    batch_size: int,
    max_tokens: int,
    hyperparams: HyperparameterConfig,
    model_config: dict
) -> Tuple[bool, float]:
    """
    Validate layer split against runtime parameters using memory estimation.

    Returns:
        (is_valid, estimated_peak_memory_mb)
    """
    from ..utils.memory_estimator import MemoryEstimator

    # 1. Estimate model size on GPU
    model_size_mb = MemoryEstimator.estimate_model_size(
        model_id=self.model_id,
        quantization=self.quant_method,
        num_parameters=model_config.get('num_parameters')
    )

    # Scale by fraction of layers on GPU
    total_layers = model_config.get('num_layers', 32)
    gpu_model_size = model_size_mb * (layers_on_gpu / total_layers)

    # 2. Estimate KV cache size WITH ACTUAL BATCH_SIZE AND MAX_TOKENS
    kv_cache_mb = MemoryEstimator.estimate_kv_cache_size(
        num_layers=layers_on_gpu,  # Only GPU layers contribute to VRAM KV cache
        hidden_size=model_config.get('hidden_size', 4096),
        num_attention_heads=model_config.get('num_attention_heads', 32),
        max_seq_length=max_tokens + 512,  # max_tokens + average prompt length
        batch_size=batch_size,  # ← ACTUAL BATCH SIZE
        dtype_bytes=2  # fp16/bfloat16
    )

    # 3. Estimate context processing memory
    context_mb = MemoryEstimator.estimate_context_size(
        max_seq_length=max_tokens + 512,
        hidden_size=model_config.get('hidden_size', 4096),
        batch_size=batch_size,
        dtype_bytes=2
    )

    # 4. Calculate total estimated memory
    estimated_peak = gpu_model_size + kv_cache_mb + context_mb

    # 5. Add 20% safety margin for overhead and variations
    estimated_peak_with_margin = estimated_peak * 1.2

    # 6. Check against available VRAM
    available_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    usable_vram = available_vram * 0.85  # Use up to 85% to avoid instability

    is_valid = estimated_peak_with_margin < usable_vram

    return is_valid, estimated_peak_with_margin
```

**Pros**:
- ✅ **FAST**: <1 second per validation (just math, no model loading)
- ✅ **Accurate enough**: MemoryEstimator formulas based on transformer architecture
- ✅ **Conservative**: 20% safety margin prevents OOM edge cases
- ✅ **Runtime-aware**: Uses actual batch_size and max_tokens
- ✅ **Good UX**: Discovery still takes ~20-30 seconds

**Cons**:
- ⚠️ **Not 100% accurate**: Estimation may be off by 10-20%
- ⚠️ **Conservative**: May recommend fewer GPU layers than actually possible
- ⚠️ **Formula-based**: Assumes standard transformer architecture

**Accuracy Assessment**:
The existing `MemoryEstimator` (src/utils/memory_estimator.py) already has formulas for:
- Model size estimation (lines 113-186)
- **KV cache size estimation** (lines 189-228) - **SUPPORTS batch_size parameter!**
- Context size estimation (lines 231-265)

These formulas are based on standard transformer math:
```
KV Cache = 2 (K+V) × batch_size × num_heads × seq_length × head_dim × dtype_bytes × num_layers
```

**Verdict**: ✅ **RECOMMENDED** - Fast, accurate enough, conservative, good UX

---

## Recommended Implementation Architecture

### Phase 1: Implement Hybrid Reference System (Approach #3)

**From LAYER_SPLIT_CACHING_ARCHITECTURES.md**:

1. **Add SplitReference to Configuration**

**File**: `src/core/types.py`
```python
@dataclass
class SplitReference:
    """Lightweight reference to a cached layer split configuration."""
    cache_key: str          # Hash of model_id + quant + GPU type
    layers_on_gpu: int      # Cached layer split value
    validation_hash: str    # Hash of validation parameters
    last_validated: str     # ISO timestamp

    # Validation metadata
    validated_batch_size: int
    validated_max_tokens: int
    validated_vram_mb: float
    estimated_peak_memory_mb: float

@dataclass
class LayerSplitConfig:
    # ... existing fields ...
    split_reference: Optional[SplitReference] = None  # NEW
```

2. **Create Layer Split Cache File**

**Location**: `cache/layer_splits/split_{cache_key}.json`
```json
{
  "cache_key": "0b529aad5e27",
  "created_timestamp": "2025-11-10T10:30:15",
  "model_info": {
    "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "quantization": "4bit",
    "num_layers": 28,
    "hidden_size": 1536,
    "num_attention_heads": 12
  },
  "hardware_info": {
    "gpu_name": "NVIDIA GeForce RTX 3060",
    "total_vram_mb": 12288,
    "cuda_version": "12.1"
  },
  "configurations": {
    "max_gpu_layers": {
      "layers_on_gpu": 24,
      "estimated_peak_memory_mb": 9830,
      "validated_batch_size": 4,
      "validated_max_tokens": 2048,
      "safety_margin": 1.2,
      "last_validated": "2025-11-10T10:32:45"
    },
    "min_gpu_layers": {
      "layers_on_gpu": 16,
      "estimated_peak_memory_mb": 6420,
      "validated_batch_size": 4,
      "validated_max_tokens": 2048,
      "safety_margin": 1.2,
      "last_validated": "2025-11-10T10:32:45"
    }
  }
}
```

**Why Two Configurations?**:
- **max_gpu_layers**: Maximum possible GPU layers for best performance (uses ~85% VRAM)
- **min_gpu_layers**: Conservative split for stability (uses ~50% VRAM)
- User can choose based on workload (max for speed, min for stability)

3. **Enhance Layer Split Discovery**

**File**: `src/utils/layer_split_finder.py`

Add new method:
```python
def discover_with_runtime_validation(
    self,
    batch_size: int,
    max_tokens: int,
    hyperparams: HyperparameterConfig,
    safety_margin: float = 1.2
) -> Tuple[int, int]:
    """
    Discover layer split with runtime parameter validation.

    Returns:
        (max_gpu_layers, min_gpu_layers)
    """
    # Step 1: Binary search for maximum GPU layers (as before)
    max_layers = self._binary_search_max_layers()

    # Step 2: Validate max_layers with runtime parameters
    is_valid, estimated_peak = self._validate_split_for_runtime(
        layers_on_gpu=max_layers,
        batch_size=batch_size,
        max_tokens=max_tokens,
        safety_margin=safety_margin
    )

    if not is_valid:
        # Reduce layers until validation passes
        max_layers = self._find_valid_split_for_runtime(
            initial_layers=max_layers,
            batch_size=batch_size,
            max_tokens=max_tokens,
            safety_margin=safety_margin
        )

    # Step 3: Find conservative minimum (50% VRAM usage)
    min_layers = self._find_conservative_split(
        max_layers=max_layers,
        target_vram_usage=0.5,
        batch_size=batch_size,
        max_tokens=max_tokens
    )

    return max_layers, min_layers
```

### Phase 2: Add User Prompt at Pipeline Execution

**File**: `src/core/pipeline.py` (at start of `process_file()`)

```python
def process_file(self, input_path: Path, ...) -> PipelineResult:
    # ... existing initialization ...

    # Check for cached layer split configurations
    if self.config.layer_split.auto_discover_splits:
        split_cache_mgr = LayerSplitCacheManager()
        cache_key = split_cache_mgr.generate_cache_key(
            model_id=self.config.model_specifier,
            quantization=self.config.quantization.method,
            gpu_name=self.device_manager.get_gpu_name()
        )

        cached_config = split_cache_mgr.load_cached_split(cache_key)

        if cached_config:
            # Validate cache against current hardware
            if split_cache_mgr.validate_hardware_match(cached_config):
                # Prompt user for choice
                choice = self._prompt_layer_split_choice(cached_config)

                if choice == "max":
                    layers_on_gpu = cached_config["configurations"]["max_gpu_layers"]["layers_on_gpu"]
                    self.logger.info(f"Using cached max GPU layers: {layers_on_gpu}")
                    self.config.layer_split.split_reference = SplitReference(
                        cache_key=cache_key,
                        layers_on_gpu=layers_on_gpu,
                        # ... copy from cached_config ...
                    )
                elif choice == "min":
                    layers_on_gpu = cached_config["configurations"]["min_gpu_layers"]["layers_on_gpu"]
                    self.logger.info(f"Using cached min GPU layers: {layers_on_gpu}")
                    # ... same as above ...
                else:  # choice == "rediscover"
                    self.logger.info("User requested re-discovery of layer split")
                    # Fall through to discovery below
            else:
                self.logger.warning("Cached layer split invalid for current hardware, re-discovering...")
        else:
            self.logger.info("No cached layer split found, running discovery...")

        # If no cached split or user chose rediscover
        if not self.config.layer_split.split_reference:
            max_layers, min_layers = self._run_layer_split_discovery()
            split_cache_mgr.save_split_configuration(cache_key, max_layers, min_layers, ...)
```

**User Prompt** (via rich.Prompt or simple input()):
```
╭─ Layer Split Configuration ─────────────────────────────────╮
│ Found cached layer split for:                               │
│   Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (4bit)  │
│   GPU: NVIDIA GeForce RTX 3060 (12GB)                      │
│                                                             │
│ Available configurations:                                   │
│   [1] Max Performance (24/28 layers on GPU, ~9.8GB VRAM)   │
│   [2] Stable/Conservative (16/28 layers on GPU, ~6.4GB)    │
│   [3] Re-discover layer split (takes ~30 seconds)          │
│                                                             │
│ Choice [1/2/3]:                                             │
╰─────────────────────────────────────────────────────────────╯
```

### Phase 3: Integration with Configuration Presets

**File**: `src/menu.py` (modify `_save_configuration()`)

```python
def _save_configuration(self, config: PipelineConfig, preset_name: str):
    # ... existing save logic ...

    # Save layer split reference if present
    if config.layer_split.split_reference:
        config_dict["layer_split_reference"] = {
            "cache_key": config.layer_split.split_reference.cache_key,
            "layers_on_gpu": config.layer_split.split_reference.layers_on_gpu,
            "validated_batch_size": config.layer_split.split_reference.validated_batch_size,
            "validated_max_tokens": config.layer_split.split_reference.validated_max_tokens,
            "last_validated": config.layer_split.split_reference.last_validated
        }

    # ... rest of save ...
```

**File**: `src/menu.py` (modify `_load_configuration()`)

```python
def _load_configuration(self, preset_name: str) -> PipelineConfig:
    # ... existing load logic ...

    # Load layer split reference if present
    if "layer_split_reference" in config_dict:
        ref_data = config_dict["layer_split_reference"]
        config.layer_split.split_reference = SplitReference(
            cache_key=ref_data["cache_key"],
            layers_on_gpu=ref_data["layers_on_gpu"],
            # ... restore all fields ...
        )

        # Validate reference is still valid
        split_cache_mgr = LayerSplitCacheManager()
        cached_config = split_cache_mgr.load_cached_split(ref_data["cache_key"])
        if not cached_config or not split_cache_mgr.validate_hardware_match(cached_config):
            logger.warning(f"Layer split reference in preset '{preset_name}' is no longer valid")
            config.layer_split.split_reference = None

    return config
```

---

## Implementation Roadmap

### Step 1: Create Layer Split Cache Manager (NEW FILE)
**File**: `src/io/layer_split_cache.py`
**Estimated Time**: 2-3 hours

**Responsibilities**:
- Generate cache keys from model/GPU/quant
- Load/save cached split configurations
- Validate hardware compatibility
- Manage cache directory structure

**Key Methods**:
```python
class LayerSplitCacheManager:
    def generate_cache_key(model_id, quantization, gpu_name) -> str
    def load_cached_split(cache_key) -> Optional[Dict]
    def save_split_configuration(cache_key, max_layers, min_layers, ...) -> bool
    def validate_hardware_match(cached_config) -> bool
    def list_cached_splits() -> List[Dict]
    def delete_cached_split(cache_key) -> bool
```

### Step 2: Enhance Layer Split Finder (MODIFY)
**File**: `src/utils/layer_split_finder.py`
**Estimated Time**: 3-4 hours

**New Methods**:
```python
def _validate_split_for_runtime(layers_on_gpu, batch_size, max_tokens, safety_margin) -> Tuple[bool, float]
def _find_valid_split_for_runtime(initial_layers, batch_size, max_tokens, safety_margin) -> int
def _find_conservative_split(max_layers, target_vram_usage, batch_size, max_tokens) -> int
def discover_with_runtime_validation(batch_size, max_tokens, hyperparams, safety_margin) -> Tuple[int, int]
```

**Integration**:
- Use MemoryEstimator for calculations
- Add logging for validation steps
- Return both max and min layer configurations

### Step 3: Update Configuration Types (MODIFY)
**File**: `src/core/types.py`
**Estimated Time**: 30 minutes

**Changes**:
- Add `SplitReference` dataclass
- Add `split_reference: Optional[SplitReference]` to `LayerSplitConfig`

### Step 4: Integrate User Prompt in Pipeline (MODIFY)
**File**: `src/core/pipeline.py`
**Estimated Time**: 2-3 hours

**Changes**:
- Check for cached split at pipeline start
- Prompt user for choice (max/min/rediscover)
- Apply chosen configuration
- Handle cache miss (run discovery)

### Step 5: Integrate with Configuration Presets (MODIFY)
**File**: `src/menu.py`
**Estimated Time**: 1-2 hours

**Changes**:
- Save split reference when saving configuration
- Load and validate split reference when loading configuration
- Display split info in configuration summary

### Step 6: Testing & Validation
**Estimated Time**: 2-3 hours

**Test Cases**:
- [ ] Discovery with different batch_sizes (1, 2, 4, 8)
- [ ] Discovery with different max_tokens (512, 1024, 2048, 4096)
- [ ] Cache save/load cycle
- [ ] Hardware mismatch detection
- [ ] User prompt flow
- [ ] Integration with configuration presets
- [ ] Memory estimation accuracy validation

---

## Expected Outcomes

### Performance Characteristics

**Discovery Time**:
- Current: ~20-30 seconds (load-only tests)
- With validation: ~25-35 seconds (+5 seconds for estimation calculations)
- **Impact**: Minimal (20% increase in discovery time)

**Memory Estimation Accuracy**:
- Expected accuracy: ±15% of actual memory usage
- Safety margin: 20% buffer
- **Result**: Conservative but safe (may recommend 1-2 fewer GPU layers than theoretically possible)

**User Experience**:
```
First Run (no cache):
1. User starts pipeline
2. Discovery runs: "Discovering optimal layer split... (30 seconds)"
3. Saves max and min configurations to cache
4. Uses discovered configuration
5. Total delay: ~30 seconds (one-time)

Subsequent Runs (with cache):
1. User starts pipeline
2. Prompt: "Use cached split? [max/min/rediscover]"
3. User chooses: "max"
4. Pipeline starts immediately
5. Total delay: ~2 seconds (user input)
```

### Failure Modes & Handling

**Scenario 1: Cache exists but hardware changed**
```
Detection: GPU name or VRAM size mismatch
Action: Auto-invalidate cache, run discovery
Message: "Hardware changed, re-discovering layer split..."
```

**Scenario 2: Cached split fails during actual generation**
```
Detection: OOM error during pipeline execution
Action: Reduce GPU layers by 2, retry
Message: "OOM detected, reducing GPU layers to {new_layers}..."
Logging: Report discrepancy for cache update
```

**Scenario 3: Estimation significantly off**
```
Detection: Actual memory usage >> estimated (>30% difference)
Action: Update cache with actual measured memory
Message: "Updating layer split cache with actual measurements..."
```

---

## Risks & Mitigations

### Risk 1: Memory Estimation Inaccuracy
**Probability**: Medium
**Impact**: Medium (may cause OOM if estimate too low)

**Mitigation**:
- Use 20% safety margin
- Conservative default (prefer fewer GPU layers)
- Fallback handling (reduce layers on OOM)
- Collect actual measurements to improve estimates over time

### Risk 2: Cache Invalidation Edge Cases
**Probability**: Low
**Impact**: Low (unnecessary re-discovery)

**Mitigation**:
- Robust hardware detection (GPU name, VRAM, CUDA version)
- Version cache files (detect format changes)
- User option to force re-discovery

### Risk 3: User Confusion About Choices
**Probability**: Medium
**Impact**: Low (suboptimal performance)

**Mitigation**:
- Clear prompt with VRAM usage shown
- Provide recommendations ("max for best performance")
- Document in user guide
- Default to "max" if user presses Enter

---

## Comparison to Alternative Approaches

### Rejected: Full Generation Testing
- **Time**: 5-10 minutes per discovery
- **Accuracy**: 100%
- **UX**: Poor (too slow)
- **Verdict**: Not worth the accuracy gain

### Recommended: Conservative Estimation
- **Time**: 25-35 seconds per discovery
- **Accuracy**: ~85% (±15%)
- **UX**: Excellent (fast, cached)
- **Verdict**: Best balance of speed and safety

### Alternative: Hybrid (Estimation + Limited Generation)
- **Time**: 1-2 minutes per discovery
- **Accuracy**: ~95%
- **UX**: Acceptable
- **Verdict**: Could implement as future enhancement

---

## Conclusion

### Recommended Approach is Sound ✅

The requested architecture is **feasible and practical** with the following modifications:

1. ✅ **Use Approach #3** (Hybrid Reference + Validation) from architecture doc
2. ✅ **Validate with memory estimation** instead of full generation tests
3. ✅ **Discover TWO configurations**: max GPU (performance) and min GPU (stability)
4. ✅ **Prompt user** at pipeline execution to choose cached configuration
5. ✅ **Integrate with configuration presets** for persistence
6. ✅ **Use conservative safety margins** (20%) to prevent OOM

### Key Benefits

- **Fast**: Discovery only ~30 seconds (vs 5-10 minutes for full generation)
- **Safe**: 20% safety margin + conservative estimation prevents OOM
- **User-Friendly**: Cached splits reused across runs (2 second startup)
- **Runtime-Aware**: Validates batch_size and max_tokens requirements
- **Flexible**: User chooses between performance (max) and stability (min)
- **Integrated**: Works with configuration preset system

### Implementation Effort

**Total Estimated Time**: 12-16 hours
- Phase 1 (Cache Manager): 2-3 hours
- Phase 2 (Enhanced Discovery): 3-4 hours
- Phase 3 (Type Updates): 30 minutes
- Phase 4 (Pipeline Integration): 2-3 hours
- Phase 5 (Preset Integration): 1-2 hours
- Phase 6 (Testing): 2-3 hours

### Next Steps

1. **Confirm Approach**: User approval of conservative estimation method
2. **Implement Phase 1**: Create LayerSplitCacheManager
3. **Implement Phase 2**: Enhance layer split discovery with validation
4. **Implement Phases 3-5**: Integration and user prompts
5. **Test & Validate**: Comprehensive testing with various configurations

---

**Status**: ✅ **READY FOR IMPLEMENTATION**
**Recommendation**: Proceed with implementation using conservative memory estimation approach

---

End of Feasibility Analysis
