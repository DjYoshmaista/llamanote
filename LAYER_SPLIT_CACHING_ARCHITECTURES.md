# Layer Split Caching Architecture Proposals
## Problem Analysis & Solution Approaches
### Date: 2025-11-09

---

## Problem Statement

**Current Behavior:**
The auto-configuration system runs layer split discovery **every time** the pipeline executes, even when using the same model with the same quantization settings. This is:
- **Time-consuming**: Takes 20-30 seconds to test different layer splits
- **Unnecessary**: The optimal split for a given model/quantization/GPU memory combination doesn't change
- **Redundant**: Discovery results are already saved to `cache/layer_splits/split_*.json` but not utilized by configuration presets
- **User-frustrating**: Delays pipeline execution even for well-tested configurations

**Existing Infrastructure:**
- `LayerSplitConfigManager` already saves/loads split configs to `cache/layer_splits/split_*.json`
- `AutoLayerSplitDiscovery` tracks tested models in `auto_discovery/discovery_state.json`
- Configuration presets saved via menu option #5 to `config/presets/*.json`
- `PipelineConfig` has `layer_split_config: Optional[LayerSplitConfig]` field
- `LayerSplitConfig` contains settings but not discovered device map

**User Request:**
Create an automatically saved and loaded configuration system that:
1. Ties discovered layer splits to saved configuration presets
2. Avoids re-running discovery on subsequent executions
3. Allows user approval/verification of cached splits
4. Maintains atomicity of configuration files
5. Handles cases where hardware changes (different GPU/memory available)

---

## Architecture Approach #1: Embedded Layer Split in Configuration Presets

### Overview
Store the discovered layer split **directly inside** the configuration preset JSON file as part of the `layer_split_config` field.

### Architecture

```
config/presets/my_podcast.json
├── mode: "podcast"
├── model_provider: "local_hf"
├── model_specifier: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
├── quantization_config: {...}
├── layer_split_config:
│   ├── enabled: true
│   ├── gpu_layers: -1
│   ├── auto_discover_splits: false  ← Disabled after first discovery
│   └── discovered_device_map:       ← NEW FIELD
│       ├── config_hash: "a3f9c2d1e8b4"
│       ├── model_id: "deepseek-ai/..."
│       ├── quant_method: "4bit"
│       ├── gpu_memory_gb: 4.0
│       ├── device_map: {...}        ← The actual discovered map
│       ├── layers_on_gpu: 24
│       ├── layers_on_cpu: 8
│       └── discovered_at: "2025-11-09T14:32:18"
└── ... (other config fields)
```

### Implementation Details

**Changes Required:**
1. Add `discovered_device_map: Optional[Dict[str, Any]]` to `LayerSplitConfig` dataclass
2. Modify pipeline initialization:
   - If `layer_split_config.discovered_device_map` exists → Use it directly
   - If not exists → Run discovery, then save to config
3. Add "Update Layer Split" menu option to refresh if hardware changes
4. On config save, include the discovered device map

**Flow:**
```
First Run:
  User loads preset "my_podcast"
    → layer_split_config.discovered_device_map is None
    → Run auto discovery
    → Save discovered map to preset JSON
    → Continue with pipeline

Subsequent Runs:
  User loads preset "my_podcast"
    → layer_split_config.discovered_device_map exists
    → Use saved device map directly (skip discovery)
    → Continue with pipeline immediately

Hardware Change:
  User selects "Update Layer Split" from menu
    → Re-run discovery
    → Overwrite discovered_device_map in preset
    → Continue
```

### Pros
✅ **Simple**: Single source of truth - one file contains everything
✅ **Atomic**: Configuration and layer split are always in sync
✅ **No File Management**: No separate cache files to manage
✅ **Clear Ownership**: Each preset owns its layer split
✅ **Easy Backup**: Copying preset file includes layer split
✅ **No Hash Mismatches**: Config and split can't get out of sync

### Cons
❌ **Duplication**: If multiple presets use same model, each stores its own copy
❌ **Large Files**: Device maps can be 100-200 lines of JSON
❌ **Version Control Noise**: Git diffs become harder to read
❌ **No Sharing**: Can't reuse discovered splits across presets
❌ **Manual Updates**: User must manually update each preset if GPU changes
❌ **No Centralized Cache**: Loses benefit of existing `LayerSplitConfigManager`

### Configuration Changes
```python
# src/core/types.py
@dataclass
class LayerSplitConfig:
    enabled: bool = True
    gpu_layers: int = -1
    auto_discover_splits: bool = True

    # NEW: Embedded discovered device map
    discovered_device_map: Optional[Dict[str, Any]] = None
    # Contains: config_hash, model_id, quant_method, gpu_memory_gb,
    #           device_map, layers_on_gpu, layers_on_cpu, discovered_at
```

### User Interaction Flow
1. User creates new preset with model "deepseek-r1"
2. First pipeline run: Discovery runs (20s), saves to preset
3. User saves preset: Device map included automatically
4. Next run: "Found cached layer split, using directly" → instant start
5. If GPU changes: Menu option "Refresh Layer Splits" → re-discovers all presets

---

## Architecture Approach #2: Separate Layer Split Cache with References

### Overview
Keep layer splits in separate cache files (`cache/layer_splits/split_*.json`) and store only a **reference hash** in the configuration preset.

### Architecture

```
config/presets/my_podcast.json
├── ... (other config)
├── layer_split_config:
│   ├── enabled: true
│   ├── auto_discover_splits: true
│   └── cached_split_hash: "a3f9c2d1e8b4"  ← NEW: Reference to cache

cache/layer_splits/split_a3f9c2d1e8b4.json
├── model_id: "deepseek-ai/..."
├── quant_method: "4bit"
├── gpu_memory_gb: 4.0
├── min_split: {...}
├── max_split: {...}
└── created_at: "2025-11-09T14:32:18"

cache/layer_splits/.split_registry.json
├── "a3f9c2d1e8b4":
│   ├── model_id: "deepseek-ai/..."
│   ├── quant_method: "4bit"
│   ├── gpu_memory_gb: 4.0
│   ├── referenced_by: ["my_podcast", "quick_notes"]
│   └── last_verified: "2025-11-09T14:32:18"
└── ... (other hashes)
```

### Implementation Details

**Changes Required:**
1. Add `cached_split_hash: Optional[str]` to `LayerSplitConfig` dataclass
2. Create `SplitCacheRegistry` class to track references
3. Pipeline initialization:
   - Check if `cached_split_hash` exists
   - Load from cache using hash
   - Verify hash matches current model/quant/memory
   - If mismatch: warn user, offer to re-discover
4. On config save: Include hash reference
5. Add "Clean Unused Splits" menu option

**Flow:**
```
First Run:
  User loads preset → cached_split_hash is None
  → Run discovery
  → Generate hash: hash(model_id + quant + gpu_mem)
  → Save to cache/layer_splits/split_{hash}.json
  → Update preset with hash
  → Add preset name to registry under this hash

Subsequent Runs:
  User loads preset → cached_split_hash = "a3f9c2d1e8b4"
  → Look up cache/layer_splits/split_a3f9c2d1e8b4.json
  → If exists: load device map (instant)
  → If not exists: warn "cache missing", re-discover

Hash Mismatch:
  User changed GPU from 4GB to 8GB
  → Hash no longer matches
  → Warn: "GPU memory changed (4GB → 8GB), cached split invalid"
  → Offer: [Re-discover] [Use Anyway] [Cancel]
```

### Pros
✅ **No Duplication**: Multiple presets can reference same cached split
✅ **Small Presets**: Only stores hash reference (12 chars)
✅ **Shared Cache**: Reuses existing `LayerSplitConfigManager` infrastructure
✅ **Easy Updates**: Update cache once, all presets benefit
✅ **Garbage Collection**: Can detect and clean unused cache files
✅ **Clear Separation**: Configuration vs. hardware-specific cache
✅ **Registry Tracking**: Know which presets use which splits

### Cons
❌ **Two Files**: Preset + cache file (can get out of sync)
❌ **Reference Complexity**: Need to validate references on load
❌ **Cache Management**: User must understand cache directory
❌ **Orphaned Caches**: Cache files can accumulate if presets deleted
❌ **Missing Cache Risk**: If cache deleted, preset breaks
❌ **Hash Collisions**: Theoretical risk (low probability)

### Configuration Changes
```python
# src/core/types.py
@dataclass
class LayerSplitConfig:
    enabled: bool = True
    gpu_layers: int = -1
    auto_discover_splits: bool = True

    # NEW: Reference to cached split
    cached_split_hash: Optional[str] = None
    # Hash format: first 12 chars of SHA256(model_id + quant + gpu_mem)
```

### User Interaction Flow
1. First run: Discovery happens, saves to cache, stores hash in preset
2. User saves preset: Hash reference saved
3. Next run: Loads from cache using hash → instant
4. Menu option "View Layer Split Cache" shows:
   - All cached splits
   - Which presets reference each
   - Option to refresh or delete unused
5. If cache missing: Auto-regenerate with confirmation

---

## Architecture Approach #3: Hybrid - Lightweight Reference + Validation Metadata

### Overview
Store a **lightweight reference** in preset with enough metadata to validate, but keep full device map in separate cache. Best of both worlds.

### Architecture

```
config/presets/my_podcast.json
├── ... (other config)
├── layer_split_config:
│   ├── enabled: true
│   ├── auto_discover_splits: false
│   └── split_reference:              ← NEW: Hybrid reference
│       ├── cache_hash: "a3f9c2d1e8b4"
│       ├── model_id: "deepseek-ai/..."  ← For validation
│       ├── quant_method: "4bit"         ← For validation
│       ├── gpu_memory_gb: 4.0           ← For validation
│       ├── layers_on_gpu: 24            ← Quick preview
│       ├── layers_on_cpu: 8             ← Quick preview
│       └── verified_at: "2025-11-09"    ← Last verification

cache/layer_splits/split_a3f9c2d1e8b4.json
├── [Full device map as before]
```

### Implementation Details

**Changes Required:**
1. Add `split_reference: Optional[Dict[str, Any]]` to `LayerSplitConfig`
2. Reference contains: hash + validation metadata + preview
3. Pipeline initialization:
   - Check if `split_reference` exists
   - Validate metadata matches current hardware
   - If valid: load full map from cache
   - If invalid: show mismatch, offer options
4. On config save: Include reference with metadata

**Flow:**
```
First Run:
  Run discovery → Save to cache → Create reference
  Reference = {
    cache_hash, model_id, quant, memory,
    layers preview, verification timestamp
  }

Load Preset:
  Read split_reference
  → Validate: model matches? ✓
  → Validate: quant matches? ✓
  → Validate: GPU memory matches? ✗ (4GB config, but 8GB available)
  → Warn: "Config expects 4GB GPU, you have 8GB"
  → Options: [Use Anyway] [Re-discover for 8GB] [Cancel]

If "Use Anyway":
  → Load cached split for 4GB
  → Continue (might not be optimal)

If "Re-discover":
  → Run discovery for 8GB
  → Update reference
  → Continue with new split
```

### Pros
✅ **Self-Validating**: Contains enough info to detect mismatches
✅ **User-Friendly**: Shows what's expected vs. what's available
✅ **Quick Preview**: Can see layer split without loading cache
✅ **Efficient**: Still avoids duplication (cache shared)
✅ **Robust**: Handles hardware changes gracefully
✅ **Transparent**: User sees exactly what's being used
✅ **Best of Both**: Combines Approach #1 and #2 advantages

### Cons
❌ **Medium Complexity**: More sophisticated than #1 or #2
❌ **Two Places to Update**: Reference + cache file
❌ **Validation Logic**: Need careful validation code
❌ **Larger Presets**: More data than just hash (but still manageable)

### Configuration Changes
```python
# src/core/types.py
@dataclass
class SplitReference:
    """Reference to cached layer split with validation metadata."""
    cache_hash: str
    model_id: str
    quant_method: str
    gpu_memory_gb: float
    layers_on_gpu: int
    layers_on_cpu: int
    verified_at: str  # ISO timestamp

@dataclass
class LayerSplitConfig:
    enabled: bool = True
    gpu_layers: int = -1
    auto_discover_splits: bool = True

    # NEW: Hybrid reference
    split_reference: Optional[SplitReference] = None
```

### User Interaction Flow
1. First run: Discovery, creates reference with metadata
2. User views preset in menu: Shows "Layer Split: 24 GPU / 8 CPU (verified 2025-11-09)"
3. User loads preset on different GPU: Auto-detects mismatch, offers options
4. Menu shows "Layer Split Status" with validation info
5. Can force re-validation from menu

---

## Architecture Approach #4: Smart Auto-Detection with Approval Workflow

### Overview
Don't store layer split in preset at all. Instead, implement smart auto-detection that:
1. Detects hardware on startup
2. Checks cache for matching configuration
3. Asks user for approval before using cached split
4. Falls back to discovery if no cache or user declines

### Architecture

```
config/presets/my_podcast.json
├── ... (other config)
├── layer_split_config:
│   ├── enabled: true
│   ├── auto_discover_splits: true
│   └── require_user_approval: true  ← NEW: Approval flag

cache/layer_splits/split_a3f9c2d1e8b4.json
├── [Same as before]

cache/layer_splits/.approval_history.json
├── "a3f9c2d1e8b4":
│   ├── approved_by_user: true
│   ├── approved_at: "2025-11-09T14:32:18"
│   ├── used_count: 42
│   └── last_used: "2025-11-10T09:15:23"
```

### Implementation Details

**Changes Required:**
1. Add `require_user_approval: bool = True` to `LayerSplitConfig`
2. Create `SplitApprovalManager` class
3. Pipeline startup flow:
   ```python
   def load_layer_split():
       hardware = detect_hardware()  # model, quant, GPU memory
       cache_hash = generate_hash(hardware)

       cached_split = load_from_cache(cache_hash)
       if cached_split:
           if config.require_user_approval:
               approved = ask_user_approval(cached_split)
               if not approved:
                   return run_discovery()
           return cached_split
       else:
           return run_discovery()
   ```
4. Add approval UI:
   ```
   Found cached layer split for this configuration:
     Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
     Quantization: 4bit
     GPU Memory: 4.0GB
     Split: 24 GPU / 8 CPU layers
     Last used: 2 days ago (42 times)

   Use this cached split? [Y/n/always/never]:
   ```

**Flow:**
```
Pipeline Start:
  Detect: model="deepseek-r1", quant="4bit", gpu="4GB"
  → Generate hash
  → Check cache
  → Found cached split!
  → Check approval history
  → If first time: Ask user
  → If previously approved: Use automatically
  → If previously declined: Skip cache, discover

User Approval Options:
  "y" - Use this time only
  "n" - Don't use, re-discover
  "always" - Always use cached splits (disable approval)
  "never" - Never use this split (blacklist it)
```

### Pros
✅ **Zero Config Storage**: No split data in preset files
✅ **User Control**: Explicit approval workflow
✅ **Audit Trail**: Tracks approval history
✅ **Flexible**: User can set "always approve" or "always discover"
✅ **Safe**: User sees what's being used before accepting
✅ **Smart Defaults**: "Always approve" after first approval
✅ **Statistics**: Shows usage count (confidence builder)

### Cons
❌ **User Interaction**: Requires user input (can be annoying)
❌ **No Preset Binding**: Layer split not tied to specific preset
❌ **Approval Fatigue**: User might get approval prompts too often
❌ **Complex Logic**: Approval history tracking adds complexity
❌ **Not Fully Automatic**: Defeats purpose if "require approval" always on

### Configuration Changes
```python
# src/core/types.py
@dataclass
class LayerSplitConfig:
    enabled: bool = True
    gpu_layers: int = -1
    auto_discover_splits: bool = True

    # NEW: Approval settings
    require_user_approval: bool = True
    # If True, ask before using cached split
    # If False, use cached split automatically
```

### User Interaction Flow
1. First pipeline run: Discovery runs, saves to cache
2. Next run: Detects cached split, asks approval:
   ```
   Use cached layer split? [Y]es / [n]o / [a]lways / n[e]ver:
   ```
3. User types "a" (always): Future runs skip approval
4. Menu option "Layer Split Preferences":
   - Enable/disable approval requirement
   - View approval history
   - Clear blacklisted splits
   - Set default action (always/never/ask)

---

## Architecture Approach #5: Configuration Profiles with Layer Split Inheritance

### Overview
Create a **two-tier system**: Base hardware profiles + Configuration presets that inherit from them.

### Architecture

```
config/hardware_profiles/gtx1650ti_4gb.json
├── profile_name: "GTX 1650 Ti (4GB)"
├── gpu_device: "NVIDIA GeForce GTX 1650 Ti"
├── gpu_memory_gb: 4.0
├── cpu_memory_gb: 28.0
├── layer_splits:
│   ├── "deepseek-r1:4bit":
│   │   ├── cache_hash: "a3f9c2d1e8b4"
│   │   ├── layers_on_gpu: 24
│   │   └── layers_on_cpu: 8
│   ├── "qwen2.5-3b:4bit":
│   │   ├── cache_hash: "b4e7f3c2a9d1"
│   │   ├── layers_on_gpu: 28
│   │   └── layers_on_cpu: 4
│   └── ... (other models)
└── created_at: "2025-11-09"

config/presets/my_podcast.json
├── ... (other config)
├── hardware_profile: "gtx1650ti_4gb"  ← NEW: Reference to profile
├── layer_split_config:
│   ├── enabled: true
│   └── inherit_from_profile: true     ← NEW: Inheritance flag

cache/layer_splits/split_*.json
├── [Full device maps as before]
```

### Implementation Details

**Changes Required:**
1. Create `HardwareProfile` class and storage system
2. Add `hardware_profile: Optional[str]` to `PipelineConfig`
3. Add `inherit_from_profile: bool = True` to `LayerSplitConfig`
4. On startup:
   - Load hardware profile
   - Get model-specific split from profile
   - If not in profile: discover and add to profile
5. Menu system:
   - "Manage Hardware Profiles" submenu
   - Auto-detect and create profile on first run
   - Switch between profiles
   - Export/import profiles

**Flow:**
```
First Run (New Hardware):
  Detect hardware → Create profile "gtx1650ti_4gb"
  Load preset "my_podcast"
  → Model: "deepseek-r1", Quant: "4bit"
  → Check profile for this model
  → Not found in profile
  → Run discovery
  → Add to profile: "deepseek-r1:4bit" → split_hash

Second Run (Same Model):
  Load profile "gtx1650ti_4gb"
  Load preset "my_podcast"
  → Model: "deepseek-r1", Quant: "4bit"
  → Check profile: Found!
  → Load from cache using hash
  → Instant start

Different Model (First Time):
  Load preset "quick_notes"
  → Model: "qwen2.5-3b", Quant: "4bit"
  → Check profile: Not found
  → Run discovery
  → Add to profile

Moving to New Machine:
  Export profile "gtx1650ti_4gb.json"
  → Copy to new machine
  → Import profile
  → All model splits available immediately
```

### Pros
✅ **Hardware-Centric**: One profile per GPU configuration
✅ **Reusable**: All presets on same hardware share profiles
✅ **Portable**: Export/import profiles between machines
✅ **Organized**: Clear separation: hardware vs. processing settings
✅ **Efficient**: Discover once per model per hardware
✅ **Scalable**: Easy to add new models to existing profile
✅ **Future-Proof**: Can add more hardware-specific settings later
✅ **Cloud-Friendly**: Could sync profiles across machines

### Cons
❌ **Most Complex**: Three-tier system (preset → profile → cache)
❌ **New Concept**: Users must understand hardware profiles
❌ **Profile Management**: Need UI to manage profiles
❌ **Indirection**: More layers between config and device map
❌ **Auto-Detection**: Must auto-detect hardware accurately
❌ **Profile Mismatch**: What if hardware changes slightly?

### Configuration Changes
```python
# NEW: src/config/hardware_profile.py
@dataclass
class HardwareProfile:
    profile_name: str
    gpu_device: str
    gpu_memory_gb: float
    cpu_memory_gb: float
    layer_splits: Dict[str, Dict[str, Any]]  # "model:quant" → split info
    created_at: str
    last_updated: str

# src/core/types.py
@dataclass
class PipelineConfig:
    ... (existing fields)

    # NEW: Reference to hardware profile
    hardware_profile: Optional[str] = None
    # If None, auto-detect on startup

@dataclass
class LayerSplitConfig:
    enabled: bool = True
    gpu_layers: int = -1

    # NEW: Inheritance from profile
    inherit_from_profile: bool = True
    # If True, get split from hardware profile
    # If False, use embedded/cached split
```

### User Interaction Flow
1. **First Run**: Auto-creates profile "gtx1650ti_4gb"
2. **Menu Option**: "View Hardware Profile"
   ```
   Current Hardware Profile: GTX 1650 Ti (4GB)

   Configured Models:
     1. deepseek-r1 (4bit) - 24 GPU / 8 CPU layers
     2. qwen2.5-3b (4bit) - 28 GPU / 4 CPU layers

   Options:
     [v] View device maps
     [r] Refresh all splits
     [e] Export profile
     [i] Import profile
     [s] Switch profile
     [c] Create new profile
   ```
3. **Export**: Saves `gtx1650ti_4gb.json` to `config/hardware_profiles/`
4. **Import**: On new machine, import profile, all splits available
5. **Switch**: If user has multiple GPUs, switch between profiles

---

## Comparison Matrix

| Feature | Approach #1 Embedded | Approach #2 Separate Cache | Approach #3 Hybrid | Approach #4 Auto-Detect | Approach #5 Profiles |
|---------|---------------------|---------------------------|-------------------|------------------------|---------------------|
| **Simplicity** | ⭐⭐⭐⭐⭐ Very Simple | ⭐⭐⭐ Moderate | ⭐⭐⭐ Moderate | ⭐⭐ Complex | ⭐ Most Complex |
| **No Duplication** | ❌ Duplicates | ✅ Shared Cache | ✅ Shared Cache | ✅ Shared Cache | ✅ Shared Cache |
| **Atomic** | ✅ Single File | ❌ Two Files | ❌ Two Files | ❌ Multiple Files | ❌ Three Tiers |
| **User Control** | ⭐⭐ Manual Update | ⭐⭐⭐ Manual + Menu | ⭐⭐⭐⭐ Auto-Validate | ⭐⭐⭐⭐⭐ Approval Flow | ⭐⭐⭐⭐ Profile Mgmt |
| **Portability** | ⭐⭐⭐⭐⭐ Copy Preset | ⭐⭐ Need Cache | ⭐⭐⭐ Reference + Cache | ⭐⭐ Cache Only | ⭐⭐⭐⭐ Export Profile |
| **Performance** | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐⭐⭐ Instant | ⭐⭐⭐⭐ May Ask User | ⭐⭐⭐⭐⭐ Instant |
| **Maintenance** | ⭐⭐ Each Preset | ⭐⭐⭐⭐ Central Cache | ⭐⭐⭐ Hybrid | ⭐⭐⭐ Approval History | ⭐⭐⭐⭐ Profile-Based |
| **Hardware Change** | ⭐⭐ Manual Each | ⭐⭐⭐ Re-Hash | ⭐⭐⭐⭐⭐ Auto-Detect | ⭐⭐⭐⭐⭐ Auto-Handle | ⭐⭐⭐ Switch Profile |
| **Multi-GPU Support** | ❌ Manual Copy | ⭐⭐ Different Hashes | ⭐⭐⭐ Validation | ⭐⭐⭐ Approval Per GPU | ⭐⭐⭐⭐⭐ Multiple Profiles |
| **Code Changes** | ⭐⭐⭐⭐ Minimal | ⭐⭐⭐ Moderate | ⭐⭐ Significant | ⭐⭐ Significant | ⭐ Extensive |

---

## Recommendation Analysis

### For Your Use Case

Based on your requirements:
- "shouldn't need to manually test... for every run" → All approaches solve this
- "tie into saved configurations" → Approaches #1, #3, #5 do this best
- "user approval" → Approaches #3, #4, #5 have this
- "ensure settings do not vary or differ" → Approaches #3, #5 have validation
- "atomicity of each configuration" → Approach #1 best, #3 second best

### My Top 3 Recommendations

**🥇 BEST: Approach #3 (Hybrid Reference + Validation)**
- **Why**: Perfect balance of simplicity and robustness
- **Pros**: Self-validating, user-friendly, efficient
- **Best For**: Your exact requirements
- **Implementation Effort**: Moderate (2-3 days)

**🥈 SECOND: Approach #5 (Hardware Profiles)**
- **Why**: Most professional, best long-term solution
- **Pros**: Scalable, portable, reusable
- **Best For**: If you plan to support multiple GPUs/machines
- **Implementation Effort**: High (4-5 days)

**🥉 THIRD: Approach #1 (Embedded)**
- **Why**: Simplest to implement and understand
- **Pros**: Truly atomic, no file management
- **Best For**: Quick fix, single-user, single-GPU setup
- **Implementation Effort**: Low (1 day)

### NOT Recommended

**Approach #2 (Separate Cache)**: Too fragile (cache can get deleted), no validation
**Approach #4 (Auto-Detect)**: Approval workflow interrupts user too often

---

## Implementation Roadmap (for Approach #3 - Recommended)

### Phase 1: Core Implementation (Day 1)
1. Add `SplitReference` dataclass to `types.py`
2. Modify `LayerSplitConfig` to include `split_reference` field
3. Update `LayerSplitConfigManager` to support reference creation

### Phase 2: Pipeline Integration (Day 2)
1. Modify pipeline initialization to check for `split_reference`
2. Add validation logic (compare reference metadata to current hardware)
3. Implement "use cached" vs. "re-discover" decision flow
4. Add user prompts for mismatch scenarios

### Phase 3: Menu Integration (Day 2-3)
1. Update config save to include split reference
2. Add "Layer Split Status" menu option
3. Add "Refresh Layer Split" menu option
4. Add validation warnings when loading presets

### Phase 4: Testing & Polish (Day 3)
1. Test with multiple presets
2. Test hardware change scenarios
3. Test cache deletion scenarios
4. Add comprehensive logging

---

## Code Snippets for Approach #3

### SplitReference Dataclass
```python
# src/core/types.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class SplitReference:
    """Reference to cached layer split with validation metadata."""
    cache_hash: str              # Hash of cache file
    model_id: str                # For validation
    quant_method: str            # "4bit", "8bit", "none"
    gpu_memory_gb: float         # Expected GPU memory
    layers_on_gpu: int           # Preview info
    layers_on_cpu: int           # Preview info
    verified_at: str             # ISO timestamp of last verification

    def matches_hardware(self, current_model: str, current_quant: str, current_gpu_gb: float) -> bool:
        """Check if this reference matches current hardware."""
        return (
            self.model_id == current_model and
            self.quant_method == current_quant and
            abs(self.gpu_memory_gb - current_gpu_gb) < 0.5  # Allow 0.5GB tolerance
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitReference':
        return cls(**data)

@dataclass
class LayerSplitConfig:
    # ... existing fields ...

    # NEW: Hybrid reference
    split_reference: Optional[SplitReference] = None
```

### Pipeline Initialization Logic
```python
# src/core/pipeline.py (in initialization)
def _load_layer_split(self):
    """Load layer split from reference or run discovery."""
    split_config = self.config.layer_split_config

    if not split_config or not split_config.enabled:
        return None

    # Check for cached reference
    if split_config.split_reference:
        ref = split_config.split_reference

        # Validate against current hardware
        current_gpu_gb = self._get_gpu_memory_gb()
        if ref.matches_hardware(self.config.model_specifier,
                                self._get_quant_method(),
                                current_gpu_gb):
            # Reference valid - load from cache
            self.logger.info(f"Using cached layer split: {ref.layers_on_gpu} GPU / {ref.layers_on_cpu} CPU")
            device_map = self._load_device_map_from_cache(ref.cache_hash)
            if device_map:
                return device_map
        else:
            # Mismatch - warn user
            self.logger.warning("Cached layer split doesn't match current hardware")
            self.logger.warning(f"  Expected: {ref.model_id}, {ref.quant_method}, {ref.gpu_memory_gb}GB")
            self.logger.warning(f"  Current: {self.config.model_specifier}, {self._get_quant_method()}, {current_gpu_gb}GB")

            choice = input("Use cached split anyway? [y/N/re-discover]: ").strip().lower()
            if choice == 'y':
                device_map = self._load_device_map_from_cache(ref.cache_hash)
                if device_map:
                    return device_map
            elif choice == 're-discover':
                pass  # Fall through to discovery
            else:
                raise PipelineError("Cannot proceed without valid layer split")

    # No reference or user wants re-discovery
    if split_config.auto_discover_splits:
        self.logger.info("Running layer split discovery...")
        min_split, max_split = self._run_discovery()

        # Create reference
        from datetime import datetime
        split_reference = SplitReference(
            cache_hash=self._generate_split_hash(),
            model_id=self.config.model_specifier,
            quant_method=self._get_quant_method(),
            gpu_memory_gb=self._get_gpu_memory_gb(),
            layers_on_gpu=min_split.layers_on_gpu,
            layers_on_cpu=min_split.layers_on_cpu,
            verified_at=datetime.now().isoformat()
        )

        # Update config
        self.config.layer_split_config.split_reference = split_reference

        # Save to preset (if user confirms)
        self._offer_to_save_split_reference(split_reference)

        return min_split.device_map

    return None
```

---

## Conclusion

**Recommended Approach: #3 (Hybrid Reference + Validation)**

This approach provides:
- ✅ Automatic caching (no redundant discovery)
- ✅ User-friendly validation (detects hardware changes)
- ✅ Preset integration (tied to saved configs)
- ✅ User approval (confirmation when needed)
- ✅ Atomicity (reference in config ensures consistency)
- ✅ Efficiency (shared cache, no duplication)

**Implementation Time**: 2-3 days
**Complexity**: Moderate
**Maintenance**: Low
**User Experience**: Excellent

This strikes the perfect balance between simplicity and functionality for your use case.

---

End of Analysis
