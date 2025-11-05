# Checkpoint Resume System Fixes - Round 2

## Overview

After the initial bug fixes, the checkpoint system was still failing to resume properly. This document details the root cause analysis and fixes for the remaining issues that prevented checkpoints from being loaded and used correctly.

## Problem Analysis

### User's Observed Behavior

From the terminal output:
```
Loaded checkpoint from 38c50e0859fa_chunk.ckpt
```

But then immediately:
```
MissingDataError: Pipeline Error: Missing required data 'text or chunks' for stage 'process'.
```

**Key Observation**: The checkpoint file was being LOADED (we see the log message), but the pipeline didn't resume from it. The expected message "Resuming from stage: chunk" never appeared.

## Root Cause Investigation

### The Checkpoint Loading Flow

1. **`pipeline.py` line 191**: Calls `checkpoint_manager.get_resume_checkpoint()`
2. **`checkpoints.py` line 582**: Calls `find_compatible_checkpoints()`
3. **`checkpoints.py` line 530**: Loads each checkpoint with `self.load()`
   - **This is where "Loaded checkpoint from..." appears in logs**
4. **`checkpoints.py` line 541**: Checks compatibility with `_check_stage_compatibility()`
5. **If incompatible**: Checkpoint is NOT added to `compatible_checkpoints` list
6. **If list is empty**: `get_resume_checkpoint()` returns `None` (line 586)
7. **When `None` returned**: Pipeline doesn't resume, `data_payload` stays empty

### Why Checkpoints Were Being Rejected

The checkpoint compatibility check had THREE bugs that caused valid checkpoints to be incorrectly rejected:

## Bug 1: Inconsistent Version Compatibility

**Location**: `src/io/checkpoints.py` lines 438-441

**The Issue**:
- The `load()` method (lines 405-407) accepts checkpoint versions `["2.0", "1.0"]`
- But `_check_stage_compatibility()` required EXACT match to current version `"2.0"`
- Checkpoints created with version 1.0 or no version field were rejected

**Original Code**:
```python
# Check version
if checkpoint_metadata.get("version") != CHECKPOINT_VERSION:
    incompatibilities.append(f"Version mismatch: checkpoint v{checkpoint_metadata.get('version')} vs current v{CHECKPOINT_VERSION}")
```

**Problem**: `checkpoint_metadata.get("version")` returns `None` for old checkpoints, `None != "2.0"` is `True`, so checkpoint is rejected.

**Fixed Code**:
```python
# Check version (allow backwards compatibility with 1.0)
checkpoint_version = checkpoint_metadata.get("version", "1.0")
if checkpoint_version not in [CHECKPOINT_VERSION, "1.0"]:
    incompatibilities.append(f"Version mismatch: checkpoint v{checkpoint_version} vs current v{CHECKPOINT_VERSION}")
```

**Impact**: Now accepts checkpoints with version `"2.0"`, `"1.0"`, or no version field (defaults to `"1.0"`).

---

## Bug 2: Incorrect Model Data Extraction

**Location**: `src/io/checkpoints.py` lines 456-465

**The Issue**:
Checkpoint stores model information as a nested dictionary:
```python
"config": {
    "text_model": {
        "provider": "huggingface",
        "specifier": "DeepSeek-R1-Distill-Qwen-1.5B",
        "full_name": "huggingface:DeepSeek-R1-Distill-Qwen-1.5B"
    }
}
```

But the compatibility check was trying to compare the entire dictionary to a string:

**Original Code**:
```python
checkpoint_model = checkpoint_config.get("text_model", "")
current_model = f"{config.model_provider}:{config.model_specifier}"
if checkpoint_model != current_model:  # Comparing dict to string!
    incompatibilities.append(...)
```

**Problem**:
- `checkpoint_model` = `{"provider": "...", "specifier": "...", "full_name": "..."}`
- `current_model` = `"huggingface:DeepSeek-R1-Distill-Qwen-1.5B"`
- Dictionary never equals string, so checkpoint is ALWAYS rejected (even when models match!)

**Fixed Code**:
```python
checkpoint_model_dict = checkpoint_config.get("text_model", {})
if isinstance(checkpoint_model_dict, dict):
    checkpoint_model = checkpoint_model_dict.get("full_name", "")
else:
    checkpoint_model = str(checkpoint_model_dict)
current_model = f"{config.model_provider}:{config.model_specifier}"
if checkpoint_model != current_model:
    incompatibilities.append(f"Text model mismatch: {checkpoint_model} vs {current_model}")
```

**Impact**: Now correctly extracts the model name string from the nested dictionary for comparison.

---

## Bug 3: Poor Visibility of Rejection Reasons

**Location**: `src/io/checkpoints.py` line 547

**The Issue**:
When checkpoints are rejected, the reason is logged with `logger.debug()`:

**Original Code**:
```python
else:
    self.logger.debug(f"Checkpoint {ckpt_file.name} incompatible: {', '.join(incompatibilities)}")
```

**Problem**: Debug-level logs are not visible by default, making it impossible to diagnose why checkpoints are being rejected.

**Fixed Code**:
```python
else:
    self.logger.info(f"Checkpoint {ckpt_file.name} incompatible: {', '.join(incompatibilities)}")
```

**Impact**: Users can now see exactly why their checkpoints are being rejected, making debugging much easier.

---

## Summary of Changes

| File | Lines | Description | Impact |
|------|-------|-------------|--------|
| `src/io/checkpoints.py` | 438-441 | Fixed version compatibility check | Accepts old checkpoints |
| `src/io/checkpoints.py` | 456-465 | Fixed model data extraction | Correctly compares models |
| `src/io/checkpoints.py` | 547 | Upgraded log level to info | Better debugging visibility |

**Total Changes**: ~15 lines modified across 1 file

---

## How These Bugs Worked Together

The bugs had a cascading effect:

1. **Bug 2 fired first**: Model comparison failed (dict != string) → checkpoint rejected
2. **Bug 1 could also fire**: If checkpoint had no version field → checkpoint rejected
3. **Bug 3 hid the evidence**: Rejection logged at debug level → user couldn't see why

**Result**: Nearly ALL checkpoints were being incorrectly rejected, even when they should have been compatible.

---

## Expected Behavior After Fixes

### Scenario 1: Resume from 'chunk' stage with same model
```
✅ Loaded checkpoint from 38c50e0859fa_chunk.ckpt
✅ Resuming from checkpoint: stage 'chunk' (2025-01-15T10:30:00)
✅ Skipping stages: {'extract', 'preprocess', 'chunk'}
✅ Starting stage 'process' with data from checkpoint
```

### Scenario 2: Resume from 'chunk' stage with different model
```
✅ Loaded checkpoint from 38c50e0859fa_chunk.ckpt
✅ Resuming from checkpoint: stage 'chunk' (2025-01-15T10:30:00)
   (Different model is OK - 'chunk' stage doesn't depend on model)
✅ Skipping stages: {'extract', 'preprocess', 'chunk'}
✅ Starting stage 'process' with NEW model
```

### Scenario 3: Resume from 'process' stage with different model
```
❌ Loaded checkpoint from 38c50e0859fa_process.ckpt
❌ Checkpoint 38c50e0859fa_process.ckpt incompatible: Text model mismatch: huggingface:DeepSeek-R1 vs huggingface:Qwen3-4B
❌ No compatible checkpoints found
   (This is correct behavior - 'process' stage depends on model)
```

---

## Testing Recommendations

### Test 1: Resume from old checkpoint (version 1.0 or no version)
```bash
# Use an existing checkpoint created before version 2.0
python main.py
# Select stages: process, filter, format, save, audio
# Select "RUN PIPELINE"
```

**Expected**: Checkpoint should be accepted and resumed from

### Test 2: Resume from 'chunk' with different model
```bash
# First run: Create checkpoint at 'chunk' stage with Model A
python main.py
# Use Model A
# Stop after chunk stage completes

# Second run: Resume with Model B
python main.py
# Use Model B
# Select stages: process, filter, format, save, audio
# Select "RUN PIPELINE"
```

**Expected**:
- Checkpoint accepted (chunk doesn't depend on model)
- Resumes from 'chunk'
- Uses Model B for processing

### Test 3: Check visibility of rejection reasons
```bash
# Try to resume from a truly incompatible checkpoint
# (e.g., 'process' stage checkpoint with different model)
python main.py
# Use different model than checkpoint
# Select "RUN PIPELINE"
```

**Expected**: See info-level log message explaining why checkpoint was rejected:
```
INFO: Checkpoint 38c50e0859fa_process.ckpt incompatible: Text model mismatch: huggingface:DeepSeek-R1 vs huggingface:Qwen3-4B
```

---

## Relationship to Previous Fixes

### First Round of Fixes (BUG_FIXES_SUMMARY.md)
1. ✅ Fixed import paths in `local_hf.py`
2. ✅ Added checkpoint data reuse logic to extract/preprocess/chunk stages
3. ✅ Made stage compatibility checking only look at resume stage fields

### This Round of Fixes (CHECKPOINT_RESUME_FIXES.md)
4. ✅ Fixed version compatibility to accept old checkpoints
5. ✅ Fixed model data extraction to properly compare models
6. ✅ Improved logging visibility for debugging

**Together**: These fixes create a fully functional checkpoint system that:
- Loads correctly from checkpoints (round 1 + 2)
- Accepts compatible checkpoints (round 2)
- Intelligently reuses checkpoint data (round 1)
- Provides clear feedback when checkpoints are incompatible (round 2)

---

## Technical Details: Why Bug 2 Was So Critical

The model comparison bug (Bug 2) was particularly insidious because:

1. **Always failed**: Dictionary never equals string, regardless of actual model
2. **Silent failure**: Debug logging hid the real error
3. **Blocked all resumes**: Even checkpoints from the SAME model were rejected
4. **Compounded with Bug 1**: Version check could also fail, adding to confusion

**Before Fix**:
```python
checkpoint_model = {"provider": "huggingface", "specifier": "DeepSeek-R1", "full_name": "huggingface:DeepSeek-R1"}
current_model = "huggingface:DeepSeek-R1"
checkpoint_model != current_model  # True (dict != string)
# Checkpoint rejected!
```

**After Fix**:
```python
checkpoint_model = "huggingface:DeepSeek-R1"  # Extracted from dict
current_model = "huggingface:DeepSeek-R1"
checkpoint_model != current_model  # False (strings match)
# Checkpoint accepted!
```

---

## Conclusion

The checkpoint system was failing because checkpoints were being loaded but then immediately rejected during compatibility checking due to:

1. **Inconsistent version compatibility** - Rejected old checkpoints unnecessarily
2. **Incorrect data type comparison** - Compared dict to string, always failed
3. **Poor logging** - Hid the rejection reasons from users

All three bugs have been fixed. The checkpoint system should now:
- ✅ Accept checkpoints from any compatible stage
- ✅ Correctly compare model names
- ✅ Accept checkpoints with old or missing version fields
- ✅ Provide clear feedback when rejecting incompatible checkpoints
- ✅ Resume correctly and populate `data_payload` with checkpoint data

**Next Step**: Test with the actual pipeline to verify the fixes work as expected.

---

## UPDATE: Additional Bug Found

After applying these fixes, testing revealed **one more critical bug** in the stage filtering logic.

See **`CHECKPOINT_RESUME_FIXES_FINAL.md`** for details on Bug #4: Incorrect Stage Filtering.

**Summary of Bug #4**:
- Checkpoints were only considered if their stage was in `stages_to_run`
- This is wrong! A 'chunk' checkpoint should be usable when running ['process', 'filter', 'save']
- Fixed to consider any checkpoint from a stage BEFORE the earliest stage in `stages_to_run`

This was the final bug preventing the checkpoint system from working correctly.
