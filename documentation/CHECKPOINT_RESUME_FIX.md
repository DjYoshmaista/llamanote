# Checkpoint Resume Fix for Partial Stage Selection

## Issue

When running only a subset of pipeline stages (e.g., `filter, format, save, audio`) with checkpoints from previous runs that include earlier stages (e.g., `process`), the pipeline fails with:

```
MissingDataError: Missing required data 'processed_chunks or chunks' for stage 'filter'
```

## Root Cause

The checkpoint manager loads checkpoint files successfully, but the pipeline doesn't properly resume from them when the user selects only later stages. The checkpoint data contains all the necessary information (`processed_chunks`), but it's not being used.

**Log evidence:**
```
Loaded checkpoint from d51efba47478_process_chunk0030.ckpt (stage: process)
```

But then later:
```
ERROR - Missing required data 'processed_chunks or chunks' for stage 'filter'
```

This means the checkpoint was **loaded** but the data wasn't **used** properly.

## Analysis

Looking at `pipeline.py` lines 175-207:

```python
if self.config.enable_checkpoints and resume_mode != "disabled":
    resume_result = self.checkpoint_manager.get_resume_checkpoint(
        input_path, self.config, self.stages_to_run  # ← Problem here
    )
    if resume_result:
        resume_from_stage, resume_data, resume_chunk_index = resume_result
        # ... sets data_payload = resume_data.copy()
```

The issue is that `get_resume_checkpoint()` is checking if the resume stage is in `stages_to_run`, and if "process" isn't in that list (because user only selected later stages), it might not return the checkpoint as valid.

## Temporary Workaround

**Option 1: Include all prerequisite stages**

When selecting stages, include all earlier stages that produce the data you need:

```
If you want to run: filter, format, save, audio
You must select: extract, preprocess, chunk, process, filter, format, save, audio
```

**Option 2: Delete old checkpoints**

```bash
rm -rf checkpoints/
```

Then run the full pipeline again.

**Option 3: Let the pipeline complete the process stage**

If there's a checkpoint at chunk 30, let the process stage complete (it will resume and finish processing remaining chunks), then run filter/format/save/audio.

## Proper Fix (To Be Implemented)

The checkpoint manager should:
1. Recognize that if a checkpoint exists for an earlier stage (e.g., "process")
2. And the user is running only later stages (e.g., "filter")
3. Then load the checkpoint data even if "process" isn't in `stages_to_run`
4. The data from completed stages should be available to later stages

### Proposed Changes

**File: `src/io/checkpoints.py`**

Modify `get_resume_checkpoint()` to check not just if the stage is in `stages_to_run`, but also if there are later stages that depend on this checkpoint's data.

```python
# Current logic (simplified):
for stage in compatible_checkpoints:
    if stage in stages_to_run:
        return checkpoint

# Proposed logic:
stage_dependencies = {
    'extract': [],
    'preprocess': ['extract'],
    'chunk': ['extract', 'preprocess'],
    'process': ['extract', 'preprocess', 'chunk'],
    'filter': ['process'],  # or 'chunk' or 'text'
    'format': ['filter'],  # or 'process' or 'text'
    'save': ['format'],  # or any earlier stage
    'audio': ['format']  # or any earlier stage with text
}

for stage, checkpoint in compatible_checkpoints:
    # Check if any stage in stages_to_run depends on this checkpoint
    for run_stage in stages_to_run:
        if stage in stage_dependencies.get(run_stage, []):
            return checkpoint
```

**File: `src/core/pipeline.py`**

Enhanced debug logging was already added (lines 190, 370-377) to help diagnose these issues.

## Current Status

✅ Debug logging added to help diagnose the issue
⚠️ Root cause identified
❌ Proper fix not yet implemented

## Workaround Summary

**For now, use Option 1**: When selecting stages to run, include all prerequisite stages:

- If running **audio only**: Select `extract, preprocess, chunk, process, filter, format, save, audio`
- If running **filter onwards**: Select `extract, preprocess, chunk, process, filter, format, save, audio`
- If running **format onwards**: Select `extract, preprocess, chunk, process, filter, format, save, audio`

This ensures checkpoints are created and used correctly.

## Future Enhancement

Implement a "smart resume" feature that:
1. Detects which checkpoints exist
2. Automatically includes necessary stages to utilize those checkpoints
3. Skips stages that are already complete (via checkpoints)
4. Only runs the stages that are actually needed

This would allow users to select "just audio" and have the system figure out what's needed.
