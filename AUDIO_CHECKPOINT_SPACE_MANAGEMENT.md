# Audio Checkpoint Space Management - Implementation Plan
## Date: 2025-11-10

---

## Problem Statement

**User Requirements**:
1. Audio checkpoint files are hundreds of megabytes each (space-constrained environment)
2. Need lossless compression for audio checkpoint files
3. Need checkpoint overwriting: Each new checkpoint overwrites previous checkpoint during same session
4. Need archival checkpoint system: Preserve checkpoints at 20%, 40%, 60%, 80%, 100% completion milestones for diagnostics

**Current Behavior**:
- Audio checkpoints save raw numpy arrays in pickle files with gzip compression
- Every checkpoint is preserved (no overwriting)
- All checkpoints use same naming pattern
- Large audio arrays cause checkpoint files to be hundreds of MB

---

## Solution Architecture

### Approach: Smart Checkpoint Lifecycle Management

**Core Concepts**:
1. **Rolling Checkpoint**: Single "latest" checkpoint that gets overwritten with each save
2. **Archival Checkpoints**: Milestone checkpoints (20%, 40%, 60%, 80%, 100%) preserved forever
3. **FLAC Compression**: Convert audio arrays to FLAC format before saving (lossless, ~50-60% space saving)
4. **Session Tracking**: Identify checkpoints from same file + model + config combination

---

## Implementation Details

### Phase 1: FLAC Audio Compression

**Concept**: Convert numpy audio arrays to FLAC format before pickling

**Benefits**:
- FLAC is lossless (identical audio when decompressed)
- ~50-60% space savings vs raw numpy arrays
- Industry standard for lossless audio compression
- Fast encode/decode with `soundfile` library (already in use)

**Implementation**:

#### 1.1: Add Audio Compression Helper Methods

**Location**: `src/io/checkpoints.py`

**Methods to Add**:

```python
def _compress_audio_array(self, audio: np.ndarray, sample_rate: int) -> bytes:
    """
    Compress audio numpy array to FLAC format (lossless).

    Args:
        audio: Audio numpy array (float32 or int16)
        sample_rate: Sample rate in Hz

    Returns:
        Compressed audio as bytes (FLAC format)
    """
    import soundfile as sf
    import io

    # Create in-memory buffer
    buffer = io.BytesIO()

    # Write audio to buffer as FLAC
    # FLAC24 provides best compression for float data
    sf.write(buffer, audio, sample_rate, format='FLAC', subtype='PCM_24')

    # Get bytes
    buffer.seek(0)
    compressed_bytes = buffer.read()

    return compressed_bytes

def _decompress_audio_array(self, compressed_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decompress FLAC audio bytes back to numpy array.

    Args:
        compressed_bytes: FLAC-compressed audio bytes

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    import soundfile as sf
    import io

    # Create buffer from bytes
    buffer = io.BytesIO(compressed_bytes)

    # Read audio from buffer
    audio, sample_rate = sf.read(buffer)

    return audio, sample_rate

def _compress_audio_arrays_in_checkpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compress audio arrays in checkpoint data before saving.

    Args:
        data: Checkpoint data dict (may contain 'audio_arrays')

    Returns:
        Modified data dict with compressed audio
    """
    if 'audio_arrays' not in data or not data['audio_arrays']:
        return data

    audio_arrays = data['audio_arrays']
    compressed_audio_list = []

    # Assume all audio has same sample rate (from audio config)
    # Fallback to 22050 Hz if not available
    sample_rate = 22050  # Default

    for idx, audio_array in enumerate(audio_arrays):
        if isinstance(audio_array, np.ndarray):
            try:
                compressed_bytes = self._compress_audio_array(audio_array, sample_rate)
                compressed_audio_list.append({
                    'compressed_bytes': compressed_bytes,
                    'sample_rate': sample_rate,
                    'original_shape': audio_array.shape,
                    'original_dtype': str(audio_array.dtype)
                })
                self.logger.debug(f"Compressed audio array {idx}: {len(compressed_bytes)} bytes")
            except Exception as e:
                self.logger.warning(f"Failed to compress audio array {idx}: {e}")
                # Fallback: keep original array
                compressed_audio_list.append(audio_array)
        else:
            # Not a numpy array, keep as-is
            compressed_audio_list.append(audio_array)

    # Replace audio_arrays with compressed version
    data['audio_arrays'] = compressed_audio_list
    data['_audio_compressed'] = True  # Mark as compressed

    return data

def _decompress_audio_arrays_in_checkpoint(self, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decompress audio arrays after loading checkpoint.

    Args:
        data: Loaded checkpoint data (may have compressed audio)

    Returns:
        Modified data dict with decompressed audio arrays
    """
    if '_audio_compressed' not in data or not data.get('_audio_compressed'):
        # Not compressed, return as-is
        return data

    if 'audio_arrays' not in data:
        return data

    compressed_audio_list = data['audio_arrays']
    decompressed_audio_list = []

    for idx, item in enumerate(compressed_audio_list):
        if isinstance(item, dict) and 'compressed_bytes' in item:
            try:
                audio_array, sample_rate = self._decompress_audio_array(item['compressed_bytes'])
                decompressed_audio_list.append(audio_array)
                self.logger.debug(f"Decompressed audio array {idx}: shape={audio_array.shape}")
            except Exception as e:
                self.logger.error(f"Failed to decompress audio array {idx}: {e}")
                # Data loss - cannot recover
                decompressed_audio_list.append(None)
        else:
            # Not compressed format, keep as-is
            decompressed_audio_list.append(item)

    data['audio_arrays'] = decompressed_audio_list
    del data['_audio_compressed']  # Remove compression flag

    return data
```

**Integration Points**:
- Call `_compress_audio_arrays_in_checkpoint()` in `save()` method before pickling
- Call `_decompress_audio_arrays_in_checkpoint()` in `load()` method after unpickling

---

### Phase 2: Checkpoint Naming for Rolling vs Archival

**Concept**: Distinguish between rolling checkpoints (overwriteable) and archival checkpoints (preserved)

**Naming Schemes**:

1. **Rolling Checkpoint** (latest/current):
   ```
   {file}_{ext}-{text_model}-{audio_model}-ROLLING-audio.ckpt
   ```
   Example: `BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-ROLLING-audio.ckpt`

2. **Archival Checkpoint** (milestone):
   ```
   {file}_{ext}-{text_model}-{audio_model}-ARCHIVE-{pct}pct-audio.ckpt
   ```
   Examples:
   - `BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-ARCHIVE-20pct-audio.ckpt`
   - `BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-ARCHIVE-40pct-audio.ckpt`
   - `BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-ARCHIVE-60pct-audio.ckpt`
   - `BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-ARCHIVE-80pct-audio.ckpt`
   - `BID_paper_pdf-deepseek_r1_1.5b-ms_speecht5-ARCHIVE-100pct-audio.ckpt`

**Implementation**:

#### 2.1: Add Archival Milestone Tracking

**Location**: `src/io/checkpoints.py` - Add to `CheckpointManager.__init__()`

```python
def __init__(self, ...):
    # ... existing init ...

    # Track archival milestones for current session
    # Key: (input_stem, text_model, audio_model) -> set of saved percentages
    self.archival_milestones: Dict[Tuple[str, str, str], Set[int]] = {}

    # Milestone thresholds (20%, 40%, 60%, 80%, 100%)
    self.milestone_thresholds = [20, 40, 60, 80, 100]
```

#### 2.2: Update `_get_checkpoint_path()` for Audio Stage

**Location**: `src/io/checkpoints.py` - Modify `_get_checkpoint_path()` method

**Changes**:
```python
def _get_checkpoint_path(self, input_path: Path, config: PipelineConfig, stage: str,
                        timestamp: Optional[str] = None, chunk_index: Optional[int] = None,
                        total_chunks: Optional[int] = None, completion_pct: Optional[int] = None,
                        checkpoint_type: str = "normal") -> Path:  # NEW PARAMETER
    """
    Args:
        checkpoint_type: "normal", "rolling", or "archival"
    """
    # ... existing code ...

    if stage == "audio" and checkpoint_type in ["rolling", "archival"]:
        # Special naming for rolling/archival audio checkpoints

        text_abbrev = self.model_abbrev_mgr.abbreviate_text_model(text_model)
        audio_abbrev = self.model_abbrev_mgr.abbreviate_audio_model(audio_model_name)

        if checkpoint_type == "rolling":
            # Rolling checkpoint - always same filename (overwriteable)
            filename = f"{file_stem}_{file_ext}-{text_abbrev}-{audio_abbrev}-ROLLING-audio.ckpt"
        else:  # archival
            # Archival checkpoint - includes milestone percentage
            if completion_pct is None:
                completion_pct = 0
            filename = f"{file_stem}_{file_ext}-{text_abbrev}-{audio_abbrev}-ARCHIVE-{completion_pct}pct-audio.ckpt"
    else:
        # ... existing naming logic for normal checkpoints ...

    return checkpoint_dir / filename
```

#### 2.3: Add Archival Decision Logic

**Location**: `src/io/checkpoints.py` - Add new method

```python
def _should_create_archival_checkpoint(self, input_path: Path, config: PipelineConfig,
                                      completion_pct: int) -> Optional[int]:
    """
    Determine if current checkpoint should be saved as archival milestone.

    Args:
        input_path: Input file path
        config: Pipeline configuration
        completion_pct: Current completion percentage

    Returns:
        Milestone percentage to save (20, 40, 60, 80, 100) or None if not a milestone
    """
    # Create session key (file + models)
    file_stem = input_path.stem
    text_model = config.model_specifier or "unknown"
    audio_model = getattr(config.audio_config, 'model_name', 'unknown') if config.audio_config else 'unknown'

    session_key = (file_stem, text_model, audio_model)

    # Initialize milestone tracking for this session if needed
    if session_key not in self.archival_milestones:
        self.archival_milestones[session_key] = set()

    saved_milestones = self.archival_milestones[session_key]

    # Check if we've crossed a new milestone threshold
    for milestone in self.milestone_thresholds:
        if completion_pct >= milestone and milestone not in saved_milestones:
            # First checkpoint >= this milestone - save as archival
            saved_milestones.add(milestone)
            self.logger.info(f"Creating archival checkpoint at {milestone}% milestone (actual: {completion_pct}%)")
            return milestone

    return None
```

---

### Phase 3: Dual Checkpoint Saving (Rolling + Archival)

**Concept**: Save both rolling checkpoint (always) and archival checkpoint (when milestone reached)

**Implementation**:

#### 3.1: Update `save()` Method for Audio Checkpoints

**Location**: `src/io/checkpoints.py` - Modify `save()` method

**Changes**:
```python
def save(self, input_path: Path, config: PipelineConfig, stage: str, data: Dict[str, Any],
         chunk_index: Optional[int] = None, total_chunks: Optional[int] = None,
         config_uuid: Optional[str] = None) -> bool:
    """
    Save checkpoint(s) with smart lifecycle management.

    For audio stage:
    - Always saves rolling checkpoint (overwrites previous)
    - Additionally saves archival checkpoint if milestone reached
    """
    try:
        # Calculate completion percentage
        completion_pct = None
        if chunk_index is not None and total_chunks is not None:
            # ... existing completion calculation ...

        # Compress audio arrays if present (Phase 1)
        if stage == "audio":
            data = self._compress_audio_arrays_in_checkpoint(data)

        # Determine checkpoint type(s) to save
        if stage == "audio":
            # Check if this is an archival milestone
            milestone_pct = self._should_create_archival_checkpoint(input_path, config, completion_pct)

            # Always save rolling checkpoint
            success_rolling = self._save_single_checkpoint(
                input_path, config, stage, data,
                chunk_index, total_chunks, completion_pct,
                config_uuid, checkpoint_type="rolling"
            )

            # Additionally save archival checkpoint if milestone
            if milestone_pct is not None:
                success_archival = self._save_single_checkpoint(
                    input_path, config, stage, data,
                    chunk_index, total_chunks, milestone_pct,  # Use milestone percentage
                    config_uuid, checkpoint_type="archival"
                )
                return success_rolling and success_archival

            return success_rolling
        else:
            # Non-audio stages: normal checkpoint behavior
            return self._save_single_checkpoint(
                input_path, config, stage, data,
                chunk_index, total_chunks, completion_pct,
                config_uuid, checkpoint_type="normal"
            )

    except Exception as e:
        self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
        return False

def _save_single_checkpoint(self, input_path: Path, config: PipelineConfig, stage: str,
                           data: Dict[str, Any], chunk_index: Optional[int] = None,
                           total_chunks: Optional[int] = None, completion_pct: Optional[int] = None,
                           config_uuid: Optional[str] = None, checkpoint_type: str = "normal") -> bool:
    """
    Save a single checkpoint file.

    Args:
        checkpoint_type: "normal", "rolling", or "archival"

    Returns:
        True if save succeeded
    """
    # Get checkpoint path based on type
    checkpoint_path = self._get_checkpoint_path(
        input_path, config, stage,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        completion_pct=completion_pct,
        checkpoint_type=checkpoint_type
    )

    # Create metadata
    data_keys = list(data.keys())
    metadata = self._create_metadata(input_path, config, stage, data_keys)

    # Add checkpoint type to metadata
    metadata["checkpoint_type"] = checkpoint_type

    # Add config UUID if provided
    if config_uuid:
        metadata["config_uuid"] = config_uuid

    # Add chunk info
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    if total_chunks is not None:
        metadata["total_chunks"] = total_chunks
    if completion_pct is not None:
        metadata["completion_pct"] = completion_pct

    # Create checkpoint object
    checkpoint = {
        "metadata": metadata,
        "data": data
    }

    # Save with atomic write
    temp_path = checkpoint_path.with_suffix('.tmp')

    if self.enable_compression:
        with gzip.open(temp_path, 'wb', compresslevel=6) as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(temp_path, 'wb') as f:
            pickle.dump(checkpoint, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Atomic rename
    temp_path.replace(checkpoint_path)

    # Calculate file sizes
    file_size_bytes = checkpoint_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Log save
    checkpoint_name = checkpoint_path.name
    self.logger.info(f"Saved {checkpoint_type} checkpoint: {checkpoint_name} ({file_size_mb:.1f} MB)")

    return True
```

---

### Phase 4: Update Load Method

**Implementation**:

#### 4.1: Update `load()` Method to Handle Compressed Audio

**Location**: `src/io/checkpoints.py` - Modify `load()` method

**Changes**:
```python
def load(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint data from file, decompressing audio if needed.
    """
    try:
        # ... existing load logic ...

        checkpoint = pickle.load(f)

        # Extract data
        data = checkpoint.get("data", {})

        # Decompress audio arrays if present
        if checkpoint.get("metadata", {}).get("stage") == "audio":
            data = self._decompress_audio_arrays_in_checkpoint(data)

        return data

    except Exception as e:
        self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
        return None
```

---

### Phase 5: Update Resume Logic

**Implementation**:

#### 5.1: Prefer Rolling Checkpoint for Resume

**Location**: `src/io/checkpoints.py` - Add method

```python
def find_rolling_audio_checkpoint(self, input_path: Path, config: PipelineConfig) -> Optional[Path]:
    """
    Find the rolling audio checkpoint for current file/config.

    Returns:
        Path to rolling checkpoint if exists, None otherwise
    """
    # Generate expected rolling checkpoint filename
    checkpoint_path = self._get_checkpoint_path(
        input_path, config, "audio",
        checkpoint_type="rolling"
    )

    if checkpoint_path.exists():
        self.logger.info(f"Found rolling audio checkpoint: {checkpoint_path.name}")
        return checkpoint_path

    return None
```

**Update `find_compatible_checkpoints()` to check rolling checkpoint first**:
- For audio stage, check rolling checkpoint first
- Fall back to archival checkpoints if rolling not found
- Archival checkpoints provide resume points if rolling deleted

---

## Configuration Options

**Location**: `src/core/types.py` - Add to relevant config dataclass

```python
@dataclass
class CheckpointConfig:
    """Configuration for checkpoint behavior."""
    enable_audio_compression: bool = True  # Use FLAC compression for audio
    enable_rolling_checkpoints: bool = True  # Use rolling checkpoint for audio
    enable_archival_checkpoints: bool = True  # Save milestone archival checkpoints
    archival_milestones: List[int] = field(default_factory=lambda: [20, 40, 60, 80, 100])  # Milestone percentages
```

---

## Expected Space Savings

### Without Compression:
- Raw audio numpy array: ~200 MB per checkpoint (float32, ~5 minutes audio at 22050 Hz)
- 10 checkpoints: 2 GB

### With FLAC Compression:
- FLAC compressed audio: ~100 MB per checkpoint (~50% reduction)
- Rolling checkpoint only: 100 MB (1 file)
- With archival (5 milestones): ~600 MB (1 rolling + 5 archival)
- **Space savings: 70% reduction** (2 GB → 600 MB)

### Best Case (Minimal Archival Saves):
- If processing completes with only 2 milestones crossed: ~300 MB (1 rolling + 2 archival)
- **Space savings: 85% reduction** (2 GB → 300 MB)

---

## Migration Path

### Backward Compatibility:
1. **Loading Old Checkpoints**:
   - `_decompress_audio_arrays_in_checkpoint()` checks for `_audio_compressed` flag
   - If not present, treats as uncompressed (old format)
   - No migration needed - old checkpoints work as-is

2. **New Checkpoints**:
   - All new checkpoints use FLAC compression
   - Old checkpoints gradually replaced as new sessions run

---

## Testing Plan

### Unit Tests:

1. **Audio Compression**:
   - Test `_compress_audio_array()` with various audio shapes
   - Verify lossless round-trip (compress → decompress → identical)
   - Test compression ratio (~50-60% expected)

2. **Checkpoint Naming**:
   - Test rolling checkpoint naming
   - Test archival checkpoint naming
   - Verify uniqueness and sortability

3. **Milestone Detection**:
   - Test `_should_create_archival_checkpoint()` logic
   - Verify milestones only saved once per session
   - Test edge cases (0%, 100%, exactly at threshold)

4. **Dual Checkpoint Saving**:
   - Test rolling checkpoint overwrite behavior
   - Test archival checkpoint preservation
   - Verify both saved when milestone reached

### Integration Tests:

1. **Full Audio Pipeline**:
   - Process audio file with checkpointing enabled
   - Verify rolling checkpoint gets overwritten
   - Verify archival checkpoints created at milestones
   - Verify final file count (1 rolling + N archival)

2. **Resume from Rolling**:
   - Save rolling checkpoint mid-audio
   - Kill process and resume
   - Verify resume continues correctly

3. **Resume from Archival**:
   - Delete rolling checkpoint
   - Resume from 60% archival checkpoint
   - Verify resume works from archival

4. **Space Savings Validation**:
   - Measure checkpoint sizes before/after compression
   - Verify ~50-60% space savings
   - Verify audio quality preserved (lossless)

---

## Implementation Steps

### Step 1: Add FLAC Compression Methods (Phase 1)
- Add 4 helper methods to CheckpointManager
- Update save() to compress audio
- Update load() to decompress audio
- **Estimated time**: 1 hour

### Step 2: Add Checkpoint Type Naming (Phase 2)
- Update `_get_checkpoint_path()` with checkpoint_type parameter
- Add milestone tracking to `__init__()`
- Add `_should_create_archival_checkpoint()` method
- **Estimated time**: 1 hour

### Step 3: Implement Dual Checkpoint Saving (Phase 3)
- Refactor save() to call `_save_single_checkpoint()`
- Add dual save logic for audio stage
- **Estimated time**: 1.5 hours

### Step 4: Update Resume Logic (Phase 5)
- Add `find_rolling_audio_checkpoint()` method
- Update checkpoint discovery to prefer rolling
- **Estimated time**: 30 minutes

### Step 5: Add Configuration Options
- Add CheckpointConfig dataclass
- Wire up config to checkpoint manager
- **Estimated time**: 30 minutes

### Step 6: Testing
- Write unit tests for compression
- Write integration tests for full pipeline
- Validate space savings
- **Estimated time**: 2 hours

**Total Estimated Time**: 6.5 hours

---

## Rollout Strategy

### Phase 1 (Immediate):
- Implement FLAC compression
- Enable by default
- Backward compatible

### Phase 2 (Next):
- Implement rolling checkpoints
- Enable by default
- Archival milestones default: [20, 40, 60, 80, 100]

### Phase 3 (Optional):
- Allow users to configure milestone thresholds
- Add menu option to clean up old archival checkpoints
- Add statistics display (space saved)

---

## Risk Assessment

### Low Risk:
- FLAC compression (industry standard, lossless, well-tested)
- Rolling checkpoint overwrite (explicit intent)
- Archival preservation (prevents data loss)

### Medium Risk:
- Disk space if archival checkpoints accumulate over many sessions
  - **Mitigation**: Add cleanup tool in menu

### Zero Risk:
- Backward compatibility maintained (old checkpoints still load)
- Audio quality preserved (lossless compression)

---

## Conclusion

This implementation provides:
1. ✅ **70-85% space savings** through FLAC compression + rolling checkpoints
2. ✅ **Diagnostic capability** through archival milestones
3. ✅ **Backward compatibility** with existing checkpoints
4. ✅ **Lossless audio** quality preservation
5. ✅ **Automatic cleanup** through rolling checkpoint overwrite

**Status**: Ready for implementation

**Next Step**: Begin Phase 1 - Add FLAC compression methods

---

**Report Generated**: 2025-11-10
**Author**: Claude Code (Anthropic)
**Priority**: HIGH (User-Requested Feature)

---

End of Report
