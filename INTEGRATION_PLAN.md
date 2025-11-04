# Integration Plan: Hierarchical Output Structure

## Overview

This document outlines the complete integration plan for the new hierarchical output structure, compression system, transcript generation, and file management features.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Modifications Required](#file-modifications-required)
3. [Configuration Changes](#configuration-changes)
4. [Pipeline Integration](#pipeline-integration)
5. [Checkpoint System Updates](#checkpoint-system-updates)
6. [Menu System Updates](#menu-system-updates)
7. [Migration Strategy](#migration-strategy)
8. [Testing Plan](#testing-plan)
9. [Rollback Procedures](#rollback-procedures)

---

## Architecture Overview

### New Directory Structure

```
llamanote-backup/
├── output/                              # Hierarchical processing outputs
│   └── {input_stem}/                    # Per input file
│       └── {session_id}/                # Per session (e.g., session1a2b3c4d)
│           ├── checkpoint_0001/         # Per checkpoint
│           │   ├── chunk_0001/          # Per chunk (every N chunks or 5000 tokens)
│           │   │   ├── chunk_0001.wav
│           │   │   ├── chunk_0001_transcript.md
│           │   │   ├── chunk_0001_original.md
│           │   │   ├── chunk_0001_report.json
│           │   │   └── {input_stem}_s{session}_ckpt0001_chk0001.ckpt → checkpoints/...
│           │   ├── checkpoint_0001_report.json
│           │   └── checkpoint_0001_transcript.md
│           ├── {input_stem}_final.wav
│           ├── {session_id}_report.json
│           ├── {session_id}_transcript.md
│           └── {session_id}_errors.log
├── generated_audio/                     # Symbolic links to all audio outputs
│   ├── {input_stem}_{session_id}_final.wav → output/.../final.wav
│   └── {input_stem}_{session_id}_chunk0001.wav → output/.../chunk_0001.wav
├── text_files/                          # Managed source files
│   ├── database.json                    # File tracking database
│   └── {input_stem}/
│       ├── original/
│       │   └── {original_file}
│       ├── metadata.json
│       └── sessions/
│           └── {session_id} → output/{input_stem}/{session_id}/
├── checkpoints/                         # Primary checkpoint storage (unchanged)
│   └── {input_stem}_s{sess}_ckpt{n}_chk{m}.ckpt
└── src/compression/                     # Compression subsystem
    ├── compress_audio.py
    ├── compress_text.py
    └── compression_config.py
```

### Data Flow

```
Input File → Text File Manager → Pipeline → Directory Manager → Output Structure
                                     ↓
                                 Session Manager
                                     ↓
                         Checkpoint Manager (updated)
                                     ↓
                        Report Generator + Transcript Generator
                                     ↓
                              Compression System
```

---

## File Modifications Required

### 1. Configuration Types (`src/core/types.py`)

**Changes needed:**

```python
# Add new configuration fields
@dataclass
class PipelineConfig:
    # ... existing fields ...

    # NEW: Checkpoint triggering
    checkpoint_interval: int = 10  # Save every N chunks
    checkpoint_token_threshold: int = 5000  # OR every N tokens (whichever first)

    # NEW: Session management
    session_id: Optional[str] = None  # Set by session manager

    # NEW: Transcript generation
    enable_transcripts: bool = False
    transcript_model: str = "openai/whisper-tiny.en"

    # NEW: Compression
    enable_auto_compression: bool = True
    compression_config: Optional[Dict[str, Any]] = None

    # NEW: Output organization
    use_hierarchical_output: bool = True  # Enable new structure
```

**Impact:** Low risk - adds new optional fields with defaults

---

### 2. Checkpoint System (`src/io/checkpoints.py`)

**Changes needed:**

#### A. Update checkpoint naming scheme

**Current:**
```python
checkpoint_path = base_dir / f"{stage}_{input_stem}.ckpt"
```

**New:**
```python
checkpoint_name = f"{input_stem}_s{session_short_id}_ckpt{ckpt_num:04d}_chk{chunk_num:04d}.ckpt"
checkpoint_path = base_dir / checkpoint_name
```

#### B. Add session tracking to metadata

```python
def save(self, ...):
    checkpoint = {
        "version": "3.0",  # Bump version
        "session_id": session_id,  # NEW
        "checkpoint_number": checkpoint_num,  # NEW
        "chunk_number": chunk_num,  # NEW
        # ... existing fields ...
    }
```

#### C. Add symbolic link creation

```python
def save(self, ..., create_symlinks: bool = True, dir_manager: Optional[DirectoryManager] = None):
    # Save checkpoint to primary location
    checkpoint_path = self._save_checkpoint(...)

    # Create symlinks in output hierarchy
    if create_symlinks and dir_manager:
        chunk_dir = dir_manager.get_chunk_dir(...)
        symlink_path = chunk_dir / checkpoint_path.name
        dir_manager.create_symlink(checkpoint_path, symlink_path)
```

#### D. Update version compatibility

```python
def load(self, checkpoint_path: Path):
    # Support v1.0, v2.0, v3.0
    if version == "3.0":
        # New format with session_id
        session_id = checkpoint.get("session_id")
        checkpoint_num = checkpoint.get("checkpoint_number", 0)
        chunk_num = checkpoint.get("chunk_number", 0)
    elif version in ["2.0", "1.0"]:
        # Legacy format - generate session info
        session_id = None
        checkpoint_num = 0
        chunk_num = 0
```

**Impact:** Medium risk - modifies existing checkpoint system, but maintains backwards compatibility

---

### 3. Pipeline (`src/core/pipeline.py`)

**Major integration point - many changes needed.**

#### A. Initialize new managers at pipeline start

```python
class Pipeline:
    def __init__(self, config: PipelineConfig):
        # ... existing initialization ...

        # NEW: Initialize managers
        self.dir_manager = DirectoryManager(
            base_output_dir=Path("output"),
            logger=self.logger
        )

        self.session_manager = SessionManager(
            sessions_db_path=Path("sessions/database.json"),
            logger=self.logger
        )

        self.text_file_manager = TextFileManager(
            text_files_dir=Path("text_files"),
            logger=self.logger
        )

        self.report_generator = ReportGenerator(logger=self.logger)

        # NEW: Transcript generator (if enabled)
        if config.enable_transcripts:
            self.transcript_generator = TranscriptGenerator(
                model_name=config.transcript_model,
                logger=self.logger
            )
        else:
            self.transcript_generator = None

        # NEW: Compression manager
        if config.compression_config:
            from ..compression import CompressionConfig
            self.compression_config = CompressionConfig.from_dict(config.compression_config)
        else:
            self.compression_config = CompressionConfig()
```

#### B. Setup session at run() start

```python
def run(self, input_file: Path, ...):
    # NEW: Get managed copy of input file
    if self.config.use_hierarchical_output:
        input_file_for_processing = self.text_file_manager.get_file_for_processing(input_file)
    else:
        input_file_for_processing = input_file

    # NEW: Create or resume session
    if resuming_from_checkpoint:
        session_id = self.session_manager.generate_session_id_from_checkpoint(
            checkpoint_metadata,
            input_file.stem
        )
    else:
        session_id = self.session_manager.create_session(
            input_file=input_file,
            config=self.config.to_dict()
        )

    # Store session_id in config
    self.config.session_id = session_id

    # NEW: Create session directory structure
    self.dir_manager.create_session_structure(input_file.stem, session_id)

    # NEW: Link session to text file
    session_dir = self.dir_manager.get_session_dir(input_file.stem, session_id)
    self.text_file_manager.update_file_sessions(
        input_file.stem,
        session_id,
        session_dir
    )
```

#### C. Token counting for checkpoint triggers

```python
class Pipeline:
    def __init__(self):
        # ... existing ...
        self.token_accumulator = 0  # Track tokens since last checkpoint
        self.checkpoint_counter = 0  # Track checkpoint number
```

In processing stage:

```python
# After processing each chunk
chunk_token_count = len(tokenizer.encode(chunk_text))
self.token_accumulator += chunk_token_count

# Check if checkpoint needed
should_checkpoint = (
    (chunk_index + 1) % self.config.checkpoint_interval == 0 or
    self.token_accumulator >= self.config.checkpoint_token_threshold
)

if should_checkpoint:
    self._save_chunk_checkpoint(...)
    self.token_accumulator = 0  # Reset
    self.checkpoint_counter += 1
```

#### D. Chunk checkpoint saving with reports

```python
def _save_chunk_checkpoint(
    self,
    chunk_index: int,
    chunk_text: str,
    audio_array: np.ndarray,
    generation_start_time: float,
    generation_end_time: float,
    ...
):
    input_stem = self.input_file.stem
    session_id = self.config.session_id

    # Create checkpoint and chunk names
    checkpoint_name = f"checkpoint_{self.checkpoint_counter:04d}"
    chunk_name = f"chunk_{chunk_index:04d}"

    # Create chunk directory
    chunk_dir = self.dir_manager.create_chunk_structure(
        input_stem,
        session_id,
        checkpoint_name,
        chunk_name
    )

    # Save audio
    audio_path = self.dir_manager.get_chunk_audio_path(
        input_stem, session_id, checkpoint_name, chunk_name,
        self.config.output_format
    )
    self._save_audio(audio_array, audio_path)

    # Save original text
    original_text_path = self.dir_manager.get_chunk_original_text_path(
        input_stem, session_id, checkpoint_name, chunk_name
    )
    with open(original_text_path, 'w') as f:
        f.write(chunk_text)

    # Generate transcript (if enabled)
    transcript_comparison = None
    if self.transcript_generator:
        transcript_path = self.dir_manager.get_chunk_transcript_path(
            input_stem, session_id, checkpoint_name, chunk_name
        )
        transcript_comparison = self.transcript_generator.generate_transcript_with_comparison(
            audio_path,
            chunk_text,
            transcript_path,
            original_text_path,
            language="en"
        )

    # Generate chunk report
    chunk_report = self.report_generator.generate_chunk_report(
        chunk_index=chunk_index,
        text_content=chunk_text,
        audio_settings={
            "model": self.config.audio_model,
            "sample_rate": self.config.sample_rate,
            "format": self.config.output_format,
            # ... other settings ...
        },
        generation_start_time=generation_start_time,
        generation_end_time=generation_end_time,
        token_count=len(tokenizer.encode(chunk_text)),
        errors=[],  # Collect during processing
        transcript_comparison=transcript_comparison
    )

    report_path = self.dir_manager.get_chunk_report_path(
        input_stem, session_id, checkpoint_name, chunk_name
    )
    self.report_generator.save_chunk_report(chunk_report, report_path)

    # Save checkpoint to primary location
    checkpoint_path = self.checkpoint_manager.save(
        stage="audio",
        input_stem=input_stem,
        data={...},
        metadata={
            "session_id": session_id,
            "checkpoint_number": self.checkpoint_counter,
            "chunk_number": chunk_index,
            ...
        },
        chunk_index=chunk_index
    )

    # Create symlink in chunk directory
    symlink_path = chunk_dir / checkpoint_path.name
    self.dir_manager.create_symlink(checkpoint_path, symlink_path)

    # Offer compression (if enabled)
    if self.config.enable_auto_compression:
        self._offer_compression(audio_path, report_path, original_text_path, transcript_path)
```

#### E. Final output handling

```python
def _finalize_output(self, final_audio_array: np.ndarray):
    input_stem = self.input_file.stem
    session_id = self.config.session_id

    # Save to session directory
    final_output_path = self.dir_manager.get_session_final_output_path(
        input_stem,
        session_id,
        self.config.output_format
    )
    self._save_audio(final_audio_array, final_output_path)

    # Create symlink in generated_audio/
    generated_audio_dir = Path("generated_audio")
    generated_audio_dir.mkdir(exist_ok=True)

    symlink_name = f"{input_stem}_{session_id}_final.{self.config.output_format}"
    symlink_path = generated_audio_dir / symlink_name

    self.dir_manager.create_symlink(final_output_path, symlink_path)

    # Generate session report
    self._generate_session_report()

    # Complete session
    self.session_manager.complete_session(session_id, final_output_path)

    # Offer final compression
    if self.config.enable_auto_compression:
        self._offer_compression(final_output_path)
```

#### F. Session report generation

```python
def _generate_session_report(self):
    input_stem = self.input_file.stem
    session_id = self.config.session_id

    # Get session data
    session_data = self.session_manager.get_session(session_id)

    # Aggregate checkpoint reports
    session_dir = self.dir_manager.get_session_dir(input_stem, session_id)
    checkpoint_reports = self.report_generator.aggregate_checkpoint_reports(session_dir)

    # Generate session report
    session_report = self.report_generator.generate_session_report(
        session_id=session_id,
        session_data=session_data,
        pipeline_config=self.config.to_dict(),
        all_stages=self.stages_to_run,
        checkpoint_reports=checkpoint_reports,
        final_statistics={
            "memory_peak_mb": self.memory_monitor.peak_memory_mb,
            "total_time_seconds": self.total_elapsed_time,
            ...
        }
    )

    # Save session report
    report_path = self.dir_manager.get_session_report_path(input_stem, session_id)
    self.report_generator.save_session_report(session_report, report_path)
```

**Impact:** High risk - extensive modifications to core pipeline logic

---

### 4. Compression Integration

#### A. Post-generation compression prompts

```python
def _offer_compression(self, *file_paths: Path):
    """Offer to compress files after generation."""
    from ..compression import AudioCompressor, TextCompressor

    audio_compressor = AudioCompressor(self.compression_config, self.logger)
    text_compressor = TextCompressor(self.compression_config, self.logger)

    for file_path in file_paths:
        # Check if file needs compression
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        if file_size_mb < self.compression_config.size_threshold_mb:
            continue

        # Determine file type
        if file_path.suffix in ['.wav', '.flac', '.mp3', '.ogg', '.opus']:
            self._offer_audio_compression(file_path, audio_compressor)
        elif file_path.suffix in ['.json', '.md', '.log', '.txt']:
            self._offer_text_compression(file_path, text_compressor)

def _offer_audio_compression(self, file_path: Path, compressor: AudioCompressor):
    """Interactive audio compression prompt."""
    info = compressor.get_audio_info(file_path)

    print(f"\n{'='*80}")
    print(f"Audio File: {file_path.name}")
    print(f"Size: {info['file_size_mb']:.2f} MB")
    print(f"Format: {info['format']} - {info['subtype']}")
    print(f"Sample Rate: {info['sample_rate']} Hz")
    print(f"Channels: {'Stereo' if info['channels'] == 2 else 'Mono'}")
    print(f"Duration: {info['duration_seconds']:.1f} seconds")
    print(f"{'='*80}")

    print("\nCompression Options:")
    print("  1. FLAC (lossless, ~50% size reduction)")
    print("  2. MP3 320kbps (near-transparent, ~10% original size)")
    print("  3. MP3 192kbps (high quality, ~6% original size)")
    print("  4. OGG Vorbis q8 (transparent, ~8% original size)")
    print("  5. Skip compression")

    choice = input("\nSelect option (1-5): ").strip()

    format_map = {
        "1": ("flac", "lossless"),
        "2": ("mp3", "high"),
        "3": ("mp3", "medium"),
        "4": ("ogg", "high")
    }

    if choice in format_map:
        fmt, quality = format_map[choice]

        # Show preview
        preview = compressor.get_compression_preview(file_path, fmt, quality)

        print(f"\nEstimated compressed size: {preview['estimated_size_mb']:.2f} MB")
        print(f"Estimated reduction: {preview['estimated_reduction_percent']:.1f}%")
        print(f"Lossless: {'Yes' if preview['is_lossless'] else 'No'}")

        if preview['potential_data_loss']:
            print("\n⚠️  WARNING: This format uses lossy compression.")
            print("   Some audio quality will be lost (though likely imperceptible).")

        confirm = input("\nProceed with compression? (y/n): ").strip().lower()

        if confirm == 'y':
            compressed = compressor.compress_audio(
                file_path,
                format_override=fmt,
                quality_override=quality
            )

            if compressed:
                delete_orig = input("Delete original file? (y/n): ").strip().lower()
                if delete_orig == 'y':
                    file_path.unlink()
                    print(f"Deleted: {file_path.name}")

def _offer_text_compression(self, file_path: Path, compressor: TextCompressor):
    """Automatic text compression with confirmation."""
    if not compressor.should_compress(file_path):
        return

    file_size_mb = file_path.stat().st_size / (1024 * 1024)

    print(f"\nFile {file_path.name} is {file_size_mb:.2f} MB (threshold: {compressor.config.size_threshold_mb} MB)")
    print(f"Compressing with gzip (lossless, 100% data integrity)...")

    compressed = compressor.compress_file(file_path)

    if compressed:
        compressed_size_mb = compressed.stat().st_size / (1024 * 1024)
        reduction = (1 - compressed_size_mb / file_size_mb) * 100
        print(f"Compressed: {file_size_mb:.2f} MB → {compressed_size_mb:.2f} MB ({reduction:.1f}% reduction)")

        delete_orig = input("Delete original file? (y/n): ").strip().lower()
        if delete_orig == 'y':
            file_path.unlink()
            print(f"Deleted: {file_path.name}")
```

**Impact:** Medium risk - adds new functionality, doesn't modify existing code

---

### 5. Menu System Updates

#### A. Main Menu (`src/main.py` or menu file)

Add new menu options:

```python
def main_menu():
    print("\nMain Menu:")
    print("  1. Process Document")
    print("  2. Resume from Checkpoint")
    print("  3. Checkpoint Management")
    print("  4. File Management")        # NEW
    print("  5. Compression")            # NEW
    print("  6. Configuration")
    print("  7. Exit")
```

#### B. File Management Menu (NEW file: `src/menu_file_management.py`)

```python
class FileManagementMenu:
    def show(self):
        print("\nFile Management:")
        print("  1. List Managed Files")
        print("  2. Add File to Database")
        print("  3. View File Details")
        print("  4. Edit File")
        print("  5. Verify File Integrity")
        print("  6. View File Sessions")
        print("  7. Remove File from Database")
        print("  8. Back")

    def edit_file(self, file_stem: str):
        """Launch editor for file."""
        from ..utils.editor import launch_editor

        file_entry = self.text_file_manager.get_file(file_stem)
        stored_path = Path(file_entry["stored_path"])

        if launch_editor(stored_path):
            print("File edited successfully")
            # Optionally recompute hash
        else:
            print("Edit failed or cancelled")
```

#### C. Compression Menu (NEW file: `src/menu_compression.py`)

```python
class CompressionMenu:
    def show(self):
        print("\nCompression:")
        print("  1. Compress Single File")
        print("  2. Batch Compress by Type")
        print("  3. Configure Compression Settings")
        print("  4. View Compression Statistics")
        print("  5. Decompress File")
        print("  6. Back")

    def batch_compress(self):
        print("\nBatch Compression:")
        print("  1. Compress all audio files")
        print("  2. Compress all reports")
        print("  3. Compress all transcripts")
        print("  4. Compress all logs")
        print("  5. Compress everything")
```

#### D. Update Checkpoint Menu (`src/menu_checkpoint.py`)

Add session-aware browsing:

```python
def browse_checkpoints_menu(self):
    # Group by input file, then by session
    print("\n" + "=" * 80)
    print(f"{'Available Checkpoints (by File and Session)':^80}")
    print("=" * 80)

    # Get all files from text file manager
    files = self.text_file_manager.list_files()

    for file_entry in files:
        file_stem = file_entry["file_stem"]
        print(f"\n📄 {file_stem}")

        # Get sessions for this file
        sessions = self.session_manager.list_sessions_for_file(file_stem)

        for session in sessions:
            session_id = session["session_id"]
            status = session["status"]
            created = session["created_at"]

            print(f"  Session: {session_id} [{status}] - {created}")

            # List checkpoints for this session
            checkpoints = self._get_checkpoints_for_session(file_stem, session_id)

            for ckpt in checkpoints:
                print(f"    ├─ {ckpt['name']} - {ckpt['stage']} - {ckpt['timestamp']}")
```

**Impact:** Medium risk - new menus, updates to existing menu

---

### 6. Editor Utility (`src/utils/editor.py`)

**NEW file:**

```python
"""
Text file editor utility.

Launches system editor with fallback chain: system default → neovim → nano → vim → vi
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

def get_default_editor() -> str:
    """Get default editor from environment or fallback chain."""
    # Check environment variables
    editor = os.environ.get('EDITOR') or os.environ.get('VISUAL')

    if editor:
        return editor

    # Fallback chain
    editors = ['neovim', 'nano', 'vim', 'vi']

    for ed in editors:
        if _command_exists(ed):
            return ed

    return 'vi'  # Last resort

def _command_exists(command: str) -> bool:
    """Check if command exists in PATH."""
    try:
        subprocess.run(
            ['which', command],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def launch_editor(file_path: Path) -> bool:
    """
    Launch editor for file.

    Args:
        file_path: File to edit

    Returns:
        True if edit successful
    """
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False

    editor = get_default_editor()

    try:
        print(f"Launching {editor}...")
        result = subprocess.run([editor, str(file_path)])
        return result.returncode == 0

    except Exception as e:
        print(f"Failed to launch editor: {e}")
        return False

if __name__ == "__main__":
    """Standalone editor script."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Edit text files")
    parser.add_argument("file", type=Path, help="File to edit")
    parser.add_argument("-e", "--editor", help="Editor to use")

    args = parser.parse_args()

    if args.editor:
        os.environ['EDITOR'] = args.editor

    success = launch_editor(args.file)
    sys.exit(0 if success else 1)
```

**Impact:** Low risk - new standalone utility

---

## Configuration Changes

### Configuration File Structure

Create `config/default.yaml`:

```yaml
# Pipeline Configuration
pipeline:
  checkpoint_interval: 10
  checkpoint_token_threshold: 5000
  enable_transcripts: false
  transcript_model: "openai/whisper-tiny.en"
  use_hierarchical_output: true

# Compression Configuration
compression:
  size_threshold_mb: 10.0
  delete_original: false
  auto_compress: true

  audio:
    format: "mp3"
    quality: "high"

  text:
    compression_level: 9
    format: "gz"

# Directory Configuration
directories:
  output: "output"
  checkpoints: "checkpoints"
  text_files: "text_files"
  generated_audio: "generated_audio"
  sessions: "sessions"
```

---

## Migration Strategy

### Migration Scripts

#### 1. `migrate_checkpoints.py` (Project Root)

```python
#!/usr/bin/env python3
"""
Migrate existing checkpoints to new structure.

This script:
1. Reads all existing checkpoints
2. Creates new directory structure
3. Updates checkpoint metadata to v3.0
4. Creates symbolic links
5. Generates reports for migrated data
"""

import json
from pathlib import Path
from src.io.checkpoints import CheckpointManager
from src.io.directory_manager import DirectoryManager
from src.io.session_manager import SessionManager

def migrate_checkpoint(checkpoint_path: Path) -> bool:
    """Migrate single checkpoint."""
    # Load old checkpoint
    checkpoint_manager = CheckpointManager(...)
    result = checkpoint_manager.load(checkpoint_path)

    if not result:
        return False

    metadata, data = result

    # Generate session ID for this checkpoint
    session_id = f"session_migrated_{checkpoint_path.stem}"

    # Update metadata to v3.0
    metadata["version"] = "3.0"
    metadata["session_id"] = session_id
    metadata["checkpoint_number"] = 0  # Default for migrated
    metadata["chunk_number"] = 0

    # Create new directory structure
    # ... create directories, save files, create symlinks ...

    # Re-save checkpoint with new metadata
    checkpoint_manager.save(...)

    return True

def main():
    checkpoints_dir = Path("checkpoints")

    if not checkpoints_dir.exists():
        print("No checkpoints directory found")
        return

    checkpoints = list(checkpoints_dir.glob("*.ckpt"))

    print(f"Found {len(checkpoints)} checkpoints to migrate")

    migrated = 0
    failed = 0

    for ckpt in checkpoints:
        print(f"Migrating: {ckpt.name}...")

        if migrate_checkpoint(ckpt):
            migrated += 1
            print(f"  ✓ Success")
        else:
            failed += 1
            print(f"  ✗ Failed")

    print(f"\nMigration complete: {migrated} succeeded, {failed} failed")

if __name__ == "__main__":
    main()
```

#### 2. `verify_migration.py` (Project Root)

```python
#!/usr/bin/env python3
"""
Verify migrated checkpoints.

Checks:
1. All checkpoints loadable
2. Directory structure correct
3. Symbolic links valid
4. Data integrity preserved
"""

def verify_checkpoint(checkpoint_path: Path) -> tuple[bool, list[str]]:
    """Verify single checkpoint."""
    errors = []

    # Load checkpoint
    # Verify version is 3.0
    # Check session_id exists
    # Verify directory structure
    # Check symlinks
    # Compare data integrity

    return len(errors) == 0, errors

def main():
    # Find all checkpoints
    # Verify each one
    # Report results
    pass

if __name__ == "__main__":
    main()
```

#### 3. `rollback_migration.py` (Project Root)

```python
#!/usr/bin/env python3
"""
Rollback migration if needed.

This script:
1. Finds all v3.0 checkpoints
2. Converts back to v2.0 format
3. Removes new directory structure (optional)
4. Restores original state
"""

def rollback_checkpoint(checkpoint_path: Path) -> bool:
    """Rollback single checkpoint to v2.0."""
    # Load v3.0 checkpoint
    # Remove new fields (session_id, etc.)
    # Save as v2.0
    pass

def main():
    confirm = input("This will rollback all migrations. Are you sure? (yes/no): ")

    if confirm.lower() != "yes":
        print("Cancelled")
        return

    # Rollback all checkpoints
    # Optionally remove new directories
    pass

if __name__ == "__main__":
    main()
```

---

## Testing Plan

### Phase 1: Unit Tests

Test each new component individually:

```bash
# Test directory manager
python -m pytest tests/test_directory_manager.py

# Test session manager
python -m pytest tests/test_session_manager.py

# Test text file manager
python -m pytest tests/test_text_file_manager.py

# Test report generator
python -m pytest tests/test_report_generator.py

# Test transcript generator
python -m pytest tests/test_transcript_generator.py

# Test compression
python -m pytest tests/test_compression.py
```

### Phase 2: Integration Tests

Test components working together:

```bash
# Test pipeline with new structure
python -m pytest tests/test_pipeline_integration.py

# Test checkpoint save/load with new format
python -m pytest tests/test_checkpoint_migration.py
```

### Phase 3: End-to-End Tests

Test complete workflow:

1. **New Processing Run:**
   ```bash
   python main.py --input test.pdf --enable-transcripts --enable-compression
   ```

   Verify:
   - Directory structure created correctly
   - Session tracked
   - Checkpoints saved with symlinks
   - Reports generated at all levels
   - Transcripts generated
   - Compression offered

2. **Resume from Checkpoint:**
   ```bash
   python main.py --resume <checkpoint>
   ```

   Verify:
   - Same session ID used
   - Processing continues correctly
   - New chunks added to same session directory

3. **Migration:**
   ```bash
   python migrate_checkpoints.py
   python verify_migration.py
   ```

   Verify:
   - All old checkpoints migrated
   - No data loss
   - New structure created

### Phase 4: Manual Testing

1. Process a small test file end-to-end
2. Browse checkpoints in menu
3. Test file management menu
4. Test compression menu
5. Test editor integration
6. Verify all symlinks work
7. Check all reports are complete

---

## Rollback Procedures

### If Integration Fails

1. **Immediate Rollback:**
   ```bash
   git checkout src/core/pipeline.py
   git checkout src/io/checkpoints.py
   ```

2. **Data Rollback:**
   ```bash
   python rollback_migration.py
   ```

3. **Configuration Rollback:**
   - Restore old config files
   - Remove new config options

### Backup Strategy

Before integration:

```bash
# Backup existing code
cp -r src src.backup.$(date +%Y%m%d)

# Backup existing data
cp -r checkpoints checkpoints.backup.$(date +%Y%m%d)
cp -r output output.backup.$(date +%Y%m%d)
```

---

## Implementation Order

Recommended order to minimize risk:

1. **Week 1: Configuration & Non-Invasive Changes**
   - Update `src/core/types.py` with new config fields
   - Create editor utility
   - Create compression menu (standalone)
   - Create file management menu (standalone)
   - Test: Menus work independently

2. **Week 2: Checkpoint System Updates**
   - Update `src/io/checkpoints.py` with v3.0 format
   - Maintain backwards compatibility
   - Test: Can load old checkpoints, save new ones

3. **Week 3: Pipeline Integration (Part 1)**
   - Initialize new managers in Pipeline.__init__
   - Add session creation/tracking
   - Add token counting
   - Test: Pipeline runs with new managers, old output still works

4. **Week 4: Pipeline Integration (Part 2)**
   - Update checkpoint saving to use new structure
   - Add report generation
   - Add transcript generation
   - Test: New outputs created in correct locations

5. **Week 5: Compression & Polish**
   - Add compression prompts
   - Add symlink creation
   - Update menus for integration
   - Test: Complete workflow end-to-end

6. **Week 6: Migration & Documentation**
   - Create migration scripts
   - Test migration on copies
   - Run migration on real data
   - Update documentation

---

## Risk Assessment

### High Risk Areas

1. **Pipeline modifications** - Core processing logic
   - Mitigation: Extensive testing, feature flags, rollback plan

2. **Checkpoint format changes** - Data compatibility
   - Mitigation: Version checking, backwards compatibility, migration tools

3. **File system operations** - Symlinks, directory creation
   - Mitigation: Atomic operations, error handling, verification

### Medium Risk Areas

1. **Menu system updates** - User interface changes
   - Mitigation: Keep old menus working, gradual rollout

2. **Configuration changes** - Breaking changes to config
   - Mitigation: Default values, migration guide

### Low Risk Areas

1. **Compression system** - Standalone, optional
2. **Report generation** - New feature, doesn't affect existing
3. **Text file manager** - Optional, doesn't affect pipeline if disabled

---

## Success Criteria

The integration is successful when:

1. ✅ All existing functionality still works
2. ✅ Old checkpoints can be loaded
3. ✅ New checkpoints use new format and structure
4. ✅ Directory structure is created correctly
5. ✅ Sessions are tracked properly
6. ✅ Reports are generated at all levels
7. ✅ Transcripts are generated (when enabled)
8. ✅ Compression works (when enabled)
9. ✅ Symlinks are created and valid
10. ✅ Menus work and are integrated
11. ✅ Migration scripts work correctly
12. ✅ No data loss occurs
13. ✅ Performance is not significantly degraded
14. ✅ All tests pass

---

## Questions for Review

Before proceeding with implementation, please review and confirm:

1. **Directory Structure:** Is the proposed hierarchy acceptable?
2. **Checkpoint Naming:** Is `{input_stem}_s{session}_ckpt{n}_chk{m}.ckpt` clear and useful?
3. **Session Management:** Should resuming always continue the same session, or offer a choice?
4. **Compression Defaults:** Should auto-compression be enabled by default?
5. **Transcript Generation:** Should transcripts be opt-in (default false) or opt-out?
6. **Migration Timing:** Migrate all at once, or allow gradual migration?
7. **Backwards Compatibility:** How long should we support old checkpoint formats?
8. **Feature Flags:** Should all new features have enable/disable flags?

---

## Next Steps

Upon approval of this plan:

1. Create feature branch: `git checkout -b feature/hierarchical-output`
2. Implement Phase 1 (Configuration & utilities)
3. Create PR for review
4. Proceed with subsequent phases
5. Final integration and testing
6. Merge to main branch

---

## Appendix: Code Examples

### Example: Complete Chunk Processing Flow

```python
# In pipeline.py
def _process_chunk_with_full_tracking(
    self,
    chunk_index: int,
    chunk_text: str
) -> bool:
    """Process a single chunk with full tracking, reporting, and compression."""

    start_time = time.time()

    # Generate audio
    audio_array = self.audio_backend.generate_audio(chunk_text)

    end_time = time.time()

    # Update token accumulator
    token_count = len(self.tokenizer.encode(chunk_text))
    self.token_accumulator += token_count

    # Check if checkpoint needed
    should_checkpoint = (
        (chunk_index + 1) % self.config.checkpoint_interval == 0 or
        self.token_accumulator >= self.config.checkpoint_token_threshold
    )

    if should_checkpoint:
        # Reset token counter
        self.token_accumulator = 0
        self.checkpoint_counter += 1

        # Create checkpoint structure
        checkpoint_name = f"checkpoint_{self.checkpoint_counter:04d}"
        chunk_name = f"chunk_{chunk_index:04d}"

        # Save all outputs
        chunk_dir = self.dir_manager.create_chunk_structure(
            self.input_stem,
            self.session_id,
            checkpoint_name,
            chunk_name
        )

        # Save audio
        audio_path = self.dir_manager.get_chunk_audio_path(
            self.input_stem,
            self.session_id,
            checkpoint_name,
            chunk_name,
            self.config.output_format
        )
        self._save_audio_file(audio_array, audio_path)

        # Save original text
        original_path = self.dir_manager.get_chunk_original_text_path(
            self.input_stem, self.session_id, checkpoint_name, chunk_name
        )
        with open(original_path, 'w', encoding='utf-8') as f:
            f.write(chunk_text)

        # Generate transcript (if enabled)
        transcript_comparison = None
        if self.config.enable_transcripts and self.transcript_generator:
            transcript_path = self.dir_manager.get_chunk_transcript_path(
                self.input_stem, self.session_id, checkpoint_name, chunk_name
            )
            transcript_comparison = self.transcript_generator.generate_transcript_with_comparison(
                audio_path,
                chunk_text,
                transcript_path,
                original_path
            )

        # Generate report
        chunk_report = self.report_generator.generate_chunk_report(
            chunk_index=chunk_index,
            text_content=chunk_text,
            audio_settings=self._get_audio_settings_dict(),
            generation_start_time=start_time,
            generation_end_time=end_time,
            token_count=token_count,
            errors=self._collect_errors(),
            transcript_comparison=transcript_comparison
        )

        report_path = self.dir_manager.get_chunk_report_path(
            self.input_stem, self.session_id, checkpoint_name, chunk_name
        )
        self.report_generator.save_chunk_report(chunk_report, report_path)

        # Save checkpoint
        checkpoint_path = self.checkpoint_manager.save(
            stage="audio",
            input_stem=self.input_stem,
            data=self._get_checkpoint_data(),
            metadata={
                "session_id": self.session_id,
                "checkpoint_number": self.checkpoint_counter,
                "chunk_number": chunk_index,
                ...
            },
            chunk_index=chunk_index,
            session_id=self.session_id
        )

        # Create symlink
        symlink_path = chunk_dir / checkpoint_path.name
        self.dir_manager.create_symlink(checkpoint_path, symlink_path)

        # Offer compression
        if self.config.enable_auto_compression:
            self._offer_compression(
                audio_path,
                report_path,
                original_path,
                transcript_path if transcript_comparison else None
            )

    return True
```

---

**End of Integration Plan**

*This document should be reviewed and approved before proceeding with implementation.*
